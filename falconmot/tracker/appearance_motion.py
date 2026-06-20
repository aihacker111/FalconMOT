"""Query Appearance-Motion (QAM) — appearance-as-motion for query-based JDE.

AMOT (center-based) estimates per-object motion with a dense cross-frame ReID
attention (`reid_motion`) that needs a per-pixel appearance map *and* a center
regression map. A query-based transformer produces neither: it emits sparse
per-instance embeddings sampled at boxes.

This module re-derives the same idea for the sparse setting, reusing the dense
feature map the ReID head already samples from (`value_proj(reid_feat)`), with
three changes that make it both portable and better-grounded than `reid_motion`:

  1. SOFT-ARGMAX instead of argmax + regression offset.  The correlation
     response itself is turned into a sub-pixel location by its expectation,
     so no separate center-regression map is required:

         A_i(p) = softmax_p( <a_i, R̂(p)> / τ ),     ĉ_i = Σ_p A_i(p) · p

  2. ENTROPY-GATED CONFIDENCE.  A diffuse response (occluded / ambiguous
     object) is untrustworthy; a sharp peak is reliable.  The cue weights
     *itself* by the normalised entropy of its own response:

         w_i = exp( −β · H̄(A_i) ),   H̄ = −Σ A log A / log(HW) ∈ [0, 1]

  3. The predicted location feeds a SIZE-ADAPTIVE Gaussian motion cost and a
     LOG-LIKELIHOOD fusion (see matching.fuse_loglik), i.e. a weighted sum of
     per-cue *distances* (= −log of a product of independent likelihoods),
     not the product-of-similarities used by `fuse_score_three`.

All tensors are torch; computation runs on whatever device the dense map is on
(GPU at inference).  Everything here is parameter-free — no extra weights, no
retraining required to prototype.  An optional cross-frame correlation loss
(documented in the paper notes) can sharpen R̂ further if needed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def normalize_dense(reid_dense: torch.Tensor) -> torch.Tensor:
    """L2-normalise a dense appearance map per spatial location.

    Args:
        reid_dense: [C, H, W] value-projected appearance map.
    Returns:
        [C, H, W] with unit-norm channel vectors at every pixel.
    """
    return F.normalize(reid_dense, dim=0, eps=1e-6)


@torch.no_grad()
def predict_centers(templates: torch.Tensor,
                    dense_hat: torch.Tensor,
                    tau: float = 0.07):
    """Predict each template's location in the current frame by correlation.

    Args:
        templates : [T, C] L2-normalised per-track appearance templates.
        dense_hat : [C, H, W] L2-normalised dense appearance map (current frame).
        tau       : softmax temperature for the correlation response.
    Returns:
        centers : [T, 2] predicted (x, y) in *map* pixel coords (float).
        entropy : [T] normalised response entropy in [0, 1] (0 = sharp peak).
        peak    : [T] max cosine similarity (template-present-in-map proxy).
    """
    C, H, W = dense_hat.shape
    if templates.numel() == 0:
        z = templates.new_zeros((0,))
        return templates.new_zeros((0, 2)), z, z

    R = dense_hat.reshape(C, H * W)               # [C, HW]
    sim = templates @ R                           # [T, HW] cosine similarity
    A = torch.softmax(sim / tau, dim=1)           # [T, HW] spatial response

    device = A.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=A.dtype),
        torch.arange(W, device=device, dtype=A.dtype),
        indexing='ij',
    )
    xs = xs.reshape(-1)                            # [HW]
    ys = ys.reshape(-1)
    cx = (A * xs).sum(dim=1)                       # [T] soft-argmax
    cy = (A * ys).sum(dim=1)

    ent = -(A * A.clamp_min(1e-12).log()).sum(dim=1) / float(np.log(H * W))
    peak = sim.max(dim=1).values
    return torch.stack([cx, cy], dim=1), ent, peak


@torch.no_grad()
def sample_dense(dense: torch.Tensor, xy_map: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a dense map at (x, y) map-pixel coordinates.

    Args:
        dense  : [C, H, W] (raw or normalised) appearance map.
        xy_map : [N, 2] sample locations (x, y) in map pixel coords.
    Returns:
        [N, C] sampled feature vectors.
    """
    C, H, W = dense.shape
    if xy_map.numel() == 0:
        return dense.new_zeros((0, C))
    x = xy_map[:, 0] / max(W - 1, 1) * 2.0 - 1.0
    y = xy_map[:, 1] / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=1).view(1, -1, 1, 2)
    samp = F.grid_sample(dense.unsqueeze(0), grid,
                         mode='bilinear', align_corners=True)   # [1, C, N, 1]
    return samp.view(C, -1).t().contiguous()                    # [N, C]


def confidence_from_entropy(entropy: np.ndarray, beta: float = 4.0) -> np.ndarray:
    """Map normalised response entropy -> motion-cue confidence weight."""
    return np.exp(-beta * np.asarray(entropy, dtype=np.float32))


# ---- coordinate helpers: feature-map space <-> original image-space ----
#
# Supports both preprocessing conventions via per-axis ratios + pad:
#   • PLAIN RESIZE (this repo): ratio_x = net_w/orig_w, ratio_y = net_h/orig_h,
#     pad_w = pad_h = 0   (anisotropic, no letterbox).
#   • LETTERBOX:               ratio_x = ratio_y = min(net/orig), pad centred.

def map_to_orig(xy_map: np.ndarray, stride: float,
                ratio_x: float, ratio_y: float,
                pad_w: float = 0.0, pad_h: float = 0.0) -> np.ndarray:
    """map pixel (x, y) at `stride` -> original-image pixel (x, y)."""
    xy = np.asarray(xy_map, dtype=np.float32)
    x = (xy[:, 0] * stride - pad_w) / ratio_x
    y = (xy[:, 1] * stride - pad_h) / ratio_y
    return np.stack([x, y], axis=1)


def orig_to_map(xy_orig: np.ndarray, stride: float,
                ratio_x: float, ratio_y: float,
                pad_w: float = 0.0, pad_h: float = 0.0) -> np.ndarray:
    """original-image pixel (x, y) -> map pixel (x, y) at `stride`."""
    xy = np.asarray(xy_orig, dtype=np.float32)
    x = (xy[:, 0] * ratio_x + pad_w) / stride
    y = (xy[:, 1] * ratio_y + pad_h) / stride
    return np.stack([x, y], axis=1)