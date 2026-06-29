"""
safa.py
=======
Scale-Adaptive Foveal Attention (SAFA) — the architectural core of Fovea-MOT.

Motivation
----------
A dense stride-4 (S4) branch explodes FLOPs because token count N = (H/4)(W/4)
and most of the image is background. SAFA mimics the human fovea: look at the
whole scene at low resolution (S8) and only "zoom in" to S4 where something is
likely to be present.

This file provides three drop-in modules:

  1. EntropyScorer          — a 1x1 conv on S8 that predicts a per-cell
                              objectness / spatial-entropy logit E. Used to
                              decide *where* to spend the expensive S4 compute.
                              Supervised by a Gaussian center heatmap (see
                              `loss_entropy` in criterion.py).

  2. SparseFeatFusion       — wraps the existing FeatFusion. The cheap detail/
                              semantic 1x1 projections run densely; the heavy
                              ConvNeXtV2 refinement only contributes on
                              high-entropy cells. Background cells fall back to
                              the cheap detail feature. A straight-through hard
                              mask makes the keep/drop decision differentiable.
                              At inference (sparse_infer=True) the heavy blocks
                              are run only on the tight bounding region of kept
                              cells -> the GFLOPs the paper reports.

  3. ScaleAdaptiveGate      — produces, per query, a soft routing distribution
                              over feature levels conditioned on the reference
                              box size s_q. Small boxes route sampling budget to
                              high-res (S4); large boxes to low-res (S16). Used
                              inside MSDeformableAttention.

Notes on the torch.sparse caveat
--------------------------------
True ragged sparsity (torch.sparse / variable token counts per image) breaks
batched grid_sample/conv and is unstable to train. The mask-gated formulation
below is the practical equivalent: it is dense and exportable during training
(so gradients and BN/LN statistics stay well-defined) yet skips the dominant
pointwise FLOPs of the refinement blocks on dropped cells at inference.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feat_fusion import ConvNeXtV2Block, LayerNorm2d


# =====================================================================
# (1) ENTROPY / OBJECTNESS SCORER AT S8
# =====================================================================
class EntropyScorer(nn.Module):
    """Predict a spatial-entropy (objectness) logit map at S8.

    Cheap by construction: a depthwise 3x3 (context) + pointwise 1x1 -> 1 logit.
    The logit is consumed both as a soft gate and, after top-rho selection, as a
    hard keep mask for the S4 branch.
    """

    def __init__(self, in_ch: int, mid_ratio: float = 0.25):
        super().__init__()
        mid = max(8, int(in_ch * mid_ratio))
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.norm = LayerNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.act = nn.GELU()
        self.head = nn.Conv2d(mid, 1, 1)
        # Focal-style prior: start mostly "background" so early training is stable.
        nn.init.constant_(self.head.bias, -2.19)

    def forward(self, s8: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.dw(s8))
        x = self.act(self.pw(x))
        return self.head(x)                       # [B,1,H8,W8] logit


def _straight_through_topk_mask(prob: torch.Tensor, keep_ratio: float,
                                tau: Optional[float] = None) -> torch.Tensor:
    """Build a hard {0,1} keep-mask with a straight-through estimator.

    Forward: hard mask (top-`keep_ratio` cells per image, optionally also any
    cell with prob > tau). Backward: gradient flows through `prob` so the scorer
    is trained end-to-end by whatever consumes the masked features.

    prob: [B,1,H,W] in (0,1). Returns mask [B,1,H,W] (same dtype).
    """
    B, _, H, W = prob.shape
    flat = prob.flatten(1)                        # [B, H*W]
    n = int(flat.shape[1])                        # static int (constant under trace)
    k = max(1, int(round(keep_ratio * n)))
    # rank-based hard selection. topk is ONNX-exportable (kthvalue is not on many
    # opsets); the k-th largest value is the smallest of the top-k.
    thresh = flat.topk(k, dim=1).values[:, -1:]   # [B,1]
    hard = (flat >= thresh).to(prob.dtype)
    if tau is not None:
        hard = torch.maximum(hard, (flat > tau).to(prob.dtype))
    hard = hard.view(B, 1, H, W)
    soft = prob
    # straight-through: value == hard, grad == d(soft)
    return hard + (soft - soft.detach())


# class SparseFeatFusion(nn.Module):
#     """Entropy-gated stride-4 fusion (the GFLOPs-saving core of SAFA).

#     Cheap paths (detail 1x1, semantic 1x1, gate) run densely. The heavy
#     ConvNeXtV2 refinement only contributes on kept (high-entropy) cells; dropped
#     cells keep the cheap detail feature. Set `keep_ratio` to control the
#     compute/accuracy trade-off (e.g. 0.25 -> heavy compute on ~25% of S4 cells).

#     Drop-in compatible with FeatFusion.forward(c1, s8) and additionally returns
#     the entropy logit so the criterion can supervise it.
#     """

#     def __init__(self, c1_ch: int, hidden_dim: int, mid_dim: Optional[int] = None,
#                  n_blocks: int = 2, expand: float = 2.0, scorer_in_ch: Optional[int] = None,
#                  keep_ratio: float = 0.25, tau: Optional[float] = None,
#                  sparse_infer: bool = True):
#         super().__init__()
#         mid = mid_dim or hidden_dim // 2
#         self.keep_ratio = float(keep_ratio)
#         self.tau = tau
#         self.sparse_infer = bool(sparse_infer)

#         # cheap projections (same as FeatFusion)
#         self.detail = nn.Conv2d(c1_ch, mid, 1, bias=False)
#         self.sem = nn.Conv2d(hidden_dim, mid, 1, bias=False)
#         self.gate_spatial = nn.Sequential(
#             nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=True), nn.Sigmoid())
#         self.alpha = nn.Parameter(torch.tensor(1.0))

#         # heavy refinement (only meaningfully active on kept cells)
#         self.blocks = nn.Sequential(*[ConvNeXtV2Block(mid, expand, k=7) for _ in range(n_blocks)])
#         self.out = nn.Conv2d(mid, hidden_dim, 1, bias=False)
#         self.out_norm = LayerNorm2d(hidden_dim)

#         # entropy scorer on S8 (defaults to S8 == hidden_dim channels)
#         self.scorer = EntropyScorer(scorer_in_ch or hidden_dim)

#     def forward(self, c1: torch.Tensor, s8: torch.Tensor,
#                 return_mask: bool = False):
#         d = self.detail(c1)                                   # [B,mid,H4,W4] cheap
#         s = self.sem(s8)
#         s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False)
#         g = self.gate_spatial(s)
#         x0 = d + self.alpha * (g * s)                         # cheap fused base

#         # entropy map (S8) -> keep mask (S4)
#         ent_logit = self.scorer(s8)                           # [B,1,H8,W8]
#         prob = ent_logit.sigmoid()
#         mask_s8 = _straight_through_topk_mask(prob, self.keep_ratio, self.tau)
#         mask_s4 = F.interpolate(mask_s8, size=d.shape[-2:], mode='nearest')

#         # The gathered fast-path uses data-dependent control flow (per-image
#         # windows) that JIT/ONNX tracing cannot capture, so it is used only in
#         # eager eval. During tracing (and training) we run the dense masked
#         # composite, which is numerically equivalent and fully static.
#         use_sparse = (not self.training) and self.sparse_infer and not torch.jit.is_tracing()
#         if use_sparse:
#             refined = self._sparse_refine(x0, mask_s4)
#         else:
#             # dense + mask composite: heavy blocks contribute only on kept cells
#             refined = self.blocks(x0 * mask_s4)
#             refined = mask_s4 * refined + (1.0 - mask_s4) * x0

#         out = self.out_norm(self.out(refined))
#         if return_mask:
#             return out, ent_logit, mask_s4
#         return out, ent_logit

#     @torch.no_grad()
#     def _sparse_refine(self, x0: torch.Tensor, mask_s4: torch.Tensor) -> torch.Tensor:
#         """Inference fast-path: run heavy blocks only on the tight active window
#         per image. This is what realises the reported GFLOPs reduction. Falls
#         back to the dense composite if a frame has no/all active cells.
#         """
#         B = x0.shape[0]
#         out = x0.clone()
#         for b in range(B):
#             m = mask_s4[b, 0] > 0.5
#             if m.sum() == 0:
#                 continue
#             ys, xs = torch.where(m)
#             y0, y1 = int(ys.min()), int(ys.max()) + 1
#             x0b, x1b = int(xs.min()), int(xs.max()) + 1
#             crop = x0[b:b + 1, :, y0:y1, x0b:x1b]
#             mcrop = mask_s4[b:b + 1, :, y0:y1, x0b:x1b]
#             # Zero inactive cells BEFORE blocks so the sparse fast-path is
#             # numerically identical to the dense composite (blocks(x0*mask)),
#             # including the 7x7 conv behaviour at drop boundaries.
#             r = self.blocks(crop * mcrop)
#             out[b:b + 1, :, y0:y1, x0b:x1b] = mcrop * r + (1.0 - mcrop) * crop
#         return out


class SparseFeatFusion(nn.Module):
    """
    Entropy-gated stride-4 fusion with Unified Bottleneck DFM.
    
    Áp dụng kỹ thuật Cổ chai (Bottleneck): Ép số channel xuống 4 lần 
    trước khi phóng to ảnh lên 4 lần diện tích (Zoom x2).
    GFLOPs được bù trừ hoàn hảo (Net cost = 0), thân thiện với ONNX.
    """

    def __init__(self, c1_ch: int, hidden_dim: int, mid_dim: Optional[int] = None,
                 n_blocks: int = 2, expand: float = 2.0, scorer_in_ch: Optional[int] = None,
                 keep_ratio: float = 0.25, tau: Optional[float] = None,
                 sparse_infer: bool = True):
        super().__init__()
        mid = mid_dim or hidden_dim // 2
        self.keep_ratio = float(keep_ratio)
        self.tau = tau
        
        self.detail = nn.Conv2d(c1_ch, mid, 1, bias=False)
        self.sem = nn.Conv2d(hidden_dim, mid, 1, bias=False)
        self.gate_spatial = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=True), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # ==========================================================
        # [THÊM MỚI]: BỘ NÉN CỔ CHAI (BOTTLENECK) ĐỂ CỨU GFLOPs
        # ==========================================================
        zoom_dim = max(16, mid // 4) # Ép channel xuống 4 lần (VD: 128 -> 32)
        
        self.compress = nn.Conv2d(mid, zoom_dim, 1, bias=False) # Nén
        
        # Khối ConvNeXtV2 giờ chỉ chạy trên số lượng channel cực nhỏ
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(zoom_dim, expand, k=7) for _ in range(n_blocks)])
        
        self.expand_conv = nn.Conv2d(zoom_dim, mid, 1, bias=False) # Bung ra lại
        # ==========================================================

        self.out = nn.Conv2d(mid, hidden_dim, 1, bias=False)
        self.out_norm = LayerNorm2d(hidden_dim)

        self.scorer = EntropyScorer(scorer_in_ch or hidden_dim)

    def forward(self, c1: torch.Tensor, s8: torch.Tensor,
                return_mask: bool = False):
        d = self.detail(c1)                                   
        s = self.sem(s8)
        s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False)
        g = self.gate_spatial(s)
        x0 = d + self.alpha * (g * s)                         

        ent_logit = self.scorer(s8)                           
        prob = ent_logit.sigmoid()
        mask_s8 = _straight_through_topk_mask(prob, self.keep_ratio, self.tau)
        mask_s4 = F.interpolate(mask_s8, size=d.shape[-2:], mode='nearest')

        # =====================================================================
        # UNIFIED DYNAMIC FOVEAL MAGNIFICATION (BOTTLENECK DFM)
        # =====================================================================
        zoom_scale = 2.0
        
        # 1. Nén Channel: Giảm 4 lần khối lượng tính toán
        x0_compressed = self.compress(x0)
        
        # 2. Phóng to ảnh (Zoom in): Tăng 4 lần khối lượng tính toán (Bù trừ = 0)
        x0_zoomed = F.interpolate(x0_compressed, scale_factor=zoom_scale, mode='bilinear', align_corners=False)
        mask_zoomed = F.interpolate(mask_s4, scale_factor=zoom_scale, mode='nearest')
        
        # 3. Học chi tiết nhỏ trên ảnh đã Zoom (GFLOPs siêu thấp vì channel rất nhỏ)
        refined_zoomed = self.blocks(x0_zoomed * mask_zoomed)
        zoomed_res = mask_zoomed * refined_zoomed + (1.0 - mask_zoomed) * x0_zoomed
        
        # 4. Thu nhỏ ảnh (Zoom out)
        refined_compressed = F.interpolate(
            zoomed_res,
            scale_factor=1.0 / zoom_scale,        # 0.5 -> trùng khớp với zoom_scale=2.0
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=False,         # QUAN TRỌNG: giữ scales tĩnh, không tính size từ shape
        )
        
        # 5. Phục hồi Channel
        refined = self.expand_conv(refined_compressed)
        # =====================================================================

        # Cộng Residual để tránh mất mát thông tin nền (Skip Connection)
        out = self.out_norm(self.out(x0 + refined))
        
        if return_mask:
            return out, ent_logit, mask_s4
        return out, ent_logit

# =====================================================================
# (3) SCALE-ADAPTIVE LEVEL ROUTING FOR DEFORMABLE ATTENTION
# =====================================================================
class ScaleAdaptiveGate(nn.Module):
    """Per-query soft routing over feature levels conditioned on box scale.

    For a query with reference box size s_q = (w, h) we build a scale embedding
    phi(s_q) = [log(w+eps), log(h+eps), log(w*h+eps), log(w/h)] and predict a
    distribution over `num_levels` via a tiny MLP. Small boxes concentrate mass
    on high-res levels, large boxes on low-res — implementing the paper's
    "spend FLOPs by object size".

    Trained with a (Gumbel-)softmax of temperature `tau`. The output gate is
    multiplied into the deformable attention weights of each level and the
    weights are then renormalised, so no sampling kernel changes are needed.
    """

    def __init__(self, num_levels: int, hidden: int = 32, tau: float = 1.0,
                 gumbel: bool = False):
        super().__init__()
        self.num_levels = num_levels
        self.tau = tau
        self.gumbel = gumbel
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, num_levels))
        # init near-uniform so early training matches the vanilla model
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _phi(wh: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        w = wh[..., 0].clamp(min=eps)
        h = wh[..., 1].clamp(min=eps)
        return torch.stack([torch.log(w), torch.log(h),
                            torch.log(w * h), torch.log(w / h)], dim=-1)

    def forward(self, ref_wh: torch.Tensor) -> torch.Tensor:
        """ref_wh: [bs, Len_q, 2] (w, h in [0,1]). Returns [bs, Len_q, num_levels]."""
        logits = self.mlp(self._phi(ref_wh))
        if self.gumbel and self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        return F.softmax(logits / self.tau, dim=-1)


def apply_level_gate(attention_weights: torch.Tensor, gate: torch.Tensor,
                     num_points_list: List[int]) -> torch.Tensor:
    """Re-weight & renormalise deformable attention weights by a per-level gate.

    attention_weights : [bs, Len_q, n_heads, sum(num_points_list)] (already softmaxed)
    gate              : [bs, Len_q, n_levels]
    Returns the gated, renormalised attention weights (same shape).
    """
    bs, Lq, nh, P = attention_weights.shape
    # expand gate to per-point: [bs, Lq, sum_points]
    per_point = torch.cat(
        [gate[..., l:l + 1].expand(bs, Lq, n) for l, n in enumerate(num_points_list)],
        dim=-1)                                            # [bs, Lq, P]
    w = attention_weights * per_point.unsqueeze(2)         # broadcast over heads
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
    return w