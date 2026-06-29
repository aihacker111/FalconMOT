"""
siwbd.py
========
Scale-Invariant Wasserstein-Bures Distance (SI-WBD) — a bounding-box loss that
replaces / complements GIoU for extremely small objects (VisDrone / UAVDT).

Idea
----
Model each box (cx, cy, w, h) as a 2-D Gaussian N(mu, Sigma) with
    mu    = [cx, cy]
    Sigma = diag(w^2/4, h^2/4)
The 2-Wasserstein (Bures) distance between predicted p and target t is

    W2^2(p,t) = ||mu_p - mu_t||^2
              + Tr( Sigma_p + Sigma_t - 2 (Sigma_p^{1/2} Sigma_t Sigma_p^{1/2})^{1/2} ).

For axis-aligned diagonal covariances this collapses to a closed, numerically
stable form:

    W2^2 = (dcx^2 + dcy^2) + ((wp - wt)^2 + (hp - ht)^2) / 4 .

Unlike IoU/GIoU this is smooth and well-defined even when boxes do not overlap,
so the gradient does not "break" for 5x5-px boxes where a 1-px shift sends GIoU
off a cliff.

Scale invariance
----------------
W2^2 is dominated by large objects. Normalising by the target area makes a
*relative* displacement incur the same penalty regardless of object size:

    L_SI-WBD = 1 - exp( - W2^2 / (C * Area(t) + eps) ).

C controls the spread; smaller C -> sharper. With C ~ O(1) the loss is in (0,1)
and behaves like a soft, scale-balanced IoU surrogate.
"""
import math
import torch


def size_blend_lambda(area: torch.Tensor,
                      center_area: float = 0.003,
                      scale: float = 1.0,
                      beta: float = 1.0,
                      log_center=None,
                      eps: float = 1e-8) -> torch.Tensor:
    """Size gate for the 'blend' overlap mode (shared by criterion & matcher).

    Trả về lam in (0,1): vật NHỎ hơn ngưỡng -> lam→1 (nghiêng SI-WBD),
    vật LỚN hơn ngưỡng -> lam→0 (nghiêng GIoU).

    Ngưỡng có thể là:
      • TĨNH    : truyền center_area (diện tích chuẩn hóa). Mặc định 0.003 ≈ 32^2px@960x544.
      • ĐỘNG    : truyền log_center (scalar/tensor = log của ngưỡng), ví dụ EMA trung vị
                  log-area của dataset. Khi có log_center thì center_area bị bỏ qua.

    Args:
        area:        diện tích box ĐÃ CHUẨN HÓA (w*h, w,h in [0,1]). [N]
        center_area: ngưỡng tĩnh theo diện tích chuẩn hóa (dùng khi log_center=None).
        scale:       độ rộng chuyển tiếp của sigmoid, theo đơn vị log-area.
        beta:        hệ số nhân thêm lên scale (tương thích siwbd_beta cũ).
        log_center:  (tùy chọn) log của ngưỡng động; ưu tiên hơn center_area.
    """
    la = torch.log(area.clamp(min=eps))
    if log_center is None:
        c = math.log(max(center_area, eps))
    else:
        c = log_center                                   # scalar hoặc tensor (broadcast)
    s = max(float(scale), eps)
    return torch.sigmoid((c - la) / (beta * s))


def gaussian_w2_sq(pred_cxcywh: torch.Tensor, tgt_cxcywh: torch.Tensor,
                   eps: float = 1e-7) -> torch.Tensor:
    """Closed-form 2-Wasserstein^2 between the two Gaussian box representations.

    pred/tgt: [..., 4] in (cx, cy, w, h). Returns [...] (per-box W2^2).
    """
    dcx = pred_cxcywh[..., 0] - tgt_cxcywh[..., 0]
    dcy = pred_cxcywh[..., 1] - tgt_cxcywh[..., 1]
    wp, hp = pred_cxcywh[..., 2].clamp(min=0), pred_cxcywh[..., 3].clamp(min=0)
    wt, ht = tgt_cxcywh[..., 2].clamp(min=0), tgt_cxcywh[..., 3].clamp(min=0)

    center = dcx * dcx + dcy * dcy
    # Bures term for diagonal covariance diag(w^2/4, h^2/4):
    #   (wp/2 - wt/2)^2 + (hp/2 - ht/2)^2 = ((wp-wt)^2 + (hp-ht)^2)/4
    shape = ((wp - wt) ** 2 + (hp - ht) ** 2) / 4.0
    return (center + shape).clamp(min=0) + eps


def si_wbd_loss(pred_cxcywh: torch.Tensor, tgt_cxcywh: torch.Tensor,
                C: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """Scale-Invariant Wasserstein-Bures Distance loss, per box (no reduction).

    pred/tgt: [N,4] normalised (cx, cy, w, h). Returns [N] in (0,1).
    """
    w2 = gaussian_w2_sq(pred_cxcywh, tgt_cxcywh, eps=eps)
    area_t = (tgt_cxcywh[..., 2].clamp(min=0) * tgt_cxcywh[..., 3].clamp(min=0))
    norm = C * area_t + eps
    return 1.0 - torch.exp(-w2 / norm)


def nwd_loss(pred_cxcywh: torch.Tensor, tgt_cxcywh: torch.Tensor,
             constant: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """Plain Normalized Gaussian Wasserstein Distance loss (global constant
    instead of per-box area). Provided as an ablation baseline against SI-WBD.
    """
    w2 = gaussian_w2_sq(pred_cxcywh, tgt_cxcywh, eps=eps)
    return 1.0 - torch.exp(-torch.sqrt(w2) / constant)