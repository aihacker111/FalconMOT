"""UAM (Uncertainty-Aware appearance-Motion) ops.

Các helper này tách độ bất định vị trí thành COVARIANCE ĐO ĐƯỢC (không tham số
giả định): độ trải của correlation response = covariance của ước lượng vị trí
theo appearance. Nhờ đó loại bỏ hoàn toàn các hyperparameter của QAM cũ
(am_kappa / am_beta / am_w_app / am_w_iou / proximity / motion_gate).

Dùng bởi falconmot/tracker/multitracker.py (MCJDETracker._uam_predict + bước
association 1). Không sửa matching.py / appearance_motion.py — chỉ thêm file này.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1) Correlation localisation TRẢ KÈM covariance của response (map-pixel^2).
#    Peak sharp -> cov nhỏ -> tin; peak khuếch tán (che khuất) -> cov lớn -> bị
#    inverse-variance fuse tự động hạ trọng số. Không cần entropy/beta/sigma.
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_centers_cov(templates: torch.Tensor,
                        dense_hat: torch.Tensor,
                        tau: float = 0.1):
    """
    Args:
        templates : [T, C] template track đã L2-normalise.
        dense_hat : [C, H, W] dense map đã L2-normalise (frame hiện tại).
        tau       : nhiệt độ softmax (độ phân giải response).
    Returns:
        centers : [T, 2]    soft-argmax (x, y) trong toạ độ map-pixel.
        cov_map : [T, 2, 2] covariance response (map-pixel^2).
        peak    : [T]       cosine lớn nhất (đại lượng chất lượng match, [-1,1]).
    """
    C, H, W = dense_hat.shape
    if templates.numel() == 0:
        z2 = templates.new_zeros((0, 2))
        return z2, templates.new_zeros((0, 2, 2)), templates.new_zeros((0,))

    R = dense_hat.reshape(C, H * W)                 # [C, HW]
    sim = templates @ R                             # [T, HW] cosine
    A = torch.softmax(sim / tau, dim=1)             # [T, HW] response

    device = A.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=A.dtype),
        torch.arange(W, device=device, dtype=A.dtype),
        indexing='ij',
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    cx = (A * xs).sum(dim=1)                         # [T] soft-argmax
    cy = (A * ys).sum(dim=1)

    # (co)variance không gian có trọng số = độ bất định vị trí
    dx = xs[None, :] - cx[:, None]                   # [T, HW]
    dy = ys[None, :] - cy[:, None]
    vxx = (A * dx * dx).sum(dim=1)                    # [T]
    vyy = (A * dy * dy).sum(dim=1)
    vxy = (A * dx * dy).sum(dim=1)

    eps = 0.25   # sàn 1/4 px^2: peak one-hot vẫn có covariance hữu hạn
    cov = torch.zeros((templates.shape[0], 2, 2), device=device, dtype=A.dtype)
    cov[:, 0, 0] = vxx + eps
    cov[:, 1, 1] = vyy + eps
    cov[:, 0, 1] = vxy
    cov[:, 1, 0] = vxy

    peak = sim.max(dim=1).values
    return torch.stack([cx, cy], dim=1), cov, peak


# ---------------------------------------------------------------------------
# 2) Scale covariance từ map-pixel^2 -> orig-pixel^2.
#    x_orig = x_map * (stride/ratio) -> J = diag(sx, sy), Σ_orig = J Σ_map Jᵀ.
# ---------------------------------------------------------------------------
def cov_map_to_orig(cov_map: np.ndarray, stride: float,
                    ratio_x: float, ratio_y: float) -> np.ndarray:
    cov_map = np.asarray(cov_map, dtype=np.float32)
    if cov_map.size == 0:
        return cov_map
    sx = stride / ratio_x
    sy = stride / ratio_y
    out = cov_map.copy()
    out[:, 0, 0] = cov_map[:, 0, 0] * sx * sx
    out[:, 1, 1] = cov_map[:, 1, 1] * sy * sy
    out[:, 0, 1] = cov_map[:, 0, 1] * sx * sy
    out[:, 1, 0] = out[:, 0, 1]
    return out


# ---------------------------------------------------------------------------
# 3) Inverse-variance fuse: Kalman (μp,Σp) ⊕ correlation (μc,Σc).
#    P = (Σp⁻¹ + Σc⁻¹)⁻¹ ,  x = P (Σp⁻¹ μp + Σc⁻¹ μc).
#    Response khuếch tán (Σc lớn) -> tự rơi về Kalman. Không tham số cân tay.
# ---------------------------------------------------------------------------
def inv_var_fuse(p_mean, p_cov, c_mean, c_cov):
    Pp = np.linalg.inv(p_cov + np.eye(2, dtype=np.float32) * 1e-6)
    if c_mean is None or c_cov is None:
        return np.asarray(p_mean, np.float32), np.linalg.inv(Pp).astype(np.float32)
    Pc = np.linalg.inv(c_cov + np.eye(2, dtype=np.float32) * 1e-6)
    P = np.linalg.inv(Pp + Pc)
    x = P @ (Pp @ np.asarray(p_mean) + Pc @ np.asarray(c_mean))
    return x.astype(np.float32), P.astype(np.float32)


# ---------------------------------------------------------------------------
# 4) Cost association UAM: cổng không gian (motion χ² HOẶC IoU) + cosine.
#    keep = [ (d²_maha ≤ chi2) OR (iou_dist ≤ iou_gate) ] AND (app_cost ≤ cos_thresh)
#    cost = app_cost nếu keep, ngược lại ∞.
#
#    IoU bảo lãnh cặp frame-kề (cue đáng tin); Mahalanobis (Kalman⊕corr) bảo lãnh
#    cặp IoU thấp (motion nhanh / phục hồi sau che khuất). Appearance luôn xếp hạng.
#    Motion KHÔNG xoá match IoU tốt — chỉ THÊM phục hồi -> UAM không thể tệ hơn
#    IoU+appearance thuần.
# ---------------------------------------------------------------------------
def maha_gate_cost(fused_means, fused_covs, det_xy, app_cost,
                   chi2=9.21, cos_thresh=0.4, iou_dists=None, iou_gate=0.7):
    """
    Args:
        fused_means : (T,2)  tâm dự đoán đã fuse (orig coords).
        fused_covs  : (T,2,2) covariance đã fuse (orig px^2).
        det_xy      : (D,2)  tâm detection (orig coords).
        app_cost    : (T,D)  cosine distance (appearance, từ sparse emb).
        chi2        : ngưỡng χ²₂ (9.21 = mức 0.99) — HẰNG SỐ THỐNG KÊ, không tune.
        cos_thresh  : trần cosine distance — nút thật sự duy nhất.
        iou_dists   : (T,D)  1-IoU, hoặc None.
        iou_gate    : ngưỡng IoU distance để bảo lãnh cặp.
    Returns:
        (T,D) cost; ô bị cổng chặn = 1e4.
    """
    T, D = app_cost.shape
    if T == 0 or D == 0:
        return app_cost
    det_xy = np.asarray(det_xy, np.float32)
    cost = np.full((T, D), 1e4, dtype=np.float32)
    for i in range(T):
        Pinv = np.linalg.inv(fused_covs[i] + np.eye(2, dtype=np.float32) * 1e-6)
        diff = det_xy - fused_means[i][None, :]             # (D, 2)
        d2 = np.einsum('di,ij,dj->d', diff, Pinv, diff)     # (D,)
        spatial = d2 <= chi2
        if iou_dists is not None:
            spatial = spatial | (iou_dists[i] <= iou_gate)
        ok = spatial & (app_cost[i] <= cos_thresh)
        cost[i, ok] = app_cost[i, ok]
    return cost