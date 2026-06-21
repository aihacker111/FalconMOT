"""Association cost / assignment utilities for the multi-class tracker.

Only the functions used by `MCJDETracker` are kept:
    - linear_assignment    : Jonker-Volgenant linear assignment (via `lap`)
    - iou_distance         : 1 - IoU cost matrix between tracks and detections
    - embedding_distance   : ReID cosine-distance cost matrix
    - fuse_score_three     : fuse IoU and ReID similarities into one cost
The legacy reid-motion / visualisation / greedy / gating helpers were unused
and have been removed.
"""

import lap
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist


def linear_assignment(cost_matrix, thresh):
    """Solve the linear assignment problem with a cost cap.

    Returns (matches, unmatched_a, unmatched_b) where `matches` is an (N, 2)
    array of [track_idx, det_idx] pairs.
    """
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))

    matches = []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """IoU matrix between two lists/arrays of boxes in tlbr format."""
    out = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if out.size == 0:
        return out
    return bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )


def iou_distance(atracks, btracks):
    """1 - IoU cost matrix between two track/detection lists (tlbr boxes)."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs, btlbrs = atracks, btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


def embedding_distance(tracks, detections, metric='cosine'):
    """ReID embedding cost matrix between tracks and detections."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([d.curr_feat for d in detections], dtype=np.float32)
    track_features = np.asarray([t.smooth_feat for t in tracks], dtype=np.float32)
    return np.maximum(0.0, cdist(track_features, det_features, metric))


def fuse_score_three(iou_cost_matrix, id_sim_matrix, detections):
    """LEGACY multiplicative fusion: cost = 1 - (iou_sim * reid_sim).

    Kept for A/B comparison. Gates ReID by IoU multiplicatively, so a strong
    appearance match cannot rescue a low-IoU pair. Prefer `fuse_loglik`.
    """
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix
    iou_sim = 1.0 - iou_cost_matrix
    id_sim = 1.0 - id_sim_matrix
    return 1.0 - (iou_sim * id_sim)


# ───────────────────── Query Appearance-Motion fusion ─────────────────────

def motion_distance(pred_xy, det_xy, track_sizes, kappa=0.1):
    """Size-adaptive Gaussian distance between predicted and detected centres.

        D^m_ij = 1 - exp( -||c_j - ĉ_i||^2 / (2 (κ·sqrt(w_i h_i))^2) )

    Args:
        pred_xy     : (T, 2) appearance-predicted track centres (orig coords).
        det_xy      : (D, 2) detection centres (orig coords).
        track_sizes : (T,) sqrt(w·h) of each track (orig pixels) — sets σ.
        kappa       : σ = κ · object-size; smaller = stricter.
    Returns:
        (T, D) motion distance in [0, 1].
    """
    T, D = len(pred_xy), len(det_xy)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    det_xy  = np.asarray(det_xy,  dtype=np.float32)
    d2 = ((pred_xy[:, None, :] - det_xy[None, :, :]) ** 2).sum(axis=2)   # (T, D)
    sigma = np.maximum(kappa * np.asarray(track_sizes, np.float32), 1.0)[:, None]
    return (1.0 - np.exp(-d2 / (2.0 * sigma ** 2))).astype(np.float32)


def fuse_loglik(d_app, d_iou, d_mot=None, w_mot=None,
                w_app=1.0, w_iou=1.0,
                proximity_gate=0.95, motion_gate=0.9):
    """Probabilistic multi-cue fusion: a per-track weighted average of cue
    *distances* (= −log of a product of independent exponential likelihoods),
    instead of a product of similarities.

        cost_ij = ( w_a d^a_ij + w_g d^g_ij + w^m_i d^m_ij )
                  / ( w_a + w_g + w^m_i )

    A spatial gate blocks pairs that are implausible by BOTH geometry and
    motion (so appearance alone cannot match across the whole frame), while a
    strong IoU *or* a strong motion prediction is enough to vouch for a pair:

        keep_ij = (d^g_ij ≤ proximity_gate) OR (d^m_ij ≤ motion_gate)

    Args:
        d_app : (T, D) appearance (cosine) distance.
        d_iou : (T, D) 1 - IoU.
        d_mot : (T, D) motion distance, or None (appearance+IoU only).
        w_mot : (T,) per-track motion confidence (entropy-gated), or None.
        w_app, w_iou : scalar cue weights.
        proximity_gate, motion_gate : spatial gating thresholds.
    Returns:
        (T, D) fused cost in [0, 1] (gated entries set to a large value).
    """
    if d_app.size == 0:
        return d_app
    T, D = d_app.shape
    cost = w_app * d_app + w_iou * d_iou
    wtot = np.full((T, 1), float(w_app + w_iou), dtype=np.float32)

    use_motion = d_mot is not None and w_mot is not None and d_mot.size > 0
    if use_motion:
        wm = np.asarray(w_mot, dtype=np.float32)[:, None]    # (T, 1)
        cost = cost + wm * d_mot
        wtot = wtot + wm

    cost = cost / np.maximum(wtot, 1e-6)

    # spatial gate: IoU or motion must vouch for the pair
    spatial_ok = d_iou <= proximity_gate
    if use_motion:
        spatial_ok = spatial_ok | (d_mot <= motion_gate)
    cost = np.where(spatial_ok, cost, 1e4).astype(np.float32)
    return cost




# """Association cost / assignment utilities for the multi-class tracker.

# Only the functions used by `MCJDETracker` are kept:
#     - linear_assignment    : Jonker-Volgenant linear assignment (via `lap`)
#     - iou_distance         : 1 - IoU cost matrix between tracks and detections
#     - embedding_distance   : ReID cosine-distance cost matrix
#     - fuse_score_three     : fuse IoU and ReID similarities into one cost
# The legacy reid-motion / visualisation / greedy / gating helpers were unused
# and have been removed.
# """

# import lap
# import numpy as np
# from cython_bbox import bbox_overlaps as bbox_ious
# from scipy.spatial.distance import cdist


# def linear_assignment(cost_matrix, thresh):
#     """Solve the linear assignment problem with a cost cap.

#     Returns (matches, unmatched_a, unmatched_b) where `matches` is an (N, 2)
#     array of [track_idx, det_idx] pairs.
#     """
#     if cost_matrix.size == 0:
#         return (np.empty((0, 2), dtype=int),
#                 tuple(range(cost_matrix.shape[0])),
#                 tuple(range(cost_matrix.shape[1])))

#     matches = []
#     _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
#     for ix, mx in enumerate(x):
#         if mx >= 0:
#             matches.append([ix, mx])

#     unmatched_a = np.where(x < 0)[0]
#     unmatched_b = np.where(y < 0)[0]
#     matches = np.asarray(matches)
#     return matches, unmatched_a, unmatched_b


# def ious(atlbrs, btlbrs):
#     """IoU matrix between two lists/arrays of boxes in tlbr format."""
#     out = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
#     if out.size == 0:
#         return out
#     return bbox_ious(
#         np.ascontiguousarray(atlbrs, dtype=np.float64),
#         np.ascontiguousarray(btlbrs, dtype=np.float64),
#     )


# def iou_distance(atracks, btracks):
#     """1 - IoU cost matrix between two track/detection lists (tlbr boxes)."""
#     if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
#        (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
#         atlbrs, btlbrs = atracks, btracks
#     else:
#         atlbrs = [track.tlbr for track in atracks]
#         btlbrs = [track.tlbr for track in btracks]
#     return 1.0 - ious(atlbrs, btlbrs)


# def embedding_distance(tracks, detections, metric='cosine'):
#     """ReID embedding cost matrix between tracks and detections."""
#     cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
#     if cost_matrix.size == 0:
#         return cost_matrix

#     det_features = np.asarray([d.curr_feat for d in detections], dtype=np.float32)
#     track_features = np.asarray([t.smooth_feat for t in tracks], dtype=np.float32)
#     return np.maximum(0.0, cdist(track_features, det_features, metric))


# def fuse_score_three(iou_cost_matrix, id_sim_matrix, detections):
#     """LEGACY multiplicative fusion: cost = 1 - (iou_sim * reid_sim).

#     Kept for A/B comparison. Gates ReID by IoU multiplicatively, so a strong
#     appearance match cannot rescue a low-IoU pair. Prefer `fuse_loglik`.
#     """
#     if iou_cost_matrix.size == 0:
#         return iou_cost_matrix
#     iou_sim = 1.0 - iou_cost_matrix
#     id_sim = 1.0 - id_sim_matrix
#     return 1.0 - (iou_sim * id_sim)


# # ───────────────────── Query Appearance-Motion fusion ─────────────────────

# def motion_distance(pred_xy, det_xy, track_sizes, kappa=0.1):
#     """Size-adaptive Gaussian distance between predicted and detected centres.

#         D^m_ij = 1 - exp( -||c_j - ĉ_i||^2 / (2 (κ·sqrt(w_i h_i))^2) )

#     Args:
#         pred_xy     : (T, 2) appearance-predicted track centres (orig coords).
#         det_xy      : (D, 2) detection centres (orig coords).
#         track_sizes : (T,) sqrt(w·h) of each track (orig pixels) — sets σ.
#         kappa       : σ = κ · object-size; smaller = stricter.
#     Returns:
#         (T, D) motion distance in [0, 1].
#     """
#     T, D = len(pred_xy), len(det_xy)
#     if T == 0 or D == 0:
#         return np.zeros((T, D), dtype=np.float32)
#     pred_xy = np.asarray(pred_xy, dtype=np.float32)
#     det_xy  = np.asarray(det_xy,  dtype=np.float32)
#     d2 = ((pred_xy[:, None, :] - det_xy[None, :, :]) ** 2).sum(axis=2)   # (T, D)
#     sigma = np.maximum(kappa * np.asarray(track_sizes, np.float32), 1.0)[:, None]
#     return (1.0 - np.exp(-d2 / (2.0 * sigma ** 2))).astype(np.float32)


# def fuse_loglik(d_app, d_iou, d_mot=None, w_mot=None,
#                 w_app=1.0, w_iou=1.0,
#                 proximity_gate=0.95, motion_gate=0.9):
#     """Probabilistic multi-cue fusion: a per-track weighted average of cue
#     *distances* (= −log of a product of independent exponential likelihoods),
#     instead of a product of similarities.

#         cost_ij = ( w_a d^a_ij + w_g d^g_ij + w^m_i d^m_ij )
#                   / ( w_a + w_g + w^m_i )

#     A spatial gate blocks pairs that are implausible by BOTH geometry and
#     motion (so appearance alone cannot match across the whole frame), while a
#     strong IoU *or* a strong motion prediction is enough to vouch for a pair:

#         keep_ij = (d^g_ij ≤ proximity_gate) OR (d^m_ij ≤ motion_gate)
#     """
#     if d_app.size == 0:
#         return d_app
#     T, D = d_app.shape
#     cost = w_app * d_app + w_iou * d_iou
#     wtot = np.full((T, 1), float(w_app + w_iou), dtype=np.float32)

#     use_motion = d_mot is not None and w_mot is not None and d_mot.size > 0
#     if use_motion:
#         wm = np.asarray(w_mot, dtype=np.float32)[:, None]    # (T, 1)
#         cost = cost + wm * d_mot
#         wtot = wtot + wm

#     cost = cost / np.maximum(wtot, 1e-6)

#     spatial_ok = d_iou <= proximity_gate
#     if use_motion:
#         spatial_ok = spatial_ok | (d_mot <= motion_gate)
#     cost = np.where(spatial_ok, cost, 1e4).astype(np.float32)
#     return cost


# # ───────────────── Uncertainty-Aware Appearance-Motion (UAM) ─────────────────

# def inv_var_fuse(p_mean, p_cov, c_mean, c_cov):
#     """Inverse-variance (precision-weighted) fusion of two 2-D Gaussian position
#     estimates — the Bayesian-optimal combination of independent estimators.

#         P = (Σp⁻¹ + Σc⁻¹)⁻¹ ,   x = P (Σp⁻¹ μp + Σc⁻¹ μc)

#     Kalman supplies (μp, Σp); the appearance correlation supplies (μc, Σc) whose
#     Σc is the response spread, so a diffuse/occluded response (large Σc) is
#     automatically down-weighted and the estimate falls back to Kalman.

#     Args:
#         p_mean : (2,)   Kalman position mean.
#         p_cov  : (2,2)  Kalman position covariance.
#         c_mean : (2,)   correlation position mean (or None → Kalman only).
#         c_cov  : (2,2)  correlation covariance (or None).
#     Returns:
#         x : (2,) fused mean,  P : (2,2) fused covariance.
#     """
#     Pp = np.linalg.inv(p_cov + np.eye(2, dtype=p_cov.dtype) * 1e-6)
#     if c_mean is None or c_cov is None:
#         return np.asarray(p_mean, np.float32), np.linalg.inv(Pp).astype(np.float32)
#     Pc = np.linalg.inv(c_cov + np.eye(2, dtype=c_cov.dtype) * 1e-6)
#     P = np.linalg.inv(Pp + Pc)
#     x = P @ (Pp @ np.asarray(p_mean) + Pc @ np.asarray(c_mean))
#     return x.astype(np.float32), P.astype(np.float32)


# def maha_gate_cost(fused_means, fused_covs, det_xy, app_cost,
#                    chi2=9.21, cos_thresh=0.4, iou_dists=None, iou_gate=0.7):
#     """Cascade cost for UAM: spatial gate (motion OR IoU) + cosine cost.

#     For every (track i, detection j):
#         d²_ij = (z_j − x̂_i)ᵀ P_i⁻¹ (z_j − x̂_i)             # motion plausibility
#         spatial_ok = (d²_ij ≤ chi2)  OR  (iou_dist_ij ≤ iou_gate)
#         keep  = spatial_ok  AND  (app_cost_ij ≤ cos_thresh)
#         cost  = app_cost_ij if keep else ∞

#     IoU vouches for adjacent-frame pairs (the reliable cue); the Kalman⊕corr
#     Mahalanobis vouches for low-IoU pairs (fast motion / occlusion recovery).
#     Appearance always ranks. Motion never *removes* a good IoU match — it only
#     *adds* recoveries, so UAM cannot underperform plain IoU+appearance.
#     """
#     T, D = app_cost.shape
#     if T == 0 or D == 0:
#         return app_cost
#     det_xy = np.asarray(det_xy, np.float32)
#     cost = np.full((T, D), 1e4, dtype=np.float32)
#     for i in range(T):
#         Pinv = np.linalg.inv(fused_covs[i] + np.eye(2, dtype=np.float32) * 1e-6)
#         diff = det_xy - fused_means[i][None, :]            # (D, 2)
#         d2 = np.einsum('di,ij,dj->d', diff, Pinv, diff)    # (D,)
#         spatial = d2 <= chi2
#         if iou_dists is not None:
#             spatial = spatial | (iou_dists[i] <= iou_gate)
#         ok = spatial & (app_cost[i] <= cos_thresh)
#         cost[i, ok] = app_cost[i, ok]
#     return cost