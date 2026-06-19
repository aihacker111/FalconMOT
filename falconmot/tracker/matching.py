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
#     """Fuse IoU and ReID similarities multiplicatively into a single cost."""
#     if iou_cost_matrix.size == 0:
#         return iou_cost_matrix
#     iou_sim = 1.0 - iou_cost_matrix
#     id_sim = 1.0 - id_sim_matrix
#     return 1.0 - (iou_sim * id_sim)



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


def fuse_score_three(iou_cost_matrix, id_sim_matrix, detections,
                     emb_weight: float = 1.0, emb_gate: float = 0.0):
    """Fuse IoU and ReID similarities into a single association cost.

    emb_weight:
        1.0  -> hành vi GỐC: cost = 1 - (iou_sim * id_sim)  (fusion nhân).
        0.0  -> IoU THUẦN: bỏ hẳn embedding, cost = 1 - iou_sim.
        giữa -> blend tuyến tính giữa cost IoU-thuần và cost fusion-nhân.
    emb_gate:
        Chỉ áp nhánh embedding ở các ô có id_sim >= ngưỡng này (embedding
        đủ tin); dưới ngưỡng thì fallback về IoU thuần. 0.0 = tắt gating.

    LƯU Ý: với emb_weight=1.0 và emb_gate=0.0, hàm trả về ĐÚNG kết quả như
    bản gốc (không clamp id_sim) để tái lập được số liệu cũ.
    """
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix

    iou_cost = iou_cost_matrix              # = 1 - iou_sim
    iou_sim  = 1.0 - iou_cost_matrix

    # IoU thuần: bỏ qua embedding hoàn toàn.
    if emb_weight <= 0.0 or id_sim_matrix is None or id_sim_matrix.size == 0:
        return iou_cost

    id_sim     = 1.0 - id_sim_matrix
    fused_cost = 1.0 - (iou_sim * id_sim)   # cost fusion-nhân (bản gốc)

    if emb_weight >= 1.0:
        cost = fused_cost
    else:
        w    = float(emb_weight)
        cost = (1.0 - w) * iou_cost + w * fused_cost

    # Gating: ô nào embedding không đủ tin -> dùng IoU thuần.
    if emb_gate > 0.0:
        low_conf = id_sim < float(emb_gate)
        cost = cost.copy()
        cost[low_conf] = iou_cost[low_conf]

    return cost