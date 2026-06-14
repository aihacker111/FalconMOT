"""
association.py — lean matching primitives for the FalconMOT tracker.

Replaces the old 800-line ``matching.py`` (most of which was dead code paths
for a dense-ReID model that no longer exists).  Everything here operates on the
per-query ReID embeddings produced by FalconJDE.

Two ideas are adapted from FusionTrack (arXiv:2505.18727), both inference-only
so they cost nothing at train time:

  * Time-decayed feature memory   (TMP + W = e^{-α·Δt}, Eq. 11)
        recent gallery features weigh more than stale ones.
  * Mutual top-k neighbour gating (NFM)
        a track↔det pair is only trusted if each is in the other's top-k
        nearest neighbours — kills spurious ReID matches → fewer ID switches.

Hard deps (``lap``, ``cython_bbox``) are used when present, with pure
numpy/scipy fallbacks so the tracker runs anywhere.
"""

import numpy as np

try:                                   # fast C assignment
    import lap
    _HAS_LAP = True
except Exception:                      # pragma: no cover
    from scipy.optimize import linear_sum_assignment
    _HAS_LAP = False

try:                                   # fast C IoU
    from cython_bbox import bbox_overlaps as _bbox_ious
    _HAS_CYBOX = True
except Exception:                      # pragma: no cover
    _HAS_CYBOX = False


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def linear_assignment(cost_matrix, thresh):
    """Solve the assignment, dropping pairs whose cost exceeds ``thresh``.

    Returns (matches[K,2], unmatched_rows, unmatched_cols).
    """
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))

    if _HAS_LAP:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[i, x[i]] for i in range(len(x)) if x[i] >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        rows, cols = linear_sum_assignment(cost_matrix)
        matches, ra, cb = [], set(), set()
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c]); ra.add(r); cb.add(c)
        unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in ra])
        unmatched_b = np.array([j for j in range(cost_matrix.shape[1]) if j not in cb])

    return np.asarray(matches), unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def _ious(atlbrs, btlbrs):
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float64)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float64)
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)

    if _HAS_CYBOX:
        return _bbox_ious(atlbrs, btlbrs)

    # numpy fallback (vectorised)
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    lt = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-7, None)


def iou_distance(tracks, dets):
    """1 - IoU cost matrix (N_tracks, N_dets) using each object's tlbr."""
    atlbrs = [t.tlbr for t in tracks]
    btlbrs = [d.tlbr for d in dets]
    return 1.0 - _ious(atlbrs, btlbrs)


# ---------------------------------------------------------------------------
# Appearance — time-decayed gallery distance (FusionTrack TMP)
# ---------------------------------------------------------------------------

def decay_gallery_distance(tracks, dets, frame_id, alpha=0.0):
    """Min cosine distance to a track's feature gallery, recency-weighted.

    For each stored gallery feature with age Δt = frame_id - stored_frame,
    its similarity is scaled by w = exp(-alpha·Δt).  ``alpha=0`` recovers the
    plain min-distance gallery (no decay).  Larger alpha trusts recent
    appearances more — useful when objects change look over time.

    Returns (N_tracks, N_dets) cost ∈ [0, 1].
    """
    cost = np.ones((len(tracks), len(dets)), dtype=np.float32)
    if cost.size == 0:
        return cost
    det_feats = np.asarray([d.curr_feat for d in dets], dtype=np.float32)   # (M, D)
    for i, track in enumerate(tracks):
        gallery = track.feat_gallery                                       # deque[(feat, frame)]
        if not gallery:
            continue
        feats  = np.stack([g[0] for g in gallery], axis=0)                 # (G, D)
        sims   = feats @ det_feats.T                                       # (G, M) cosine sim
        if alpha > 0.0:
            ages = np.array([frame_id - g[1] for g in gallery], dtype=np.float32)
            w    = np.exp(-alpha * np.clip(ages, 0, None))[:, None]        # (G, 1)
            sims = sims * w                                                # decayed sim
        best = sims.max(axis=0)                                            # (M,)
        cost[i] = np.clip(1.0 - best, 0.0, 1.0)
    return cost


def fuse_additive(iou_cost, reid_cost, w_iou=0.5):
    """Additive IoU+ReID fusion: 1 - (w_iou·IoUsim + w_reid·ReIDsim).

    Robust when IoU=0 (camera pan / fast motion) because the ReID term keeps
    the pair alive — unlike multiplicative fusion which collapses to 0.
    """
    if iou_cost.size == 0:
        return iou_cost
    w_reid = 1.0 - w_iou
    return 1.0 - (w_iou * (1.0 - iou_cost) + w_reid * (1.0 - reid_cost))


# ---------------------------------------------------------------------------
# Mutual top-k neighbour gating (FusionTrack NFM)
# ---------------------------------------------------------------------------

def mutual_topk_gate(cost, k=2, penalty=1.0):
    """Suppress non-mutual-neighbour pairs in a cost matrix.

    A pair (i, j) is kept only if j is among row i's k cheapest columns AND
    i is among column j's k cheapest rows.  Non-mutual pairs get ``penalty``
    added (pushing them above the assignment threshold).  This is the core of
    FusionTrack's Neighbor Filtering Mechanism, applied within a single view.

    No-op when either side has ≤ k entries.
    """
    if cost.size == 0 or cost.shape[0] <= k or cost.shape[1] <= k:
        return cost
    row_topk = np.argsort(cost, axis=1)[:, :k]            # (N, k) cols nearest per row
    col_topk = np.argsort(cost, axis=0)[:k, :]            # (k, M) rows nearest per col

    keep = np.zeros_like(cost, dtype=bool)
    rows = np.arange(cost.shape[0])[:, None]
    keep[rows, row_topk] = True                           # row-side candidates
    col_keep = np.zeros_like(cost, dtype=bool)
    cols = np.arange(cost.shape[1])[None, :]
    col_keep[col_topk, cols] = True                       # col-side candidates

    mutual = keep & col_keep
    out = cost.copy()
    out[~mutual] += penalty
    return out


# ---------------------------------------------------------------------------
# Appearance-aware duplicate removal
# ---------------------------------------------------------------------------

def remove_duplicates(tracks_a, tracks_b, frame_id, iou_thr=0.15, reid_thr=0.3):
    """Drop near-duplicate tracks across two pools.

    A pair is a duplicate only if it overlaps spatially (IoU > 1-iou_thr) AND
    looks alike (gallery dist < reid_thr) — prevents merging distinct but
    adjacent objects.  The younger track is dropped.
    """
    if not tracks_a or not tracks_b:
        return tracks_a, tracks_b
    iou_d  = iou_distance(tracks_a, tracks_b)
    reid_d = decay_gallery_distance(tracks_a, tracks_b, frame_id)
    pairs = np.where((iou_d < iou_thr) & (reid_d < reid_thr))

    drop_a, drop_b = set(), set()
    for p, q in zip(*pairs):
        age_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        age_q = tracks_b[q].frame_id - tracks_b[q].start_frame
        (drop_b if age_p > age_q else drop_a).add(q if age_p > age_q else p)

    res_a = [t for i, t in enumerate(tracks_a) if i not in drop_a]
    res_b = [t for i, t in enumerate(tracks_b) if i not in drop_b]
    return res_a, res_b


# ---------------------------------------------------------------------------
# Pool set ops
# ---------------------------------------------------------------------------

def join_tracks(a, b):
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def sub_tracks(a, b):
    remove = {t.track_id for t in b}
    return [t for t in a if t.track_id not in remove]
