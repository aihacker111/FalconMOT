"""Associator — everything about *how tracks and detections are matched*.

Takes raw per-cue cost matrices (appearance / IoU / motion) and is solely
responsible for: building the fused cost, applying spatial gates, the optional
velocity-direction (OCM) term, and running the linear assignment. Each cascade
stage the tracker needs is one method here, so `MCJDETracker.update()` reads as
a list of stage calls rather than 150 lines of matrix juggling.

Stages
  • first()        : appearance + IoU (+ motion) log-likelihood fusion, gated.
  • iou()          : pure-IoU fallback (stage-2 and unconfirmed tracks).
  • reid_revive()  : NEW. appearance-only revival of *lost* tracks across large
                     spatial gaps — relative (mutual-NN + ratio) gate, no spatial
                     gate. The main lever against fragmentation / post-occlusion IDs.
"""

from __future__ import annotations

import numpy as np

from falconmot.tracker import matching


class Associator:
    def __init__(self, cfg):
        self.cfg = cfg

    # ───────────────────────────── primitives ────────────────────────────
    @staticmethod
    def appearance_distance(tracks, dets):
        return matching.embedding_distance(tracks, dets)

    @staticmethod
    def _assign(cost, thresh):
        return matching.linear_assignment(cost, thresh=thresh)

    # ─────────────────────────── fusion policy ───────────────────────────
    def _fuse(self, d_app, d_iou, d_mot, w_mot, tracks=None, dets=None):
        """Per-track weighted average of cue *distances* with a spatial gate.

            cost = (w_a d_app + w_g d_iou + w_m d_mot [+ w_o d_ocm]) / Σw
            keep = (d_iou <= proximity_gate) OR (d_mot <= motion_gate)
        """
        if d_app.size == 0:
            return d_app
        cfg = self.cfg
        T, D = d_app.shape
        cost = cfg.appearance_weight * d_app + cfg.iou_weight * d_iou
        wtot = np.full((T, 1), float(cfg.appearance_weight + cfg.iou_weight), np.float32)

        use_motion = d_mot is not None and w_mot is not None and d_mot.size > 0
        if use_motion:
            wm = np.asarray(w_mot, np.float32)[:, None]
            cost = cost + wm * d_mot
            wtot = wtot + wm

        if cfg.use_ocm and tracks is not None and dets is not None:
            d_ocm = self._ocm_cost(tracks, dets)
            if d_ocm is not None:
                cost = cost + cfg.ocm_weight * d_ocm
                wtot = wtot + cfg.ocm_weight

        cost = cost / np.maximum(wtot, 1e-6)

        spatial_ok = d_iou <= cfg.proximity_gate
        if use_motion:
            spatial_ok = spatial_ok | (d_mot <= cfg.motion_gate)
        return np.where(spatial_ok, cost, 1e4).astype(np.float32)

    def _ocm_cost(self, tracks, dets):
        """Velocity-direction consistency (OC-SORT OCM), in [0, 1].

        Compares each track's recent observation direction to the direction
        from its last box to a candidate detection. Pairs moving against the
        track's heading are penalised. Tracks with <2 observations contribute 0.
        """
        T, D = len(tracks), len(dets)
        if T == 0 or D == 0:
            return None
        det_c = np.array([[d.tlwh[0] + d.tlwh[2] * 0.5,
                           d.tlwh[1] + d.tlwh[3] * 0.5] for d in dets], np.float32)
        cost = np.zeros((T, D), np.float32)
        for i, t in enumerate(tracks):
            hist = getattr(t, 'tlwh_deque', None)
            if not hist or len(hist) < 2:
                continue
            (_, prev), (_, last) = hist[-2], hist[-1]
            pc = np.array([prev[0] + prev[2] / 2, prev[1] + prev[3] / 2], np.float32)
            lc = np.array([last[0] + last[2] / 2, last[1] + last[3] / 2], np.float32)
            v = lc - pc
            nv = np.linalg.norm(v)
            if nv < 1e-3:
                continue
            v = v / nv
            cand = det_c - lc
            nc = np.linalg.norm(cand, axis=1, keepdims=True)
            cand = cand / np.maximum(nc, 1e-6)
            cos = (cand @ v).clip(-1.0, 1.0)
            cost[i] = (1.0 - cos) * 0.5
        return cost

    # ──────────────────────────── cascade stages ─────────────────────────
    def first(self, tracks, dets, d_app, d_iou, d_mot, w_mot):
        """Stage-1 association. Returns (matches, u_tracks, u_dets)."""
        if self.cfg.legacy_fuse:
            cost = matching.fuse_score_three(d_iou, d_app, dets)
            return self._assign(cost, 0.6)
        cost = self._fuse(d_app, d_iou, d_mot, w_mot, tracks, dets)
        return self._assign(cost, self.cfg.match_thresh)

    def iou(self, tracks, dets, thresh):
        """Pure-IoU association (stage-2, unconfirmed)."""
        cost = matching.iou_distance(tracks, dets)
        return self._assign(cost, thresh)

    def _relative_gate(self, cost):
        """Boolean mask of appearance matches that are *relatively* convincing.

        Scale-free: instead of asking "is this distance below 0.30?", it asks
        "does the best match win clearly?". A pair (i, j) is allowed iff

            • Lowe ratio: best_i <= reid_ratio * second_best_i  (best beats runner-up)
            • mutual NN  : j is i's nearest det AND i is j's nearest track

        Both criteria depend on the *shape* of this frame's distance distribution,
        so they don't need re-tuning when the embedding scale drifts between
        sequences. A loose absolute backstop (reid_gate_max) is applied on top.
        """
        cfg = self.cfg
        T, D = cost.shape
        allowed = np.zeros((T, D), dtype=bool)

        # row-wise best / second-best (Lowe ratio test per track)
        if D >= 2:
            part = np.partition(cost, 1, axis=1)
            best, second = part[:, 0], part[:, 1]
        else:
            best = cost[:, 0]
            second = np.full(T, np.inf, np.float32)
        ratio_ok = best <= np.maximum(cfg.reid_ratio * second, 1e-6)   # (T,)

        row_arg = cost.argmin(axis=1)                                  # (T,)
        col_arg = cost.argmin(axis=0)                                  # (D,)
        for i in range(T):
            j = row_arg[i]
            if not ratio_ok[i]:
                continue
            if cfg.reid_mutual and col_arg[j] != i:
                continue
            if cost[i, j] > cfg.reid_gate_max:
                continue
            allowed[i, j] = True
        return allowed

    def reid_revive(self, lost_tracks, dets):
        """Appearance-only revival of lost tracks across large spatial gaps.

        No spatial gate (a confident appearance match may reconnect a track that
        moved far during occlusion). Gating is *relative* (mutual-NN + Lowe
        ratio, see `_relative_gate`) rather than an absolute cosine ceiling, plus
        an area-ratio sanity check so a tiny box is never fused with a huge one.
        """
        cfg = self.cfg
        T, D = len(lost_tracks), len(dets)
        if T == 0 or D == 0:
            return np.empty((0, 2), int), tuple(range(T)), tuple(range(D))

        cost = matching.embedding_distance(lost_tracks, dets)   # (T, D) cosine
        t_area = np.array([max(t.tlwh[2] * t.tlwh[3], 1.0) for t in lost_tracks], np.float32)
        d_area = np.array([max(d.tlwh[2] * d.tlwh[3], 1.0) for d in dets], np.float32)
        ratio = np.maximum(t_area[:, None] / d_area[None, :],
                           d_area[None, :] / t_area[:, None])
        cost = np.where(ratio <= cfg.reid_area_gate, cost, 1e4).astype(np.float32)

        # keep only relatively-convincing entries, then solve assignment on them
        allowed = self._relative_gate(cost)
        cost = np.where(allowed, cost, 1e4).astype(np.float32)
        return self._assign(cost, cfg.reid_gate_max)
