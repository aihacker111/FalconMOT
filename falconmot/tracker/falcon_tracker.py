"""
falcon_tracker.py — multi-class online tracker for FalconJDE.

Per-class 4-stage association (carried over from the old MCTrackerV2, which was
the strongest variant) with two FusionTrack-inspired upgrades and the dead
dense-ReID path removed:

  Stage 1  confirmed tracks  vs  all dets        IoU+ReID, NFM-gated   thr .60
  Stage 2  remaining tracks  vs  remaining dets  IoU+ReID, NFM-gated   thr .70
  Stage 3  LOST tracks       vs  remaining dets  decayed ReID, gated   thr .50
  Stage 4  unconfirmed       vs  remaining dets  IoU+ReID              thr .50
  → new tracks from leftover high-confidence dets

Improvements vs old tracker
  * time-decayed gallery matching (FusionTrack TMP / W = e^{-α·Δt})
  * mutual top-k neighbour gating on appearance stages (FusionTrack NFM)
  * separated lost pool, NSA-Kalman, GMC, appearance-aware dedup (kept)
  * dense-ReID / motion attention removed — FalconJDE emits per-query ReID
"""

from collections import defaultdict

import numpy as np

from falconmot.tracking_utils.kalman_filter import KalmanFilter
from falconmot.tracking_utils.gmc import GMC
from . import association as A
from .base import TrackState
from .track import Track


class FalconTracker:
    def __init__(self, opt, frame_rate=30):
        self.opt          = opt
        self.num_classes  = opt.num_classes
        self.det_thresh   = opt.conf_thres
        self.buffer_size  = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        # FusionTrack knobs (safe defaults; tune via opts if desired)
        self.decay_alpha = getattr(opt, 'reid_decay_alpha', 0.02)   # TMP recency
        self.nfm_topk    = getattr(opt, 'nfm_topk', 2)              # NFM mutual-k
        self.use_nfm     = getattr(opt, 'use_nfm', True)

        # GMC fusion-weight ramp (thay cho ngưỡng cứng norm<30 từng gây nhảy
        # benchmark): nội suy mềm w_iou theo độ lớn dịch chuyển camera.
        self.w_iou_hi    = getattr(opt, 'w_iou_hi', 0.5)   # GMC tin cậy (motion nhỏ)
        self.w_iou_lo    = getattr(opt, 'w_iou_lo', 0.3)   # GMC kém tin (motion lớn)
        self.gmc_band_lo = getattr(opt, 'gmc_band_lo', 20.0)
        self.gmc_band_hi = getattr(opt, 'gmc_band_hi', 40.0)

        self.tracked = defaultdict(list)
        self.lost    = defaultdict(list)
        self.removed = defaultdict(list)

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.gmc = self._make_gmc()
        self._curr_img = None

    # ------------------------------------------------------------------ api
    def _make_gmc(self):
        """Tạo GMC mới (re-seed RNG, prevFrame sạch). Dùng chung cho init & reset."""
        return GMC(
            method='sparseOptFlow', verbose=[None, False],
            seed=getattr(self.opt, 'gmc_seed', 0),
            deterministic=getattr(self.opt, 'gmc_deterministic', True),
        )

    def set_image(self, img):
        self._curr_img = img

    def reset(self):
        self.tracked = defaultdict(list)
        self.lost    = defaultdict(list)
        self.removed = defaultdict(list)
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        # Tạo lại GMC: tránh rò rỉ prevFrame giữa các sequence + re-seed RNG
        self.gmc = self._make_gmc()
        self._curr_img = None

    # ------------------------------------------------------------------ helpers
    def _gmc_fusion_weight(self, gmc_H):
        """Nội suy mềm w_iou theo độ lớn dịch chuyển camera (||translation||).

        Thay cho ngưỡng cứng `0.5 if norm<30 else 0.3` — vốn lật nhánh chỉ vì
        một dao động sub-pixel của gmc_H, khuếch đại variance benchmark. Ở đây
        w_iou giảm tuyến tính từ w_iou_hi (motion nhỏ, GMC đáng tin) xuống
        w_iou_lo (motion lớn) trong dải [gmc_band_lo, gmc_band_hi].
        """
        if gmc_H is None:
            return self.w_iou_hi
        trans = float(np.linalg.norm(gmc_H[:2, 2]))
        lo, hi = self.gmc_band_lo, self.gmc_band_hi
        if hi <= lo:
            return self.w_iou_hi if trans < hi else self.w_iou_lo
        t = (trans - lo) / (hi - lo)
        t = min(1.0, max(0.0, t))                      # clip [0,1]
        return self.w_iou_hi * (1.0 - t) + self.w_iou_lo * t

    def _reid_cost(self, tracks, dets, gated):
        cost = A.decay_gallery_distance(tracks, dets, self.frame_id, self.decay_alpha)
        if gated and self.use_nfm:
            cost = A.mutual_topk_gate(cost, k=self.nfm_topk)
        return cost

    # ------------------------------------------------------------------ update
    def update(self, dets_per_class, h_orig, w_orig):
        self.frame_id += 1
        if self.frame_id == 1:
            Track.reset_counts(self.num_classes)

        # global motion compensation (once per frame)
        gmc_H = None
        if self._curr_img is not None:
            try:
                r = self.gmc.apply(self._curr_img, None)
                gmc_H = r[0] if isinstance(r, tuple) else r
            except Exception:
                gmc_H = None
        w_iou = self._gmc_fusion_weight(gmc_H)

        output = defaultdict(list)

        for cls_id in range(self.num_classes):
            dets = dets_per_class.get(cls_id, [])

            unconfirmed = [t for t in self.tracked[cls_id] if not t.is_activated]
            confirmed   = [t for t in self.tracked[cls_id] if t.is_activated]
            lost_pool   = self.lost[cls_id]

            Track.multi_predict(confirmed)
            Track.multi_predict(unconfirmed)
            Track.multi_predict(lost_pool)
            if gmc_H is not None:
                Track.multi_gmc(confirmed, gmc_H)
                Track.multi_gmc(unconfirmed, gmc_H)
                Track.multi_gmc(lost_pool, gmc_H)

            activated, refind, lost_now, removed_now = [], [], [], []

            # ── Stage 1: confirmed vs all dets ──────────────────────────
            cost = A.fuse_additive(A.iou_distance(confirmed, dets),
                                   self._reid_cost(confirmed, dets, gated=True), w_iou)
            m, u_trk, u_det = A.linear_assignment(cost, thresh=0.6)
            for it, idd in m:
                confirmed[it].update(dets[idd], self.frame_id, float(cost[it, idd]))
                activated.append(confirmed[it])

            # ── Stage 2: remaining confirmed vs remaining dets ──────────
            r_trk  = [confirmed[i] for i in u_trk]
            r_dets = [dets[i] for i in u_det]
            cost2 = A.fuse_additive(A.iou_distance(r_trk, r_dets),
                                    self._reid_cost(r_trk, r_dets, gated=True), w_iou)
            m2, u_trk2, u_det2 = A.linear_assignment(cost2, thresh=0.7)
            for it, idd in m2:
                r_trk[it].update(r_dets[idd], self.frame_id, float(cost2[it, idd]))
                activated.append(r_trk[it])
            for it in u_trk2:
                if r_trk[it].state != TrackState.Lost:
                    r_trk[it].mark_lost(); lost_now.append(r_trk[it])

            # ── Stage 3: LOST vs remaining dets (decayed ReID only) ─────
            r_dets3 = [r_dets[i] for i in u_det2]
            if lost_pool and r_dets3:
                cost3 = self._reid_cost(lost_pool, r_dets3, gated=True)
                m3, _, u_det3 = A.linear_assignment(cost3, thresh=0.5)
                for it, idd in m3:
                    lost_pool[it].re_activate(r_dets3[idd], self.frame_id, new_id=False)
                    refind.append(lost_pool[it])
            else:
                u_det3 = list(range(len(r_dets3)))

            # ── Stage 4: unconfirmed vs remaining dets ──────────────────
            r_dets4 = [r_dets3[i] for i in u_det3]
            if unconfirmed and r_dets4:
                cost4 = A.fuse_additive(A.iou_distance(unconfirmed, r_dets4),
                                        self._reid_cost(unconfirmed, r_dets4, gated=False), w_iou)
                m4, u_unc, u_det4 = A.linear_assignment(cost4, thresh=0.5)
            else:
                m4, u_unc, u_det4 = [], range(len(unconfirmed)), range(len(r_dets4))
            for it, idd in m4:
                unconfirmed[it].update(r_dets4[idd], self.frame_id, float(cost4[it, idd]))
                activated.append(unconfirmed[it])
            for it in u_unc:
                unconfirmed[it].mark_removed(); removed_now.append(unconfirmed[it])

            # ── new tracks ──────────────────────────────────────────────
            for i in u_det4:
                det = r_dets4[i]
                if det.score >= self.det_thresh:
                    det.activate(self.kalman_filter, self.frame_id)
                    activated.append(det)

            # ── age out lost ────────────────────────────────────────────
            for t in lost_pool:
                if self.frame_id - t.end_frame > self.max_time_lost:
                    t.mark_removed(); removed_now.append(t)

            # ── refresh pools ───────────────────────────────────────────
            self.tracked[cls_id] = [t for t in self.tracked[cls_id]
                                    if t.state == TrackState.Tracked]
            self.tracked[cls_id] = A.join_tracks(
                A.join_tracks(self.tracked[cls_id], activated), refind)
            self.lost[cls_id] = A.sub_tracks(self.lost[cls_id], self.tracked[cls_id])
            self.lost[cls_id].extend(lost_now)
            self.lost[cls_id] = A.sub_tracks(self.lost[cls_id], self.removed[cls_id])
            self.removed[cls_id].extend(removed_now)
            self.tracked[cls_id], self.lost[cls_id] = A.remove_duplicates(
                self.tracked[cls_id], self.lost[cls_id], self.frame_id)

            output[cls_id] = [t for t in self.tracked[cls_id] if t.is_activated]

        return output