"""Multi-class JDE tracker — orchestration only.

`MCJDETracker` is now a thin lifecycle controller. All motion math lives in
`MotionModel`, all matching policy in `Associator`. The per-class cascade reads
as a short sequence of stage calls:

    predict → GMC → stage-1 fuse → stage-2 IoU → stage-3 re-ID revive
            → spawn/retire → bookkeeping

Two online robustness upgrades are wired in here (the policy lives in the two
collaborators, the *decisions* live here):

  • conditional appearance EMA — a matched detection only refreshes a track's
    embedding/template when it is confident AND not visibly occluded, so the
    appearance bank is not poisoned in crowds (the main IDs source on the
    busy sequences).
  • lost-track revival — `Associator.reid_revive` reconnects a lost track to a
    far-away detection by appearance, then `MotionModel.revive` (ORU) heals the
    drifted Kalman state. Together they cut FM and post-occlusion IDs.

The model stays decoupled: `update()` receives pre-decoded detections
(`dict[cls_id] -> list[MCTrack]`).
"""

from collections import deque, defaultdict

import numpy as np

from falconmot.tracker import matching
from falconmot.tracker.config import TrackerCfg
from falconmot.tracker.motion import MotionModel
from falconmot.tracker.association import Associator
from .basetrack import MCBaseTrack, TrackState

id2cls = {
    0: 'pedestrian', 1: 'people',   2: 'bicycle',  3: 'car',
    4: 'van',        5: 'truck',    6: 'tricycle', 7: 'awning-tricycle',
    8: 'bus',        9: 'motor',
}


class MCTrack(MCBaseTrack):
    """A single tracklet of one object class. Kalman math is delegated to MotionModel."""

    def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
        self.cls_id = cls_id
        self._tlwh = np.asarray(tlwh, dtype=np.float32)

        self.motion = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.alpha = 0.9
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)

        # QAM value-space appearance template (set by the tracker from the dense map)
        self.template = None
        self.pred_xy = None

        self.curr_tlwh = np.asarray(tlwh, dtype=np.float32)
        self.tlwh_deque = deque([], maxlen=30)

    # ───────────────────────── appearance memory ─────────────────────────
    def update_template(self, feat, gain):
        """Absorb a value-space template by `gain` (0 = freeze, 1 = replace)."""
        if feat is None or gain <= 0.0:
            return
        feat = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        if self.template is None:
            self.template = feat
        else:
            self.template = (1.0 - gain) * self.template + gain * feat
            tn = np.linalg.norm(self.template)
            if tn > 0:
                self.template = self.template / tn

    def update_features(self, feat, gain=None):
        """EMA-update the embedding. `gain` is the per-frame absorption rate;
        at construction time `gain=None` initialises the bank outright."""
        feat = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        elif gain is not None and gain > 0.0:
            self.smooth_feat = (1.0 - gain) * self.smooth_feat + gain * feat
        self.features.append(feat)
        sn = np.linalg.norm(self.smooth_feat)
        if sn > 0:
            self.smooth_feat = self.smooth_feat / sn

    # ──────────────────────────── lifecycle ──────────────────────────────
    def activate(self, motion, frame_id):
        self.motion = motion
        self.track_id = self.next_id(self.cls_id)
        self.mean, self.covariance = motion.initiate(self._tlwh)
        self.curr_tlwh = self._tlwh
        self.track_len = 0
        self.state = TrackState.Tracked
        self.tlwh_deque.append((frame_id, self._tlwh))
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, gap=1, app_gain=0.0, new_id=False):
        """Revive a lost track. ORU heals the drifted Kalman state across `gap`."""
        last_tlwh = self.tlwh_deque[-1][1] if self.tlwh_deque else self.tlwh
        self.mean, self.covariance = self.motion.revive(
            self.mean, self.covariance, last_tlwh, new_track.tlwh, gap, new_track.score)
        self.update_features(new_track.curr_feat, gain=app_gain)
        self.update_template(new_track.template, gain=app_gain)

        self.curr_tlwh = new_track.curr_tlwh
        self.tlwh_deque.append((frame_id, new_track.curr_tlwh))
        self.track_len = 0
        self.frame_id = frame_id
        self.state = TrackState.Tracked
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, app_gain=0.0):
        """Update a matched track. `app_gain` scales how much the embedding absorbs."""
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.motion.correct(
            self.mean, self.covariance, new_tlwh, new_track.score)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        self.curr_tlwh = new_tlwh
        self.tlwh_deque.append((frame_id, new_track.curr_tlwh))
        self.update_features(new_track.curr_feat, gain=app_gain)
        self.update_template(new_track.template, gain=app_gain)

    # ───────────────────────────── geometry ──────────────────────────────
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(
            self.cls_id, self.track_id, self.start_frame, self.end_frame)


class MCJDETracker(object):
    """Per-class ByteTrack-style tracker. Decoupled from the detector."""

    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.num_classes = opt.num_classes
        self.det_thresh = opt.conf_thres

        self.cfg = TrackerCfg.from_opt(opt)
        self.motion = MotionModel(self.cfg)
        self.assoc = Associator(self.cfg)

        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        self.frame_id = 0
        self._curr_img = None

    # ───────────────────────────── inputs ────────────────────────────────
    def set_image(self, img):
        self._curr_img = img

    def set_dense(self, reid_dense, stride, ratio_x, ratio_y, pad_w=0.0, pad_h=0.0):
        self.motion.set_dense(reid_dense, stride, ratio_x, ratio_y, pad_w, pad_h)

    def reset(self):
        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)
        self.frame_id = 0
        self.motion = MotionModel(self.cfg)

    # ───────────────────────── appearance hygiene ────────────────────────
    def _app_quality(self, dets):
        """Per-detection EMA gain ∈ [0, app_gain_max].

        quality q = score * (1 - occ)^p   with occ = max IoU with another det.
        gain = q * app_gain_max. A confident, isolated detection absorbs fully;
        a faint or occluded one barely moves the embedding — no hard threshold.
        """
        n = len(dets)
        if n == 0:
            return []
        score = np.array([float(d.score) for d in dets], np.float32).clip(0.0, 1.0)
        occ = np.zeros(n, np.float32)
        if n > 1:
            tlbrs = [d.tlbr for d in dets]
            iou = matching.ious(tlbrs, tlbrs)
            np.fill_diagonal(iou, 0.0)
            occ = iou.max(axis=1).astype(np.float32)
        q = score * np.power(1.0 - occ, self.cfg.app_occ_power)
        return (q * self.cfg.app_gain_max).tolist()

    # ────────────────────────────── update ───────────────────────────────
    def update(self, dets_per_class, h_orig, w_orig):
        self.frame_id += 1
        if self.frame_id == 1:
            MCTrack.init_count(self.num_classes)

        output_tracks_dict = defaultdict(list)

        for cls_id in range(self.num_classes):
            dets = dets_per_class.get(cls_id, [])
            gains = self._app_quality(dets)

            # split confirmed vs unconfirmed (first-frame-only) tracks
            unconfirmed, tracked = [], []
            for t in self.tracked_tracks_dict[cls_id]:
                (tracked if t.is_activated else unconfirmed).append(t)

            # predict
            self.motion.predict(self.lost_tracks_dict[cls_id])
            self.motion.predict(tracked)
            pool = join_tracks(tracked, self.lost_tracks_dict[cls_id])

            # camera-motion compensation (computed once per frame, cached)
            self.motion.apply_gmc(pool, self._curr_img, self.frame_id)
            self.motion.apply_gmc(unconfirmed, self._curr_img, self.frame_id)

            # QAM templates for detections
            if self.cfg.use_qam and self.motion.has_dense:
                self.motion.sample_template(dets)

            activated, refined, newly_lost, removed = [], [], [], []

            def _match(track, det, di):
                """Apply a (track, det) match: update or revive, with adaptive EMA gain."""
                g = gains[di] if di < len(gains) else self.cfg.app_gain_max
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id, app_gain=g)
                    activated.append(track)
                else:
                    gap = self.frame_id - track.frame_id
                    track.re_activate(det, self.frame_id, gap=gap,
                                      app_gain=g, new_id=False)
                    refined.append(track)

            # ── stage 1: appearance + IoU (+ motion) ──
            emb_d = self.assoc.appearance_distance(pool, dets)
            iou_d, mot_d, w_mot = self.motion.gating(pool, dets)
            m, u_track, u_det = self.assoc.first(pool, dets, emb_d, iou_d, mot_d, w_mot)
            for it, idd in m:
                _match(pool[it], dets[idd], idd)

            remain_tracks = [pool[i] for i in u_track]
            remain_dets = [dets[i] for i in u_det]
            remain_didx = list(u_det)

            # ── stage 2: pure IoU ──
            m2, u_track2, u_det2 = self.assoc.iou(remain_tracks, remain_dets, self.cfg.iou_thresh)
            for it, idd in m2:
                _match(remain_tracks[it], remain_dets[idd], remain_didx[idd])

            r_tracks = [remain_tracks[i] for i in u_track2]
            r_dets = [remain_dets[i] for i in u_det2]
            r_didx = [remain_didx[i] for i in u_det2]

            # ── stage 3 (NEW): appearance-only revival of lost tracks ──
            lost_remain = [t for t in r_tracks if t.state == TrackState.Lost]
            if lost_remain and r_dets:
                m3, _, u_det3 = self.assoc.reid_revive(lost_remain, r_dets)
                revived = set()
                for it, idd in m3:
                    t = lost_remain[it]
                    gap = self.frame_id - t.frame_id
                    g = gains[r_didx[idd]] if r_didx[idd] < len(gains) else self.cfg.app_gain_max
                    t.re_activate(r_dets[idd], self.frame_id, gap=gap,
                                  app_gain=g, new_id=False)
                    refined.append(t)
                    revived.add(id(t))
                r_tracks = [t for t in r_tracks if id(t) not in revived]
                r_dets = [r_dets[i] for i in u_det3]
                r_didx = [r_didx[i] for i in u_det3]

            # tracks still unmatched -> mark lost (only those not already lost)
            for t in r_tracks:
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    newly_lost.append(t)

            # ── unconfirmed tracks (IoU only) ──
            m4, u_unc, u_det4 = self.assoc.iou(unconfirmed, r_dets, self.cfg.unconfirmed_thresh)
            for it, idd in m4:
                g = gains[r_didx[idd]] if r_didx[idd] < len(gains) else self.cfg.app_gain_max
                unconfirmed[it].update(r_dets[idd], self.frame_id, app_gain=g)
                activated.append(unconfirmed[it])
            for it in u_unc:
                unconfirmed[it].mark_removed()
                removed.append(unconfirmed[it])

            # ── spawn new tracks ──
            for idd in u_det4:
                det = r_dets[idd]
                if det.score < self.det_thresh:
                    continue
                det.activate(self.motion, self.frame_id)
                activated.append(det)

            # ── retire stale lost tracks ──
            for t in self.lost_tracks_dict[cls_id]:
                if self.frame_id - t.end_frame > self.cfg.max_lost:
                    t.mark_removed()
                    removed.append(t)

            # ── bookkeeping ──
            self.tracked_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(
                join_tracks(self.tracked_tracks_dict[cls_id], activated), refined)
            self.lost_tracks_dict[cls_id] = sub_tracks(
                self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(newly_lost)
            self.lost_tracks_dict[cls_id] = sub_tracks(
                self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed)
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
                remove_duplicate_tracks(self.tracked_tracks_dict[cls_id],
                                        self.lost_tracks_dict[cls_id])

            output_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.is_activated]

        return output_tracks_dict


# ───────────────────────────── track-list utils ──────────────────────────
def join_tracks(t_list_a, t_list_b):
    exists, res = {}, []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_tracks(t_list_a, t_list_b):
    tracks = {t.track_id: t for t in t_list_a}
    for t in t_list_b:
        tracks.pop(t.track_id, None)
    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    p_dist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(p_dist < 0.15)
    dup_a, dup_b = [], []
    for p, q in zip(*pairs):
        time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        time_q = tracks_b[q].frame_id - tracks_b[q].start_frame
        if time_p > time_q:
            dup_b.append(q)
        else:
            dup_a.append(p)
    res_a = [t for i, t in enumerate(tracks_a) if i not in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if i not in dup_b]
    return res_a, res_b
