"""MotionModel — everything about *how an object moves*.

Wraps three motion sources behind one interface so the tracker never touches a
Kalman matrix or a dense feature map directly:

  • Kalman filter  : constant-velocity state, NSA-scaled correction, and the
                     Observation-Centric Re-Update (ORU) used when a lost track
                     is revived after a gap.
  • GMC            : per-frame affine camera-motion compensation (computed once,
                     cached by frame id, applied to any track subset).
  • QAM            : appearance-as-motion — predicts each track's centre in the
                     current frame by cross-frame correlation on the dense ReID
                     map, and turns that into a size-adaptive motion distance
                     with an entropy-gated per-track confidence.

The model exposes exactly what `Associator` needs: the geometric IoU distance,
the QAM motion distance, and the motion confidence weights. It also owns the
state-mutating Kalman ops (`initiate`, `predict`, `correct`, `revive`) so the
`MCTrack` lifecycle methods stay one-liners.
"""

from __future__ import annotations

import numpy as np

from falconmot.tracker import matching
from falconmot.tracker import appearance_motion as am
from falconmot.tracker.kalman_filter import KalmanFilter
from falconmot.tracker.gmc import GMC


def _xyah(tlwh):
    """(t,l,w,h) -> (cx,cy,aspect,h)."""
    ret = np.asarray(tlwh, dtype=np.float32).copy()
    ret[:2] += ret[2:] / 2.0
    ret[2] /= max(ret[3], 1e-6)
    return ret


class MotionModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.gmc = GMC(method='sparseOptFlow', verbose=[None, False]) if cfg.use_gmc else None

        # per-frame dense appearance state (set via set_dense)
        self._dense_hat = None     # [C,H,W] L2-normalised (torch)
        self._dense_raw = None     # [C,H,W] raw (torch, for sampling)
        self._tf = None            # (stride, ratio_x, ratio_y, pad_w, pad_h)

        # GMC affine cached per frame id so the per-class loop recomputes once
        self._gmc_frame = -1
        self._gmc_H = None

    # ─────────────────────────── dense / QAM I/O ──────────────────────────
    def set_dense(self, reid_dense, stride, ratio_x, ratio_y, pad_w=0.0, pad_h=0.0):
        if reid_dense is None:
            self._dense_hat = self._dense_raw = self._tf = None
            return
        self._dense_raw = reid_dense
        self._dense_hat = am.normalize_dense(reid_dense)
        self._tf = (float(stride), float(ratio_x), float(ratio_y),
                    float(pad_w), float(pad_h))

    @property
    def has_dense(self):
        return self._dense_hat is not None

    def sample_template(self, dets):
        """Sample each detection's value-space appearance template from the map."""
        if self._dense_raw is None or len(dets) == 0:
            return
        import torch
        stride, rx, ry, pw, ph = self._tf
        centers = np.array([[d.tlwh[0] + d.tlwh[2] * 0.5,
                             d.tlwh[1] + d.tlwh[3] * 0.5] for d in dets], np.float32)
        cmap = am.orig_to_map(centers, stride, rx, ry, pw, ph)
        xy = torch.from_numpy(cmap).to(self._dense_raw.device).float()
        feats = am.sample_dense(self._dense_raw, xy).cpu().numpy()
        for d, f in zip(dets, feats):
            n = np.linalg.norm(f)
            d.template = (f / n) if n > 0 else f

    # ─────────────────────────── Kalman state ops ─────────────────────────
    def initiate(self, tlwh):
        return self.kf.initiate(_xyah(tlwh))

    def predict(self, tracks):
        """Vectorised constant-velocity prediction (in place)."""
        if not tracks:
            return
        from falconmot.tracker.basetrack import TrackState
        mean = np.asarray([t.mean.copy() for t in tracks])
        cov = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                mean[i][7] = 0          # zero height-velocity for non-active tracks
        mean, cov = self.kf.multi_predict(mean, cov)
        for i, t in enumerate(tracks):
            t.mean, t.covariance = mean[i], cov[i]

    def correct(self, mean, cov, tlwh, score):
        """Measurement update; NSA-scaled when enabled."""
        meas = _xyah(tlwh)
        if self.cfg.use_nsa:
            return self.kf.update_nsa(mean, cov, meas, score)
        return self.kf.update(mean, cov, meas)

    def revive(self, mean, cov, last_tlwh, new_tlwh, gap, score):
        """Observation-Centric Re-Update (OC-SORT).

        After a `gap`-frame absence the constant-velocity state has drifted and
        its velocity points the wrong way. We walk the Kalman state along the
        *virtual* straight trajectory between the last real observation and the
        new one, re-aligning velocity/covariance, then apply the real update.
        This is the main defence against post-occlusion ID switches.
        """
        if not self.cfg.use_oru or gap <= 1:
            return self.correct(mean, cov, new_tlwh, score)

        last = _xyah(last_tlwh)
        new = _xyah(new_tlwh)
        steps = int(min(gap, 30))
        for k in range(1, steps):
            virt = last + (new - last) * (k / float(steps))
            mean, cov = self.kf.predict(mean, cov)
            # virtual observations are interpolated, not measured -> trust them little
            mean, cov = self.kf.update_nsa(mean, cov, virt, 0.3)
        return self.correct(mean, cov, new_tlwh, score)

    # ─────────────────────────────── GMC ──────────────────────────────────
    def apply_gmc(self, tracks, img, frame_id):
        """Apply this frame's affine camera-motion compensation to `tracks`."""
        if self.gmc is None or img is None or not tracks:
            return
        if frame_id != self._gmc_frame:
            try:
                res = self.gmc.apply(img, None)
                self._gmc_H = res[0] if isinstance(res, tuple) else res
            except Exception:
                self._gmc_H = None
            self._gmc_frame = frame_id
        H = self._gmc_H
        if H is None:
            return
        R = H[:2, :2]
        s = max(R[0, 0], R[1, 1])
        R8 = np.kron(np.eye(4), np.array([[s, 0], [0, s]]))
        t = H[:2, 2]
        for st in tracks:
            m = R8.dot(st.mean)
            m[:2] += t
            st.mean = m
            st.covariance = R8.dot(st.covariance).dot(R8.T)

    # ───────────────────────── association costs ──────────────────────────
    def iou_distance(self, tracks, dets):
        return matching.iou_distance(tracks, dets)

    def gating(self, pool, dets):
        """Return (iou_dist, motion_dist, motion_weight) for a (pool, dets) pair.

        motion_dist / motion_weight come from QAM when a dense map is available
        and the cue is enabled; otherwise motion is absent (None) and the
        associator falls back to appearance + IoU.
        """
        iou_d = matching.iou_distance(pool, dets)
        if not (self.cfg.use_qam and self.has_dense and pool and dets):
            return iou_d, None, None

        import torch
        T, D = len(pool), len(dets)
        d_mot = np.ones((T, D), dtype=np.float32)
        w_mot = np.zeros((T,), dtype=np.float32)

        tmpls, idx = [], []
        for i, t in enumerate(pool):
            if t.template is not None:
                tmpls.append(t.template)
                idx.append(i)
        if not idx:
            return iou_d, d_mot, w_mot

        stride, rx, ry, pw, ph = self._tf
        tt = torch.from_numpy(np.stack(tmpls)).to(self._dense_hat.device).float()
        tt = torch.nn.functional.normalize(tt, dim=1)
        centers_map, ent, _peak = am.predict_centers(tt, self._dense_hat, tau=self.cfg.qam_tau)
        centers_map = centers_map.cpu().numpy()
        ent = ent.cpu().numpy()

        pred_orig = am.map_to_orig(centers_map, stride, rx, ry, pw, ph)
        det_orig = np.array([[d.tlwh[0] + d.tlwh[2] * 0.5,
                              d.tlwh[1] + d.tlwh[3] * 0.5] for d in dets], np.float32)
        sizes = np.array([np.sqrt(max(pool[i].tlwh[2] * pool[i].tlwh[3], 1.0))
                          for i in idx], np.float32)
        dm = matching.motion_distance(pred_orig, det_orig, sizes, kappa=self.cfg.qam_kappa)
        wm = am.confidence_from_entropy(ent, beta=self.cfg.qam_beta)

        for k, i in enumerate(idx):
            d_mot[i] = dm[k]
            w_mot[i] = wm[k]
            pool[i].pred_xy = pred_orig[k]
        return iou_d, d_mot, w_mot
