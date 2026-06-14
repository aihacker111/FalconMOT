"""
track.py — a single tracked object for FalconMOT.

Consolidates the good parts of the old MCTrackV2:
  * NSA Kalman update     — measurement noise scaled by detection confidence
  * feature gallery        — last-N raw L2-normed embeddings (no EMA drift),
                             now stamped with frame_id for time-decay matching
  * adaptive EMA           — smooth_feat update rate scales with match quality
The dense-ReID/motion machinery of the old tracker is gone (FalconJDE emits a
per-query embedding directly, so no dense feature map is needed).
"""

from collections import deque

import numpy as np

from falconmot.tracking_utils.kalman_filter import KalmanFilter
from .base import BaseTrack, TrackState


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat, num_classes, cls_id, buff_size=30):
        self.cls_id = cls_id
        self._tlwh  = np.asarray(tlwh, dtype=np.float32)

        self.kalman_filter = None
        self.mean = self.covariance = None
        self.is_activated = False

        self.score      = float(score)
        self.track_len  = 0
        self.curr_tlwh  = self._tlwh.copy()

        # gallery of (L2-normed feature, frame_id) for time-decayed matching
        self.feat_gallery = deque([], maxlen=buff_size)
        self.curr_feat    = None
        self.smooth_feat  = None
        self.alpha        = 0.9

        self.update_features(feat, frame_id=0, match_conf=1.0)

    # ------------------------------------------------------------------ feats
    def update_features(self, feat, frame_id, match_conf=1.0):
        feat = np.asarray(feat, dtype=np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        self.curr_feat = feat
        self.feat_gallery.append((feat, frame_id))

        self.alpha = 0.5 + 0.4 * float(np.clip(match_conf, 0.0, 1.0))
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-8)

    # ------------------------------------------------------------------ kalman
    def predict(self):
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if not tracks:
            return
        mean = np.asarray([t.mean.copy() for t in tracks])
        cov  = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                mean[i][7] = 0
        mean, cov = Track.shared_kalman.multi_predict(mean, cov)
        for i in range(len(tracks)):
            tracks[i].mean, tracks[i].covariance = mean[i], cov[i]

    @staticmethod
    def multi_gmc(tracks, H=np.eye(2, 3)):
        if not tracks:
            return
        R = H[:2, :2]
        s = max(R[0, 0], R[1, 1])
        R8 = np.kron(np.eye(4), np.array([[s, 0], [0, s]]))
        t = H[:2, 2]
        for tr in tracks:
            m = R8.dot(tr.mean)
            m[:2] += t
            tr.mean = m
            tr.covariance = R8.dot(tr.covariance).dot(R8.T)

    # ------------------------------------------------------------------ life
    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id(self.cls_id)
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.curr_tlwh = self._tlwh.copy()
        self.track_len = 0
        self.state = TrackState.Tracked
        self.is_activated = (frame_id == 1)
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, det, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update_nsa(
            self.mean, self.covariance, self.tlwh_to_xyah(det.tlwh), confidence=det.score)
        self.update_features(det.curr_feat, frame_id, match_conf=det.score)
        self.curr_tlwh = det.curr_tlwh.copy()
        self.track_len = 0
        self.frame_id = frame_id
        self.score = det.score
        self.state = TrackState.Tracked
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id(self.cls_id)

    def update(self, det, frame_id, match_cost=0.0):
        self.frame_id = frame_id
        self.track_len += 1
        self.mean, self.covariance = self.kalman_filter.update_nsa(
            self.mean, self.covariance, self.tlwh_to_xyah(det.tlwh), confidence=det.score)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = det.score
        self.curr_tlwh = det.tlwh.copy()
        match_conf = 1.0 - float(np.clip(match_cost, 0.0, 1.0))
        self.update_features(det.curr_feat, frame_id, match_conf=match_conf)

    # ------------------------------------------------------------------ coords
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

    def __repr__(self):
        return f'Track(c{self.cls_id}-{self.track_id}|{self.start_frame}-{self.end_frame})'
