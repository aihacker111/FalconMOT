# """Multi-class JDE tracker.

# `MCJDETracker` is model-decoupled: it receives pre-decoded detections
# (`dict[cls_id] -> list[MCTrack]`) and runs an independent ByteTrack-style
# association cascade per class, with Kalman motion prediction, global motion
# compensation (GMC) and ReID-embedding fusion.
# """

# from collections import deque, defaultdict

# import numpy as np

# from falconmot.tracker import matching
# from falconmot.tracking_utils.kalman_filter import KalmanFilter
# from falconmot.tracking_utils.gmc import GMC
# from .basetrack import MCBaseTrack, TrackState

# # VisDrone 10-class id -> name (used for visualisation / debugging)
# id2cls = {
#     0: 'pedestrian', 1: 'people',   2: 'bicycle',  3: 'car',
#     4: 'van',        5: 'truck',    6: 'tricycle', 7: 'awning-tricycle',
#     8: 'bus',        9: 'motor',
# }


# class MCTrack(MCBaseTrack):
#     """A single tracklet of one object class."""

#     shared_kalman = KalmanFilter()

#     def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
#         self.cls_id = cls_id
#         self._tlwh = np.asarray(tlwh, dtype=np.float32)

#         self.kalman_filter = None
#         self.mean, self.covariance = None, None
#         self.is_activated = False

#         self.score = score
#         self.track_len = 0

#         self.smooth_feat = None
#         self.alpha = 0.9
#         self.update_features(temp_feat)
#         self.features = deque([], maxlen=buff_size)

#         self.curr_tlwh = np.asarray(tlwh, dtype=np.float32)
#         self.tlwh_deque = deque([], maxlen=30)

#     def update_features(self, feat, alpha=None):
#         """L2-normalise the embedding and update the EMA `smooth_feat`."""
#         feat /= np.linalg.norm(feat)
#         self.alpha = (1.0 - alpha) if alpha is not None else 0.9

#         self.curr_feat = feat
#         if self.smooth_feat is None:
#             self.smooth_feat = feat
#         else:
#             self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
#         self.features.append(feat)
#         self.smooth_feat /= np.linalg.norm(self.smooth_feat)

#     @staticmethod
#     def multi_predict(tracks):
#         """Vectorised Kalman prediction for a list of tracks."""
#         if len(tracks) == 0:
#             return
#         multi_mean = np.asarray([t.mean.copy() for t in tracks])
#         multi_covariance = np.asarray([t.covariance for t in tracks])
#         for i, t in enumerate(tracks):
#             if t.state != TrackState.Tracked:
#                 multi_mean[i][7] = 0
#         multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(
#             multi_mean, multi_covariance)
#         for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#             tracks[i].mean = mean
#             tracks[i].covariance = cov

#     @staticmethod
#     def multi_gmc(stracks, H=np.eye(2, 3)):
#         """Apply a global-motion-compensation affine `H` to track states."""
#         if len(stracks) == 0:
#             return
#         R = H[:2, :2]
#         # Use a single (larger) uniform scale factor for numerical stability.
#         larger_scale = max(R[0, 0], R[1, 1])
#         R = np.array([[larger_scale, 0], [0, larger_scale]])
#         R8x8 = np.kron(np.eye(4, dtype=float), R)
#         t = H[:2, 2]
#         for i, st in enumerate(stracks):
#             mean = R8x8.dot(st.mean)
#             mean[:2] += t
#             st.mean = mean
#             st.covariance = R8x8.dot(st.covariance).dot(R8x8.transpose())

#     def activate(self, kalman_filter, frame_id):
#         """Start a new track."""
#         self.kalman_filter = kalman_filter
#         self.track_id = self.next_id(self.cls_id)
#         self.mean, self.covariance = self.kalman_filter.initiate(
#             self.tlwh_to_xyah(self._tlwh))
#         self.curr_tlwh = self._tlwh
#         self.track_len = 0
#         self.state = TrackState.Tracked
#         self.tlwh_deque.append((frame_id, self._tlwh))

#         # Only first-frame detections are reported as confirmed immediately.
#         if frame_id == 1:
#             self.is_activated = True

#         self.frame_id = frame_id
#         self.start_frame = frame_id

#     def re_activate(self, new_track, frame_id, new_id=False):
#         """Reactivate a lost track from a new matched detection."""
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
#         self.update_features(new_track.curr_feat)

#         self.curr_tlwh = new_track.curr_tlwh
#         self.tlwh_deque.append((frame_id, new_track.curr_tlwh))

#         self.track_len = 0
#         self.frame_id = frame_id
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         if new_id:
#             self.track_id = self.next_id(self.cls_id)

#     def update(self, new_track, frame_id, alpha=None, update_feature=True):
#         """Update a matched track with a new detection."""
#         self.frame_id = frame_id
#         self.track_len += 1

#         new_tlwh = new_track.tlwh
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.score = new_track.score

#         self.curr_tlwh = new_tlwh
#         self.tlwh_deque.append((frame_id, new_track.curr_tlwh))
#         if update_feature:
#             self.update_features(new_track.curr_feat, alpha)

#     @property
#     def tlwh(self):
#         """Current position as (top-left-x, top-left-y, width, height)."""
#         if self.mean is None:
#             return self._tlwh.copy()
#         ret = self.mean[:4].copy()
#         ret[2] *= ret[3]
#         ret[:2] -= ret[2:] / 2
#         return ret

#     @property
#     def tlbr(self):
#         """Current position as (min-x, min-y, max-x, max-y)."""
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret

#     @staticmethod
#     def tlwh_to_xyah(tlwh):
#         """Convert (t, l, w, h) -> (center-x, center-y, aspect, height)."""
#         ret = np.asarray(tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= ret[3]
#         return ret

#     def to_xyah(self):
#         return self.tlwh_to_xyah(self.tlwh)

#     def __repr__(self):
#         return 'OT_({}-{})_({}-{})'.format(
#             self.cls_id, self.track_id, self.start_frame, self.end_frame)


# class MCJDETracker(object):
#     """Per-class ByteTrack-style tracker for ECDet/Falcon-JDE detections.

#     The model is decoupled: `update()` receives pre-decoded detections rather
#     than running the network internally.
#     """

#     def __init__(self, opt, frame_rate=30):
#         self.opt = opt
#         self.num_classes = opt.num_classes
#         self.det_thresh = opt.conf_thres
#         self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
#         self.max_time_lost = self.buffer_size

#         self.tracked_tracks_dict = defaultdict(list)
#         self.lost_tracks_dict = defaultdict(list)
#         self.removed_tracks_dict = defaultdict(list)

#         self.frame_id = 0
#         self.kalman_filter = KalmanFilter()

#         self.gmc = GMC(method='sparseOptFlow', verbose=[None, False])
#         self._curr_img = None   # set via set_image() before update()

#     def set_image(self, img):
#         """Provide the raw BGR frame (for GMC) before calling update()."""
#         self._curr_img = img

#     def reset(self):
#         """Clear all state — call between sequences."""
#         self.tracked_tracks_dict = defaultdict(list)
#         self.lost_tracks_dict = defaultdict(list)
#         self.removed_tracks_dict = defaultdict(list)
#         self.frame_id = 0
#         self.kalman_filter = KalmanFilter()

#     def update(self, dets_per_class, h_orig, w_orig):
#         """Run one tracking step.

#         Args:
#             dets_per_class : dict[cls_id] -> list[MCTrack] high-conf detections
#             h_orig, w_orig : original image height / width
#         Returns:
#             dict[cls_id] -> list[MCTrack] of active output tracks
#         """
#         self.frame_id += 1
#         if self.frame_id == 1:
#             MCTrack.init_count(self.num_classes)

#         activated_tracks_dict = defaultdict(list)
#         refined_tracks_dict = defaultdict(list)
#         lost_tracks_dict = defaultdict(list)
#         removed_tracks_dict = defaultdict(list)
#         output_tracks_dict = defaultdict(list)

#         # Global motion compensation — computed once per frame.
#         gmc_H = None
#         if self._curr_img is not None:
#             try:
#                 gmc_result = self.gmc.apply(self._curr_img, None)
#                 gmc_H = gmc_result[0] if isinstance(gmc_result, tuple) else gmc_result
#             except Exception:
#                 gmc_H = None

#         for cls_id in range(self.num_classes):
#             cls_detects = dets_per_class.get(cls_id, [])

#             unconfirmed_dict = defaultdict(list)
#             tracked_tracks_dict = defaultdict(list)
#             for track in self.tracked_tracks_dict[cls_id]:
#                 if not track.is_activated:
#                     unconfirmed_dict[cls_id].append(track)
#                 else:
#                     tracked_tracks_dict[cls_id].append(track)

#             MCTrack.multi_predict(self.lost_tracks_dict[cls_id])
#             MCTrack.multi_predict(tracked_tracks_dict[cls_id])

#             track_pool_dict = defaultdict(list)
#             track_pool_dict[cls_id] = join_tracks(
#                 tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

#             if gmc_H is not None:
#                 MCTrack.multi_gmc(track_pool_dict[cls_id], gmc_H)
#                 MCTrack.multi_gmc(unconfirmed_dict[cls_id], gmc_H)

#             # --- Step 1: first association — ReID embedding + IoU ---
#             dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detects)
#             dist_iou = matching.iou_distance(track_pool_dict[cls_id], cls_detects)
#             dist_iou = matching.fuse_score_three(dist_iou, dists, cls_detects)
#             matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.6)

#             for i_tracked, i_det in matches:
#                 track = track_pool_dict[cls_id][i_tracked]
#                 det = cls_detects[i_det]
#                 if track.state == TrackState.Tracked:
#                     track.update(det, self.frame_id)
#                     activated_tracks_dict[cls_id].append(track)
#                 else:
#                     track.re_activate(det, self.frame_id, new_id=False)
#                     refined_tracks_dict[cls_id].append(track)

#             # --- Step 2: second association — IoU only ---
#             cls_detects_r = [cls_detects[i] for i in u_detection]
#             r_tracked_tracks = [track_pool_dict[cls_id][i]
#                                 for i in u_track if track_pool_dict[cls_id][i].state]
#             dist_iou = matching.iou_distance(r_tracked_tracks, cls_detects_r)
#             matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.8)

#             for i_tracked, i_det in matches:
#                 track = r_tracked_tracks[i_tracked]
#                 det = cls_detects_r[i_det]
#                 if track.state == TrackState.Tracked:
#                     track.update(det, self.frame_id)
#                     activated_tracks_dict[cls_id].append(track)
#                 else:
#                     track.re_activate(det, self.frame_id, new_id=False)
#                     refined_tracks_dict[cls_id].append(track)

#             # Tracks still unmatched after step 2 -> mark lost.
#             for it in u_track:
#                 track = r_tracked_tracks[it]
#                 if track.state != TrackState.Lost:
#                     track.mark_lost()
#                     lost_tracks_dict[cls_id].append(track)

#             # --- Unconfirmed tracks (only one beginning frame) ---
#             cls_detects_unc = [cls_detects_r[i] for i in u_detection]
#             dist_iou = matching.iou_distance(unconfirmed_dict[cls_id], cls_detects_unc)
#             matches, u_unconfirmed, u_detection = matching.linear_assignment(dist_iou, thresh=0.5)

#             for i_tracked, i_det in matches:
#                 unconfirmed_dict[cls_id][i_tracked].update(cls_detects_unc[i_det], self.frame_id)
#                 activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
#             for it in u_unconfirmed:
#                 unconfirmed_dict[cls_id][it].mark_removed()
#                 removed_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][it])

#             # --- Initialise new tracks ---
#             for i_new in u_detection:
#                 track = cls_detects_unc[i_new]
#                 if track.score < self.det_thresh:
#                     continue
#                 track.activate(self.kalman_filter, self.frame_id)
#                 activated_tracks_dict[cls_id].append(track)

#             # --- Age out lost tracks ---
#             for track in self.lost_tracks_dict[cls_id]:
#                 if self.frame_id - track.end_frame > self.max_time_lost:
#                     track.mark_removed()
#                     removed_tracks_dict[cls_id].append(track)

#             # --- Bookkeeping ---
#             self.tracked_tracks_dict[cls_id] = [
#                 t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked]
#             self.tracked_tracks_dict[cls_id] = join_tracks(
#                 join_tracks(self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id]),
#                 refined_tracks_dict[cls_id])
#             self.lost_tracks_dict[cls_id] = sub_tracks(
#                 self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
#             self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
#             self.lost_tracks_dict[cls_id] = sub_tracks(
#                 self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
#             self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
#             self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
#                 remove_duplicate_tracks(self.tracked_tracks_dict[cls_id],
#                                         self.lost_tracks_dict[cls_id])

#             output_tracks_dict[cls_id] = [
#                 t for t in self.tracked_tracks_dict[cls_id] if t.is_activated]

#         return output_tracks_dict


# def join_tracks(t_list_a, t_list_b):
#     """Union of two track lists, de-duplicated by track_id."""
#     exists = {}
#     res = []
#     for t in t_list_a:
#         exists[t.track_id] = 1
#         res.append(t)
#     for t in t_list_b:
#         if not exists.get(t.track_id, 0):
#             exists[t.track_id] = 1
#             res.append(t)
#     return res


# def sub_tracks(t_list_a, t_list_b):
#     """Tracks in `t_list_a` whose ids are not in `t_list_b`."""
#     tracks = {t.track_id: t for t in t_list_a}
#     for t in t_list_b:
#         tracks.pop(t.track_id, None)
#     return list(tracks.values())


# def remove_duplicate_tracks(tracks_a, tracks_b):
#     """Drop near-identical (IoU > 0.85) tracks, keeping the longer-lived one."""
#     p_dist = matching.iou_distance(tracks_a, tracks_b)
#     pairs = np.where(p_dist < 0.15)
#     dup_a, dup_b = [], []
#     for p, q in zip(*pairs):
#         time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
#         time_q = tracks_b[q].frame_id - tracks_b[q].start_frame
#         if time_p > time_q:
#             dup_b.append(q)
#         else:
#             dup_a.append(p)
#     res_a = [t for i, t in enumerate(tracks_a) if i not in dup_a]
#     res_b = [t for i, t in enumerate(tracks_b) if i not in dup_b]
#     return res_a, res_b



"""Multi-class JDE tracker.

`MCJDETracker` is model-decoupled: it receives pre-decoded detections
(`dict[cls_id] -> list[MCTrack]`) and runs an independent ByteTrack-style
association cascade per class, with Kalman motion prediction, global motion
compensation (GMC) and ReID-embedding fusion.
"""

from collections import deque, defaultdict

import numpy as np

from falconmot.tracker import matching
from falconmot.tracking_utils.kalman_filter import KalmanFilter
from falconmot.tracking_utils.gmc import GMC
from .basetrack import MCBaseTrack, TrackState

# VisDrone 10-class id -> name (used for visualisation / debugging)
id2cls = {
    0: 'pedestrian', 1: 'people',   2: 'bicycle',  3: 'car',
    4: 'van',        5: 'truck',    6: 'tricycle', 7: 'awning-tricycle',
    8: 'bus',        9: 'motor',
}


class MCTrack(MCBaseTrack):
    """A single tracklet of one object class."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
        self.cls_id = cls_id
        self._tlwh = np.asarray(tlwh, dtype=np.float32)

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.alpha = 0.9
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)

        self.curr_tlwh = np.asarray(tlwh, dtype=np.float32)
        self.tlwh_deque = deque([], maxlen=30)

    def update_features(self, feat, alpha=None):
        """L2-normalise the embedding and update the EMA `smooth_feat`."""
        feat /= np.linalg.norm(feat)
        self.alpha = (1.0 - alpha) if alpha is not None else 0.9

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    @staticmethod
    def multi_predict(tracks):
        """Vectorised Kalman prediction for a list of tracks."""
        if len(tracks) == 0:
            return
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_covariance = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            tracks[i].mean = mean
            tracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Apply a global-motion-compensation affine `H` to track states."""
        if len(stracks) == 0:
            return
        R = H[:2, :2]
        # Use a single (larger) uniform scale factor for numerical stability.
        larger_scale = max(R[0, 0], R[1, 1])
        R = np.array([[larger_scale, 0], [0, larger_scale]])
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = H[:2, 2]
        for i, st in enumerate(stracks):
            mean = R8x8.dot(st.mean)
            mean[:2] += t
            st.mean = mean
            st.covariance = R8x8.dot(st.covariance).dot(R8x8.transpose())

    def activate(self, kalman_filter, frame_id):
        """Start a new track."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id(self.cls_id)
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))
        self.curr_tlwh = self._tlwh
        self.track_len = 0
        self.state = TrackState.Tracked
        self.tlwh_deque.append((frame_id, self._tlwh))

        # Only first-frame detections are reported as confirmed immediately.
        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track from a new matched detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.update_features(new_track.curr_feat)

        self.curr_tlwh = new_track.curr_tlwh
        self.tlwh_deque.append((frame_id, new_track.curr_tlwh))

        self.track_len = 0
        self.frame_id = frame_id
        self.state = TrackState.Tracked
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, alpha=None, update_feature=True):
        """Update a matched track with a new detection."""
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        self.curr_tlwh = new_tlwh
        self.tlwh_deque.append((frame_id, new_track.curr_tlwh))
        if update_feature:
            self.update_features(new_track.curr_feat, alpha)

    @property
    def tlwh(self):
        """Current position as (top-left-x, top-left-y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Current position as (min-x, min-y, max-x, max-y)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert (t, l, w, h) -> (center-x, center-y, aspect, height)."""
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
    """Per-class ByteTrack-style tracker for ECDet/Falcon-JDE detections.

    The model is decoupled: `update()` receives pre-decoded detections rather
    than running the network internally.
    """

    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.num_classes = opt.num_classes
        self.det_thresh = opt.conf_thres
        # Trọng số nhánh ReID trong association (xem matching.fuse_score_three).
        #   emb_weight=1.0 -> như cũ; 0.0 -> IoU thuần (để kiểm chứng ReID hại).
        self.emb_weight = float(getattr(opt, 'emb_weight', 1.0))
        self.emb_gate   = float(getattr(opt, 'emb_gate', 0.0))
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

        self.gmc = GMC(method='sparseOptFlow', verbose=[None, False])
        self._curr_img = None   # set via set_image() before update()

    def set_image(self, img):
        """Provide the raw BGR frame (for GMC) before calling update()."""
        self._curr_img = img

    def reset(self):
        """Clear all state — call between sequences."""
        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

    def update(self, dets_per_class, h_orig, w_orig):
        """Run one tracking step.

        Args:
            dets_per_class : dict[cls_id] -> list[MCTrack] high-conf detections
            h_orig, w_orig : original image height / width
        Returns:
            dict[cls_id] -> list[MCTrack] of active output tracks
        """
        self.frame_id += 1
        if self.frame_id == 1:
            MCTrack.init_count(self.num_classes)

        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        # Global motion compensation — computed once per frame.
        gmc_H = None
        if self._curr_img is not None:
            try:
                gmc_result = self.gmc.apply(self._curr_img, None)
                gmc_H = gmc_result[0] if isinstance(gmc_result, tuple) else gmc_result
            except Exception:
                gmc_H = None

        for cls_id in range(self.num_classes):
            cls_detects = dets_per_class.get(cls_id, [])

            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_tracks_dict[cls_id].append(track)

            MCTrack.multi_predict(self.lost_tracks_dict[cls_id])
            MCTrack.multi_predict(tracked_tracks_dict[cls_id])

            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(
                tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            if gmc_H is not None:
                MCTrack.multi_gmc(track_pool_dict[cls_id], gmc_H)
                MCTrack.multi_gmc(unconfirmed_dict[cls_id], gmc_H)

            # --- Step 1: first association — ReID embedding + IoU ---
            dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detects)
            dist_iou = matching.iou_distance(track_pool_dict[cls_id], cls_detects)
            dist_iou = matching.fuse_score_three(
                dist_iou, dists, cls_detects,
                emb_weight=self.emb_weight, emb_gate=self.emb_gate)
            matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.6)

            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detects[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            # --- Step 2: second association — IoU only ---
            cls_detects_r = [cls_detects[i] for i in u_detection]
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state]
            dist_iou = matching.iou_distance(r_tracked_tracks, cls_detects_r)
            matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.8)

            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detects_r[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            # Tracks still unmatched after step 2 -> mark lost.
            for it in u_track:
                track = r_tracked_tracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            # --- Unconfirmed tracks (only one beginning frame) ---
            cls_detects_unc = [cls_detects_r[i] for i in u_detection]
            dist_iou = matching.iou_distance(unconfirmed_dict[cls_id], cls_detects_unc)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dist_iou, thresh=0.5)

            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detects_unc[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                unconfirmed_dict[cls_id][it].mark_removed()
                removed_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][it])

            # --- Initialise new tracks ---
            for i_new in u_detection:
                track = cls_detects_unc[i_new]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            # --- Age out lost tracks ---
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            # --- Bookkeeping ---
            self.tracked_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(
                join_tracks(self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id]),
                refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(
                self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(
                self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
                remove_duplicate_tracks(self.tracked_tracks_dict[cls_id],
                                        self.lost_tracks_dict[cls_id])

            output_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.is_activated]

        return output_tracks_dict


def join_tracks(t_list_a, t_list_b):
    """Union of two track lists, de-duplicated by track_id."""
    exists = {}
    res = []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_tracks(t_list_a, t_list_b):
    """Tracks in `t_list_a` whose ids are not in `t_list_b`."""
    tracks = {t.track_id: t for t in t_list_a}
    for t in t_list_b:
        tracks.pop(t.track_id, None)
    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    """Drop near-identical (IoU > 0.85) tracks, keeping the longer-lived one."""
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