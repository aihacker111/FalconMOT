"""Multi-class JDE tracker.

`MCJDETracker` is model-decoupled: it receives pre-decoded detections
(`dict[cls_id] -> list[MCTrack]`) and runs an independent ByteTrack-style
association cascade per class, with Kalman motion prediction, global motion
compensation (GMC) and ReID-embedding fusion.
"""

from collections import deque, defaultdict

import numpy as np

from falconmot.tracker import matching
from falconmot.tracker.kalman_filter import KalmanFilter
from falconmot.tracker.gmc import GMC
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

        # Query Appearance-Motion: value-space appearance template (set by the
        # tracker from the dense map). Used to correlate against the next frame.
        self.template = None
        self.pred_xy = None    # appearance-predicted centre (orig coords), if any

        self.curr_tlwh = np.asarray(tlwh, dtype=np.float32)
        self.tlwh_deque = deque([], maxlen=30)

    def update_template(self, feat, alpha=0.9):
        """EMA-update the value-space appearance template (L2-normalised)."""
        if feat is None:
            return
        feat = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        if self.template is None:
            self.template = feat
        else:
            self.template = alpha * self.template + (1.0 - alpha) * feat
            tn = np.linalg.norm(self.template)
            if tn > 0:
                self.template = self.template / tn

    # def update_features(self, feat, alpha=None):
    #     """L2-normalise the embedding and update the EMA `smooth_feat`."""
    #     feat /= np.linalg.norm(feat)
    #     self.alpha = (1.0 - alpha) if alpha is not None else 0.9

    #     self.curr_feat = feat
    #     if self.smooth_feat is None:
    #         self.smooth_feat = feat
    #     else:
    #         self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
    #     self.features.append(feat)
    #     self.smooth_feat /= np.linalg.norm(self.smooth_feat)
    def update_features(self, feat, score=None):
        """L2-normalise the embedding and update the EMA `smooth_feat`."""
        feat /= np.linalg.norm(feat)
        
        # [QUAN TRỌNG] Phục hồi biến curr_feat để dùng cho file matching.py
        self.curr_feat = feat 
        
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if score is not None:
                # Điểm số thấp -> alpha cao -> tin tưởng nhiều hơn vào lịch sử đặc trưng cũ (smooth_feat)
                alpha = 1.0 - 0.1 * score
            else:
                alpha = 0.90
                
            self.smooth_feat = alpha * self.smooth_feat + (1.0 - alpha) * feat
            
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

    # @staticmethod
    # def multi_gmc(stracks, H=np.eye(2, 3)):
    #     """Apply a global-motion-compensation affine `H` to track states."""
    #     if len(stracks) == 0:
    #         return
    #     R = H[:2, :2]
    #     # Use a single (larger) uniform scale factor for numerical stability.
    #     larger_scale = max(R[0, 0], R[1, 1])
    #     R = np.array([[larger_scale, 0], [0, larger_scale]])
    #     R8x8 = np.kron(np.eye(4, dtype=float), R)
    #     t = H[:2, 2]
    #     for i, st in enumerate(stracks):
    #         mean = R8x8.dot(st.mean)
    #         mean[:2] += t
    #         st.mean = mean
    #         st.covariance = R8x8.dot(st.covariance).dot(R8x8.transpose())
    @staticmethod
    def multi_gmc(tracks, H):
        if H is None or np.array_equal(H, np.eye(2, 3)):
            return

        R = H[:2, :2]
        t = H[:2, 2]

        # Tạo ma trận biến đổi trạng thái 8x8 cho Kalman Filter (x, y, a, h, vx, vy, va, vh)
        R_8x8 = np.eye(8, dtype=np.float32)
        R_8x8[:2, :2] = R      # Xoay vị trí trung tâm x, y
        R_8x8[4:6, 4:6] = R    # Xoay vận tốc vx, vy

        for track in tracks:
            if track.mean is not None:
                # 1. Warp vị trí trung tâm (x, y)
                track.mean[:2] = np.dot(R, track.mean[:2]) + t
                # 2. Warp vận tốc trung tâm (vx, vy)
                track.mean[4:6] = np.dot(R, track.mean[4:6])
                # 3. Warp toàn bộ ma trận hiệp phương sai hệ thống
                track.covariance = np.dot(R_8x8, np.dot(track.covariance, R_8x8.T))

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
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.mean, self.covariance = self.kalman_filter.update_nsa(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), new_track.score
        )
        self.update_features(new_track.curr_feat)
        self.update_template(new_track.template)

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
            self.update_template(new_track.template, alpha=0.9)

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
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

        self.gmc = GMC(method='sparseOptFlow', verbose=[None, False])
        self._curr_img = None   # set via set_image() before update()

        # ── Query Appearance-Motion (QAM) config ──
        self.use_am       = getattr(opt, 'use_appearance_motion', False)
        self.legacy_fuse  = getattr(opt, 'legacy_fuse', False)
        self.am_tau       = getattr(opt, 'am_tau', 0.07)
        self.am_kappa     = getattr(opt, 'am_kappa', 0.1)
        self.am_beta      = getattr(opt, 'am_beta', 4.0)
        self.w_app        = getattr(opt, 'am_w_app', 1.0)
        self.w_iou        = getattr(opt, 'am_w_iou', 1.0)
        self.match_thresh = getattr(opt, 'match_thresh', 0.7)
        self.proximity_gate = getattr(opt, 'proximity_thresh', 0.95)
        self.motion_gate    = getattr(opt, 'motion_gate', 0.9)

        # per-frame dense appearance state (set via set_dense())
        self._dense_hat = None    # [C,H,W] L2-normalised current map (torch)
        self._dense_raw = None    # [C,H,W] raw current map (torch, for sampling)
        self._tf = None           # (stride, ratio, pad_w, pad_h)

    def set_image(self, img):
        """Provide the raw BGR frame (for GMC) before calling update()."""
        self._curr_img = img

    def set_dense(self, reid_dense, stride, ratio_x, ratio_y, pad_w=0.0, pad_h=0.0):
        """Provide the current-frame dense appearance map + image transform.

        Coordinate convention must MATCH the box decode in the postprocessor:
          • plain resize (this repo): ratio_x=net_w/orig_w, ratio_y=net_h/orig_h,
            pad_w=pad_h=0 (anisotropic, no letterbox).
          • letterbox: ratio_x=ratio_y=min(net/orig), pad centred.

        Args:
            reid_dense : torch [C,H,W] value-projected appearance map (or None).
            stride     : feature stride of the map w.r.t. network input.
            ratio_x, ratio_y, pad_w, pad_h : net<->orig mapping (per axis).
        """
        if reid_dense is None:
            self._dense_hat = self._dense_raw = self._tf = None
            return
        from falconmot.tracker import appearance_motion as am
        self._dense_raw = reid_dense
        self._dense_hat = am.normalize_dense(reid_dense)
        self._tf = (float(stride), float(ratio_x), float(ratio_y),
                    float(pad_w), float(pad_h))

    @staticmethod
    def _centers_orig(items):
        """(N,2) box centres in original-image coords from a list of MCTrack."""
        return np.array([[t.tlwh[0] + t.tlwh[2] * 0.5,
                          t.tlwh[1] + t.tlwh[3] * 0.5] for t in items],
                        dtype=np.float32)

    def _sample_det_templates(self, dets):
        """Sample each detection's value-space appearance template from the map."""
        if self._dense_raw is None or len(dets) == 0:
            return
        import torch
        from falconmot.tracker import appearance_motion as am
        stride, rx, ry, pw, ph = self._tf
        cmap = am.orig_to_map(self._centers_orig(dets), stride, rx, ry, pw, ph)
        xy = torch.from_numpy(cmap).to(self._dense_raw.device).float()
        feats = am.sample_dense(self._dense_raw, xy).cpu().numpy()   # (D, C)
        for d, f in zip(dets, feats):
            n = np.linalg.norm(f)
            d.template = (f / n) if n > 0 else f

    def _appearance_motion(self, pool, dets):
        """Predict pool-track centres by correlation; return (d_mot, w_mot).

        d_mot : (T, D) size-adaptive Gaussian motion distance.
        w_mot : (T,)   entropy-gated confidence (0 for tracks without template).
        """
        import torch
        from falconmot.tracker import appearance_motion as am
        T, D = len(pool), len(dets)
        d_mot = np.ones((T, D), dtype=np.float32)
        w_mot = np.zeros((T,), dtype=np.float32)

        tmpls, idx = [], []
        for i, t in enumerate(pool):
            if t.template is not None:
                tmpls.append(t.template)
                idx.append(i)
        if not idx:
            return d_mot, w_mot

        stride, rx, ry, pw, ph = self._tf
        tt = torch.from_numpy(np.stack(tmpls)).to(self._dense_hat.device).float()
        tt = torch.nn.functional.normalize(tt, dim=1)
        centers_map, ent, _peak = am.predict_centers(tt, self._dense_hat, tau=self.am_tau)
        centers_map = centers_map.cpu().numpy()
        ent = ent.cpu().numpy()

        pred_orig = am.map_to_orig(centers_map, stride, rx, ry, pw, ph)        # (K, 2)
        det_orig  = self._centers_orig(dets)                                  # (D, 2)
        sizes = np.array([np.sqrt(max(pool[i].tlwh[2] * pool[i].tlwh[3], 1.0))
                          for i in idx], dtype=np.float32)
        dm = matching.motion_distance(pred_orig, det_orig, sizes, kappa=self.am_kappa)
        wm = am.confidence_from_entropy(ent, beta=self.am_beta)

        for k, i in enumerate(idx):
            d_mot[i] = dm[k]
            w_mot[i] = wm[k]
            pool[i].pred_xy = pred_orig[k]
        return d_mot, w_mot

    def reset(self):
        """Clear all state — call between sequences."""
        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

    def remove_cross_class_duplicates(self, tracked_dict, lost_dict, iou_thresh=0.75):
        """
        Loại bỏ các Bounding Box đè nhau giữa các Class khác nhau để tăng MOTA,
        nhưng giữ lại Track ngầm (đẩy vào Lost) để bảo vệ IDF1 khỏi hiện tượng Class Flickering.
        """
        all_tracks = []
        # Chỉ xét những track thực sự đang hiện diện (Tracked) trên màn hình ở frame này
        for cls_id in tracked_dict.keys():
            for t in tracked_dict[cls_id]:
                if t.is_activated and t.state == TrackState.Tracked:
                    all_tracks.append(t)

        if len(all_tracks) < 2:
            return

        from falconmot.tracker import matching
        dists = matching.iou_distance(all_tracks, all_tracks)
        
        # Tìm các cặp đè nhau (IoU > thresh)
        pairs = np.where(dists < (1.0 - iou_thresh))
        
        for i, j in zip(pairs[0], pairs[1]):
            if i >= j: 
                continue
                
            t1 = all_tracks[i]
            t2 = all_tracks[j]
            
            # Chỉ xử lý nếu 2 track khác Class
            if t1.cls_id == t2.cls_id:
                continue
                
            # Đảm bảo cả 2 chưa bị xử lý bởi vòng lặp trước đó
            if t1.state != TrackState.Tracked or t2.state != TrackState.Tracked:
                continue
                
            # LUẬT 1: Bảo vệ Track trưởng thành (Chống nhiễu chớp nhoáng)
            if t1.track_len > 3 and t2.track_len <= 2:
                t2.mark_removed()
                continue
            elif t2.track_len > 3 and t1.track_len <= 2:
                t1.mark_removed()
                continue

            # LUẬT 2: Soft Suppression dựa trên Score (Bảo vệ IDF1)
            if t1.score >= t2.score:
                t2.mark_lost() # Tạm thời ẩn t2 đi, không xóa vĩnh viễn
            else:
                t1.mark_lost() # Tạm thời ẩn t1 đi, không xóa vĩnh viễn

    # def update(self, dets_per_class, h_orig, w_orig):
    #     """Run one tracking step.

    #     Args:
    #         dets_per_class : dict[cls_id] -> list[MCTrack] high-conf detections
    #         h_orig, w_orig : original image height / width
    #     Returns:
    #         dict[cls_id] -> list[MCTrack] of active output tracks
    #     """
    #     self.frame_id += 1
    #     if self.frame_id == 1:
    #         MCTrack.init_count(self.num_classes)

    #     activated_tracks_dict = defaultdict(list)
    #     refined_tracks_dict = defaultdict(list)
    #     lost_tracks_dict = defaultdict(list)
    #     removed_tracks_dict = defaultdict(list)
    #     output_tracks_dict = defaultdict(list)

    #     # Global motion compensation — computed once per frame.
    #     gmc_H = None
    #     if self._curr_img is not None:
    #         try:
    #             gmc_result = self.gmc.apply(self._curr_img, None)
    #             gmc_H = gmc_result[0] if isinstance(gmc_result, tuple) else gmc_result
    #         except Exception:
    #             gmc_H = None

    #     for cls_id in range(self.num_classes):
    #         cls_detects = dets_per_class.get(cls_id, [])

    #         unconfirmed_dict = defaultdict(list)
    #         tracked_tracks_dict = defaultdict(list)
    #         for track in self.tracked_tracks_dict[cls_id]:
    #             if not track.is_activated:
    #                 unconfirmed_dict[cls_id].append(track)
    #             else:
    #                 tracked_tracks_dict[cls_id].append(track)

    #         MCTrack.multi_predict(self.lost_tracks_dict[cls_id])
    #         MCTrack.multi_predict(tracked_tracks_dict[cls_id])

    #         track_pool_dict = defaultdict(list)
    #         track_pool_dict[cls_id] = join_tracks(
    #             tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

    #         if gmc_H is not None:
    #             MCTrack.multi_gmc(track_pool_dict[cls_id], gmc_H)
    #             MCTrack.multi_gmc(unconfirmed_dict[cls_id], gmc_H)

    #         # Sample value-space appearance templates for detections (QAM).
    #         if self.use_am and self._dense_raw is not None:
    #             self._sample_det_templates(cls_detects)

    #         # --- Step 1: first association — appearance + IoU (+ motion) ---
    #         pool = track_pool_dict[cls_id]
    #         emb_d = matching.embedding_distance(pool, cls_detects)
    #         iou_d = matching.iou_distance(pool, cls_detects)

    #         if self.legacy_fuse or not self.use_am:
    #             cost = matching.fuse_score_three(iou_d, emb_d, cls_detects)
    #             thr  = 0.6
    #         else:
    #             d_mot, w_mot = None, None
    #             if self._dense_hat is not None and len(pool) > 0 and len(cls_detects) > 0:
    #                 d_mot, w_mot = self._appearance_motion(pool, cls_detects)
    #             cost = matching.fuse_loglik(
    #                 emb_d, iou_d, d_mot, w_mot,
    #                 w_app=self.w_app, w_iou=self.w_iou,
    #                 proximity_gate=self.proximity_gate, motion_gate=self.motion_gate)
    #             thr  = self.match_thresh
    #         matches, u_track, u_detection = matching.linear_assignment(cost, thresh=thr)

    #         for i_tracked, i_det in matches:
    #             track = track_pool_dict[cls_id][i_tracked]
    #             det = cls_detects[i_det]
    #             if track.state == TrackState.Tracked:
    #                 track.update(det, self.frame_id)
    #                 activated_tracks_dict[cls_id].append(track)
    #             else:
    #                 track.re_activate(det, self.frame_id, new_id=False)
    #                 refined_tracks_dict[cls_id].append(track)

    #         # --- Step 2: second association — IoU only ---
    #         cls_detects_r = [cls_detects[i] for i in u_detection]
    #         r_tracked_tracks = [track_pool_dict[cls_id][i]
    #                             for i in u_track if track_pool_dict[cls_id][i].state]
    #         dist_iou = matching.iou_distance(r_tracked_tracks, cls_detects_r)
    #         matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.8)

    #         for i_tracked, i_det in matches:
    #             track = r_tracked_tracks[i_tracked]
    #             det = cls_detects_r[i_det]
    #             if track.state == TrackState.Tracked:
    #                 track.update(det, self.frame_id)
    #                 activated_tracks_dict[cls_id].append(track)
    #             else:
    #                 track.re_activate(det, self.frame_id, new_id=False)
    #                 refined_tracks_dict[cls_id].append(track)

    #         # Tracks still unmatched after step 2 -> mark lost.
    #         for it in u_track:
    #             track = r_tracked_tracks[it]
    #             if track.state != TrackState.Lost:
    #                 track.mark_lost()
    #                 lost_tracks_dict[cls_id].append(track)

    #         # --- Unconfirmed tracks (only one beginning frame) ---
    #         cls_detects_unc = [cls_detects_r[i] for i in u_detection]
    #         dist_iou = matching.iou_distance(unconfirmed_dict[cls_id], cls_detects_unc)
    #         matches, u_unconfirmed, u_detection = matching.linear_assignment(dist_iou, thresh=0.5)

    #         for i_tracked, i_det in matches:
    #             unconfirmed_dict[cls_id][i_tracked].update(cls_detects_unc[i_det], self.frame_id)
    #             activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
    #         for it in u_unconfirmed:
    #             unconfirmed_dict[cls_id][it].mark_removed()
    #             removed_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][it])

    #         # --- Initialise new tracks ---
    #         for i_new in u_detection:
    #             track = cls_detects_unc[i_new]
    #             if track.score < self.det_thresh:
    #                 continue
    #             track.activate(self.kalman_filter, self.frame_id)
    #             activated_tracks_dict[cls_id].append(track)

    #         # --- Age out lost tracks ---
    #         for track in self.lost_tracks_dict[cls_id]:
    #             if self.frame_id - track.end_frame > self.max_time_lost:
    #                 track.mark_removed()
    #                 removed_tracks_dict[cls_id].append(track)

    #         # --- Bookkeeping ---
    #         self.tracked_tracks_dict[cls_id] = [
    #             t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked]
    #         self.tracked_tracks_dict[cls_id] = join_tracks(
    #             join_tracks(self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id]),
    #             refined_tracks_dict[cls_id])
    #         self.lost_tracks_dict[cls_id] = sub_tracks(
    #             self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
    #         self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
    #         self.lost_tracks_dict[cls_id] = sub_tracks(
    #             self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
    #         self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
    #         self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
    #             remove_duplicate_tracks(self.tracked_tracks_dict[cls_id],
    #                                     self.lost_tracks_dict[cls_id])
    #         output_tracks_dict[cls_id] = [
    #             t for t in self.tracked_tracks_dict[cls_id] if t.is_activated]
            
    #     return output_tracks_dict
    def update(self, dets_per_class, h_orig, w_orig):
        """Run one tracking step."""
        self.frame_id += 1
        if self.frame_id == 1:
            MCTrack.init_count(self.num_classes)

        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)

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

            # Sample value-space appearance templates for detections (QAM).
            if self.use_am and self._dense_raw is not None:
                self._sample_det_templates(cls_detects)

            # --- Step 1: first association — appearance + IoU (+ motion) ---
            pool = track_pool_dict[cls_id]
            emb_d = matching.embedding_distance(pool, cls_detects)
            iou_d = matching.iou_distance(pool, cls_detects)

            if self.legacy_fuse or not self.use_am:
                cost = matching.fuse_score_three(iou_d, emb_d, cls_detects)
                thr  = 0.6
            else:
                d_mot, w_mot = None, None
                if self._dense_hat is not None and len(pool) > 0 and len(cls_detects) > 0:
                    d_mot, w_mot = self._appearance_motion(pool, cls_detects)
                cost = matching.fuse_loglik(
                    emb_d, iou_d, d_mot, w_mot,
                    w_app=self.w_app, w_iou=self.w_iou,
                    proximity_gate=self.proximity_gate, motion_gate=self.motion_gate)
                thr  = self.match_thresh
            matches, u_track, u_detection = matching.linear_assignment(cost, thresh=thr)

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

        # =====================================================================
        # [NOVELTY]: Re-Bookkeeping Đa Lớp (Cross-Class)
        # Thực hiện SAU KHI đã tổng hợp đủ Tracks từ TẤT CẢ các Classes
        # =====================================================================
        self.remove_cross_class_duplicates(self.tracked_tracks_dict, self.lost_tracks_dict, iou_thresh=0.75)
        
        output_tracks_dict = defaultdict(list)
        for cls_id in range(self.num_classes):
            new_tracked = []
            for t in self.tracked_tracks_dict[cls_id]:
                # Tái phân bổ các Track vừa bị ép ngầm (mark_lost/mark_removed) về đúng chỗ
                if t.state == TrackState.Tracked:
                    new_tracked.append(t)
                elif t.state == TrackState.Lost:
                    self.lost_tracks_dict[cls_id].append(t)
                elif t.state == TrackState.Removed:
                    self.removed_tracks_dict[cls_id].append(t)
            
            self.tracked_tracks_dict[cls_id] = new_tracked
            
            # Xuất kết quả cuối cùng (Loại bỏ triệt để Ghost Output)
            output_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.is_activated and t.state == TrackState.Tracked
            ]
            
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