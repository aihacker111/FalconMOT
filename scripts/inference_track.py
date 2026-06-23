# """
# inference_track.py — Smooth Video Tracking Inference cho FalconJDE.

# Bản FULL TRACKING: bật toàn bộ năng lực tracker để output mượt nhất có thể.

# So với bản cũ, bản này thêm:
#   1. Query Appearance-Motion (QAM / UAM): nạp dense appearance map vào tracker mỗi
#      frame (set_dense). Đây là phần quan trọng nhất — giúp giữ track_id ổn định,
#      giảm ID-switch và giảm giật box. Bản cũ chạy tracker nhưng KHÔNG đưa dense map
#      -> tracker phải match bằng IoU đơn thuần -> nháy/đổi id nhiều hơn.
#   2. GMC (Global Motion Compensation): bù chuyển động camera — đã có sẵn trong
#      tracker.update(), chỉ cần set_image() đúng mỗi frame (đã làm).
#   3. Letterbox preprocessing tuỳ chọn (--letterbox): giữ đúng tỉ lệ ảnh -> detect
#      chính xác hơn -> track ổn định hơn. Mặc định plain-resize như bản cũ.
#   4. Smoothing 3 lớp ở hậu xử lý:
#         a. Lọc track quá ngắn (chống đốm nháy 1-2 frame).
#         b. Nội suy lấp khoảng trống (interpolation, chống mất box ngắn hạn).
#         c. Làm mượt quỹ đạo bằng moving-average có cửa sổ đối xứng (giảm rung box).
# """

# import os
# import cv2
# import torch
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm

# import _paths  # noqa: F401
# from falconmot.models.model import create_model, load_model
# from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
# from falconmot.tracker import FalconTracker, Track           # = MCJDETracker / MCTrack
# from falconmot.tracking_utils import visualization as vis
# from falconmot.tracking_utils.timer import Timer
# from falconmot.opts import opts

# # ==========================================
# # CẤU HÌNH CLASS MAP
# # ==========================================
# MERGED_CLS_MAP = {
#     1: 'pedestrian', 2: 'bicycle', 3: 'car', 4: 'truck',
#     5: 'tricycle', 6: 'bus', 7: 'motor'
# }
# # Model output (cls_id) 0-indexed
# CLS_NAMES_0_IDX = {k - 1: v for k, v in MERGED_CLS_MAP.items()}


# class FastVideoTracker:
#     def __init__(self, opt, video_fps: int):
#         self.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 and torch.cuda.is_available() else 'cpu'
#         self.num_cls = opt.num_classes
#         self.min_area = opt.min_box_area
#         self.net_w, self.net_h = opt.img_size           # img_size = (W, H)
#         self.use_letterbox = getattr(opt, 'letterbox', False)
#         self.use_am = getattr(opt, 'use_appearance_motion', False)

#         print(f'Loading model onto {self.device}...')
#         self.model = create_model(opt.arch, opt)
#         self.model = load_model(self.model, opt.load_model)
#         self.model = self.model.to(self.device).eval()

#         # ── QAM: yêu cầu model trả về dense appearance map ──
#         if self.use_am:
#             _m = getattr(self.model, 'module', self.model)
#             _m.return_reid_dense = True
#             print('[QAM] Query Appearance-Motion: ON (dense appearance enabled)')
#         else:
#             print('[QAM] Query Appearance-Motion: OFF (dùng IoU + sparse reid). '
#                   'Bật --use_appearance_motion để mượt hơn.')

#         self.postprocessor = FalconJDEPostProcessor(
#             num_classes=opt.num_classes,
#             num_top_queries=getattr(opt, 'K', 300),
#             conf_thres=opt.conf_thres,
#             use_focal_loss=True,
#         )
#         # Letterbox -> postprocessor cần biết net_hw để giải mã box đúng.
#         # Plain-resize -> KHÔNG set_net_hw (postprocessor dùng nhánh scale * orig).
#         if self.use_letterbox:
#             self.postprocessor.set_net_hw(self.net_h, self.net_w)

#         self.tracker = FalconTracker(opt, frame_rate=video_fps)
#         self.timer = Timer()
#         self._orig_sizes = None

#     # ----------------------------------------------------------------------
#     # Preprocess: trả về (tensor, transform) để dense map khớp với box decode.
#     # transform = (ratio_x, ratio_y, pad_w, pad_h)
#     # ----------------------------------------------------------------------
#     def preprocess(self, img_bgr):
#         orig_h, orig_w = img_bgr.shape[:2]

#         if self.use_letterbox:
#             r = min(self.net_h / orig_h, self.net_w / orig_w)
#             new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
#             resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
#             canvas = np.full((self.net_h, self.net_w, 3), 114, dtype=np.uint8)
#             pad_w = (self.net_w - new_w) * 0.5
#             pad_h = (self.net_h - new_h) * 0.5
#             top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
#             canvas[top:top + new_h, left:left + new_w] = resized
#             img_proc = canvas
#             transform = (r, r, pad_w, pad_h)
#         else:
#             img_proc = cv2.resize(img_bgr, (self.net_w, self.net_h),
#                                   interpolation=cv2.INTER_AREA)
#             transform = (self.net_w / orig_w, self.net_h / orig_h, 0.0, 0.0)

#         img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
#         img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
#         return img_tensor.unsqueeze(0).to(self.device), transform

#     def _decode_detections(self, res: dict) -> defaultdict:
#         dets = defaultdict(list)
#         if len(res['scores']) == 0:
#             return dets

#         boxes = res['boxes'].cpu().numpy()
#         scores = res['scores'].cpu().numpy()
#         labels = res['labels'].cpu().numpy()
#         reid = res['reid'].cpu().numpy() if 'reid' in res else None

#         ws = boxes[:, 2] - boxes[:, 0]
#         hs = boxes[:, 3] - boxes[:, 1]
#         valid_idx = np.where((ws > 0) & (hs > 0))[0]

#         for i in valid_idx:
#             cls_id = int(labels[i])
#             tlwh = np.array([boxes[i, 0], boxes[i, 1], ws[i], hs[i]], dtype=np.float32)
#             emb = reid[i] if reid is not None else np.zeros(1, dtype=np.float32)
#             dets[cls_id].append(Track(tlwh, float(scores[i]), emb, self.num_cls, cls_id))

#         return dets

#     def get_tracks_for_frame(self, frame_bgr):
#         """Chạy AI + tracker (full QAM) cho 1 frame, trả về list track."""
#         orig_h, orig_w = frame_bgr.shape[:2]
#         if self._orig_sizes is None:
#             self._orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

#         blob, transform = self.preprocess(frame_bgr)

#         self.timer.tic()
#         with torch.no_grad():
#             output = self.model(blob)
#             res = self.postprocessor(output, self._orig_sizes)[0]
#             dets = self._decode_detections(res)
#         self.timer.toc()

#         # GMC cần ảnh gốc
#         self.tracker.set_image(frame_bgr)

#         # ── QAM: nạp dense appearance map cho frame hiện tại ──
#         # Toạ độ phải khớp với cách postprocessor giải mã box (transform ở trên).
#         if isinstance(output, dict) and output.get('reid_dense') is not None:
#             rx, ry, pad_w, pad_h = transform
#             self.tracker.set_dense(
#                 output['reid_dense'][0],                       # [C,H,W]
#                 stride=output['reid_dense_stride'],
#                 ratio_x=rx, ratio_y=ry, pad_w=pad_w, pad_h=pad_h,
#             )

#         online_targets = self.tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

#         frame_results = []
#         for cls_id, tracks in online_targets.items():
#             for t in tracks:
#                 if t.track_id < 0:
#                     continue
#                 if t.curr_tlwh[2] * t.curr_tlwh[3] > self.min_area:
#                     frame_results.append((cls_id, t.track_id, t.curr_tlwh.copy(), t.score))

#         return frame_results


# # ==========================================================================
# # HẬU XỬ LÝ — 3 LỚP LÀM MƯỢT
# # ==========================================================================
# def _build_track_dict(all_results):
#     """Gom theo (cls_id, track_id): track_dict[cls][tid][frame] = (tlwh, score)."""
#     track_dict = defaultdict(lambda: defaultdict(dict))
#     for frame_id, targets in all_results.items():
#         for cls_id, trk_id, tlwh, score in targets:
#             track_dict[cls_id][trk_id][frame_id] = (np.asarray(tlwh, np.float32), score)
#     return track_dict


# def filter_short_tracks(track_dict, min_len=5):
#     """Bỏ track quá ngắn (đốm nháy 1-vài frame) — nguồn chính của hiện tượng nháy."""
#     removed = 0
#     for cls_id in list(track_dict.keys()):
#         for trk_id in list(track_dict[cls_id].keys()):
#             if len(track_dict[cls_id][trk_id]) < min_len:
#                 del track_dict[cls_id][trk_id]
#                 removed += 1
#     print(f"  • Lọc track ngắn (<{min_len} frames): bỏ {removed} track.")
#     return track_dict


# def interpolate_tracks(track_dict, max_gap=15):
#     """Nội suy tuyến tính lấp khoảng trống khi model miss object ngắn hạn."""
#     filled = 0
#     for cls_id, trks in track_dict.items():
#         for trk_id, frames_data in trks.items():
#             frame_ids = sorted(frames_data.keys())
#             for i in range(len(frame_ids) - 1):
#                 f1, f2 = frame_ids[i], frame_ids[i + 1]
#                 gap = f2 - f1
#                 if 1 < gap <= max_gap:
#                     box1, score1 = frames_data[f1]
#                     box2, score2 = frames_data[f2]
#                     for step in range(1, gap):
#                         w = step / gap
#                         f_interp = f1 + step
#                         frames_data[f_interp] = (
#                             box1 + (box2 - box1) * w,
#                             score1 + (score2 - score1) * w,
#                         )
#                         filled += 1
#     print(f"  • Nội suy (gap ≤ {max_gap}): điền {filled} box.")
#     return track_dict


# def smooth_tracks_moving_average(track_dict, window=5):
#     """Làm mượt quỹ đạo bằng moving-average cửa sổ đối xứng (giảm rung box).

#     Trung bình đối xứng -> không gây trễ (lag) như EMA. window lẻ là tốt nhất.
#     """
#     if window <= 1:
#         return track_dict
#     half = window // 2
#     for cls_id, trks in track_dict.items():
#         for trk_id, frames_data in trks.items():
#             frame_ids = sorted(frames_data.keys())
#             if len(frame_ids) < 3:
#                 continue
#             boxes = np.stack([frames_data[f][0] for f in frame_ids])   # (N,4)
#             scores = np.array([frames_data[f][1] for f in frame_ids])
#             n = len(frame_ids)
#             smoothed = boxes.copy()
#             for i in range(n):
#                 a, b = max(0, i - half), min(n, i + half + 1)
#                 smoothed[i] = boxes[a:b].mean(axis=0)
#             for i, f in enumerate(frame_ids):
#                 frames_data[f] = (smoothed[i].astype(np.float32), float(scores[i]))
#     print(f"  • Làm mượt quỹ đạo (moving-average window={window}).")
#     return track_dict


# def track_dict_to_frames(track_dict):
#     """Trả về results_smoothed[frame_id] = [(cls_id, trk_id, tlwh, score), ...]."""
#     results = defaultdict(list)
#     for cls_id, trks in track_dict.items():
#         for trk_id, frames_data in trks.items():
#             for frame_id, (tlwh, score) in frames_data.items():
#                 results[frame_id].append((cls_id, trk_id, tlwh, score))
#     return results


# def main():
#     opt = opts()
#     opt.parser.add_argument('--input_video', type=str, required=True)
#     opt.parser.add_argument('--output_video', type=str, default='output_smoothed.mp4')
#     opt.parser.add_argument('--max_interp_gap', type=int, default=15,
#                             help='Khoảng trống tối đa để nối track (chống mất box ngắn).')
#     opt.parser.add_argument('--min_track_len', type=int, default=5,
#                             help='Bỏ track ngắn hơn ngưỡng này (chống nháy đốm).')
#     opt.parser.add_argument('--smooth_window', type=int, default=5,
#                             help='Cửa sổ moving-average làm mượt quỹ đạo (lẻ, 1 = tắt).')
#     opt.parser.add_argument('--letterbox', action='store_true',
#                             help='Dùng letterbox thay vì plain-resize (chính xác hơn).')
#     parsed_opt = opt.init()

#     if not os.path.exists(parsed_opt.input_video):
#         raise FileNotFoundError(f"Cannot find input video: {parsed_opt.input_video}")

#     cap = cv2.VideoCapture(parsed_opt.input_video)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     tracker = FastVideoTracker(parsed_opt, video_fps=fps)

#     # -----------------------------------------------------
#     # PHASE 1: INFERENCE + TRACKING (chạy AI + full tracker)
#     # -----------------------------------------------------
#     print("\n--- PHASE 1: AI Inference + Tracking ---")
#     raw_results = {}
#     frame_id = 0
#     pbar = tqdm(total=total_frames, desc="Tracking")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_id += 1
#         raw_results[frame_id] = tracker.get_tracks_for_frame(frame)
#         pbar.update(1)
#     pbar.close()
#     cap.release()

#     # -----------------------------------------------------
#     # PHASE 2: SMOOTHING (lọc ngắn -> nội suy -> làm mượt)
#     # -----------------------------------------------------
#     print("\n--- PHASE 2: Smoothing Tracks ---")
#     track_dict = _build_track_dict(raw_results)
#     track_dict = filter_short_tracks(track_dict, min_len=parsed_opt.min_track_len)
#     track_dict = interpolate_tracks(track_dict, max_gap=parsed_opt.max_interp_gap)
#     track_dict = smooth_tracks_moving_average(track_dict, window=parsed_opt.smooth_window)
#     smoothed_results = track_dict_to_frames(track_dict)

#     # -----------------------------------------------------
#     # PHASE 3: RENDER VIDEO
#     # -----------------------------------------------------
#     print("\n--- PHASE 3: Rendering Video ---")
#     cap = cv2.VideoCapture(parsed_opt.input_video)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(parsed_opt.output_video, fourcc, fps, (width, height))

#     avg_fps = 1.0 / max(1e-5, tracker.timer.average_time)
#     frame_id = 0
#     pbar = tqdm(total=total_frames, desc="Rendering")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_id += 1

#         targets = smoothed_results.get(frame_id, [])
#         tlwhs, tids, scores = defaultdict(list), defaultdict(list), defaultdict(list)
#         for (cls_id, trk_id, tlwh, score) in targets:
#             tlwhs[cls_id].append(tlwh)
#             tids[cls_id].append(trk_id)
#             scores[cls_id].append(score)

#         annotated_frame = vis.plot_tracks(
#             image=frame,
#             tlwhs_dict=tlwhs,
#             obj_ids_dict=tids,
#             num_classes=tracker.num_cls,
#             scores=scores,
#             frame_id=frame_id,
#             fps=avg_fps,
#             cls_id2name=CLS_NAMES_0_IDX,
#         )
#         out.write(annotated_frame)
#         pbar.update(1)
#     pbar.close()
#     cap.release()
#     out.release()

#     print(f"\n✅ Hoàn thành! Video đã lưu tại: {parsed_opt.output_video}")
#     print(f"🚀 Tốc độ AI inference: {avg_fps:.2f} FPS")


# if __name__ == '__main__':
#     main()





"""
inference_track.py — Smooth Video Tracking Inference cho FalconJDE.

Bản FULL TRACKING: bật toàn bộ năng lực tracker để output mượt nhất có thể.

So với bản cũ, bản này thêm:
  1. Query Appearance-Motion (QAM / UAM): nạp dense appearance map vào tracker mỗi
     frame (set_dense). Đây là phần quan trọng nhất — giúp giữ track_id ổn định,
     giảm ID-switch và giảm giật box. Bản cũ chạy tracker nhưng KHÔNG đưa dense map
     -> tracker phải match bằng IoU đơn thuần -> nháy/đổi id nhiều hơn.
  2. GMC (Global Motion Compensation): bù chuyển động camera — đã có sẵn trong
     tracker.update(), chỉ cần set_image() đúng mỗi frame (đã làm).
  3. Letterbox preprocessing tuỳ chọn (--letterbox): giữ đúng tỉ lệ ảnh -> detect
     chính xác hơn -> track ổn định hơn. Mặc định plain-resize như bản cũ.
  4. Smoothing 3 lớp ở hậu xử lý:
        a. Lọc track quá ngắn (chống đốm nháy 1-2 frame).
        b. Nội suy lấp khoảng trống (interpolation, chống mất box ngắn hạn).
        c. Làm mượt quỹ đạo bằng moving-average có cửa sổ đối xứng (giảm rung box).
"""

import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import _paths  # noqa: F401
from falconmot.models.model import create_model, load_model
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracker import FalconTracker, Track           # = MCJDETracker / MCTrack
from falconmot.tracking_utils import visualization as vis
from falconmot.tracking_utils.timer import Timer
from falconmot.opts import opts

# ==========================================
# CẤU HÌNH CLASS MAP
# ==========================================
MERGED_CLS_MAP = {
    1: 'pedestrian', 2: 'bicycle', 3: 'car', 4: 'truck',
    5: 'tricycle', 6: 'bus', 7: 'motor'
}
# Model output (cls_id) 0-indexed
CLS_NAMES_0_IDX = {k - 1: v for k, v in MERGED_CLS_MAP.items()}


class FastVideoTracker:
    def __init__(self, opt, video_fps: int):
        self.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 and torch.cuda.is_available() else 'cpu'
        self.num_cls = opt.num_classes
        self.min_area = opt.min_box_area
        self.net_w, self.net_h = opt.img_size           # img_size = (W, H)
        self.use_letterbox = getattr(opt, 'letterbox', False)
        self.use_am = getattr(opt, 'use_appearance_motion', False)

        print(f'Loading model onto {self.device}...')
        self.model = create_model(opt.arch, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(self.device).eval()

        # ── QAM: yêu cầu model trả về dense appearance map ──
        if self.use_am:
            _m = getattr(self.model, 'module', self.model)
            _m.return_reid_dense = True
            print('[QAM] Query Appearance-Motion: ON (dense appearance enabled)')
        else:
            print('[QAM] Query Appearance-Motion: OFF (dùng IoU + sparse reid). '
                  'Bật --use_appearance_motion để mượt hơn.')

        self.postprocessor = FalconJDEPostProcessor(
            num_classes=opt.num_classes,
            num_top_queries=getattr(opt, 'K', 300),
            conf_thres=opt.conf_thres,
            use_focal_loss=True,
        )
        # Letterbox -> postprocessor cần biết net_hw để giải mã box đúng.
        # Plain-resize -> KHÔNG set_net_hw (postprocessor dùng nhánh scale * orig).
        if self.use_letterbox:
            self.postprocessor.set_net_hw(self.net_h, self.net_w)

        self.tracker = FalconTracker(opt, frame_rate=video_fps)
        self.timer = Timer()
        self._orig_sizes = None

    # ----------------------------------------------------------------------
    # Preprocess: trả về (tensor, transform) để dense map khớp với box decode.
    # transform = (ratio_x, ratio_y, pad_w, pad_h)
    # ----------------------------------------------------------------------
    def preprocess(self, img_bgr):
        orig_h, orig_w = img_bgr.shape[:2]

        if self.use_letterbox:
            r = min(self.net_h / orig_h, self.net_w / orig_w)
            new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
            resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((self.net_h, self.net_w, 3), 114, dtype=np.uint8)
            pad_w = (self.net_w - new_w) * 0.5
            pad_h = (self.net_h - new_h) * 0.5
            top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
            canvas[top:top + new_h, left:left + new_w] = resized
            img_proc = canvas
            transform = (r, r, pad_w, pad_h)
        else:
            img_proc = cv2.resize(img_bgr, (self.net_w, self.net_h),
                                  interpolation=cv2.INTER_AREA)
            transform = (self.net_w / orig_w, self.net_h / orig_h, 0.0, 0.0)

        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device), transform

    def _decode_detections(self, res: dict) -> defaultdict:
        dets = defaultdict(list)
        if len(res['scores']) == 0:
            return dets

        boxes = res['boxes'].cpu().numpy()
        scores = res['scores'].cpu().numpy()
        labels = res['labels'].cpu().numpy()
        reid = res['reid'].cpu().numpy() if 'reid' in res else None

        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        valid_idx = np.where((ws > 0) & (hs > 0))[0]

        for i in valid_idx:
            cls_id = int(labels[i])
            tlwh = np.array([boxes[i, 0], boxes[i, 1], ws[i], hs[i]], dtype=np.float32)
            emb = reid[i] if reid is not None else np.zeros(1, dtype=np.float32)
            dets[cls_id].append(Track(tlwh, float(scores[i]), emb, self.num_cls, cls_id))

        return dets

    def get_tracks_for_frame(self, frame_bgr):
        """Chạy AI + tracker (full QAM) cho 1 frame, trả về list track."""
        orig_h, orig_w = frame_bgr.shape[:2]
        if self._orig_sizes is None:
            self._orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

        blob, transform = self.preprocess(frame_bgr)

        self.timer.tic()
        with torch.no_grad():
            output = self.model(blob)
            res = self.postprocessor(output, self._orig_sizes)[0]
            dets = self._decode_detections(res)
        self.timer.toc()

        # GMC cần ảnh gốc
        self.tracker.set_image(frame_bgr)

        # ── QAM: nạp dense appearance map cho frame hiện tại ──
        # Toạ độ phải khớp với cách postprocessor giải mã box (transform ở trên).
        if isinstance(output, dict) and output.get('reid_dense') is not None:
            rx, ry, pad_w, pad_h = transform
            self.tracker.set_dense(
                output['reid_dense'],                          # [C,H,W] (model đã bỏ batch)
                stride=output['reid_dense_stride'],
                ratio_x=rx, ratio_y=ry, pad_w=pad_w, pad_h=pad_h,
            )

        online_targets = self.tracker.update(dets)

        frame_results = []
        for cls_id, tracks in online_targets.items():
            for t in tracks:
                if t.track_id < 0:
                    continue
                if t.curr_tlwh[2] * t.curr_tlwh[3] > self.min_area:
                    frame_results.append((cls_id, t.track_id, t.curr_tlwh.copy(), t.score))

        return frame_results


# ==========================================================================
# HẬU XỬ LÝ — 3 LỚP LÀM MƯỢT
# ==========================================================================
def _build_track_dict(all_results):
    """Gom theo (cls_id, track_id): track_dict[cls][tid][frame] = (tlwh, score)."""
    track_dict = defaultdict(lambda: defaultdict(dict))
    for frame_id, targets in all_results.items():
        for cls_id, trk_id, tlwh, score in targets:
            track_dict[cls_id][trk_id][frame_id] = (np.asarray(tlwh, np.float32), score)
    return track_dict


def filter_short_tracks(track_dict, min_len=5):
    """Bỏ track quá ngắn (đốm nháy 1-vài frame) — nguồn chính của hiện tượng nháy."""
    removed = 0
    for cls_id in list(track_dict.keys()):
        for trk_id in list(track_dict[cls_id].keys()):
            if len(track_dict[cls_id][trk_id]) < min_len:
                del track_dict[cls_id][trk_id]
                removed += 1
    print(f"  • Lọc track ngắn (<{min_len} frames): bỏ {removed} track.")
    return track_dict


def interpolate_tracks(track_dict, max_gap=15):
    """Nội suy tuyến tính lấp khoảng trống khi model miss object ngắn hạn."""
    filled = 0
    for cls_id, trks in track_dict.items():
        for trk_id, frames_data in trks.items():
            frame_ids = sorted(frames_data.keys())
            for i in range(len(frame_ids) - 1):
                f1, f2 = frame_ids[i], frame_ids[i + 1]
                gap = f2 - f1
                if 1 < gap <= max_gap:
                    box1, score1 = frames_data[f1]
                    box2, score2 = frames_data[f2]
                    for step in range(1, gap):
                        w = step / gap
                        f_interp = f1 + step
                        frames_data[f_interp] = (
                            box1 + (box2 - box1) * w,
                            score1 + (score2 - score1) * w,
                        )
                        filled += 1
    print(f"  • Nội suy (gap ≤ {max_gap}): điền {filled} box.")
    return track_dict


def smooth_tracks_moving_average(track_dict, window=5):
    """Làm mượt quỹ đạo bằng moving-average cửa sổ đối xứng (giảm rung box).

    Trung bình đối xứng -> không gây trễ (lag) như EMA. window lẻ là tốt nhất.
    """
    if window <= 1:
        return track_dict
    half = window // 2
    for cls_id, trks in track_dict.items():
        for trk_id, frames_data in trks.items():
            frame_ids = sorted(frames_data.keys())
            if len(frame_ids) < 3:
                continue
            boxes = np.stack([frames_data[f][0] for f in frame_ids])   # (N,4)
            scores = np.array([frames_data[f][1] for f in frame_ids])
            n = len(frame_ids)
            smoothed = boxes.copy()
            for i in range(n):
                a, b = max(0, i - half), min(n, i + half + 1)
                smoothed[i] = boxes[a:b].mean(axis=0)
            for i, f in enumerate(frame_ids):
                frames_data[f] = (smoothed[i].astype(np.float32), float(scores[i]))
    print(f"  • Làm mượt quỹ đạo (moving-average window={window}).")
    return track_dict


def track_dict_to_frames(track_dict):
    """Trả về results_smoothed[frame_id] = [(cls_id, trk_id, tlwh, score), ...]."""
    results = defaultdict(list)
    for cls_id, trks in track_dict.items():
        for trk_id, frames_data in trks.items():
            for frame_id, (tlwh, score) in frames_data.items():
                results[frame_id].append((cls_id, trk_id, tlwh, score))
    return results


def main():
    opt = opts()
    opt.parser.add_argument('--input_video', type=str, required=True)
    opt.parser.add_argument('--output_video', type=str, default='output_smoothed.mp4')
    opt.parser.add_argument('--max_interp_gap', type=int, default=15,
                            help='Khoảng trống tối đa để nối track (chống mất box ngắn).')
    opt.parser.add_argument('--min_track_len', type=int, default=5,
                            help='Bỏ track ngắn hơn ngưỡng này (chống nháy đốm).')
    opt.parser.add_argument('--smooth_window', type=int, default=5,
                            help='Cửa sổ moving-average làm mượt quỹ đạo (lẻ, 1 = tắt).')
    opt.parser.add_argument('--letterbox', action='store_true',
                            help='Dùng letterbox thay vì plain-resize (chính xác hơn).')
    parsed_opt = opt.init()

    if not os.path.exists(parsed_opt.input_video):
        raise FileNotFoundError(f"Cannot find input video: {parsed_opt.input_video}")

    cap = cv2.VideoCapture(parsed_opt.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = FastVideoTracker(parsed_opt, video_fps=fps)

    # -----------------------------------------------------
    # PHASE 1: INFERENCE + TRACKING (chạy AI + full tracker)
    # -----------------------------------------------------
    print("\n--- PHASE 1: AI Inference + Tracking ---")
    raw_results = {}
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Tracking")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        raw_results[frame_id] = tracker.get_tracks_for_frame(frame)
        pbar.update(1)
    pbar.close()
    cap.release()

    # -----------------------------------------------------
    # PHASE 2: SMOOTHING (lọc ngắn -> nội suy -> làm mượt)
    # -----------------------------------------------------
    print("\n--- PHASE 2: Smoothing Tracks ---")
    track_dict = _build_track_dict(raw_results)
    track_dict = filter_short_tracks(track_dict, min_len=parsed_opt.min_track_len)
    track_dict = interpolate_tracks(track_dict, max_gap=parsed_opt.max_interp_gap)
    track_dict = smooth_tracks_moving_average(track_dict, window=parsed_opt.smooth_window)
    smoothed_results = track_dict_to_frames(track_dict)

    # -----------------------------------------------------
    # PHASE 3: RENDER VIDEO
    # -----------------------------------------------------
    print("\n--- PHASE 3: Rendering Video ---")
    cap = cv2.VideoCapture(parsed_opt.input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(parsed_opt.output_video, fourcc, fps, (width, height))

    avg_fps = 1.0 / max(1e-5, tracker.timer.average_time)
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Rendering")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        targets = smoothed_results.get(frame_id, [])
        tlwhs, tids, scores = defaultdict(list), defaultdict(list), defaultdict(list)
        for (cls_id, trk_id, tlwh, score) in targets:
            tlwhs[cls_id].append(tlwh)
            tids[cls_id].append(trk_id)
            scores[cls_id].append(score)

        annotated_frame = vis.plot_tracks(
            image=frame,
            tlwhs_dict=tlwhs,
            obj_ids_dict=tids,
            num_classes=tracker.num_cls,
            scores=scores,
            frame_id=frame_id,
            fps=avg_fps,
            cls_id2name=CLS_NAMES_0_IDX,
        )
        out.write(annotated_frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()

    print(f"\n✅ Hoàn thành! Video đã lưu tại: {parsed_opt.output_video}")
    print(f"🚀 Tốc độ AI inference: {avg_fps:.2f} FPS")


if __name__ == '__main__':
    main()