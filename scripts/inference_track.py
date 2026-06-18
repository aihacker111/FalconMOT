"""
inference_track.py — Smooth Video Tracking Inference for FalconJDE.
Có nội suy (Interpolation) để chống nháy box và map đúng tên Class.
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
from falconmot.tracker import FalconTracker, Track
from falconmot.tracking_utils import visualization as vis
from falconmot.tracking_utils.timer import Timer
from falconmot.opts import opts

# ==========================================
# CẤU HÌNH CLASS MAP MÀ BẠN CUNG CẤP
# ==========================================
MERGED_CLS_MAP = {
    1: 'pedestrian', 2: 'bicycle', 3: 'car', 4: 'truck',
    5: 'tricycle', 6: 'bus', 7: 'motor'
}

# Chuyển về 0-indexed vì model output (cls_id) bắt đầu từ 0
CLS_NAMES_0_IDX = {k - 1: v for k, v in MERGED_CLS_MAP.items()}


class FastVideoTracker:
    def __init__(self, opt, video_fps: int):
        self.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 and torch.cuda.is_available() else 'cpu'
        self.num_cls = opt.num_classes
        self.min_area = opt.min_box_area
        self.net_w, self.net_h = opt.img_size
        
        print(f'Loading model onto {self.device}...')
        self.model = create_model(opt.arch, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(self.device).eval()

        self.postprocessor = FalconJDEPostProcessor(
            num_classes=opt.num_classes,
            num_top_queries=getattr(opt, 'K', 300),
            conf_thres=opt.conf_thres,
            use_focal_loss=True,
        )

        self.tracker = FalconTracker(opt, frame_rate=video_fps)
        self.timer = Timer()
        self._orig_sizes = None

    def preprocess(self, img_bgr):
        img_resized = cv2.resize(img_bgr, (self.net_w, self.net_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

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
        """Chạy AI và trả về list tọa độ, không vẽ ngay để tích luỹ"""
        orig_h, orig_w = frame_bgr.shape[:2]
        if self._orig_sizes is None:
            self._orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

        blob = self.preprocess(frame_bgr)

        self.timer.tic()
        with torch.no_grad():
            output = self.model(blob)
            res = self.postprocessor(output, self._orig_sizes)[0]
            dets = self._decode_detections(res)
        self.timer.toc()

        self.tracker.set_image(frame_bgr)
        online_targets = self.tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

        frame_results = []
        for cls_id, tracks in online_targets.items():
            for t in tracks:
                if t.curr_tlwh[2] * t.curr_tlwh[3] > self.min_area:
                    # Lưu lại: cls_id, track_id, tọa độ tlwh, độ tự tin (score)
                    frame_results.append((cls_id, t.track_id, t.curr_tlwh.copy(), t.score))
                    
        return frame_results


def interpolate_tracks(all_results, max_gap=15):
    """
    Thuật toán nội suy: lấp đầy khoảng trống nếu model bị miss object trong thời gian ngắn (chống nháy)
    max_gap: Số frame tối đa bị mất tín hiệu được phép nối lại (15 frames = 0.5s ở 30FPS).
    """
    # Gom nhóm theo track_id: tracks[cls_id][track_id][frame_id] = (tlwh, score)
    track_dict = defaultdict(lambda: defaultdict(dict))
    
    for frame_id, targets in all_results.items():
        for cls_id, trk_id, tlwh, score in targets:
            track_dict[cls_id][trk_id][frame_id] = (tlwh, score)

    # Thực hiện nội suy tuyến tính (Linear Interpolation)
    for cls_id, trks in track_dict.items():
        for trk_id, frames_data in trks.items():
            frame_ids = sorted(frames_data.keys())
            for i in range(len(frame_ids) - 1):
                f1, f2 = frame_ids[i], frame_ids[i+1]
                gap = f2 - f1
                
                # Nếu bị miss từ 2 đến max_gap frames -> Tiến hành điền vào chỗ trống
                if 1 < gap <= max_gap:
                    box1, score1 = frames_data[f1]
                    box2, score2 = frames_data[f2]
                    
                    for step in range(1, gap):
                        f_interp = f1 + step
                        weight = step / gap
                        box_interp = box1 + (box2 - box1) * weight
                        score_interp = score1 + (score2 - score1) * weight
                        # Cập nhật vào dữ liệu nội suy
                        frames_data[f_interp] = (box_interp, score_interp)

    # Trả ngược lại cấu trúc theo frame_id: results_smoothed[frame_id] = [...]
    results_smoothed = defaultdict(list)
    for cls_id, trks in track_dict.items():
        for trk_id, frames_data in trks.items():
            for frame_id, (tlwh, score) in frames_data.items():
                results_smoothed[frame_id].append((cls_id, trk_id, tlwh, score))
                
    return results_smoothed


def main():
    opt = opts()
    opt.parser.add_argument('--input_video', type=str, required=True)
    opt.parser.add_argument('--output_video', type=str, default='output_smoothed.mp4')
    opt.parser.add_argument('--max_interp_gap', type=int, default=15, help='Khoảng trống max để nối track (chống nháy)')
    parsed_opt = opt.init()

    if not os.path.exists(parsed_opt.input_video):
        raise FileNotFoundError(f"Cannot find input video: {parsed_opt.input_video}")

    cap = cv2.VideoCapture(parsed_opt.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = FastVideoTracker(parsed_opt, video_fps=fps)
    
    # -----------------------------------------------------
    # PHASE 1: INFERENCE (Chạy AI lấy dữ liệu, KHÔNG vẽ)
    # -----------------------------------------------------
    print(f"\n--- PHASE 1: AI Inference ---")
    raw_results = {}
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Detecting")

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
    # PHASE 2: INTERPOLATION (Làm mượt, chống nháy)
    # -----------------------------------------------------
    print("\n--- PHASE 2: Smoothing Tracks ---")
    smoothed_results = interpolate_tracks(raw_results, max_gap=parsed_opt.max_interp_gap)

    # -----------------------------------------------------
    # PHASE 3: RENDER VIDEO (Vẽ hình và lưu ra file)
    # -----------------------------------------------------
    print("\n--- PHASE 3: Rendering Video ---")
    cap = cv2.VideoCapture(parsed_opt.input_video) # Mở lại video từ đầu
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(parsed_opt.output_video, fourcc, fps, (width, height))
    
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Rendering")

    avg_fps = 1.0 / max(1e-5, tracker.timer.average_time)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        
        # Lấy dữ liệu đã được làm mượt
        targets = smoothed_results.get(frame_id, [])
        
        tlwhs, tids, scores = defaultdict(list), defaultdict(list), defaultdict(list)
        for (cls_id, trk_id, tlwh, score) in targets:
            tlwhs[cls_id].append(tlwh)
            tids[cls_id].append(trk_id)
            scores[cls_id].append(score)

        # Vẽ frame
        annotated_frame = vis.plot_tracks(
            image=frame,
            tlwhs_dict=tlwhs,
            obj_ids_dict=tids,
            num_classes=tracker.num_cls,
            scores=scores,
            frame_id=frame_id,
            fps=avg_fps,
            cls_id2name=CLS_NAMES_0_IDX  # <-- DÙNG CLASS MAP Ở ĐÂY
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