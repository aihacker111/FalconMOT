"""
inference_track.py — Pure Video Inference for FalconJDE.

No evaluation, no filtering, no metrics. Just fast tracking inference.
Input: A video file (.mp4, .avi)
Output: A new video file with bounding boxes and tracking IDs.
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

        # Basic fallback names for visualization (Class 0, Class 1, etc.)
        self.cls_names = {i: f"Cls {i}" for i in range(self.num_cls)}

    def preprocess(self, img_bgr):
        """Fast image preprocessing: resize -> RGB -> normalize -> tensor."""
        img_resized = cv2.resize(img_bgr, (self.net_w, self.net_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def _decode_detections(self, res: dict) -> defaultdict:
        """Vectorized conversion of raw outputs to Track objects."""
        dets = defaultdict(list)
        if len(res['scores']) == 0:
            return dets

        boxes = res['boxes'].cpu().numpy()
        scores = res['scores'].cpu().numpy()
        labels = res['labels'].cpu().numpy()
        reid = res['reid'].cpu().numpy() if 'reid' in res else None

        # Fast valid-box filter
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        valid_idx = np.where((ws > 0) & (hs > 0))[0]

        for i in valid_idx:
            cls_id = int(labels[i])
            tlwh = np.array([boxes[i, 0], boxes[i, 1], ws[i], hs[i]], dtype=np.float32)
            emb = reid[i] if reid is not None else np.zeros(1, dtype=np.float32)
            dets[cls_id].append(Track(tlwh, float(scores[i]), emb, self.num_cls, cls_id))
            
        return dets

    def process_frame(self, frame_bgr, frame_id):
        """Run tracking on a single frame and return the annotated image."""
        orig_h, orig_w = frame_bgr.shape[:2]

        # Cache original sizes (constant for the whole video)
        if self._orig_sizes is None:
            self._orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

        # 1. Preprocess
        blob = self.preprocess(frame_bgr)

        # 2. Inference (No Gradient = Faster & Less VRAM)
        self.timer.tic()
        with torch.no_grad():
            output = self.model(blob)
            res = self.postprocessor(output, self._orig_sizes)[0]
            dets = self._decode_detections(res)
        self.timer.toc()

        # 3. Update Tracker
        self.tracker.set_image(frame_bgr)
        online_targets = self.tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

        # 4. Collect Valid Tracks
        tlwhs, tids, scores = defaultdict(list), defaultdict(list), defaultdict(list)
        for cls_id, tracks in online_targets.items():
            for t in tracks:
                if t.curr_tlwh[2] * t.curr_tlwh[3] > self.min_area:
                    tlwhs[cls_id].append(t.curr_tlwh)
                    tids[cls_id].append(t.track_id)
                    scores[cls_id].append(t.score)

        # 5. Draw visualization
        current_fps = 1.0 / max(1e-5, self.timer.average_time)
        annotated_frame = vis.plot_tracks(
            image=frame_bgr,
            tlwhs_dict=tlwhs,
            obj_ids_dict=tids,
            num_classes=self.num_cls,
            scores=scores,
            frame_id=frame_id,
            fps=current_fps,
            cls_id2name=self.cls_names
        )
        return annotated_frame


def main():
    opt = opts()
    # Add simple custom arguments for video
    opt.parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    opt.parser.add_argument('--output_video', type=str, default='output.mp4', help='Path to save output video')
    parsed_opt = opt.init()

    if not os.path.exists(parsed_opt.input_video):
        raise FileNotFoundError(f"Cannot find input video: {parsed_opt.input_video}")

    # 1. Open Video
    cap = cv2.VideoCapture(parsed_opt.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. Setup Tracker & Writer
    tracker = FastVideoTracker(parsed_opt, video_fps=fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(parsed_opt.output_video, fourcc, fps, (width, height))

    print(f"\n--- Starting Video Tracking ---")
    print(f"Input: {parsed_opt.input_video}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    
    # 3. Main Loop
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        annotated_frame = tracker.process_frame(frame, frame_id)
        out.write(annotated_frame)
        pbar.update(1)

    # 4. Cleanup
    pbar.close()
    cap.release()
    out.release()
    
    avg_fps = 1.0 / max(1e-5, tracker.timer.average_time)
    print(f"\nDone! Output saved to: {parsed_opt.output_video}")
    print(f"Average Inference Speed: {avg_fps:.2f} FPS")


if __name__ == '__main__':
    main()