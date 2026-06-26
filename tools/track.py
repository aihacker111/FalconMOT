"""Unified tracking inference for FalconMOT.

Runs detection + multi-object tracking and produces an annotated output. Three
input modes are auto-detected from ``--source`` (or forced with ``--mode``):

  * video      : a video file (``.mp4``, ``.avi``, ...). Offline 3-phase pipeline
                 (inference+track -> trajectory smoothing -> render) that writes a
                 smooth annotated video.
  * images     : a folder of frames (an image sequence). Same offline pipeline as
                 ``video``; writes an annotated video and, optionally, frames.
  * realtime   : a webcam index (e.g. ``0``) or a stream URL. Online per-frame
                 tracking shown live; the annotated video can be saved too. Global
                 smoothing is unavailable online, so only the tracker's own
                 temporal smoothing applies.

The appearance-motion (QAM) dense-embedding path and global motion compensation
(GMC) are enabled exactly as in training/eval, so tracking quality matches the
reported benchmark numbers.

Examples:
    # Video file -> smooth annotated video
    python tools/track.py --source demo.mp4 --output_video out/demo_tracked.mp4 \\
        --arch falcon_jde --load_model exp/mot/run/model_best.pth \\
        --input-wh 1088 640 --use_appearance_motion

    # Image sequence (folder of frames)
    python tools/track.py --source /data/seq/images --fps 30 \\
        --load_model model_best.pth --input-wh 1088 640

    # Real-time webcam (camera 0), live window + saved video
    python tools/track.py --source 0 --mode realtime --show \\
        --output_video out/webcam.mp4 --load_model model_best.pth
"""
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

import _paths  # noqa: F401  (sys.path bootstrap)
from falconmot import create_model, load_model
from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracker import FalconTracker, Track
from falconmot.tracker.utils import visualization as vis
from falconmot.tracker.utils.timer import Timer
from falconmot.cfg import opts

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Class map: 7-class model output (0-indexed) -> human-readable names.
MERGED_CLS_MAP = {
    1: "pedestrian", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "bus", 7: "motor",
}
CLS_NAMES_0_IDX = {k - 1: v for k, v in MERGED_CLS_MAP.items()}

# MOT-Challenge result line: frame,id,x,y,w,h,score,cls,-1,-1
_MOT_FMT = "{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},{cls},-1,-1\n"


class FalconVideoTracker:
    """Per-frame detector + tracker wrapper with optional dense QAM and GMC."""

    def __init__(self, opt, video_fps: int):
        self.device = (
            f"cuda:{opt.gpus[0]}"
            if opt.gpus[0] >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        self.num_cls = opt.num_classes
        self.min_area = opt.min_box_area
        self.net_w, self.net_h = opt.img_size  # img_size = (W, H)
        self.use_letterbox = getattr(opt, "letterbox", False)
        self.use_am = getattr(opt, "use_appearance_motion", False)

        print(f"Loading model onto {self.device}...")
        self.model = create_model(opt.arch, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(self.device).eval()

        # QAM: ask the model to return the dense appearance map.
        if self.use_am:
            inner = getattr(self.model, "module", self.model)
            inner.return_reid_dense = True
            print("[QAM] Query Appearance-Motion: ON (dense appearance enabled)")
        else:
            print("[QAM] Query Appearance-Motion: OFF (IoU + sparse ReID). "
                  "Pass --use_appearance_motion for smoother tracks.")

        self.postprocessor = FalconJDEPostProcessor(
            num_classes=opt.num_classes,
            num_top_queries=getattr(opt, "K", 300),
            conf_thres=opt.conf_thres,
            use_focal_loss=True,
        )
        # Letterbox -> the postprocessor needs net_hw to decode boxes correctly.
        # Plain resize -> do NOT set_net_hw (it uses the scale * orig branch).
        if self.use_letterbox:
            self.postprocessor.set_net_hw(self.net_h, self.net_w)

        self.tracker = FalconTracker(opt, frame_rate=video_fps)
        self.timer = Timer()
        self._orig_sizes = None

    def preprocess(self, img_bgr):
        """Return (tensor, transform) so the dense map aligns with box decoding.

        transform = (ratio_x, ratio_y, pad_w, pad_h)
        """
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
            img_proc = cv2.resize(img_bgr, (self.net_w, self.net_h), interpolation=cv2.INTER_AREA)
            transform = (self.net_w / orig_w, self.net_h / orig_h, 0.0, 0.0)

        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device), transform

    def _decode_detections(self, res: dict) -> defaultdict:
        """Convert postprocessor output into per-class lists of Track objects."""
        dets = defaultdict(list)
        if len(res["scores"]) == 0:
            return dets

        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        reid = res["reid"].cpu().numpy() if "reid" in res else None

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
        """Run detector + tracker (full QAM) on one frame; return a track list."""
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

        # GMC needs the original frame.
        self.tracker.set_image(frame_bgr)

        # QAM: feed the dense appearance map for the current frame. Coordinates
        # must match how the postprocessor decoded the boxes (transform above).
        if isinstance(output, dict) and output.get("reid_dense") is not None:
            rx, ry, pad_w, pad_h = transform
            self.tracker.set_dense(
                output["reid_dense"],  # [C,H,W] (batch already removed by the model)
                stride=output["reid_dense_stride"],
                ratio_x=rx, ratio_y=ry, pad_w=pad_w, pad_h=pad_h,
            )

        online_targets = self.tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

        frame_results = []
        for cls_id, tracks in online_targets.items():
            for t in tracks:
                if t.track_id < 0:
                    continue
                if t.curr_tlwh[2] * t.curr_tlwh[3] > self.min_area:
                    frame_results.append((cls_id, t.track_id, t.curr_tlwh.copy(), t.score))
        return frame_results


# ===========================================================================
#  Offline trajectory smoothing (three passes)
# ===========================================================================
def _build_track_dict(all_results):
    """Group by (cls_id, track_id): track_dict[cls][tid][frame] = (tlwh, score)."""
    track_dict = defaultdict(lambda: defaultdict(dict))
    for frame_id, targets in all_results.items():
        for cls_id, trk_id, tlwh, score in targets:
            track_dict[cls_id][trk_id][frame_id] = (np.asarray(tlwh, np.float32), score)
    return track_dict


def filter_short_tracks(track_dict, min_len=5):
    """Drop very short tracks (1-few frame blips) — the main source of flicker."""
    removed = 0
    for cls_id in list(track_dict.keys()):
        for trk_id in list(track_dict[cls_id].keys()):
            if len(track_dict[cls_id][trk_id]) < min_len:
                del track_dict[cls_id][trk_id]
                removed += 1
    print(f"  - Filter short tracks (<{min_len} frames): removed {removed}.")
    return track_dict


def interpolate_tracks(track_dict, max_gap=15):
    """Linearly interpolate to fill short detection gaps (missed objects)."""
    filled = 0
    for _cls_id, trks in track_dict.items():
        for _trk_id, frames_data in trks.items():
            frame_ids = sorted(frames_data.keys())
            for i in range(len(frame_ids) - 1):
                f1, f2 = frame_ids[i], frame_ids[i + 1]
                gap = f2 - f1
                if 1 < gap <= max_gap:
                    box1, score1 = frames_data[f1]
                    box2, score2 = frames_data[f2]
                    for step in range(1, gap):
                        w = step / gap
                        frames_data[f1 + step] = (
                            box1 + (box2 - box1) * w,
                            score1 + (score2 - score1) * w,
                        )
                        filled += 1
    print(f"  - Interpolate (gap <= {max_gap}): filled {filled} boxes.")
    return track_dict


def smooth_tracks_moving_average(track_dict, window=5):
    """Smooth trajectories with a symmetric moving average (reduces box jitter).

    A symmetric average introduces no lag (unlike an EMA). An odd window works
    best.
    """
    if window <= 1:
        return track_dict
    half = window // 2
    for _cls_id, trks in track_dict.items():
        for _trk_id, frames_data in trks.items():
            frame_ids = sorted(frames_data.keys())
            if len(frame_ids) < 3:
                continue
            boxes = np.stack([frames_data[f][0] for f in frame_ids])  # (N,4)
            scores = np.array([frames_data[f][1] for f in frame_ids])
            n = len(frame_ids)
            smoothed = boxes.copy()
            for i in range(n):
                a, b = max(0, i - half), min(n, i + half + 1)
                smoothed[i] = boxes[a:b].mean(axis=0)
            for i, f in enumerate(frame_ids):
                frames_data[f] = (smoothed[i].astype(np.float32), float(scores[i]))
    print(f"  - Smooth trajectories (moving-average window={window}).")
    return track_dict


def track_dict_to_frames(track_dict):
    """Return results[frame_id] = [(cls_id, trk_id, tlwh, score), ...]."""
    results = defaultdict(list)
    for cls_id, trks in track_dict.items():
        for trk_id, frames_data in trks.items():
            for frame_id, (tlwh, score) in frames_data.items():
                results[frame_id].append((cls_id, trk_id, tlwh, score))
    return results


# ===========================================================================
#  Frame sources and rendering helpers
# ===========================================================================
def _resolve_mode(source: str, forced_mode: str) -> str:
    """Decide the input mode from --mode or by inspecting --source."""
    if forced_mode != "auto":
        return forced_mode
    if source.isdigit():
        return "realtime"
    if osp.isdir(source):
        return "images"
    if source.lower().startswith(("rtsp://", "http://", "https://")):
        return "realtime"
    return "video"


def _list_sequence_frames(folder: str):
    return sorted(
        osp.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(_IMG_EXTS)
    )


def _render_targets(frame, targets, tracker):
    """Annotate one frame with the given (cls, id, tlwh, score) targets."""
    tlwhs, tids, scores = defaultdict(list), defaultdict(list), defaultdict(list)
    for cls_id, trk_id, tlwh, score in targets:
        tlwhs[cls_id].append(tlwh)
        tids[cls_id].append(trk_id)
        scores[cls_id].append(score)
    return vis.plot_tracks(
        image=frame,
        tlwhs_dict=tlwhs,
        obj_ids_dict=tids,
        num_classes=tracker.num_cls,
        scores=scores,
        frame_id=0,
        fps=0.0,
        cls_id2name=CLS_NAMES_0_IDX,
    )


def _write_mot_results(path, results):
    """Write all frame results in MOT-Challenge text format."""
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for frame_id in sorted(results.keys()):
            for cls_id, trk_id, tlwh, score in results[frame_id]:
                x, y, w, h = tlwh
                f.write(_MOT_FMT.format(
                    frame=frame_id, tid=trk_id, x=x, y=y, w=w, h=h,
                    score=score, cls=cls_id + 1,
                ))
    print(f"MOT results written to: {path}")


# ===========================================================================
#  Offline pipeline (video / image-sequence) and online pipeline (realtime)
# ===========================================================================
def run_offline(opt, frames_iter, num_frames, width, height, fps):
    """Three-phase offline tracking: inference+track -> smooth -> render."""
    tracker = FalconVideoTracker(opt, video_fps=fps)

    print("\n--- Phase 1: detection + tracking ---")
    raw_results = {}
    frame_cache = [] if opt.cache_frames else None
    frame_id = 0
    for frame in tqdm(frames_iter, total=num_frames, desc="Tracking"):
        frame_id += 1
        raw_results[frame_id] = tracker.get_tracks_for_frame(frame)
        if frame_cache is not None:
            frame_cache.append(frame)

    print("\n--- Phase 2: smoothing tracks ---")
    track_dict = _build_track_dict(raw_results)
    track_dict = filter_short_tracks(track_dict, min_len=opt.min_track_len)
    track_dict = interpolate_tracks(track_dict, max_gap=opt.max_interp_gap)
    track_dict = smooth_tracks_moving_average(track_dict, window=opt.smooth_window)
    smoothed = track_dict_to_frames(track_dict)

    if opt.save_mot:
        _write_mot_results(opt.save_mot, smoothed)

    print("\n--- Phase 3: rendering video ---")
    os.makedirs(osp.dirname(osp.abspath(opt.output_video)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(opt.output_video, fourcc, fps, (width, height))

    # Re-create the frame source for rendering if frames were not cached.
    render_iter = frame_cache if frame_cache is not None else _rebuild_source(opt)
    frame_id = 0
    for frame in tqdm(render_iter, total=num_frames, desc="Rendering"):
        frame_id += 1
        annotated = _render_targets(frame, smoothed.get(frame_id, []), tracker)
        out.write(annotated)
    out.release()
    if isinstance(render_iter, _VideoFrameSource):
        render_iter.release()

    avg_fps = 1.0 / max(1e-5, tracker.timer.average_time)
    print(f"\nDone. Annotated video saved to: {opt.output_video}")
    print(f"Detector speed: {avg_fps:.2f} FPS")


def run_realtime(opt, cap, width, height, fps):
    """Online per-frame tracking from a live source, shown and/or saved."""
    tracker = FalconVideoTracker(opt, video_fps=fps)

    out = None
    if opt.output_video:
        os.makedirs(osp.dirname(osp.abspath(opt.output_video)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(opt.output_video, fourcc, fps, (width, height))

    mot_results = {} if opt.save_mot else None
    print("\n--- Real-time tracking (press 'q' to stop) ---")
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            targets = tracker.get_tracks_for_frame(frame)
            if mot_results is not None:
                mot_results[frame_id] = targets

            annotated = _render_targets(frame, targets, tracker)
            live_fps = 1.0 / max(1e-5, tracker.timer.average_time)
            cv2.putText(annotated, f"{live_fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if out is not None:
                out.write(annotated)
            if opt.show:
                cv2.imshow("FalconMOT - realtime", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if out is not None:
            out.release()
        if opt.show:
            cv2.destroyAllWindows()

    if mot_results is not None:
        _write_mot_results(opt.save_mot, mot_results)
    print(f"\nStopped after {frame_id} frames. Average detector speed: "
          f"{1.0 / max(1e-5, tracker.timer.average_time):.2f} FPS")


# ---- frame-source helpers used by the offline renderer --------------------
class _VideoFrameSource:
    """Iterable over the frames of a video file (used for the render pass)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()


def _rebuild_source(opt):
    if opt._mode == "images":
        return (cv2.imread(p) for p in _list_sequence_frames(opt.source))
    return _VideoFrameSource(opt.source)


def main():
    opt_parser = opts()
    p = opt_parser.parser
    p.add_argument("--source", type=str, required=True,
                   help="video file, folder of frames, webcam index, or stream URL")
    p.add_argument("--mode", type=str, default="auto",
                   choices=["auto", "video", "images", "realtime"],
                   help="input mode (default: auto-detect from --source)")
    p.add_argument("--output_video", type=str, default="out/tracked.mp4",
                   help="path for the annotated output video ('' to disable in realtime)")
    p.add_argument("--save_mot", type=str, default="",
                   help="optional path to write MOT-Challenge format results")
    p.add_argument("--fps", type=int, default=30,
                   help="frame rate to assume for image sequences / cameras")
    p.add_argument("--show", action="store_true",
                   help="show a live preview window (realtime mode)")
    p.add_argument("--cache_frames", action="store_true",
                   help="keep frames in RAM to avoid re-reading the source for rendering")
    p.add_argument("--max_interp_gap", type=int, default=15,
                   help="max gap (frames) to bridge when interpolating tracks")
    p.add_argument("--min_track_len", type=int, default=5,
                   help="drop tracks shorter than this many frames")
    p.add_argument("--smooth_window", type=int, default=5,
                   help="moving-average window for trajectory smoothing (odd; 1 = off)")
    p.add_argument("--letterbox", action="store_true",
                   help="use letterbox instead of plain resize (more accurate)")
    opt = opt_parser.init()

    opt._mode = _resolve_mode(opt.source, opt.mode)
    print(f"Input mode: {opt._mode}")

    if opt._mode == "realtime":
        cap = cv2.VideoCapture(int(opt.source) if opt.source.isdigit() else opt.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {opt.source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or opt.fps
        if not opt.output_video:
            opt.output_video = ""
        run_realtime(opt, cap, width, height, fps)
        return

    if opt._mode == "images":
        frames = _list_sequence_frames(opt.source)
        if not frames:
            raise FileNotFoundError(f"No images found in: {opt.source}")
        sample = cv2.imread(frames[0])
        height, width = sample.shape[:2]
        fps = opt.fps
        frames_iter = (cv2.imread(p) for p in frames)
        run_offline(opt, frames_iter, len(frames), width, height, fps)
        return

    # video mode
    if not osp.exists(opt.source):
        raise FileNotFoundError(f"Cannot find input video: {opt.source}")
    cap = cv2.VideoCapture(opt.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or opt.fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_iter = (lambda c: (f for ok, f in iter(lambda: c.read(), (False, None)) if ok))(cap)
    run_offline(opt, frames_iter, total, width, height, fps)
    cap.release()


if __name__ == "__main__":
    main()
