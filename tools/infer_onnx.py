"""
infer_onnx.py — Run ECDetJDE ONNX model on a video / image folder.

Usage:
    python infer_onnx.py \
        --model ecdet_jde.onnx \
        --source path/to/images_or_video \
        --conf_thres 0.4 --nms_thres 0.45 \
        --num_classes 10 --reid_dim 128 \
        --save_dir outputs/onnx_result

No PyTorch required at inference time — only onnxruntime + cv2 + numpy.
"""

from __future__ import annotations
import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# ImageNet normalisation (must match training)
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, net_h: int, net_w: int,
              color=(127.5, 127.5, 127.5)) -> tuple[np.ndarray, float, int, int]:
    """Resize + pad to (net_h, net_w). Returns (padded_bgr, ratio, pad_left, pad_top)."""
    h, w = img.shape[:2]
    ratio = min(net_h / h, net_w / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    dw, dh = (net_w - new_w) * 0.5, (net_h - new_h) * 0.5
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, left, top


def preprocess(img_bgr: np.ndarray, net_h: int, net_w: int) -> tuple[np.ndarray, tuple, tuple]:
    """BGR image → (1,3,H,W) float32 blob + letterbox params for coord recovery."""
    orig_h, orig_w = img_bgr.shape[:2]
    lb, ratio, pad_left, pad_top = letterbox(img_bgr, net_h, net_w)
    rgb = lb[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    blob = rgb.transpose(2, 0, 1)[None]                    # (1, 3, H, W)
    return np.ascontiguousarray(blob), (orig_h, orig_w), (ratio, pad_left, pad_top)


# ---------------------------------------------------------------------------
# Postprocessing  (numpy, no PyTorch)
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.clip(-88, 88)))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Simple greedy NMS. boxes: (N,4) xyxy. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int32)


def postprocess(
    pred_logits: np.ndarray,            # (1, N, num_classes)
    pred_boxes:  np.ndarray,            # (1, N, 4) cxcywh norm in letterbox space
    pred_reid:   np.ndarray,            # (1, N, reid_dim)
    orig_hw: tuple[int, int],
    lb_params: tuple[float, int, int],  # (ratio, pad_left, pad_top)
    conf_thres: float,
    nms_thres:  float,
    num_classes: int,
) -> dict[int, np.ndarray]:
    """
    Returns {cls_id: (M, 6+reid_dim)}  columns = [x1,y1,x2,y2,score,cls_id, ...reid]
    All coords in original image pixel space.
    """
    orig_h, orig_w = orig_hw
    ratio, pad_left, pad_top = lb_params

    logits = pred_logits[0]     # (N, num_classes)
    boxes  = pred_boxes[0]      # (N, 4) cxcywh norm
    reid   = pred_reid[0]       # (N, D)

    scores_all = sigmoid(logits)    # (N, num_classes)

    # cxcywh (letterbox-norm) → xyxy (original image px)
    net_h, net_w = round(orig_h * ratio) + 2 * pad_top, round(orig_w * ratio) + 2 * pad_left
    # Actually net_h and net_w are fixed at export time — compute from params
    cx = boxes[:, 0]; cy = boxes[:, 1]; bw = boxes[:, 2]; bh = boxes[:, 3]
    # We know the network input was net_h × net_w; reconstruct from letterbox params
    # The postprocessor stores net sizes implicitly — we pass them via lb_params ratio
    # Inverse: pixel in letterbox → remove padding → scale to original
    # boxes are normalised in [0,1] relative to net_h, net_w
    # We don't have net_h/net_w here directly but can recover:
    #   new_w = round(orig_w * ratio), new_h = round(orig_h * ratio)
    new_w = round(orig_w * ratio)
    new_h = round(orig_h * ratio)
    net_w_actual = new_w + pad_left + (pad_left if (new_w + 2 * pad_left) % 2 == 0 else pad_left + 1)
    # Simpler: boxes are normalised to [0,1] in net space; derive from ratio/pad
    # x_px_in_net = cx * net_w_actual, then remove pad and scale
    # We'll pass net_hw to this function instead — see caller below

    results: dict[int, np.ndarray] = {}
    return results      # placeholder — see _postprocess_with_nethw below


def _postprocess_with_nethw(
    pred_logits, pred_boxes, pred_reid,
    orig_hw, net_hw, conf_thres, nms_thres, num_classes,
):
    """Full postprocess knowing net_hw explicitly."""
    orig_h, orig_w = orig_hw
    net_h, net_w   = net_hw

    ratio = min(net_h / orig_h, net_w / orig_w)
    new_w = round(orig_w * ratio)
    new_h = round(orig_h * ratio)
    dw = (net_w - new_w) * 0.5
    dh = (net_h - new_h) * 0.5

    logits = pred_logits[0]
    boxes  = pred_boxes[0]
    reid   = pred_reid[0]

    scores_all = sigmoid(logits)

    # cxcywh norm → xyxy in original image space
    cx_px = boxes[:, 0] * net_w
    cy_px = boxes[:, 1] * net_h
    bw_px = boxes[:, 2] * net_w
    bh_px = boxes[:, 3] * net_h
    cx_o = (cx_px - dw) / ratio
    cy_o = (cy_px - dh) / ratio
    bw_o = bw_px / ratio
    bh_o = bh_px / ratio
    x1 = np.clip(cx_o - bw_o * 0.5, 0, orig_w)
    y1 = np.clip(cy_o - bh_o * 0.5, 0, orig_h)
    x2 = np.clip(cx_o + bw_o * 0.5, 0, orig_w)
    y2 = np.clip(cy_o + bh_o * 0.5, 0, orig_h)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)   # (N, 4)

    results: dict[int, np.ndarray] = {}
    for cls_id in range(num_classes):
        cls_scores = scores_all[:, cls_id]
        keep = cls_scores >= conf_thres
        if keep.sum() == 0:
            results[cls_id] = np.zeros((0, 6), dtype=np.float32)
            continue
        cb = boxes_xyxy[keep]
        cs = cls_scores[keep]
        cr = reid[keep]
        nms_idx = nms(cb, cs, nms_thres)
        cb, cs, cr = cb[nms_idx], cs[nms_idx], cr[nms_idx]
        cls_col = np.full((cs.shape[0], 1), cls_id, dtype=np.float32)
        dets = np.concatenate([cb, cs[:, None], cls_col], axis=-1)   # (M, 6)
        results[cls_id] = dets
    return results


# ---------------------------------------------------------------------------
# Visualisation (minimal)
# ---------------------------------------------------------------------------
_PALETTE = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31), (255, 178, 29),
    ( 70, 179, 174), (  0, 194, 251), (  0, 212,  82), (  0, 128, 128),
    (148,  24, 255), (113, 148, 184),
]

def draw_dets(img: np.ndarray, results: dict[int, np.ndarray]) -> np.ndarray:
    out = img.copy()
    for cls_id, dets in results.items():
        color = _PALETTE[cls_id % len(_PALETTE)]
        for det in dets:
            x1, y1, x2, y2, score = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f'{cls_id}:{score:.2f}', (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


# ---------------------------------------------------------------------------
# Inference session
# ---------------------------------------------------------------------------
class ONNXInferencer:
    def __init__(self, model_path: str, providers=None, net_h: int = 608, net_w: int = 1088,
                 conf_thres: float = 0.4, nms_thres: float = 0.45,
                 num_classes: int = 10):
        providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session   = ort.InferenceSession(model_path, providers=providers)
        self.net_h     = net_h
        self.net_w     = net_w
        self.conf_thres = conf_thres
        self.nms_thres  = nms_thres
        self.num_classes = num_classes
        self.in_name   = self.session.get_inputs()[0].name
        print(f'[ORT] Providers: {self.session.get_providers()}')
        print(f'[ORT] Input: {self.in_name}  {self.session.get_inputs()[0].shape}')

    def infer(self, img_bgr: np.ndarray):
        """img_bgr: HxWx3 uint8. Returns dets dict + reid dict."""
        blob, orig_hw, _ = preprocess(img_bgr, self.net_h, self.net_w)
        pred_logits, pred_boxes, pred_reid = self.session.run(None, {self.in_name: blob})

        dets = _postprocess_with_nethw(
            pred_logits, pred_boxes, pred_reid,
            orig_hw, (self.net_h, self.net_w),
            self.conf_thres, self.nms_thres, self.num_classes,
        )
        return dets


# ---------------------------------------------------------------------------
# Source  (images / video file / webcam)
# ---------------------------------------------------------------------------
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _is_webcam(source: str) -> bool:
    return source.isdigit()


class Source:
    """Unified iterator over image-folder, video file, or webcam."""

    def __init__(self, source: str):
        self.source = source
        self.cap:  cv2.VideoCapture | None = None
        self.files: list[str] = []
        self.is_cam   = _is_webcam(source)
        self.is_video = False
        self.fps      = 30.0
        self.width    = 0
        self.height   = 0

        if self.is_cam:
            self.cap = cv2.VideoCapture(int(source))
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open webcam index {source}')
            self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'[Source] Webcam {source}  {self.width}×{self.height} @ {self.fps:.0f} fps')

        elif os.path.isdir(source):
            self.files = sorted(
                f for f in glob.glob(os.path.join(source, '*.*'))
                if os.path.splitext(f)[1].lower() in _IMG_EXTS
            )
            if not self.files:
                raise RuntimeError(f'No images found in {source}')
            sample = cv2.imread(self.files[0])
            self.height, self.width = sample.shape[:2]
            print(f'[Source] Image folder  {len(self.files)} frames  {self.width}×{self.height}')

        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open video: {source}')
            self.is_video = True
            self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'[Source] Video  {total} frames  {self.width}×{self.height} @ {self.fps:.1f} fps')

    def __iter__(self):
        if self.files:
            for i, fp in enumerate(self.files, 1):
                img = cv2.imread(fp)
                if img is not None:
                    yield i, img
        else:
            fid = 0
            while self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                fid += 1
                yield fid, frame

    def release(self):
        if self.cap:
            self.cap.release()


class VideoWriter:
    """Save visualised frames to a video file (optional)."""

    # Codec candidates tried in order; first one that works is used
    _CODECS = [
        ('avc1',  '.mp4'),   # H.264 — preferred on macOS / Linux with ffmpeg
        ('mp4v',  '.mp4'),   # MPEG-4
        ('XVID',  '.avi'),   # XVID — reliable fallback
        ('MJPG',  '.avi'),   # Motion JPEG — always available
    ]

    def __init__(self, path: str, fps: float, width: int, height: int):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        base = os.path.splitext(path)[0]
        self.writer: cv2.VideoWriter | None = None
        for codec, ext in self._CODECS:
            out_path = base + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if w.isOpened():
                self.writer = w
                self.path   = out_path
                print(f'[VideoWriter] codec={codec}  → {out_path}  {width}×{height} @ {fps:.1f} fps')
                break
            w.release()
        if self.writer is None:
            raise RuntimeError('No working video codec found (tried avc1, mp4v, XVID, MJPG)')

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    inferencer = ONNXInferencer(
        model_path  = args.model,
        net_h       = args.img_h,
        net_w       = args.img_w,
        conf_thres  = args.conf_thres,
        nms_thres   = args.nms_thres,
        num_classes = args.num_classes,
    )

    src = Source(args.source)
    is_cam = src.is_cam

    # Video writer — always for video/webcam, optional for images
    writer: VideoWriter | None = None
    if src.is_video or is_cam or args.save_video:
        out_path = os.path.join(args.save_dir, 'output.mp4')
        writer = VideoWriter(out_path, src.fps, src.width, src.height)

    show = args.show or is_cam   # always show live for webcam
    if show:
        cv2.namedWindow('ECDetJDE', cv2.WINDOW_NORMAL)

    total_dets = 0
    times: list[float] = []

    try:
        for fid, img in src:
            t0 = time.perf_counter()
            dets = inferencer.infer(img)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            n = sum(len(v) for v in dets.values())
            total_dets += n

            fps_cur = 1.0 / (sum(times[-30:]) / min(len(times), 30) + 1e-9)

            vis = draw_dets(img, dets)
            # Overlay FPS + frame counter
            cv2.putText(vis, f'FPS {fps_cur:.1f}  Frame {fid}  Dets {n}',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(vis)
            elif not (src.is_video or is_cam):
                # Image folder → save individual frames
                cv2.imwrite(os.path.join(args.save_dir, f'{fid:05d}.jpg'), vis)

            if show:
                cv2.imshow('ECDetJDE', vis)
                key = cv2.waitKey(1 if (is_cam or src.is_video) else 0) & 0xFF
                if key == ord('q') or key == 27:    # q / ESC to quit
                    print('\n[Info] Interrupted by user.')
                    break

            if fid % 30 == 0 and not is_cam:
                print(f'Frame {fid:5d} | dets {n:3d} | {fps_cur:.1f} fps')

    finally:
        src.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    avg_fps = len(times) / (sum(times) + 1e-9)
    print(f'\nDone. Frames: {len(times)} | Avg FPS: {avg_fps:.1f} | Total dets: {total_dets}')
    if writer:
        print(f'Video saved → {writer.path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='ECDetJDE ONNX inference — images / video / webcam',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model',       required=True,
                   help='path to .onnx model')
    p.add_argument('--source',      required=True,
                   help='image folder | video file | webcam index (e.g. 0)')
    p.add_argument('--save_dir',    default='outputs/onnx',
                   help='output directory')
    p.add_argument('--img_h',       type=int,   default=608)
    p.add_argument('--img_w',       type=int,   default=1088)
    p.add_argument('--conf_thres',  type=float, default=0.25)
    p.add_argument('--nms_thres',   type=float, default=0.45)
    p.add_argument('--num_classes', type=int,   default=10)
    p.add_argument('--show',        action='store_true',
                   help='display live window (auto-on for webcam)')
    p.add_argument('--save_video',  action='store_true',
                   help='force video output even for image folders')
    main(p.parse_args())
