"""Detection demo for FalconMOT (image / folder mode).

Runs the FalconJDE detector on a single image or a folder of images and writes
the visualized results. Tracking is not used here; for tracking on a video,
image sequence, or webcam, use ``tools/track.py``.

Example:
    python tools/detect.py \\
        --arch falcon_jde --load_model exp/mot/run/model_best.pth \\
        --input_path /data/demo_images --output_dir out/detect \\
        --input-wh 1088 640 --conf_thres 0.4
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch

import _paths  # noqa: F401  (sys.path bootstrap)
from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot import create_model, load_model
from falconmot.cfg.args import opts

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# VisDrone 7-class label set used by the released model (0-indexed).
VISDRONE_CLASSES = [
    "pedestrian",
    "bicycle",
    "car",
    "van",
    "truck",
    "bus",
    "motor",
]

# Fixed per-class colors for drawing (deterministic across runs).
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(VISDRONE_CLASSES), 3), dtype=np.uint8)

# ImageNet normalization (matches the training preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_plain(img, net_h, net_w):
    """Plain resize to (net_w, net_h) without letterbox padding."""
    return cv2.resize(img, (net_w, net_h), interpolation=cv2.INTER_AREA)


def to_tensor(img_bgr):
    """Preprocess a BGR image into a (1, 3, H, W) tensor.

    Converts BGR -> RGB, scales to [0, 1], then applies ImageNet mean/std
    normalization, and finally moves to channel-first layout with a batch dim.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _nms(boxes, scores, iou_thr=0.6):
    """Standard greedy NMS on (x1, y1, x2, y2) boxes."""
    if len(boxes) == 0:
        return np.empty((0,), int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, int)


@torch.no_grad()
def infer_image(model, post, img_bgr, net_h, net_w, device):
    """Run the detector on a full image and return (boxes, scores, labels)."""
    h0, w0 = img_bgr.shape[:2]
    blob = to_tensor(resize_plain(img_bgr, net_h, net_w)).to(device)
    out = model(blob)
    res = post(out, torch.tensor([[h0, w0]], device=device))[0]
    return (
        res["boxes"].cpu().numpy(),
        res["scores"].cpu().numpy(),
        res["labels"].cpu().numpy(),
    )


@torch.no_grad()
def infer_tiled(model, post, img_bgr, net_h, net_w, device, rows=2, cols=2, overlap=0.2):
    """SAHI-style sliced inference for tiny objects in large frames.

    The image is split into ``rows x cols`` overlapping tiles (plus the full
    frame); detections from all tiles are merged back to image coordinates and
    de-duplicated with per-class NMS.
    """
    h0, w0 = img_bgr.shape[:2]
    th, tw = int(h0 / rows), int(w0 / cols)
    oy, ox = int(th * overlap), int(tw * overlap)
    all_b, all_s, all_l = [], [], []

    crops = [(0, 0, w0, h0)]
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, c * tw - ox)
            y1 = max(0, r * th - oy)
            x2 = min(w0, (c + 1) * tw + ox)
            y2 = min(h0, (r + 1) * th + oy)
            crops.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in crops:
        b, s, l = infer_image(model, post, img_bgr[y1:y2, x1:x2], net_h, net_w, device)
        if len(b):
            b = b.copy()
            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1
            all_b.append(b)
            all_s.append(s)
            all_l.append(l)

    if not all_b:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), int)

    boxes = np.concatenate(all_b)
    scores = np.concatenate(all_s)
    labels = np.concatenate(all_l)

    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        keep = _nms(boxes[idx], scores[idx])
        keep_all.extend(idx[keep])
    keep_all = np.array(keep_all, int)
    return boxes[keep_all], scores[keep_all], labels[keep_all]


def visualize(img, boxes, scores, labels, conf_thres=0.3):
    """Draw boxes, class names and scores above the confidence threshold."""
    vis_img = img.copy()
    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, labels):
        if score < conf_thres:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        class_name = VISDRONE_CLASSES[cls_id] if cls_id < len(VISDRONE_CLASSES) else f"cls_{cls_id}"
        color = [int(c) for c in COLORS[cls_id % len(COLORS)]]

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_name} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(
            vis_img, label_text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return vis_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="path to an image or a folder of images")
    ap.add_argument("--output_dir", default="out/inference_results", help="directory to save visualized results")
    ap.add_argument("--conf_thres", type=float, default=0.4, help="confidence threshold for drawing boxes")
    ap.add_argument("--max_dets", type=int, default=300, help="maximum number of objects to keep")
    ap.add_argument("--tile", action="store_true", help="enable SAHI-style sliced inference for tiny objects")
    ap.add_argument("--tile_grid", type=int, nargs=2, default=[2, 2], help="tile grid as ROWS COLS")
    a, unknown = ap.parse_known_args()

    opt = opts().init(unknown)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_w, net_h = opt.input_wh[0], opt.input_wh[1]

    print(f"Initializing model '{opt.arch}'...")
    model = create_model(opt.arch, opt)
    assert opt.load_model, "The --load_model argument is required"
    model = load_model(model, opt.load_model)
    model = model.to(device).eval()

    # Plain resize -> do not call set_net_hw (the postprocessor rescales by norm * orig).
    post = FalconJDEPostProcessor(num_classes=opt.num_classes, num_top_queries=a.max_dets)

    if os.path.isdir(a.input_path):
        img_files = sorted(
            p for p in glob.glob(os.path.join(a.input_path, "*"))
            if p.lower().endswith(_IMG_EXTS)
        )
    else:
        img_files = [a.input_path] if a.input_path.lower().endswith(_IMG_EXTS) else []

    print(f"Found {len(img_files)} image(s). Net size: {net_w}x{net_h} | tiled: {a.tile}")
    if not img_files:
        print("No valid images found.")
        return

    os.makedirs(a.output_dir, exist_ok=True)

    for n, path in enumerate(img_files):
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read image: {path}")
            continue

        t0 = time.time()
        if a.tile:
            boxes, scores, labels = infer_tiled(
                model, post, img, net_h, net_w, device,
                rows=a.tile_grid[0], cols=a.tile_grid[1],
            )
        else:
            boxes, scores, labels = infer_image(model, post, img, net_h, net_w, device)

        if len(scores) > a.max_dets:
            top = scores.argsort()[::-1][: a.max_dets]
            boxes, scores, labels = boxes[top], scores[top], labels[top]

        print(f"[{n + 1}/{len(img_files)}] {os.path.basename(path)} | {(time.time() - t0) * 1000:.1f} ms")

        vis_img = visualize(img, boxes, scores, labels, conf_thres=a.conf_thres)
        cv2.imwrite(os.path.join(a.output_dir, os.path.basename(path)), vis_img)

    print(f"Done. Visualized results saved to: {a.output_dir}")


if __name__ == "__main__":
    main()
