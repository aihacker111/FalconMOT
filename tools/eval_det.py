"""
eval_det.py -- evaluate a FalconJDE detector on VisDrone-DET with the
correct protocol (COCO AP @ maxDets=500, the VisDrone standard — NOT COCO's
default 100), and optionally export per-image results in VisDrone submission
format for the official toolkit / test-dev server.

Two ways to evaluate:
  (1) --gt_json present  → compute AP/AP50/AP75/APs/m/l + AR1/AR10/AR500 with
                            pycocotools at maxDets=500 (matches val-set SOTA tables).
  (2) test-dev (no GT)   → use --export_dir to write VisDrone .txt per image,
                            zip and submit to the evaluation server.

Optional --tile enables SAHI-style sliced inference (keeps tiny objects at full
resolution) — recommended for VisDrone's large frames.

Usage (val, with GT):
    python tools/eval_det.py \
        --arch falcon_jde --load_model exp/.../model_best.pth \
        --img_dir  /data/VisDrone2019-DET-COCO/val/images \
        --gt_json  /data/VisDrone2019-DET-COCO/val/annotations/instances_val.json \
        --input-wh 1088 640 --num_queries 500

Usage (test-dev submission):
    python tools/eval_det.py \
        --arch falcon_jde --load_model exp/.../model_best.pth \
        --img_dir /data/VisDrone2019-DET-test-dev/images \
        --export_dir out/visdrone_testdev_results --tile
"""

import _paths            # noqa: F401  (sys.path bootstrap)
import os
import json
import argparse
import glob

import numpy as np
import cv2
import torch

from falconmot.cfg.args import opts
from falconmot.nn import create_model, load_model
from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------------------
# Preprocess (letterbox, matches coco_detection._letterbox)
# ---------------------------------------------------------------------------

def resize_plain(img, net_h, net_w):
    # Plain resize to (net_w, net_h) -- no letterbox/pad
    return cv2.resize(img, (net_w, net_h), interpolation=cv2.INTER_AREA)


def to_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def infer_image(model, post, img_bgr, net_h, net_w, device):
    """Run the model on one (letterboxed) image → (boxes_xyxy, scores, labels) in orig px."""
    H0, W0 = img_bgr.shape[:2]
    lb = resize_plain(img_bgr, net_h, net_w)
    x = to_tensor(lb).to(device)
    out = model(x)
    res = post(out, torch.tensor([[H0, W0]], device=device))[0]
    return (res['boxes'].cpu().numpy(),
            res['scores'].cpu().numpy(),
            res['labels'].cpu().numpy())


def _nms(boxes, scores, iou_thr=0.6):
    if len(boxes) == 0:
        return np.empty((0,), int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, int)


@torch.no_grad()
def infer_tiled(model, post, img_bgr, net_h, net_w, device, rows=2, cols=2, overlap=0.2):
    """SAHI-style sliced inference: detect on overlapping crops at higher
    effective resolution, map back to full image, then NMS. Helps tiny objects."""
    H0, W0 = img_bgr.shape[:2]
    th, tw = int(H0 / rows), int(W0 / cols)
    oy, ox = int(th * overlap), int(tw * overlap)
    all_b, all_s, all_l = [], [], []
    # tiles + the full frame (catches large objects)
    crops = [(0, 0, W0, H0)]
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, c * tw - ox); y1 = max(0, r * th - oy)
            x2 = min(W0, (c + 1) * tw + ox); y2 = min(H0, (r + 1) * th + oy)
            crops.append((x1, y1, x2, y2))
    for (x1, y1, x2, y2) in crops:
        b, s, l = infer_image(model, post, img_bgr[y1:y2, x1:x2], net_h, net_w, device)
        if len(b):
            b = b.copy(); b[:, [0, 2]] += x1; b[:, [1, 3]] += y1
            all_b.append(b); all_s.append(s); all_l.append(l)
    if not all_b:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), int)
    boxes = np.concatenate(all_b); scores = np.concatenate(all_s); labels = np.concatenate(all_l)
    # class-wise NMS to merge duplicates from overlapping tiles
    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        keep = _nms(boxes[idx], scores[idx])
        keep_all.extend(idx[keep])
    keep_all = np.array(keep_all, int)
    return boxes[keep_all], scores[keep_all], labels[keep_all]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_json',    default='', help='COCO GT json (val) → compute AP@500')
    ap.add_argument('--img_dir',    required=True)
    ap.add_argument('--export_dir', default='', help='write VisDrone-format .txt per image')
    ap.add_argument('--conf_thres', type=float, default=0.001,
                    help='keep low for AP eval; raise (~0.3) for submission')
    ap.add_argument('--max_dets',   type=int, default=500)
    ap.add_argument('--tile', action='store_true', help='SAHI-style sliced inference')
    ap.add_argument('--tile_grid', type=int, nargs=2, default=[2, 2])
    a, unknown = ap.parse_known_args()

    # FalconJDE model opts (arch, input-wh, num_queries, etc.) via the project parser
    opt = opts().init(unknown)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_w, net_h = opt.input_wh[0], opt.input_wh[1]

    print('Creating model...')
    model = create_model(opt.arch, opt)
    assert opt.load_model, '--load_model is required'
    model, _, _ = load_model(model, opt.load_model)
    model = model.to(device).eval()

    post = FalconJDEPostProcessor(num_classes=opt.num_classes,
                                  num_top_queries=a.max_dets)
    # Plain resize -> do not call set_net_hw (postprocessor inverts via norm*orig)

    # image list (+ map filename → coco image_id if GT given)
    name2id, gt = {}, None
    if a.gt_json:
        gt = json.load(open(a.gt_json))
        name2id = {im['file_name']: im['id'] for im in gt['images']}
        img_files = [os.path.join(a.img_dir, im['file_name']) for im in gt['images']]
    else:
        img_files = sorted(p for p in glob.glob(os.path.join(a.img_dir, '*'))
                           if p.lower().endswith(_IMG_EXTS))
    print(f'Evaluating {len(img_files)} images  net={net_w}x{net_h}  '
          f'tiled={a.tile}  maxDets={a.max_dets}')

    if a.export_dir:
        os.makedirs(a.export_dir, exist_ok=True)

    dt_coco = []
    for n, path in enumerate(img_files):
        img = cv2.imread(path)
        if img is None:
            continue
        if a.tile:
            boxes, scores, labels = infer_tiled(model, post, img, net_h, net_w, device,
                                                rows=a.tile_grid[0], cols=a.tile_grid[1])
        else:
            boxes, scores, labels = infer_image(model, post, img, net_h, net_w, device)

        keep = scores >= a.conf_thres
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        # cap to max_dets by score
        if len(scores) > a.max_dets:
            top = scores.argsort()[::-1][:a.max_dets]
            boxes, scores, labels = boxes[top], scores[top], labels[top]

        fname = os.path.basename(path)

        if a.gt_json and fname in name2id:
            iid = name2id[fname]
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                dt_coco.append({'image_id': iid, 'category_id': int(l) + 1,
                                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                'score': float(s)})

        if a.export_dir:
            with open(os.path.join(a.export_dir, os.path.splitext(fname)[0] + '.txt'), 'w') as f:
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    # VisDrone: bbox_left,bbox_top,w,h,score,category(1-idx),-1,-1
                    f.write(f'{x1:.2f},{y1:.2f},{(x2-x1):.2f},{(y2-y1):.2f},'
                            f'{s:.4f},{int(l)+1},-1,-1\n')

        if (n + 1) % 100 == 0:
            print(f'  {n+1}/{len(img_files)}')

    if a.export_dir:
        print(f'[export] VisDrone-format results → {a.export_dir} '
              f'(zip & submit, or run the official toolkit)')

    # ── COCO AP @ maxDets=500 (VisDrone protocol) ─────────────────────────
    if a.gt_json:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(a.gt_json)
        if not dt_coco:
            print('[eval] no detections — AP=0'); return
        coco_dt = coco_gt.loadRes(dt_coco)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.maxDets = [1, 10, a.max_dets]      # VisDrone: AP @ maxDets=500
        ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
        print(f'\n[VisDrone protocol, maxDets={a.max_dets}]')
        print(f'  AP={s[0]:.4f}  AP50={s[1]:.4f}  AP75={s[2]:.4f}  '
              f'APs={s[3]:.4f}  APm={s[4]:.4f}  APl={s[5]:.4f}')
        print(f'  AR1={s[6]:.4f}  AR10={s[7]:.4f}  AR{a.max_dets}={s[8]:.4f}')


if __name__ == '__main__':
    main()