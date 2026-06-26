"""
eval_det_visdrone.py -- VisDrone-correct mAP detection evaluation on VisDrone test-dev.

Implements the following protocol (consistent with gen_dataset_visdrone.py):
  - maxDets = 500  (VisDrone images may have 200-600+ objects per frame)
  - ignore regions: score=0 / cls_id=0 or 11 -> a detection overlapping
    such a region is not counted as FP (intersection/det_area >= 0.5)
  - No truncation filtering: keep all GT including heavily truncated objects
    so mAP is consistent with training data (gen_dataset keeps all truncation)

test_dev layout:
    <data_dir>/VisDrone2019/test_dev/
        annotations/   <seq>.txt   MOT-format GT
        sequences/     <seq>/      image frames

VisDrone annotation per line:
    frame, track_id, x1, y1, w, h, score, cls_id(1-indexed), truncation, occlusion
    score=0 or cls_id=0/11 → ignored region (kept as ignore, not as GT)

Usage:
    python tools/eval_det_visdrone.py \
        --arch falcon_jde  \
        --load_model exp/mot/run/model_best.pth \
        --data_dir /path/to/datasets \
        --input-wh 864 480 --eval_spatial_size 480 864 \
        --num_classes 10 --reid_dim 128 \
        --K 300 --use_s4 --gpus 0 --quiet
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import os.path as osp
from collections import defaultdict

import torch
from tqdm import tqdm

import _paths  # noqa: F401  (sys.path bootstrap)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from falconmot.models.model import create_model, load_model
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracking_utils.log import logger
from falconmot.tracking_utils.timer import Timer
import falconmot.datasets.dataset.jde as datasets
from falconmot.opts import opts


# ---------------------------------------------------------------------------
# VisDrone class names (0-indexed internally, 1-indexed in annotations)
# ---------------------------------------------------------------------------

VISDRONE_CLASSES = {
    1: 'pedestrian', 2: 'people',   3: 'bicycle',  4: 'car',
    5: 'van',        6: 'truck',    7: 'tricycle',  8: 'awning-tricycle',
    9: 'bus',       10: 'motor',
}


# ---------------------------------------------------------------------------
# GT parser
# ---------------------------------------------------------------------------

def parse_visdrone_gt(ann_path):
    """Parse VisDrone MOT annotation.

    Returns:
        gt_dict:     frame_id → [(x1,y1,w,h,cls_1idx), ...]  valid GT boxes
        ignore_dict: frame_id → [(x1,y1,w,h), ...]           ignored regions

    Filtering is consistent with gen_dataset_visdrone.py:
        - score=0 OR cls_id=0 OR cls_id=11 → ignore region
        - No occlusion filtering  : every occluded object is a valid GT
        - No truncation filtering : every truncated object is a valid GT
          (gen_dataset keeps them in training too -> consistent mAP)
    """
    gt     = defaultdict(list)
    ignore = defaultdict(list)
    if not osp.isfile(ann_path):
        return gt, ignore
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            x1, y1   = float(parts[2]), float(parts[3])
            w,  h    = float(parts[4]), float(parts[5])
            score    = int(parts[6])
            cls_id   = int(parts[7])   # 1-indexed

            if w <= 0 or h <= 0:
                continue

            # Ignore regions: score=0, cls_id=0 (unlabeled), cls_id=11 (others)
            if score == 0 or cls_id == 0 or cls_id == 11:
                ignore[frame_id].append((x1, y1, w, h))
                continue

            gt[frame_id].append((x1, y1, w, h, cls_id))
    return gt, ignore



def overlaps_ignore(x1, y1, w, h, ignore_boxes, thr=0.5):
    """True if detection is mostly inside an ignored region.

    Uses intersection/det_area (not IoU) — matches the VisDrone MATLAB toolkit
    logic: a small detection fully inside a large ignore region is filtered even
    when standard IoU would be low.
    """
    det_area = w * h
    if det_area <= 0 or not ignore_boxes:
        return False
    x2, y2 = x1 + w, y1 + h
    for (ix, iy, iw, ih) in ignore_boxes:
        inter_w = max(0.0, min(x2, ix + iw) - max(x1, ix))
        inter_h = max(0.0, min(y2, iy + ih) - max(y1, iy))
        if inter_w * inter_h / det_area >= thr:
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(opt):
    if getattr(opt, 'quiet', False):
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    net_w, net_h = opt.img_size          # img_size = (W, H)
    num_classes  = opt.num_classes

    postprocessor = FalconJDEPostProcessor(num_classes=num_classes, num_top_queries=500)
    # Plain resize: no set_net_hw -> postprocessor inverts via norm*orig

    # ── Paths ─────────────────────────────────────────────────────────────────
    test_dev_root = osp.join(opt.data_dir, 'VisDrone2019', 'test_dev')
    seq_root      = osp.join(test_dev_root, 'sequences')
    ann_dir       = osp.join(test_dev_root, 'annotations')

    if not osp.isdir(seq_root):
        raise FileNotFoundError(f'sequences dir not found: {seq_root}')

    seqs = sorted([d for d in os.listdir(seq_root)
                   if osp.isdir(osp.join(seq_root, d))])
    logger.info('Found %d sequences', len(seqs))

    # ── Model ─────────────────────────────────────────────────────────────────
    print('Creating model...')
    model = create_model(opt.arch, opt)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    # ── COCO accumulators ─────────────────────────────────────────────────────
    coco_images  = []
    coco_gt_anns = []
    coco_dt_anns = []
    ann_id  = 0
    img_id  = 0
    timer   = Timer()

    # ── Sequence loop ─────────────────────────────────────────────────────────
    seq_bar = tqdm(seqs, desc='sequences', unit='seq',
                   bar_format='{l_bar}{bar:15}{r_bar}',
                   disable=getattr(opt, 'quiet', False))

    for seq in seq_bar:
        gt_dict, ignore_dict = parse_visdrone_gt(osp.join(ann_dir, f'{seq}.txt'))
        loader      = datasets.LoadImages(osp.join(seq_root, seq), opt.img_size)
        seq_frames  = 0
        seq_timer   = Timer()

        for frame_id, (path, img, img0) in enumerate(loader, start=1):
            img_id += 1
            orig_h, orig_w = img0.shape[:2]

            coco_images.append({'id': img_id, 'height': orig_h, 'width': orig_w})

            # ── GT annotations for this frame ──────────────────────────────
            for (x1, y1, w, h, cls_1idx) in gt_dict.get(frame_id, []):
                ann_id += 1
                coco_gt_anns.append({
                    'id':          ann_id,
                    'image_id':    img_id,
                    'category_id': cls_1idx,          # already 1-indexed
                    'bbox':        [x1, y1, w, h],    # already pixel xywh
                    'area':        float(w * h),
                    'iscrowd':     0,
                })

            # ── Model forward ──────────────────────────────────────────────
            blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)
            orig_sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
            timer.tic()
            seq_timer.tic()
            with torch.no_grad():
                outputs = model(blob)
            timer.toc()
            seq_timer.toc()
            seq_frames += 1

            # -- Decode predictions (same as the model PostProcessor) -------
            results      = postprocessor(outputs, orig_sizes)   # list[dict]
            res          = results[0]
            sc_t         = res['scores'].cpu()          # (K,)
            cls_t        = res['labels'].cpu()          # (K,)  0-indexed
            boxes_xyxy   = res['boxes'].cpu().numpy()   # (K, 4) pixel xyxy

            frame_ignores = ignore_dict.get(frame_id, [])

            for i in range(len(sc_t)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                w_ = x2 - x1;  h_ = y2 - y1
                if w_ <= 0 or h_ <= 0:
                    continue
                if overlaps_ignore(x1, y1, w_, h_, frame_ignores):
                    continue
                coco_dt_anns.append({
                    'image_id':    img_id,
                    'category_id': int(cls_t[i]) + 1,   # 0-idx → 1-idx
                    'bbox':        [float(x1), float(y1), float(w_), float(h_)],
                    'score':       float(sc_t[i]),
                })

        seq_fps = seq_frames / max(seq_timer.total_time, 1e-5)
        seq_bar.set_postfix({'frames': seq_frames, 'fps': f'{seq_fps:.1f}'}, refresh=False)

    # ── COCOeval ──────────────────────────────────────────────────────────────
    total_frames = timer.calls
    avg_fps      = total_frames / max(timer.total_time, 1e-5)
    logger.info('Frames: %d  |  Avg FPS: %.1f', total_frames, avg_fps)

    print('\n' + '=' * 60)
    print(f'  VisDrone mAP — test-dev  '
          f'({len(seqs)} seqs, {total_frames} frames)')
    print('=' * 60)

    categories = [{'id': i + 1, 'name': VISDRONE_CLASSES.get(i + 1, str(i))}
                  for i in range(num_classes)]

    gt_dict_coco = {
        'images':      coco_images,
        'annotations': coco_gt_anns,
        'categories':  categories,
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_dict_coco
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_dt_anns)

    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.params.maxDets = [1, 10, 500]   # Fix 1: VisDrone uses 500, not 100
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    s = ev.stats
    metrics = {
        'mAP':    float(s[0]),
        'mAP50':  float(s[1]),
        'mAP75':  float(s[2]),
        'mAP_s':  float(s[3]),
        'mAP_m':  float(s[4]),
        'mAP_l':  float(s[5]),
        'AR@1':   float(s[6]),
        'AR@10':  float(s[7]),
        'AR@500': float(s[8]),   # Fix 1: was AR@100
        'AR_s':   float(s[9]),
        'AR_m':   float(s[10]),
        'AR_l':   float(s[11]),
    }

    print('=' * 60)
    print(f"  mAP        : {metrics['mAP']:.4f}")
    print(f"  mAP50      : {metrics['mAP50']:.4f}")
    print(f"  mAP75      : {metrics['mAP75']:.4f}")
    print(f"  mAP_small  : {metrics['mAP_s']:.4f}")
    print(f"  mAP_medium : {metrics['mAP_m']:.4f}")
    print(f"  mAP_large  : {metrics['mAP_l']:.4f}")
    print(f"  AR@500     : {metrics['AR@500']:.4f}")
    print('=' * 60)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    opt.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 else 'cpu'
    main(opt)