"""
Visualise the FalconMOT training augmentation pipeline (coco_detection.py path).

Mirrors VisDroneCocoDataset.__getitem__ exactly, single-frame path:
    Original -> HSV -> Letterbox -> [SmallObjZoom] -> Geometric -> [GridMask] -> Flip

Geometric step at runtime is EITHER affine OR homography (mutually exclusive).
The pipeline figure shows the affine branch; a second figure is a gallery of
random homography warps so you can see the synthetic viewpoint diversity.

Two figures per sample:
  Figure 1 - step-by-step pipeline (affine branch)
  Figure 2 - homography gallery (letterboxed + N random perspective warps)

Reads labels from a COCO json (the format used for training) by default, or
from VisDrone JDE-format labels_with_ids/ via --jde_root.

Usage:
    # COCO json (matches training data)
    python tools/visualize_augmentation.py \
        --coco_json /workspace/VisDrone2019-COCO/val/annotations/instances_val.json \
        --img_dir   /workspace/VisDrone2019-COCO/val/images \
        --n 4 --no_show

    # tune homography + opt-in augments
    python tools/visualize_augmentation.py --coco_json ... --img_dir ... \
        --homography_strength 0.14 --small_obj_zoom --gridmask

    # JDE-format fallback
    python tools/visualize_augmentation.py --jde_root VisDrone2019-7cls --split val
"""

import sys
import os
import json
import argparse
import random
import warnings
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from falconmot.datasets.augment import (
    augment_hsv,
    random_affine,
    random_homography_warp,
    _camera_homography,
    _apply_homography,
)


# ---------------------------------------------------------------------------
# Class palette / names
# ---------------------------------------------------------------------------

CLS_NAMES = {
    0: 'pedestrian', 1: 'people',  2: 'bicycle',  3: 'car',
    4: 'van',        5: 'truck',   6: 'tricycle', 7: 'awning-tri',
    8: 'bus',        9: 'motor',
}
_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 81, 200, 120), ( 60, 143, 255),
    (255,  56, 132), (133, 132, 255), (255, 191,  56), ( 56, 255, 220),
    (200,  80,  50), ( 80, 200, 255),
]


def cls_color(c):
    return _PALETTE[int(c) % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Letterbox (matches coco_detection._letterbox: pads to 114)
# ---------------------------------------------------------------------------

def letterbox(img, net_h, net_w, color=(114, 114, 114)):
    h, w = img.shape[:2]
    ratio = min(net_h / h, net_w / w)
    nw, nh = round(w * ratio), round(h * ratio)
    dw, dh = (net_w - nw) * 0.5, (net_h - nh) * 0.5
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, left, top


# ---------------------------------------------------------------------------
# Label conversions  (labels = (N,6) [cls, tid, cx, cy, w, h] normalized)
# ---------------------------------------------------------------------------

def norm_to_xyxy_px(labels, ratio, pad_w, pad_h, w0, h0):
    """normalized cxcywh (orig) -> pixel xyxy in letterboxed space."""
    if len(labels) == 0:
        return np.zeros((0, 6), np.float32)
    out = labels.copy()
    out[:, 2] = ratio * w0 * (labels[:, 2] - labels[:, 4] * 0.5) + pad_w
    out[:, 3] = ratio * h0 * (labels[:, 3] - labels[:, 5] * 0.5) + pad_h
    out[:, 4] = ratio * w0 * (labels[:, 2] + labels[:, 4] * 0.5) + pad_w
    out[:, 5] = ratio * h0 * (labels[:, 3] + labels[:, 5] * 0.5) + pad_h
    return out


def xyxy_px_to_norm(lbs, net_w, net_h):
    if len(lbs) == 0:
        return np.zeros((0, 6), np.float32)
    out = lbs.copy()
    out[:, 2] = (lbs[:, 2] + lbs[:, 4]) * 0.5 / net_w
    out[:, 3] = (lbs[:, 3] + lbs[:, 5]) * 0.5 / net_h
    out[:, 4] = (lbs[:, 4] - lbs[:, 2]) / net_w
    out[:, 5] = (lbs[:, 5] - lbs[:, 3]) / net_h
    return out


def obj_stats(labels_norm, w, h):
    if len(labels_norm) == 0:
        return 0, 0.0
    bw, bh = labels_norm[:, 4] * w, labels_norm[:, 5] * h
    return len(labels_norm), float(np.median(np.sqrt(bw ** 2 + bh ** 2)))


def draw_boxes_norm(img_bgr, labels_norm, h, w):
    out = img_bgr.copy()
    for lb in labels_norm:
        cx, cy, bw, bh = lb[2] * w, lb[3] * h, lb[4] * w, lb[5] * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), cls_color(int(lb[0])), 1)
    return out


def draw_boxes_xyxy(img_bgr, lbs):
    out = img_bgr.copy()
    for lb in lbs:
        x1, y1, x2, y2 = map(int, lb[2:6])
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), cls_color(int(lb[0])), 1)
    return out


# ---------------------------------------------------------------------------
# Ported opt-in augments (faithful copies of coco_detection methods)
# ---------------------------------------------------------------------------

def small_obj_zoom(img, lbs, p=1.0):
    """Pixel-xyxy zoom crop anchored on a small object."""
    if random.random() > p or len(lbs) == 0:
        return img, lbs
    H, W = img.shape[:2]
    ws, hs = lbs[:, 4] - lbs[:, 2], lbs[:, 5] - lbs[:, 3]
    small = np.where((ws * hs) / (W * H) < 0.002)[0]
    if len(small) == 0:
        small = np.arange(len(lbs))
    a = lbs[random.choice(small)]
    cx, cy = (a[2] + a[4]) * 0.5, (a[3] + a[5]) * 0.5
    s = random.uniform(0.30, 0.55)
    cw, ch = max(64, int(W * s)), max(64, int(H * s))
    left = random.randint(max(0, int(cx) - cw + 1), max(0, min(W - cw, int(cx))))
    top  = random.randint(max(0, int(cy) - ch + 1), max(0, min(H - ch, int(cy))))
    ccx, ccy = (lbs[:, 2] + lbs[:, 4]) * 0.5, (lbs[:, 3] + lbs[:, 5]) * 0.5
    inside = (ccx >= left) & (ccx < left + cw) & (ccy >= top) & (ccy < top + ch)
    if not inside.any():
        return img, lbs
    crop = img[top:top + ch, left:left + cw]
    out_img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    sx, sy = W / cw, H / ch
    nl = lbs[inside].copy()
    nl[:, 2] = np.clip((lbs[inside, 2] - left) * sx, 0, W)
    nl[:, 3] = np.clip((lbs[inside, 3] - top) * sy, 0, H)
    nl[:, 4] = np.clip((lbs[inside, 4] - left) * sx, 0, W)
    nl[:, 5] = np.clip((lbs[inside, 5] - top) * sy, 0, H)
    return out_img, nl


def gridmask(img, p=1.0):
    """Erase a regular grid of squares to simulate partial occlusion."""
    if random.random() > p:
        return img
    H, W = img.shape[:2]
    d = random.randint(40, 80)
    r = int(d * random.uniform(0.35, 0.55))
    dx, dy = random.randint(0, d), random.randint(0, d)
    tile = np.ones((d, d), np.uint8)
    tile[:r, :r] = 0
    mask = np.tile(tile, ((H + d) // d + 1, (W + d) // d + 1))[dy:dy + H, dx:dx + W]
    return img * mask[:, :, None]


# ---------------------------------------------------------------------------
# Sample loaders
# ---------------------------------------------------------------------------

def load_coco(coco_json, img_dir, max_n, seed):
    with open(coco_json) as f:
        data = json.load(f)
    cat_ids = sorted(c['id'] for c in data['categories'])
    cat2idx = {cid: i for i, cid in enumerate(cat_ids)}
    imgs = {im['id']: im for im in data['images']}
    anns = defaultdict(list)
    for a in data['annotations']:
        anns[a['image_id']].append(a)
    ids = [i for i in imgs if anns[i]]
    random.Random(seed).shuffle(ids)
    samples = []
    for img_id in ids[:max_n]:
        im = imgs[img_id]
        img = cv2.imread(os.path.join(img_dir, im['file_name']))
        if img is None:
            continue
        h, w = im['height'], im['width']
        labels = []
        for a in anns[img_id]:
            x, y, bw, bh = a['bbox']
            labels.append([cat2idx.get(a['category_id'], 0), 0,
                           (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
        samples.append((img, np.array(labels, np.float32), im['file_name']))
    return samples


def load_jde(root, split, max_n, seed):
    img_root = os.path.join(root, split, 'images')
    lbl_root = os.path.join(root, split, 'labels_with_ids')
    pairs = []
    for seq in sorted(os.listdir(img_root)):
        d = os.path.join(img_root, seq)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = os.path.splitext(fn)[0]
                pairs.append((os.path.join(d, fn),
                              os.path.join(lbl_root, seq, stem + '.txt')))
    random.Random(seed).shuffle(pairs)
    samples = []
    for ip, lp in pairs:
        if not (os.path.isfile(lp) and os.path.getsize(lp) > 0):
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            labels = np.loadtxt(lp, dtype=np.float32).reshape(-1, 6)
        samples.append((img, labels, os.path.basename(ip)))
        if len(samples) >= max_n:
            break
    return samples


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pipeline(img_bgr, labels, net_h, net_w, args, axes):
    """Step-by-step, mirrors coco_detection.__getitem__ single-frame path."""
    orig_h, orig_w = img_bgr.shape[:2]
    panels = []

    # [1] Original
    n, m = obj_stats(labels, orig_w, orig_h)
    panels.append((draw_boxes_norm(img_bgr, labels, orig_h, orig_w),
                   '[1] Original  {}x{}\nobjs={} med={:.1f}px'.format(orig_w, orig_h, n, m)))

    # [2] HSV
    hsv = img_bgr.copy()
    augment_hsv(hsv, fraction=0.50)
    panels.append((draw_boxes_norm(hsv, labels, orig_h, orig_w), '[2] HSV (S+V +-50%)'))

    # [3] Letterbox -> pixel xyxy
    lb_img, ratio, pw, ph = letterbox(hsv, net_h, net_w)
    lbs = norm_to_xyxy_px(labels, ratio, pw, ph, orig_w, orig_h)
    panels.append((draw_boxes_xyxy(lb_img, lbs),
                   '[3] Letterbox  {}x{}'.format(net_w, net_h)))

    # [4] Small-object zoom (opt-in)
    if args.small_obj_zoom:
        lb_img, lbs = small_obj_zoom(lb_img.copy(), lbs.copy(), p=1.0)
        panels.append((draw_boxes_xyxy(lb_img, lbs),
                       '[4] SmallObjZoom\nobjs={}'.format(len(lbs))))

    # [5] Geometric - affine branch
    af_img, af_lbs, _ = random_affine(
        lb_img.copy(), lbs.copy(),
        degrees=(-5, 5), translate=(0.10, 0.10),
        scale=(0.50, 1.20), shear=(-2, 2))
    panels.append((draw_boxes_xyxy(af_img, af_lbs),
                   '[5] Affine branch\nobjs={}'.format(len(af_lbs))))

    cur_img, cur_lbs = af_img, af_lbs

    # [6] GridMask (opt-in)
    if args.gridmask:
        cur_img = gridmask(cur_img.copy(), p=1.0)
        panels.append((draw_boxes_xyxy(cur_img, cur_lbs), '[6] GridMask'))

    # [final] flip
    fin = cur_img.copy()
    fin_norm = xyxy_px_to_norm(cur_lbs, net_w, net_h)
    if random.random() > 0.5:
        fin = np.fliplr(fin)
        if len(fin_norm):
            fin_norm[:, 2] = 1.0 - fin_norm[:, 2]
    n, m = obj_stats(fin_norm, net_w, net_h)
    panels.append((draw_boxes_norm(fin, fin_norm, net_h, net_w),
                   '[final] +Flip\nobjs={} med={:.1f}px'.format(n, m)))

    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im[:, :, ::-1]); ax.set_title(title, fontsize=8, pad=3); ax.axis('off')
    for ax in axes[len(panels):]:
        ax.axis('off')


def fig_homography(img_bgr, labels, net_h, net_w, strength, axes):
    """Letterboxed reference + a smooth drone-flight trajectory of warps.

    Sweeps a banking fly-through: yaw pans left->right, roll banks into the
    turn, pitch makes a gentle tilt arc, altitude bobs slightly — so the
    panels read like consecutive frames of a drone manoeuvring, not random
    distortions. Each warp is a physically-valid camera-pose homography.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    lb_img, ratio, pw, ph = letterbox(img_bgr, net_h, net_w)
    lbs = norm_to_xyxy_px(labels, ratio, pw, ph, orig_w, orig_h)

    axes[0].imshow(draw_boxes_xyxy(lb_img, lbs)[:, :, ::-1])
    axes[0].set_title('Letterboxed (ref)\nobjs={}'.format(len(lbs)),
                      fontsize=8, pad=3, color='#2060c0')
    axes[0].axis('off')

    n = len(axes) - 1
    ang = strength * 110.0
    ts = np.linspace(0.0, 1.0, n)
    for ax, t in zip(axes[1:], ts):
        yaw   = (2 * t - 1) * ang                 # pan across the turn
        pitch = ang * 0.6 * np.sin(np.pi * t)     # gentle tilt arc
        roll  = -(2 * t - 1) * ang * 0.4          # bank into the turn
        scale = 1.0 + 0.05 * np.sin(np.pi * t)    # slight altitude bob
        H = _camera_homography(net_w, net_h, pitch, yaw, roll, scale)
        wim, wlbs, _ = _apply_homography(lb_img.copy(), lbs.copy(), H,
                                         (114, 114, 114), 4)
        ax.imshow(draw_boxes_xyxy(wim, wlbs)[:, :, ::-1])
        ax.set_title('yaw={:+.0f} pitch={:+.0f}\nroll={:+.0f}  objs={}'.format(
            yaw, pitch, roll, len(wlbs)), fontsize=8, pad=3, color='#208040')
        ax.axis('off')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco_json', default='', help='COCO instances json (training format)')
    p.add_argument('--img_dir',   default='', help='image dir for --coco_json')
    p.add_argument('--jde_root',  default='', help='VisDrone JDE-format root (fallback)')
    p.add_argument('--split',     default='val', choices=['train', 'val'])
    p.add_argument('--input_w', type=int, default=864)
    p.add_argument('--input_h', type=int, default=480)
    p.add_argument('--n',       type=int, default=4)
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--out_dir', default='out/aug_vis')
    p.add_argument('--no_show', action='store_true')
    p.add_argument('--homography_strength', type=float, default=0.12)
    p.add_argument('--homography_views',    type=int,   default=5)
    p.add_argument('--small_obj_zoom', action='store_true', help='show SmallObjZoom step')
    p.add_argument('--gridmask',       action='store_true', help='show GridMask step')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.coco_json:
        if not args.img_dir:
            print('[ERROR] --img_dir required with --coco_json'); sys.exit(1)
        samples = load_coco(args.coco_json, args.img_dir, args.n, args.seed)
    elif args.jde_root:
        samples = load_jde(args.jde_root, args.split, args.n, args.seed)
    else:
        print('[ERROR] provide --coco_json (+ --img_dir) or --jde_root'); sys.exit(1)

    if not samples:
        print('[ERROR] no labelled samples found'); sys.exit(1)
    print('Visualising {} samples  net={}x{}'.format(len(samples), args.input_w, args.input_h))

    n_pipe = 5 + int(args.small_obj_zoom) + int(args.gridmask) + 1
    for idx, (img, labels, name) in enumerate(samples):
        tag = os.path.splitext(os.path.basename(name))[0]

        f1, ax1 = plt.subplots(1, n_pipe, figsize=(3.6 * n_pipe, 4.2))
        f1.suptitle('[Pipeline] {}'.format(name), fontsize=9, y=1.02)
        fig_pipeline(img, labels, args.input_h, args.input_w, args, np.atleast_1d(ax1))
        plt.tight_layout()
        o1 = os.path.join(args.out_dir, '{}_pipeline.png'.format(tag))
        plt.savefig(o1, dpi=120, bbox_inches='tight'); print('  [{}] {}'.format(idx + 1, o1))
        if not args.no_show: plt.show()
        plt.close(f1)

        ncol = 1 + args.homography_views
        f2, ax2 = plt.subplots(1, ncol, figsize=(3.4 * ncol, 4.0))
        f2.suptitle('[Homography viewpoints] {}  strength={}'.format(name, args.homography_strength),
                    fontsize=9, y=1.02)
        fig_homography(img, labels, args.input_h, args.input_w,
                       args.homography_strength, np.atleast_1d(ax2))
        plt.tight_layout()
        o2 = os.path.join(args.out_dir, '{}_homography.png'.format(tag))
        plt.savefig(o2, dpi=120, bbox_inches='tight'); print('        {}'.format(o2))
        if not args.no_show: plt.show()
        plt.close(f2)

    print('\nDone -> {}'.format(os.path.abspath(args.out_dir)))


if __name__ == '__main__':
    main()
