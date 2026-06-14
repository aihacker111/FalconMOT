"""
VisDrone COCO-format dataset for DEIM-JDE training.

Augmentation pipeline — AMOT-exact with letterbox:
  1. augment_hsv  : random S+V scaling (fraction=0.50) on raw BGR image
  2. resize       : plain resize to (width, height) — KHÔNG letterbox, KHÔNG pad
  3. random_affine: degrees=(-5,5), translate=0.10, scale=(0.50,1.20), shear=(-2,2)
  4. horizontal flip (50%)

Batch dict keys:
    input          : (3, H, W) float32 tensor, scaled to [0, 1]
    detr_boxes     : (max_objs, 4) cxcywh normalized [0,1]
    detr_labels    : (max_objs,)   int64, 0-indexed class id  (-1 = padding)
    detr_track_ids : (max_objs,)   int64, 0-indexed global track id (-1 = padding)
    detr_num_objs  : int64 scalar — number of valid entries
    orig_hw        : (2,) int64 [orig_H, orig_W] before letterbox (for COCO eval)
    coco_image_id  : int64 — COCO image id (for eval)
    lb_pad         : (2,) int32 [pad_w, pad_h] exact pixel offsets from letterbox
"""

import glob
import os
import json
import random
import numpy as np
import cv2
import torch
from collections import defaultdict

from falconmot.datasets.augment import augment_hsv, random_affine, random_homography_warp


def _letterbox(img, height, width, color=(127.5, 127.5, 127.5)):
    shape = img.shape[:2]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width  - new_shape[0]) * 0.5
    dh = (height - new_shape[1]) * 0.5
    top,  bottom = round(dh - 0.1), round(dh + 0.1)
    left, right  = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, left, top


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class VisDroneCocoDataset(torch.utils.data.Dataset):
    """
    COCO-format VisDrone dataset for DEIM-JDE training/validation.

    Args:
        opt      : training options (from opts.py)
        img_root : directory containing images
        ann_file : path to COCO JSON (instances_train.json / instances_val.json)
        augment  : whether to apply training augmentations
    """


    def __init__(self, opt, img_root: str, ann_file: str, augment: bool = False):
        self.opt      = opt
        self.augment  = augment
        self.width    = opt.input_wh[0]   # W
        self.height   = opt.input_wh[1]   # H
        self.max_objs = getattr(opt, 'K', 300)

        # Required by opts.update_dataset_info_and_set_heads
        self.default_input_wh = [self.height, self.width]   # [H, W] — opts unpacks as input_h, input_w
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # no-op (normalization removed)
        self.std  = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Epoch-aware augmentation cutoff (AMOT augments all epochs by default)
        self.cur_epoch  = 0
        stop_epoch      = getattr(opt, 'stop_epoch', -1)
        self.stop_epoch = opt.num_epochs if stop_epoch < 0 else stop_epoch

        # ── Load COCO JSON ────────────────────────────────────────────────
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        self.img_info = {img['id']: img for img in coco['images']}
        self.img_ids  = [img['id'] for img in coco['images']]
        self.img_root = img_root

        self._ann_by_img = defaultdict(list)
        for ann in coco['annotations']:
            self._ann_by_img[ann['image_id']].append(ann)

        # ── nID_dict: per-class unique track ID count (for ArcFace sizing) ──
        max_tid = defaultdict(int)
        for ann in coco['annotations']:
            cls_id_0 = ann['category_id'] - 1
            tid      = ann['track_id']
            if tid + 1 > max_tid[cls_id_0]:
                max_tid[cls_id_0] = tid + 1
        self.nID_dict    = dict(max_tid)
        self.num_classes = len(coco['categories'])

        # ── New augmentation flags ────────────────────────────────────────
        # Temporal Mosaic: 4 frames from same sequence → 2×2 mosaic
        self.use_temporal_mosaic   = getattr(opt, 'temporal_mosaic',      False)
        self.temporal_mosaic_prob  = getattr(opt, 'temporal_mosaic_prob', 0.5)
        # Standard 4-image mosaic (DETR/YOLO-style; random images + random center)
        self.use_mosaic            = getattr(opt, 'mosaic',               False)
        self.mosaic_prob           = getattr(opt, 'mosaic_prob',          0.5)
        # Small Object Zoom: crop toward tiny objects before affine
        self.use_small_obj_zoom    = getattr(opt, 'small_obj_zoom',       False)
        self.small_obj_zoom_prob   = getattr(opt, 'small_obj_zoom_prob',  0.5)

        # Random perspective (homography) warp — synthetic viewpoint diversity.
        # Used as an ALTERNATIVE to affine (mutually exclusive) so each sample
        # gets exactly one geometric transform (no compounding distortion).
        self.use_homography        = getattr(opt, 'homography',           True)
        self.homography_prob       = getattr(opt, 'homography_prob',      0.3)
        self.homography_strength   = getattr(opt, 'homography_strength',  0.12)
        # GridMask: erase grid pattern to simulate partial occlusion
        self.use_gridmask          = getattr(opt, 'gridmask',             False)
        self.gridmask_prob         = getattr(opt, 'gridmask_prob',        0.3)

        # ── Sequence index for temporal augmentation ──────────────────────
        self._build_seq_index()

        print(f'[VisDroneCocoDataset] {len(self.img_ids)} images  '
              f'{len(coco["annotations"])} annotations  '
              f'classes={self.num_classes}  augment={augment}')
        print(f'  temporal_mosaic={self.use_temporal_mosaic}  '
              f'small_obj_zoom={self.use_small_obj_zoom}  '
              f'gridmask={self.use_gridmask}  '
              f'seq_index={self._has_seq_idx} ({len(self._seq_to_ids)} seqs)')
        for cid, n in sorted(self.nID_dict.items()):
            print(f'  class {cid}: {n} unique IDs')

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        self.cur_epoch = epoch

    def __len__(self):
        return len(self.img_ids)

    # ------------------------------------------------------------------
    # Raw loader
    # ------------------------------------------------------------------

    def _load_raw(self, img_id: int):
        """Return BGR uint8 image + (N,6) [cls_0, tid_0, cx, cy, w, h] norm."""
        info     = self.img_info[img_id]
        img_path = os.path.join(self.img_root, info['file_name'])
        img      = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Cannot read {img_path}')

        H, W  = img.shape[:2]
        anns   = self._ann_by_img.get(img_id, [])
        labels = np.zeros((len(anns), 6), dtype=np.float32)
        for i, ann in enumerate(anns):
            x1, y1, bw, bh = ann['bbox']
            labels[i] = [
                ann['category_id'] - 1,   # cls  0-indexed
                ann['track_id'],           # tid  0-indexed global
                (x1 + bw * 0.5) / W,      # cx   normalized
                (y1 + bh * 0.5) / H,      # cy   normalized
                bw / W,                    # w    normalized
                bh / H,                    # h    normalized
            ]
        return img, labels

    # ------------------------------------------------------------------
    # Sequence index (used by temporal mosaic)
    # ------------------------------------------------------------------

    def _build_seq_index(self):
        """Group img_ids by seq_id, sorted by frame_id."""
        seq_frames = {}
        for img_id, info in self.img_info.items():
            seq = info.get('seq_id')
            if seq is None:
                continue
            seq_frames.setdefault(seq, []).append((info.get('frame_id', 0), img_id))

        self._seq_to_ids = {}   # seq_id → [img_id, ...] sorted by frame_id
        self._id_to_pos  = {}   # img_id → (seq_id, position)
        for seq, frames in seq_frames.items():
            frames.sort()
            ids = [fid for _, fid in frames]
            self._seq_to_ids[seq] = ids
            for pos, img_id in enumerate(ids):
                self._id_to_pos[img_id] = (seq, pos)
        self._has_seq_idx = len(self._seq_to_ids) > 0

    # ------------------------------------------------------------------
    # Temporal Mosaic
    # ------------------------------------------------------------------

    def _temporal_mosaic(self, anchor_img_id: int):
        """Build 2×2 mosaic from 4 frames of the same sequence.

        Picks 3 partner frames at random temporal offsets [+5..+30 frames]
        relative to the anchor. Each tile gets independent HSV augmentation.
        Labels are remapped from per-frame cxcywh-norm to mosaic-norm space.
        """
        seq, pos = self._id_to_pos.get(anchor_img_id, (None, 0))
        if seq is not None:
            seq_ids = self._seq_to_ids[seq]
            n = len(seq_ids)
            max_off = max(5, min(30, n // 4))
            partners = []
            for sign in [1, -1, 1]:
                off = random.randint(5, max_off) * sign
                partners.append(seq_ids[max(0, min(n - 1, pos + off))])
        else:
            # Fallback: random frames from anywhere
            partners = random.choices(self.img_ids, k=3)

        four_ids = [anchor_img_id] + partners
        random.shuffle(four_ids)

        mid_x, mid_y = self.width // 2, self.height // 2
        placements = [
            (0,     0,     mid_x,      mid_y),
            (mid_x, 0,     self.width,  mid_y),
            (0,     mid_y, mid_x,      self.height),
            (mid_x, mid_y, self.width,  self.height),
        ]
        mosaic = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        all_lbs = []

        for img_id, (tx1, ty1, tx2, ty2) in zip(four_ids, placements):
            tile_img, tile_lbs = self._load_raw(img_id)
            # Independent HSV per tile → more colour diversity
            augment_hsv(tile_img, fraction=0.50)
            tw, th = tx2 - tx1, ty2 - ty1
            mosaic[ty1:ty2, tx1:tx2] = cv2.resize(
                tile_img, (tw, th), interpolation=cv2.INTER_LINEAR)
            if len(tile_lbs) > 0:
                lbs = tile_lbs.copy()
                lbs[:, 2] = tile_lbs[:, 2] * tw / self.width  + tx1 / self.width
                lbs[:, 3] = tile_lbs[:, 3] * th / self.height + ty1 / self.height
                lbs[:, 4] = tile_lbs[:, 4] * tw / self.width
                lbs[:, 5] = tile_lbs[:, 5] * th / self.height
                all_lbs.append(lbs)

        combined = (np.concatenate(all_lbs, 0) if all_lbs
                    else np.zeros((0, 6), dtype=np.float32))
        return mosaic, combined   # img (H,W,3) BGR, labels cxcywh-norm

    def _mosaic(self, anchor_img_id: int):
        """Standard 4-image mosaic (DETR/YOLO-style).

        Picks the anchor + 3 random images from anywhere in the dataset and
        tiles them around a RANDOM center → strong scale/context/position
        diversity (unlike temporal mosaic which uses same-sequence frames and a
        fixed 2x2 split). Returns a net-size image + cxcywh-norm labels, same
        contract as _temporal_mosaic.
        """
        W, H = self.width, self.height
        ids = [anchor_img_id] + random.choices(self.img_ids, k=3)
        random.shuffle(ids)

        cx = random.randint(int(0.30 * W), int(0.70 * W))   # random mosaic center
        cy = random.randint(int(0.30 * H), int(0.70 * H))
        placements = [
            (0,  0,  cx, cy),          # top-left
            (cx, 0,  W,  cy),          # top-right
            (0,  cy, cx, H),           # bottom-left
            (cx, cy, W,  H),           # bottom-right
        ]
        mosaic  = np.full((H, W, 3), 114, dtype=np.uint8)
        all_lbs = []

        for img_id, (tx1, ty1, tx2, ty2) in zip(ids, placements):
            tw, th = tx2 - tx1, ty2 - ty1
            if tw <= 1 or th <= 1:
                continue
            tile_img, tile_lbs = self._load_raw(img_id)
            augment_hsv(tile_img, fraction=0.50)            # independent HSV per tile
            mosaic[ty1:ty2, tx1:tx2] = cv2.resize(
                tile_img, (tw, th), interpolation=cv2.INTER_LINEAR)
            if len(tile_lbs) > 0:
                lbs = tile_lbs.copy()
                lbs[:, 2] = tile_lbs[:, 2] * tw / W + tx1 / W
                lbs[:, 3] = tile_lbs[:, 3] * th / H + ty1 / H
                lbs[:, 4] = tile_lbs[:, 4] * tw / W
                lbs[:, 5] = tile_lbs[:, 5] * th / H
                all_lbs.append(lbs)

        combined = (np.concatenate(all_lbs, 0) if all_lbs
                    else np.zeros((0, 6), dtype=np.float32))
        return mosaic, combined

    # ------------------------------------------------------------------
    # Small Object Zoom
    # ------------------------------------------------------------------

    def _small_obj_zoom(self, img, lbs_xyxy, p=0.5):
        """Zoom crop anchored on a small object.

        Works on pixel-xyxy labels (after letterbox conversion).
        Returns resized crop at full network size + remapped labels.
        Probability p controls how often the zoom is applied.
        """
        if random.random() > p or len(lbs_xyxy) == 0:
            return img, lbs_xyxy

        H, W = img.shape[:2]
        ws = lbs_xyxy[:, 4] - lbs_xyxy[:, 2]
        hs = lbs_xyxy[:, 5] - lbs_xyxy[:, 3]
        norm_areas = (ws * hs) / (W * H)
        small_idx  = np.where(norm_areas < 0.002)[0]
        if len(small_idx) == 0:
            small_idx = np.arange(len(lbs_xyxy))   # fallback: any object

        anchor  = lbs_xyxy[random.choice(small_idx)]
        cx_px   = (anchor[2] + anchor[4]) * 0.5
        cy_px   = (anchor[3] + anchor[5]) * 0.5

        scale = random.uniform(0.30, 0.55)
        cw = max(64, int(W * scale))
        ch = max(64, int(H * scale))

        left = random.randint(max(0, int(cx_px) - cw + 1),
                              max(0, min(W - cw, int(cx_px))))
        top  = random.randint(max(0, int(cy_px) - ch + 1),
                              max(0, min(H - ch, int(cy_px))))

        # Keep only objects with center inside crop
        cx_c = (lbs_xyxy[:, 2] + lbs_xyxy[:, 4]) * 0.5
        cy_c = (lbs_xyxy[:, 3] + lbs_xyxy[:, 5]) * 0.5
        inside = ((cx_c >= left) & (cx_c < left + cw) &
                  (cy_c >= top)  & (cy_c < top  + ch))
        if not inside.any():
            return img, lbs_xyxy

        crop = img[top:top + ch, left:left + cw]
        img_out = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

        sx, sy = W / cw, H / ch
        new_lbs = lbs_xyxy[inside].copy()
        new_lbs[:, 2] = np.clip((lbs_xyxy[inside, 2] - left) * sx, 0, W)
        new_lbs[:, 3] = np.clip((lbs_xyxy[inside, 3] - top)  * sy, 0, H)
        new_lbs[:, 4] = np.clip((lbs_xyxy[inside, 4] - left) * sx, 0, W)
        new_lbs[:, 5] = np.clip((lbs_xyxy[inside, 5] - top)  * sy, 0, H)

        return img_out, new_lbs

    # ------------------------------------------------------------------
    # GridMask
    # ------------------------------------------------------------------

    def _gridmask(self, img, p=0.3):
        """Erase a regular grid of squares to simulate partial occlusion.

        Vectorized: builds the mask via numpy tile/slice ops, no Python loop.
        """
        if random.random() > p:
            return img
        H, W = img.shape[:2]
        d  = random.randint(40, 80)
        r  = int(d * random.uniform(0.35, 0.55))
        dx = random.randint(0, d)
        dy = random.randint(0, d)
        # Build one tile (d×d), then tile it to cover (H+d)×(W+d), then crop
        tile = np.ones((d, d), dtype=np.uint8)
        tile[:r, :r] = 0
        reps_h = (H + d) // d + 1
        reps_w = (W + d) // d + 1
        mask = np.tile(tile, (reps_h, reps_w))[dy:dy + H, dx:dx + W]
        return img * mask[:, :, None]

    # ------------------------------------------------------------------
    # __getitem__ — augmentation pipeline
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        img_id         = self.img_ids[idx]
        img, labels    = self._load_raw(img_id)
        info           = self.img_info[img_id]
        orig_h, orig_w = info['height'], info['width']

        do_aug       = self.augment and self.cur_epoch < self.stop_epoch
        mosaic_used  = False
        pad_w = pad_h = 0   # letterbox padding (0 when mosaic used)

        # ── [A] Temporal Mosaic path ──────────────────────────────────────
        # 4 consecutive frames from same sequence → richer positive density,
        # same-scene background consistency, independent HSV per tile.
        if (do_aug and self.use_temporal_mosaic and self._has_seq_idx
                and random.random() < self.temporal_mosaic_prob):
            img, labels = self._temporal_mosaic(img_id)
            # labels now in cxcywh-norm (mosaic space) → convert to xyxy pixel
            if len(labels) > 0:
                lbs = labels.copy()
                lbs[:, 2] = (labels[:, 2] - labels[:, 4] * 0.5) * self.width
                lbs[:, 3] = (labels[:, 3] - labels[:, 5] * 0.5) * self.height
                lbs[:, 4] = (labels[:, 2] + labels[:, 4] * 0.5) * self.width
                lbs[:, 5] = (labels[:, 3] + labels[:, 5] * 0.5) * self.height
            else:
                lbs = np.zeros((0, 6), dtype=np.float32)
            mosaic_used = True

        # ── [A2] Standard Mosaic path (4 random images + random center) ───
        if (not mosaic_used and do_aug and self.use_mosaic
                and random.random() < self.mosaic_prob):
            img, labels = self._mosaic(img_id)
            if len(labels) > 0:
                lbs = labels.copy()
                lbs[:, 2] = (labels[:, 2] - labels[:, 4] * 0.5) * self.width
                lbs[:, 3] = (labels[:, 3] - labels[:, 5] * 0.5) * self.height
                lbs[:, 4] = (labels[:, 2] + labels[:, 4] * 0.5) * self.width
                lbs[:, 5] = (labels[:, 3] + labels[:, 5] * 0.5) * self.height
            else:
                lbs = np.zeros((0, 6), dtype=np.float32)
            mosaic_used = True
        if not mosaic_used:
            # B1: HSV on raw image
            if do_aug:
                augment_hsv(img, fraction=0.50)

            # B2: Plain resize về (W, H) — KHÔNG letterbox/pad (kiểu RT-DETR/DEIM)
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # B3: cxcywh norm → pixel xyxy trong canvas (norm bất biến qua plain resize)
            if len(labels) > 0:
                lbs = labels.copy()
                lbs[:, 2] = (labels[:, 2] - labels[:, 4] * 0.5) * self.width
                lbs[:, 3] = (labels[:, 3] - labels[:, 5] * 0.5) * self.height
                lbs[:, 4] = (labels[:, 2] + labels[:, 4] * 0.5) * self.width
                lbs[:, 5] = (labels[:, 3] + labels[:, 5] * 0.5) * self.height
            else:
                lbs = np.zeros((0, 6), dtype=np.float32)

            # B4: Small Object Zoom — crop toward tiny objects before affine.
            # Increases pixel area of small objects → stronger gradient signal.
            if do_aug and self.use_small_obj_zoom:
                img, lbs = self._small_obj_zoom(
                    img, lbs, p=self.small_obj_zoom_prob)

        # ── [C] Geometric transform (both paths): perspective OR affine ──
        # One transform per sample. Homography simulates a different camera
        # viewpoint (perspective foreshortening); affine handles scale/shift.
        if do_aug:
            if self.use_homography and random.random() < self.homography_prob:
                img, lbs, _ = random_homography_warp(
                    img, lbs, strength=self.homography_strength)
            else:
                img, lbs, _ = random_affine(
                    img, lbs,
                    degrees=(-5, 5),
                    translate=(0.10, 0.10),
                    scale=(0.50, 1.20),
                    shear=(-2, 2),
                )

        # ── [D] pixel xyxy → cxcywh normalized ───────────────────────────
        if len(lbs) > 0:
            out = lbs.copy()
            out[:, 2] = (lbs[:, 2] + lbs[:, 4]) * 0.5 / self.width
            out[:, 3] = (lbs[:, 3] + lbs[:, 5]) * 0.5 / self.height
            out[:, 4] = (lbs[:, 4] - lbs[:, 2])       / self.width
            out[:, 5] = (lbs[:, 5] - lbs[:, 3])       / self.height
            labels = out
        else:
            labels = lbs

        # ── [E] GridMask — simulate partial occlusion ─────────────────────
        if do_aug and self.use_gridmask:
            img = self._gridmask(img, p=self.gridmask_prob)

        # ── [F] Horizontal flip (50%) ─────────────────────────────────────
        if do_aug and random.random() > 0.5:
            img = np.fliplr(img)
            if len(labels) > 0:
                labels[:, 2] = 1.0 - labels[:, 2]

        # ── [G] BGR → RGB, scale to [0, 1] ───────────────────────────────
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # ── Pack DETR-format targets ──────────────────────────────────────
        num_objs       = min(len(labels), self.max_objs)
        detr_boxes     = np.zeros((self.max_objs, 4),  dtype=np.float32)
        detr_labels    = np.full((self.max_objs,), -1, dtype=np.int64)
        detr_track_ids = np.full((self.max_objs,), -1, dtype=np.int64)

        for k in range(num_objs):
            lb = labels[k]
            detr_boxes[k]     = lb[2:6]
            detr_labels[k]    = int(lb[0])
            detr_track_ids[k] = int(lb[1])

        return {
            'input':          img,
            'detr_boxes':     detr_boxes,
            'detr_labels':    detr_labels,
            'detr_track_ids': detr_track_ids,
            'detr_num_objs':  np.array(num_objs, dtype=np.int64),
            'orig_hw':        np.array([orig_h, orig_w], dtype=np.int64),
            'coco_image_id':  np.array(img_id, dtype=np.int64),
            'lb_pad':         np.array([pad_w, pad_h], dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# Inference loader — mirrors val preprocessing of VisDroneCocoDataset exactly:
#   letterbox → BGR→RGB → /255.0  (no augmentation, no normalization)
# Use this in track.py so train/infer preprocessing is identical.
# ---------------------------------------------------------------------------

class LoadImagesForTracking:
    """
    Iterate over image files for tracking inference.

    Preprocessing matches VisDroneCocoDataset val path exactly:
      1. plain resize to (width, height) — no letterbox, no pad
      2. BGR → RGB, /255.0
      3. (3, H, W) float32 numpy array

    Args:
        path     : directory of images or single image file
        img_size : (W, H) — must match training --input-wh
    """

    _IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}

    def __init__(self, path, img_size=(896, 512)):
        self.width  = img_size[0]
        self.height = img_size[1]
        self.count  = 0
        self.frame_rate = 10   # placeholder for compatibility with LoadImages

        if isinstance(path, list):
            self.files = path
        elif os.path.isdir(path):
            all_files = sorted(glob.glob(os.path.join(path, '*.*')))
            self.files = [f for f in all_files
                          if os.path.splitext(f)[1].lower() in self._IMG_EXTS]
        elif os.path.isfile(path):
            self.files = [path]
        else:
            self.files = []

        assert len(self.files) > 0, f'No images found in {path}'
        self.nF = len(self.files)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration

        img_path = self.files[self.count]
        img0 = cv2.imread(img_path)   # BGR uint8
        assert img0 is not None, f'Failed to load {img_path}'

        img = cv2.resize(img0, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))   # (3, H, W)

        return img_path, img, img0

    def __getitem__(self, idx):
        idx      = idx % self.nF
        img_path = self.files[idx]
        img0     = cv2.imread(img_path)
        assert img0 is not None, f'Failed to load {img_path}'
        img = cv2.resize(img0, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return img_path, img, img0

    def __len__(self):
        return self.nF