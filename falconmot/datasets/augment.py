# """
# Augmentation helpers (AMOT / JDE-style).
# All label ops use (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh unless noted.

# Two augmentation modes are supported:

# AMOT-exact pipeline (--amot_aug):
#   Raw image:
#     1. augment_hsv  — S+V only, fraction=0.50
#   After letterbox:
#     2. random_affine — degrees=(-5,5), translate=0.10, scale=(0.50,1.20), shear=(-2,2)
#     3. horizontal flip (50%)

# Extended pipeline (default):
#   1. random_photometric_distort  (replaces HSV-only)
#   2. random_zoom_out / random_bias_crop / mosaic (optional)
#   [letterbox to network size]
#   3. apply_appearance_augments — photometric + CLAHE + motion_blur
#   4. horizontal + vertical flip
# """

# import math
# import random
# import numpy as np
# import cv2


# # ---------------------------------------------------------------------------
# # Coordinate helpers
# # ---------------------------------------------------------------------------

# def cxcywh_to_xyxy(boxes, width, height):
#     """(N,4) cxcywh normalized → (N,4) xyxy pixel"""
#     x1 = (boxes[:, 0] - boxes[:, 2] / 2) * width
#     y1 = (boxes[:, 1] - boxes[:, 3] / 2) * height
#     x2 = (boxes[:, 0] + boxes[:, 2] / 2) * width
#     y2 = (boxes[:, 1] + boxes[:, 3] / 2) * height
#     return np.stack([x1, y1, x2, y2], axis=1)


# def xyxy_to_cxcywh(boxes, width, height):
#     """(N,4) xyxy pixel → (N,4) cxcywh normalized"""
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
#     w  = (boxes[:, 2] - boxes[:, 0]) / width
#     h  = (boxes[:, 3] - boxes[:, 1]) / height
#     return np.stack([cx, cy, w, h], axis=1)


# def sanitize_boxes(labels, width, height, min_size=2):
#     """Clip boxes to image and drop degenerate ones.
#     labels: (N, 6) [cls, tid, cx, cy, w, h] normalized.
#     """
#     if len(labels) == 0:
#         return labels
#     boxes = cxcywh_to_xyxy(labels[:, 2:6], width, height)
#     np.clip(boxes[:, [0, 2]], 0, width,  out=boxes[:, [0, 2]])
#     np.clip(boxes[:, [1, 3]], 0, height, out=boxes[:, [1, 3]])
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     keep = (w >= min_size) & (h >= min_size)
#     if not keep.any():
#         return np.zeros((0, labels.shape[1]), dtype=labels.dtype)
#     out = labels[keep].copy()
#     out[:, 2:6] = xyxy_to_cxcywh(boxes[keep], width, height)
#     return out


# # ---------------------------------------------------------------------------
# # 1. Photometric distortion (EdgeCrafter: RandomPhotometricDistort, p=0.5)
# # ---------------------------------------------------------------------------

# def random_photometric_distort(img,
#                                 brightness_delta=32,
#                                 contrast_range=(0.5, 1.5),
#                                 saturation_range=(0.5, 1.5),
#                                 hue_delta=18):
#     """Random brightness / contrast / saturation / hue on BGR uint8 image.

#     Matches torchvision RandomPhotometricDistort: each sub-operation applied
#     independently with p=0.5; contrast is randomly applied before or after
#     color-space ops.
#     """
#     img = img.astype(np.float32)

#     # Brightness
#     if random.random() < 0.5:
#         img += random.uniform(-brightness_delta, brightness_delta)

#     # Contrast (randomly placed before or after HSV ops)
#     apply_contrast_first = random.random() < 0.5
#     if apply_contrast_first and random.random() < 0.5:
#         img *= random.uniform(*contrast_range)

#     # Lazy HSV: only convert if at least one of sat/hue will fire
#     apply_saturation = random.random() < 0.5
#     apply_hue        = random.random() < 0.5
#     if apply_saturation or apply_hue:
#         img = np.clip(img, 0, 255).astype(np.uint8)
#         img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
#         if apply_saturation:
#             img_hsv[:, :, 1] *= random.uniform(*saturation_range)
#         if apply_hue:
#             img_hsv[:, :, 0] += random.uniform(-hue_delta, hue_delta)
#             img_hsv[:, :, 0] %= 180.0
#         np.clip(img_hsv[:, :, 1], 0, 255, out=img_hsv[:, :, 1])
#         np.clip(img_hsv[:, :, 2], 0, 255, out=img_hsv[:, :, 2])
#         img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

#     # Contrast (after HSV if not applied first)
#     if not apply_contrast_first and random.random() < 0.5:
#         img *= random.uniform(*contrast_range)

#     return np.clip(img, 0, 255).astype(np.uint8)


# # ---------------------------------------------------------------------------
# # 2. Random zoom out (EdgeCrafter: RandomZoomOut, fill=0)
# # ---------------------------------------------------------------------------

# def random_zoom_out(img, labels, max_scale=2.0, fill_value=0, p=0.5):
#     """Place the image on a larger canvas (zoom out), then adjust labels.

#     labels: (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh.
#     Returns: (img_canvas, labels_adjusted) — labels still normalized to new canvas.
#     """
#     if random.random() > p:
#         return img, labels

#     h, w = img.shape[:2]
#     scale  = random.uniform(1.0, max_scale)
#     new_h  = int(h * scale)
#     new_w  = int(w * scale)

#     canvas = np.full((new_h, new_w, 3), fill_value, dtype=img.dtype)
#     top  = random.randint(0, new_h - h)
#     left = random.randint(0, new_w - w)
#     canvas[top:top + h, left:left + w] = img

#     if len(labels) > 0:
#         out = labels.copy()
#         out[:, 2] = (labels[:, 2] * w + left) / new_w   # cx
#         out[:, 3] = (labels[:, 3] * h + top)  / new_h   # cy
#         out[:, 4] = labels[:, 4] * w / new_w             # bw
#         out[:, 5] = labels[:, 5] * h / new_h             # bh
#         labels = out

#     return canvas, labels


# def random_zoom_in(img, labels, min_scale=0.6, max_scale=0.9, min_ioo=0.3):
#     """Zoom in by cropping a sub-region so small objects appear larger after letterbox.

#     Scale is uniform in [min_scale, max_scale]:
#       0.6 → objects ~1.67× larger   0.9 → objects ~1.11× larger
#     Crop is anchored on a random GT box so at least one object is always inside.
#     Objects are included via IoO ≥ min_ioo.

#     Call this on the raw source image before letterbox.
#     labels: (N, 6) [cls, tid, cx, cy, w, h] normalised cxcywh.
#     """
#     if len(labels) == 0:
#         return img, labels

#     h, w  = img.shape[:2]
#     scale = random.uniform(min_scale, max_scale)
#     crop_h = max(32, int(h * scale))
#     crop_w = max(32, int(w * scale))

#     cx_px = labels[:, 2] * w
#     cy_px = labels[:, 3] * h
#     anchor = random.randint(0, len(labels) - 1)
#     ax, ay = cx_px[anchor], cy_px[anchor]

#     left = int(np.clip(ax - crop_w * random.uniform(0.2, 0.8), 0, w - crop_w))
#     top  = int(np.clip(ay - crop_h * random.uniform(0.2, 0.8), 0, h - crop_h))
#     x2, y2 = left + crop_w, top + crop_h

#     boxes_px  = cxcywh_to_xyxy(labels[:, 2:6], w, h)
#     obj_areas = np.maximum(
#         (boxes_px[:, 2] - boxes_px[:, 0]) * (boxes_px[:, 3] - boxes_px[:, 1]), 1.0
#     )
#     ix1   = np.maximum(boxes_px[:, 0], left)
#     iy1   = np.maximum(boxes_px[:, 1], top)
#     ix2   = np.minimum(boxes_px[:, 2], x2)
#     iy2   = np.minimum(boxes_px[:, 3], y2)
#     inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
#     mask  = (inter / obj_areas) >= min_ioo

#     if not mask.any():
#         return img, labels

#     crop_labels = labels[mask].copy()
#     crop_labels[:, 2] = (crop_labels[:, 2] * w - left) / crop_w
#     crop_labels[:, 3] = (crop_labels[:, 3] * h - top)  / crop_h
#     crop_labels[:, 4] =  crop_labels[:, 4] * w / crop_w
#     crop_labels[:, 5] =  crop_labels[:, 5] * h / crop_h
#     np.clip(crop_labels[:, 2:6], 0.0, 1.0, out=crop_labels[:, 2:6])

#     return img[top:y2, left:x2].copy(), crop_labels


# # ---------------------------------------------------------------------------
# # 3. Random IoU crop (EdgeCrafter: RandomIoUCrop, p=0.8, min_scale=0.3)
# # ---------------------------------------------------------------------------

# def random_bias_crop(img, labels,
#                      min_scale=0.25, max_scale=0.85,
#                      beta_alpha=2.0, beta_beta=5.0,
#                      p=0.5):
#     """Scale-biased random crop for UAV small-object detection.

#     Replaces SSD-style IoU crop which was slow (IoU threshold 0.7/0.9 almost
#     never satisfied → all trials wasted).

#     Scale is sampled from Beta(α, β) mapped to [min_scale, max_scale]:
#       Beta(2, 5)  → mean ≈ 0.29  → effective crop ≈ 25–50% of image area
#       This zooms INTO the scene, making small UAV objects (10px) appear
#       larger (~30-40px) → easier for the model to learn early on.

#     Acceptance: first crop whose centre contains ≥1 GT box is used.
#     No trials loop — crop is placed to guarantee an object is inside,
#     so it always succeeds in O(1).

#     Args:
#         img:        (H, W, 3) BGR uint8 numpy array
#         labels:     (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh
#         min_scale:  minimum crop side as fraction of image side
#         max_scale:  maximum crop side as fraction of image side
#         beta_alpha: Beta distribution α  (< β biases toward smaller scales)
#         beta_beta:  Beta distribution β
#         p:          probability of applying the crop
#     """
#     if random.random() >= p or len(labels) == 0:
#         return img, labels

#     h, w = img.shape[:2]

#     # Sample scale from Beta distribution → biased toward smaller crops
#     raw   = float(np.random.beta(beta_alpha, beta_beta))
#     scale = min_scale + raw * (max_scale - min_scale)

#     crop_h = max(32, int(h * scale))
#     crop_w = max(32, int(w * scale))

#     # Pixel centres of all GT boxes
#     cx_px = (labels[:, 2] * w).astype(np.float32)
#     cy_px = (labels[:, 3] * h).astype(np.float32)

#     # Pick a random GT box as the anchor — crop is placed so its centre
#     # is guaranteed inside the crop window.  No rejection loop needed.
#     anchor_idx = random.randint(0, len(labels) - 1)
#     ax, ay     = cx_px[anchor_idx], cy_px[anchor_idx]

#     # Crop origin: anchor centre must be inside [left, left+crop_w)
#     left_min = max(0, int(ax) - crop_w + 1)
#     left_max = min(w - crop_w, int(ax))
#     left     = random.randint(left_min, max(left_min, left_max))

#     top_min  = max(0, int(ay) - crop_h + 1)
#     top_max  = min(h - crop_h, int(ay))
#     top      = random.randint(top_min, max(top_min, top_max))

#     # Keep all boxes whose centre is inside the crop
#     keep = ((cx_px >= left) & (cx_px < left + crop_w) &
#             (cy_px >= top)  & (cy_px < top  + crop_h))

#     img_crop  = img[top:top + crop_h, left:left + crop_w]

#     boxes_px  = cxcywh_to_xyxy(labels[:, 2:6], w, h)
#     new_px    = boxes_px[keep].copy()
#     new_px[:, [0, 2]] = np.clip(new_px[:, [0, 2]] - left, 0, crop_w)
#     new_px[:, [1, 3]] = np.clip(new_px[:, [1, 3]] - top,  0, crop_h)

#     new_labels         = labels[keep].copy()
#     new_labels[:, 2:6] = xyxy_to_cxcywh(new_px, crop_w, crop_h)

#     return img_crop, new_labels


# # ---------------------------------------------------------------------------
# # Appearance augmentations (image-only, no label changes)
# # ---------------------------------------------------------------------------

# def random_motion_blur(img, kernel_size_range=(3, 7), p=0.20):
#     """Directional motion blur simulating UAV camera pan/tilt/roll.

#     A line kernel at a random angle is convolved with the image.
#     kernel_size_range: (min, max) blur length — odd values only.
#     """
#     if random.random() >= p:
#         return img
#     lo, hi = kernel_size_range
#     size   = random.randrange(lo, hi + 1, 2)
#     angle  = random.uniform(0.0, 180.0)
#     mid    = size // 2
#     rad    = np.deg2rad(angle)
#     cos_a, sin_a = np.cos(rad), np.sin(rad)
#     # Vectorised kernel build — no Python for loop
#     offsets = np.arange(size) - mid
#     xs = np.round(mid + offsets * cos_a).astype(np.int32)
#     ys = np.round(mid + offsets * sin_a).astype(np.int32)
#     valid  = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
#     kernel = np.zeros((size, size), dtype=np.float32)
#     kernel[ys[valid], xs[valid]] = 1.0
#     s = kernel.sum()
#     if s > 0:
#         kernel /= s
#     return cv2.filter2D(img, ddepth=-1, kernel=kernel)


# # ---------------------------------------------------------------------------
# # Appearance augmentation pipeline
# # ---------------------------------------------------------------------------

# def apply_appearance_augments(img):
#     """Post-letterbox appearance augments — lean pipeline tuned for VisDrone.

#     Order and rationale (from dataset analysis):
#       1. photometric_distort  — always; mild color/brightness variation (p=0.5 per sub-op)
#       2. motion_blur          — p=0.20; kernel (3,7)px — UAV shake without destroying features

#     CLAHE removed: as a stochastic train-only op it shifts the train/test
#     contrast distribution; if local-contrast boosting is wanted, apply it
#     deterministically as preprocessing at BOTH train and inference instead.

#     Removed (data-driven):
#       - sensor_noise: std=25 → 88% objects SNR<1, completely masks tiny features
#       - fog: reduces already-low median contrast (10 gray levels) by 10–40%
#       - jpeg_compression: 8×8 blocks destroy 8–16px objects (1–2 blocks per object)
#     """
#     img = random_photometric_distort(img)
#     img = random_motion_blur(img)
#     return img


# # ---------------------------------------------------------------------------
# # Small-object Copy-Paste augmentation
# # ---------------------------------------------------------------------------

# def copy_paste_small_objects(img, labels,
#                               max_area=0.01,
#                               max_paste=5,
#                               p=0.5):
#     """Copy small GT boxes and paste them at random positions in the same image.

#     Selects objects whose normalized area (w*h) < max_area, copies their
#     image patches, pastes at random valid locations, and appends new labels.
#     The original boxes are preserved — only new copies are added.

#     labels: (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh.
#     max_area: normalized area threshold for "small" (default 0.01 = 1% of image).
#     max_paste: max number of objects to paste.
#     """
#     if random.random() >= p or len(labels) == 0:
#         return img, labels

#     h, w = img.shape[:2]
#     areas = labels[:, 4] * labels[:, 5]
#     small_idx = np.where(areas < max_area)[0]
#     if len(small_idx) == 0:
#         return img, labels

#     img_out    = img.copy()
#     new_labels = []
#     n_paste    = min(max_paste, len(small_idx))
#     chosen     = random.sample(list(small_idx), n_paste)

#     for idx in chosen:
#         cx, cy, bw, bh = labels[idx, 2], labels[idx, 3], labels[idx, 4], labels[idx, 5]

#         # Source patch in pixel coords
#         x1 = max(0, int((cx - bw / 2) * w))
#         y1 = max(0, int((cy - bh / 2) * h))
#         x2 = min(w, int((cx + bw / 2) * w))
#         y2 = min(h, int((cy + bh / 2) * h))
#         if x2 - x1 < 2 or y2 - y1 < 2:
#             continue

#         patch = img[y1:y2, x1:x2]
#         ph, pw = patch.shape[:2]

#         # Random destination (constrained inside image)
#         dx = random.randint(0, max(0, w - pw))
#         dy = random.randint(0, max(0, h - ph))

#         img_out[dy:dy + ph, dx:dx + pw] = patch

#         new_lb = labels[idx].copy()
#         new_lb[2] = (dx + pw / 2.0) / w
#         new_lb[3] = (dy + ph / 2.0) / h
#         # w/h are unchanged (same patch size)
#         new_labels.append(new_lb)

#     if new_labels:
#         labels = np.concatenate([labels, np.array(new_labels, dtype=labels.dtype)], axis=0)

#     return img_out, labels


# # ---------------------------------------------------------------------------
# # Mosaic with scale bias (4-image mosaic, some tiles zoomed out)
# # ---------------------------------------------------------------------------

# def mosaic_with_scale_bias(imgs_labels,
#                             output_w, output_h,
#                             scale_bias_prob=0.5,
#                             scale_min=0.3,
#                             scale_max=0.6,
#                             fill_value=114):
#     """Compose 4 images into a 2×2 mosaic; some tiles are zoomed-out (scale biased).

#     imgs_labels : list of 4 (img_bgr, labels_cxcywh_norm) tuples.
#     output_w/h  : final mosaic size (= network input size).
#     scale_bias_prob : probability each tile is zoomed out.
#     scale_min/max   : zoom-out scale range for biased tiles (0.3–0.6 makes objects
#                       appear at 30-60% of their original size → simulates distance).
#     fill_value  : background fill for zoomed-out tiles (default 114 = grey).

#     Returns: (mosaic_bgr: H×W×3 uint8, labels: M×6 normalized cxcywh).
#     """
#     mosaic = np.full((output_h, output_w, 3), fill_value, dtype=np.uint8)
#     mid_x  = output_w // 2
#     mid_y  = output_h // 2

#     # (x1,y1,x2,y2) destination in mosaic for each tile
#     placements = [
#         (0,    0,    mid_x, mid_y),    # top-left
#         (mid_x, 0,   output_w, mid_y), # top-right
#         (0,    mid_y, mid_x, output_h),# bottom-left
#         (mid_x, mid_y, output_w, output_h), # bottom-right
#     ]

#     all_labels = []

#     for (img, labels), (tx1, ty1, tx2, ty2) in zip(imgs_labels, placements):
#         tile_w = tx2 - tx1
#         tile_h = ty2 - ty1
#         orig_h, orig_w = img.shape[:2]

#         if random.random() < scale_bias_prob:
#             # Zoomed-out tile — small canvas with black surround
#             scale   = random.uniform(scale_min, scale_max)
#             # Clamp to tile size so the patch always fits inside the tile
#             small_w = min(tile_w, max(4, int(orig_w * scale)))
#             small_h = min(tile_h, max(4, int(orig_h * scale)))
#             small   = cv2.resize(img, (small_w, small_h),
#                                  interpolation=cv2.INTER_AREA)

#             # Random offset within tile (guaranteed non-negative after clamping)
#             ox = random.randint(0, tile_w - small_w)
#             oy = random.randint(0, tile_h - small_h)

#             tile = np.full((tile_h, tile_w, 3), fill_value, dtype=np.uint8)
#             tile[oy:oy + small_h, ox:ox + small_w] = small

#             # Adjust labels: use small_w/small_h (actual resized dims after clamping),
#             # NOT orig_w*scale — they differ when tile is smaller than orig*scale.
#             if len(labels) > 0:
#                 lbs = labels.copy()
#                 lbs[:, 2] = (labels[:, 2] * small_w + ox) / tile_w
#                 lbs[:, 3] = (labels[:, 3] * small_h + oy) / tile_h
#                 lbs[:, 4] = labels[:, 4] * small_w / tile_w
#                 lbs[:, 5] = labels[:, 5] * small_h / tile_h
#             else:
#                 lbs = labels
#         else:
#             # Normal tile — stretch to tile size
#             tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
#             lbs  = labels.copy() if len(labels) > 0 else labels
#             # labels are already normalized to source image → same normalized value
#             # when resized uniformly, so no coord change needed

#         mosaic[ty1:ty2, tx1:tx2] = tile

#         # Convert tile-normalized → mosaic-normalized
#         if len(lbs) > 0:
#             ml = lbs.copy()
#             ml[:, 2] = (lbs[:, 2] * tile_w + tx1) / output_w
#             ml[:, 3] = (lbs[:, 3] * tile_h + ty1) / output_h
#             ml[:, 4] = lbs[:, 4] * tile_w / output_w
#             ml[:, 5] = lbs[:, 5] * tile_h / output_h
#             all_labels.append(ml)

#     if all_labels:
#         labels_out = np.concatenate(all_labels, axis=0)
#     else:
#         labels_out = np.zeros((0, 6), dtype=np.float32)

#     return mosaic, labels_out


# # ---------------------------------------------------------------------------
# # AMOT-exact random affine (rotation + scale + translate + shear)
# # Ported from AMOT/src/lib/datasets/dataset/jde.py::random_affine
# #
# # Labels format: (N, 6) [cls, tid, x1_px, y1_px, x2_px, y2_px]  ← pixel xyxy
# # (NOT the cxcywh-norm format used elsewhere in this file)
# # The caller must convert to xyxy before passing and convert back after.
# # ---------------------------------------------------------------------------

# def random_affine(img, targets=None,
#                   degrees=(-5, 5),
#                   translate=(0.10, 0.10),
#                   scale=(0.50, 1.20),
#                   shear=(-2, 2),
#                   borderValue=(127.5, 127.5, 127.5)):
#     """Random affine transform matching AMOT's exact implementation.

#     targets : None  OR  (N, 6) float32 [cls, tid, x1, y1, x2, y2] pixel xyxy.
#     Returns (warped_img, filtered_targets, M) when targets is not None,
#             warped_img                       when targets is None.

#     Filtering: boxes with w≤4, h≤4, area_ratio≤0.1, aspect_ratio≥10 are dropped.
#     """
#     border = 0
#     height = img.shape[0]
#     width  = img.shape[1]

#     # Rotation + Scale
#     R = np.eye(3)
#     a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
#     s = random.random() * (scale[1]   - scale[0])   + scale[0]
#     R[:2] = cv2.getRotationMatrix2D(
#         angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

#     # Translation
#     T = np.eye(3)
#     T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border
#     T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border

#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
#     S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

#     M = S @ T @ R  # combined: ORDER MATTERS
#     imw = cv2.warpPerspective(img, M, dsize=(width, height),
#                               flags=cv2.INTER_NEAREST,
#                               borderValue=borderValue)

#     if targets is not None:
#         if len(targets) > 0:
#             n      = targets.shape[0]
#             points = targets[:, 2:6].copy()
#             area0  = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

#             # Warp all 4 corners of each box
#             xy = np.ones((n * 4, 3))
#             xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
#             xy = (xy @ M.T)[:, :2].reshape(n, 8)

#             x  = xy[:, [0, 2, 4, 6]]
#             y  = xy[:, [1, 3, 5, 7]]
#             xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

#             # Reduce box size for large rotations
#             radians  = a * math.pi / 180
#             reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
#             xc = (xy[:, 2] + xy[:, 0]) / 2
#             yc = (xy[:, 3] + xy[:, 1]) / 2
#             w  = (xy[:, 2] - xy[:, 0]) * reduction
#             h  = (xy[:, 3] - xy[:, 1]) * reduction
#             xy = np.concatenate(
#                 (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)).reshape(4, n).T

#             # Clip to image boundary
#             np.clip(xy[:, 0], 0, width,  out=xy[:, 0])
#             np.clip(xy[:, 2], 0, width,  out=xy[:, 2])
#             np.clip(xy[:, 1], 0, height, out=xy[:, 1])
#             np.clip(xy[:, 3], 0, height, out=xy[:, 3])

#             w    = xy[:, 2] - xy[:, 0]
#             h    = xy[:, 3] - xy[:, 1]
#             area = w * h
#             ar   = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
#             keep = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

#             targets        = targets[keep]
#             targets[:, 2:6] = xy[keep]

#         return imw, targets, M
#     else:
#         return imw


# # ---------------------------------------------------------------------------
# # Random perspective (homography) warp — synthetic multi-viewpoint augment
# # ---------------------------------------------------------------------------

# def _camera_homography(w, h, pitch_deg, yaw_deg, roll_deg,
#                        scale=1.0, tx=0.0, ty=0.0, fov_deg=70.0):
#     """Homography induced by a drone camera pose change.

#     Pure camera rotation about the optical centre gives H = K·R·K⁻¹, which is
#     an EXACT image homography for ANY scene (no planar assumption, no parallax)
#     — so pitch/yaw/roll are physically correct. Altitude (scale) and lateral
#     drone motion (tx, ty) are added on top; those are exact only for a planar
#     ground, hence kept small.

#       pitch : tilt fwd/back  → vertical foreshortening (the oblique "drone" look)
#       yaw   : pan left/right → horizontal foreshortening
#       roll  : bank           → in-plane-ish rotation
#     """
#     f = 0.5 * w / np.tan(np.radians(fov_deg) * 0.5)      # focal from horizontal FOV
#     K = np.array([[f, 0, w * 0.5], [0, f, h * 0.5], [0, 0, 1]], dtype=np.float64)
#     Kinv = np.linalg.inv(K)

#     cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
#     cy, sy = np.cos(np.radians(yaw_deg)),   np.sin(np.radians(yaw_deg))
#     cr, sr = np.cos(np.radians(roll_deg)),  np.sin(np.radians(roll_deg))
#     Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])      # pitch
#     Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      # yaw
#     Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])      # roll
#     H_rot = K @ (Rx @ Ry @ Rz) @ Kinv

#     # altitude (scale about centre) + lateral pan
#     S = np.array([[scale, 0, (1 - scale) * w * 0.5 + tx],
#                   [0, scale, (1 - scale) * h * 0.5 + ty],
#                   [0, 0, 1]], dtype=np.float64)
#     return (S @ H_rot).astype(np.float32)


# def _apply_homography(img, targets, H, borderValue, min_box):
#     h, w = img.shape[:2]
#     warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
#                                  borderMode=cv2.BORDER_CONSTANT,
#                                  borderValue=borderValue)
#     if targets is None:
#         return warped
#     if len(targets) == 0:
#         return warped, targets, H

#     x1, y1, x2, y2 = targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]
#     corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
#                         np.stack([x2, y2], 1), np.stack([x1, y2], 1)], axis=1)
#     warped_c = cv2.perspectiveTransform(
#         corners.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 4, 2)
#     nx1 = np.clip(warped_c[:, :, 0].min(1), 0, w)
#     ny1 = np.clip(warped_c[:, :, 1].min(1), 0, h)
#     nx2 = np.clip(warped_c[:, :, 0].max(1), 0, w)
#     ny2 = np.clip(warped_c[:, :, 1].max(1), 0, h)

#     keep = ((nx2 - nx1) > min_box) & ((ny2 - ny1) > min_box)
#     if keep.sum() == 0:
#         return img, targets, np.eye(3, dtype=np.float32)       # warp emptied frame
#     out = targets[keep].copy()
#     out[:, 2], out[:, 3] = nx1[keep], ny1[keep]
#     out[:, 4], out[:, 5] = nx2[keep], ny2[keep]
#     return warped, out, H


# def random_homography_warp(img, targets=None,
#                            strength=0.12,
#                            borderValue=(127.5, 127.5, 127.5),
#                            min_box=4):
#     """Warp a frame as if a drone changed its viewpoint mid-flight.

#     Instead of arbitrary corner jitter, a physically-valid camera pose change
#     is sampled (pitch / yaw / roll + small altitude & lateral motion) and the
#     induced homography H = K·R·K⁻¹ (with scale/translation) is applied. This
#     produces the realistic oblique foreshortening of a tilting/banking drone
#     rather than impossible "twist" distortions.

#     `strength` (≈0.08–0.18) scales the motion envelope:
#         max tilt ≈ strength·110°  (0.12 → ~13°), roll ≈ 0.5×, altitude ±~6%,
#         pan ±~10% of frame. Rotation is exact for any scene; the small
#         scale/pan terms assume a near-planar ground.

#     Same I/O contract as ``random_affine``:
#         targets : None OR (N, 6) [cls, tid, x1, y1, x2, y2] pixel xyxy
#         returns (img, filtered_targets, H) with targets, else warped img.
#     """
#     h, w = img.shape[:2]
#     ang = strength * 110.0
#     pitch = np.random.uniform(-ang, ang)
#     yaw   = np.random.uniform(-ang, ang)
#     roll  = np.random.uniform(-0.5 * ang, 0.5 * ang)
#     scale = 1.0 + np.random.uniform(-0.5, 0.5) * strength      # altitude ±~6%
#     tx    = np.random.uniform(-0.2, 0.2) * strength * w         # lateral pan
#     ty    = np.random.uniform(-0.2, 0.2) * strength * h
#     H = _camera_homography(w, h, pitch, yaw, roll, scale, tx, ty)
#     return _apply_homography(img, targets, H, borderValue, min_box)


# # ---------------------------------------------------------------------------
# # Legacy: HSV-only augmentation (kept for reference, replaced by
# # random_photometric_distort in the main pipeline)
# # ---------------------------------------------------------------------------

# def augment_hsv(img, fraction=0.5):
#     """Random S/V scaling on BGR uint8 image (in-place).
#     Kept for backward compatibility — prefer random_photometric_distort.
#     """
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     S = img_hsv[:, :, 1].astype(np.float32)
#     V = img_hsv[:, :, 2].astype(np.float32)

#     a = (random.random() * 2 - 1) * fraction + 1
#     S *= a
#     if a > 1:
#         np.clip(S, a_min=0, a_max=255, out=S)

#     a = (random.random() * 2 - 1) * fraction + 1
#     V *= a
#     if a > 1:
#         np.clip(V, a_min=0, a_max=255, out=V)

#     img_hsv[:, :, 1] = S.astype(np.uint8)
#     img_hsv[:, :, 2] = V.astype(np.uint8)
#     cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
#     return img







"""
Augmentation helpers (AMOT / JDE-style).
All label ops use (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh unless noted.
"""

import math
import random
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes, width, height):
    """(N,4) cxcywh normalized → (N,4) xyxy pixel"""
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * width
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * height
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * width
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return np.stack([x1, y1, x2, y2], axis=1)

def xyxy_to_cxcywh(boxes, width, height):
    """(N,4) xyxy pixel → (N,4) cxcywh normalized"""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    w  = (boxes[:, 2] - boxes[:, 0]) / width
    h  = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([cx, cy, w, h], axis=1)

def sanitize_boxes(labels, width, height, min_size=2):
    """Clip boxes to image and drop degenerate ones."""
    if len(labels) == 0:
        return labels
    boxes = cxcywh_to_xyxy(labels[:, 2:6], width, height)
    np.clip(boxes[:, [0, 2]], 0, width,  out=boxes[:, [0, 2]])
    np.clip(boxes[:, [1, 3]], 0, height, out=boxes[:, [1, 3]])
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    if not keep.any():
        return np.zeros((0, labels.shape[1]), dtype=labels.dtype)
    out = labels[keep].copy()
    out[:, 2:6] = xyxy_to_cxcywh(boxes[keep], width, height)
    return out

# ---------------------------------------------------------------------------
# 1. Photometric distortion
# ---------------------------------------------------------------------------

def random_photometric_distort(img, brightness_delta=32, contrast_range=(0.5, 1.5),
                               saturation_range=(0.5, 1.5), hue_delta=18):
    img = img.astype(np.float32)

    if random.random() < 0.5:
        img += random.uniform(-brightness_delta, brightness_delta)

    apply_contrast_first = random.random() < 0.5
    if apply_contrast_first and random.random() < 0.5:
        img *= random.uniform(*contrast_range)

    apply_saturation = random.random() < 0.5
    apply_hue        = random.random() < 0.5
    if apply_saturation or apply_hue:
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        if apply_saturation:
            img_hsv[:, :, 1] *= random.uniform(*saturation_range)
        if apply_hue:
            img_hsv[:, :, 0] += random.uniform(-hue_delta, hue_delta)
            img_hsv[:, :, 0] %= 180.0
        np.clip(img_hsv[:, :, 1], 0, 255, out=img_hsv[:, :, 1])
        np.clip(img_hsv[:, :, 2], 0, 255, out=img_hsv[:, :, 2])
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if not apply_contrast_first and random.random() < 0.5:
        img *= random.uniform(*contrast_range)

    return np.clip(img, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# 2. Random zoom out & Crop
# ---------------------------------------------------------------------------

def random_zoom_out(img, labels, max_scale=2.0, fill_value=0, p=0.5):
    """Place the image on a larger canvas (zoom out), simulating altitude gain."""
    if random.random() > p:
        return img, labels

    h, w = img.shape[:2]
    scale  = random.uniform(1.0, max_scale)
    new_h  = int(h * scale)
    new_w  = int(w * scale)

    # Use fill_value (0 = black)
    canvas = np.full((new_h, new_w, 3), fill_value, dtype=img.dtype)
    top  = random.randint(0, new_h - h)
    left = random.randint(0, new_w - w)
    canvas[top:top + h, left:left + w] = img

    if len(labels) > 0:
        out = labels.copy()
        out[:, 2] = (labels[:, 2] * w + left) / new_w   # cx
        out[:, 3] = (labels[:, 3] * h + top)  / new_h   # cy
        out[:, 4] = labels[:, 4] * w / new_w             # bw
        out[:, 5] = labels[:, 5] * h / new_h             # bh
        labels = out

    return canvas, labels


def random_crop(img, labels, scale_range=(0.6, 1.0), p=0.5):
    """Random crop for object detection."""
    if random.random() > p or len(labels) == 0:
        return img, labels

    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    crop_h = max(32, int(h * scale))
    crop_w = max(32, int(w * scale))

    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    img_crop = img[top:top+crop_h, left:left+crop_w]

    boxes_px = cxcywh_to_xyxy(labels[:, 2:6], w, h)
    boxes_px[:, [0, 2]] -= left
    boxes_px[:, [1, 3]] -= top
    
    cx = (boxes_px[:, 0] + boxes_px[:, 2]) / 2.0
    cy = (boxes_px[:, 1] + boxes_px[:, 3]) / 2.0
    
    np.clip(boxes_px[:, [0, 2]], 0, crop_w, out=boxes_px[:, [0, 2]])
    np.clip(boxes_px[:, [1, 3]], 0, crop_h, out=boxes_px[:, [1, 3]])
    
    bw = boxes_px[:, 2] - boxes_px[:, 0]
    bh = boxes_px[:, 3] - boxes_px[:, 1]
    keep = (cx >= 0) & (cx < crop_w) & (cy >= 0) & (cy < crop_h) & (bw > 2) & (bh > 2)
    
    new_labels = labels[keep].copy()
    if len(new_labels) > 0:
        new_labels[:, 2:6] = xyxy_to_cxcywh(boxes_px[keep], crop_w, crop_h)
        
    return img_crop, new_labels

# ---------------------------------------------------------------------------
# Appearance augmentations
# ---------------------------------------------------------------------------

def random_motion_blur(img, kernel_size_range=(3, 7), p=0.20):
    if random.random() >= p:
        return img
    lo, hi = kernel_size_range
    size   = random.randrange(lo, hi + 1, 2)
    angle  = random.uniform(0.0, 180.0)
    mid    = size // 2
    rad    = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    offsets = np.arange(size) - mid
    xs = np.round(mid + offsets * cos_a).astype(np.int32)
    ys = np.round(mid + offsets * sin_a).astype(np.int32)
    valid  = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[ys[valid], xs[valid]] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)

def apply_appearance_augments(img):
    img = random_photometric_distort(img)
    img = random_motion_blur(img)
    return img

# ---------------------------------------------------------------------------
# Copy-Paste & Mosaic
# ---------------------------------------------------------------------------

def copy_paste_small_objects(img, labels, max_area=0.01, max_paste=5, p=0.5):
    if random.random() >= p or len(labels) == 0:
        return img, labels

    h, w = img.shape[:2]
    areas = labels[:, 4] * labels[:, 5]
    small_idx = np.where(areas < max_area)[0]
    if len(small_idx) == 0:
        return img, labels

    img_out    = img.copy()
    new_labels = []
    n_paste    = min(max_paste, len(small_idx))
    chosen     = random.sample(list(small_idx), n_paste)

    for idx in chosen:
        cx, cy, bw, bh = labels[idx, 2], labels[idx, 3], labels[idx, 4], labels[idx, 5]

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        patch = img[y1:y2, x1:x2]
        ph, pw = patch.shape[:2]

        dx = random.randint(0, max(0, w - pw))
        dy = random.randint(0, max(0, h - ph))

        img_out[dy:dy + ph, dx:dx + pw] = patch

        new_lb = labels[idx].copy()
        new_lb[2] = (dx + pw / 2.0) / w
        new_lb[3] = (dy + ph / 2.0) / h
        new_labels.append(new_lb)

    if new_labels:
        labels = np.concatenate([labels, np.array(new_labels, dtype=labels.dtype)], axis=0)

    return img_out, labels

def mosaic_with_scale_bias(imgs_labels, output_w, output_h, scale_bias_prob=0.5,
                           scale_min=0.3, scale_max=0.6, fill_value=0):
    mosaic = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    mid_x  = output_w // 2
    mid_y  = output_h // 2

    placements = [
        (0,    0,    mid_x, mid_y),
        (mid_x, 0,   output_w, mid_y),
        (0,    mid_y, mid_x, output_h),
        (mid_x, mid_y, output_w, output_h),
    ]

    all_labels = []

    for (img, labels), (tx1, ty1, tx2, ty2) in zip(imgs_labels, placements):
        tile_w = tx2 - tx1
        tile_h = ty2 - ty1
        orig_h, orig_w = img.shape[:2]

        if random.random() < scale_bias_prob:
            scale   = random.uniform(scale_min, scale_max)
            small_w = min(tile_w, max(4, int(orig_w * scale)))
            small_h = min(tile_h, max(4, int(orig_h * scale)))
            small   = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

            ox = random.randint(0, tile_w - small_w)
            oy = random.randint(0, tile_h - small_h)

            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            tile[oy:oy + small_h, ox:ox + small_w] = small

            if len(labels) > 0:
                lbs = labels.copy()
                lbs[:, 2] = (labels[:, 2] * small_w + ox) / tile_w
                lbs[:, 3] = (labels[:, 3] * small_h + oy) / tile_h
                lbs[:, 4] = labels[:, 4] * small_w / tile_w
                lbs[:, 5] = labels[:, 5] * small_h / tile_h
            else:
                lbs = labels
        else:
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            lbs  = labels.copy() if len(labels) > 0 else labels

        mosaic[ty1:ty2, tx1:tx2] = tile

        if len(lbs) > 0:
            ml = lbs.copy()
            ml[:, 2] = (lbs[:, 2] * tile_w + tx1) / output_w
            ml[:, 3] = (lbs[:, 3] * tile_h + ty1) / output_h
            ml[:, 4] = lbs[:, 4] * tile_w / output_w
            ml[:, 5] = lbs[:, 5] * tile_h / output_h
            all_labels.append(ml)

    if all_labels:
        labels_out = np.concatenate(all_labels, axis=0)
    else:
        labels_out = np.zeros((0, 6), dtype=np.float32)

    return mosaic, labels_out

# ---------------------------------------------------------------------------
# AMOT-exact random affine
# ---------------------------------------------------------------------------

def random_affine(img, targets=None, degrees=(-5, 5), translate=(0.10, 0.10),
                  scale=(0.50, 1.20), shear=(-2, 2), borderValue=(0, 0, 0)):
    border = 0
    height = img.shape[0]
    width  = img.shape[1]

    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1]   - scale[0])   + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * height + border
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * width + border

    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

    M = S @ T @ R  
    imw = cv2.warpPerspective(img, M, dsize=(width, height),
                              flags=cv2.INTER_NEAREST,
                              borderValue=borderValue)

    if targets is not None:
        if len(targets) > 0:
            n      = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0  = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            x  = xy[:, [0, 2, 4, 6]]
            y  = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            radians  = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            xc = (xy[:, 2] + xy[:, 0]) / 2
            yc = (xy[:, 3] + xy[:, 1]) / 2
            w  = (xy[:, 2] - xy[:, 0]) * reduction
            h  = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)).reshape(4, n).T

            np.clip(xy[:, 0], 0, width,  out=xy[:, 0])
            np.clip(xy[:, 2], 0, width,  out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])

            w    = xy[:, 2] - xy[:, 0]
            h    = xy[:, 3] - xy[:, 1]
            area = w * h
            ar   = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            keep = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets        = targets[keep]
            targets[:, 2:6] = xy[keep]

        return imw, targets, M
    else:
        return imw

# ---------------------------------------------------------------------------
# Random perspective warp
# ---------------------------------------------------------------------------

def _camera_homography(w, h, pitch_deg, yaw_deg, roll_deg, scale=1.0, tx=0.0, ty=0.0, fov_deg=70.0):
    f = 0.5 * w / np.tan(np.radians(fov_deg) * 0.5)
    K = np.array([[f, 0, w * 0.5], [0, f, h * 0.5], [0, 0, 1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    cy, sy = np.cos(np.radians(yaw_deg)),   np.sin(np.radians(yaw_deg))
    cr, sr = np.cos(np.radians(roll_deg)),  np.sin(np.radians(roll_deg))
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])      
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])      
    H_rot = K @ (Rx @ Ry @ Rz) @ Kinv

    S = np.array([[scale, 0, (1 - scale) * w * 0.5 + tx],
                  [0, scale, (1 - scale) * h * 0.5 + ty],
                  [0, 0, 1]], dtype=np.float64)
    return (S @ H_rot).astype(np.float32)

def _apply_homography(img, targets, H, borderValue, min_box):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=borderValue)
    if targets is None:
        return warped
    if len(targets) == 0:
        return warped, targets, H

    x1, y1, x2, y2 = targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1), np.stack([x1, y2], 1)], axis=1)
    warped_c = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 4, 2)
    nx1 = np.clip(warped_c[:, :, 0].min(1), 0, w)
    ny1 = np.clip(warped_c[:, :, 1].min(1), 0, h)
    nx2 = np.clip(warped_c[:, :, 0].max(1), 0, w)
    ny2 = np.clip(warped_c[:, :, 1].max(1), 0, h)

    keep = ((nx2 - nx1) > min_box) & ((ny2 - ny1) > min_box)
    if keep.sum() == 0:
        return img, targets, np.eye(3, dtype=np.float32)       
    out = targets[keep].copy()
    out[:, 2], out[:, 3] = nx1[keep], ny1[keep]
    out[:, 4], out[:, 5] = nx2[keep], ny2[keep]
    return warped, out, H

def random_homography_warp(img, targets=None, strength=0.12, borderValue=(0, 0, 0), min_box=4):
    h, w = img.shape[:2]
    ang = strength * 110.0
    pitch = np.random.uniform(-ang, ang)
    yaw   = np.random.uniform(-ang, ang)
    roll  = np.random.uniform(-0.5 * ang, 0.5 * ang)
    scale = 1.0 + np.random.uniform(-0.5, 0.5) * strength      
    tx    = np.random.uniform(-0.2, 0.2) * strength * w         
    ty    = np.random.uniform(-0.2, 0.2) * strength * h
    H = _camera_homography(w, h, pitch, yaw, roll, scale, tx, ty)
    return _apply_homography(img, targets, H, borderValue, min_box)

# ---------------------------------------------------------------------------
# Legacy: HSV-only
# ---------------------------------------------------------------------------

def augment_hsv(img, fraction=0.5):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img