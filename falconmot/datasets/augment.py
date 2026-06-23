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


def _fill_region(img, x0, y0, ew, eh, mode='random'):
    """Tô đè vùng [y0:y0+eh, x0:x0+ew] của ảnh BGR uint8 theo `mode`.
 
    mode:
      'random' : nhiễu uniform 0..255 (Random-Erasing chuẩn).
      'mean'   : xám 114 (giá trị letterbox của repo).
      'patch'  : copy 1 vùng ngẫu nhiên KHÁC trong cùng ảnh -> occluder giống
                 nội dung scene thật (tự nhiên hơn nhiễu, rất hợp drone-view).
    """
    h, w = img.shape[:2]
    x0 = int(max(0, min(x0, w - 1)))
    y0 = int(max(0, min(y0, h - 1)))
    ew = int(max(1, min(ew, w - x0)))
    eh = int(max(1, min(eh, h - y0)))
    c = img.shape[2]
    if mode == 'mean':
        img[y0:y0 + eh, x0:x0 + ew] = 114
    elif mode == 'patch':
        sx = random.randint(0, max(0, w - ew))
        sy = random.randint(0, max(0, h - eh))
        img[y0:y0 + eh, x0:x0 + ew] = img[sy:sy + eh, sx:sx + ew]
    else:  # 'random'
        img[y0:y0 + eh, x0:x0 + ew] = np.random.randint(0, 256, (eh, ew, c), dtype=np.uint8)
    return img
 
 
def object_aware_occlusion(img, labels, p=0.5, occ_obj_frac=0.3,
                           frac_lo=0.20, frac_hi=0.45, mode='patch',
                           min_obj_px=12):
    """Che MỘT PHẦN (dải cạnh) một số object để mô phỏng occlusion cho ReID/MOT.
 
    Dùng DẢI CẠNH (top/bottom/left/right) với frac < 0.5 nên TÂM box không bao giờ
    bị che -> an toàn cho dense ReID (sample tại tâm GT vẫn trúng phần thấy được).
 
    img    : BGR uint8 (H, W, 3), kích thước mạng.
    labels : [N,6] = [cls, tid, cx, cy, w, h] normalized [0,1]. KHÔNG bị sửa.
    p              : xác suất áp dụng cho cả ảnh.
    occ_obj_frac   : tỉ lệ object bị che mỗi ảnh.
    frac_lo/hi     : phần kích thước box bị che theo 1 chiều (ép < 0.5).
    mode           : 'patch' | 'random' | 'mean' (xem _fill_region).
    min_obj_px     : bỏ qua object có cạnh nhỏ hơn ngưỡng (che là mất hẳn).
    """
    if random.random() >= p or labels is None or len(labels) == 0:
        return img, labels
    h, w = img.shape[:2]
    n = len(labels)
    k = max(1, int(round(n * occ_obj_frac)))
    sel = np.random.choice(n, size=min(k, n), replace=False)
    frac_hi = min(frac_hi, 0.49)   # đảm bảo tâm box luôn nhìn thấy
 
    for i in sel:
        bw = float(labels[i, 4]) * w
        bh = float(labels[i, 5]) * h
        if min(bw, bh) < min_obj_px:
            continue
        x1 = (float(labels[i, 2]) - float(labels[i, 4]) * 0.5) * w
        y1 = (float(labels[i, 3]) - float(labels[i, 5]) * 0.5) * h
        frac = random.uniform(frac_lo, frac_hi)
        side = random.randint(0, 3)
        if side == 0:      # dải trên
            ew, eh, ox, oy = bw, bh * frac, x1, y1
        elif side == 1:    # dải dưới
            ew, eh = bw, bh * frac
            ox, oy = x1, y1 + bh - eh
        elif side == 2:    # dải trái
            ew, eh, ox, oy = bw * frac, bh, x1, y1
        else:              # dải phải
            ew, eh = bw * frac, bh
            ox, oy = x1 + bw - ew, y1
        _fill_region(img, ox, oy, ew, eh, mode)
    return img, labels
 
 
def random_erasing(img, labels=None, p=0.3, sl=0.02, sh=0.20, r1=0.3,
                   mode='random', max_attempts=10):
    """Random Erasing mức ẢNH (Zhong et al. 2020). Nhãn KHÔNG đổi.
 
    sl,sh : khoảng diện tích vùng xoá / diện tích ảnh.
    r1    : aspect ratio trong [r1, 1/r1].
    """
    if random.random() >= p:
        return img, labels
    h, w = img.shape[:2]
    area = h * w
    for _ in range(max_attempts):
        te = random.uniform(sl, sh) * area
        ar = random.uniform(r1, 1.0 / r1)
        eh = int(round(math.sqrt(te * ar)))
        ew = int(round(math.sqrt(te / ar)))
        if 0 < ew < w and 0 < eh < h:
            x0 = random.randint(0, w - ew)
            y0 = random.randint(0, h - eh)
            _fill_region(img, x0, y0, ew, eh, mode)
            break
    return img, labels
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