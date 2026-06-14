"""
test_augment.py — Visualize augmentation strategies on real VisDrone frames.

Usage:
    python tools/test_augment.py \
        --data_root /Users/tinvo0908/Desktop/uav-train/VisDrone2019-7cls \
        --split train \
        --out_dir  /tmp/aug_vis

Outputs one PNG per augmentation strategy showing original vs augmented
with bounding boxes drawn.
"""

import os
import sys
import random
import argparse
import math

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

# ── colour palette per class ─────────────────────────────────────────────────
_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 255,  51),
    ( 51, 255,  51), ( 51, 200, 255), ( 51,  51, 255), (200,  51, 255),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_frame(seq_img_dir, seq_lbl_dir, frame_name):
    """Return (BGR uint8, labels (N,6) [cls,tid,cx,cy,w,h] norm)."""
    img = cv2.imread(os.path.join(seq_img_dir, frame_name))
    lbl_path = os.path.join(seq_lbl_dir,
                            frame_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    labels = np.zeros((0, 6), dtype=np.float32)
    if os.path.isfile(lbl_path):
        raw = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 6)
        if len(raw):
            labels = raw
    return img, labels


def draw_boxes(img, labels, thickness=1):
    """Draw cxcywh-norm boxes on a copy of img."""
    out = img.copy()
    H, W = out.shape[:2]
    for lb in labels:
        cls_id = int(lb[0]) % len(_PALETTE)
        cx, cy, bw, bh = lb[2], lb[3], lb[4], lb[5]
        x1 = int((cx - bw/2) * W);  y1 = int((cy - bh/2) * H)
        x2 = int((cx + bw/2) * W);  y2 = int((cy + bh/2) * H)
        color = _PALETTE[cls_id]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def letterbox(img, height, width, color=(127.5, 127.5, 127.5)):
    shape = img.shape[:2]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1]*ratio), round(shape[0]*ratio))
    dw = (width  - new_shape[0]) * 0.5
    dh = (height - new_shape[1]) * 0.5
    top, bottom = round(dh-0.1), round(dh+0.1)
    left, right = round(dw-0.1), round(dw+0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, left, top


def letterbox_labels(labels, orig_w, orig_h, ratio, pad_w, pad_h, net_w, net_h):
    if len(labels) == 0:
        return labels
    out = labels.copy()
    out[:, 2] = (labels[:, 2] * orig_w * ratio + pad_w) / net_w
    out[:, 3] = (labels[:, 3] * orig_h * ratio + pad_h) / net_h
    out[:, 4] = labels[:, 4] * orig_w * ratio / net_w
    out[:, 5] = labels[:, 5] * orig_h * ratio / net_h
    return out


def random_affine(img, targets, degrees=(-5,5), translate=(.1,.1),
                  scale=(.5,1.2), shear=(-2,2), borderValue=(127.5,127.5,127.5)):
    border = 0
    height, width = img.shape[:2]
    R = np.eye(3)
    a = random.random()*(degrees[1]-degrees[0])+degrees[0]
    s = random.random()*(scale[1]-scale[0])+scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a,center=(width/2,height/2),scale=s)
    T = np.eye(3)
    T[0,2] = (random.random()*2-1)*translate[0]*height+border
    T[1,2] = (random.random()*2-1)*translate[1]*width+border
    S = np.eye(3)
    S[0,1] = math.tan((random.random()*(shear[1]-shear[0])+shear[0])*math.pi/180)
    S[1,0] = math.tan((random.random()*(shear[1]-shear[0])+shear[0])*math.pi/180)
    M = S @ T @ R
    imw = cv2.warpPerspective(img, M, dsize=(width,height),
                              flags=cv2.INTER_LINEAR, borderValue=borderValue)
    if targets is not None and len(targets):
        n = targets.shape[0]
        pts = targets[:,2:6].copy()
        area0 = (pts[:,2]-pts[:,0])*(pts[:,3]-pts[:,1])
        xy = np.ones((n*4,3))
        xy[:,:2] = pts[:,[0,1,2,3,0,3,2,1]].reshape(n*4,2)
        xy = (xy @ M.T)[:,:2].reshape(n,8)
        x = xy[:,[0,2,4,6]]; y = xy[:,[1,3,5,7]]
        xy = np.concatenate((x.min(1),y.min(1),x.max(1),y.max(1))).reshape(4,n).T
        radians = a*math.pi/180
        reduction = max(abs(math.sin(radians)),abs(math.cos(radians)))**0.5
        xc=(xy[:,2]+xy[:,0])/2; yc=(xy[:,3]+xy[:,1])/2
        w=(xy[:,2]-xy[:,0])*reduction; h=(xy[:,3]-xy[:,1])*reduction
        xy=np.concatenate((xc-w/2,yc-h/2,xc+w/2,yc+h/2)).reshape(4,n).T
        np.clip(xy[:,0],0,width,out=xy[:,0]); np.clip(xy[:,2],0,width,out=xy[:,2])
        np.clip(xy[:,1],0,height,out=xy[:,1]); np.clip(xy[:,3],0,height,out=xy[:,3])
        w=xy[:,2]-xy[:,0]; h=xy[:,3]-xy[:,1]; area=w*h
        ar=np.maximum(w/(h+1e-16),h/(w+1e-16))
        keep=(w>4)&(h>4)&(area/(area0+1e-16)>0.1)&(ar<10)
        targets=targets[keep]; targets[:,2:6]=xy[keep]
    return imw, targets, M


def side_by_side(left, right, label_left='Original', label_right='Augmented'):
    H = max(left.shape[0], right.shape[0])
    W = left.shape[1] + right.shape[1] + 10
    out = np.full((H, W, 3), 50, dtype=np.uint8)
    out[:left.shape[0], :left.shape[1]] = left
    out[:right.shape[0], left.shape[1]+10:] = right
    cv2.putText(out, label_left,  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(out, label_right, (left.shape[1]+20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return out


# ── augmentation strategies ───────────────────────────────────────────────────

NET_W, NET_H = 896, 512


def aug_current_pipeline(img, labels):
    """Current AMOT pipeline: HSV + letterbox + affine + hflip."""
    # HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for ch in [1, 2]:
        a = (random.random()*2-1)*0.5+1
        img_hsv[:,:,ch] *= a
        np.clip(img_hsv[:,:,ch], 0, 255, out=img_hsv[:,:,ch])
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    h0, w0 = img.shape[:2]
    img_lb, ratio, pw, ph = letterbox(img, NET_H, NET_W)
    lbs_lb = letterbox_labels(labels, w0, h0, ratio, pw, ph, NET_W, NET_H)
    # Convert to pixel xyxy for affine
    if len(lbs_lb):
        lbs_xy = lbs_lb.copy()
        lbs_xy[:,2] = (lbs_lb[:,2]-lbs_lb[:,4]/2)*NET_W
        lbs_xy[:,3] = (lbs_lb[:,3]-lbs_lb[:,5]/2)*NET_H
        lbs_xy[:,4] = (lbs_lb[:,2]+lbs_lb[:,4]/2)*NET_W
        lbs_xy[:,5] = (lbs_lb[:,3]+lbs_lb[:,5]/2)*NET_H
    else:
        lbs_xy = np.zeros((0,6), dtype=np.float32)
    img_aff, lbs_aff, _ = random_affine(img_lb, lbs_xy)
    if len(lbs_aff):
        out = lbs_aff.copy()
        out[:,2]=(lbs_aff[:,2]+lbs_aff[:,4])*.5/NET_W
        out[:,3]=(lbs_aff[:,3]+lbs_aff[:,5])*.5/NET_H
        out[:,4]=(lbs_aff[:,4]-lbs_aff[:,2])/NET_W
        out[:,5]=(lbs_aff[:,5]-lbs_aff[:,3])/NET_H
        lbs_final = out
    else:
        lbs_final = lbs_aff
    if random.random() > 0.5:
        img_aff = np.fliplr(img_aff)
        if len(lbs_final): lbs_final[:,2] = 1-lbs_final[:,2]
    return img_aff, lbs_final


def aug_temporal_mosaic(frames_labels, net_w=NET_W, net_h=NET_H):
    """4 frames from same sequence → 2×2 mosaic."""
    assert len(frames_labels) >= 4
    mosaic = np.full((net_h, net_w, 3), 114, dtype=np.uint8)
    mid_x, mid_y = net_w//2, net_h//2
    placements = [(0,0,mid_x,mid_y),(mid_x,0,net_w,mid_y),
                  (0,mid_y,mid_x,net_h),(mid_x,mid_y,net_w,net_h)]
    all_labels = []
    random.shuffle(frames_labels)
    for (img, labels), (tx1,ty1,tx2,ty2) in zip(frames_labels[:4], placements):
        tw, th = tx2-tx1, ty2-ty1
        tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        mosaic[ty1:ty2, tx1:tx2] = tile
        if len(labels):
            lbs = labels.copy()
            lbs[:,2] = (labels[:,2]*tw + tx1)/net_w
            lbs[:,3] = (labels[:,3]*th + ty1)/net_h
            lbs[:,4] =  labels[:,4]*tw/net_w
            lbs[:,5] =  labels[:,5]*th/net_h
            all_labels.append(lbs)
    combined = np.concatenate(all_labels, 0) if all_labels else np.zeros((0,6),np.float32)
    return mosaic, combined


def aug_small_object_zoom(img, labels, min_area=0.001, zoom=0.40):
    """Zoom into region containing small objects."""
    H, W = img.shape[:2]
    areas = labels[:,4]*labels[:,5]
    small = labels[areas < min_area]
    if len(small) == 0:
        small = labels  # fallback to all objects
    if len(small) == 0:
        return img, labels
    # Pick a random small object as anchor
    anchor = small[random.randint(0, len(small)-1)]
    cx_px = anchor[2]*W; cy_px = anchor[3]*H
    cw = max(64, int(W*zoom)); ch = max(64, int(H*zoom))
    left = int(np.clip(cx_px - cw*random.uniform(0.2,0.8), 0, W-cw))
    top  = int(np.clip(cy_px - ch*random.uniform(0.2,0.8), 0, H-ch))
    crop = img[top:top+ch, left:left+cw]
    # Keep objects whose center is inside crop
    cx_px_all = labels[:,2]*W; cy_px_all = labels[:,3]*H
    inside = ((cx_px_all>=left)&(cx_px_all<left+cw)&
              (cy_px_all>=top) &(cy_px_all<top+ch))
    if not inside.any():
        return img, labels
    new_lbs = labels[inside].copy()
    new_lbs[:,2] = (labels[inside,2]*W - left)/cw
    new_lbs[:,3] = (labels[inside,3]*H - top)/ch
    new_lbs[:,4] =  labels[inside,4]*W/cw
    new_lbs[:,5] =  labels[inside,5]*H/ch
    crop_resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    return crop_resized, new_lbs


def aug_gridmask(img, labels, ratio=0.4, d_range=(40,80)):
    """GridMask: erase regular grid of squares."""
    H, W = img.shape[:2]
    d = random.randint(*d_range)
    delta_x = random.randint(0, d)
    delta_y = random.randint(0, d)
    mask = np.ones((H, W), dtype=np.float32)
    for y in range(-d+delta_y, H, d):
        for x in range(-d+delta_x, W, d):
            r = int(d * ratio)
            y1 = max(0, y); y2 = min(H, y+r)
            x1 = max(0, x); x2 = min(W, x+r)
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = 0
    out = (img.astype(np.float32) * mask[:,:,None]).astype(np.uint8)
    return out, labels  # boxes unchanged


def aug_temporal_copypaste(img_target, lbs_target, img_src, lbs_src,
                            max_paste=4, max_area=0.005):
    """Copy small objects from src frame, paste into target at offset positions."""
    H, W = img_target.shape[:2]
    areas = lbs_src[:,4]*lbs_src[:,5]
    small_idx = np.where(areas < max_area)[0]
    if len(small_idx) == 0:
        return img_target, lbs_target
    out_img = img_target.copy()
    new_lbs = []
    n_paste = min(max_paste, len(small_idx))
    chosen = random.sample(list(small_idx), n_paste)
    for idx in chosen:
        lb = lbs_src[idx]
        cx,cy,bw,bh = lb[2],lb[3],lb[4],lb[5]
        x1=max(0,int((cx-bw/2)*W)); y1=max(0,int((cy-bh/2)*H))
        x2=min(W,int((cx+bw/2)*W)); y2=min(H,int((cy+bh/2)*H))
        if x2-x1<2 or y2-y1<2: continue
        patch = img_src[y1:y2,x1:x2]
        ph,pw = patch.shape[:2]
        # Paste at random position with slight motion offset
        dx = random.randint(-30,30); dy = random.randint(-20,20)
        nx = int(np.clip(x1+dx, 0, W-pw))
        ny = int(np.clip(y1+dy, 0, H-ph))
        out_img[ny:ny+ph, nx:nx+pw] = patch
        new_lb = lb.copy()
        new_lb[2] = (nx+pw/2)/W; new_lb[3] = (ny+ph/2)/H
        new_lbs.append(new_lb)
    if new_lbs:
        all_lbs = np.concatenate([lbs_target,np.array(new_lbs)], 0)
    else:
        all_lbs = lbs_target
    return out_img, all_lbs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/Users/tinvo0908/Desktop/uav-train/VisDrone2019-7cls')
    ap.add_argument('--split',     default='train')
    ap.add_argument('--out_dir',   default='/tmp/aug_vis')
    ap.add_argument('--seq',       default='', help='specific sequence name (optional)')
    ap.add_argument('--n_frames',  type=int, default=5, help='frames to visualize per aug')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img_root = os.path.join(args.data_root, args.split, 'images')
    lbl_root = os.path.join(args.data_root, args.split, 'labels_with_ids')

    # Pick sequence
    seqs = sorted(os.listdir(img_root))
    seq_name = args.seq if args.seq else random.choice(seqs)
    print(f'[Sequence] {seq_name}')

    seq_img_dir = os.path.join(img_root, seq_name)
    seq_lbl_dir = os.path.join(lbl_root, seq_name)
    all_frames  = sorted(os.listdir(seq_img_dir))

    # Load a window of consecutive frames
    start_idx = random.randint(0, max(0, len(all_frames)-30))
    window = all_frames[start_idx:start_idx+20]

    frames_data = []
    for fname in window:
        img, lbs = load_frame(seq_img_dir, seq_lbl_dir, fname)
        if img is not None:
            frames_data.append((img, lbs, fname))

    print(f'Loaded {len(frames_data)} frames from {seq_name}')
    if len(frames_data) < 4:
        print('[Error] Not enough frames.'); sys.exit(1)

    # ── 1. Original ────────────────────────────────────────────────────────
    print('Visualizing: original')
    for img, lbs, fname in frames_data[:args.n_frames]:
        viz = draw_boxes(img, lbs)
        h0,w0 = img.shape[:2]
        lb_img, ratio, pw, ph = letterbox(img, NET_H, NET_W)
        lbs_lb = letterbox_labels(lbs, w0, h0, ratio, pw, ph, NET_W, NET_H)
        viz_lb = draw_boxes(lb_img, lbs_lb)
        out = side_by_side(viz, viz_lb, 'Original (1904×1071)', f'Letterbox ({NET_W}×{NET_H})')
        cv2.imwrite(os.path.join(args.out_dir, f'1_original_{fname}'), out)

    # ── 2. Current pipeline ────────────────────────────────────────────────
    print('Visualizing: current AMOT pipeline')
    for img, lbs, fname in frames_data[:args.n_frames]:
        h0,w0 = img.shape[:2]
        lb_img,ratio,pw,ph = letterbox(img,NET_H,NET_W)
        lbs_lb = letterbox_labels(lbs,w0,h0,ratio,pw,ph,NET_W,NET_H)
        orig_viz = draw_boxes(lb_img, lbs_lb)
        aug_img, aug_lbs = aug_current_pipeline(img.copy(), lbs.copy())
        aug_viz = draw_boxes(aug_img, aug_lbs)
        out = side_by_side(orig_viz, aug_viz, 'After letterbox', 'After AMOT aug')
        cv2.imwrite(os.path.join(args.out_dir, f'2_amot_aug_{fname}'), out)

    # ── 3. Temporal Mosaic ────────────────────────────────────────────────
    print('Visualizing: temporal mosaic')
    for i in range(args.n_frames):
        # Pick 4 frames spaced out in the sequence
        indices = sorted(random.sample(range(len(frames_data)), min(4, len(frames_data))))
        fl = [(frames_data[j][0].copy(), frames_data[j][1].copy()) for j in indices]
        if len(fl) < 4:
            fl = fl * (4 // len(fl) + 1)
        mos_img, mos_lbs = aug_temporal_mosaic(fl[:4])
        mos_viz = draw_boxes(mos_img, mos_lbs)
        ref_img, ref_lbs, _ = frames_data[indices[0]]
        h0,w0=ref_img.shape[:2]
        lb_img,ratio,pw,ph=letterbox(ref_img,NET_H,NET_W)
        lbs_lb=letterbox_labels(ref_lbs,w0,h0,ratio,pw,ph,NET_W,NET_H)
        ref_viz = draw_boxes(lb_img, lbs_lb)
        fname = f'{i:04d}.jpg'
        labels_info = f'Mosaic frames: {[frames_data[j][2] for j in indices]}'
        out = side_by_side(ref_viz, mos_viz, 'Single frame', 'Temporal Mosaic (4 frames)')
        cv2.imwrite(os.path.join(args.out_dir, f'3_temporal_mosaic_{fname}'), out)

    # ── 4. Small Object Zoom ──────────────────────────────────────────────
    print('Visualizing: small object zoom')
    for img, lbs, fname in frames_data[:args.n_frames]:
        h0,w0=img.shape[:2]
        lb_img,ratio,pw,ph=letterbox(img,NET_H,NET_W)
        lbs_lb=letterbox_labels(lbs,w0,h0,ratio,pw,ph,NET_W,NET_H)
        orig_viz = draw_boxes(lb_img, lbs_lb)
        zoom_img, zoom_lbs = aug_small_object_zoom(lb_img.copy(), lbs_lb.copy())
        zoom_viz = draw_boxes(zoom_img, zoom_lbs)
        # Show object count
        n_small = int((lbs_lb[:,4]*lbs_lb[:,5] < 0.001).sum()) if len(lbs_lb) else 0
        out = side_by_side(orig_viz, zoom_viz,
                           f'Original ({len(lbs_lb)} objs, {n_small} tiny)',
                           f'Small-obj Zoom ({len(zoom_lbs)} objs)')
        cv2.imwrite(os.path.join(args.out_dir, f'4_small_zoom_{fname}'), out)

    # ── 5. GridMask ───────────────────────────────────────────────────────
    print('Visualizing: GridMask')
    for img, lbs, fname in frames_data[:args.n_frames]:
        h0,w0=img.shape[:2]
        lb_img,ratio,pw,ph=letterbox(img,NET_H,NET_W)
        lbs_lb=letterbox_labels(lbs,w0,h0,ratio,pw,ph,NET_W,NET_H)
        orig_viz = draw_boxes(lb_img, lbs_lb)
        gm_img, gm_lbs = aug_gridmask(lb_img.copy(), lbs_lb.copy())
        gm_viz = draw_boxes(gm_img, gm_lbs)
        out = side_by_side(orig_viz, gm_viz, 'Original', 'GridMask (occlusion sim)')
        cv2.imwrite(os.path.join(args.out_dir, f'5_gridmask_{fname}'), out)

    # ── 6. Temporal Copy-Paste ────────────────────────────────────────────
    print('Visualizing: temporal copy-paste')
    for i in range(args.n_frames):
        if i+3 >= len(frames_data): break
        img_t,  lbs_t,  fname_t  = frames_data[i]
        img_src,lbs_src,fname_src= frames_data[i+3]
        h0,w0=img_t.shape[:2]
        lb_t,ratio,pw,ph=letterbox(img_t,NET_H,NET_W)
        lbs_t_lb=letterbox_labels(lbs_t,w0,h0,ratio,pw,ph,NET_W,NET_H)
        lb_src,_,_,_=letterbox(img_src,NET_H,NET_W)
        lbs_src_lb=letterbox_labels(lbs_src,w0,h0,ratio,pw,ph,NET_W,NET_H)
        orig_viz = draw_boxes(lb_t, lbs_t_lb)
        cp_img, cp_lbs = aug_temporal_copypaste(
            lb_t.copy(), lbs_t_lb.copy(), lb_src, lbs_src_lb)
        cp_viz = draw_boxes(cp_img, cp_lbs)
        out = side_by_side(orig_viz, cp_viz,
                           f'Frame T ({fname_t})',
                           f'+ objects from T+3 ({fname_src})')
        cv2.imwrite(os.path.join(args.out_dir, f'6_temporal_copypaste_{fname_t}'), out)

    print(f'\nAll visualizations saved to: {args.out_dir}')
    print('Files:')
    for f in sorted(os.listdir(args.out_dir)):
        print(f'  {f}')


if __name__ == '__main__':
    main()
