"""
Convert a VisDrone-DET split (per-image .txt annotations) to a COCO json.

VisDrone-DET layout (as uploaded):
    VisDrone2019-DET-train/
        images/        0000001_xxxx.jpg ...
        annotations/   0000001_xxxx.txt ...   (one txt per image)

Each annotation line:
    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
    category: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
              6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
    score=0 marks an ignored region (dropped here).

Output COCO json is consumed by VisDroneCocoDataset:
    category_id ∈ 1..10  (dataset does category_id-1 → 0-indexed class)
    each annotation carries track_id=0  (DET has no identities; ReID is off
    in --train_single_det mode anyway).

Usage:
    python scripts/visdrone_det_to_coco.py \
        --images_dir /data/VisDrone2019-DET-train/images \
        --ann_dir    /data/VisDrone2019-DET-train/annotations \
        --out        /data/VisDrone2019-DET-train/instances_train.json
"""

import os
import json
import argparse
import glob

VISDRONE_DET_CATEGORIES = [
    (1, 'pedestrian'), (2, 'people'), (3, 'bicycle'), (4, 'car'), (5, 'van'),
    (6, 'truck'), (7, 'tricycle'), (8, 'awning-tricycle'), (9, 'bus'), (10, 'motor'),
]
_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
def _image_size(path):
    """Return (W, H) without decoding the whole image when possible."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size                      # (W, H)
    except Exception:
        import cv2
        im = cv2.imread(path)
        if im is None:
            return None
        h, w = im.shape[:2]
        return (w, h)


def convert(images_dir, ann_dir, out_path, out_images_dir=None, blackout=True):
    import cv2
    img_paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, '*'))
        if p.lower().endswith(_IMG_EXTS)
    )
    if not img_paths:
        raise SystemExit(f'[ERROR] no images found in {images_dir}')

    if blackout:
        if not out_images_dir:
            out_images_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), 'images')
        os.makedirs(out_images_dir, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    n_box = n_skip_cat = n_skip_score = n_no_txt = n_ignore = 0

    for img_id, img_path in enumerate(img_paths, start=1):
        fname = os.path.basename(img_path)
        txt = os.path.join(ann_dir, os.path.splitext(fname)[0] + '.txt')

        # ── parse annotations: split valid boxes vs ignore boxes ──────────
        valid, ignore = [], []
        if os.path.isfile(txt):
            with open(txt) as f:
                for line in f:
                    line = line.strip().rstrip(',')
                    if not line:
                        continue
                    parts = line.replace(' ', '').split(',')
                    if len(parts) < 6:
                        continue
                    try:
                        x, y, bw, bh = (float(parts[0]), float(parts[1]),
                                        float(parts[2]), float(parts[3]))
                        score = float(parts[4]); cat = int(float(parts[5]))
                    except ValueError:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    # ignored region (cat 0/11 or score 0) → blackout, not a label
                    if cat < 1 or cat > 10 or score == 0:
                        ignore.append((x, y, bw, bh))
                        if cat < 1 or cat > 10:
                            n_skip_cat += 1
                        else:
                            n_skip_score += 1
                        continue
                    valid.append((x, y, bw, bh, cat))
        else:
            n_no_txt += 1

        # ── read image; black out ignore regions (matches MOT converter) ──
        if blackout:
            img = cv2.imread(img_path)
            if img is None:
                print(f'  [warn] cannot read {fname}, skipped'); continue
            H, W = img.shape[:2]
            for (x, y, bw, bh) in ignore:
                xi, yi = int(max(0, x)), int(max(0, y))
                img[yi:yi + int(bh), xi:xi + int(bw)] = 0
                n_ignore += 1
            cv2.imwrite(os.path.join(out_images_dir, fname), img)
        else:
            size = _image_size(img_path)
            if size is None:
                print(f'  [warn] cannot read {fname}, skipped'); continue
            W, H = size
            n_ignore += len(ignore)

        images.append({'id': img_id, 'file_name': fname, 'width': W, 'height': H})

        for (x, y, bw, bh, cat) in valid:
            x = max(0.0, min(x, W - 1)); y = max(0.0, min(y, H - 1))
            bw = min(bw, W - x);          bh = min(bh, H - y)
            if bw <= 1 or bh <= 1:
                continue
            annotations.append({
                'id': ann_id, 'image_id': img_id, 'category_id': cat,
                'bbox': [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                'area': round(bw * bh, 2), 'iscrowd': 0,
                'track_id': 0,
            })
            ann_id += 1; n_box += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': cid, 'name': name, 'supercategory': 'object'}
                       for cid, name in VISDRONE_DET_CATEGORIES],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f)

    print(f'[done] {len(images)} images, {n_box} boxes → {out_path}')
    if blackout:
        print(f'       blacked-out {n_ignore} ignore regions → cleaned images in {out_images_dir}')
        print(f'       (point --data_cfg train_img to this cleaned-images dir)')
    print(f'       skipped: cat(ignored/others)={n_skip_cat}  score0={n_skip_score}  '
          f'images_without_txt={n_no_txt}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True)
    p.add_argument('--ann_dir',    required=True)
    p.add_argument('--out',        required=True, help='output COCO json path')
    p.add_argument('--out_images_dir', default='',
                   help='where to write cleaned images (default: <out_dir>/images)')
    p.add_argument('--no_blackout', action='store_true',
                   help='do NOT black out ignore regions (json only, images untouched). '
                        'Default blacks them out to match the MOT converter.')
    a = p.parse_args()
    convert(a.images_dir, a.ann_dir, a.out,
            out_images_dir=(a.out_images_dir or None),
            blackout=not a.no_blackout)


if __name__ == '__main__':
    main()
