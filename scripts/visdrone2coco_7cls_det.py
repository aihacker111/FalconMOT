# """
# Convert a VisDrone-DET split (per-image .txt annotations) to a COCO json.
# With support for merging specific classes:
#     pedestrian + people -> pedestrian
#     van + truck -> truck
#     tricycle + awning-tricycle -> tricycle
# """

# import os
# import json
# import argparse
# import glob

# # Danh sách class gốc (10 classes)
# VISDRONE_DET_CATEGORIES = [
#     (1, 'pedestrian'), (2, 'people'), (3, 'bicycle'), (4, 'car'), (5, 'van'),
#     (6, 'truck'), (7, 'tricycle'), (8, 'awning-tricycle'), (9, 'bus'), (10, 'motor'),
# ]

# # Danh sách class sau khi merge (7 classes)
# MERGED_CATEGORIES = [
#     (1, 'pedestrian'), (2, 'bicycle'), (3, 'car'), (4, 'truck'),
#     (5, 'tricycle'), (6, 'bus'), (7, 'motor')
# ]

# # Từ điển ánh xạ id cũ sang id mới
# CLASS_MAPPING = {
#     1: 1,   # pedestrian -> pedestrian (1)
#     2: 1,   # people -> pedestrian (1)
#     3: 2,   # bicycle -> bicycle (2)
#     4: 3,   # car -> car (3)
#     5: 4,   # van -> truck (4)
#     6: 4,   # truck -> truck (4)
#     7: 5,   # tricycle -> tricycle (5)
#     8: 5,   # awning-tricycle -> tricycle (5)
#     9: 6,   # bus -> bus (6)
#     10: 7   # motor -> motor (7)
# }

# _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# def _image_size(path):
#     """Return (W, H) without decoding the whole image when possible."""
#     try:
#         from PIL import Image
#         with Image.open(path) as im:
#             return im.size                      # (W, H)
#     except Exception:
#         import cv2
#         im = cv2.imread(path)
#         if im is None:
#             return None
#         h, w = im.shape[:2]
#         return (w, h)


# def convert(images_dir, ann_dir, out_path, out_images_dir=None, blackout=True, merge=False):
#     import cv2
#     img_paths = sorted(
#         p for p in glob.glob(os.path.join(images_dir, '*'))
#         if p.lower().endswith(_IMG_EXTS)
#     )
#     if not img_paths:
#         raise SystemExit(f'[ERROR] no images found in {images_dir}')

#     if blackout:
#         if not out_images_dir:
#             out_images_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), 'images')
#         os.makedirs(out_images_dir, exist_ok=True)

#     images, annotations = [], []
#     ann_id = 1
#     n_box = n_skip_cat = n_skip_score = n_no_txt = n_ignore = 0

#     for img_id, img_path in enumerate(img_paths, start=1):
#         fname = os.path.basename(img_path)
#         txt = os.path.join(ann_dir, os.path.splitext(fname)[0] + '.txt')

#         # ── parse annotations: split valid boxes vs ignore boxes ──────────
#         valid, ignore = [], []
#         if os.path.isfile(txt):
#             with open(txt) as f:
#                 for line in f:
#                     line = line.strip().rstrip(',')
#                     if not line:
#                         continue
#                     parts = line.replace(' ', '').split(',')
#                     if len(parts) < 6:
#                         continue
#                     try:
#                         x, y, bw, bh = (float(parts[0]), float(parts[1]),
#                                         float(parts[2]), float(parts[3]))
#                         score = float(parts[4]); cat = int(float(parts[5]))
#                     except ValueError:
#                         continue
#                     if bw <= 0 or bh <= 0:
#                         continue
                    
#                     # ignored region (cat 0/11 or score 0) → blackout, not a label
#                     if cat < 1 or cat > 10 or score == 0:
#                         ignore.append((x, y, bw, bh))
#                         if cat < 1 or cat > 10:
#                             n_skip_cat += 1
#                         else:
#                             n_skip_score += 1
#                         continue
                    
#                     # ── ÁP DỤNG HÀM MERGE Ở ĐÂY ──
#                     if merge:
#                         cat = CLASS_MAPPING[cat]
                        
#                     valid.append((x, y, bw, bh, cat))
#         else:
#             n_no_txt += 1

#         # ── read image; black out ignore regions (matches MOT converter) ──
#         if blackout:
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f'  [warn] cannot read {fname}, skipped'); continue
#             H, W = img.shape[:2]
#             for (x, y, bw, bh) in ignore:
#                 xi, yi = int(max(0, x)), int(max(0, y))
#                 img[yi:yi + int(bh), xi:xi + int(bw)] = 0
#                 n_ignore += 1
#             cv2.imwrite(os.path.join(out_images_dir, fname), img)
#         else:
#             size = _image_size(img_path)
#             if size is None:
#                 print(f'  [warn] cannot read {fname}, skipped'); continue
#             W, H = size
#             n_ignore += len(ignore)

#         images.append({'id': img_id, 'file_name': fname, 'width': W, 'height': H})

#         for (x, y, bw, bh, cat) in valid:
#             x = max(0.0, min(x, W - 1)); y = max(0.0, min(y, H - 1))
#             bw = min(bw, W - x);          bh = min(bh, H - y)
#             if bw <= 1 or bh <= 1:
#                 continue
#             annotations.append({
#                 'id': ann_id, 'image_id': img_id, 'category_id': cat,
#                 'bbox': [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
#                 'area': round(bw * bh, 2), 'iscrowd': 0,
#                 'track_id': 0,
#             })
#             ann_id += 1; n_box += 1

#     # ── CẬP NHẬT CATEGORIES INFO VÀO COCO JSON ──
#     final_categories = MERGED_CATEGORIES if merge else VISDRONE_DET_CATEGORIES
    
#     coco = {
#         'images': images,
#         'annotations': annotations,
#         'categories': [{'id': cid, 'name': name, 'supercategory': 'object'}
#                        for cid, name in final_categories],
#     }
    
#     os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
#     with open(out_path, 'w') as f:
#         json.dump(coco, f)

#     print(f'[done] {len(images)} images, {n_box} boxes → {out_path}')
#     if merge:
#         print(f'       [info] Classes merged to 7 categories successfully.')
#     if blackout:
#         print(f'       blacked-out {n_ignore} ignore regions → cleaned images in {out_images_dir}')
#         print(f'       (point --data_cfg train_img to this cleaned-images dir)')
#     print(f'       skipped: cat(ignored/others)={n_skip_cat}  score0={n_skip_score}  '
#           f'images_without_txt={n_no_txt}')


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--images_dir', required=True)
#     p.add_argument('--ann_dir',    required=True)
#     p.add_argument('--out',        required=True, help='output COCO json path')
#     p.add_argument('--out_images_dir', default='',
#                    help='where to write cleaned images (default: <out_dir>/images)')
#     p.add_argument('--no_blackout', action='store_true',
#                    help='do NOT black out ignore regions (json only, images untouched). '
#                         'Default blacks them out to match the MOT converter.')
#     # Thêm cờ merge_classes
#     p.add_argument('--merge_classes', action='store_true',
#                    help='Merge similar classes (pedestrian+people, van+truck, etc) into 7 classes')
#     a = p.parse_args()
    
#     convert(a.images_dir, a.ann_dir, a.out,
#             out_images_dir=(a.out_images_dir or None),
#             blackout=not a.no_blackout,
#             merge=a.merge_classes)


# if __name__ == '__main__':
#     main()




"""
gen_dataset_visdrone_det_coco.py
Convert VisDrone-DET split (per-image .txt annotations) → COCO JSON.

Class mapping (10 → 7):
    pedestrian (1) + people (2) → pedestrian (1)
    bicycle    (3)              → bicycle   (2)
    car        (4)              → car       (3)
    van        (5)              → van       (4)   ← kept separate from truck
    truck      (6)              → truck     (5)
    tricycle   (7)              → DROPPED
    awning-tri (8)              → DROPPED
    bus        (9)              → bus       (6)
    motor      (10)             → motor     (7)

Supports two eval protocols built on top of the same 7-class COCO json:
    5-class AMOT:        pedestrian / car / van / truck / bus
    4-class competition: pedestrian / car / motor / bicycle
"""

import os
import json
import argparse
import glob

# ── Original VisDrone-DET categories (10 classes) ───────────────────────────
VISDRONE_DET_CATEGORIES = [
    (1, 'pedestrian'), (2, 'people'),   (3, 'bicycle'),        (4, 'car'),
    (5, 'van'),        (6, 'truck'),    (7, 'tricycle'),        (8, 'awning-tricycle'),
    (9, 'bus'),        (10, 'motor'),
]

# ── Target 7-class categories ────────────────────────────────────────────────
TARGET_CATEGORIES = [
    (1, 'pedestrian'),
    (2, 'bicycle'),
    (3, 'car'),
    (4, 'van'),
    (5, 'truck'),
    (6, 'bus'),
    (7, 'motor'),
]

# ── Mapping: old cat_id → new cat_id  (None = DROP) ─────────────────────────
#   1  pedestrian      → 1  pedestrian
#   2  people          → 1  pedestrian   (merge)
#   3  bicycle         → 2  bicycle
#   4  car             → 3  car
#   5  van             → 4  van
#   6  truck           → 5  truck
#   7  tricycle        → None (drop)
#   8  awning-tricycle → None (drop)
#   9  bus             → 6  bus
#  10  motor           → 7  motor
CLASS_MAPPING: dict[int, int | None] = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: None,   # drop
    8: None,   # drop
    9: 6,
    10: 7,
}

# Convenience: eval-protocol views (for documentation / downstream filtering)
PROTOCOLS = {
    '5class_amot':   [1, 3, 4, 5, 6],   # pedestrian, car, van, truck, bus
    '4class_comp':   [1, 2, 3, 7],       # pedestrian, bicycle, car, motor
}

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


# ── Image-size helper ────────────────────────────────────────────────────────
def _image_size(path: str):
    """Return (W, H) without decoding full image."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size          # (W, H)
    except Exception:
        import cv2
        im = cv2.imread(path)
        if im is None:
            return None
        h, w = im.shape[:2]
        return (w, h)


# ── Main converter ───────────────────────────────────────────────────────────
def convert(images_dir: str, ann_dir: str, out_path: str,
            out_images_dir: str = None, blackout: bool = True) -> None:
    import cv2

    img_paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, '*'))
        if p.lower().endswith(_IMG_EXTS)
    )
    if not img_paths:
        raise SystemExit(f'[ERROR] no images found in {images_dir}')

    if blackout:
        if not out_images_dir:
            out_images_dir = os.path.join(
                os.path.dirname(os.path.abspath(out_path)), 'images')
        os.makedirs(out_images_dir, exist_ok=True)

    images, annotations = [], []
    ann_id = 1

    # Stats counters
    n_box = 0
    n_dropped_class = 0   # tricycle / awning-tri
    n_skip_score    = 0   # score == 0  (ignore region)
    n_skip_cat      = 0   # cat 0 or 11
    n_no_txt        = 0
    n_ignore_blur   = 0   # regions blacked-out

    # Per-class box counter for final report
    class_counts: dict[int, int] = {cid: 0 for cid, _ in TARGET_CATEGORIES}

    for img_id, img_path in enumerate(img_paths, start=1):
        fname = os.path.basename(img_path)
        txt   = os.path.join(ann_dir, os.path.splitext(fname)[0] + '.txt')

        valid:  list[tuple] = []   # (x, y, bw, bh, new_cat_id)
        ignore: list[tuple] = []   # (x, y, bw, bh)

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
                        score = float(parts[4])
                        cat   = int(float(parts[5]))
                    except ValueError:
                        continue

                    if bw <= 0 or bh <= 0:
                        continue

                    # ── Ignore regions (cat 0/11 or score 0) → blackout only ──
                    if cat < 1 or cat > 10 or score == 0:
                        ignore.append((x, y, bw, bh))
                        if cat < 1 or cat > 10:
                            n_skip_cat += 1
                        else:
                            n_skip_score += 1
                        continue

                    # ── Apply 7-class mapping ──────────────────────────────
                    new_cat = CLASS_MAPPING[cat]
                    if new_cat is None:
                        # tricycle / awning-tricycle: silently drop
                        n_dropped_class += 1
                        continue

                    valid.append((x, y, bw, bh, new_cat))
        else:
            n_no_txt += 1

        # ── Process image ──────────────────────────────────────────────────
        if blackout:
            img = cv2.imread(img_path)
            if img is None:
                print(f'  [warn] cannot read {fname}, skipped')
                continue
            H, W = img.shape[:2]
            for (x, y, bw, bh) in ignore:
                xi, yi = int(max(0, x)), int(max(0, y))
                img[yi:yi + int(bh), xi:xi + int(bw)] = 0
                n_ignore_blur += 1
            cv2.imwrite(os.path.join(out_images_dir, fname), img)
        else:
            size = _image_size(img_path)
            if size is None:
                print(f'  [warn] cannot read {fname}, skipped')
                continue
            W, H = size
            n_ignore_blur += len(ignore)

        images.append({'id': img_id, 'file_name': fname, 'width': W, 'height': H})

        for (x, y, bw, bh, new_cat) in valid:
            # Clamp to image boundary
            x  = max(0.0, min(x, W - 1))
            y  = max(0.0, min(y, H - 1))
            bw = min(bw, W - x)
            bh = min(bh, H - y)
            if bw <= 1 or bh <= 1:
                continue

            annotations.append({
                'id':          ann_id,
                'image_id':    img_id,
                'category_id': new_cat,
                'bbox':        [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                'area':        round(bw * bh, 2),
                'iscrowd':     0,
                'track_id':    0,       # DET task: no track_id
            })
            ann_id += 1
            n_box  += 1
            class_counts[new_cat] += 1

    # ── Write COCO JSON ────────────────────────────────────────────────────
    coco = {
        'images':      images,
        'annotations': annotations,
        'categories': [
            {'id': cid, 'name': name, 'supercategory': 'object'}
            for cid, name in TARGET_CATEGORIES
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f)

    # ── Report ─────────────────────────────────────────────────────────────
    cat_name = {cid: name for cid, name in TARGET_CATEGORIES}
    print(f'\n[done] {len(images)} images  {n_box} boxes → {out_path}')
    print('  Per-class box counts (7-class target):')
    for cid in sorted(class_counts):
        print(f'    [{cid}] {cat_name[cid]:<15s}: {class_counts[cid]:,}')
    print(f'  Dropped (tricycle/awning-tri)  : {n_dropped_class:,}')
    print(f'  Ignored (score=0)              : {n_skip_score:,}')
    print(f'  Skipped (cat 0/11)             : {n_skip_cat:,}')
    print(f'  Images without .txt            : {n_no_txt}')
    if blackout:
        print(f'  Blacked-out ignore regions     : {n_ignore_blur:,}')
        print(f'  Cleaned images dir             : {out_images_dir}')
    print()
    print('  Eval-protocol class subsets:')
    for proto, ids in PROTOCOLS.items():
        names = [cat_name[i] for i in ids]
        print(f'    {proto}: {names}')


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='Convert VisDrone-DET → COCO JSON (7-class, drop tricycle).')
    p.add_argument('--images_dir',     required=True,
                   help='Path to folder containing .jpg images')
    p.add_argument('--ann_dir',        required=True,
                   help='Path to folder containing per-image .txt annotations')
    p.add_argument('--out',            required=True,
                   help='Output COCO JSON path (e.g. ./ann/instances_train.json)')
    p.add_argument('--out_images_dir', default='',
                   help='Where to write blackout-cleaned images '
                        '(default: <out_dir>/images/). Ignored with --no_blackout.')
    p.add_argument('--no_blackout',    action='store_true',
                   help='Skip writing cleaned images; only produce the JSON.')
    a = p.parse_args()

    convert(
        images_dir    = a.images_dir,
        ann_dir       = a.ann_dir,
        out_path      = a.out,
        out_images_dir= a.out_images_dir or None,
        blackout      = not a.no_blackout,
    )


if __name__ == '__main__':
    main()