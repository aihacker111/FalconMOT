"""
gen_dataset_visdrone_coco.py — Convert VisDrone2019-MOT to COCO JSON format.
With support for merging specific classes and safely handling track_ids.

Output structure:
    CONVERTED_ROOT/
    ├── train/
    │   ├── images/
    │   └── annotations/instances_train.json
    └── val/
        ├── images/
        └── annotations/instances_val.json
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ── Class mapping ────────────────────────────────────────────────────────────
CLS_MAP = {
    1: 'pedestrian', 2: 'people',   3: 'bicycle',        4: 'car',
    5: 'van',        6: 'truck',    7: 'tricycle',        8: 'awning-tricycle',
    9: 'bus',       10: 'motor',
}

MERGED_CLS_MAP = {
    1: 'pedestrian', 2: 'bicycle', 3: 'car', 4: 'truck',
    5: 'tricycle', 6: 'bus', 7: 'motor'
}

CLASS_MAPPING = {
    1: 1,   # pedestrian -> pedestrian
    2: 1,   # people -> pedestrian
    3: 2,   # bicycle -> bicycle
    4: 3,   # car -> car
    5: 4,   # van -> truck
    6: 4,   # truck -> truck
    7: 5,   # tricycle -> tricycle
    8: 5,   # awning-tricycle -> tricycle
    9: 6,   # bus -> bus
    10: 7   # motor -> motor
}

def get_categories(merge: bool):
    target_map = MERGED_CLS_MAP if merge else CLS_MAP
    return [{'id': k, 'name': v, 'supercategory': 'object'} for k, v in target_map.items()]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_ann_file(path: str) -> np.ndarray:
    """Fast annotation parse using numpy. Returns (N, 10) int32 array."""
    with open(path, 'r') as f:
        raw = f.read()
    if not raw.strip():
        return np.zeros((0, 10), dtype=np.int32)
    # Replace commas with spaces so np.fromstring can parse
    data = np.fromstring(raw.replace(',', ' '), dtype=np.int32, sep=' ')
    cols = 10
    rows = len(data) // cols
    return data[: rows * cols].reshape(rows, cols)


def _process_frame(src_path: str, dst_path: str, ignore_boxes: list) -> tuple:
    """Read src frame, black-out ignore regions, write dst. Returns (H, W)."""
    img = cv2.imread(src_path)
    if img is None:
        return None
    for box in ignore_boxes:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img[y: y + h, x: x + w] = 0
    if not os.path.isfile(dst_path):
        cv2.imwrite(dst_path, img)
    return img.shape[0], img.shape[1]


# ── Core converter ────────────────────────────────────────────────────────────

def convert_split(src_root: str, dst_root: str, split: str,
                  workers: int = 4, merge: bool = False) -> None:

    seq_dir = os.path.join(src_root, 'sequences')
    ann_dir = os.path.join(src_root, 'annotations')

    dst_img_root = os.path.join(dst_root, 'images')
    dst_ann_dir  = os.path.join(dst_root, 'annotations')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_ann_dir,  exist_ok=True)

    images_list = []
    anns_list   = []

    img_id      = 0
    ann_id      = 0
    
    num_classes = 7 if merge else 10
    track_start = [0] * num_classes   # global per-class ID offset across sequences
    active_map  = MERGED_CLS_MAP if merge else CLS_MAP

    seq_names = sorted(os.listdir(seq_dir))

    for seq in tqdm(seq_names, desc=f'[{split}]'):
        seq_img_dir  = os.path.join(seq_dir, seq)
        seq_ann_file = os.path.join(ann_dir, seq + '.txt')
        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_file)):
            print(f'  [skip] {seq}')
            continue

        dst_seq_dir = os.path.join(dst_img_root, seq)
        os.makedirs(dst_seq_dir, exist_ok=True)

        # ── 1. Parse annotations ─────────────────────────────────────────
        arr = _parse_ann_file(seq_ann_file)
        if arr.shape[0] == 0:
            continue

        is_ignore = (arr[:, 6] == 0) | (arr[:, 7] == 0) | (arr[:, 7] == 11)
        is_valid  = (arr[:, 6] == 1) & (arr[:, 7] > 0)  & (arr[:, 7] < 11)

        ignore_by_frame: dict[int, list] = defaultdict(list)
        valid_by_frame:  dict[int, list] = defaultdict(list)

        for row in arr[is_ignore]:
            ignore_by_frame[int(row[0])].append(row[2:6])

        for row in arr[is_valid]:
            valid_by_frame[int(row[0])].append(row)

        # ── 2. Build track-id rank dict safely with Tuple (raw_cls, raw_tid) ─────
        track_ids_per_cls: dict[int, set] = defaultdict(set)
        for rows in valid_by_frame.values():
            for row in rows:
                raw_cls_1 = int(row[7])
                raw_tid   = int(row[1])
                new_cls_1 = CLASS_MAPPING[raw_cls_1] if merge else raw_cls_1
                new_cls_0 = new_cls_1 - 1
                
                # Lưu trữ theo tuple để tránh đụng độ ID giữa các class cũ bị merge lại
                track_ids_per_cls[new_cls_0].add((raw_cls_1, raw_tid))

        # rank_map[new_cls_0][(raw_cls_1, raw_tid)] → local_rank
        rank_map = {
            cls_id: {unique_key: rank for rank, unique_key in enumerate(sorted(keys))}
            for cls_id, keys in track_ids_per_cls.items()
        }
        seq_n_ids = {cls_id: len(keys) for cls_id, keys in track_ids_per_cls.items()}

        tqdm.write(f'  {seq}: ' + '  '.join(
            f'{active_map[c+1]}={n}' for c, n in sorted(seq_n_ids.items())))

        # ── 3. Parallel image I/O ─────────────────────────────────────────
        frame_ids   = sorted(valid_by_frame.keys())
        frame_sizes: dict[int, tuple] = {}   

        def _job(fr_id: int):
            fr_name  = f'{fr_id:07d}.jpg'
            src_path = os.path.join(seq_img_dir, fr_name)
            dst_path = os.path.join(dst_seq_dir, fr_name)
            hw = _process_frame(src_path, dst_path, ignore_by_frame.get(fr_id, []))
            return fr_id, hw

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_job, fid): fid for fid in frame_ids}
            for fut in as_completed(futs):
                fr_id, hw = fut.result()
                if hw is not None:
                    frame_sizes[fr_id] = hw

        # ── 4. Build COCO records (sequential, deterministic order) ──────
        for fr_id in frame_ids:
            if fr_id not in frame_sizes:
                continue
            H, W = frame_sizes[fr_id]
            fr_name  = f'{fr_id:07d}.jpg'
            rel_path = f'{seq}/{fr_name}'

            images_list.append({
                'id':       img_id,
                'file_name': rel_path,
                'height':   H,
                'width':    W,
                'seq_id':   seq,
                'frame_id': fr_id,           
            })

            for row in valid_by_frame[fr_id]:
                raw_cls_1 = int(row[7])
                raw_tid   = int(row[1])
                new_cls_1 = CLASS_MAPPING[raw_cls_1] if merge else raw_cls_1
                new_cls_0 = new_cls_1 - 1
                
                x1 = float(row[2]);  y1 = float(row[3])
                bw = float(row[4]);  bh = float(row[5])
                if bw <= 0 or bh <= 0:
                    continue

                # Tra cứu rank an toàn bằng Tuple
                local_rank = rank_map[new_cls_0][(raw_cls_1, raw_tid)]
                track_id   = local_rank + track_start[new_cls_0]

                anns_list.append({
                    'id':          ann_id,
                    'image_id':    img_id,
                    'category_id': new_cls_1,
                    'bbox':        [x1, y1, bw, bh],
                    'area':        bw * bh,
                    'iscrowd':     0,
                    'track_id':    track_id,
                })
                ann_id += 1

            img_id += 1

        # Advance global track-id offsets
        for cls_id in range(num_classes):
            track_start[cls_id] += seq_n_ids.get(cls_id, 0)

    # ── 5. Write JSON ─────────────────────────────────────────────────────
    coco = {
        'images':      images_list,
        'annotations': anns_list,
        'categories':  get_categories(merge),
    }
    out_json = os.path.join(dst_ann_dir, f'instances_{split}.json')
    with open(out_json, 'w') as f:
        json.dump(coco, f, separators=(',', ':'))

    # ── 6. Stats ──────────────────────────────────────────────────────────
    print(f'\n[{split}] {len(images_list):,} images  {len(anns_list):,} annotations')
    if merge:
        print(f'  [Info] Classes merged securely into 7 categories.')
    print(f'  → {out_json}')

    max_tid: dict[int, int] = defaultdict(int)
    for ann in anns_list:
        c = ann['category_id'] - 1
        if ann['track_id'] + 1 > max_tid[c]:
            max_tid[c] = ann['track_id'] + 1
            
    print('  Track IDs generated:')
    for c in sorted(max_tid):
        print(f'    {active_map[c+1]}: {max_tid[c]} unique object IDs')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Convert VisDrone2019-MOT → COCO JSON for training.')
    ap.add_argument('--visdrone_root', default='/workspace')
    ap.add_argument('--output_root',   default='/workspace/VisDrone2019-COCO')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'],
                    choices=['train', 'val', 'test-dev'])
    ap.add_argument('--workers', type=int, default=4,
                    help='parallel threads for image I/O per sequence')
    # Thêm cờ merge_classes
    ap.add_argument('--merge_classes', action='store_true',
                   help='Merge specific classes to 7 classes safely handling track_ids')
    args = ap.parse_args()

    for split in args.splits:
        src = os.path.join(args.visdrone_root, f'VisDrone2019-MOT-{split}')
        dst = os.path.join(args.output_root, split)
        if not os.path.isdir(src):
            print(f'[Error] Not found: {src}')
            continue
        convert_split(src, dst, split, workers=args.workers, merge=args.merge_classes)


if __name__ == '__main__':
    main()