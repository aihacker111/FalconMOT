"""
visdrone2coco_5cls_benchmark_mot.py
Chuyển đổi VisDrone2019-MOT sang COCO JSON, merge về 5 class benchmark:
    1: pedestrian, 2: car, 3: van, 4: truck, 5: bus
    (Trong đó: pedestrian = pedestrian + people. Các class khác bị drop: bicycle, tricycle, awning-tricycle, motor)

Logic ảnh: KHÔNG bôi đen (blackout) các class bị drop — giữ nguyên pixel; 
chỉ bôi đen các vùng IGNORE thực sự của VisDrone (score==0 hoặc cls==0 hoặc cls==11).

Track-ID guarantee:
    • Sử dụng TrackIDManager để cấp ID global, an toàn và không bị đụng độ 
      khi merge class (vd: people -> pedestrian).
    • Record ảnh chứa đầy đủ `seq_id` và `frame_id` cho LoadCocoSequencesForTracking.
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ── Class definitions ─────────────────────────────────────────────────────────

# Target 5-class cat_id → name (Benchmark)
TARGET_CLS_MAP: dict[int, str] = {
    1: 'pedestrian',
    2: 'car',
    3: 'van',
    4: 'truck',
    5: 'bus',
}

# old cat_id → new cat_id  (None = DROP)
CLASS_MAPPING: dict[int, int | None] = {
    1: 1,       # pedestrian → pedestrian
    2: 1,       # people     → pedestrian  (merge)
    3: None,    # bicycle    → DROP
    4: 2,       # car        → car
    5: 3,       # van        → van
    6: 4,       # truck      → truck
    7: None,    # tricycle   → DROP
    8: None,    # awning-tri → DROP
    9: 5,       # bus        → bus
    10: None,   # motor      → DROP
}

NUM_TARGET_CLASSES = len(TARGET_CLS_MAP)   # 5


def _get_categories() -> list[dict]:
    return [{'id': k, 'name': v, 'supercategory': 'object'}
            for k, v in TARGET_CLS_MAP.items()]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _parse_ann_file(path: str) -> np.ndarray:
    """Parse annotation VisDrone -> (N, 10) int32. Cắt phần dư không đủ 10 cột."""
    with open(path, 'r') as f:
        raw = f.read()
    if not raw.strip():
        return np.zeros((0, 10), dtype=np.int32)
    data = np.fromstring(raw.replace(',', ' '), dtype=np.int32, sep=' ')
    cols = 10
    rows = len(data) // cols
    return data[:rows * cols].reshape(rows, cols)


def _process_frame(src_path: str, dst_path: str, ignore_boxes: list, overwrite: bool) -> tuple | None:
    """Đọc frame gốc, CHỈ bôi đen vùng ignore thực sự, ghi ra dst. Trả về (H, W)."""
    img = cv2.imread(src_path)
    if img is None:
        return None
    for box in ignore_boxes:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img[y:y + h, x:x + w] = 0
    if overwrite or not os.path.isfile(dst_path):
        cv2.imwrite(dst_path, img)
    return img.shape[0], img.shape[1]


# ── Track-ID management ───────────────────────────────────────────────────────

class TrackIDManager:
    """Manages globally-unique, per-class track IDs across sequences."""

    def __init__(self):
        # track_start[new_cat_id - 1]  (0-indexed)
        self._start = [0] * NUM_TARGET_CLASSES

    def build_seq_map(self, valid_rows: list[np.ndarray]) -> dict:
        """
        seq_map[(old_cat, old_tid)] → global_track_id
        """
        per_new_cat: dict[int, set] = defaultdict(set)
        for row in valid_rows:
            old_cat = int(row[7])
            old_tid = int(row[1])
            new_cat = CLASS_MAPPING[old_cat]
            if new_cat is None:
                continue
            new_cat_idx = new_cat - 1
            per_new_cat[new_cat_idx].add((old_cat, old_tid))

        seq_map: dict[tuple, int] = {}
        n_per_cls: dict[int, int] = {}
        for new_cat_idx, keys in per_new_cat.items():
            sorted_keys = sorted(keys)
            for rank, key in enumerate(sorted_keys):
                seq_map[key] = rank + self._start[new_cat_idx]
            n_per_cls[new_cat_idx] = len(sorted_keys)

        for new_cat_idx, n in n_per_cls.items():
            self._start[new_cat_idx] += n

        return seq_map

    def totals(self) -> dict[int, int]:
        return {idx + 1: n for idx, n in enumerate(self._start) if n > 0}


# ── Core converter ────────────────────────────────────────────────────────────

def convert_split(src_root: str, dst_root: str, split: str,
                  workers: int = 4, overwrite: bool = False) -> None:

    seq_dir = os.path.join(src_root, 'sequences')
    ann_dir = os.path.join(src_root, 'annotations')

    dst_img_root = os.path.join(dst_root, 'images')
    dst_ann_dir  = os.path.join(dst_root, 'annotations')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_ann_dir,  exist_ok=True)

    images_list: list[dict] = []
    anns_list:   list[dict] = []

    img_id   = 0
    ann_id   = 0
    tid_mgr  = TrackIDManager()

    n_dropped  = 0
    n_box      = 0
    class_box_counts: dict[int, int] = {k: 0 for k in TARGET_CLS_MAP}

    seq_names = sorted(os.listdir(seq_dir))

    for seq in tqdm(seq_names, desc=f'[{split}]'):
        seq_img_dir  = os.path.join(seq_dir, seq)
        seq_ann_file = os.path.join(ann_dir, seq + '.txt')
        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_file)):
            continue

        dst_seq_dir = os.path.join(dst_img_root, seq)
        os.makedirs(dst_seq_dir, exist_ok=True)

        arr = _parse_ann_file(seq_ann_file)
        if arr.shape[0] == 0:
            continue

        is_ignore = (arr[:, 6] == 0) | (arr[:, 7] == 0) | (arr[:, 7] == 11)
        is_valid  = (arr[:, 6] == 1) & (arr[:, 7] >= 1) & (arr[:, 7] <= 10)

        ignore_by_frame: dict[int, list] = defaultdict(list)
        valid_by_frame:  dict[int, list] = defaultdict(list)

        for row in arr[is_ignore]:
            ignore_by_frame[int(row[0])].append(row[2:6])
        for row in arr[is_valid]:
            valid_by_frame[int(row[0])].append(row)

        all_valid_rows = [row for rows in valid_by_frame.values() for row in rows]
        seq_map = tid_mgr.build_seq_map(all_valid_rows)

        frame_ids    = sorted(valid_by_frame.keys())
        frame_sizes: dict[int, tuple] = {}

        def _job(fr_id: int):
            fr_name  = f'{fr_id:07d}.jpg'
            src_path = os.path.join(seq_img_dir, fr_name)
            dst_path = os.path.join(dst_seq_dir, fr_name)
            hw = _process_frame(src_path, dst_path, ignore_by_frame.get(fr_id, []), overwrite)
            return fr_id, hw

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_job, fid): fid for fid in frame_ids}
            for fut in as_completed(futs):
                fr_id, hw = fut.result()
                if hw is not None:
                    frame_sizes[fr_id] = hw

        for fr_id in frame_ids:
            if fr_id not in frame_sizes:
                continue
            H, W   = frame_sizes[fr_id]
            rel_path = f'{seq}/{fr_id:07d}.jpg'

            images_list.append({
                'id':        img_id,
                'file_name': rel_path,
                'height':    H,
                'width':     W,
                'seq_id':    seq,        # BAT BUOC cho LoadCocoSequencesForTracking
                'frame_id':  fr_id,      # BAT BUOC
            })

            for row in valid_by_frame[fr_id]:
                old_cat = int(row[7])
                old_tid = int(row[1])
                new_cat = CLASS_MAPPING[old_cat]

                if new_cat is None:
                    n_dropped += 1
                    continue

                x1, y1, bw, bh = (float(row[2]), float(row[3]),
                                   float(row[4]), float(row[5]))
                if bw <= 0 or bh <= 0:
                    continue

                global_tid = seq_map[(old_cat, old_tid)]

                anns_list.append({
                    'id':          ann_id,
                    'image_id':    img_id,
                    'category_id': new_cat,
                    'bbox':        [x1, y1, bw, bh],
                    'area':        bw * bh,
                    'iscrowd':     0,
                    'track_id':    global_tid,
                })
                ann_id += 1
                n_box  += 1
                class_box_counts[new_cat] += 1

            img_id += 1

    coco = {
        'images':      images_list,
        'annotations': anns_list,
        'categories':  _get_categories(),
    }
    out_json = os.path.join(dst_ann_dir, f'instances_{split}.json')
    with open(out_json, 'w') as f:
        json.dump(coco, f, separators=(',', ':'))

    print(f'\n[{split}] {len(images_list):,} images  {n_box:,} annotations')
    print('  Per-class box counts:')
    for cid in sorted(class_box_counts):
        print(f'    [{cid}] {TARGET_CLS_MAP[cid]:<15s}: {class_box_counts[cid]:,}')
    print(f'  Dropped classes (bicycle/tricycles/motor): {n_dropped:,}')

    totals = tid_mgr.totals()
    print('  Unique track IDs per class:')
    for cid in sorted(totals):
        print(f'    [{cid}] {TARGET_CLS_MAP[cid]:<15s}: {totals[cid]:,}')
    print(f'  → {out_json}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Convert VisDrone2019-MOT → COCO JSON (5-class benchmark).')
    ap.add_argument('--visdrone_root', default='/workspace/VisDrone2019',
                    help='Root dir containing VisDrone2019-MOT-{train,val,test-dev}/')
    ap.add_argument('--output_root',   default='/workspace/VisDrone2019-COCO-5cls',
                    help='Output root dir')
    ap.add_argument('--splits', nargs='+',
                    default=['test-dev'],
                    choices=['train', 'val', 'test-dev'])
    ap.add_argument('--workers', type=int, default=8,
                    help='Parallel threads for image I/O per sequence')
    ap.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing images')
    args = ap.parse_args()

    for split in args.splits:
        src = os.path.join(args.visdrone_root, f'VisDrone2019-MOT-{split}')
        dst = os.path.join(args.output_root, split)
        if not os.path.isdir(src):
            print(f'[Error] Not found: {src}')
            continue
        convert_split(src, dst, split, workers=args.workers, overwrite=args.overwrite)


if __name__ == '__main__':
    main()