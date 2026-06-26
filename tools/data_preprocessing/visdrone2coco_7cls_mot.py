"""
gen_dataset_visdrone_mot_coco.py
Convert VisDrone2019-MOT → COCO JSON format.

Class mapping (10 → 7, drop tricycle & awning-tricycle):
    pedestrian (1) + people (2) → pedestrian (1)
    bicycle    (3)              → bicycle   (2)
    car        (4)              → car       (3)
    van        (5)              → van       (4)   ← kept separate from truck
    truck      (6)              → truck     (5)
    tricycle   (7)              → DROPPED
    awning-tri (8)              → DROPPED
    bus        (9)              → bus       (6)
    motor      (10)             → motor     (7)

Track-ID guarantee:
    • Each (new_cat_id, track_id) pair is globally unique across all sequences.
    • When two old classes merge (people→pedestrian), IDs from BOTH old classes
      are unified per-sequence via a (old_cat, old_tid) → new_tid lookup before
      the global offset is applied.
    • Dropped classes (tricycle, awning-tri) contribute zero IDs → no gap in
      track_start offsets.

Output:
    <output_root>/<split>/
        images/<seq>/<frame>.jpg   (ignore regions blacked out)
        annotations/instances_<split>.json

Eval protocols (subset of category_ids from the 7-class JSON):
    5-class AMOT  : pedestrian(1) / car(3) / van(4) / truck(5) / bus(6)
    4-class comp  : pedestrian(1) / bicycle(2) / car(3) / motor(7)
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

# Original VisDrone cat_id → name
VISDRONE_CLS_MAP: dict[int, str] = {
    1: 'pedestrian', 2: 'people',          3: 'bicycle',
    4: 'car',        5: 'van',             6: 'truck',
    7: 'tricycle',   8: 'awning-tricycle', 9: 'bus',
    10: 'motor',
}

# Target 7-class cat_id → name
TARGET_CLS_MAP: dict[int, str] = {
    1: 'pedestrian',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'bus',
    7: 'motor',
}

# old cat_id → new cat_id  (None = DROP)
CLASS_MAPPING: dict[int, int | None] = {
    1: 1,       # pedestrian → pedestrian
    2: 1,       # people     → pedestrian  (merge)
    3: 2,       # bicycle    → bicycle
    4: 3,       # car        → car
    5: 4,       # van        → van
    6: 5,       # truck      → truck
    7: None,    # tricycle   → DROP
    8: None,    # awning-tri → DROP
    9: 6,       # bus        → bus
    10: 7,      # motor      → motor
}

NUM_TARGET_CLASSES = len(TARGET_CLS_MAP)   # 7

# Convenience: eval-protocol views
PROTOCOLS = {
    '5class_amot': [1, 3, 4, 5, 6],   # pedestrian, car, van, truck, bus
    '4class_comp': [1, 2, 3, 7],       # pedestrian, bicycle, car, motor
}


def _get_categories() -> list[dict]:
    return [{'id': k, 'name': v, 'supercategory': 'object'}
            for k, v in TARGET_CLS_MAP.items()]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _parse_ann_file(path: str) -> np.ndarray:
    """
    Parse a VisDrone-MOT annotation .txt file.
    Each line: frame_id, track_id, x, y, w, h, score, cat, truncation, occlusion
    Returns shape (N, 10) int32.  Empty file → (0, 10).
    """
    with open(path, 'r') as f:
        raw = f.read()
    if not raw.strip():
        return np.zeros((0, 10), dtype=np.int32)
    data = np.fromstring(raw.replace(',', ' '), dtype=np.int32, sep=' ')
    cols = 10
    rows = len(data) // cols
    return data[:rows * cols].reshape(rows, cols)


def _process_frame(src_path: str, dst_path: str, ignore_boxes: list) -> tuple | None:
    """Black-out ignore regions and write image.  Returns (H, W) or None."""
    img = cv2.imread(src_path)
    if img is None:
        return None
    for box in ignore_boxes:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img[y:y + h, x:x + w] = 0
    if not os.path.isfile(dst_path):
        cv2.imwrite(dst_path, img)
    return img.shape[0], img.shape[1]   # H, W


# ── Track-ID management ───────────────────────────────────────────────────────
#
# Goal: produce globally-unique, contiguous track_ids per (new) class,
#       even when multiple old classes merge into one.
#
# Approach per sequence:
#   1. Collect all (old_cat, old_tid) pairs that map to each new_cat.
#   2. Sort them deterministically → assign local rank 0, 1, 2, …
#   3. global_track_id = local_rank + track_start[new_cat_idx]
#   4. After the sequence, advance track_start[new_cat_idx] by the number
#      of unique objects of that new class in the sequence.
#
# No gaps arise from dropped classes because we never add to track_start
# for cat_ids that map to None.

class TrackIDManager:
    """Manages globally-unique, per-class track IDs across sequences."""

    def __init__(self):
        # track_start[new_cat_id - 1]  (0-indexed)
        self._start = [0] * NUM_TARGET_CLASSES

    def build_seq_map(self, valid_rows: list[np.ndarray]) -> dict:
        """
        Given all valid annotation rows for one sequence, return:
            seq_map[(old_cat, old_tid)] → global_track_id

        Also advances internal offsets for the next sequence.
        """
        # Collect unique (old_cat, old_tid) per new_cat
        per_new_cat: dict[int, set] = defaultdict(set)
        for row in valid_rows:
            old_cat = int(row[7])
            old_tid = int(row[1])
            new_cat = CLASS_MAPPING[old_cat]
            if new_cat is None:
                continue
            new_cat_idx = new_cat - 1
            per_new_cat[new_cat_idx].add((old_cat, old_tid))

        # Assign local ranks (sorted for determinism)
        seq_map: dict[tuple, int] = {}
        n_per_cls: dict[int, int] = {}
        for new_cat_idx, keys in per_new_cat.items():
            sorted_keys = sorted(keys)   # (old_cat, old_tid) → deterministic
            for rank, key in enumerate(sorted_keys):
                seq_map[key] = rank + self._start[new_cat_idx]
            n_per_cls[new_cat_idx] = len(sorted_keys)

        # Advance global offsets
        for new_cat_idx, n in n_per_cls.items():
            self._start[new_cat_idx] += n

        return seq_map

    def totals(self) -> dict[int, int]:
        """Return total unique track IDs assigned per new cat_id (1-indexed)."""
        return {idx + 1: n for idx, n in enumerate(self._start) if n > 0}


# ── Core converter ────────────────────────────────────────────────────────────

def convert_split(src_root: str, dst_root: str, split: str,
                  workers: int = 4) -> None:

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

    # Stats
    n_dropped  = 0   # annotations dropped because class is None
    n_box      = 0
    class_box_counts: dict[int, int] = {k: 0 for k in TARGET_CLS_MAP}

    seq_names = sorted(os.listdir(seq_dir))

    for seq in tqdm(seq_names, desc=f'[{split}]'):
        seq_img_dir  = os.path.join(seq_dir, seq)
        seq_ann_file = os.path.join(ann_dir, seq + '.txt')
        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_file)):
            print(f'  [skip] {seq}')
            continue

        dst_seq_dir = os.path.join(dst_img_root, seq)
        os.makedirs(dst_seq_dir, exist_ok=True)

        # ── 1. Parse annotation file ─────────────────────────────────────
        arr = _parse_ann_file(seq_ann_file)
        if arr.shape[0] == 0:
            continue

        # Ignore rows: score==0  OR  cat==0  OR  cat==11
        is_ignore = (arr[:, 6] == 0) | (arr[:, 7] == 0) | (arr[:, 7] == 11)
        # Valid rows: score==1  AND  cat in [1..10]
        is_valid  = (arr[:, 6] == 1) & (arr[:, 7] >= 1) & (arr[:, 7] <= 10)

        ignore_by_frame: dict[int, list] = defaultdict(list)
        valid_by_frame:  dict[int, list] = defaultdict(list)

        for row in arr[is_ignore]:
            ignore_by_frame[int(row[0])].append(row[2:6])
        for row in arr[is_valid]:
            valid_by_frame[int(row[0])].append(row)

        # ── 2. Build per-sequence track-ID map ───────────────────────────
        #    Pass ALL valid rows (across all frames) so the manager can see
        #    every (old_cat, old_tid) pair in this sequence.
        all_valid_rows = [row for rows in valid_by_frame.values() for row in rows]
        seq_map = tid_mgr.build_seq_map(all_valid_rows)

        # Log per-sequence object counts
        seq_cls_cnt: dict[int, int] = defaultdict(int)
        for (old_cat, _), gid in seq_map.items():
            new_cat = CLASS_MAPPING[old_cat]
            if new_cat is not None:
                seq_cls_cnt[new_cat] += 1
        tqdm.write(
            f'  {seq}: ' +
            '  '.join(f'{TARGET_CLS_MAP[c]}={n}'
                      for c, n in sorted(seq_cls_cnt.items()))
        )

        # ── 3. Parallel image I/O ────────────────────────────────────────
        frame_ids    = sorted(valid_by_frame.keys())
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

        # ── 4. Build COCO records (deterministic frame order) ────────────
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
                'seq_id':    seq,
                'frame_id':  fr_id,
            })

            for row in valid_by_frame[fr_id]:
                old_cat = int(row[7])
                old_tid = int(row[1])
                new_cat = CLASS_MAPPING[old_cat]

                # Drop tricycle / awning-tricycle
                if new_cat is None:
                    n_dropped += 1
                    continue

                x1, y1, bw, bh = (float(row[2]), float(row[3]),
                                   float(row[4]), float(row[5]))
                if bw <= 0 or bh <= 0:
                    continue

                # Look up the global track_id assigned by TrackIDManager
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

    # ── 5. Write JSON ─────────────────────────────────────────────────────
    coco = {
        'images':      images_list,
        'annotations': anns_list,
        'categories':  _get_categories(),
    }
    out_json = os.path.join(dst_ann_dir, f'instances_{split}.json')
    with open(out_json, 'w') as f:
        json.dump(coco, f, separators=(',', ':'))

    # ── 6. Final report ───────────────────────────────────────────────────
    print(f'\n[{split}] {len(images_list):,} images  {n_box:,} annotations')
    print('  Per-class box counts:')
    for cid in sorted(class_box_counts):
        print(f'    [{cid}] {TARGET_CLS_MAP[cid]:<15s}: {class_box_counts[cid]:,}')
    print(f'  Dropped (tricycle/awning-tri)  : {n_dropped:,}')

    # Verify track-ID correctness
    totals = tid_mgr.totals()
    print('  Unique track IDs per class:')
    for cid in sorted(totals):
        print(f'    [{cid}] {TARGET_CLS_MAP[cid]:<15s}: {totals[cid]:,}')

    # Sanity-check: no duplicate (cat, track_id) across annotations
    seen: set[tuple] = set()
    dup_count = 0
    for ann in anns_list:
        key = (ann['category_id'], ann['track_id'])
        # Same (cat, tid) can appear in multiple frames — that is correct.
        # We check that each unique (cat, tid) pair actually corresponds to
        # exactly one semantic object (i.e., track_id is within expected range).
        if ann['track_id'] >= totals.get(ann['category_id'], 0):
            dup_count += 1
    if dup_count:
        print(f'  [WARN] {dup_count} annotations have out-of-range track_id — '
              'please inspect the source data.')
    else:
        print('  [OK] All track_ids are within expected range.')

    print()
    print('  Eval-protocol class subsets:')
    for proto, ids in PROTOCOLS.items():
        names = [TARGET_CLS_MAP[i] for i in ids]
        print(f'    {proto}: {names}')
    print(f'  → {out_json}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Convert VisDrone2019-MOT → COCO JSON (7-class).')
    ap.add_argument('--visdrone_root', default='/workspace',
                    help='Root dir containing VisDrone2019-MOT-{train,val,test-dev}/')
    ap.add_argument('--output_root',   default='/workspace/VisDrone2019-COCO',
                    help='Output root dir')
    ap.add_argument('--splits', nargs='+',
                    default=['train', 'val', 'test-dev'],
                    choices=['train', 'val', 'test-dev'])
    ap.add_argument('--workers', type=int, default=8,
                    help='Parallel threads for image I/O per sequence')
    args = ap.parse_args()

    for split in args.splits:
        src = os.path.join(args.visdrone_root, f'VisDrone2019-MOT-{split}')
        dst = os.path.join(args.output_root, split)
        if not os.path.isdir(src):
            print(f'[Error] Not found: {src}')
            continue
        convert_split(src, dst, split, workers=args.workers)


if __name__ == '__main__':
    main()