"""
visdrone2coco_4cls_competition_mot.py
Chuyển đổi VisDrone2019-MOT sang COCO JSON, merge về 4 class (competition):
    1: person, 2: car, 3: motorcycle, 4: bicycle
    (Drop hoàn toàn: van, truck, tricycle, awning-tricycle, bus)

Khớp track_4cls.py (remap 7cls model -> 4cls eval) và class names tương ứng.

Logic ảnh: KHÔNG bôi đen (blackout) các class bị drop (van/truck/tricycle/bus) —
giữ nguyên pixel; chỉ bôi đen các vùng IGNORE thực sự của VisDrone (score==0 hoặc
cls==0 (unlabeled) hoặc cls==11 (others)).

Mỗi image record ghi 'seq_id' và 'frame_id' (BẮT BUỘC cho
LoadCocoSequencesForTracking gom frame theo sequence + sort theo frame thật).

Output structure:
    OUTPUT_ROOT/
    └── <split>/
        ├── images/<seq>/<frame:07d>.jpg
        └── annotations/instances_<split>.json
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Class mapping 4 class competition (1-indexed target) ─────────────────────
TARGET_CLS_MAP = {
    1: 'person', 2: 'car', 3: 'motorcycle', 4: 'bicycle'
}
NUM_CLASSES = len(TARGET_CLS_MAP)

# VisDrone gốc 1-indexed -> 4 class target (None = DROP)
#   1 pedestrian  2 people  3 bicycle  4 car   5 van
#   6 truck       7 tricycle 8 awning-tricycle 9 bus 10 motor
CLASS_MAPPING = {
    1: 1, 2: 1,      # pedestrian, people     -> person      (1)
    3: 4,            # bicycle                -> bicycle     (4)
    4: 2,            # car                    -> car         (2)
    5: None,         # van                    -> DROP
    6: None,         # truck                  -> DROP
    7: None,         # tricycle               -> DROP
    8: None,         # awning-tricycle        -> DROP
    9: None,         # bus                    -> DROP
    10: 3,           # motor                  -> motorcycle  (3)
}


def get_categories():
    return [{'id': k, 'name': v, 'supercategory': 'object'}
            for k, v in TARGET_CLS_MAP.items()]


def _parse_ann_file(path: str) -> np.ndarray:
    """Parse annotation VisDrone -> (N, 10) int32. Cắt phần dư không đủ 10 cột."""
    with open(path, 'r') as f:
        raw = f.read()
    if not raw.strip():
        return np.zeros((0, 10), dtype=np.int32)
    data = np.fromstring(raw.replace(',', ' '), dtype=np.int32, sep=' ')
    cols = 10
    rows = len(data) // cols
    return data[: rows * cols].reshape(rows, cols)


def _process_frame(src_path: str, dst_path: str, ignore_boxes: list,
                   overwrite: bool) -> tuple:
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


def convert_split(src_root, dst_root, split, workers=8, overwrite=False):
    seq_dir = os.path.join(src_root, 'sequences')
    ann_dir = os.path.join(src_root, 'annotations')
    dst_img_root = os.path.join(dst_root, 'images')
    dst_ann_dir = os.path.join(dst_root, 'annotations')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    images_list, anns_list = [], []
    img_id, ann_id = 0, 0
    track_start = [0] * NUM_CLASSES          # offset track-id toàn cục theo class
    total_dropped = 0

    for seq in tqdm(sorted(os.listdir(seq_dir)), desc=f'[{split}]'):
        seq_img_dir = os.path.join(seq_dir, seq)
        seq_ann_file = os.path.join(ann_dir, seq + '.txt')
        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_file)):
            continue

        dst_seq_dir = os.path.join(dst_img_root, seq)
        os.makedirs(dst_seq_dir, exist_ok=True)

        arr = _parse_ann_file(seq_ann_file)
        if arr.shape[0] == 0:
            continue

        is_ignore = (arr[:, 6] == 0) | (arr[:, 7] == 0) | (arr[:, 7] == 11)
        is_valid = (arr[:, 6] == 1) & (arr[:, 7] > 0) & (arr[:, 7] < 11)

        # Vùng ignore thật -> luôn bôi đen.
        ignore_by_frame = defaultdict(list)
        for row in arr[is_ignore]:
            ignore_by_frame[int(row[0])].append(row[2:6])

        # Object hợp lệ: giữ class merge được, drop van/truck/tricycle/bus
        # (KHÔNG bôi đen — giữ nguyên pixel).
        valid_by_frame = defaultdict(list)
        seq_dropped = 0
        for row in arr[is_valid]:
            if CLASS_MAPPING[int(row[7])] is None:
                seq_dropped += 1
                continue
            valid_by_frame[int(row[0])].append(row)
        total_dropped += seq_dropped

        # Rank track-id theo tuple (raw_cls, raw_tid): pedestrian#3 != people#3
        # dù cùng merge vào 'person' -> hai object khác nhau không bị gán chung id.
        track_ids_per_cls = defaultdict(set)
        for rows in valid_by_frame.values():
            for row in rows:
                new_cls_0 = CLASS_MAPPING[int(row[7])] - 1
                track_ids_per_cls[new_cls_0].add((int(row[7]), int(row[1])))
        rank_map = {
            c: {key: r for r, key in enumerate(sorted(keys))}
            for c, keys in track_ids_per_cls.items()
        }

        # Ghi ảnh (song song).
        frame_ids = sorted(valid_by_frame.keys())
        frame_sizes = {}

        def _job(fr_id):
            src_path = os.path.join(seq_img_dir, f'{fr_id:07d}.jpg')
            dst_path = os.path.join(dst_seq_dir, f'{fr_id:07d}.jpg')
            return fr_id, _process_frame(src_path, dst_path,
                                         ignore_by_frame.get(fr_id, []), overwrite)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fr_id, hw in [fut.result() for fut in
                              as_completed([pool.submit(_job, fid) for fid in frame_ids])]:
                if hw:
                    frame_sizes[fr_id] = hw

        # Build COCO records (tuần tự, thứ tự xác định).
        for fr_id in frame_ids:
            if fr_id not in frame_sizes:
                continue
            H, W = frame_sizes[fr_id]
            images_list.append({
                'id':        img_id,
                'file_name': f'{seq}/{fr_id:07d}.jpg',
                'height':    H,
                'width':     W,
                'seq_id':    seq,        # <-- BAT BUOC cho tracking loader
                'frame_id':  fr_id,      # <-- BAT BUOC cho tracking loader (frame that)
            })
            for row in valid_by_frame[fr_id]:
                new_cls = CLASS_MAPPING[int(row[7])]
                bw, bh = float(row[4]), float(row[5])
                if bw <= 0 or bh <= 0:
                    continue
                anns_list.append({
                    'id':          ann_id,
                    'image_id':    img_id,
                    'category_id': new_cls,
                    'bbox':        [float(row[2]), float(row[3]), bw, bh],
                    'area':        bw * bh,
                    'iscrowd':     0,
                    'track_id':    rank_map[new_cls - 1][(int(row[7]), int(row[1]))]
                                   + track_start[new_cls - 1],
                })
                ann_id += 1
            img_id += 1

        for c in range(NUM_CLASSES):
            track_start[c] += len(track_ids_per_cls[c])

    out_json = os.path.join(dst_ann_dir, f'instances_{split}.json')
    with open(out_json, 'w') as f:
        json.dump({'images': images_list, 'annotations': anns_list,
                   'categories': get_categories()}, f)

    print(f'\n[{split}] {len(images_list):,} images  {len(anns_list):,} annotations')
    print(f'  [Info] Merged -> {NUM_CLASSES} class: ' + ' / '.join(TARGET_CLS_MAP.values()))
    print(f'  [Info] Dropped {total_dropped:,} van/truck/tricycle/bus objects (pixels kept)')
    print(f'  -> {out_json}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='VisDrone2019-MOT -> COCO JSON, 4-class competition merge '
                    '(person/car/motorcycle/bicycle, drop van/truck/tricycle/bus).')
    ap.add_argument('--visdrone_root', default='/workspace/VisDrone2019')
    ap.add_argument('--output_root', default='/workspace/VisDrone2019-COCO-4cls')
    ap.add_argument('--splits', nargs='+', default=['test-dev'],
                    choices=['train', 'val', 'test-dev'],
                    help='cac split can convert (mac dinh: test-dev)')
    ap.add_argument('--workers', type=int, default=8,
                    help='so thread I/O anh song song')
    ap.add_argument('--overwrite', action='store_true',
                    help='ghi de anh da ton tai (mac dinh bo qua anh da co)')
    args = ap.parse_args()

    for split in args.splits:
        src = os.path.join(args.visdrone_root, f'VisDrone2019-MOT-{split}')
        dst = os.path.join(args.output_root, split)
        if not os.path.isdir(src):
            print(f'[Error] Not found: {src}')
            continue
        convert_split(src, dst, split, workers=args.workers, overwrite=args.overwrite)