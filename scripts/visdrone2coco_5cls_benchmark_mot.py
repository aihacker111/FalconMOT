# """
# gen_dataset_visdrone_coco_5cls.py — Convert VisDrone2019-MOT to COCO JSON,
# MERGED into the 5-class benchmark scheme:

#     pedestrian / car / truck / tricycle / bus   (DROP bicycle + motor)

# Mapping (khớp eval_mode='5class_merge_benchmark' trong opts.py/track.py):
#     pedestrian(1) + people(2)            → pedestrian (1)
#     car(4)                               → car        (2)
#     van(5)    + truck(6)                 → truck      (3)
#     tricycle(7) + awning-tricycle(8)     → tricycle   (4)
#     bus(9)                               → bus         (5)
#     bicycle(3), motor(10)                → DROP

# Hai lớp bị drop (bicycle, motor) mặc định được BLACK-OUT khỏi ảnh (giống
# ignore regions) để detector không bị phạt vì vật thể không nhãn. Dùng
# --no_blackout_dropped nếu muốn giữ nguyên pixel (annotation vẫn bị bỏ).

# Output structure:
#     OUTPUT_ROOT/
#     ├── train/
#     │   ├── images/
#     │   └── annotations/instances_train.json
#     └── val/
#         ├── images/
#         └── annotations/instances_val.json
# """

# import os
# import json
# import argparse
# import numpy as np
# import cv2
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm


# # ── Class mapping ────────────────────────────────────────────────────────────
# # VisDrone gốc (1-indexed):
# #   1 pedestrian  2 people   3 bicycle  4 car         5 van
# #   6 truck       7 tricycle 8 awning-tricycle  9 bus 10 motor

# # Lớp đích (1-indexed) sau khi merge.
# TARGET_CLS_MAP = {
#     1: 'pedestrian',
#     2: 'car',
#     3: 'truck',
#     4: 'tricycle',
#     5: 'bus',
# }
# NUM_CLASSES = len(TARGET_CLS_MAP)   # 5

# # raw VisDrone 1-indexed class → target 1-indexed class (None = DROP).
# CLASS_MAPPING = {
#     1: 1,      # pedestrian        → pedestrian
#     2: 1,      # people            → pedestrian
#     3: None,   # bicycle           → DROP
#     4: 2,      # car               → car
#     5: 3,      # van               → truck
#     6: 3,      # truck             → truck
#     7: 4,      # tricycle          → tricycle
#     8: 4,      # awning-tricycle   → tricycle
#     9: 5,      # bus               → bus
#     10: None,  # motor             → DROP
# }


# def get_categories():
#     return [{'id': k, 'name': v, 'supercategory': 'object'}
#             for k, v in TARGET_CLS_MAP.items()]


# # ── Helpers ──────────────────────────────────────────────────────────────────

# def _parse_ann_file(path: str) -> np.ndarray:
#     """Fast annotation parse using numpy. Returns (N, 10) int32 array."""
#     with open(path, 'r') as f:
#         raw = f.read()
#     if not raw.strip():
#         return np.zeros((0, 10), dtype=np.int32)
#     # Replace commas with spaces so np.fromstring can parse
#     data = np.fromstring(raw.replace(',', ' '), dtype=np.int32, sep=' ')
#     cols = 10
#     rows = len(data) // cols
#     return data[: rows * cols].reshape(rows, cols)


# def _process_frame(src_path: str, dst_path: str, ignore_boxes: list,
#                    overwrite: bool = False) -> tuple:
#     """Read src frame, black-out ignore regions, write dst. Returns (H, W)."""
#     img = cv2.imread(src_path)
#     if img is None:
#         return None
#     for box in ignore_boxes:
#         x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#         img[y: y + h, x: x + w] = 0
#     if overwrite or not os.path.isfile(dst_path):
#         cv2.imwrite(dst_path, img)
#     return img.shape[0], img.shape[1]


# # ── Core converter ────────────────────────────────────────────────────────────

# def convert_split(src_root: str, dst_root: str, split: str,
#                   workers: int = 8, blackout_dropped: bool = True,
#                   overwrite: bool = False) -> None:

#     seq_dir = os.path.join(src_root, 'sequences')
#     ann_dir = os.path.join(src_root, 'annotations')

#     dst_img_root = os.path.join(dst_root, 'images')
#     dst_ann_dir  = os.path.join(dst_root, 'annotations')
#     os.makedirs(dst_img_root, exist_ok=True)
#     os.makedirs(dst_ann_dir,  exist_ok=True)

#     images_list = []
#     anns_list   = []

#     img_id = 0
#     ann_id = 0

#     track_start = [0] * NUM_CLASSES   # global per-class ID offset across sequences
#     total_dropped = 0

#     seq_names = sorted(os.listdir(seq_dir))

#     for seq in tqdm(seq_names, desc=f'[{split}]'):
#         seq_img_dir  = os.path.join(seq_dir, seq)
#         seq_ann_file = os.path.join(ann_dir, seq + '.txt')
#         if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_file)):
#             print(f'  [skip] {seq}')
#             continue

#         dst_seq_dir = os.path.join(dst_img_root, seq)
#         os.makedirs(dst_seq_dir, exist_ok=True)

#         # ── 1. Parse annotations ─────────────────────────────────────────
#         arr = _parse_ann_file(seq_ann_file)
#         if arr.shape[0] == 0:
#             continue

#         is_ignore = (arr[:, 6] == 0) | (arr[:, 7] == 0) | (arr[:, 7] == 11)
#         is_valid  = (arr[:, 6] == 1) & (arr[:, 7] > 0)  & (arr[:, 7] < 11)

#         ignore_by_frame: dict[int, list] = defaultdict(list)
#         valid_by_frame:  dict[int, list] = defaultdict(list)

#         # Ignore regions thật → luôn black-out.
#         for row in arr[is_ignore]:
#             ignore_by_frame[int(row[0])].append(row[2:6])

#         # Tách object hợp lệ thành: giữ (kept) vs drop (bicycle/motor).
#         seq_dropped = 0
#         for row in arr[is_valid]:
#             raw_cls_1 = int(row[7])
#             tgt = CLASS_MAPPING[raw_cls_1]
#             if tgt is None:                       # bicycle / motor → DROP
#                 seq_dropped += 1
#                 if blackout_dropped:
#                     ignore_by_frame[int(row[0])].append(row[2:6])
#                 continue
#             valid_by_frame[int(row[0])].append(row)
#         total_dropped += seq_dropped

#         # ── 2. Build track-id rank dict an toàn bằng tuple (raw_cls, raw_tid) ──
#         # Tuple key giữ van#3 ≠ truck#3 dù cả hai cùng merge vào 'truck',
#         # tránh hai object khác nhau bị gán chung 1 track_id.
#         track_ids_per_cls: dict[int, set] = defaultdict(set)
#         for rows in valid_by_frame.values():
#             for row in rows:
#                 raw_cls_1 = int(row[7])
#                 raw_tid   = int(row[1])
#                 new_cls_0 = CLASS_MAPPING[raw_cls_1] - 1
#                 track_ids_per_cls[new_cls_0].add((raw_cls_1, raw_tid))

#         rank_map = {
#             cls_id: {key: rank for rank, key in enumerate(sorted(keys))}
#             for cls_id, keys in track_ids_per_cls.items()
#         }
#         seq_n_ids = {cls_id: len(keys) for cls_id, keys in track_ids_per_cls.items()}

#         tqdm.write(f'  {seq}: ' + '  '.join(
#             f'{TARGET_CLS_MAP[c + 1]}={n}' for c, n in sorted(seq_n_ids.items()))
#             + (f'   [drop={seq_dropped}]' if seq_dropped else ''))

#         # ── 3. Parallel image I/O ─────────────────────────────────────────
#         frame_ids   = sorted(valid_by_frame.keys())
#         frame_sizes: dict[int, tuple] = {}

#         def _job(fr_id: int):
#             fr_name  = f'{fr_id:07d}.jpg'
#             src_path = os.path.join(seq_img_dir, fr_name)
#             dst_path = os.path.join(dst_seq_dir, fr_name)
#             hw = _process_frame(src_path, dst_path,
#                                 ignore_by_frame.get(fr_id, []), overwrite)
#             return fr_id, hw

#         with ThreadPoolExecutor(max_workers=workers) as pool:
#             futs = {pool.submit(_job, fid): fid for fid in frame_ids}
#             for fut in as_completed(futs):
#                 fr_id, hw = fut.result()
#                 if hw is not None:
#                     frame_sizes[fr_id] = hw

#         # ── 4. Build COCO records (sequential, deterministic order) ──────
#         for fr_id in frame_ids:
#             if fr_id not in frame_sizes:
#                 continue
#             H, W = frame_sizes[fr_id]
#             fr_name  = f'{fr_id:07d}.jpg'
#             rel_path = f'{seq}/{fr_name}'

#             images_list.append({
#                 'id':        img_id,
#                 'file_name': rel_path,
#                 'height':    H,
#                 'width':     W,
#                 'seq_id':    seq,
#                 'frame_id':  fr_id,
#             })

#             for row in valid_by_frame[fr_id]:
#                 raw_cls_1 = int(row[7])
#                 raw_tid   = int(row[1])
#                 new_cls_1 = CLASS_MAPPING[raw_cls_1]
#                 new_cls_0 = new_cls_1 - 1

#                 x1 = float(row[2]);  y1 = float(row[3])
#                 bw = float(row[4]);  bh = float(row[5])
#                 if bw <= 0 or bh <= 0:
#                     continue

#                 local_rank = rank_map[new_cls_0][(raw_cls_1, raw_tid)]
#                 track_id   = local_rank + track_start[new_cls_0]

#                 anns_list.append({
#                     'id':          ann_id,
#                     'image_id':    img_id,
#                     'category_id': new_cls_1,
#                     'bbox':        [x1, y1, bw, bh],
#                     'area':        bw * bh,
#                     'iscrowd':     0,
#                     'track_id':    track_id,
#                 })
#                 ann_id += 1

#             img_id += 1

#         # Advance global track-id offsets
#         for cls_id in range(NUM_CLASSES):
#             track_start[cls_id] += seq_n_ids.get(cls_id, 0)

#     # ── 5. Write JSON ─────────────────────────────────────────────────────
#     coco = {
#         'images':      images_list,
#         'annotations': anns_list,
#         'categories':  get_categories(),
#     }
#     out_json = os.path.join(dst_ann_dir, f'instances_{split}.json')
#     with open(out_json, 'w') as f:
#         json.dump(coco, f, separators=(',', ':'))

#     # ── 6. Stats ──────────────────────────────────────────────────────────
#     print(f'\n[{split}] {len(images_list):,} images  {len(anns_list):,} annotations')
#     print(f'  [Info] Merged into {NUM_CLASSES} classes: '
#           + ' / '.join(TARGET_CLS_MAP.values()))
#     print(f'  [Info] Dropped {total_dropped:,} bicycle/motor objects'
#           + (' (blacked out)' if blackout_dropped else ' (pixels kept)'))
#     print(f'  → {out_json}')

#     max_tid: dict[int, int] = defaultdict(int)
#     for ann in anns_list:
#         c = ann['category_id'] - 1
#         if ann['track_id'] + 1 > max_tid[c]:
#             max_tid[c] = ann['track_id'] + 1

#     print('  Track IDs generated:')
#     for c in sorted(max_tid):
#         print(f'    {TARGET_CLS_MAP[c + 1]}: {max_tid[c]} unique object IDs')


# # ── CLI ───────────────────────────────────────────────────────────────────────

# def main():
#     ap = argparse.ArgumentParser(
#         description='Convert VisDrone2019-MOT → COCO JSON, 5-class merge '
#                     '(pedestrian/car/truck/tricycle/bus, drop bicycle+motor).')
#     ap.add_argument('--visdrone_root', default='/workspace')
#     ap.add_argument('--output_root',   default='/workspace/VisDrone2019-COCO-5cls')
#     ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test-dev'],
#                     choices=['train', 'val', 'test-dev'])
#     ap.add_argument('--workers', type=int, default=8,
#                     help='parallel threads for image I/O per sequence')
#     ap.add_argument('--no_blackout_dropped', action='store_true',
#                     help='KHÔNG black-out vùng bicycle/motor (giữ nguyên pixel; '
#                          'annotation vẫn bị drop)')
#     ap.add_argument('--overwrite', action='store_true',
#                     help='ghi đè ảnh đã tồn tại (cần khi đổi chế độ blackout giữa '
#                          'các lần chạy — mặc định bỏ qua ảnh đã có)')
#     args = ap.parse_args()

#     for split in args.splits:
#         src = os.path.join(args.visdrone_root, f'VisDrone2019-MOT-{split}')
#         dst = os.path.join(args.output_root, split)
#         if not os.path.isdir(src):
#             print(f'[Error] Not found: {src}')
#             continue
#         convert_split(src, dst, split,
#                       workers=args.workers,
#                       blackout_dropped=not args.no_blackout_dropped,
#                       overwrite=args.overwrite)


# if __name__ == '__main__':
#     main()



"""
gen_dataset_visdrone_coco_5cls.py
Chuyển đổi VisDrone2019-MOT sang COCO JSON, merge về 5 class chuẩn:
    1: pedestrian, 2: car, 3: truck, 4: tricycle, 5: bus
    (Drop hoàn toàn: bicycle, motor)

Logic: Không bôi đen (blackout) các class bị drop, chỉ bôi đen vùng ignore thực sự.
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Class mapping 5 class ────────────────────────────────────────────────────
TARGET_CLS_MAP = {
    1: 'pedestrian', 2: 'car', 3: 'truck', 4: 'tricycle', 5: 'bus'
}
NUM_CLASSES = len(TARGET_CLS_MAP)

# Mapping từ 10 class gốc VisDrone -> 5 class target (None = DROP)
CLASS_MAPPING = {
    1: 1, 2: 1,      # pedestrian, people -> 1
    3: None,         # bicycle -> DROP
    4: 2,            # car -> 2
    5: 3, 6: 3,      # van, truck -> 3
    7: 4, 8: 4,      # tricycle, awning-tricycle -> 4
    9: 5,            # bus -> 5
    10: None         # motor -> DROP
}

def get_categories():
    return [{'id': k, 'name': v, 'supercategory': 'object'} for k, v in TARGET_CLS_MAP.items()]

def _parse_ann_file(path: str) -> np.ndarray:
    with open(path, 'r') as f: raw = f.read()
    if not raw.strip(): return np.zeros((0, 10), dtype=np.int32)
    data = np.fromstring(raw.replace(',', ' '), dtype=np.int32, sep=' ')
    return data.reshape(-1, 10)

def _process_frame(src_path: str, dst_path: str, ignore_boxes: list, overwrite: bool) -> tuple:
    img = cv2.imread(src_path)
    if img is None: return None
    # CHỈ bôi đen các vùng ignore thực sự (ví dụ vùng mờ, vùng lỗi)
    for box in ignore_boxes:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img[y:y+h, x:x+w] = 0
    if overwrite or not os.path.isfile(dst_path):
        cv2.imwrite(dst_path, img)
    return img.shape[0], img.shape[1]

def convert_split(src_root, dst_root, split, workers=8, overwrite=False):
    seq_dir, ann_dir = os.path.join(src_root, 'sequences'), os.path.join(src_root, 'annotations')
    dst_img_root, dst_ann_dir = os.path.join(dst_root, 'images'), os.path.join(dst_root, 'annotations')
    os.makedirs(dst_img_root, exist_ok=True); os.makedirs(dst_ann_dir, exist_ok=True)

    images_list, anns_list = [], []
    img_id, ann_id = 0, 0
    track_start = [0] * NUM_CLASSES
    
    for seq in tqdm(sorted(os.listdir(seq_dir)), desc=f'[{split}]'):
        seq_img_dir = os.path.join(seq_dir, seq)
        seq_ann_file = os.path.join(ann_dir, seq + '.txt')
        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_file)): continue

        dst_seq_dir = os.path.join(dst_img_root, seq)
        os.makedirs(dst_seq_dir, exist_ok=True)

        arr = _parse_ann_file(seq_ann_file)
        if arr.shape[0] == 0: continue

        is_ignore = (arr[:, 6] == 0) | (arr[:, 7] == 0) | (arr[:, 7] == 11)
        is_valid = (arr[:, 6] == 1) & (arr[:, 7] > 0) & (arr[:, 7] < 11)

        ignore_by_frame = defaultdict(list)
        for row in arr[is_ignore]: ignore_by_frame[int(row[0])].append(row[2:6])

        valid_by_frame = defaultdict(list)
        for row in arr[is_valid]:
            tgt = CLASS_MAPPING[int(row[7])]
            if tgt is not None: valid_by_frame[int(row[0])].append(row)

        track_ids_per_cls = defaultdict(set)
        for rows in valid_by_frame.values():
            for row in rows: track_ids_per_cls[CLASS_MAPPING[int(row[7])]-1].add((int(row[7]), int(row[1])))
        
        rank_map = {c: {key: r for r, key in enumerate(sorted(keys))} for c, keys in track_ids_per_cls.items()}

        frame_ids = sorted(valid_by_frame.keys())
        frame_sizes = {}
        
        def _job(fr_id):
            src_path = os.path.join(seq_img_dir, f'{fr_id:07d}.jpg')
            dst_path = os.path.join(dst_seq_dir, f'{fr_id:07d}.jpg')
            return fr_id, _process_frame(src_path, dst_path, ignore_by_frame.get(fr_id, []), overwrite)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fr_id, hw in [fut.result() for fut in as_completed([pool.submit(_job, fid) for fid in frame_ids])]:
                if hw: frame_sizes[fr_id] = hw

        for fr_id in frame_ids:
            if fr_id not in frame_sizes: continue
            images_list.append({'id': img_id, 'file_name': f'{seq}/{fr_id:07d}.jpg', 'height': frame_sizes[fr_id][0], 'width': frame_sizes[fr_id][1]})
            for row in valid_by_frame[fr_id]:
                new_cls = CLASS_MAPPING[int(row[7])]
                anns_list.append({
                    'id': ann_id, 'image_id': img_id, 'category_id': new_cls,
                    'bbox': [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                    'area': float(row[4]*row[5]), 'iscrowd': 0,
                    'track_id': rank_map[new_cls-1][(int(row[7]), int(row[1]))] + track_start[new_cls-1]
                })
                ann_id += 1
            img_id += 1
        
        for c in range(NUM_CLASSES): track_start[c] += len(track_ids_per_cls[c])

    with open(os.path.join(dst_ann_dir, f'instances_{split}.json'), 'w') as f:
        json.dump({'images': images_list, 'annotations': anns_list, 'categories': get_categories()}, f)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--visdrone_root', default='/workspace/VisDrone2019')
    ap.add_argument('--output_root', default='/workspace/VisDrone2019-COCO-5cls')
    args = ap.parse_args()
    for split in ['train', 'val', 'test-dev']:
        convert_split(os.path.join(args.visdrone_root, f'VisDrone2019-MOT-{split}'), os.path.join(args.output_root, split), split)