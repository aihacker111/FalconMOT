"""
gen_dataset_visdrone.py — Convert VisDrone2019-MOT annotations to JDE training format.

Annotation format per line:
    frame_id, track_id, x1, y1, w, h, score, cls_id(1-indexed), truncation, occlusion

Filtering rules (consistent with evaluate.py):
    Ignore regions (bôi đen trên ảnh, không phải GT):
        - score == 0          : annotator đánh dấu bỏ qua
        - cls_id == 0         : vùng chưa được gán nhãn
        - cls_id == 11        : danh mục "others" nằm ngoài 10 class
    Valid GT objects (đưa vào label file):
        - score == 1 AND 1 <= cls_id <= 10
        - Không filter theo occlusion  : eval cũng tính tất cả object bị che
        - Không filter theo truncation : giữ lại để tính mAP đầy đủ (nhất quán với evaluate.py)
"""

import os
import copy
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm

cls2id = {
    'pedestrian':       0,
    'people':           1,
    'bicycle':          2,
    'car':              3,
    'van':              4,
    'truck':            5,
    'tricycle':         6,
    'awning-tricycle':  7,
    'bus':              8,
    'motor':            9,
}

id2cls = {v: k for k, v in cls2id.items()}


def draw_ignore_regions(img, boxes):
    """Bôi đen các vùng ignore trên ảnh (in-place).

    boxes: list of [x1, y1, w, h] (pixel, integer OK)
    """
    if img is None:
        print('[Err]: Input image is None!')
        return -1
    for box in boxes:
        x, y, w, h = [int(v + 0.5) for v in box]
        img[y: y + h, x: x + w] = 0
    return img


def gen_dot_train_file(data_root, rel_path, out_root, f_name='detrac.train'):
    """Tạo file index liệt kê đường dẫn ảnh tương đối."""
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('[Err]: invalid root')
        return

    out_f_path = os.path.join(out_root, f_name)
    cnt = 0
    with open(out_f_path, 'w') as f:
        root = data_root + rel_path
        seqs = sorted(os.listdir(root))
        for seq in tqdm(seqs):
            img_dir = os.path.join(root, seq)
            rel_dir = rel_path + '/' + seq
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(img_dir, img_name)
                    if os.path.isfile(img_path):
                        item = (rel_dir + '/' + img_name).replace(data_root + '/', '')
                        f.write(item + '\n')
                        cnt += 1

    print('Total {:d} images'.format(cnt))


def gen_track_dataset(src_root, dst_root, viz_root=None):
    """Convert VisDrone MOT split sang định dạng JDE (images + labels_with_ids).

    Args:
        src_root: thư mục gốc VisDrone MOT split
                  (phải có sequences/ và annotations/ bên trong)
        dst_root: thư mục đầu ra (tạo mới nếu chưa có)
        viz_root: nếu set, lưu ảnh visualization có vẽ bbox
    """
    if not os.path.isdir(src_root):
        print('[Err]: invalid src dir:', src_root)
        return

    dst_img_root = os.path.join(dst_root, 'images')
    dst_txt_root = os.path.join(dst_root, 'labels_with_ids')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_txt_root, exist_ok=True)

    # global track-id offset mỗi class (tích luỹ qua các seq)
    track_start_id_dict = {cls_id: 0 for cls_id in id2cls}

    frame_cnt = 0
    seq_names = sorted(os.listdir(os.path.join(src_root, 'sequences')))

    for seq in tqdm(seq_names, desc='sequences'):
        print('Processing {}:'.format(seq))

        seq_img_dir    = os.path.join(src_root, 'sequences', seq)
        seq_ann_path   = os.path.join(src_root, 'annotations', seq + '.txt')
        dst_seq_img_dir = os.path.join(dst_img_root, seq)
        dst_seq_txt_dir = os.path.join(dst_txt_root, seq)

        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_path)):
            print('[Warning]: missing img dir or annotation file for', seq)
            continue

        os.makedirs(dst_seq_img_dir, exist_ok=True)
        os.makedirs(dst_seq_txt_dir, exist_ok=True)

        # ── Parse annotation ──────────────────────────────────────────────────
        with open(seq_ann_path, 'r', encoding='utf-8') as f_r:
            label_lines = f_r.readlines()

        n_lines = len(label_lines)
        seq_label_array = np.zeros((n_lines, 10), dtype=np.int32)
        for i, line in enumerate(label_lines):
            parts = [int(x) for x in line.strip().split(',')]
            seq_label_array[i, :len(parts)] = parts

        # Ignore regions: score=0 OR cls_id=0 (unlabeled) OR cls_id=11 (others)
        # Cả ba trường hợp đều bị bôi đen trên ảnh và không đưa vào GT.
        is_ignore = (
            (seq_label_array[:, 6] == 0) |   # score == 0
            (seq_label_array[:, 7] == 0)  |   # cls_id == 0  (unlabeled)
            (seq_label_array[:, 7] == 11)      # cls_id == 11 (others)
        )

        # Valid objects: score=1 AND cls_id 1-10
        # Không filter occlusion và truncation để:
        #   1. Model học detect cả object bị che/bị cắt
        #   2. evaluate.py cũng không filter → mAP nhất quán
        is_valid = (
            (seq_label_array[:, 6] == 1) &
            (seq_label_array[:, 7] > 0)  &
            (seq_label_array[:, 7] < 11)
        )

        seq_ignore_box_dict = defaultdict(list)
        seq_objs_label_dict = defaultdict(list)

        for label in seq_label_array[is_ignore]:
            frame_id = label[0]
            seq_ignore_box_dict[frame_id].append(label[2:6])  # [x1,y1,w,h]

        for label in seq_label_array[is_valid]:
            frame_id = label[0]
            seq_objs_label_dict[frame_id].append(label)

        # ── Build per-class track-id lookup for this seq ──────────────────────
        tmp_ids_dict = defaultdict(set)
        for fr_labels in seq_objs_label_dict.values():
            for label in fr_labels:
                cls_id    = int(label[7]) - 1   # 1-indexed → 0-indexed
                target_id = int(label[1])
                tmp_ids_dict[cls_id].add(target_id)

        seq_cls_target_ids_dict = {}
        seq_max_tra_id_dict = {k: 0 for k in id2cls}
        for cls_id, ids in tmp_ids_dict.items():
            sorted_ids = sorted(ids)
            seq_cls_target_ids_dict[cls_id] = sorted_ids
            seq_max_tra_id_dict[cls_id]     = len(sorted_ids)

        for k in id2cls:
            print('  {} max_track_id={:d}  start_id={:d}'.format(
                id2cls[k], seq_max_tra_id_dict[k], track_start_id_dict[k]))

        # ── Per-frame processing ───────────────────────────────────────────────
        for fr_id, fr_labels in seq_objs_label_dict.items():
            fr_name = '{:07d}.jpg'.format(fr_id)
            fr_path = os.path.join(seq_img_dir, fr_name)
            if not os.path.isfile(fr_path):
                print('[Err]: missing frame', fr_path)
                continue

            img = cv2.imread(fr_path, cv2.IMREAD_COLOR)
            if img is None:
                print('[Err]: empty image', fr_path)
                continue

            H, W, _ = img.shape

            # Bôi đen ignore regions (cls_id=0, cls_id=11, score=0)
            draw_ignore_regions(img, seq_ignore_box_dict.get(fr_id, []))

            dst_img_path = os.path.join(dst_seq_img_dir, fr_name)
            if not os.path.isfile(dst_img_path):
                cv2.imwrite(dst_img_path, img)

            # Visualization
            if viz_root is not None:
                viz_dir = os.path.join(viz_root, seq)
                os.makedirs(viz_dir, exist_ok=True)
                img_viz = copy.deepcopy(img)

            # ── Generate label file ────────────────────────────────────────
            fr_label_strs = []
            for label in fr_labels:
                cls_id    = int(label[7]) - 1   # 0-indexed
                target_id = int(label[1])

                # remap raw track_id → global sequential id (per class)
                local_rank = seq_cls_target_ids_dict[cls_id].index(target_id)
                track_id   = local_rank + 1 + track_start_id_dict[cls_id]

                bbox_left   = float(label[2])
                bbox_top    = float(label[3])
                bbox_width  = float(label[4])
                bbox_height = float(label[5])

                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                if viz_root is not None:
                    pt1 = (int(bbox_left + 0.5), int(bbox_top + 0.5))
                    pt2 = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))
                    cv2.rectangle(img_viz, pt1, pt2, (0, 255, 0), 2)
                    label_text = '{} id:{}'.format(id2cls[cls_id], track_id)
                    cv2.putText(img_viz, label_text, (pt1[0], pt1[1] - 4),
                                cv2.FONT_HERSHEY_PLAIN, 1.1, (225, 255, 255), 1)

                # Normalize to [0, 1] in letterbox-original space
                cx = (bbox_left + bbox_width  * 0.5) / W
                cy = (bbox_top  + bbox_height * 0.5) / H
                bw = bbox_width  / W
                bh = bbox_height / H

                fr_label_strs.append(
                    '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        cls_id, track_id, cx, cy, bw, bh))

            if viz_root is not None:
                cv2.imwrite(os.path.join(viz_dir, fr_name), img_viz)

            label_f_path = os.path.join(dst_seq_txt_dir, fr_name.replace('.jpg', '.txt'))
            with open(label_f_path, 'w', encoding='utf-8') as f_w:
                f_w.writelines(fr_label_strs)

            frame_cnt += 1

        # Cập nhật global track-id offset sau mỗi seq
        for cls_id in id2cls:
            track_start_id_dict[cls_id] += seq_max_tra_id_dict[cls_id]

        print('Done: {}\n'.format(seq))

    print('Total {:d} frames converted.'.format(frame_cnt))


if __name__ == '__main__':
    DATASET_ROOT   = '/workspace'
    CONVERTED_ROOT = '/workspace/VisDrone2019'

    # Step 1: Convert train set → CONVERTED_ROOT/train/images/ + labels_with_ids/
    gen_track_dataset(
        src_root=f'{DATASET_ROOT}/VisDrone2019-MOT-train',
        dst_root=f'{CONVERTED_ROOT}/train',
        viz_root=None,
    )

    # Step 2: Convert val set → CONVERTED_ROOT/val/images/ + labels_with_ids/
    gen_track_dataset(
        src_root=f'{DATASET_ROOT}/VisDrone2019-MOT-val',
        dst_root=f'{CONVERTED_ROOT}/val',
        viz_root=None,
    )

    # Step 3: Generate index files (.train / .val)
    gen_dot_train_file(
        data_root=f'{CONVERTED_ROOT}/',
        rel_path='train/images',
        out_root=CONVERTED_ROOT,
        f_name='VisDrone.train',
    )
    gen_dot_train_file(
        data_root=f'{CONVERTED_ROOT}/',
        rel_path='val/images',
        out_root=CONVERTED_ROOT,
        f_name='VisDrone.val',
    )
