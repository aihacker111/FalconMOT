# """
# io.py — MOT result / annotation I/O utilities.

# read_mot_results hỗ trợ:
#   - VisDrone MOT annotations (default khi filename không chứa 'MOT16-'/'MOT17-')
#   - MOT16/17 annotations

# VisDrone GT filter (is_gt=True):
#     Hợp lệ : score==1 AND 1 <= cls_id <= 10
#     Ignore  : score==0 OR cls_id==0 (unlabeled) OR cls_id==11 (others)

# VisDrone ignore filter (is_ignore=True):
#     Chỉ thu thập các row là ignore region (score==0 | cls_id==0 | cls_id==11)
#     để evaluation.py lọc FP nằm trong vùng ignore.

# Class-offset trong GT target_id:
#     VisDrone track_id là per-class (pedestrian id=1 và car id=1 là hai object khác nhau).
#     Khi merge tất cả class vào cùng 1 motmetrics accumulator, offset đảm bảo ID không đụng nhau:
#         global_id = original_track_id + (cls_id_1indexed - 1) * 1_000_000
#     Prediction file dùng offset tương tự (cls_id_0indexed * 1_000_000) nên matching đúng.
# """

# import os
# from typing import Dict

# import numpy as np

# from falconmot.tracking_utils.log import logger


# def write_results(filename, results_dict: Dict, data_type: str):
#     if not filename:
#         return
#     path = os.path.dirname(filename)
#     if path:
#         os.makedirs(path, exist_ok=True)

#     if data_type in ('mot', 'mcmot', 'lab'):
#         save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
#     elif data_type == 'kitti':
#         save_format = ('{frame} {id} pedestrian -1 -1 -10 '
#                        '{x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n')
#     else:
#         raise ValueError(f'Unknown data_type: {data_type}')

#     with open(filename, 'w') as f:
#         for frame_id, frame_data in results_dict.items():
#             if data_type == 'kitti':
#                 frame_id -= 1
#             for tlwh, track_id in frame_data:
#                 if track_id < 0:
#                     continue
#                 x1, y1, w, h = tlwh
#                 x2, y2 = x1 + w, y1 + h
#                 line = save_format.format(
#                     frame=frame_id, id=track_id,
#                     x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
#                 f.write(line)

#     logger.info('Save results to {}'.format(filename))


# def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
#     if data_type in ('mot', 'lab'):
#         return read_mot_results(filename, is_gt, is_ignore)
#     raise ValueError(f'Unknown data type: {data_type}')


# # ---------------------------------------------------------------------------
# # MOT16/17 constants (used only when filename contains 'MOT16-' / 'MOT17-')
# # ---------------------------------------------------------------------------
# _MOT1617_VALID_LABELS  = {1}
# _MOT1617_IGNORE_LABELS = {2, 7, 8, 12}

# # Class-offset multiplier: ensures per-class track IDs are globally unique
# # when all classes are merged into a single motmetrics accumulator.
# _CLS_ID_OFFSET = 1_000_000

# # Evaluation mode skip sets (1-indexed cls_ids to skip from VisDrone GT)
# # 10class: keep all,  5class: AMOT protocol,  4class: competition (person/car/moto/bicycle)
# _EVAL_SKIP_1IDX = {
#     '10class': set(),
#     '5class':  {2, 3, 7, 8, 10},   # skip: people, bicycle, tricycle, awning-tri, motor
#     '4class':  {2, 5, 6, 7, 8, 9}, # skip: people, van, truck, tricycle, awning-tri, bus
# }
# _eval_mode = '10class'


# def set_eval_mode(mode: str):
#     """Set evaluation class subset: '10class' | '5class' | '4class'."""
#     global _eval_mode
#     assert mode in _EVAL_SKIP_1IDX, f'Unknown eval_mode: {mode}'
#     _eval_mode = mode


# def read_mot_results(filename, is_gt, is_ignore):
#     """Parse MOT-format annotation or result file.

#     Returns:
#         results_dict: frame_id → [(tlwh, target_id, score), ...]
#     """
#     results_dict = {}
#     if not os.path.isfile(filename):
#         return results_dict

#     is_mot1617 = 'MOT16-' in filename or 'MOT17-' in filename

#     with open(filename, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split(',')
#             if len(parts) < 6:
#                 continue

#             fid = int(parts[0])
#             if fid < 1:
#                 continue

#             x1, y1, w, h = map(float, parts[2:6])
#             if w <= 0 or h <= 0:
#                 continue

#             target_id = int(parts[1])
#             tlwh      = (x1, y1, w, h)

#             # ── GT or ignore pass ──────────────────────────────────────────
#             if is_gt or is_ignore:
#                 if len(parts) < 8:
#                     continue
#                 ann_score  = int(float(parts[6]))
#                 ann_cls_id = int(float(parts[7]))   # 1-indexed

#                 if is_mot1617:
#                     if is_gt:
#                         if ann_score == 0 or ann_cls_id not in _MOT1617_VALID_LABELS:
#                             continue
#                     else:  # is_ignore
#                         if ann_cls_id not in _MOT1617_IGNORE_LABELS:
#                             continue
#                 else:
#                     # VisDrone: score=0 OR cls_id=0 OR cls_id=11 → ignore region
#                     is_ignore_row = (ann_score == 0 or ann_cls_id == 0 or ann_cls_id == 11)
#                     if is_gt and is_ignore_row:
#                         continue
#                     if is_ignore and not is_ignore_row:
#                         continue

#                     if is_gt:
#                         skip_set = _EVAL_SKIP_1IDX[_eval_mode]
#                         if skip_set and ann_cls_id in skip_set:
#                             continue
#                         # Always offset: VisDrone track IDs are per-class, single accumulator
#                         # needs globally unique IDs across all eval modes.
#                         target_id = target_id + (ann_cls_id - 1) * _CLS_ID_OFFSET

#                 results_dict.setdefault(fid, [])
#                 results_dict[fid].append((tlwh, target_id, 1))

#             # ── Prediction pass ────────────────────────────────────────────
#             else:
#                 pred_score = float(parts[6]) if len(parts) > 6 else 1.0
#                 results_dict.setdefault(fid, [])
#                 results_dict[fid].append((tlwh, target_id, pred_score))

#     return results_dict


# def unzip_objs(objs):
#     if len(objs) > 0:
#         tlwhs, ids, scores = zip(*objs)
#     else:
#         tlwhs, ids, scores = [], [], []
#     tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
#     return tlwhs, ids, scores



"""
io.py — MOT result / annotation I/O utilities.

read_mot_results hỗ trợ:
  - VisDrone MOT annotations (default khi filename không chứa 'MOT16-'/'MOT17-')
  - MOT16/17 annotations

VisDrone GT filter (is_gt=True):
    Hợp lệ : score==1 AND 1 <= cls_id <= 10
    Ignore  : score==0 OR cls_id==0 (unlabeled) OR cls_id==11 (others)

VisDrone ignore filter (is_ignore=True):
    Chỉ thu thập các row là ignore region (score==0 | cls_id==0 | cls_id==11)
    để evaluation.py lọc FP nằm trong vùng ignore.

Class-offset trong GT target_id:
    VisDrone track_id là per-class (pedestrian id=1 và car id=1 là hai object khác nhau).
    Khi merge tất cả class vào cùng 1 motmetrics accumulator, offset đảm bảo ID không đụng nhau:
        global_id = original_track_id + (cls_id_1indexed - 1) * 1_000_000
    Prediction file dùng offset tương tự (cls_id_0indexed * 1_000_000) nên matching đúng.
"""

import os
from typing import Dict

import numpy as np

from falconmot.tracking_utils.log import logger
from falconmot.tracking_utils import class_remap


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if path:
        os.makedirs(path, exist_ok=True)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = ('{frame} {id} pedestrian -1 -1 -10 '
                       '{x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n')
    else:
        raise ValueError(f'Unknown data_type: {data_type}')

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id,
                    x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)

    logger.info('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        return read_mot_results(filename, is_gt, is_ignore)
    raise ValueError(f'Unknown data type: {data_type}')


# ---------------------------------------------------------------------------
# MOT16/17 constants (used only when filename contains 'MOT16-' / 'MOT17-')
# ---------------------------------------------------------------------------
_MOT1617_VALID_LABELS  = {1}
_MOT1617_IGNORE_LABELS = {2, 7, 8, 12}

# Class-offset multiplier: ensures per-class track IDs are globally unique
# when all classes are merged into a single motmetrics accumulator.
_CLS_ID_OFFSET = 1_000_000

# Evaluation mode skip sets (1-indexed cls_ids to skip from VisDrone GT)
# 10class: keep all,  5class: AMOT protocol (skip),  4class: competition (skip)
_EVAL_SKIP_1IDX = {
    '10class': set(),
    '5class':  {2, 3, 7, 8, 10},   # skip: people, bicycle, tricycle, awning-tri, motor
    '4class':  {2, 5, 6, 7, 8, 9}, # skip: people, van, truck, tricycle, awning-tri, bus
}

# eval_mode -> tên profile trong class_remap.py (None = không merge, dùng skip ở trên)
# '5class_merge_benchmark' / '5class_merge_competition': MERGE thay vì skip —
# xem class_remap.py để biết lý do và mapping chi tiết.
_EVAL_MERGE_PROFILE = {
    '10class':                   None,
    '5class':                    None,
    '4class':                    None,
    '5class_merge_benchmark':    '5class_merge_benchmark',
    '5class_merge_competition':  '5class_merge_competition',
}

_eval_mode = '10class'


def set_eval_mode(mode: str):
    """Set evaluation class subset: '10class' | '5class' | '4class' | '5class_merge'."""
    global _eval_mode
    assert mode in _EVAL_MERGE_PROFILE, (
        f'Unknown eval_mode: {mode}. Available: {list(_EVAL_MERGE_PROFILE)}')
    _eval_mode = mode
    class_remap.set_merge_profile(_EVAL_MERGE_PROFILE[mode])


def read_mot_results(filename, is_gt, is_ignore):
    """Parse MOT-format annotation or result file.

    Returns:
        results_dict: frame_id → [(tlwh, target_id, score), ...]
    """
    results_dict = {}
    if not os.path.isfile(filename):
        return results_dict

    is_mot1617 = 'MOT16-' in filename or 'MOT17-' in filename

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue

            fid = int(parts[0])
            if fid < 1:
                continue

            x1, y1, w, h = map(float, parts[2:6])
            if w <= 0 or h <= 0:
                continue

            target_id = int(parts[1])
            tlwh      = (x1, y1, w, h)

            # ── GT or ignore pass ──────────────────────────────────────────
            if is_gt or is_ignore:
                if len(parts) < 8:
                    continue
                ann_score  = int(float(parts[6]))
                ann_cls_id = int(float(parts[7]))   # 1-indexed

                if is_mot1617:
                    if is_gt:
                        if ann_score == 0 or ann_cls_id not in _MOT1617_VALID_LABELS:
                            continue
                    else:  # is_ignore
                        if ann_cls_id not in _MOT1617_IGNORE_LABELS:
                            continue
                else:
                    # VisDrone: score=0 OR cls_id=0 OR cls_id=11 → ignore region
                    is_ignore_row = (ann_score == 0 or ann_cls_id == 0 or ann_cls_id == 11)
                    if is_gt and is_ignore_row:
                        continue
                    if is_ignore and not is_ignore_row:
                        continue

                    if is_gt:
                        merge_profile = class_remap.get_active_profile()
                        if merge_profile is not None:
                            # MERGE mode: nhiều raw class -> cùng 1 target id;
                            # None = profile chủ động drop (không có nhóm tương đương).
                            if merge_profile.remap(ann_cls_id - 1) is None:
                                continue
                        else:
                            # Legacy SKIP mode ('10class' / '5class' / '4class')
                            skip_set = _EVAL_SKIP_1IDX[_eval_mode]
                            if skip_set and ann_cls_id in skip_set:
                                continue
                        # Offset LUÔN theo RAW class (không theo target sau merge):
                        # 2 raw class merge vào cùng target (vd van+truck) vẫn phải
                        # có offset khác nhau, nếu không track_id trùng số giữa 2 raw
                        # class (vd van#3 và truck#3) sẽ bị coi là CÙNG MỘT object.
                        target_id = target_id + (ann_cls_id - 1) * _CLS_ID_OFFSET

                results_dict.setdefault(fid, [])
                results_dict[fid].append((tlwh, target_id, 1))

            # ── Prediction pass ────────────────────────────────────────────
            else:
                pred_score = float(parts[6]) if len(parts) > 6 else 1.0
                results_dict.setdefault(fid, [])
                results_dict[fid].append((tlwh, target_id, pred_score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores
