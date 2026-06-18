"""MOT result / annotation I/O.

Supports VisDrone MOT and MOT16/17 annotation formats.

VisDrone GT filtering (is_gt=True):
    Valid  : score == 1 AND 1 <= cls_id <= 10
    Ignore : score == 0 OR cls_id == 0 (unlabeled) OR cls_id == 11 (others)

Class offset for GT target ids:
    VisDrone track ids are per-class (pedestrian #1 and car #1 are different
    objects). When all classes are merged into a single motmetrics
    accumulator, an offset keeps ids globally unique:
        global_id = original_track_id + (cls_id_1indexed - 1) * 1_000_000
    The prediction file uses the same offset, so matching stays correct.
"""

import os

import numpy as np


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        return read_mot_results(filename, is_gt, is_ignore)
    raise ValueError(f'Unknown data type: {data_type}')


# MOT16/17 label sets (used only when filename contains 'MOT16-' / 'MOT17-')
_MOT1617_VALID_LABELS = {1}
_MOT1617_IGNORE_LABELS = {2, 7, 8, 12}

# Class-offset multiplier: ensures per-class track ids are globally unique
# when all classes are merged into a single motmetrics accumulator.
_CLS_ID_OFFSET = 1_000_000

# Evaluation mode skip sets (1-indexed cls_ids to skip from VisDrone GT).
#   10class: keep all,  5class: AMOT protocol,
#   4class: competition (person / car / moto / bicycle)
_EVAL_SKIP_1IDX = {
    '10class': set(),
    '5class':  {2, 3, 7, 8, 10},     # skip: people, bicycle, tricycle, awning-tri, motor
    '4class':  {2, 5, 6, 7, 8, 9},   # skip: people, van, truck, tricycle, awning-tri, bus
}
_eval_mode = '10class'


def set_eval_mode(mode: str):
    """Set evaluation class subset: '10class' | '5class' | '4class'."""
    global _eval_mode
    assert mode in _EVAL_SKIP_1IDX, f'Unknown eval_mode: {mode}'
    _eval_mode = mode


def read_mot_results(filename, is_gt, is_ignore):
    """Parse a MOT-format annotation or result file.

    Returns:
        dict: frame_id -> [(tlwh, target_id, score), ...]
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
            tlwh = (x1, y1, w, h)

            # --- GT or ignore pass ---
            if is_gt or is_ignore:
                if len(parts) < 8:
                    continue
                ann_score = int(float(parts[6]))
                ann_cls_id = int(float(parts[7]))   # 1-indexed

                if is_mot1617:
                    if is_gt:
                        if ann_score == 0 or ann_cls_id not in _MOT1617_VALID_LABELS:
                            continue
                    elif ann_cls_id not in _MOT1617_IGNORE_LABELS:
                        continue
                else:
                    # VisDrone: score=0 OR cls_id=0 OR cls_id=11 -> ignore region
                    is_ignore_row = (ann_score == 0 or ann_cls_id == 0 or ann_cls_id == 11)
                    if is_gt and is_ignore_row:
                        continue
                    if is_ignore and not is_ignore_row:
                        continue
                    if is_gt:
                        skip_set = _EVAL_SKIP_1IDX[_eval_mode]
                        if skip_set and ann_cls_id in skip_set:
                            continue
                        # VisDrone ids are per-class -> offset for a single accumulator.
                        target_id = target_id + (ann_cls_id - 1) * _CLS_ID_OFFSET

                results_dict.setdefault(fid, [])
                results_dict[fid].append((tlwh, target_id, 1))

            # --- Prediction pass ---
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
