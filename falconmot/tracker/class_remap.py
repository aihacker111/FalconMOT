# """class_remap.py — Remap model output class ids between training schema and eval schema.

# Model trains on 7 classes (visdrone2coco_7cls_mot.py):
#     0: pedestrian   (merged: ped + people)
#     1: bicycle
#     2: car
#     3: truck        (merged: van + truck)
#     4: tricycle     (merged: tricycle + awning-tricycle)
#     5: bus
#     6: motor

# Benchmark eval uses 5 classes (visdrone2coco_5cls_benchmark_mot.py):
#     0: pedestrian
#     1: car
#     2: truck
#     3: tricycle
#     4: bus

# Drop: bicycle (7cls idx 1), motor (7cls idx 6).

# Usage in track_ECDet.py:
#     from falconmot.tracker.class_remap import remap_dets_7cls_to_5cls, CLS5_NAMES

#     dets_remapped = remap_dets_7cls_to_5cls(dets)   # dict[0..6] -> dict[0..4]
# """

# from typing import Dict, List

# # ── 7cls model output (0-indexed) ────────────────────────────────────────────
# CLS7_NAMES = {
#     0: 'pedestrian',
#     1: 'bicycle',
#     2: 'car',
#     3: 'truck',
#     4: 'tricycle',
#     5: 'bus',
#     6: 'motor',
# }

# # ── 5cls benchmark GT (0-indexed) ────────────────────────────────────────────
# CLS5_NAMES = {
#     0: 'pedestrian',
#     1: 'car',
#     2: 'truck',
#     3: 'tricycle',
#     4: 'bus',
# }

# # 7cls 0-idx -> 5cls 0-idx  (None = drop)
# _REMAP_7_TO_5 = {
#     0: 0,    # pedestrian -> pedestrian
#     1: None, # bicycle    -> DROP
#     2: 1,    # car        -> car
#     3: 2,    # truck      -> truck
#     4: 3,    # tricycle   -> tricycle
#     5: 4,    # bus        -> bus
#     6: None, # motor      -> DROP
# }

# NUM_CLS_TRAIN = 7   # model head output channels
# NUM_CLS_EVAL  = 5   # benchmark GT classes


# def remap_dets_7cls_to_5cls(dets: Dict[int, List]) -> Dict[int, List]:
#     """Filter & remap model detections from 7-class space to 5-class eval space.

#     Args:
#         dets: dict[cls_id_7 (0-indexed)] -> list[MCTrack]
#               (output of ECDetSequenceRunner._decode_detections)

#     Returns:
#         dict[cls_id_5 (0-indexed)] -> list[MCTrack], with MCTrack.cls_id updated.
#     """
#     out: Dict[int, List] = {}
#     for cls7, track_list in dets.items():
#         cls5 = _REMAP_7_TO_5.get(cls7)
#         if cls5 is None:
#             continue   # drop bicycle / motor
#         remapped = []
#         for t in track_list:
#             t.cls_id = cls5   # in-place patch; MCTrack.cls_id drives track-id offset
#             remapped.append(t)
#         out[cls5] = out.get(cls5, []) + remapped
#     return out





# # ── Skip set for ECDetSequenceRunner (5-class eval, model has 7 output classes)
# # After remap the runner only sees indices 0..4 — no need to skip anything.
# # Kept here for documentation; track_ECDet imports this directly.
# SKIP_SET_AFTER_REMAP: set = set()



"""class_remap.py — Remap model output class ids between training schema and eval schema.

Model trains on 7 classes (0-indexed internally):
    0: pedestrian   (merged: ped + people)
    1: bicycle
    2: car
    3: van
    4: truck
    5: bus
    6: motor

Benchmark eval uses 5 classes (0-indexed internally):
    0: pedestrian
    1: car
    2: van
    3: truck
    4: bus

Drop: bicycle (7cls idx 1), motor (7cls idx 6).

Usage in track_ECDet.py:
    from falconmot.tracker.class_remap import remap_dets_7cls_to_5cls, CLS5_NAMES

    dets_remapped = remap_dets_7cls_to_5cls(dets)   # dict[0..6] -> dict[0..4]
"""

from typing import Dict, List

# ── 7cls model output (0-indexed) ────────────────────────────────────────────
CLS7_NAMES = {
    0: 'pedestrian',
    1: 'bicycle',
    2: 'car',
    3: 'van',
    4: 'truck',
    5: 'bus',
    6: 'motor',
}

# ── 5cls benchmark GT (0-indexed) ────────────────────────────────────────────
CLS5_NAMES = {
    0: 'pedestrian',
    1: 'car',
    2: 'van',
    3: 'truck',
    4: 'bus',
}

# 7cls 0-idx -> 5cls 0-idx  (None = drop)
_REMAP_7_TO_5 = {
    0: 0,    # pedestrian -> pedestrian
    1: None, # bicycle    -> DROP
    2: 1,    # car        -> car
    3: 2,    # van        -> van
    4: 3,    # truck      -> truck
    5: 4,    # bus        -> bus
    6: None, # motor      -> DROP
}

NUM_CLS_TRAIN = 7   # model head output channels
NUM_CLS_EVAL  = 5   # benchmark GT classes


def remap_dets_7cls_to_5cls(dets: Dict[int, List]) -> Dict[int, List]:
    """Filter & remap model detections from 7-class space to 5-class eval space.

    Args:
        dets: dict[cls_id_7 (0-indexed)] -> list[MCTrack]
              (output of ECDetSequenceRunner._decode_detections)

    Returns:
        dict[cls_id_5 (0-indexed)] -> list[MCTrack], with MCTrack.cls_id updated.
    """
    out: Dict[int, List] = {}
    for cls7, track_list in dets.items():
        cls5 = _REMAP_7_TO_5.get(cls7)
        if cls5 is None:
            continue   # drop bicycle / motor
        
        remapped = []
        for t in track_list:
            t.cls_id = cls5   # in-place patch; MCTrack.cls_id drives track-id offset
            remapped.append(t)
            
        out[cls5] = out.get(cls5, []) + remapped
    return out


# ── Skip set for ECDetSequenceRunner (5-class eval, model has 7 output classes)
# After remap the runner only sees indices 0..4 — no need to skip anything.
# Kept here for documentation; track_ECDet imports this directly.
SKIP_SET_AFTER_REMAP: set = set()