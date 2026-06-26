"""coco_gt_reader.py — Build GT frame dicts directly from COCO JSON (5cls benchmark).

Replaces io.read_mot_results when GT comes from instances_test-dev.json
(visdrone2coco_5cls_benchmark_mot.py output) instead of raw VisDrone .txt files.

GT COCO JSON already has:
  * Only 5 valid classes (bicycle + motor already dropped at gen time).
  * seq_id + frame_id per image record.
  * track_id globally unique per class (with _CLS_ID_OFFSET already baked in
    by the gen script via rank_map + track_start).

So we do NOT need to re-apply cls_id offset here — just pass track_id through.
The prediction side (track_ECDet) still applies the same offset via cls_id *
_CLS_ID_OFFSET when writing the result .txt, so matching stays consistent.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Must match _CLS_ID_OFFSET in track_ECDet.py and io.py
_CLS_ID_OFFSET = 1_000_000


def _global_track_id(track_id: int, category_id_1idx: int) -> int:
    """Make track_id globally unique across classes (same formula as io.py)."""
    return track_id + (category_id_1idx - 1) * _CLS_ID_OFFSET


def load_coco_gt_for_seq(
    ann_file: str,
    seq_id: str,
) -> Tuple[Dict[int, List], Dict[int, List]]:
    """Load GT and ignore dicts for one sequence from a COCO JSON file.

    Returns:
        gt_frame_dict     : frame_id -> [(tlwh, global_track_id, 1), ...]
        ignore_frame_dict : frame_id -> [(tlwh, -1, 1), ...]   (always empty for
                            5cls COCO — ignore regions are baked into pixels, not
                            stored as annotations; kept for Evaluator API compat)
    """
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # Index: image_id -> (frame_id, seq_id)
    img_meta: Dict[int, Tuple[int, str]] = {}
    for img in coco['images']:
        sid = img.get('seq_id') or os.path.dirname(img['file_name']) or '_root'
        img_meta[img['id']] = (int(img.get('frame_id', 0)), sid)

    gt_frame_dict: Dict[int, List] = defaultdict(list)

    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_meta:
            continue
        frame_id, sid = img_meta[img_id]
        if sid != seq_id:
            continue

        x1, y1, bw, bh = ann['bbox']
        if bw <= 0 or bh <= 0:
            continue

        tlwh = (x1, y1, bw, bh)
        cat_1idx = ann['category_id']               # 1-indexed, already 5-cls
        track_id = ann.get('track_id', 0)
        global_id = _global_track_id(track_id, cat_1idx)

        gt_frame_dict[frame_id].append((tlwh, global_id, 1))

    return dict(gt_frame_dict), {}   # ignore dict empty (pixels already blacked)


class CocoGTEvaluator:
    """Drop-in replacement for tracking_utils.evaluation.Evaluator when GT
    comes from a COCO JSON file (5cls benchmark) instead of raw VisDrone .txt.

    API is identical to Evaluator so track_ECDet needs minimal changes.
    """

    def __init__(self, ann_file: str, seq_id: str):
        import motmetrics as mm
        self._mm = mm
        self.seq_id = seq_id
        self.gt_frame_dict, self.gt_ignore_frame_dict = \
            load_coco_gt_for_seq(ann_file, seq_id)
        self.acc = mm.MOTAccumulator(auto_id=True)

    def reset_accumulator(self):
        self.acc = self._mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id: int, trk_tlwhs, trk_ids):
        import motmetrics as mm

        trk_tlwhs = np.asarray(trk_tlwhs, dtype=float).reshape(-1, 4)
        trk_ids   = list(trk_ids)

        gt_objs = self.gt_frame_dict.get(frame_id, [])
        if gt_objs:
            gt_tlwhs_raw, gt_ids, _ = zip(*gt_objs)
        else:
            gt_tlwhs_raw, gt_ids = [], []
        gt_tlwhs = np.asarray(gt_tlwhs_raw, dtype=float).reshape(-1, 4)

        # Ignore regions: empty for 5cls COCO (already baked into image pixels)
        iou_dist = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        self.acc.update(list(gt_ids), trk_ids, iou_dist)

    def eval_file(self, filename: str):
        """Parse prediction .txt and accumulate per frame."""
        self.reset_accumulator()

        result_dict: Dict[int, List] = defaultdict(list)
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 7:
                    continue
                fid  = int(parts[0])
                tid  = int(parts[1])
                x1, y1, w, h = map(float, parts[2:6])
                if w <= 0 or h <= 0:
                    continue
                result_dict[fid].append(((x1, y1, w, h), tid))

        for fid in sorted(result_dict):
            tlwhs = [r[0] for r in result_dict[fid]]
            tids  = [r[1] for r in result_dict[fid]]
            self.eval_frame(fid, tlwhs, tids)

        return self.acc