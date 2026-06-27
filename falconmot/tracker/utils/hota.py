"""hota.py -- HOTA / DetA / AssA / LocA evaluation for FalconMOT.

py-motmetrics (used by eval_mot_5cls.py) computes MOTA / IDF1 but NOT HOTA.
HOTA is computed with the official TrackEval metric. This module is a drop-in:
it re-reads the GT (COCO JSON, via the existing load_coco_gt_for_seq) and the
per-sequence prediction .txt files that eval_mot_5cls.py already writes, splits
detections per class (VisDrone convention = class-averaged HOTA), and runs
TrackEval's HOTA metric in memory -- no extra GT dump, no change to the runner.

Install TrackEval once:
    pip install git+https://github.com/JonathonLuiten/TrackEval.git
    # (or: git clone ... && pip install -e TrackEval)

Usage (add 3 lines at the end of eval_mot_5cls.main, see bottom of this file):
    from falconmot.tracker.utils.hota import evaluate_hota_from_results
    evaluate_hota_from_results(ann_file, result_root, src.seqs, CLS5_NAMES)

The class split uses id // CLS_ID_OFFSET, which matches how both GT and the
prediction .txt bake the class into the (globally unique) track id.
"""
from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

# --- NUMPY PATCH FOR TRACKEVAL (NumPy >= 1.24 compatibility) ---
# TrackEval relies on deprecated types (np.float, np.int, np.bool) that were 
# removed in NumPy 1.24. This runtime patch maps them to valid types so 
# TrackEval runs seamlessly without requiring any source code modifications.
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'object'):
    np.object = object
# ---------------------------------------------------------------

from falconmot.tracker.utils.coco_gt_reader import load_coco_gt_for_seq

CLS_ID_OFFSET = 1_000_000


# --------------------------------------------------------------------------
# Geometry: raw IoU similarity (NOT thresholded -- HOTA integrates over alpha)
# --------------------------------------------------------------------------
def box_iou_matrix(gt_tlwh: np.ndarray, tr_tlwh: np.ndarray) -> np.ndarray:
    """IoU in [0,1] between GT and tracker boxes given as (x,y,w,h).

    Returns an [num_gt, num_tr] matrix; 0 where there is no overlap.
    Unlike mm.distances.iou_matrix(..., max_iou=0.5) this does NOT gate at 0.5,
    because HOTA needs the true similarity at every localisation threshold.
    """
    G, T = len(gt_tlwh), len(tr_tlwh)
    if G == 0 or T == 0:
        return np.zeros((G, T), dtype=np.float64)
    g = np.asarray(gt_tlwh, dtype=np.float64).reshape(-1, 4)
    t = np.asarray(tr_tlwh, dtype=np.float64).reshape(-1, 4)
    gx1, gy1, gw, gh = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    tx1, ty1, tw, th = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    gx2, gy2 = gx1 + gw, gy1 + gh
    tx2, ty2 = tx1 + tw, ty1 + th

    ix1 = np.maximum(gx1[:, None], tx1[None, :])
    iy1 = np.maximum(gy1[:, None], ty1[None, :])
    ix2 = np.minimum(gx2[:, None], tx2[None, :])
    iy2 = np.minimum(gy2[:, None], ty2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_g = (gw * gh)[:, None]
    area_t = (tw * th)[None, :]
    union = area_g + area_t - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _read_pred_txt(path: str) -> Dict[int, List]:
    """Parse a result .txt -> frame_id -> [(tlwh, global_track_id), ...]."""
    out: Dict[int, List] = defaultdict(list)
    if not osp.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            fid, tid = int(p[0]), int(p[1])
            x, y, w, h = map(float, p[2:6])
            if w <= 0 or h <= 0:
                continue
            out[fid].append(((x, y, w, h), tid))
    return out


# --------------------------------------------------------------------------
# Per-(class, sequence) frame store -> TrackEval HOTA
# --------------------------------------------------------------------------
def _remap(per_frame_ids: List[List[int]]):
    """Map arbitrary (offset) ids to contiguous 0..N-1 ids, as TrackEval wants."""
    uniq = sorted({int(i) for fr in per_frame_ids for i in fr})
    lut = {v: k for k, v in enumerate(uniq)}
    remapped = [np.array([lut[int(i)] for i in fr], dtype=int) for fr in per_frame_ids]
    return remapped, len(uniq)


class HOTACollector:
    """Accumulate per (class, sequence) frames and run TrackEval's HOTA."""

    def __init__(self, class_names: Sequence[str]):
        self.class_names = list(class_names)
        # store[cls][seq] = {'gt':[ids/frame], 'trk':[ids/frame], 'sim':[matrix/frame]}
        self.store: Dict[str, Dict[str, dict]] = {c: {} for c in self.class_names}

    def add_frame(self, cls_name, seq, gt_ids, trk_ids, sim):
        d = self.store[cls_name].setdefault(seq, {'gt': [], 'trk': [], 'sim': []})
        d['gt'].append(list(gt_ids))
        d['trk'].append(list(trk_ids))
        d['sim'].append(np.asarray(sim, dtype=np.float64))

    def _seq_data(self, sd: dict) -> dict:
        gt_ids, n_gt = _remap(sd['gt'])
        trk_ids, n_trk = _remap(sd['trk'])
        return {
            'num_timesteps': len(gt_ids),
            'num_gt_ids': n_gt,
            'num_tracker_ids': n_trk,
            'num_gt_dets': int(sum(len(g) for g in gt_ids)),
            'num_tracker_dets': int(sum(len(t) for t in trk_ids)),
            'gt_ids': gt_ids,
            'tracker_ids': trk_ids,
            'similarity_scores': sd['sim'],
        }

    @staticmethod
    def _scalar(res: dict, key: str) -> float:
        v = res[key]
        return float(np.mean(v)) if hasattr(v, '__len__') else float(v)

    def compute(self) -> dict:
        """Return {'per_class': {cls: {...}}, 'overall': {...}} of HOTA scalars."""
        from trackeval.metrics import HOTA  # imported here so the dep is optional
        hota = HOTA()
        per_class = {}
        for c in self.class_names:
            seqs = self.store.get(c, {})
            if not seqs:
                continue
            seq_res = {seq: hota.eval_sequence(self._seq_data(sd))
                       for seq, sd in seqs.items()}
            cls_res = hota.combine_sequences(seq_res)
            per_class[c] = {
                'HOTA': self._scalar(cls_res, 'HOTA'),
                'DetA': self._scalar(cls_res, 'DetA'),
                'AssA': self._scalar(cls_res, 'AssA'),
                'LocA': self._scalar(cls_res, 'LocA'),
                'DetRe': self._scalar(cls_res, 'DetRe'),
                'AssRe': self._scalar(cls_res, 'AssRe'),
            }
        # VisDrone-style class-averaged HOTA
        keys = ['HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe']
        overall = {k: float(np.mean([per_class[c][k] for c in per_class])) for k in keys} \
            if per_class else {k: 0.0 for k in keys}
        return {'per_class': per_class, 'overall': overall}


# --------------------------------------------------------------------------
# One-call entry point (no change to the runner / CocoGTEvaluator needed)
# --------------------------------------------------------------------------
def evaluate_hota_from_results(ann_file: str,
                               result_root: str,
                               seqs: Sequence[str],
                               class_names: Dict[int, str],
                               offset: int = CLS_ID_OFFSET) -> dict:
    """Compute HOTA from the per-seq result .txt files already written by the runner.

    Args:
        ann_file    : COCO 5-class GT json (same one passed to eval_mot_5cls).
        result_root : folder holding '<seq>.txt' prediction files.
        seqs        : sequence ids to evaluate.
        class_names : {0-idx -> name}, e.g. CLS5_NAMES.
    Returns the dict from HOTACollector.compute() and prints a table.
    """
    names = [class_names[i] for i in sorted(class_names)]
    coll = HOTACollector(names)

    for seq in seqs:
        gt_frames, _ = load_coco_gt_for_seq(ann_file, seq)        # frame -> [(tlwh,gid,1)]
        pred_frames = _read_pred_txt(osp.join(result_root, f'{seq}.txt'))
        all_fids = sorted(set(gt_frames) | set(pred_frames))      # aligned timesteps

        for cls0 in sorted(class_names):
            cname = class_names[cls0]
            for fid in all_fids:
                gt = [(tlwh, gid) for (tlwh, gid, _) in gt_frames.get(fid, [])
                      if gid // offset == cls0]
                pr = [(tlwh, tid) for (tlwh, tid) in pred_frames.get(fid, [])
                      if tid // offset == cls0]
                gt_box = np.array([b for b, _ in gt], dtype=np.float64).reshape(-1, 4)
                pr_box = np.array([b for b, _ in pr], dtype=np.float64).reshape(-1, 4)
                sim = box_iou_matrix(gt_box, pr_box)
                coll.add_frame(cname, seq, [i for _, i in gt], [i for _, i in pr], sim)

    res = coll.compute()

    # ---- pretty print ----
    print('\n[HOTA] per-class (TrackEval):')
    hdr = f'{"class":<12}{"HOTA":>8}{"DetA":>8}{"AssA":>8}{"LocA":>8}{"DetRe":>8}{"AssRe":>8}'
    print(hdr)
    for c, m in res['per_class'].items():
        print(f'{c:<12}{m["HOTA"]*100:>8.2f}{m["DetA"]*100:>8.2f}{m["AssA"]*100:>8.2f}'
              f'{m["LocA"]*100:>8.2f}{m["DetRe"]*100:>8.2f}{m["AssRe"]*100:>8.2f}')
    o = res['overall']
    print('-' * len(hdr))
    print(f'{"AVG":<12}{o["HOTA"]*100:>8.2f}{o["DetA"]*100:>8.2f}{o["AssA"]*100:>8.2f}'
          f'{o["LocA"]*100:>8.2f}{o["DetRe"]*100:>8.2f}{o["AssRe"]*100:>8.2f}')
    return res