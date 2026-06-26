"""eval_mot_4cls.py -- Tracking + MOTMetrics evaluation on the 4-class competition split.

Model train 7 classes (visdrone2coco_7cls_mot.py):
    0:ped  1:bicycle  2:car  3:truck  4:tricycle  5:bus  6:motor

GT test-dev 4 classes (visdrone2coco_4cls_competition_mot.py):
    0:person  1:car  2:motorcycle  3:bicycle

Pipeline:
    LoadCocoSequencesForTracking (COCO JSON)
        -> SequenceRunner4cls.run()
            -> remap_dets_7cls_to_4cls()   ← KEY: drop truck/tricycle/bus, remap ids
        -> CocoGTEvaluator.eval_file()     (GT from COCO JSON)
        -> motmetrics MOTA / IDF1

Input:
    --track_img_root /data/VisDrone2019-COCO-4cls/test-dev/images
    --track_ann_file /data/VisDrone2019-COCO-4cls/test-dev/annotations/instances_test-dev.json
    --load_model     /path/to/checkpoint.pth
    (+ standard FalconMOT opts: --arch, --input-wh, --conf_thres, --use_appearance_motion, ...)
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List

import cv2
import motmetrics as mm
import numpy as np
import torch

import _paths  # noqa: F401

from falconmot.models.model import create_model, load_model
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracker.multitracker import MCJDETracker, MCTrack
from falconmot.tracker.class_remap import NUM_CLS_TRAIN   # model head = 7 (fixed)
from falconmot.tracking_utils import visualization as vis
from falconmot.tracking_utils.coco_gt_reader import CocoGTEvaluator
from falconmot.tracking_utils.log import logger
from falconmot.tracking_utils.timer import Timer
from falconmot.tracking_utils.utils import mkdir_if_missing
from falconmot.datasets.dataset.coco_detection import LoadCocoSequencesForTracking
from falconmot.opts import opts

# ── 7cls model output -> 4cls competition remap (self-contained) ─────────────
#   7cls (0-idx): 0 ped, 1 bicycle, 2 car, 3 truck, 4 tricycle, 5 bus, 6 motor
#   4cls (0-idx): 0 person, 1 car, 2 motorcycle, 3 bicycle
CLS4_NAMES = {0: 'person', 1: 'car', 2: 'motorcycle', 3: 'bicycle'}
NUM_CLS_EVAL_4 = 4

_REMAP_7_TO_4 = {
    0: 0,     # pedestrian -> person
    1: 3,     # bicycle    -> bicycle
    2: 1,     # car        -> car
    3: None,  # truck      -> DROP
    4: None,  # tricycle   -> DROP
    5: None,  # bus        -> DROP
    6: 2,     # motor      -> motorcycle
}


def remap_dets_7cls_to_4cls(dets: Dict[int, List]) -> Dict[int, List]:
    """Filter & remap detections from 7-class space to 4-class competition space.

    Args:
        dets: dict[cls_id_7 (0-indexed)] -> list[MCTrack]
    Returns:
        dict[cls_id_4 (0-indexed)] -> list[MCTrack], MCTrack.cls_id patched in place.
    """
    out: Dict[int, List] = {}
    for cls7, track_list in dets.items():
        cls4 = _REMAP_7_TO_4.get(cls7)
        if cls4 is None:
            continue   # drop truck / tricycle / bus
        for t in track_list:
            t.cls_id = cls4   # MCTrack.cls_id drives the per-class track-id offset
        out[cls4] = out.get(cls4, []) + track_list
    return out


_CLS_ID_OFFSET = 1_000_000

_RESULT_FMT = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},{cls_id},-1,-1\n'


class SequenceRunner4cls:
    """Runner with a built-in 7->4 class remap.

      1. Tracker is initialised with num_classes=NUM_CLS_EVAL_4 (track-id offset over 4 classes).
      2. After decoding (7 classes), remap_dets_7cls_to_4cls() drops truck/tricycle/bus and
         shifts the class index before feeding the tracker.
      3. When writing the result file, cls_id is a 4-class index (1-indexed).
      4. Query Appearance-Motion (QAM) is unchanged -- the dense-map wiring is identical.
    """

    def __init__(self, opt, frame_rate: int = 30):
        net_w, net_h = opt.img_size
        self._net_wh  = (net_w, net_h)
        self.device   = opt.device
        self.min_area = getattr(opt, 'min_box_area', 100)

        self.model = opt.model
        # Query Appearance-Motion needs the dense appearance map from the model.
        if getattr(opt, 'use_appearance_motion', False):
            _m = getattr(self.model, 'module', self.model)
            _m.return_reid_dense = True

        self.postprocessor = FalconJDEPostProcessor(
            num_classes=NUM_CLS_TRAIN,           # model head = 7
            num_top_queries=getattr(opt, 'K', 300),
            conf_thres=opt.conf_thres,
            use_focal_loss=True,
        )

        # Tracker runs in the 4-class space (after remap).
        opt_tracker        = type('Opt', (), dict(vars(opt)))()   # shallow copy
        opt_tracker.num_classes = NUM_CLS_EVAL_4
        self.tracker = MCJDETracker(opt_tracker, frame_rate)

        self.timer       = Timer()
        self._orig_sizes = None

    def _decode_detections(self, res: dict) -> defaultdict:
        """Postprocessor output (7cls) -> per-class MCTrack list, trong 7cls space."""
        dets = defaultdict(list)
        if len(res['scores']) == 0:
            return dets

        boxes_np  = res['boxes'].cpu().numpy()
        scores_np = res['scores'].cpu().numpy()
        labels_np = res['labels'].cpu().numpy()
        reid_np   = res['reid'].cpu().numpy() if 'reid' in res else None

        ws = boxes_np[:, 2] - boxes_np[:, 0]
        hs = boxes_np[:, 3] - boxes_np[:, 1]
        valid = np.where((ws > 0) & (hs > 0))[0]

        for i in valid:
            cls_id = int(labels_np[i])
            tlwh   = np.array([boxes_np[i, 0], boxes_np[i, 1], ws[i], hs[i]], dtype=np.float32)
            emb    = reid_np[i] if reid_np is not None else np.zeros(1, dtype=np.float32)
            dets[cls_id].append(MCTrack(tlwh, float(scores_np[i]), emb, NUM_CLS_EVAL_4, cls_id))
        return dets

    def _collect_tracks(self, online_targets_dict: dict):
        tlwhs, tids, scores = defaultdict(list), defaultdict(list), defaultdict(list)
        for cls_id, tracks in online_targets_dict.items():
            for t in tracks:
                w, h = t.curr_tlwh[2], t.curr_tlwh[3]
                if w * h > self.min_area:
                    tlwhs[cls_id].append(t.curr_tlwh)
                    tids[cls_id].append(t.track_id)
                    scores[cls_id].append(t.score)
        return tlwhs, tids, scores

    def run(self, seq_loader, result_path: str,
            save_dir=None, show_image: bool = False):
        """Run one sequence and write the MOT result file in the 4-class scheme."""
        if save_dir:
            mkdir_if_missing(save_dir)

        self.tracker.reset()
        self.timer       = Timer()
        self._orig_sizes = None

        with open(result_path, 'w') as f_out:
            for frame_id, img, img0 in seq_loader:
                orig_h, orig_w = img0.shape[:2]

                if self._orig_sizes is None:
                    self._orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

                blob = torch.from_numpy(img[None]).to(self.device)

                self.timer.tic()
                with torch.no_grad():
                    output = self.model(blob)
                    res    = self.postprocessor(output, self._orig_sizes)[0]
                    dets_7cls = self._decode_detections(res)
                    # ── KEY: remap 7cls -> 4cls, drop truck/tricycle/bus ──
                    dets_4cls = remap_dets_7cls_to_4cls(dets_7cls)
                self.timer.toc()

                self.tracker.set_image(img0)

                # Query Appearance-Motion: hand the dense map + the SAME image
                # transform the postprocessor uses to decode boxes.
                if 'reid_dense' in output:
                    if self.postprocessor._net_hw is not None:
                        net_h, net_w = self.postprocessor._net_hw
                        r = min(net_h / orig_h, net_w / orig_w)
                        rx = ry = r
                        pad_w = (net_w - orig_w * r) * 0.5
                        pad_h = (net_h - orig_h * r) * 0.5
                    else:
                        net_w, net_h = self._net_wh
                        rx, ry = net_w / orig_w, net_h / orig_h
                        pad_w = pad_h = 0.0
                    self.tracker.set_dense(
                        output['reid_dense'][0],            # [C,H,W]
                        stride=output['reid_dense_stride'],
                        ratio_x=rx, ratio_y=ry, pad_w=pad_w, pad_h=pad_h)

                online_targets = self.tracker.update(dets_4cls, h_orig=orig_h, w_orig=orig_w)

                tlwhs, tids, tscores = self._collect_tracks(online_targets)

                # Write result: cls_id 0-indexed -> 1-indexed to match the MOT format
                for cls4_0idx in range(NUM_CLS_EVAL_4):
                    for tlwh, tid, sc in zip(
                            tlwhs[cls4_0idx], tids[cls4_0idx], tscores[cls4_0idx]):
                        if tid < 0:
                            continue
                        f_out.write(_RESULT_FMT.format(
                            frame=frame_id,
                            id=tid + cls4_0idx * _CLS_ID_OFFSET,
                            x1=tlwh[0], y1=tlwh[1], w=tlwh[2], h=tlwh[3],
                            score=sc,
                            cls_id=cls4_0idx + 1,   # 1-indexed
                        ))

                if show_image or save_dir:
                    online_im = vis.plot_tracks(
                        image=img0,
                        tlwhs_dict=tlwhs,
                        obj_ids_dict=tids,
                        num_classes=NUM_CLS_EVAL_4,
                        scores=tscores,
                        frame_id=frame_id,
                        fps=1.0 / max(1e-5, self.timer.average_time),
                    )
                    if show_image:
                        cv2.imshow('online_im', online_im)
                    if save_dir:
                        cv2.imwrite(osp.join(save_dir, f'{frame_id:05d}.jpg'), online_im)

                if frame_id % 30 == 0:
                    logger.info('Frame %d  %.2f fps', frame_id,
                                1.0 / max(1e-5, self.timer.average_time))

        logger.info('Saved -> %s', result_path)
        return self.timer.average_time, self.timer.calls


def main(opt, ann_file: str, img_root: str, exp_name: str,
         save_images: bool = False, show_image: bool = False):
    logger.setLevel(logging.INFO)

    result_root = osp.join(osp.dirname(img_root), 'results', exp_name)
    mkdir_if_missing(result_root)

    src = LoadCocoSequencesForTracking(ann_file, img_root, img_size=opt.img_size)

    runner = SequenceRunner4cls(opt)
    accs, names = [], []
    timer_avgs, timer_calls = [], []

    for seq_id in src.seqs:
        logger.info('--- seq: %s', seq_id)
        result_filename = osp.join(result_root, f'{seq_id}.txt')
        output_dir      = (osp.join(osp.dirname(img_root), 'outputs', exp_name, seq_id)
                           if save_images else None)

        seq_loader = src.sequence(seq_id)
        ta, tc = runner.run(seq_loader, result_filename,
                            save_dir=output_dir, show_image=show_image)
        timer_avgs.append(ta)
        timer_calls.append(tc)

        evaluator = CocoGTEvaluator(ann_file, seq_id)
        acc = evaluator.eval_file(result_filename)
        accs.append(acc)
        names.append(seq_id)
        logger.info('Done seq: %s', seq_id)

    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time    = np.dot(timer_avgs, timer_calls)
    avg_fps     = 1.0 / max(all_time / max(np.sum(timer_calls), 1), 1e-5)
    logger.info('Total %.2fs  FPS %.2f', all_time, avg_fps)

    print(f'\n[Eval] 4-class competition: {" / ".join(CLS4_NAMES.values())}')
    metrics = mm.metrics.motchallenge_metrics
    mh      = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names)
    print(strsummary)

    import pandas as pd
    out_xlsx = osp.join(result_root, f'summary_{exp_name}.xlsx')
    summary.to_excel(pd.ExcelWriter(out_xlsx))
    logger.info('Summary saved -> %s', out_xlsx)


if __name__ == '__main__':
    opt = opts().init()

    opt.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 else 'cpu'

    img_root = getattr(opt, 'track_img_root',
                       '/data/VisDrone2019-COCO-4cls/test-dev/images')
    ann_file = getattr(opt, 'track_ann_file',
                       '/data/VisDrone2019-COCO-4cls/test-dev/annotations/instances_test-dev.json')

    assert osp.isdir(img_root),  f'img_root not found: {img_root}'
    assert osp.isfile(ann_file), f'ann_file not found: {ann_file}'

    print('Creating model...')
    opt.model = create_model(opt.arch, opt)
    opt.model = load_model(opt.model, opt.load_model)
    opt.model = opt.model.to(opt.device).eval()

    main(
        opt,
        ann_file=ann_file,
        img_root=img_root,
        exp_name='falcon_visdrone_4cls',
        show_image=False,
        save_images=True,
    )