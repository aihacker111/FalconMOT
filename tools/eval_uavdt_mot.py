"""eval_mot_uavdt.py -- Tracking + MOTMetrics(+HOTA) evaluation on UAV-DT.

Mirror of eval_mot_5cls.py, adapted for UAV-DT.

Model trains 7 classes (visdrone, 0-indexed):
    0:ped 1:bicycle 2:car 3:van 4:truck 5:bus 6:motor

UAV-DT GT (from tools/data_preprocessing/uavdt2coco_mot.py) has only vehicles.
Two eval protocols (pick with --uavdt_scheme):
    vehicle1 (default) : single class 'vehicle'  <-- the standard UAVDT MOT protocol
                         model {car,van,truck,bus} -> vehicle ; drop ped/bicycle/motor
    uavdt3             : native 3 classes car/truck/bus
                         model car->0, truck->1, bus->2 ; drop the rest
The GT json must be generated with the MATCHING --class_scheme:
    vehicle1  <-> uavdt2coco_mot.py --class_scheme vehicle1
    uavdt3    <-> uavdt2coco_mot.py --class_scheme uavdt3

Pipeline (identical to the 5-class benchmark):
    LoadCocoSequencesForTracking (COCO JSON)
      -> UAVDTSequenceRunner.run()  (remap 7cls -> UAVDT eval space)
      -> CocoGTEvaluator.eval_file()  (GT from COCO JSON)
      -> motmetrics MOTA/IDF1 (+ HOTA via tools' hota.py)

Input:
    --track_img_root  /data/UAVDT-COCO/test/images
    --track_ann_file  /data/UAVDT-COCO/test/annotations/instances_test.json
    --uavdt_scheme    vehicle1|uavdt3
    --load_model      /path/to/checkpoint.pth
    (+ standard FalconMOT opts: --arch, --input-wh, --conf_thres, ...)
"""

from __future__ import absolute_import, division, print_function

import logging
import os.path as osp
from collections import defaultdict
from typing import Dict, List

import cv2
import motmetrics as mm
import numpy as np
import torch

import _paths  # noqa: F401

from falconmot import create_model, load_model
from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracker.multitracker import MCJDETracker, MCTrack
from falconmot.tracker.class_remap import CLS7_NAMES, NUM_CLS_TRAIN
from falconmot.tracker.utils import mkdir_if_missing
from falconmot.tracker.utils import visualization as vis
from falconmot.tracker.utils.coco_gt_reader import CocoGTEvaluator
from falconmot.utils.log import logger
from falconmot.tracker.utils.timer import Timer
from falconmot.data.dataset import LoadCocoSequencesForTracking
from falconmot.cfg import opts

_CLS_ID_OFFSET = 1_000_000
_RESULT_FMT = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},{cls_id},-1,-1\n'


# ── UAV-DT eval schemes: map model 7cls (0-idx) -> UAVDT eval class (0-idx) ────
# (None / absent = drop). Mirrors class_remap._REMAP_7_TO_5 in spirit.
UAVDT_SCHEMES: Dict[str, dict] = {
    # Standard UAVDT MOT protocol: everything drivable -> one 'vehicle' class.
    'vehicle1': {
        'names': {0: 'vehicle'},
        'num_eval': 1,
        'map7': {2: 0, 3: 0, 4: 0, 5: 0},          # car, van, truck, bus -> vehicle
    },
    # Native UAVDT 3 classes.
    'uavdt3': {
        'names': {0: 'car', 1: 'truck', 2: 'bus'},
        'num_eval': 3,
        'map7': {2: 0, 4: 1, 5: 2},                # car->0, truck->1, bus->2 (van dropped)
    },
}


def remap_dets_7cls_to_uavdt(dets: Dict[int, List], map7: Dict[int, int]) -> Dict[int, List]:
    """Filter & remap model detections from 7-class space to the UAVDT eval space.

    Patches MCTrack.cls_id in place (cls_id drives the per-class track-id offset),
    exactly like class_remap.remap_dets_7cls_to_5cls.
    """
    out: Dict[int, List] = {}
    for cls7, track_list in dets.items():
        new_cls = map7.get(cls7)
        if new_cls is None:
            continue
        remapped = []
        for t in track_list:
            t.cls_id = new_cls
            remapped.append(t)
        out[new_cls] = out.get(new_cls, []) + remapped
    return out


class UAVDTSequenceRunner:
    """Sequence runner with a 7cls -> UAVDT-eval remap (mirror of ECDetSequenceRunner5cls)."""

    def __init__(self, opt, scheme: dict, frame_rate: int = 30):
        net_w, net_h = opt.img_size
        self._net_wh = (net_w, net_h)
        self.device = opt.device
        self.min_area = getattr(opt, 'min_box_area', 100)
        self.scheme = scheme
        self.num_eval = scheme['num_eval']
        self.map7 = scheme['map7']

        self.model = opt.model
        if getattr(opt, 'use_appearance_motion', False):
            _m = getattr(self.model, 'module', self.model)
            _m.return_reid_dense = True

        self.postprocessor = FalconJDEPostProcessor(
            num_classes=NUM_CLS_TRAIN,               # model head = 7
            num_top_queries=getattr(opt, 'K', 300),
            conf_thres=opt.conf_thres,
            use_focal_loss=True,
            nms_emb_relax=0.45,
        )
        # Tracker runs in the UAVDT eval space; num_classes = num_eval so the
        # track-id offset = cls_eval0idx * 1_000_000, matching the GT side.
        opt_tracker = type('Opt', (), dict(vars(opt)))()
        opt_tracker.num_classes = self.num_eval
        self.tracker = MCJDETracker(opt_tracker, frame_rate)

        self.timer = Timer()
        self._orig_sizes = None

    def _decode_detections(self, res: dict) -> defaultdict:
        """Postprocessor output (7cls) -> per-class MCTrack list, in 7cls space."""
        dets = defaultdict(list)
        if len(res['scores']) == 0:
            return dets
        boxes_np = res['boxes'].cpu().numpy()
        scores_np = res['scores'].cpu().numpy()
        labels_np = res['labels'].cpu().numpy()
        reid_np = res['reid'].cpu().numpy() if 'reid' in res else None

        ws = boxes_np[:, 2] - boxes_np[:, 0]
        hs = boxes_np[:, 3] - boxes_np[:, 1]
        valid = np.where((ws > 0) & (hs > 0))[0]
        for i in valid:
            cls_id = int(labels_np[i])
            tlwh = np.array([boxes_np[i, 0], boxes_np[i, 1], ws[i], hs[i]], dtype=np.float32)
            emb = reid_np[i] if reid_np is not None else np.zeros(1, dtype=np.float32)
            # num_classes = num_eval: track-id counter keyed per eval slot (post-remap)
            dets[cls_id].append(MCTrack(tlwh, float(scores_np[i]), emb, self.num_eval, cls_id))
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

    def run(self, seq_loader, result_path: str, save_dir=None, show_image: bool = False):
        if save_dir:
            mkdir_if_missing(save_dir)
        self.tracker.reset()
        self.timer = Timer()
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
                    res = self.postprocessor(output, self._orig_sizes)[0]
                    dets_7cls = self._decode_detections(res)
                    dets_eval = remap_dets_7cls_to_uavdt(dets_7cls, self.map7)
                self.timer.toc()

                self.tracker.set_image(img0)

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
                        output['reid_dense'], stride=output['reid_dense_stride'],
                        ratio_x=rx, ratio_y=ry, pad_w=pad_w, pad_h=pad_h)

                online_targets = self.tracker.update(dets_eval, h_orig=orig_h, w_orig=orig_w)
                tlwhs, tids, tscores = self._collect_tracks(online_targets)

                for cls0 in range(self.num_eval):
                    for tlwh, tid, sc in zip(tlwhs[cls0], tids[cls0], tscores[cls0]):
                        if tid < 0:
                            continue
                        f_out.write(_RESULT_FMT.format(
                            frame=frame_id, id=tid + cls0 * _CLS_ID_OFFSET,
                            x1=tlwh[0], y1=tlwh[1], w=tlwh[2], h=tlwh[3],
                            score=sc, cls_id=cls0 + 1))

                if show_image or save_dir:
                    online_im = vis.plot_tracks(
                        image=img0, tlwhs_dict=tlwhs, obj_ids_dict=tids,
                        num_classes=self.num_eval, scores=tscores, frame_id=frame_id,
                        fps=1.0 / max(1e-5, self.timer.average_time))
                    if show_image:
                        cv2.imshow('online_im', online_im)
                    if save_dir:
                        cv2.imwrite(osp.join(save_dir, f'{frame_id:05d}.jpg'), online_im)

                if frame_id % 30 == 0:
                    logger.info('Frame %d  %.2f fps', frame_id,
                                1.0 / max(1e-5, self.timer.average_time))

        logger.info('Saved -> %s', result_path)
        return self.timer.average_time, self.timer.calls


def main(opt, ann_file: str, img_root: str, exp_name: str, scheme_name: str,
         save_images: bool = False, show_image: bool = False):
    logger.setLevel(logging.INFO)
    scheme = UAVDT_SCHEMES[scheme_name]

    result_root = osp.join(osp.dirname(img_root), 'results', exp_name)
    mkdir_if_missing(result_root)

    src = LoadCocoSequencesForTracking(ann_file, img_root, img_size=opt.img_size)
    runner = UAVDTSequenceRunner(opt, scheme)
    accs, names = [], []
    timer_avgs, timer_calls = [], []

    for seq_id in src.seqs:
        logger.info('--- seq: %s', seq_id)
        result_filename = osp.join(result_root, f'{seq_id}.txt')
        output_dir = (osp.join(osp.dirname(img_root), 'outputs', exp_name, seq_id)
                      if save_images else None)
        seq_loader = src.sequence(seq_id)
        ta, tc = runner.run(seq_loader, result_filename,
                            save_dir=output_dir, show_image=show_image)
        timer_avgs.append(ta)
        timer_calls.append(tc)

        evaluator = CocoGTEvaluator(ann_file, seq_id)
        accs.append(evaluator.eval_file(result_filename))
        names.append(seq_id)
        logger.info('Done seq: %s', seq_id)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_fps = 1.0 / max(all_time / max(np.sum(timer_calls), 1), 1e-5)
    logger.info('Total %.2fs  FPS %.2f', all_time, avg_fps)

    print(f'\n[Eval] UAV-DT ({scheme_name}): {" / ".join(scheme["names"].values())}')
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)

    import pandas as pd
    out_xlsx = osp.join(result_root, f'summary_{exp_name}.xlsx')
    summary.to_excel(pd.ExcelWriter(out_xlsx))
    logger.info('Summary saved -> %s', out_xlsx)

    # ── HOTA / DetA / AssA via TrackEval (per-class, class-averaged) ──
    try:
        from falconmot.tracker.utils.hota import evaluate_hota_from_results
        evaluate_hota_from_results(ann_file, result_root, src.seqs, scheme['names'])
    except ImportError:
        logger.warning('TrackEval not installed -> skipping HOTA. '
                       'pip install git+https://github.com/JonathonLuiten/TrackEval.git')


if __name__ == '__main__':
    opt = opts().init()
    opt.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 else 'cpu'

    # --track_img_root / --track_ann_file : the UAVDT-COCO output of uavdt2coco_mot.py
    img_root = getattr(opt, 'track_img_root', '/data/UAVDT-COCO/test/images')
    ann_file = getattr(opt, 'track_ann_file',
                       '/data/UAVDT-COCO/test/annotations/instances_test.json')
    scheme_name = getattr(opt, 'uavdt_scheme', 'uavdt3')
    assert scheme_name in UAVDT_SCHEMES, f'--uavdt_scheme must be one of {list(UAVDT_SCHEMES)}'
    assert osp.isdir(img_root), f'img_root not found: {img_root}'
    assert osp.isfile(ann_file), f'ann_file not found: {ann_file}'

    print('Creating model...')
    opt.model = create_model(opt.arch, opt)
    opt.model = load_model(opt.model, opt.load_model)
    opt.model = opt.model.to(opt.device).eval()

    main(opt, ann_file=ann_file, img_root=img_root,
         exp_name=f'falconmot_uavdt_{scheme_name}', scheme_name=scheme_name,
         show_image=False, save_images=False)