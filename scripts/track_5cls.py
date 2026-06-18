
"""track_ECDet_5cls.py — Tracking + MOTMetrics eval trên 5-class benchmark.

Model train 7 classes (visdrone2coco_7cls_mot.py):
    0:ped  1:bicycle  2:car  3:truck  4:tricycle  5:bus  6:motor

GT test-dev 5 classes (visdrone2coco_5cls_benchmark_mot.py):
    0:ped  1:car  2:truck  3:tricycle  4:bus

Pipeline:
    LoadCocoSequencesForTracking (COCO JSON)
        -> ECDetSequenceRunner.run()
            -> remap_dets_7cls_to_5cls()   ← KEY: drop bicycle/motor, remap ids
        -> CocoGTEvaluator.eval_file()     ← GT từ COCO JSON, không phải raw .txt
        -> motmetrics MOTA / IDF1

Input:
    --img_root   /data/VisDrone2019-COCO-5cls/test-dev/images
    --ann_file   /data/VisDrone2019-COCO-5cls/test-dev/annotations/instances_test-dev.json
    --load_model /path/to/checkpoint.pth
    (+ standard FalconMOT opts: --arch, --input-wh, --conf_thres, etc.)
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import os.path as osp
from collections import defaultdict

import cv2
import motmetrics as mm
import numpy as np
import torch

import _paths  # noqa: F401

from falconmot.models.model import create_model, load_model
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracker.multitracker import MCJDETracker, MCTrack
from falconmot.tracker.class_remap import (
    remap_dets_7cls_to_5cls,
    CLS5_NAMES,
    NUM_CLS_TRAIN,
    NUM_CLS_EVAL,
    SKIP_SET_AFTER_REMAP,
)
from falconmot.tracking_utils import visualization as vis
from falconmot.tracking_utils.coco_gt_reader import CocoGTEvaluator
from falconmot.tracking_utils.log import logger
from falconmot.tracking_utils.timer import Timer
from falconmot.tracking_utils.utils import mkdir_if_missing
from falconmot.datasets.dataset.coco_detection import LoadCocoSequencesForTracking
from falconmot.opts import opts

_CLS_ID_OFFSET = 1_000_000

_RESULT_FMT = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},{cls_id},-1,-1\n'


class ECDetSequenceRunner5cls:
    """ECDet runner với 7->5 class remap built-in.

    Khác biệt so với ECDetSequenceRunner gốc:
      1. Tracker init với NUM_CLS_TRAIN=7 (giữ nguyên, tracker không biết về remap).
      2. Sau decode, gọi remap_dets_7cls_to_5cls() để drop bicycle/motor và
         shift class index trước khi đưa vào tracker.
      3. Khi ghi result file, cls_id là 5-cls index (1-indexed).
      4. num_cls trong MCTrack được set = NUM_CLS_EVAL để track_id offset
         chỉ cộng theo 5 class.
    """

    def __init__(self, opt, frame_rate: int = 30):
        net_w, net_h = opt.img_size
        self.device   = opt.device
        self.min_area = getattr(opt, 'min_box_area', 100)

        self.model = opt.model

        self.postprocessor = FalconJDEPostProcessor(
            num_classes=NUM_CLS_TRAIN,           # model head = 7
            num_top_queries=getattr(opt, 'K', 300),
            conf_thres=opt.conf_thres,
            use_focal_loss=True,
        )
        self.postprocessor.set_net_hw(net_h, net_w)

        # Tracker chạy trên 5cls space (sau remap); num_classes=5 đảm bảo
        # track_id offset = cls_5idx * 1_000_000, khớp với GT side.
        opt_tracker        = type('Opt', (), dict(vars(opt)))()   # shallow copy
        opt_tracker.num_classes = NUM_CLS_EVAL
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
            # num_classes=NUM_CLS_EVAL: track_id counter keyed per 5cls slot
            dets[cls_id].append(MCTrack(tlwh, float(scores_np[i]), emb, NUM_CLS_EVAL, cls_id))
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
        """Chạy một sequence, ghi MOT result file theo 5-cls scheme.

        seq_loader: _CocoSeqIterator — yield (frame_id:int, img, img0)
        """
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
                    # ── KEY: remap 7cls -> 5cls, drop bicycle(1) + motor(6) ──
                    dets_5cls = remap_dets_7cls_to_5cls(dets_7cls)
                self.timer.toc()

                self.tracker.set_image(img0)
                online_targets = self.tracker.update(dets_5cls, h_orig=orig_h, w_orig=orig_w)

                tlwhs, tids, tscores = self._collect_tracks(online_targets)

                # Ghi result: cls_id 0-indexed -> 1-indexed để khớp format MOT
                for cls5_0idx in range(NUM_CLS_EVAL):
                    for tlwh, tid, sc in zip(
                            tlwhs[cls5_0idx], tids[cls5_0idx], tscores[cls5_0idx]):
                        if tid < 0:
                            continue
                        f_out.write(_RESULT_FMT.format(
                            frame=frame_id,
                            id=tid + cls5_0idx * _CLS_ID_OFFSET,
                            x1=tlwh[0], y1=tlwh[1], w=tlwh[2], h=tlwh[3],
                            score=sc,
                            cls_id=cls5_0idx + 1,   # 1-indexed
                        ))

                if show_image or save_dir:
                    online_im = vis.plot_tracks(
                        image=img0,
                        tlwhs_dict=tlwhs,
                        obj_ids_dict=tids,
                        num_classes=NUM_CLS_EVAL,
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

    # Load sequences từ COCO JSON — không hardcode seq list
    src = LoadCocoSequencesForTracking(ann_file, img_root, img_size=opt.img_size)

    runner = ECDetSequenceRunner5cls(opt)
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

        # Eval với COCO JSON GT (5cls), không cần raw .txt
        evaluator = CocoGTEvaluator(ann_file, seq_id)
        acc = evaluator.eval_file(result_filename)
        accs.append(acc)
        names.append(seq_id)
        logger.info('Done seq: %s', seq_id)

    # Summary
    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time    = np.dot(timer_avgs, timer_calls)
    avg_fps     = 1.0 / max(all_time / max(np.sum(timer_calls), 1), 1e-5)
    logger.info('Total %.2fs  FPS %.2f', all_time, avg_fps)

    print(f'\n[Eval] 5-class benchmark: {" / ".join(CLS5_NAMES.values())}')
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

    # ── Required args (thêm vào opts hoặc truyền trực tiếp) ──────────────
    # --img_root  : thư mục ảnh (images/) của bộ test-dev COCO 5cls
    # --ann_file  : file instances_test-dev.json
    # Nếu opts chưa có 2 field này, dùng argparse thêm vào hoặc hardcode:
    img_root = getattr(opt, 'track_img_root',
                       '/data/VisDrone2019-COCO-5cls/test-dev/images')
    ann_file = getattr(opt, 'track_ann_file',
                       '/data/VisDrone2019-COCO-5cls/test-dev/annotations/instances_test-dev.json')

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
        exp_name='ecdet_visdrone_5cls',
        show_image=False,
        save_images=False,
    )