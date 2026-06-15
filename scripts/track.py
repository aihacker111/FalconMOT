"""
track.py — Multi-class online tracking inference for FalconJDE.

Luồng:
  LoadImages → SequenceRunner.run() → write incremental MOT results
             → Evaluator.eval_file() → motmetrics MOTA/IDF1

global_track_id = per-class track_id + cls_id_0indexed * 1_000_000
  → VisDrone track IDs are per-class (pedestrian id=1 ≠ car id=1)
  → offset đảm bảo IDs unique trong single-accumulator motmetrics
  → io.py GT side dùng offset tương tự
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

import _paths  # noqa: F401  (sys.path bootstrap)

from falconmot.models.model import create_model, load_model
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
from falconmot.tracker import FalconTracker, Track
from falconmot.tracking_utils import visualization as vis
from falconmot.tracking_utils.evaluation import Evaluator
from falconmot.tracking_utils.io import set_eval_mode
from falconmot.tracking_utils.log import logger
from falconmot.tracking_utils.timer import Timer
from falconmot.tracking_utils.utils import mkdir_if_missing
from falconmot.datasets.dataset.coco_detection import (
    LoadImagesForTracking,
    LoadCocoSequencesForTracking,
)
from falconmot.opts import opts

# Must match io.py._CLS_ID_OFFSET
_CLS_ID_OFFSET = 1_000_000

# 0-indexed cls_ids to skip per eval mode
_EVAL_SKIP_0IDX = {
    '10class': set(),
    '5class':  {1, 2, 6, 7, 9},    # people, bicycle, tricycle, awning-tri, motor
    '4class':  {1, 4, 5, 6, 7, 8}, # people, van, truck, tricycle, awning-tri, bus
}

_VISDRONE_SEQS = [
    'uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v',
    'uav0000077_00720_v', 'uav0000088_00290_v', 'uav0000119_02301_v',
    'uav0000120_04775_v', 'uav0000161_00000_v', 'uav0000188_00000_v',
    'uav0000201_00000_v', 'uav0000249_00001_v', 'uav0000249_02688_v',
    'uav0000297_00000_v', 'uav0000297_02761_v', 'uav0000306_00230_v',
    'uav0000355_00001_v', 'uav0000370_00001_v',
]

_RESULT_FMT = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},{cls_id},-1,-1\n'


# ---------------------------------------------------------------------------
# SequenceRunner — model + tracker for one sequence
# ---------------------------------------------------------------------------

class SequenceRunner:
    """
    Encapsulates postprocessor + tracker for a single sequence.

    Improvements over procedural version:
      - Incremental write: results flushed per frame, no RAM accumulation
      - Cached orig_sizes: tensor created once per sequence, not per frame
      - Vectorized box filter in _decode_detections
      - Single-pass track collection
    """

    def __init__(self, opt, frame_rate: int = 30):
        net_w, net_h   = opt.img_size
        self.device     = opt.device
        self.num_cls    = opt.num_classes
        self.min_area   = opt.min_box_area
        self.eval_mode  = getattr(opt, 'eval_mode', '10class')
        self.skip_set   = _EVAL_SKIP_0IDX.get(self.eval_mode, set())

        self.model = opt.model

        self.postprocessor = FalconJDEPostProcessor(
            num_classes     = opt.num_classes,
            num_top_queries = getattr(opt, 'K', 300),
            conf_thres      = opt.conf_thres,
            use_focal_loss  = True,
        )
        # No-letterbox full flow: box decode dùng simple-scale theo orig_size.
        # KHÔNG gọi set_net_hw (đã thành no-op) — tránh hiểu nhầm còn letterbox.

        self.tracker     = FalconTracker(opt, frame_rate)
        self.timer       = Timer()
        self._orig_sizes = None   # cached once per sequence

    # ------------------------------------------------------------------

    def _decode_detections(self, res: dict) -> defaultdict:
        """Postprocessor output → per-class Track list."""
        dets = defaultdict(list)
        if len(res['scores']) == 0:
            return dets

        boxes_np  = res['boxes'].cpu().numpy()
        scores_np = res['scores'].cpu().numpy()
        labels_np = res['labels'].cpu().numpy()
        reid_np   = res['reid'].cpu().numpy() if 'reid' in res else None

        # Vectorized valid-box filter (w > 0, h > 0)
        ws    = boxes_np[:, 2] - boxes_np[:, 0]
        hs    = boxes_np[:, 3] - boxes_np[:, 1]
        valid = np.where((ws > 0) & (hs > 0))[0]

        for i in valid:
            cls_id = int(labels_np[i])
            tlwh   = np.array([boxes_np[i, 0], boxes_np[i, 1], ws[i], hs[i]],
                               dtype=np.float32)
            emb    = reid_np[i] if reid_np is not None else np.zeros(1, dtype=np.float32)
            dets[cls_id].append(
                Track(tlwh, float(scores_np[i]), emb, self.num_cls, cls_id))
        return dets

    def _collect_tracks(self, online_targets_dict: dict):
        """Single-pass: collect active tracks filtered by min_box_area."""
        tlwhs  = defaultdict(list)
        tids   = defaultdict(list)
        scores = defaultdict(list)
        for cls_id, tracks in online_targets_dict.items():
            for t in tracks:
                w, h = t.curr_tlwh[2], t.curr_tlwh[3]
                if w * h > self.min_area:
                    tlwhs[cls_id].append(t.curr_tlwh)
                    tids[cls_id].append(t.track_id)
                    scores[cls_id].append(t.score)
        return tlwhs, tids, scores

    def _visualize(self, img0, tlwhs, tids, scores, frame_id, save_dir, show_image):
        online_im = vis.plot_tracks(
            image        = img0,
            tlwhs_dict   = tlwhs,
            obj_ids_dict = tids,
            num_classes  = self.num_cls,
            scores       = scores,
            frame_id     = frame_id,
            fps          = 1.0 / max(1e-5, self.timer.average_time),
        )
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir:
            cv2.imwrite(osp.join(save_dir, f'{frame_id:05d}.jpg'), online_im)

    # ------------------------------------------------------------------

    def run(self, data_loader, result_path: str,
            save_dir=None, show_image: bool = False):
        """
        Run one sequence end-to-end.
        Writes MOT result file incrementally (no RAM accumulation).
        Returns (num_frames, avg_time_per_frame, n_timed_frames).
        """
        if save_dir:
            mkdir_if_missing(save_dir)

        self.tracker.reset()   # clear state from previous sequence
        self.timer       = Timer()
        self._orig_sizes = None
        counter          = 0

        with open(result_path, 'w') as f_out:
            for meta, img, img0 in data_loader:
                counter += 1
                # COCO loader trả frame_id THẬT (int) để khớp GT của Evaluator;
                # LoadImagesForTracking trả img_path (str) → dùng bộ đếm tuần tự.
                frame_id = meta if isinstance(meta, int) else counter
                orig_h, orig_w = img0.shape[:2]

                # Cache orig_sizes — constant within a sequence
                if self._orig_sizes is None:
                    self._orig_sizes = torch.tensor(
                        [[orig_h, orig_w]], device=self.device)

                blob = torch.from_numpy(img[None]).to(self.device)

                self.timer.tic()
                with torch.no_grad():
                    output = self.model(blob)
                    res    = self.postprocessor(output, self._orig_sizes)[0]
                    dets   = self._decode_detections(res)
                self.timer.toc()

                self.tracker.set_image(img0)
                online_targets = self.tracker.update(
                    dets, h_orig=orig_h, w_orig=orig_w)

                tlwhs, tids, tscores = self._collect_tracks(online_targets)

                # Incremental write — no results_dict accumulation
                for cls_id in range(self.num_cls):
                    if cls_id in self.skip_set:
                        continue
                    for tlwh, tid, sc in zip(
                            tlwhs[cls_id], tids[cls_id], tscores[cls_id]):
                        if tid < 0:
                            continue
                        f_out.write(_RESULT_FMT.format(
                            frame  = frame_id,
                            id     = tid + cls_id * _CLS_ID_OFFSET,
                            x1=tlwh[0], y1=tlwh[1], w=tlwh[2], h=tlwh[3],
                            score  = sc,
                            cls_id = cls_id + 1,
                        ))

                if show_image or save_dir:
                    self._visualize(img0, tlwhs, tids, tscores,
                                    frame_id, save_dir, show_image)

                if frame_id % 30 == 0:
                    logger.info('Frame %d  %.2f fps', frame_id,
                                1.0 / max(1e-5, self.timer.average_time))

        logger.info('Saved results → %s', result_path)
        return frame_id, self.timer.average_time, self.timer.calls


# ---------------------------------------------------------------------------
# main — iterate over sequences, evaluate with motmetrics
# ---------------------------------------------------------------------------

def main(opt, data_root: str, ann_root: str, seqs,
         exp_name: str, save_images: bool = False, show_image: bool = False):
    logger.setLevel(logging.INFO)

    result_root = osp.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)

    runner      = SequenceRunner(opt)
    accs        = []
    timer_avgs  = []
    timer_calls = []

    for seq in seqs:
        logger.info('start seq: %s', seq)

        output_dir = (osp.join(data_root, '..', 'outputs', exp_name, seq)
                      if save_images else None)

        dataloader      = LoadImagesForTracking(osp.join(data_root, seq), opt.img_size)
        result_filename = osp.join(result_root, f'{seq}.txt')

        nf, ta, tc = runner.run(
            dataloader, result_filename,
            save_dir=output_dir, show_image=show_image,
        )
        timer_avgs.append(ta)
        timer_calls.append(tc)

        logger.info('Evaluate seq: %s', seq)
        accs.append(Evaluator(ann_root, seq, 'mot').eval_file(result_filename))

    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time    = np.dot(timer_avgs, timer_calls)
    avg_fps     = 1.0 / max(all_time / max(np.sum(timer_calls), 1), 1e-5)
    logger.info('Total time %.2fs  FPS %.2f', all_time, avg_fps)

    metrics    = mm.metrics.motchallenge_metrics
    mh         = mm.metrics.create()
    summary    = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters = mh.formatters,
        namemap    = mm.io.motchallenge_metric_names,
    )
    print(strsummary)
    Evaluator.save_summary(summary, osp.join(result_root, f'summary_{exp_name}.xlsx'))


def _summarize(accs, seqs, timer_avgs, timer_calls, result_root, exp_name):
    """In FPS + bảng motmetrics + lưu xlsx (dùng chung cho mọi nguồn dữ liệu)."""
    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time    = np.dot(timer_avgs, timer_calls)
    avg_fps     = 1.0 / max(all_time / max(np.sum(timer_calls), 1), 1e-5)
    logger.info('Total time %.2fs  FPS %.2f', all_time, avg_fps)

    metrics    = mm.metrics.motchallenge_metrics
    mh         = mm.metrics.create()
    summary    = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters = mh.formatters,
        namemap    = mm.io.motchallenge_metric_names,
    )
    print(strsummary)
    Evaluator.save_summary(summary, osp.join(result_root, f'summary_{exp_name}.xlsx'))


# ---------------------------------------------------------------------------
# main_coco — nguồn dữ liệu lấy TỪ COCO JSON (đồng bộ với coco_detection/training)
# ---------------------------------------------------------------------------

def main_coco(opt, ann_file: str, img_root: str, gt_root: str,
              exp_name: str, save_images: bool = False, show_image: bool = False):
    """
    Tracking dùng đúng dataset của training:
      * ảnh + danh sách seq/frame lấy từ COCO JSON (ann_file, img_root)
      * tiền xử lý y hệt val (plain resize + /255, no letterbox)
      * GT để đánh giá vẫn đọc từ annotation VisDrone thô (gt_root/<seq>.txt) —
        toạ độ trùng với COCO nên MOTA/IDF1 không đổi; frame_id khớp do loader
        trả về frame_id thật.
    """
    logger.setLevel(logging.INFO)

    result_root = osp.join(osp.dirname(img_root.rstrip('/')), 'results', exp_name)
    mkdir_if_missing(result_root)

    src    = LoadCocoSequencesForTracking(ann_file, img_root, opt.img_size)
    runner = SequenceRunner(opt)
    accs, timer_avgs, timer_calls = [], [], []

    for seq in src.seqs:
        logger.info('start seq: %s (%d frames)', seq, src.num_frames(seq))

        output_dir = (osp.join(result_root, 'outputs', seq) if save_images else None)
        dataloader      = src.sequence(seq)
        result_filename = osp.join(result_root, f'{seq}.txt')

        nf, ta, tc = runner.run(
            dataloader, result_filename,
            save_dir=output_dir, show_image=show_image,
        )
        timer_avgs.append(ta)
        timer_calls.append(tc)

        logger.info('Evaluate seq: %s', seq)
        accs.append(Evaluator(gt_root, seq, 'mot').eval_file(result_filename))

    _summarize(accs, src.seqs, timer_avgs, timer_calls, result_root, exp_name)

if __name__ == '__main__':
    opt = opts().init()

    eval_mode = getattr(opt, 'eval_mode', '10class')
    set_eval_mode(eval_mode)
    _mode_info = {
        '10class': 'all 10 VisDrone classes (official)',
        '5class':  'pedestrian / car / van / truck / bus (AMOT protocol)',
        '4class':  'person / car / motorcycle / bicycle (competition)',
    }
    print(f'[Eval] {eval_mode}: {_mode_info[eval_mode]}')

    opt.device = f'cuda:{opt.gpus[0]}' if opt.gpus[0] >= 0 else 'cpu'

    print('Creating model...')
    opt.model = create_model(opt.arch, opt)
    opt.model = load_model(opt.model, opt.load_model)
    opt.model = opt.model.to(opt.device).eval()

    if not opt.test_visdrone:
        raise ValueError('No test dataset configured. Set --test_visdrone.')

    if getattr(opt, 'track_from_coco', False):
        # ── Nguồn tracking = COCO JSON (đồng bộ với coco_detection/training) ──
        import json as _json
        ann_file = opt.track_ann_file
        img_root = opt.track_img_root
        if not ann_file or not img_root:
            with open(opt.data_cfg) as _f:
                _cfg = _json.load(_f)
            ann_file = ann_file or _cfg['val_ann']   # COCO JSON chỉ có train/val
            img_root = img_root or _cfg['val_img']
        gt_root = opt.track_gt_root
        if not gt_root:
            raise ValueError(
                'Cần --track_gt_root: thư mục annotation VisDrone thô của split '
                'tracking (vd .../VisDrone2019-MOT-val/annotations) để Evaluator dựng GT.')
        main_coco(
            opt,
            ann_file    = ann_file,
            img_root    = img_root,
            gt_root     = gt_root,
            exp_name    = 'ecdet_visdrone',
            show_image  = False,
            save_images = True,
        )
    else:
        # ── Fallback: nguồn thô từ test_dev/sequences (KHÔNG đồng bộ blackout) ──
        data_root = osp.join(opt.data_dir, 'VisDrone2019/test_dev/sequences')
        ann_root  = osp.join(opt.data_dir, 'VisDrone2019/test_dev/annotations')

        main(
            opt,
            data_root   = data_root,
            ann_root    = ann_root,
            seqs        = _VISDRONE_SEQS,
            exp_name    = 'ecdet_visdrone',
            show_image  = False,
            save_images = True,
        )