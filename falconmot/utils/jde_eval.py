"""
COCO mAP evaluation utilities for detection (VisDrone / COCO JSON format).

CocoJsonEvaluator — dành cho VisDroneCocoDataset (COCO JSON format).
Load GT trực tiếp từ COCO JSON, dùng real image_id, giống DEIMv2.
Output boxes 1-indexed category_id để khớp COCO convention.
"""

import os
import json
import tempfile
import numpy as np
import torch


def _visdrone_ap(coco_eval, max_det=500):
    """Extract AP/AR at maxDets=max_det directly from the precision/recall
    arrays — avoids pycocotools' summarize() defaulting AP to maxDets=100
    (which returns -1 when 100 is not in params.maxDets)."""
    p = coco_eval.params
    try:
        m = list(p.maxDets).index(max_det)
    except ValueError:
        m = len(p.maxDets) - 1
    m1  = 0
    m10 = min(1, len(p.maxDets) - 1)
    prec = coco_eval.eval['precision']     # [T,R,K,A,M]
    rec  = coco_eval.eval['recall']        # [T,K,A,M]
    def _m(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float('nan')
    i50 = 0
    i75 = min(5, prec.shape[0] - 1)
    return {
        'AP':   _m(prec[:, :, :, 0, m]),
        'AP50': _m(prec[i50, :, :, 0, m]),
        'AP75': _m(prec[i75, :, :, 0, m]),
        'APs':  _m(prec[:, :, :, 1, m]),
        'APm':  _m(prec[:, :, :, 2, m]),
        'APl':  _m(prec[:, :, :, 3, m]),
        'AR':   _m(rec[:, :, 0, m]),
        'AR1':  _m(rec[:, :, 0, m1]),
        'AR10': _m(rec[:, :, 0, m10]),
        'ARs':  _m(rec[:, :, 1, m]),
        'ARm':  _m(rec[:, :, 2, m]),
        'ARl':  _m(rec[:, :, 3, m]),
    }


def _print_metrics(m, max_det=500):
    """In bảng AP/AR đầy đủ theo VisDrone protocol."""
    d = max_det
    print(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={d} ] = {m['AP']:.3f}")
    print(f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets={d} ] = {m['AP50']:.3f}")
    print(f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets={d} ] = {m['AP75']:.3f}")
    print(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets={d} ] = {m['APs']:.3f}")
    print(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets={d} ] = {m['APm']:.3f}")
    print(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets={d} ] = {m['APl']:.3f}")
    print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {m['AR1']:.3f}")
    print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {m['AR10']:.3f}")
    print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={d} ] = {m['AR']:.3f}")
    print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets={d} ] = {m['ARs']:.3f}")
    print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets={d} ] = {m['ARm']:.3f}")
    print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets={d} ] = {m['ARl']:.3f}")


# ---------------------------------------------------------------------------
# Helper: convert xyxy → xywh (cho COCO result format)
# ---------------------------------------------------------------------------

def _xyxy_to_xywh(boxes: torch.Tensor) -> list:
    """(K, 4) xyxy pixel → list of [x, y, w, h]"""
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1).tolist()


# ===========================================================================
# 1. CocoJsonEvaluator — DEIMv2-style, dùng COCO JSON GT trực tiếp
# ===========================================================================

class CocoJsonEvaluator:
    """
    COCO mAP evaluator cho VisDroneCocoDataset.

    Dùng GT từ COCO JSON file (không rebuild từ batch), khớp 1-1 với
    cách DEIMv2 evaluate.

    Expects batch dict có key 'coco_image_id' (set trong VisDroneCocoDataset).

    Predictions từ FalconJDEPostProcessor:
        result['boxes']  — xyxy pixel trong original image space  ✓
        result['labels'] — 0-indexed class id → phải +1 cho COCO 1-indexed
        result['scores'] — float confidence
    """

    def __init__(self, ann_file: str, iou_types=('bbox',)):
        try:
            from faster_coco_eval import COCO, COCOeval_faster as COCOeval
            self._coco_module = 'faster'
        except ImportError:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            self._coco_module = 'pycocotools'

        self.COCO    = COCO
        self.COCOeval = COCOeval

        self.coco_gt  = COCO(ann_file)
        self.iou_types = list(iou_types)
        self._dt_anns  = []   # accumulated detection results

    def reset(self):
        self._dt_anns = []

    def update(self, dt_results: list, batch: dict):
        """
        Args:
            dt_results    : list[dict] from FalconJDEPostProcessor (one per image)
            batch         : batch dict — must contain 'coco_image_id'
        """
        img_ids = batch['coco_image_id']   # (B,) int64

        for b, res in enumerate(dt_results):
            img_id = int(img_ids[b].item()) if torch.is_tensor(img_ids[b]) \
                     else int(img_ids[b])

            if len(res['scores']) == 0:
                continue

            boxes_xywh = _xyxy_to_xywh(res['boxes'].cpu())
            scores     = res['scores'].cpu().tolist()
            labels     = res['labels'].cpu().tolist()

            for k in range(len(scores)):
                self._dt_anns.append({
                    'image_id':    img_id,
                    'category_id': int(labels[k]) + 1,   # 0-indexed → 1-indexed (COCO)
                    'bbox':        boxes_xywh[k],
                    'score':       float(scores[k]),
                })

    def summarize(self) -> dict:
        if not self._dt_anns:
            print('[CocoJsonEvaluator] No detections — mAP = 0')
            self.reset()
            return {'AP': 0.0, 'AP50': 0.0}

        coco_dt   = self.coco_gt.loadRes(self._dt_anns)
        
        # SỬA Ở ĐÂY: Không gán vào self.coco_gt.params vì nó không tồn tại
        coco_eval = self.COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        # Chỉ gán vào đối tượng eval vừa khởi tạo
        coco_eval.params.maxDets = [1, 10, 500]   # VisDrone protocol
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        # KHÔNG gọi coco_eval.summarize() — nó in dòng "AP@maxDets=100 = -1"
        # hardcode bên trong pycocotools/faster_coco_eval bất kể params.maxDets.
        # Thay vào đó đọc trực tiếp từ precision/recall arrays qua _visdrone_ap().
        metrics = _visdrone_ap(coco_eval, max_det=500)

        # In bảng kết quả theo VisDrone protocol (maxDets=500)
        _print_metrics(metrics)
            
        self.reset()
        return metrics