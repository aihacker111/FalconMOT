"""utils/eval.py — merged from jde_eval.py, coco_eval.py, post_process.py"""
"""
COCO mAP evaluation utilities for detection (VisDrone / COCO JSON format).

CocoJsonEvaluator — for VisDroneCocoDataset (COCO JSON format).
Loads GT directly from COCO JSON using the real image_id, like DEIMv2.
Outputs boxes with 1-indexed category_id to match the COCO convention.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os
import json
import tempfile
import numpy as np
import torch
import contextlib
import copy
from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util

from falconmot.utils import dist as dist_utils
from .image import transform_preds

__all__ = ['CocoEvaluator',]


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
    """Print the full AP/AR table following the VisDrone protocol."""
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
# 1. CocoJsonEvaluator — DEIMv2-style, uses COCO JSON GT directly
# ===========================================================================

class CocoJsonEvaluator:
    """
    COCO mAP evaluator cho VisDroneCocoDataset.

    Uses GT from the COCO JSON file (not rebuilt from batches), matching
    the way DEIMv2 evaluates exactly.

    Expects the batch dict to have a 'coco_image_id' key (set in VisDroneCocoDataset).

    Predictions from FalconJDEPostProcessor:
        result['boxes']  — xyxy pixel trong original image space  ✓
        result['labels'] — 0-indexed class id -> must be +1 for COCO's 1-indexed ids
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

        # NOTE: do not assign to self.coco_gt.params because it does not exist
        coco_eval = self.COCOeval(self.coco_gt, coco_dt, 'bbox')

        # Assign only to the freshly created eval object
        coco_eval.params.maxDets = [1, 10, 500]   # VisDrone protocol

        coco_eval.evaluate()
        coco_eval.accumulate()
        # Do NOT call coco_eval.summarize() — it prints "AP@maxDets=100 = -1"
        # hardcoded inside pycocotools/faster_coco_eval regardless of params.maxDets.
        # Instead, read directly from the precision/recall arrays via _visdrone_ap().
        metrics = _visdrone_ap(coco_eval, max_det=500)

        # Print the result table following the VisDrone protocol (maxDets=500)
        _print_metrics(metrics)

        self.reset()
        return metrics

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.params.maxDets = [1, 10, 500]
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()


def ctdet_post_process(dets, c, s, h, w, num_classes):
    """
    :param dets:
    :param c: center
    :param s: scale
    :param h: height
    :param w: width
    :param num_classes:
    :return:
    """
    # dets: batch x max_dets x dim
    # return 1-based class det dict

    ret = []
    for i in range(dets.shape[0]):  # each image in the batch
        top_preds = {}  # result dict(key: class_id(start from 0), value: obj_num×5)
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]

        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j] = dets[i, inds, :]

        ret.append(top_preds)

    return ret
