from __future__ import absolute_import, division, print_function

from .dataset.jde import JointDataset
from .dataset.coco_detection import VisDroneCocoDataset


def get_dataset(dataset: str, task: str):
    """
    Return dataset class for (dataset, task) combination.

    dataset:
        'jde'        — JDE format (.train index files + labels_with_ids/)
        'coco'       — COCO JSON format (produced by gen_dataset_visdrone_coco.py)
    task:
        'mot'        — multi-object tracking (detection + ReID)
    """
    if task == 'mot':
        if dataset == 'coco':
            return VisDroneCocoDataset
        return JointDataset   # default: jde format
    return None
