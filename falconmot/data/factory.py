from __future__ import absolute_import, division, print_function

from .dataset import VisDroneCocoDataset


def get_dataset(dataset: str, task: str):
    """Return dataset class for (dataset, task).

    dataset: 'coco' — COCO JSON format (VisDrone, gen_dataset_visdrone_coco.py)
    task:    'mot'  — multi-object tracking (detection + ReID)
    """
    if task == 'mot' and dataset == 'coco':
        return VisDroneCocoDataset
    raise ValueError(f"Unsupported (dataset={dataset!r}, task={task!r}); only ('coco','mot') is supported.")
