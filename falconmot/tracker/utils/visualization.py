"""Tracking visualisation helpers.

Only `plot_tracks` is kept — it draws coloured boxes with track ids and,
optionally, class names and confidence scores. The legacy plotting variants
were unused and have been removed.
"""

import cv2
import numpy as np


def mkdir_if_missing(d):
    """Create directory `d` (and parents) if it does not already exist."""
    if d:
        os.makedirs(d, exist_ok=True)

        
def get_color(idx):
    idx = idx * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def plot_tracks(image,
                tlwhs_dict,
                obj_ids_dict,
                num_classes,
                scores=None,
                frame_id=0,
                fps=0.0,
                cls_id2name=None):
    """Draw per-class tracks on a frame.

    Args:
        image        : BGR frame (numpy array).
        tlwhs_dict   : dict[cls_id] -> list of (x, y, w, h) boxes.
        obj_ids_dict : dict[cls_id] -> list of track ids.
        num_classes  : number of object classes.
        scores       : optional dict[cls_id] -> list of confidence scores.
        frame_id, fps: kept for API compatibility (not drawn).
        cls_id2name  : optional dict[cls_id] -> class name, drawn next to id.
    Returns:
        The annotated frame (a copy of `image`).
    """
    img = np.ascontiguousarray(np.copy(image))

    text_scale = max(0.8, image.shape[1] / 1600.0)
    text_thickness = 1
    line_thickness = min(2, int(image.shape[1] / 500.0))

    for cls_id in range(num_classes):
        cls_tlwhs = tlwhs_dict.get(cls_id, [])
        obj_ids = obj_ids_dict.get(cls_id, [])
        cls_scores = scores.get(cls_id, []) if isinstance(scores, dict) else None

        for i, tlwh_i in enumerate(cls_tlwhs):
            x1, y1, w, h = tlwh_i
            int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            color = get_color(abs(obj_id))

            label = str(obj_id)
            if cls_id2name is not None and cls_id in cls_id2name:
                label = '{} {}'.format(cls_id2name[cls_id], obj_id)
            if cls_scores is not None and i < len(cls_scores):
                label = '{} {:.2f}'.format(label, cls_scores[i])

            cv2.rectangle(img, int_box[0:2], int_box[2:4], color=color, thickness=line_thickness)

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)
            cv2.rectangle(img, (int(x1), int(y1)),
                          (int(x1) + text_w, int(y1) + text_h + 4),
                          color=color, thickness=-1)
            cv2.putText(img, label, (int(x1), int(y1) + text_h),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255),
                        thickness=text_thickness, lineType=cv2.LINE_AA)

    return img
