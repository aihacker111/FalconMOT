"""
base.py — track lifecycle primitives for the FalconMOT tracker.

A single, per-class track-id counter lives on ``BaseTrack`` so that ids are
unique *within* a class (VisDrone uses per-class ids: pedestrian #1 ≠ car #1).
"""

from collections import defaultdict

import numpy as np


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Per-class track-id allocation + minimal lifecycle state."""

    _count = defaultdict(int)

    track_id = 0
    is_activated = False
    state = TrackState.New

    score = 0.0
    start_frame = 0
    frame_id = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id(cls_id: int) -> int:
        BaseTrack._count[cls_id] += 1
        return BaseTrack._count[cls_id]

    @staticmethod
    def reset_counts(num_classes: int) -> None:
        for cls_id in range(num_classes):
            BaseTrack._count[cls_id] = 0

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


# VisDrone class id → name (kept here so the whole codebase has one source).
ID2CLS = {
    0: 'pedestrian', 1: 'people',   2: 'bicycle',          3: 'car',
    4: 'van',        5: 'truck',    6: 'tricycle',          7: 'awning-tricycle',
    8: 'bus',        9: 'motor',
}
