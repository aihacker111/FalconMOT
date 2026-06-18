# encoding=utf-8
"""Base classes for multi-class tracking.

Only the multi-class base (`MCBaseTrack`) is kept — it maintains a *per-class*
track-id counter so that, e.g., pedestrian #1 and car #1 are distinct objects.
The original single-object `BaseTrack` was unused and has been removed.
"""

from collections import OrderedDict, defaultdict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class MCBaseTrack(object):
    """Multi-class base track with a per-class id counter."""

    _count_dict = defaultdict(int)   # cls_id -> last assigned id

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera placeholder
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id(cls_id):
        MCBaseTrack._count_dict[cls_id] += 1
        return MCBaseTrack._count_dict[cls_id]

    @staticmethod
    def init_count(num_classes):
        """Reset the id counter for every object class."""
        for cls_id in range(num_classes):
            MCBaseTrack._count_dict[cls_id] = 0

    @staticmethod
    def reset_track_count(cls_id):
        MCBaseTrack._count_dict[cls_id] = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
