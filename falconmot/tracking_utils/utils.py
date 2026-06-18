"""Small filesystem helper used across the tracking pipeline.

The original module was a large YOLOv3/JDE grab-bag (anchor encoding,
non-max-suppression, AP computation, etc.) that is no longer used by this
transformer-based pipeline. Only `mkdir_if_missing` was actually referenced,
so the rest has been removed.
"""

import os


def mkdir_if_missing(d):
    """Create directory `d` (and parents) if it does not already exist."""
    if d:
        os.makedirs(d, exist_ok=True)
