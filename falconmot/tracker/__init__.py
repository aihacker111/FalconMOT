"""FalconMOT tracking package.

Exposes the multi-class tracker and its tracklet class. The `FalconTracker` /
`Track` aliases are provided so callers can use either the historical
`MCJDETracker` / `MCTrack` names or the FalconMOT-branded names.
"""

from .basetrack import MCBaseTrack, TrackState
from .multitracker import MCJDETracker, MCTrack, id2cls

# Branded aliases
FalconTracker = MCJDETracker
Track = MCTrack

__all__ = [
    'MCJDETracker', 'MCTrack',
    'FalconTracker', 'Track',
    'MCBaseTrack', 'TrackState',
    'id2cls',
]
