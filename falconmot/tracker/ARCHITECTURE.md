# FalconMOT — Tracking module

Clean, single-tracker design. Per-frame the model emits detections with a ReID
embedding per query; the tracker associates them into trajectories online.

## Package layout (`lib/tracker/`)

| file | role |
|------|------|
| `base.py` | `TrackState`, `BaseTrack` (per-class id counter), `ID2CLS` names |
| `track.py` | `Track` — one object: NSA-Kalman + time-stamped feature gallery + adaptive-EMA |
| `association.py` | matching primitives: IoU, decayed-gallery ReID, additive fusion, LAPJV/scipy assignment, NFM mutual-top-k gate, dedup |
| `falcon_tracker.py` | `FalconTracker` — per-class 4-stage online pipeline |
| `__init__.py` | public exports |

Kalman (`tracking_utils/kalman_filter.py`, with `update_nsa`) and global motion
compensation (`tracking_utils/gmc.py`) are reused as-is.

## Association pipeline (per class, per frame)

```
predict (Kalman) + GMC compensate
 ├ Stage 1  confirmed   vs all dets        IoU+ReID, NFM-gated   thr 0.60
 ├ Stage 2  remaining   vs remaining dets  IoU+ReID, NFM-gated   thr 0.70  → unmatched ⇒ lost
 ├ Stage 3  LOST pool   vs remaining dets  decayed-ReID, gated   thr 0.50  → re-activate
 ├ Stage 4  unconfirmed vs remaining dets  IoU+ReID              thr 0.50
 └ leftover high-conf dets ⇒ new tracks ; age-out lost ; appearance-aware dedup
```

## FusionTrack-inspired upgrades (arXiv:2505.18727), inference-only

* **Time-decayed feature memory** (TMP, `W = e^{-α·Δt}`) — gallery similarity is
  recency-weighted (`reid_decay_alpha`, default 0.02), so stale appearances
  count less. Re-association after occlusion stays robust.
* **Mutual top-k neighbour gating** (NFM) — an appearance match is trusted only
  if track and detection are each in the other's top-k nearest neighbours
  (`nfm_topk`=2, `use_nfm`=True). Cuts spurious ReID matches → higher IDF1, fewer ID switches.

## Removed (vs old code)

* Legacy `multitracker.py` / `multitracker_v2.py` duplicates → one `FalconTracker`.
* 800-line `matching.py` with dead dense-ReID/motion-attention paths → lean `association.py`.
* Dense-ReID feature-map path: FalconJDE outputs a per-query embedding, so no
  dense map / `ecdet_reid_motion` is needed.
* Orphaned `tracking_utils/nms.py`, `parse_config.py`.

## Tunable opts (optional)

`--reid_decay_alpha` (0.02), `--nfm_topk` (2), `--use_nfm` — all have safe defaults
read via `getattr`, so existing configs keep working.
