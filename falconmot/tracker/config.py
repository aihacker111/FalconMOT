"""Tracker configuration.

A single lean dataclass replaces the ~15 `getattr(opt, 'am_*', ...)` lookups
that used to live in `MCJDETracker.__init__`. Hyper-parameters that are purely
*internal* to a sub-module (QAM softmax temperature, entropy sharpness, …) are
kept here but consumed only by `MotionModel`, so they never appear on the
tracker's public surface.

`TrackerCfg.from_opt(opt)` maps the legacy argparse namespace onto these
fields, so existing launch scripts keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackerCfg:
    # ── association weights ───────────────────────────────────────────────
    appearance_weight: float = 1.0      # was am_w_app
    iou_weight: float = 1.0             # was am_w_iou

    # ── first-association gates / thresholds ──────────────────────────────
    match_thresh: float = 0.7          # cost ceiling, stage-1 (QAM) assignment
    proximity_gate: float = 0.95       # iou_dist above this needs motion to vouch
    motion_gate: float = 0.9           # motion can vouch only below this

    # ── second & unconfirmed IoU stages ───────────────────────────────────
    iou_thresh: float = 0.8            # stage-2 IoU ceiling
    unconfirmed_thresh: float = 0.5

    # ── NEW: appearance-only re-identification of lost tracks ─────────────
    # Relative gate (scale-free): a revival is accepted on how much the best
    # appearance match *beats the runner-up*, not on an absolute cosine number —
    # so it transfers across sequences/datasets without re-tuning.
    reid_ratio: float = 0.80           # Lowe ratio: best_dist <= ratio * second_best
    reid_gate_max: float = 0.50        # loose absolute backstop (safety net only)
    reid_mutual: bool = True           # require mutual nearest-neighbour
    reid_area_gate: float = 4.0        # reject pairs whose area ratio exceeds this

    # ── lifecycle ─────────────────────────────────────────────────────────
    max_lost: int = 60                 # frames a lost track survives (was 30)

    # ── appearance-bank hygiene (adaptive EMA, "dynamic appearance") ──────
    # The embedding absorbs a new observation in proportion to its *quality*
    #   q = score * (1 - occlusion);   gain = q * app_gain_max
    # A low-score or occluded detection has q≈0 → embedding barely moves (this
    # replaces the old skip/score/occ hard gates with one continuous knob).
    app_gain_max: float = 0.10         # max EMA absorption at quality==1 (≈ old 1-momentum)
    app_occ_power: float = 1.0         # >1 punishes partial occlusion more sharply

    # ── motion model toggles ──────────────────────────────────────────────
    use_qam: bool = True               # appearance-as-motion cue (was use_appearance_motion)
    use_nsa: bool = True               # confidence-scaled Kalman update
    use_oru: bool = True               # observation-centric re-update on revival
    use_ocm: bool = False              # velocity-direction consistency cost (ablation)
    use_gmc: bool = True
    legacy_fuse: bool = False          # multiplicative fuse_score_three (A/B only)

    # ── QAM internals (consumed by MotionModel, hidden from tracker API) ──
    qam_tau: float = 0.07
    qam_kappa: float = 0.10
    qam_beta: float = 4.0

    # ── OCM internals ─────────────────────────────────────────────────────
    ocm_weight: float = 0.20

    @classmethod
    def from_opt(cls, opt) -> "TrackerCfg":
        """Build a config from the legacy argparse namespace (back-compatible)."""
        g = lambda name, default: getattr(opt, name, default)
        track_buffer = int(g('track_buffer', 30))
        frame_rate = int(g('frame_rate', 30))
        max_lost = int(frame_rate / 30.0 * track_buffer)
        # let an explicit --max_lost override the buffer-derived value
        max_lost = int(g('max_lost', max(max_lost, cls.max_lost)))
        return cls(
            appearance_weight=float(g('am_w_app', 1.0)),
            iou_weight=float(g('am_w_iou', 1.0)),
            match_thresh=float(g('match_thresh', 0.7)),
            proximity_gate=float(g('proximity_thresh', 0.95)),
            motion_gate=float(g('motion_gate', 0.9)),
            reid_ratio=float(g('reid_ratio', 0.80)),
            reid_gate_max=float(g('reid_gate_max', 0.50)),
            reid_mutual=bool(g('reid_mutual', True)),
            reid_area_gate=float(g('reid_area_gate', 4.0)),
            max_lost=max_lost,
            app_gain_max=float(g('app_gain_max', 0.10)),
            app_occ_power=float(g('app_occ_power', 1.0)),
            use_qam=bool(g('use_appearance_motion', False)),
            use_nsa=bool(g('use_nsa', True)),
            use_oru=bool(g('use_oru', True)),
            use_ocm=bool(g('use_ocm', False)),
            use_gmc=bool(g('use_gmc', True)),
            legacy_fuse=bool(g('legacy_fuse', False)),
            qam_tau=float(g('am_tau', 0.07)),
            qam_kappa=float(g('am_kappa', 0.10)),
            qam_beta=float(g('am_beta', 4.0)),
            ocm_weight=float(g('ocm_weight', 0.20)),
        )
