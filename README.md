# FalconMOT

UAV edge multi-object **detection + tracking**. FalconJDE detector (DINOv3-STA
backbone → hybrid encoder → DETR-style decoder + ReID head) with a clean online
multi-class tracker.

## Structure

```
FalconMOT/
├── falconmot/                 # the Python package (import falconmot)
│   ├── opts.py                # CLI / config
│   ├── logger.py
│   ├── models/                # FalconJDE model + criterion + postprocessor
│   │   └── falcon_jde/        #   backbone (dinov3), hybrid_encoder, decoder, ...
│   ├── datasets/              # data loading (jde / coco), augmentation
│   ├── engine/                # training (trainers, factory)
│   ├── tracker/               # online tracker (base, track, association, falcon_tracker)
│   ├── tracking_utils/        # kalman, gmc, eval, io, visualization
│   ├── optim/                 # schedulers
│   ├── utils/                 # image / post-process / coco-eval helpers
│   └── cfg/                   # dataset json configs
├── scripts/                   # entrypoints (run these)
│   ├── train.py  evaluate.py  track.py  export_onnx.py  infer_onnx.py
│   └── gen_dataset_visdrone*.py
├── tools/                     # profiling / counting / visualization helpers
└── pyproject.toml
```

## Install / run

```bash
# editable install (recommended)
pip install -e .
python scripts/train.py --arch falcon_jde ...

# or run without installing — scripts bootstrap sys.path via scripts/_paths.py
python scripts/train.py ...
```

Detection losses: `loss_cls` + `loss_bbox` + `loss_giou` (DETR-style).
Tracking: see `falconmot/tracker/ARCHITECTURE.md`.
