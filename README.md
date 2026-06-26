# FalconMOT

**FalconMOT** is a DINOv3-backbone joint detection-and-embedding (JDE) tracker
for multi-object tracking (MOT) in drone / UAV video. It targets the
[VisDrone](http://aiskyeye.com/) and [UAVDT](https://sites.google.com/view/grli-uavdt/)
benchmarks, where scenes are crowded with small, fast-moving objects seen from a
moving camera.

A single network — **FalconJDE** — performs detection and appearance-embedding
extraction in one forward pass, and a lightweight multi-class tracker associates
detections across frames.

## Highlights

- **DINOv3 ViT backbone** with a spatial-prior adapter, a hybrid encoder, and a
  DEIM / D-FINE-style decoder for accurate small-object detection.
- **Decoupled ReID head** producing dense **Query Appearance-Motion (QAM)**
  embeddings, so association uses appearance *and* motion cues.
- **Global Motion Compensation (GMC)** to stabilise association under aggressive
  camera motion.
- **Unified inference tool** (`tools/track.py`) covering video files, image
  sequences, and real-time webcam / stream input, with optional trajectory
  smoothing and MOT-Challenge result export.
- Reproducible **training and evaluation** scripts for VisDrone-DET, plus the
  5-class benchmark and 4-class competition MOT protocols.

## Repository layout

```
FalconMOT/
├── configs/                 # dataset / experiment JSON configs
├── falconmot/               # the installable package
│   ├── models/              # FalconJDE model wiring (+ falcon_jde/ internals)
│   ├── tracker/             # multi-class JDE tracker + class remapping
│   ├── datasets/            # COCO-format dataset + augmentation pipeline
│   ├── engine/              # trainer / detection / mot loops
│   ├── optim/               # optimizer / schedule helpers
│   ├── utils/               # detection & MOT evaluation utilities
│   └── tracking_utils/      # timer, visualization, I/O helpers
└── tools/                   # command-line entry points
    ├── train.py             # training (detection stage 1, full JDE stage 2)
    ├── detect.py            # detection-only demo (image / folder)
    ├── track.py             # unified tracking (video / images / realtime)
    ├── eval_det.py          # VisDrone-DET COCO AP @ maxDets=500
    ├── eval_det_visdrone.py # VisDrone-correct detection mAP (ignore regions)
    ├── eval_mot_4cls.py     # MOTMetrics eval, 4-class competition split
    ├── eval_mot_5cls.py     # MOTMetrics eval, 5-class benchmark split
    ├── export_onnx.py       # export the detector to ONNX
    ├── infer_onnx.py        # run inference with an ONNX model
    └── datasets/            # VisDrone -> COCO/JDE dataset converters
```

## Installation

```bash
# 1. (recommended) create an environment
python -m venv .venv && source .venv/bin/activate

# 2. install a CUDA-matched PyTorch build first if you need GPU support
#    see https://pytorch.org for the right command for your CUDA version

# 3. install FalconMOT and its dependencies
pip install -e .
```

The tools also run without installing the package — `tools/_paths.py` adds the
repo root to `sys.path` automatically.

## Data preparation

The model trains on COCO-format JSON annotations. Converters for the raw
VisDrone splits live in `tools/datasets/`:

```bash
# VisDrone-MOT -> COCO JSON (7-class training schema)
python tools/datasets/visdrone2coco_7cls_mot.py \
    --visdrone_root /data/VisDrone2019-MOT \
    --output_root /data/visdrone_coco --splits train val

# VisDrone-DET -> COCO JSON
python tools/datasets/visdrone2coco_7cls_det.py \
    --images_dir /data/VisDrone2019-DET/train/images \
    --ann_dir    /data/VisDrone2019-DET/train/annotations \
    --out        /data/visdrone_det_coco/train.json \
    --out_images_dir /data/visdrone_det_coco/train/images
```

Point the relevant config in `configs/` (e.g. `configs/visdrone_coco.json`) at
your converted dataset root.

## Training

Training has two stages: a detection-only warm-up, then full JDE training that
adds the ReID head.

```bash
# Stage 1 — detection only
python tools/train.py --arch falcon_jde --exp_id visdrone_s1 \
    --data_cfg configs/visdrone_coco.json \
    --input-wh 1088 640 --num_queries 500 \
    --batch_size 8 --num_epochs 30 --train_single_det

# Stage 2 — full JDE (load the stage-1 checkpoint, drop --train_single_det)
python tools/train.py --arch falcon_jde --exp_id visdrone_s2 \
    --data_cfg configs/visdrone_coco.json \
    --input-wh 1088 640 --num_queries 500 \
    --load_model exp/mot/visdrone_s1/model_last.pth
```

## Inference

### Detection demo (image or folder)

```bash
python tools/detect.py --arch falcon_jde \
    --load_model exp/mot/visdrone_s2/model_best.pth \
    --input_path /data/demo_images --output_dir out/detect \
    --input-wh 1088 640 --conf_thres 0.4
```

### Tracking (unified)

`tools/track.py` auto-detects the input mode from `--source`; override with
`--mode {auto,video,images,realtime}`.

```bash
# Video file -> smooth annotated video
python tools/track.py --source demo.mp4 --output_video out/demo_tracked.mp4 \
    --arch falcon_jde --load_model exp/mot/visdrone_s2/model_best.pth \
    --input-wh 1088 640 --use_appearance_motion

# Folder of frames (image sequence)
python tools/track.py --source /data/seq/images --fps 30 \
    --load_model model_best.pth --input-wh 1088 640

# Real-time webcam (camera 0): live window + saved video
python tools/track.py --source 0 --mode realtime --show \
    --output_video out/webcam.mp4 --load_model model_best.pth
```

Useful tracking flags: `--save_mot results.txt` (MOT-Challenge output),
`--min_track_len`, `--max_interp_gap`, `--smooth_window` (offline trajectory
smoothing), and `--letterbox` (letterbox instead of plain resize).

## Evaluation

```bash
# Detection AP (COCO @ maxDets=500, VisDrone standard)
python tools/eval_det.py --arch falcon_jde --load_model model_best.pth \
    --img_dir /data/visdrone/val/images \
    --gt_json /data/visdrone/val/annotations/instances_val.json \
    --input-wh 1088 640 --num_queries 500

# MOT metrics (IDF1 / MOTA) on the 5-class benchmark
python tools/eval_mot_5cls.py --load_model model_best.pth --input-wh 1088 640

# MOT metrics on the 4-class competition split
python tools/eval_mot_4cls.py --load_model model_best.pth --input-wh 1088 640
```

## ONNX export

```bash
python tools/export_onnx.py --arch falcon_jde --load_model model_best.pth \
    --input-wh 1088 640 --onnx_path model.onnx
python tools/infer_onnx.py --model model.onnx --source /data/demo_images
```

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Citation

If you use FalconMOT in your research, please cite it (see
[CITATION.cff](CITATION.cff)).
