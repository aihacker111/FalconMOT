# FalconMOT — FalconJDE

**Joint Detection & Embedding (JDE) Multi-Object Tracking cho video drone / aerial (VisDrone, UAVDT, VTMOT).**

FalconJDE là một mô hình *query-based* (họ RT-DETR / DEIM) phát hiện vật thể **và** trích xuất đặc trưng tái nhận dạng (ReID) trong **một lần forward duy nhất**, kết hợp với một multi-class tracker có Kalman filter, bù chuyển động camera (GMC) và **Query Appearance-Motion (QAM)** để theo vết mượt, ít đổi ID.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Huấn luyện (2 stage)](#5-huấn-luyện-2-stage)
6. [Đánh giá (Detection mAP & Tracking MOTA/IDF1)](#6-đánh-giá)
7. [Inference](#7-inference)
8. [Export ONNX](#8-export-onnx)
9. [Sơ đồ class & các scheme remap](#9-sơ-đồ-class--các-scheme-remap)
10. [Các tham số quan trọng](#10-các-tham-số-quan-trọng)
11. [Ghi chú & FAQ](#11-ghi-chú--faq)

---

## 1. Tổng quan kiến trúc

```
                ┌─────────────────────────────────────────────────────────┐
   ảnh BGR ───► │  DINOv3STAs backbone  (ViT-tiny + STA adapter)          │
                │      │ feats {s8, s16, s32}  (+ s4 nếu --use_s4)         │
                │      ▼                                                    │
                │  HybridEncoder (DEIM)  — fuse đa tỉ lệ, AIFI/CCFM        │
                │      ▼                                                    │
                │  DEIMTransformer decoder  (300 queries, deformable attn) │
                │      │  pred_logits / pred_boxes (cxcywh chuẩn hoá)       │
                │      ▼                                                    │
                │  ReIDHead  — sample feature map tại mỗi box (deform attn) │
                │      │  pred_reid (embedding) + reid_dense (QAM)          │
                └──────┼──────────────────────────────────────────────────┘
                       ▼
            FalconJDEPostProcessor  → boxes (xyxy pixel) + scores + labels + reid
                       ▼
            MCJDETracker (FalconTracker) → online tracks (per-class id)
```

**Thành phần chính** (`falconmot/models/falcon_jde/`):

| Module | File | Vai trò |
|---|---|---|
| Backbone | `backbone/dinov3/`, `backbone/vit_tiny.py`, `backbone/dinov3_adapter.py` | DINOv3 ViT-tiny + **STA** (Spatial-aware adapter) sinh feature pyramid. |
| Encoder | `hybrid_encoder.py`, `feat_fusion.py` | Hợp nhất đa tỉ lệ kiểu DEIM. |
| Decoder | `decoder.py`, `dfine_decoder.py`, `deim_utils.py`, `dfine_utils.py` | Transformer decoder query-based + denoising training. |
| ReID head | (trong `model.py`) | Sample feature map tại box bằng deformable-attn → embedding; thêm `dense_appearance()` cho QAM. |
| Loss | `criterion.py`, `matcher.py`, `denoising.py`, `box_ops.py` | Hungarian matcher + detection loss + ReID loss, cân bằng bằng **uncertainty weights**. |
| Postprocess | `postprocessor.py` | Giải mã box (letterbox **hoặc** plain-resize) + lọc theo `conf_thres`. |

**Triết lý JDE "không xung đột gradient"** (xem chú thích trong `model.py`):
- Query (`hs`) và box được **detach** → chỉ làm *con trỏ* chỉ chỗ cần lấy đặc trưng, bảo vệ ngữ nghĩa localization/classification của detector.
- Feature map **vẫn nối gradient** (có thể scale bằng `--reid_grad_scale`) → gradient ReID chảy về encoder/backbone để trunk học đặc trưng định danh.
- Detection vs ReID được cân bằng bằng *learnable uncertainty weights* trong criterion (không hard stop-gradient).

**Tracker** (`falconmot/tracker/multitracker.py` — class `MCJDETracker`, alias `FalconTracker`):
- Đa lớp, mỗi lớp có dải `track_id` riêng (offset `cls_id * 1_000_000` khi ghi file).
- **Kalman filter** (`tracking_utils/kalman_filter.py`) dự đoán chuyển động.
- **GMC** (`tracking_utils/gmc.py`, mặc định `sparseOptFlow`) bù rung/chuyển động camera.
- **Query Appearance-Motion (QAM)**: tương quan template ngoại hình với *dense appearance map* để dự đoán tâm vật thể; chi phí phối hợp tính bằng `matching.fuse_loglik` (kết hợp appearance + IoU + motion với cổng entropy).

---

## 2. Cấu trúc thư mục

```
.
├── scripts/                         # Điểm vào (entry points) chạy bằng CLI
│   ├── _paths.py                    # bootstrap sys.path (import được "falconmot")
│   ├── train.py                     # Huấn luyện (stage-1 det / stage-2 tracking)
│   ├── train_stage1_det.sh          # Ví dụ lệnh train stage-1 detection
│   │
│   ├── evaluate.py                  # mAP trên VisDrone test-dev (maxDets=500, ignore-region)
│   ├── evaluate_det.py              # mAP VisDrone-DET (+ xuất file submission test-dev)
│   │
│   ├── inference.py                 # Detect trên ẢNH / thư mục ảnh (có --tile SAHI)
│   ├── inference_track.py           # ★ Tracking trên VIDEO + làm mượt (QAM + smoothing)
│   ├── track_4cls.py                # Tracking + MOTMetrics, remap 7→4 lớp (competition)
│   ├── track_5cls.py                # Tracking + MOTMetrics, remap 7→5 lớp (benchmark)
│   ├── track_ECDet.py               # Tracking + MOTMetrics, đầy đủ lớp
│   │
│   ├── export_onnx.py               # Xuất model sang ONNX
│   ├── infer_onnx.py                # Chạy ONNX (chỉ cần onnxruntime, không cần torch)
│   │
│   ├── gen_dataset_visdrone.py      # VisDrone-MOT → định dạng JDE (label .txt)
│   ├── gen_dataset_visdrone_coco.py # VisDrone-MOT → COCO JSON
│   ├── visdrone_det_to_coco.py      # VisDrone-DET → COCO JSON
│   ├── visdrone2coco_7cls_det.py    # COCO 7-lớp cho detection
│   ├── visdrone2coco_7cls_mot.py    # COCO 7-lớp cho MOT (merge 10→7 lớp)
│   ├── visdrone2coco_5cls_benchmark_mot.py   # GT 5-lớp benchmark
│   └── visdrone2coco_4cls_competition_mot.py # GT 4-lớp competition
│
└── falconmot/                       # Thư viện lõi
    ├── opts.py                      # TẤT CẢ tham số CLI (rất quan trọng)
    ├── logger.py
    ├── cfg/                         # Cấu hình dữ liệu (JSON): đường dẫn ảnh/ann
    │   ├── visdrone_coco_det.json   #   detection
    │   ├── visdrone_coco.json       #   MOT
    │   ├── UAVDT.json, VTMOT.json, ...
    │
    ├── models/
    │   ├── model.py                 # create_model / load_model / save_model
    │   ├── data_parallel.py, scatter_gather.py
    │   └── falcon_jde/              # (xem bảng ở mục 1)
    │
    ├── datasets/
    │   ├── dataset_factory.py
    │   ├── augment.py               # mosaic, copy-paste, gridmask, homography, ...
    │   └── dataset/                 # jde.py, coco_detection.py, ...
    │
    ├── engine/                      # Vòng lặp huấn luyện
    │   ├── base_trainer.py, mot.py, train_factory.py
    │   └── stage.py                 # Quản lý 2-phase fine-tuning của stage-2
    │
    ├── tracker/
    │   ├── multitracker.py          # MCJDETracker / MCTrack  (= FalconTracker / Track)
    │   ├── appearance_motion.py     # QAM: predict_centers, sample_dense, ...
    │   ├── matching.py              # iou/embedding distance, fuse_loglik
    │   ├── basetrack.py, class_remap.py
    │
    ├── tracking_utils/
    │   ├── kalman_filter.py, gmc.py
    │   ├── visualization.py         # vẽ box/id lên frame (plot_tracks)
    │   ├── coco_gt_reader.py, evaluation.py, io.py, timer.py, log.py
    │
    └── utils/
        ├── coco_eval.py, jde_eval.py, post_process.py, image.py, ...
```

---

## 3. Cài đặt môi trường

Yêu cầu: **Python ≥ 3.10**, **CUDA GPU** (khuyến nghị cho train/inference video).

```bash
# (khuyến nghị) tạo môi trường riêng
conda create -n falconmot python=3.10 -y
conda activate falconmot

# PyTorch (chọn bản CUDA phù hợp máy bạn, ví dụ cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Phụ thuộc chính
pip install opencv-python numpy tqdm \
            pycocotools motmetrics pandas openpyxl \
            onnx onnxruntime-gpu scipy lap cython_bbox
```

> **Lưu ý import:** mọi script trong `scripts/` đều `import _paths` ở dòng đầu để tự thêm thư mục gốc repo vào `sys.path`, nhờ đó `import falconmot...` hoạt động mà **không cần** `pip install -e .`. Hãy chạy script từ thư mục gốc repo.

---

## 4. Chuẩn bị dữ liệu

Project huấn luyện trên **VisDrone2019** (DET cho stage-1, MOT cho stage-2). Quy ước nhãn VisDrone gốc 10 lớp được **gộp về 7 lớp** khi train:

```
0:pedestrian  1:bicycle  2:car  3:truck  4:tricycle  5:bus  6:motor
```

### 4.1. Chuyển VisDrone-DET → COCO (cho stage-1)

```bash
python scripts/visdrone_det_to_coco.py        # hoặc visdrone2coco_7cls_det.py
```

### 4.2. Chuyển VisDrone-MOT → COCO (cho stage-2 / eval tracking)

```bash
# COCO JSON dùng để train MOT (gộp 10→7 lớp, xử lý track_id an toàn)
python scripts/visdrone2coco_7cls_mot.py \
    --visdrone_root /workspace \
    --output_root  /workspace/VisDrone2019-COCO-7cls \
    --splits train val --workers 8
```

**Quy tắc lọc nhãn** (đồng nhất giữa `gen_dataset_*` và `evaluate.py`):
- `score == 0` hoặc `cls_id ∈ {0, 11}` → **ignore region** (không phải GT, không tính FP).
- GT hợp lệ: `score == 1` và `1 ≤ cls_id ≤ 10`. **Không** lọc theo occlusion/truncation.

### 4.3. Sinh GT eval theo từng scheme

```bash
python scripts/visdrone2coco_5cls_benchmark_mot.py     # GT 5-lớp benchmark
python scripts/visdrone2coco_4cls_competition_mot.py   # GT 4-lớp competition
```

### 4.4. Cấu hình đường dẫn dữ liệu

Sửa các file JSON trong `falconmot/cfg/` cho khớp máy bạn, ví dụ `visdrone_coco_det.json`:

```json
{
  "root":      "/workspace/VisDrone2019-DET-COCO",
  "train_ann": "/workspace/VisDrone2019-DET-COCO/train/annotations/instances_train.json",
  "train_img": "/workspace/VisDrone2019-DET-COCO/train/images",
  "val_ann":   "/workspace/VisDrone2019-DET-COCO/val/annotations/instances_val.json",
  "val_img":   "/workspace/VisDrone2019-DET-COCO/val/images_val"
}
```

---

## 5. Huấn luyện (2 stage)

Pipeline khuyến nghị: **Stage-1 train detector** → **Stage-2 train JDE (thêm ReID)**.

### Stage-1 — Detection-only

Tắt hoàn toàn ReID head/loss bằng `--train_single_det`. Có sẵn script mẫu:

```bash
bash scripts/train_stage1_det.sh
# tương đương:
python scripts/train.py \
  --task mot --arch falcon_jde --dataset coco --train_single_det \
  --exp_id falcon_stage1_det \
  --data_cfg falconmot/cfg/visdrone_coco_det.json \
  --val_cfg  falconmot/cfg/visdrone_coco_det.json \
  --deim_pretrained pretrained_model/deimv2_dinov3_s_coco.pth \
  --input-wh 960 544 --eval_spatial_size 544 960 \
  --num_queries 300 --use_s4 \
  --batch_size 16 --grad_accum 4 --num_workers 16 --gpus 0 \
  --lr 5e-4 --num_epochs 60 \
  --lr_scheduler cosine --warmup_epochs 2 --no_aug_epochs 5 --lr_min_factor 0.01 \
  --mosaic --mosaic_prob 0.3
```

### Stage-2 — Tracking (JDE) với 2 phase tự động

Nạp checkpoint stage-1 bằng `--load_model`, bỏ `--train_single_det`. Bộ quản lý `engine/stage.py` tự chia 2 phase:

- **Phase 0 (ReID warmup):** đóng băng backbone + encoder + decoder (+ nhánh S4), **chỉ** train `reid_head` + ReID classifier. BatchNorm running-stats bị khoá → mAP không đổi.
- **Phase 1 (joint):** mở băng encoder + decoder (+ S4), backbone vẫn đóng băng; `id_weight` tăng dần `0 → mục tiêu` trong `--id_warmup_epochs` epoch đầu.

> Optimizer + LR scheduler được **tự dựng lại** ở ranh giới phase vì tập tham số trainable thay đổi.

```bash
python scripts/train.py \
  --task mot --arch falcon_jde --dataset coco \
  --exp_id falcon_stage2_track \
  --load_model exp/mot/falcon_stage1_det/model_best.pth \
  --data_cfg falconmot/cfg/visdrone_coco.json \
  --val_cfg  falconmot/cfg/visdrone_coco.json \
  --input-wh 864 480 --eval_spatial_size 480 864 \
  --use_s4 --reid_dim 128 \
  --reid_warmup_epochs 3 --id_warmup_epochs 3 \
  --tri --id_weight 1.0 \
  --batch_size 8 --num_epochs 30 --gpus 0
```

**Đánh giá tracking trong lúc train:** bật `--track_val` (xem mục 10) để theo dõi IDF1/MOTA mỗi vài epoch và chọn `model_best.pth` theo điểm kết hợp (`--track_val_w_idf1`, `--track_val_w_mota`).

---

## 6. Đánh giá

### 6.1. Detection mAP

```bash
# VisDrone-DET val (có GT) — COCO AP @ maxDets=500 (chuẩn VisDrone, KHÔNG phải 100)
python scripts/evaluate_det.py \
  --arch falcon_jde --load_model exp/.../model_best.pth \
  --img_dir /data/VisDrone2019-DET-COCO/val/images \
  --gt_json /data/VisDrone2019-DET-COCO/val/annotations/instances_val.json \
  --input-wh 1088 640 --num_queries 500 --use_s4
# Thêm --tile để bật sliced inference (SAHI) cho vật thể nhỏ.

# test-dev (không GT) → xuất file submission
python scripts/evaluate_det.py ... --export_dir submit/  # rồi zip & nộp lên server
```

`evaluate.py` là bản đánh giá mAP "đúng chuẩn VisDrone" (maxDets=500, bỏ qua ignore-region, không lọc truncation).

### 6.2. Tracking MOTA / IDF1 (motmetrics)

Chọn script theo scheme lớp đánh giá:

```bash
# 4 lớp competition (person/car/motorcycle/bicycle) — remap 7→4, drop truck/tricycle/bus
python scripts/track_4cls.py \
  --arch falcon_jde --load_model exp/.../model_best.pth \
  --track_img_root /data/VisDrone2019-COCO-4cls/test-dev/images \
  --track_ann_file /data/VisDrone2019-COCO-4cls/test-dev/annotations/instances_test-dev.json \
  --input-wh 864 480 --conf_thres 0.4 --use_s4 --use_appearance_motion

# 5 lớp benchmark (ped/car/truck/tricycle/bus) — remap 7→5, drop bicycle/motor
python scripts/track_5cls.py  ...

# đầy đủ lớp
python scripts/track_ECDet.py ...
```

Kết quả MOTA/IDF1 in ra console (motchallenge format) và lưu `summary_*.xlsx`.

---

## 7. Inference

### 7.1. Trên ảnh / thư mục ảnh (chỉ detection)

```bash
python scripts/inference.py \
  --input_path /path/to/images_or_image.jpg \
  --output_dir out/inference_results \
  --arch falcon_jde --load_model exp/.../model_best.pth \
  --input-wh 864 480 --conf_thres 0.4 \
  --tile --tile_grid 2 2     # tuỳ chọn: SAHI cho vật nhỏ
```

### 7.2. Trên video (tracking + làm mượt) — `inference_track.py` ★

Script này chạy **full tracking** (QAM + Kalman + GMC) rồi làm mượt 3 lớp ở hậu xử lý để video output đỡ nháy/giật nhất:

1. **Lọc track ngắn** (`--min_track_len`) — bỏ đốm nháy 1–vài frame.
2. **Nội suy** (`--max_interp_gap`) — lấp khoảng trống khi model miss object ngắn hạn.
3. **Làm mượt quỹ đạo** (`--smooth_window`) — moving-average đối xứng, giảm rung box, không gây trễ.

```bash
python scripts/inference_track.py \
  --input_video  in.mp4 \
  --output_video out_smoothed.mp4 \
  --arch falcon_jde --load_model exp/.../model_best.pth \
  --input-wh 864 480 --conf_thres 0.4 --use_s4 \
  --use_appearance_motion \         # ★ bật QAM → giữ ID ổn định, mượt hơn
  --min_track_len 5 --max_interp_gap 15 --smooth_window 5
```

**Pipeline 3 phase:** PHASE 1 detect+track toàn bộ frame → PHASE 2 làm mượt → PHASE 3 render video. (Đọc lại video 2 lần nên cần file video tồn tại trên đĩa.)

> Mẹo mượt nhất: bật `--use_appearance_motion`, `--smooth_window` để số **lẻ** (5 hoặc 7). Tăng window → mượt hơn nhưng box bám hơi trễ với vật di chuyển nhanh. Nếu model train kiểu letterbox, thêm `--letterbox` để box khớp chính xác.

### 7.3. Trên ONNX (không cần PyTorch)

```bash
python scripts/infer_onnx.py \
  --model falcon_jde.onnx --source path/to/video_or_images \
  --conf_thres 0.4 --nms_thres 0.45 \
  --num_classes 7 --reid_dim 128 --save_dir outputs/onnx_result
```

---

## 8. Export ONNX

```bash
# Từ checkpoint đã train (4-scale + S4)
python scripts/export_onnx.py \
  --use_s4 --load_model exp/.../model_best.pth \
  --onnx_path falcon_jde.onnx \
  --img_h 480 --img_w 864 --opset 17

# Kiểm tra graph với trọng số ngẫu nhiên
python scripts/export_onnx.py --dummy --use_s4 --onnx_path falcon_jde_dummy.onnx
```

> **Quan trọng:** truyền **đúng các flag kiến trúc** như khi train (`--use_s4`, `--train_single_det`, `--reid_head_type`, ...) để graph ONNX khớp checkpoint.

---

## 9. Sơ đồ class & các scheme remap

**Model luôn xuất 7 lớp** (đầu ra cố định). Khi đánh giá theo bộ GT khác, dùng remap trong `falconmot/tracker/class_remap.py`:

| 7-cls (train, 0-idx) | → 5-cls benchmark | → 4-cls competition |
|---|---|---|
| 0 pedestrian | 0 pedestrian | 0 person |
| 1 bicycle | — *(drop)* | 3 bicycle |
| 2 car | 1 car | 1 car |
| 3 truck | 2 truck | — *(drop)* |
| 4 tricycle | 3 tricycle | — *(drop)* |
| 5 bus | 4 bus | — *(drop)* |
| 6 motor | — *(drop)* | 2 motorcycle |

`track_id` toàn cục = `track_id_trong_lớp + cls_id_0idx * 1_000_000` (giữ id duy nhất trong một accumulator motmetrics; phía GT trong `io.py` áp cùng offset).

---

## 10. Các tham số quan trọng

Tất cả định nghĩa trong `falconmot/opts.py`. Một số nhóm hay dùng:

**Mô hình / kiến trúc**
| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--arch` | `falcon_jde` | Kiến trúc (hiện chỉ hỗ trợ `falcon_jde`). |
| `--use_s4` / `--no_s4_aux` | off / on | Bật nhánh stride-4 cho vật nhỏ + head phụ. |
| `--num_queries` | 300 | Số query của decoder. |
| `--num_dec_layers` | 4 | Số tầng decoder. |
| `--reid_dim` | 128 | Chiều embedding ReID. |
| `--reid_grad_scale` | 1.0 | Cường độ gradient ReID về trunk (1.0 = JDE coupling đầy đủ). |
| `--input-wh` | `864 480` | Kích thước mạng (W H). |
| `--eval_spatial_size` | — | (H W) cho denoising/eval; thường đảo của input-wh. |

**Huấn luyện**
| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--deim_pretrained` | — | Checkpoint khởi tạo (DEIM/DINOv3). |
| `--train_single_det` | off | Train detector-only (stage-1, tắt ReID). |
| `--lr` / `--backbone_lr_factor` | 5e-4 / 0.05 | LR & hệ số LR riêng cho backbone. |
| `--lr_scheduler` | `flat_cosine` | `cosine` / `flat_cosine` / ... |
| `--reid_warmup_epochs`, `--id_warmup_epochs` | 0 | Warmup phase-0 và ramp `id_weight` phase-1. |
| `--tri`, `--use_arcface`, `--rep` | off | Các loss ReID phụ (triplet / arcface / repulsion). |
| Augment | — | `--mosaic`, `--copy_paste`, `--gridmask`, `--homography`, `--small_obj_zoom`, `--temporal_mosaic`. |

**Tracking**
| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--conf_thres` | 0.4 | Ngưỡng tin cậy detection. |
| `--track_buffer` | 30 | Số frame giữ track đã mất (theo 30 FPS). |
| `--use_appearance_motion` | off | **Bật QAM** (dense appearance-motion) → ổn định ID, mượt hơn. |
| `--am_tau`, `--am_kappa`, `--am_beta` | 0.07 / 0.1 / 4.0 | Tham số QAM (nhiệt độ tương quan, scale motion, cổng entropy). |
| `--am_w_app`, `--am_w_iou` | 1.0 / 1.0 | Trọng số appearance vs IoU khi fuse. |
| `--match_thresh` | 0.7 | Ngưỡng linear assignment. |
| `--proximity_thresh`, `--motion_gate` | 0.95 / 0.9 | Cổng lọc cặp match xa/không hợp lý. |

**Riêng `inference_track.py`** (đã thêm): `--input_video`, `--output_video`, `--max_interp_gap`, `--min_track_len`, `--smooth_window`, `--letterbox`.

---

## 11. Ghi chú & FAQ

**Letterbox vs plain-resize.** `inference.py` và bản mặc định của `inference_track.py` dùng **plain-resize** (không letterbox) → postprocessor giải mã box bằng nhánh `scale * orig` (không gọi `set_net_hw`). Các script eval qua `LoadCocoSequencesForTracking` dùng **letterbox** → postprocessor gọi `set_net_hw`. **Toạ độ của dense appearance map (QAM) phải khớp đúng cách giải mã box** — `inference_track.py` đã tự tính transform phù hợp cho cả hai chế độ.

**QAM cần gì?** Tracker tự bật khi model trả về `reid_dense`. Điều này chỉ xảy ra khi `model.return_reid_dense = True` **và** model không ở chế độ train. Các script tracking set cờ này khi có `--use_appearance_motion`. Stride của dense map = 4 nếu `--use_s4`, ngược lại 8.

**Tại sao box bị nháy/đổi ID?** Thường do (a) chưa bật `--use_appearance_motion`, (b) `conf_thres` quá cao làm rớt box ngắt quãng, hoặc (c) thiếu hậu xử lý. Với video, dùng `inference_track.py` + QAM + 3 lớp smoothing (mục 7.2).

**`model_best.pth` chọn theo gì?** Stage-1 theo mAP; stage-2 (nếu bật `--track_val`) theo điểm kết hợp `w_idf1 * IDF1 + w_mota * MOTA`.

**Không import được `falconmot`?** Chạy script từ thư mục gốc repo (dòng `import _paths` lo việc thêm path); đừng `cd scripts/` rồi chạy bằng đường dẫn tương đối lạ.

**GPU bắt buộc không?** Train/inference video gần như bắt buộc GPU. Suy luận ảnh lẻ vẫn chạy được trên CPU nhưng chậm.

---

*FalconMOT / FalconJDE — DINOv3STAs + HybridEncoder + DEIMTransformer + ReID head, kèm multi-class tracker (Kalman + GMC + Query Appearance-Motion).*