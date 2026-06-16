#!/usr/bin/env bash
# Stage-1: VisDrone-DET detection-only training (no ReID head / loss).
# Edit paths in falconmot/cfg/visdrone_coco_det.json before running.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train.py \
  --task mot --arch falcon_jde --dataset coco \
  --train_single_det \
  --exp_id falcon_stage1_det \
  --data_cfg falconmot/cfg/visdrone_coco_det.json \
  --val_cfg  falconmot/cfg/visdrone_coco_det.json \
  --deim_pretrained pretrained_model/deimv2_dinov3_s_coco.pth \
  --input-wh 960 544 \
  --eval_spatial_size 544 960 \
  --num_queries 300 \
  --batch_size 16 --grad_accum 4 --num_workers 16 --gpus 0 \
  --lr 5e-4 --num_epochs 60 \
  --lr_scheduler cosine --warmup_epochs 2 --no_aug_epochs 5 --lr_min_factor 0.01 \
  --val_intervals 1 \
  --use_s4 \
  --mosaic --mosaic_prob 0.3 \
  "$@"
