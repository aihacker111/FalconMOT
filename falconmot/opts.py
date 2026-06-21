from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ── Basic ──────────────────────────────────────────────────────────
        self.parser.add_argument('--task',     default='mot',  help='mot')
        self.parser.add_argument('--train_single_det', action='store_true', default=False,
                                 help='Stage-1 detection-only training (VisDrone-DET): no ReID head, '
                                      'no ReID loss, no temporal mosaic. Save checkpoint, then run '
                                      'stage-2 tracking fine-tune with --load_model (no flag).')
        self.parser.add_argument('--dataset',  default='coco',
                                 help='coco (COCO JSON format) | jde (legacy JDE index files)')
        self.parser.add_argument('--exp_id',   default='default')
        self.parser.add_argument('--test',     action='store_true')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume',   action='store_true',
                                 help='resume training; reloads optimizer')

        # ── System ─────────────────────────────────────────────────────────
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, comma-separated for multiple GPUs')
        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument('--not_cuda_benchmark', action='store_true')
        self.parser.add_argument('--seed', type=int, default=317)

        # ── Logging / checkpoint ───────────────────────────────────────────
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='print loss every N iters (0 = progress bar)')
        self.parser.add_argument('--hide_data_time', action='store_true')
        self.parser.add_argument('--quiet', action='store_true', default=False,
                                 help='suppress per-frame tqdm output')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save checkpoint every epoch')
        self.parser.add_argument('--val_intervals', type=int, default=1,
                                 help='run COCO mAP eval every N epochs')

        # ── Model ──────────────────────────────────────────────────────────
        self.parser.add_argument('--arch', default='falcon_jde',
                                 help='falcon_jde (DINOv3STAs + HybridEncoder + DEIMTransformer)')

        # DINOv3STAs backbone
        self.parser.add_argument('--dinov3_name', default='vit_tiny',
                                 help='vit_tiny | dinov3_small | dinov3_base')
        self.parser.add_argument('--dinov3_weights', default='',
                                 help='path to backbone weights (.pth/.pt)')
        self.parser.add_argument('--dinov3_embed_dim', type=int, default=192)
        self.parser.add_argument('--dinov3_num_heads', type=int, default=3)
        self.parser.add_argument('--dinov3_interaction_indexes', type=int, nargs='+',
                                 default=[5, 8, 11],
                                 help='ViT layer indices to extract (S8/S16/S32)')
        self.parser.add_argument('--use_sta', action='store_true', default=True,
                                 help='use Spatial Prior Module')
        self.parser.add_argument('--no_sta', dest='use_sta', action='store_false')
        self.parser.add_argument('--conv_inplane', type=int, default=16,
                                 help='SpatialPriorModule base channels')
        self.parser.add_argument('--hidden_dim', type=int, default=192,
                                 help='shared hidden dim for encoder and decoder')

        # Encoder
        self.parser.add_argument('--enc_dim_ff',    type=float, default=512)
        self.parser.add_argument('--enc_expansion', type=float, default=0.34)
        self.parser.add_argument('--enc_depth_mult', type=float, default=0.67)

        # Decoder
        self.parser.add_argument('--num_queries',   type=int, default=300)
        self.parser.add_argument('--num_dec_layers', type=int, default=4)
        self.parser.add_argument('--dec_dim_ff',    type=int, default=512)
        self.parser.add_argument('--num_denoising', type=int, default=100)
        self.parser.add_argument('--reg_max',       type=int, default=32)

        # S4 branch (stride-4 for small/distant object detection)
        self.parser.add_argument('--use_s4', action='store_true', default=False,
                                 help='Add stride-4 branch: decoder uses [S4,S8,S16,S32].')
        self.parser.add_argument('--use_s4_aux', action='store_true', default=True,
                                 help='enable S4 auxiliary heatmap loss head during training')
        self.parser.add_argument('--no_s4_aux', dest='use_s4_aux', action='store_false',
                                 help='disable S4 auxiliary loss/head; no aux gradient')
        # ── Gộp train + val thành một tập train ──────────────────────────
        self.parser.add_argument('--merge_val_into_train', action='store_true', default=False,
                                 help='Gộp cả train_ann/img và val_ann/img (từ data_cfg) vào '
                                      'tập train. Offset image_id/seq_id/track_id để không '
                                      'va chạm. Lưu ý: không còn val tách biệt để đánh giá.')
        # ── ReID head ────────────────────────────────────────────────────
        # NOTE: reid_head_type is DEPRECATED. The model now uses a single
        # appearance ReID head (deformable-sample of the shared feature map).
        # The flag is kept only so older scripts don't break; it is ignored.
        self.parser.add_argument('--reid_head_type', default='transformer',
                                 choices=['transformer', 'context_aware', 'mlp'],
                                 help='DEPRECATED / ignored — a single ReID head is always used.')
        self.parser.add_argument('--reid_num_points', type=int, default=8,
                                 help='số điểm deformable-sample/box cho ReID head')
        self.parser.add_argument('--reid_grad_scale', type=float, default=1.0,
                                 help='độ mạnh gradient ReID chảy vào trunk qua feature map '
                                      '(1.0 = full JDE coupling; hạ về ~0.1 nếu detection bị nhiễu).')

        # Pretrained / spatial size
        self.parser.add_argument('--deim_pretrained', default='',
                                 help='pretrained DEIM checkpoint (weights only, no optimizer)')
        self.parser.add_argument('--eval_spatial_size', type=int, nargs=2,
                                 default=[480, 864], help='[H, W] for anchor pre-generation')
        # ── Stage-2 two-phase fine-tune ─────────────────────────────────────
        self.parser.add_argument('--reid_warmup_epochs', type=int, default=0,
                                help='Phase 0: số epoch chỉ train reid_head + classifiers '
                                    '(detector freeze). 0 = bỏ Phase 0, vào thẳng joint.')
        self.parser.add_argument('--reid_warmup_lr', type=float, default=-1.0,
                                help='LR cho Phase 0 (-1 = dùng --lr). Có thể đặt cao hơn '
                                    'vì chỉ train vài head nhỏ, vd 1e-3.')
        self.parser.add_argument('--keep_backbone_frozen', action='store_true', default=False,
                                help='Phase 1: giữ backbone freeze (khuyến nghị).')
        self.parser.add_argument('--unfreeze_backbone', dest='keep_backbone_frozen',
                                action='store_false')
        self.parser.add_argument('--id_warmup_epochs', type=int, default=0,
                                help='Phase 1: ramp id_weight 0->id_weight qua N epoch đầu.')
        # ── Input resolution ───────────────────────────────────────────────
        self.parser.add_argument('--input-wh', type=int, nargs=2, default=[864, 480],
                                 help='network input W H')
        self.parser.add_argument('--input_res', type=int, default=-1)
        self.parser.add_argument('--input_h',   type=int, default=-1)
        self.parser.add_argument('--input_w',   type=int, default=-1)

        # ── Training ───────────────────────────────────────────────────────
        self.parser.add_argument('--lr', type=float, default=5e-4)
        self.parser.add_argument('--weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--backbone_lr_factor', type=float, default=0.05,
                                 help='backbone LR = lr × this factor (pretrained DINOv3 fine-tune; '
                                      '0.05 conservative, 0.1 = DETR/D-FINE default).')
        self.parser.add_argument('--reid_lr_factor', type=float, default=1.0,
                                 help='ReID branch LR = lr × this factor. reid_head + ID classifiers '
                                      'are trained from scratch while the rest is pretrained, so a '
                                      'higher factor (e.g. 3–5) speeds ReID convergence without '
                                      'destabilising the detector. s_det/s_id stay at base LR.')
        self.parser.add_argument('--lr_step', type=str, default='10,20',
                                 help='epochs to drop LR (step scheduler)')
        self.parser.add_argument('--warmup_iters', type=int, default=2000,
                                 help='quadratic LR warmup steps')
        self.parser.add_argument('--no_aug_epochs', type=int, default=2,
                                 help='final constant-LR epochs (no-aug phase)')
        self.parser.add_argument('--lr_gamma', type=float, default=0.5,
                                 help='min_lr = lr × lr_gamma (flat_cosine)')
        self.parser.add_argument('--lr_scheduler', type=str, default='flat_cosine',
                                 choices=['cosine', 'step', 'flat_cosine'])
        self.parser.add_argument('--warmup_epochs', type=float, default=0.0,
                                 help='linear warmup epochs (cosine/step scheduler)')
        self.parser.add_argument('--lr_drop', type=int, default=-1,
                                 help='step-drop epoch (-1 = num_epochs)')
        self.parser.add_argument('--lr_min_factor', type=float, default=0.0,
                                 help='cosine floor as fraction of base LR')
        self.parser.add_argument('--clip_max_norm', type=float, default=0.1,
                                 help='gradient clipping max norm (0 = disabled)')
        self.parser.add_argument('--use_amp', action='store_true', default=True)
        self.parser.add_argument('--no_amp', dest='use_amp', action='store_false')
        self.parser.add_argument('--num_epochs',       type=int, default=30)
        self.parser.add_argument('--batch_size',       type=int, default=8)
        self.parser.add_argument('--master_batch_size', type=int, default=-1)
        self.parser.add_argument('--grad_accum', type=int, default=1,
                                 help='gradient accumulation steps')
        self.parser.add_argument('--num_iters', type=int, default=-1)
        self.parser.add_argument('--trainval', action='store_true')

        # ── Augmentation ───────────────────────────────────────────────────
        self.parser.add_argument('--stop_epoch', type=int, default=-1,
                                 help='epoch to disable aug (-1 = always on)')
        self.parser.add_argument('--copy_paste', action='store_true', default=False)
        self.parser.add_argument('--copy_paste_prob', type=float, default=0.5)
        self.parser.add_argument('--copy_paste_max_area', type=float, default=0.01)
        self.parser.add_argument('--copy_paste_n', type=int, default=5)
        self.parser.add_argument('--mosaic', action='store_true', default=False)
        self.parser.add_argument('--mosaic_prob', type=float, default=0.5)
        self.parser.add_argument('--mosaic_scale_bias_prob', type=float, default=0.5)
        self.parser.add_argument('--mosaic_scale_min', type=float, default=0.3)
        self.parser.add_argument('--mosaic_scale_max', type=float, default=0.6)

        # ── Dataset config ─────────────────────────────────────────────────
        self.parser.add_argument('--data_cfg', type=str,
                                 default='falconmot/cfg/visdrone_coco.json')
        self.parser.add_argument('--val_cfg', type=str, default='',
                                 help='JSON config for val split; activates COCO mAP eval')
        self.parser.add_argument('--debug_val_batches', type=int, default=0,
                                 help='limit eval to N batches (0 = full)')
        self.parser.add_argument('--data_dir', type=str, default='',
                                 help='root for tracking eval data (track.py)')

        # ── Loss ───────────────────────────────────────────────────────────
        self.parser.add_argument('--id_weight', type=float, default=0.0,
                                 help='ReID loss weight (0 = detection only)')
        self.parser.add_argument('--tri', action='store_true',
                                 help='add triplet loss to ReID')
        self.parser.add_argument('--rep', action='store_true', default=False,
                                 help='enable RepulsionLoss')
        self.parser.add_argument('--rep_weight', type=float, default=0.5)
        self.parser.add_argument('--use_arcface', action='store_true', default=False,
                                 help='use ArcFace for ReID classification. Default OFF = plain '
                                      'CE + emb_scale (the stable FairMOT/AMOT recipe).')
        self.parser.add_argument('--s_det_init', type=float, default=2.5,
                                 help='init for uncertainty weight s_det ≈ log(initial loss_det). '
                                      'Read first-iter loss_det from the log and set log() of it.')
        self.parser.add_argument('--s_id_init', type=float, default=1.85,
                                 help='init for uncertainty weight s_id ≈ log(initial loss_reid).')

        # ── Sequence-aware augmentation ────────────────────────────────────
        self.parser.add_argument('--temporal_mosaic', action='store_true', default=False,
                                 help='4 frames from same sequence → 2×2 mosaic; '
                                      'increases positive-sample density per forward pass')
        self.parser.add_argument('--temporal_mosaic_prob', type=float, default=0.5,
                                 help='probability of using temporal mosaic vs single frame')
        self.parser.add_argument('--small_obj_zoom', action='store_true', default=False,
                                 help='zoom crop anchored on tiny objects before affine; '
                                      'boosts gradient signal for objects <0.2%% of image area')
        self.parser.add_argument('--small_obj_zoom_prob', type=float, default=0.5,
                                 help='probability of applying small-object zoom per sample')
        self.parser.add_argument('--gridmask', action='store_true', default=False,
                                 help='GridMask: erase regular grid pattern to simulate occlusion')
        self.parser.add_argument('--gridmask_prob', type=float, default=0.3,
                                 help='probability of applying gridmask per sample')
        # Random perspective (homography) warp — synthetic viewpoint augment
        self.parser.add_argument('--homography', action='store_true', default=True,
                                 help='random perspective warp (alt to affine) for viewpoint diversity')
        self.parser.add_argument('--no_homography', dest='homography', action='store_false')
        self.parser.add_argument('--homography_prob', type=float, default=0.3,
                                 help='probability of using homography instead of affine')
        self.parser.add_argument('--homography_strength', type=float, default=0.12,
                                 help='corner-jitter fraction of image size (0.08-0.15 sane)')
        # ── ReID ───────────────────────────────────────────────────────────
        self.parser.add_argument('--reid_dim', type=int, default=128)
        self.parser.add_argument('--reid_cls_ids', default='0,1,2,3,4,5,6,7,8,9')

        # ── Tracking nguồn COCO (đồng bộ với training/val) ──
        self.parser.add_argument('--track_from_coco', action='store_true', default=False,
                                help='lấy ảnh + danh sách seq/frame từ COCO JSON thay vì '
                                    'test_dev/sequences thô (đồng bộ tiền xử lý với training)')
        self.parser.add_argument('--track_ann_file', type=str, default='',
                                help='COCO JSON annotation cho tracking; rỗng → dùng val_ann trong data_cfg')
        self.parser.add_argument('--track_img_root', type=str, default='',
                                help='thư mục ảnh tương ứng track_ann_file; rỗng → dùng val_img trong data_cfg')
        self.parser.add_argument('--track_gt_root', type=str, default='',
                                help='thư mục annotation VisDrone thô của split tracking '
                                    '(vd .../VisDrone2019-MOT-val/annotations) để Evaluator dựng GT')
        # ── Tracking inference ─────────────────────────────────────────────
        self.parser.add_argument('--eval_mode', type=str, default='10class',
                                 choices=['10class', '5class', '4class',
                                          '5class_merge_benchmark',
                                          '5class_merge_competition'],
                                 help='Evaluation class subset: '
                                      '10class=all VisDrone (default, with ID offset); '
                                      '5class=pedestrian/car/van/truck/bus, SKIP (AMOT protocol); '
                                      '4class=person/car/motorcycle/bicycle, SKIP (competition); '
                                      '5class_merge_benchmark=pedestrian/car/truck/tricycle/bus, '
                                      'MERGE (drops bicycle+motor, no equivalent group); '
                                      '5class_merge_competition=person/car/truck/motorcycle/bicycle, '
                                      'MERGE (truck=van+truck; drops tricycle+awning-tricycle+bus, '
                                      'no equivalent group)')
        self.parser.add_argument('--K', type=int, default=300,
                                 help='max detections per image at inference')
        self.parser.add_argument('--conf_thres', type=float, default=0.4,
                                 help='detection confidence threshold')
        self.parser.add_argument('--track_buffer', type=int, default=30)
        # ── Query Appearance-Motion (QAM) association ──
        self.parser.add_argument('--use_appearance_motion', action='store_true',
                                 help='enable appearance-as-motion association: predict each '
                                      'track position by cross-frame correlation on the dense '
                                      'reid map (soft-argmax), entropy-gated, fused by '
                                      'log-likelihood. Falls back to legacy fusion if off.')
        self.parser.add_argument('--legacy_fuse', action='store_true',
                                 help='force the old multiplicative fuse_score_three (A/B).')
        self.parser.add_argument('--am_tau', type=float, default=0.07,
                                 help='softmax temperature for the correlation response.')
        self.parser.add_argument('--am_kappa', type=float, default=0.1,
                                 help='motion sigma = kappa * sqrt(w*h); smaller = stricter.')
        self.parser.add_argument('--am_beta', type=float, default=4.0,
                                 help='entropy->confidence sharpness: w = exp(-beta*entropy).')
        self.parser.add_argument('--am_w_app', type=float, default=1.0,
                                 help='appearance (cosine) cue weight in log-lik fusion.')
        self.parser.add_argument('--am_w_iou', type=float, default=1.0,
                                 help='IoU cue weight in log-lik fusion.')
        self.parser.add_argument('--match_thresh', type=float, default=0.7,
                                 help='cost ceiling for the first (QAM) association.')
        self.parser.add_argument('--proximity_thresh', type=float, default=0.95,
                                 help='IoU-distance spatial gate: a pair with iou_dist above '
                                      'this needs motion to vouch for it.')
        self.parser.add_argument('--motion_gate', type=float, default=0.9,
                                 help='motion-distance spatial gate: motion can vouch for a '
                                      'low-IoU pair only if its motion_dist is below this.')
        # ── Tracking-metric validation (chọn model_best theo IDF1/MOTA) ──
        self.parser.add_argument('--emb_weight', type=float, default=1.0,
                                 help='ReID weight in association fusion. '
                                      '1.0=original, 0.0=IoU-only, in-between=blend.')
        self.parser.add_argument('--emb_gate', type=float, default=0.0,
                                 help='Chỉ áp embedding khi id_sim >= ngưỡng này; '
                                      'dưới ngưỡng dùng IoU thuần. 0.0=tắt gating.')
        self.parser.add_argument('--track_val', action='store_true',
                                 help='bật validation tracking (IDF1/MOTA) để chọn model_best '
                                      'thay cho COCO mAP. Cần --val_cfg có val_ann + val_img (COCO).')
        self.parser.add_argument('--track_val_intervals', type=int, default=1,
                                 help='chạy tracking-eval mỗi N epoch (0 = dùng --val_intervals).')
        self.parser.add_argument('--track_val_fp16', type=int, default=1,
                                 help='forward detector bằng fp16 khi tracking-eval (1=bật, nhanh hơn).')
        self.parser.add_argument('--track_val_in_phase0', action='store_true',
                                 help='cũng chạy tracking-eval trong Phase 0 (mặc định bỏ qua vì '
                                      'detector đóng băng; bật nếu muốn đo riêng cải thiện reid_head).')
        self.parser.add_argument('--track_val_w_idf1', type=float, default=0.6,
                                 help='trọng số IDF1 trong track_score = w_idf1*IDF1 + w_mota*MOTA.')
        self.parser.add_argument('--track_val_w_mota', type=float, default=0.4,
                                 help='trọng số MOTA trong track_score.')
        self.parser.add_argument('--frame_rate', type=int, default=30,
                                 help='frame-rate giả định cho tracker (buffer = frame_rate/30 * track_buffer).')
        # ── Tracker (FusionTrack-inspired, inference-only) ───────────────
        self.parser.add_argument('--reid_decay_alpha', type=float, default=0.02,
                                 help='time-decay rate for gallery ReID memory (W=e^{-alpha*dt}); 0 = no decay')
        self.parser.add_argument('--nfm_topk', type=int, default=2,
                                 help='mutual top-k for Neighbor Filtering gate on appearance matches')
        self.parser.add_argument('--use_nfm', action='store_true', default=True,
                                 help='enable mutual top-k NFM gating (reduces ID switches)')
        self.parser.add_argument('--no_nfm', dest='use_nfm', action='store_false')
        # ── GMC determinism + w_iou ramp (benchmark ổn định) ─────────────
        self.parser.add_argument('--gmc_seed', type=int, default=0,
                                 help='seed RNG của OpenCV cho RANSAC trong GMC -> benchmark không đổi')
        self.parser.add_argument('--gmc_deterministic', action='store_true', default=True,
                                 help='re-seed cv2 RNG mỗi frame để RANSAC tất định')
        self.parser.add_argument('--no_gmc_deterministic', dest='gmc_deterministic',
                                 action='store_false')
        self.parser.add_argument('--w_iou_hi', type=float, default=0.5,
                                 help='trọng số IoU khi GMC tin cậy (camera motion nhỏ)')
        self.parser.add_argument('--w_iou_lo', type=float, default=0.3,
                                 help='trọng số IoU khi GMC kém tin (camera motion lớn)')
        self.parser.add_argument('--gmc_band_lo', type=float, default=20.0,
                                 help='||translation|| bắt đầu giảm w_iou (ramp dưới)')
        self.parser.add_argument('--gmc_band_hi', type=float, default=40.0,
                                 help='||translation|| đạt w_iou_lo (ramp trên)')
        self.parser.add_argument('--min-box-area', type=float, default=100,
                                 help='filter boxes smaller than this area (px²)')
        self.parser.add_argument('--test_visdrone', default=True)
        self.parser.add_argument('--test_uavdt',    default=False)

        # ── Distributed ────────────────────────────────────────────────────
        self.parser.add_argument('--local-rank', type=int, default=0)

    def parse(self, args=''):
        opt = self.parser.parse_args() if args == '' else self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus     = [int(g) for g in opt.gpus.split(',')]
        opt.lr_step  = [int(s) for s in opt.lr_step.split(',')]

        if opt.lr_drop < 0:
            opt.lr_drop = opt.num_epochs

        if opt.trainval:
            opt.val_intervals = 100_000_000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest = opt.batch_size - opt.master_batch_size
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            chunk = rest // (len(opt.gpus) - 1)
            if i < rest % (len(opt.gpus) - 1):
                chunk += 1
            opt.chunk_sizes.append(chunk)
        print('chunk_sizes:', opt.chunk_sizes)

        opt.root_dir  = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.exp_dir   = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir  = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('Output will be saved to', opt.save_dir)

        # Stage-1 detection-only: VisDrone-DET has no track IDs.
        if opt.train_single_det:
            opt.use_reid           = False
            opt.id_weight          = 0.0
            opt.temporal_mosaic    = False
            opt.reid_warmup_epochs = 0
            opt.id_warmup_epochs   = 0
            s4_tag = '+s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''
            print('[train_single_det] stage-1 detection-only: '
                  f'ReID head OFF | losses: cls+bbox+giou{s4_tag} | '
                  f'use_s4={getattr(opt, "use_s4", False)} | '
                  f'mosaic={getattr(opt, "mosaic", False)} | temporal_mosaic=OFF')
        else:
            opt.use_reid = True

        if opt.resume and opt.load_model == '':
            base = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
            opt.load_model = os.path.join(base, 'model_last.pth')

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_input_wh
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes   = dataset.num_classes
        print('num_classes:', opt.num_classes)

        for reid_id in opt.reid_cls_ids.split(','):
            if int(reid_id) > opt.num_classes - 1:
                if getattr(opt, 'train_single_det', False):
                    break
                print('[Err]: reid_cls_ids conflicts with num_classes')
                return

        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h   = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w   = opt.input_w if opt.input_w > 0 else input_w
        opt.input_res = max(opt.input_h, opt.input_w)

        if opt.task == 'mot':
            # Always expose nID_dict (criterion reads it even when ReID is off).
            opt.nID_dict = dataset.nID_dict
        else:
            assert 0, 'task not defined!'

        return opt

    def init(self, args=''):
        opt = self.parse(args)

        default_dataset_info = {
            'mot': {
                'default_input_wh': [opt.input_wh[1], opt.input_wh[0]],
                'num_classes':       len(opt.reid_cls_ids.split(',')),
                'mean': [0.485, 0.456, 0.406],
                'std':  [0.229, 0.224, 0.225],
                'dataset':  'coco',
                'nID_dict': {},
            },
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    setattr(self, k, v)

        h_w = default_dataset_info[opt.task]['default_input_wh']
        opt.img_size = (h_w[1], h_w[0])
        print('Net input: {:d}×{:d}'.format(h_w[1], h_w[0]))

        dataset     = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt









# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import argparse
# import os


# class opts(object):
#     def __init__(self):
#         self.parser = argparse.ArgumentParser()

#         # ── Basic ──────────────────────────────────────────────────────────
#         self.parser.add_argument('--task',     default='mot',  help='mot')
#         self.parser.add_argument('--train_single_det', action='store_true', default=False,
#                                  help='Stage-1 detection-only training (VisDrone-DET): no ReID head, '
#                                       'no ReID loss, no temporal mosaic. Save checkpoint, then run '
#                                       'stage-2 tracking fine-tune with --load_model (no flag).')
#         self.parser.add_argument('--dataset',  default='coco',
#                                  help='coco (COCO JSON format) | jde (legacy JDE index files)')
#         self.parser.add_argument('--exp_id',   default='default')
#         self.parser.add_argument('--test',     action='store_true')
#         self.parser.add_argument('--load_model', default='',
#                                  help='path to pretrained model')
#         self.parser.add_argument('--resume',   action='store_true',
#                                  help='resume training; reloads optimizer')

#         # ── System ─────────────────────────────────────────────────────────
#         self.parser.add_argument('--gpus', default='0',
#                                  help='-1 for CPU, comma-separated for multiple GPUs')
#         self.parser.add_argument('--num_workers', type=int, default=8)
#         self.parser.add_argument('--not_cuda_benchmark', action='store_true')
#         self.parser.add_argument('--seed', type=int, default=317)

#         # ── Logging / checkpoint ───────────────────────────────────────────
#         self.parser.add_argument('--print_iter', type=int, default=0,
#                                  help='print loss every N iters (0 = progress bar)')
#         self.parser.add_argument('--hide_data_time', action='store_true')
#         self.parser.add_argument('--quiet', action='store_true', default=False,
#                                  help='suppress per-frame tqdm output')
#         self.parser.add_argument('--save_all', action='store_true',
#                                  help='save checkpoint every epoch')
#         self.parser.add_argument('--val_intervals', type=int, default=1,
#                                  help='run COCO mAP eval every N epochs')

#         # ── Model ──────────────────────────────────────────────────────────
#         self.parser.add_argument('--arch', default='falcon_jde',
#                                  help='falcon_jde (DINOv3STAs + HybridEncoder + DEIMTransformer)')

#         # DINOv3STAs backbone
#         self.parser.add_argument('--dinov3_name', default='vit_tiny',
#                                  help='vit_tiny | dinov3_small | dinov3_base')
#         self.parser.add_argument('--dinov3_weights', default='',
#                                  help='path to backbone weights (.pth/.pt)')
#         self.parser.add_argument('--dinov3_embed_dim', type=int, default=192)
#         self.parser.add_argument('--dinov3_num_heads', type=int, default=3)
#         self.parser.add_argument('--dinov3_interaction_indexes', type=int, nargs='+',
#                                  default=[5, 8, 11],
#                                  help='ViT layer indices to extract (S8/S16/S32)')
#         self.parser.add_argument('--use_sta', action='store_true', default=True,
#                                  help='use Spatial Prior Module')
#         self.parser.add_argument('--no_sta', dest='use_sta', action='store_false')
#         self.parser.add_argument('--conv_inplane', type=int, default=16,
#                                  help='SpatialPriorModule base channels')
#         self.parser.add_argument('--hidden_dim', type=int, default=192,
#                                  help='shared hidden dim for encoder and decoder')

#         # Encoder
#         self.parser.add_argument('--enc_dim_ff',    type=float, default=512)
#         self.parser.add_argument('--enc_expansion', type=float, default=0.34)
#         self.parser.add_argument('--enc_depth_mult', type=float, default=0.67)

#         # Decoder
#         self.parser.add_argument('--num_queries',   type=int, default=300)
#         self.parser.add_argument('--num_dec_layers', type=int, default=4)
#         self.parser.add_argument('--dec_dim_ff',    type=int, default=512)
#         self.parser.add_argument('--num_denoising', type=int, default=100)
#         self.parser.add_argument('--reg_max',       type=int, default=32)

#         # S4 branch (stride-4 for small/distant object detection)
#         self.parser.add_argument('--use_s4', action='store_true', default=False,
#                                  help='Add stride-4 branch: decoder uses [S4,S8,S16,S32].')
#         self.parser.add_argument('--use_s4_aux', action='store_true', default=True,
#                                  help='enable S4 auxiliary heatmap loss head during training')
#         self.parser.add_argument('--no_s4_aux', dest='use_s4_aux', action='store_false',
#                                  help='disable S4 auxiliary loss/head; no aux gradient')
#         # ── Gộp train + val thành một tập train ──────────────────────────
#         self.parser.add_argument('--merge_val_into_train', action='store_true', default=False,
#                                  help='Gộp cả train_ann/img và val_ann/img (từ data_cfg) vào '
#                                       'tập train. Offset image_id/seq_id/track_id để không '
#                                       'va chạm. Lưu ý: không còn val tách biệt để đánh giá.')
#         # ── ReID head ────────────────────────────────────────────────────
#         # NOTE: reid_head_type is DEPRECATED. The model now uses a single
#         # appearance ReID head (deformable-sample of the shared feature map).
#         # The flag is kept only so older scripts don't break; it is ignored.
#         self.parser.add_argument('--reid_head_type', default='transformer',
#                                  choices=['transformer', 'context_aware', 'mlp'],
#                                  help='DEPRECATED / ignored — a single ReID head is always used.')
#         self.parser.add_argument('--reid_num_points', type=int, default=8,
#                                  help='số điểm deformable-sample/box cho ReID head')
#         self.parser.add_argument('--reid_grad_scale', type=float, default=1.0,
#                                  help='độ mạnh gradient ReID chảy vào trunk qua feature map '
#                                       '(1.0 = full JDE coupling; hạ về ~0.1 nếu detection bị nhiễu).')

#         # Pretrained / spatial size
#         self.parser.add_argument('--deim_pretrained', default='',
#                                  help='pretrained DEIM checkpoint (weights only, no optimizer)')
#         self.parser.add_argument('--eval_spatial_size', type=int, nargs=2,
#                                  default=[480, 864], help='[H, W] for anchor pre-generation')
#         # ── Stage-2 two-phase fine-tune ─────────────────────────────────────
#         self.parser.add_argument('--reid_warmup_epochs', type=int, default=0,
#                                 help='Phase 0: số epoch chỉ train reid_head + classifiers '
#                                     '(detector freeze). 0 = bỏ Phase 0, vào thẳng joint.')
#         self.parser.add_argument('--reid_warmup_lr', type=float, default=-1.0,
#                                 help='LR cho Phase 0 (-1 = dùng --lr). Có thể đặt cao hơn '
#                                     'vì chỉ train vài head nhỏ, vd 1e-3.')
#         self.parser.add_argument('--keep_backbone_frozen', action='store_true', default=False,
#                                 help='Phase 1: giữ backbone freeze (khuyến nghị).')
#         self.parser.add_argument('--unfreeze_backbone', dest='keep_backbone_frozen',
#                                 action='store_false')
#         self.parser.add_argument('--id_warmup_epochs', type=int, default=0,
#                                 help='Phase 1: ramp id_weight 0->id_weight qua N epoch đầu.')
#         # ── Input resolution ───────────────────────────────────────────────
#         self.parser.add_argument('--input-wh', type=int, nargs=2, default=[864, 480],
#                                  help='network input W H')
#         self.parser.add_argument('--input_res', type=int, default=-1)
#         self.parser.add_argument('--input_h',   type=int, default=-1)
#         self.parser.add_argument('--input_w',   type=int, default=-1)

#         # ── Training ───────────────────────────────────────────────────────
#         self.parser.add_argument('--lr', type=float, default=5e-4)
#         self.parser.add_argument('--weight_decay', type=float, default=1e-4)
#         self.parser.add_argument('--backbone_lr_factor', type=float, default=0.05,
#                                  help='backbone LR = lr × this factor (pretrained DINOv3 fine-tune; '
#                                       '0.05 conservative, 0.1 = DETR/D-FINE default).')
#         self.parser.add_argument('--reid_lr_factor', type=float, default=1.0,
#                                  help='ReID branch LR = lr × this factor. reid_head + ID classifiers '
#                                       'are trained from scratch while the rest is pretrained, so a '
#                                       'higher factor (e.g. 3–5) speeds ReID convergence without '
#                                       'destabilising the detector. s_det/s_id stay at base LR.')
#         self.parser.add_argument('--lr_step', type=str, default='10,20',
#                                  help='epochs to drop LR (step scheduler)')
#         self.parser.add_argument('--warmup_iters', type=int, default=2000,
#                                  help='quadratic LR warmup steps')
#         self.parser.add_argument('--no_aug_epochs', type=int, default=2,
#                                  help='final constant-LR epochs (no-aug phase)')
#         self.parser.add_argument('--lr_gamma', type=float, default=0.5,
#                                  help='min_lr = lr × lr_gamma (flat_cosine)')
#         self.parser.add_argument('--lr_scheduler', type=str, default='flat_cosine',
#                                  choices=['cosine', 'step', 'flat_cosine'])
#         self.parser.add_argument('--warmup_epochs', type=float, default=0.0,
#                                  help='linear warmup epochs (cosine/step scheduler)')
#         self.parser.add_argument('--lr_drop', type=int, default=-1,
#                                  help='step-drop epoch (-1 = num_epochs)')
#         self.parser.add_argument('--lr_min_factor', type=float, default=0.0,
#                                  help='cosine floor as fraction of base LR')
#         self.parser.add_argument('--clip_max_norm', type=float, default=0.1,
#                                  help='gradient clipping max norm (0 = disabled)')
#         self.parser.add_argument('--use_amp', action='store_true', default=True)
#         self.parser.add_argument('--no_amp', dest='use_amp', action='store_false')
#         self.parser.add_argument('--num_epochs',       type=int, default=30)
#         self.parser.add_argument('--batch_size',       type=int, default=8)
#         self.parser.add_argument('--master_batch_size', type=int, default=-1)
#         self.parser.add_argument('--grad_accum', type=int, default=1,
#                                  help='gradient accumulation steps')
#         self.parser.add_argument('--num_iters', type=int, default=-1)
#         self.parser.add_argument('--trainval', action='store_true')

#         # ── Augmentation ───────────────────────────────────────────────────
#         self.parser.add_argument('--stop_epoch', type=int, default=-1,
#                                  help='epoch to disable aug (-1 = always on)')
#         self.parser.add_argument('--copy_paste', action='store_true', default=False)
#         self.parser.add_argument('--copy_paste_prob', type=float, default=0.5)
#         self.parser.add_argument('--copy_paste_max_area', type=float, default=0.01)
#         self.parser.add_argument('--copy_paste_n', type=int, default=5)
#         self.parser.add_argument('--mosaic', action='store_true', default=False)
#         self.parser.add_argument('--mosaic_prob', type=float, default=0.5)
#         self.parser.add_argument('--mosaic_scale_bias_prob', type=float, default=0.5)
#         self.parser.add_argument('--mosaic_scale_min', type=float, default=0.3)
#         self.parser.add_argument('--mosaic_scale_max', type=float, default=0.6)

#         # ── Dataset config ─────────────────────────────────────────────────
#         self.parser.add_argument('--data_cfg', type=str,
#                                  default='falconmot/cfg/visdrone_coco.json')
#         self.parser.add_argument('--val_cfg', type=str, default='',
#                                  help='JSON config for val split; activates COCO mAP eval')
#         self.parser.add_argument('--debug_val_batches', type=int, default=0,
#                                  help='limit eval to N batches (0 = full)')
#         self.parser.add_argument('--data_dir', type=str, default='',
#                                  help='root for tracking eval data (track.py)')

#         # ── Loss ───────────────────────────────────────────────────────────
#         self.parser.add_argument('--id_weight', type=float, default=1.0,
#                                  help='ReID loss weight (0 = detection only)')
#         self.parser.add_argument('--tri', action='store_true',
#                                  help='add triplet loss to ReID')
#         self.parser.add_argument('--rep', action='store_true', default=False,
#                                  help='enable RepulsionLoss')
#         self.parser.add_argument('--rep_weight', type=float, default=0.5)
#         self.parser.add_argument('--use_arcface', action='store_true', default=False,
#                                  help='use ArcFace for ReID classification. Default OFF = plain '
#                                       'CE + emb_scale (the stable FairMOT/AMOT recipe).')
#         self.parser.add_argument('--s_det_init', type=float, default=2.5,
#                                  help='init for uncertainty weight s_det ≈ log(initial loss_det). '
#                                       'Read first-iter loss_det from the log and set log() of it.')
#         self.parser.add_argument('--s_id_init', type=float, default=1.85,
#                                  help='init for uncertainty weight s_id ≈ log(initial loss_reid).')

#         # ── Sequence-aware augmentation ────────────────────────────────────
#         self.parser.add_argument('--temporal_mosaic', action='store_true', default=False,
#                                  help='4 frames from same sequence → 2×2 mosaic; '
#                                       'increases positive-sample density per forward pass')
#         self.parser.add_argument('--temporal_mosaic_prob', type=float, default=0.5,
#                                  help='probability of using temporal mosaic vs single frame')
#         self.parser.add_argument('--small_obj_zoom', action='store_true', default=False,
#                                  help='zoom crop anchored on tiny objects before affine; '
#                                       'boosts gradient signal for objects <0.2%% of image area')
#         self.parser.add_argument('--small_obj_zoom_prob', type=float, default=0.5,
#                                  help='probability of applying small-object zoom per sample')
#         self.parser.add_argument('--gridmask', action='store_true', default=False,
#                                  help='GridMask: erase regular grid pattern to simulate occlusion')
#         self.parser.add_argument('--gridmask_prob', type=float, default=0.3,
#                                  help='probability of applying gridmask per sample')
#         # Random perspective (homography) warp — synthetic viewpoint augment
#         self.parser.add_argument('--homography', action='store_true', default=True,
#                                  help='random perspective warp (alt to affine) for viewpoint diversity')
#         self.parser.add_argument('--no_homography', dest='homography', action='store_false')
#         self.parser.add_argument('--homography_prob', type=float, default=0.3,
#                                  help='probability of using homography instead of affine')
#         self.parser.add_argument('--homography_strength', type=float, default=0.12,
#                                  help='corner-jitter fraction of image size (0.08-0.15 sane)')
#         # ── ReID ───────────────────────────────────────────────────────────
#         self.parser.add_argument('--reid_dim', type=int, default=128)
#         self.parser.add_argument('--reid_cls_ids', default='0,1,2,3,4,5,6,7,8,9')

#         # ── Tracking nguồn COCO (đồng bộ với training/val) ──
#         self.parser.add_argument('--track_from_coco', action='store_true', default=False,
#                                 help='lấy ảnh + danh sách seq/frame từ COCO JSON thay vì '
#                                     'test_dev/sequences thô (đồng bộ tiền xử lý với training)')
#         self.parser.add_argument('--track_ann_file', type=str, default='',
#                                 help='COCO JSON annotation cho tracking; rỗng → dùng val_ann trong data_cfg')
#         self.parser.add_argument('--track_img_root', type=str, default='',
#                                 help='thư mục ảnh tương ứng track_ann_file; rỗng → dùng val_img trong data_cfg')
#         self.parser.add_argument('--track_gt_root', type=str, default='',
#                                 help='thư mục annotation VisDrone thô của split tracking '
#                                     '(vd .../VisDrone2019-MOT-val/annotations) để Evaluator dựng GT')
#         # ── Tracking inference ─────────────────────────────────────────────
#         self.parser.add_argument('--eval_mode', type=str, default='10class',
#                                  choices=['10class', '5class', '4class',
#                                           '5class_merge_benchmark',
#                                           '5class_merge_competition'],
#                                  help='Evaluation class subset: '
#                                       '10class=all VisDrone (default, with ID offset); '
#                                       '5class=pedestrian/car/van/truck/bus, SKIP (AMOT protocol); '
#                                       '4class=person/car/motorcycle/bicycle, SKIP (competition); '
#                                       '5class_merge_benchmark=pedestrian/car/truck/tricycle/bus, '
#                                       'MERGE (drops bicycle+motor, no equivalent group); '
#                                       '5class_merge_competition=person/car/truck/motorcycle/bicycle, '
#                                       'MERGE (truck=van+truck; drops tricycle+awning-tricycle+bus, '
#                                       'no equivalent group)')
#         self.parser.add_argument('--K', type=int, default=300,
#                                  help='max detections per image at inference')
#         self.parser.add_argument('--conf_thres', type=float, default=0.4,
#                                  help='detection confidence threshold')
#         self.parser.add_argument('--track_buffer', type=int, default=30)
#         # ── Query Appearance-Motion (QAM) association ──
#         self.parser.add_argument('--use_appearance_motion', action='store_true',
#                                  help='enable appearance-as-motion association: predict each '
#                                       'track position by cross-frame correlation on the dense '
#                                       'reid map (soft-argmax), entropy-gated, fused by '
#                                       'log-likelihood. Falls back to legacy fusion if off.')
#         self.parser.add_argument('--uam_chi2', type=float, default=9.21,
#                                  help='UAM Mahalanobis gate (chi-square, 2-DOF): 5.99 = .95, '
#                                       '9.21 = .99 (more occlusion-tolerant). Statistical '
#                                       'constant, rarely tuned.')
#         self.parser.add_argument('--uam_cos_thresh', type=float, default=0.4,
#                                  help='UAM max appearance (cosine) distance for a valid match — '
#                                       'the single real association knob.')
#         self.parser.add_argument('--uam_iou_gate', type=float, default=0.7,
#                                  help='UAM IoU fallback gate (1-IoU): a pair also passes the '
#                                       'spatial gate if iou_dist <= this, so IoU still vouches '
#                                       'for adjacent-frame matches. Raise to be more permissive.')
#         self.parser.add_argument('--legacy_fuse', action='store_true',
#                                  help='force the old multiplicative fuse_score_three (A/B).')
#         self.parser.add_argument('--am_tau', type=float, default=0.07,
#                                  help='softmax temperature for the correlation response.')
#         self.parser.add_argument('--am_kappa', type=float, default=0.1,
#                                  help='motion sigma = kappa * sqrt(w*h); smaller = stricter.')
#         self.parser.add_argument('--am_beta', type=float, default=4.0,
#                                  help='entropy->confidence sharpness: w = exp(-beta*entropy).')
#         self.parser.add_argument('--am_w_app', type=float, default=1.0,
#                                  help='appearance (cosine) cue weight in log-lik fusion.')
#         self.parser.add_argument('--am_w_iou', type=float, default=1.0,
#                                  help='IoU cue weight in log-lik fusion.')
#         self.parser.add_argument('--match_thresh', type=float, default=0.7,
#                                  help='cost ceiling for the first (QAM) association.')
#         self.parser.add_argument('--proximity_thresh', type=float, default=0.95,
#                                  help='IoU-distance spatial gate: a pair with iou_dist above '
#                                       'this needs motion to vouch for it.')
#         self.parser.add_argument('--motion_gate', type=float, default=0.9,
#                                  help='motion-distance spatial gate: motion can vouch for a '
#                                       'low-IoU pair only if its motion_dist is below this.')
#         # ── Tracking-metric validation (chọn model_best theo IDF1/MOTA) ──
#         self.parser.add_argument('--emb_weight', type=float, default=1.0,
#                                  help='ReID weight in association fusion. '
#                                       '1.0=original, 0.0=IoU-only, in-between=blend.')
#         self.parser.add_argument('--emb_gate', type=float, default=0.0,
#                                  help='Chỉ áp embedding khi id_sim >= ngưỡng này; '
#                                       'dưới ngưỡng dùng IoU thuần. 0.0=tắt gating.')
#         self.parser.add_argument('--track_val', action='store_true',
#                                  help='bật validation tracking (IDF1/MOTA) để chọn model_best '
#                                       'thay cho COCO mAP. Cần --val_cfg có val_ann + val_img (COCO).')
#         self.parser.add_argument('--track_val_intervals', type=int, default=1,
#                                  help='chạy tracking-eval mỗi N epoch (0 = dùng --val_intervals).')
#         self.parser.add_argument('--track_val_fp16', type=int, default=1,
#                                  help='forward detector bằng fp16 khi tracking-eval (1=bật, nhanh hơn).')
#         self.parser.add_argument('--track_val_in_phase0', action='store_true',
#                                  help='cũng chạy tracking-eval trong Phase 0 (mặc định bỏ qua vì '
#                                       'detector đóng băng; bật nếu muốn đo riêng cải thiện reid_head).')
#         self.parser.add_argument('--track_val_w_idf1', type=float, default=0.6,
#                                  help='trọng số IDF1 trong track_score = w_idf1*IDF1 + w_mota*MOTA.')
#         self.parser.add_argument('--track_val_w_mota', type=float, default=0.4,
#                                  help='trọng số MOTA trong track_score.')
#         self.parser.add_argument('--frame_rate', type=int, default=30,
#                                  help='frame-rate giả định cho tracker (buffer = frame_rate/30 * track_buffer).')
#         # ── Tracker (FusionTrack-inspired, inference-only) ───────────────
#         self.parser.add_argument('--reid_decay_alpha', type=float, default=0.02,
#                                  help='time-decay rate for gallery ReID memory (W=e^{-alpha*dt}); 0 = no decay')
#         self.parser.add_argument('--nfm_topk', type=int, default=2,
#                                  help='mutual top-k for Neighbor Filtering gate on appearance matches')
#         self.parser.add_argument('--use_nfm', action='store_true', default=True,
#                                  help='enable mutual top-k NFM gating (reduces ID switches)')
#         self.parser.add_argument('--no_nfm', dest='use_nfm', action='store_false')
#         # ── GMC determinism + w_iou ramp (benchmark ổn định) ─────────────
#         self.parser.add_argument('--gmc_seed', type=int, default=0,
#                                  help='seed RNG của OpenCV cho RANSAC trong GMC -> benchmark không đổi')
#         self.parser.add_argument('--gmc_deterministic', action='store_true', default=True,
#                                  help='re-seed cv2 RNG mỗi frame để RANSAC tất định')
#         self.parser.add_argument('--no_gmc_deterministic', dest='gmc_deterministic',
#                                  action='store_false')
#         self.parser.add_argument('--w_iou_hi', type=float, default=0.5,
#                                  help='trọng số IoU khi GMC tin cậy (camera motion nhỏ)')
#         self.parser.add_argument('--w_iou_lo', type=float, default=0.3,
#                                  help='trọng số IoU khi GMC kém tin (camera motion lớn)')
#         self.parser.add_argument('--gmc_band_lo', type=float, default=20.0,
#                                  help='||translation|| bắt đầu giảm w_iou (ramp dưới)')
#         self.parser.add_argument('--gmc_band_hi', type=float, default=40.0,
#                                  help='||translation|| đạt w_iou_lo (ramp trên)')
#         self.parser.add_argument('--min-box-area', type=float, default=100,
#                                  help='filter boxes smaller than this area (px²)')
#         self.parser.add_argument('--test_visdrone', default=True)
#         self.parser.add_argument('--test_uavdt',    default=False)

#         # ── Distributed ────────────────────────────────────────────────────
#         self.parser.add_argument('--local-rank', type=int, default=0)

#     def parse(self, args=''):
#         opt = self.parser.parse_args() if args == '' else self.parser.parse_args(args)

#         opt.gpus_str = opt.gpus
#         opt.gpus     = [int(g) for g in opt.gpus.split(',')]
#         opt.lr_step  = [int(s) for s in opt.lr_step.split(',')]

#         if opt.lr_drop < 0:
#             opt.lr_drop = opt.num_epochs

#         if opt.trainval:
#             opt.val_intervals = 100_000_000

#         if opt.master_batch_size == -1:
#             opt.master_batch_size = opt.batch_size // len(opt.gpus)
#         rest = opt.batch_size - opt.master_batch_size
#         opt.chunk_sizes = [opt.master_batch_size]
#         for i in range(len(opt.gpus) - 1):
#             chunk = rest // (len(opt.gpus) - 1)
#             if i < rest % (len(opt.gpus) - 1):
#                 chunk += 1
#             opt.chunk_sizes.append(chunk)
#         print('chunk_sizes:', opt.chunk_sizes)

#         opt.root_dir  = os.path.join(os.path.dirname(__file__), '..', '..')
#         opt.exp_dir   = os.path.join(opt.root_dir, 'exp', opt.task)
#         opt.save_dir  = os.path.join(opt.exp_dir, opt.exp_id)
#         opt.debug_dir = os.path.join(opt.save_dir, 'debug')
#         print('Output will be saved to', opt.save_dir)

#         # Stage-1 detection-only: VisDrone-DET has no track IDs.
#         if opt.train_single_det:
#             opt.use_reid           = False
#             opt.id_weight          = 0.0
#             opt.temporal_mosaic    = False
#             opt.reid_warmup_epochs = 0
#             opt.id_warmup_epochs   = 0
#             s4_tag = '+s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''
#             print('[train_single_det] stage-1 detection-only: '
#                   f'ReID head OFF | losses: cls+bbox+giou{s4_tag} | '
#                   f'use_s4={getattr(opt, "use_s4", False)} | '
#                   f'mosaic={getattr(opt, "mosaic", False)} | temporal_mosaic=OFF')
#         else:
#             opt.use_reid = True

#         if opt.resume and opt.load_model == '':
#             base = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
#             opt.load_model = os.path.join(base, 'model_last.pth')

#         return opt

#     def update_dataset_info_and_set_heads(self, opt, dataset):
#         input_h, input_w = dataset.default_input_wh
#         opt.mean, opt.std = dataset.mean, dataset.std
#         opt.num_classes   = dataset.num_classes
#         print('num_classes:', opt.num_classes)

#         for reid_id in opt.reid_cls_ids.split(','):
#             if int(reid_id) > opt.num_classes - 1:
#                 if getattr(opt, 'train_single_det', False):
#                     break
#                 print('[Err]: reid_cls_ids conflicts with num_classes')
#                 return

#         input_h = opt.input_res if opt.input_res > 0 else input_h
#         input_w = opt.input_res if opt.input_res > 0 else input_w
#         opt.input_h   = opt.input_h if opt.input_h > 0 else input_h
#         opt.input_w   = opt.input_w if opt.input_w > 0 else input_w
#         opt.input_res = max(opt.input_h, opt.input_w)

#         if opt.task == 'mot':
#             # Always expose nID_dict (criterion reads it even when ReID is off).
#             opt.nID_dict = dataset.nID_dict
#         else:
#             assert 0, 'task not defined!'

#         return opt

#     def init(self, args=''):
#         opt = self.parse(args)

#         default_dataset_info = {
#             'mot': {
#                 'default_input_wh': [opt.input_wh[1], opt.input_wh[0]],
#                 'num_classes':       len(opt.reid_cls_ids.split(',')),
#                 'mean': [0.485, 0.456, 0.406],
#                 'std':  [0.229, 0.224, 0.225],
#                 'dataset':  'coco',
#                 'nID_dict': {},
#             },
#         }

#         class Struct:
#             def __init__(self, entries):
#                 for k, v in entries.items():
#                     setattr(self, k, v)

#         h_w = default_dataset_info[opt.task]['default_input_wh']
#         opt.img_size = (h_w[1], h_w[0])
#         print('Net input: {:d}×{:d}'.format(h_w[1], h_w[0]))

#         dataset     = Struct(default_dataset_info[opt.task])
#         opt.dataset = dataset.dataset
#         opt = self.update_dataset_info_and_set_heads(opt, dataset)
#         return opt