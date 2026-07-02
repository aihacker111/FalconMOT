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
                                 help='STAGE 1: Detection-only training. No ReID head, '
                                      'no ReID loss. Save checkpoint, then run '
                                      'Stage-2 tracking fine-tune with --train_reid_only.')
        self.parser.add_argument('--train_stage2_mot', action='store_true', default=False,
                                 help='STAGE 2: MOT Fine-tuning. Freeze Backbone & Encoder, '
                                      'train ONLY Decoder & ReID head.')
        self.parser.add_argument('--dataset',  default='coco',
                                 help='coco (COCO JSON format) — only supported format')
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
        self.parser.add_argument('--conv_inplane', type=int, default=32,
                                 help='SpatialPriorModule base channels (32: richer S4 detail)')
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
        self.parser.add_argument('--num_denoising', type=int, default=100,
                                 help='CDN denoising queries (200: faster convergence on dense VisDrone scenes)')
        self.parser.add_argument('--reg_max',       type=int, default=32)
        self.parser.add_argument('--mal_alpha',     type=float, default=None,
                                 help='DEIM MAL negative weight; None=original repo default, try 0.2 if negatives are too strong')
 
        # S4 branch (stride-4 for small/distant object detection)
        self.parser.add_argument('--use_s4', action='store_true', default=False,
                                 help='Add stride-4 branch: decoder uses [S4,S8,S16,S32].')
        self.parser.add_argument('--use_s4_aux', action='store_true', default=False,
                                 help='enable S4 auxiliary heatmap loss (default OFF: overlaps enc_score_head)')
        self.parser.add_argument('--no_s4_aux', dest='use_s4_aux', action='store_false',
                                 help='disable S4 auxiliary loss/head; no aux gradient')
 
        # ── Fovea-MOT: SAFA (Scale-Adaptive Foveal Attention) ───────────────
        self.parser.add_argument('--use_safa', action='store_true', default=False,
                                 help='Entropy-gated sparse S4 fusion + scale-adaptive '
                                      'deformable routing (requires --use_s4).')
        self.parser.add_argument('--safa_keep_ratio', type=float, default=0.25,
                                 help='fraction of S8 cells kept for heavy S4 refinement '
                                      '(0.25 -> ~75%% GFLOPs cut on the S4 branch).')
        self.parser.add_argument('--no_safa_scale_adaptive', dest='safa_scale_adaptive',
                                 action='store_false', default=True,
                                 help='disable the per-query scale-adaptive level routing in the decoder.')
        self.parser.add_argument('--use_entropy_aux', action='store_true', default=True,
                                 help='supervise the SAFA entropy scorer with a center heatmap.')
        self.parser.add_argument('--no_entropy_aux', dest='use_entropy_aux', action='store_false')
        self.parser.add_argument('--entropy_weight', type=float, default=0.5)
 
        # ── Fovea-MOT: SI-WBD bounding-box loss ─────────────────────────────
        self.parser.add_argument('--use_siwbd', action='store_true', default=False,
                                 help='Scale-Invariant Wasserstein-Bures box loss for tiny objects.')
        self.parser.add_argument('--siwbd_C', type=float, default=0.5,
                                 help='area-normalisation spread (smaller = sharper).')
        self.parser.add_argument('--siwbd_weight', type=float, default=2.0,
                                 help='weight of loss_siwbd (only used in box_reg_mode=add).')
        self.parser.add_argument('--box_reg_mode', default='blend',
                                 choices=['add', 'replace', 'blend'],
                                 help="overlap-regression term when --use_siwbd: "
                                      "add=GIoU+SI-WBD (2 signals), replace=SI-WBD only, "
                                      "blend=size-gated convex mix (small->SI-WBD, large->GIoU).")
        self.parser.add_argument('--siwbd_beta', type=float, default=1.0,
                                 help='blend: size-gate sharpness multiplier (smaller = harder switch).')
        self.parser.add_argument('--siwbd_gate_center', type=float, default=0.003,
                                 help='blend: ABSOLUTE small/large boundary as NORMALISED box area '
                                      '(w*h, w,h in [0,1]). Default 0.003 ~= 32^2 px on a 960x544 canvas. '
                                      'Objects below this lean to SI-WBD, above to GIoU. Scale with '
                                      'resolution: center ~= (s/W)*(s/H) for a side of s px.')
        self.parser.add_argument('--siwbd_gate_scale', type=float, default=1.0,
                                 help='blend: transition width of the gate in log-area units (fixed, '
                                      'replaces the old batch-std estimate).')
        self.parser.add_argument('--siwbd_gate_dynamic', action='store_true', default=False,
                                 help='blend: EMA-track the gate center from the DATASET log-area '
                                      'distribution instead of a fixed threshold. Stable (not per-batch '
                                      'noisy) and auto-adapts to dataset/resolution. Init from '
                                      '--siwbd_gate_center.')
        self.parser.add_argument('--siwbd_gate_momentum', type=float, default=0.99,
                                 help='blend: EMA momentum for the dynamic gate center (closer to 1 = '
                                      'slower/steadier).')
        self.parser.add_argument('--siwbd_replaces_giou', action='store_true', default=False,
                                 help='DEPRECATED: equivalent to --box_reg_mode replace.')
 
        # ── Fovea-MOT: T-UCL uncertainty-weighted ReID ──────────────────────
        self.parser.add_argument('--use_tucl', action='store_true', default=False,
                                 help='down-weight ReID loss on low-quality (small/blurry) boxes.')
        self.parser.add_argument('--tucl_lambda', type=float, default=0.05,
                                 help='weight of the -log(w) anti-collapse regulariser.')
 
        # -- Merge train + val into a single training set --
        self.parser.add_argument('--merge_val_into_train', action='store_true', default=False,
                                 help='Merge both train_ann/img and val_ann/img (from data_cfg) into '
                                      'the training set. image_id/seq_id/track_id are offset to avoid '
                                      'collisions. Note: no separate val set remains for evaluation.')
        # ── ReID head ────────────────────────────────────────────────────
        self.parser.add_argument('--reid_num_points', type=int, default=8,
                                 help='number of deformable sample points per box for the ReID head')
        self.parser.add_argument('--reid_grad_scale', type=float, default=0.1,
                                 help='strength of the ReID gradient flowing into the trunk via the feature map '
                                      '(1.0 = full JDE coupling; lower to ~0.1 if detection gets noisy).')
 
        # Pretrained / spatial size
        self.parser.add_argument('--deim_pretrained', default='',
                                 help='pretrained DEIM checkpoint (weights only, no optimizer)')
        self.parser.add_argument('--eval_spatial_size', type=int, nargs=2,
                                 default=[480, 864], help='[H, W] for anchor pre-generation')
 
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
 
        # ── Augmentation ───────────────────────────────────────────────────
        self.parser.add_argument('--stop_epoch', type=int, default=-1,
                                 help='epoch to disable aug (-1 = always on)')
        self.parser.add_argument('--mosaic', action='store_true', default=False)
        self.parser.add_argument('--mosaic_prob', type=float, default=0.5)
 
        # ── Dataset config ─────────────────────────────────────────────────
        self.parser.add_argument('--data_cfg', type=str,
                                 default='configs/visdrone_coco.json')
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
        self.parser.add_argument('--use_arcface', action='store_true', default=False,
                                 help='use ArcFace for ReID classification. Default OFF = plain '
                                      'CE + emb_scale (the stable FairMOT/AMOT recipe).')
        self.parser.add_argument('--s_det_init', type=float, default=2.5,
                                 help='init for uncertainty weight s_det ≈ log(initial loss_det). '
                                      'Read first-iter loss_det from the log and set log() of it.')
        self.parser.add_argument('--s_id_init', type=float, default=2.25,
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
        self.parser.add_argument('--homography', action='store_true', default=False,
                                 help='random perspective warp (alt to affine) for viewpoint diversity')
        self.parser.add_argument('--no_homography', dest='homography', action='store_false')
        self.parser.add_argument('--homography_prob', type=float, default=0.3,
                                 help='probability of using homography instead of affine')
        self.parser.add_argument('--homography_strength', type=float, default=0.12,
                                 help='corner-jitter fraction of image size (0.08-0.15 sane)')
        self.parser.add_argument('--obj_occlusion', action='store_true', default=False)
        self.parser.add_argument('--no_obj_occlusion', dest='obj_occlusion', action='store_false')
        self.parser.add_argument('--obj_occ_prob', type=float, default=0.5)
        self.parser.add_argument('--obj_occ_frac', type=float, default=0.3)
        self.parser.add_argument('--obj_occ_mode', type=str,   default='patch',
                            choices=['patch', 'random', 'mean'])
        self.parser.add_argument('--random_erasing', action='store_true')
        self.parser.add_argument('--re_prob', type=float, default=0.25)
 
        # ── ReID ───────────────────────────────────────────────────────────
        self.parser.add_argument('--reid_dim', type=int, default=128)
        self.parser.add_argument('--reid_cls_ids', default='0,1,2,3,4,5,6,7,8,9')
 
        # ── Tracking inference ─────────────────────────────────────────────
        self.parser.add_argument('--K', type=int, default=300,
                                 help='max detections per image at inference')
        self.parser.add_argument('--conf_thres', type=float, default=0.4,
                                 help='detection confidence threshold')
        self.parser.add_argument('--track_buffer', type=int, default=30,
                                 help='frames a lost track is kept alive')
        self.parser.add_argument('--frame_rate', type=int, default=30,
                                 help='assumed frame rate for the tracker (buffer = frame_rate/30 * track_buffer).')
        self.parser.add_argument('--min-box-area', type=float, default=100,
                                 help='filter boxes smaller than this area (px²)')
        self.parser.add_argument('--track_ann_file', type=str, default='',
                                 help='COCO JSON annotation for the tracking split (eval_mot_* tools); '
                                      'empty -> tool default')
        self.parser.add_argument('--track_img_root', type=str, default='',
                                 help='image directory matching track_ann_file; empty -> tool default')
        # ── Query Appearance-Motion (QAM) association ──
        self.parser.add_argument('--use_appearance_motion', action='store_true',
                                 help='enable appearance-as-motion association')
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
                                 help='IoU-distance spatial gate')
        self.parser.add_argument('--motion_gate', type=float, default=0.9,
                                 help='motion-distance spatial gate')
 
        # ── Tracking validation during training (MOTA / IDF1) ───────────────
        # Runs automatically every --track_val_intervals epochs (stage-2/joint)
        # whenever val_cfg provides val sequences; model_best.pth is selected by
        # track_score = (w_idf1*IDF1 + w_mota*MOTA) / (w_idf1 + w_mota).
        self.parser.add_argument('--track_val_intervals', type=int, default=1,
                                 help='run tracking eval every N epochs (0 = use --val_intervals).')
        self.parser.add_argument('--track_val_fp16', type=int, default=1,
                                 help='run the detector forward in fp16 during tracking eval (1=on, faster).')
        self.parser.add_argument('--track_val_loader_threads', type=int, default=4,
                                 help='worker threads that read+preprocess frames during track eval '
                                      '(overlaps disk I/O with GPU inference).')
        self.parser.add_argument('--track_val_prefetch', type=int, default=8,
                                 help='number of frames read ahead per sequence during track eval.')
        self.parser.add_argument('--track_val_gmc', type=int, default=0,
                                 help='run GMC (camera-motion compensation) during tracking eval. '
                                      'Default 0: GMC is serial CPU optical flow (~5-15 ms/frame) '
                                      'and stalls the GPU; enable only if val has strong camera motion.')
        self.parser.add_argument('--track_val_w_idf1', type=float, default=0.6,
                                 help='IDF1 weight in track_score.')
        self.parser.add_argument('--track_val_w_mota', type=float, default=0.4,
                                 help='MOTA weight in track_score.')
 
        # ── ReID dense map (stride-4) ───────────────────────────────────────
        self.parser.add_argument('--reid_use_s4_dense', action='store_true',
                      help='Dense ReID map at stride-4 (small objects). Requires c1=_s4_feat.')
        self.parser.add_argument('--reid_lr_factor', type=float, default=10.0,
                      help='LR multiplier for reid_head + ID classifiers (fresh params '
                           'need ~1e-4+ while pretrained trunk fine-tunes at a small lr).')
 
        # ── OSD — Orthogonal Subspace Decoupling ────────────────────────────
        self.parser.add_argument('--use_osd', type=int, default=1,
                      help='1 = split the shared feature map into orthogonal det/ReID '
                           'subspaces via a learned rotation Q in O(C); task-gradient '
                           'inner product at the branch point is exactly 0. 0 = off.')
        self.parser.add_argument('--osd_id_ratio', type=float, default=0.33,
                      help='fraction of rotated channels routed to the ReID subspace. '
                           'Default 0.33: reid_feat (p2) is stride-4 and rich in detail; '
                           'rank 48 (ratio 0.25) is too tight to feed reid_dim=128 after '
                           'the nonlinear tower — 64 dims is the sweet spot. '
                           '(sweep 0.25 / 0.33 / 0.4 in ablations).')
        self.parser.add_argument('--decorr_weight', type=float, default=0.02,
                      help='weight of the cross-covariance (Barlow-style) penalty that '
                           'enforces statistical independence between the two subspaces.')
 
    def parse(self, args=''):
        opt = self.parser.parse_args() if args == '' else self.parser.parse_args(args)
 
        opt.gpus_str = opt.gpus
        opt.gpus     = [int(g) for g in opt.gpus.split(',')]
        opt.lr_step  = [int(s) for s in opt.lr_step.split(',')]
 
        if opt.lr_drop < 0:
            opt.lr_drop = opt.num_epochs
 
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
 
        # Check mutual exclusion
        if opt.train_single_det and opt.train_stage2_mot:
            raise ValueError("Error: --train_single_det and --train_reid_only cannot be used together!")
 
        # Stage-1 detection-only: VisDrone-DET has no track IDs.
        if opt.train_single_det:
            opt.use_reid           = False
            opt.id_weight          = 0.0
            opt.temporal_mosaic    = False
            s4_tag = '+s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''
            print('[train_single_det] STAGE 1 (Detection-only): '
                  f'ReID head OFF | losses: cls+bbox+giou{s4_tag} | '
                  f'use_s4={getattr(opt, "use_s4", False)} | '
                  f'mosaic={getattr(opt, "mosaic", False)} | temporal_mosaic=OFF')
        elif opt.train_stage2_mot:
            opt.use_reid = True
            print('[train_reid_only] STAGE 2 (ReID-only): '
                  'Detector FROZEN | Training ONLY ReID head & Orthogonal loss.')
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