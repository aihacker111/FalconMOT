from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os
import json

import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.utils.data
from torchvision.transforms import transforms as T

import _paths  # noqa: F401  (sys.path bootstrap)
from falconmot.opts import opts
from falconmot.models.model import create_model, load_model, save_model
from falconmot.models.data_parallel import DataParallel
from falconmot.logger import Logger
from falconmot.datasets.dataset_factory import get_dataset
from falconmot.engine.train_factory import train_factory
from falconmot.engine import stage as stage_mgr
from falconmot.utils.jde_eval import CocoJsonEvaluator, JDECocoEvaluator
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor
# ── Tracking-metric validation (motmetrics IDF1/MOTA) ──────────────────────
from collections import defaultdict as _defaultdict
from falconmot.tracker.multitracker import MCJDETracker, MCTrack
from falconmot.tracking_utils.coco_gt_reader import CocoGTEvaluator
from falconmot.tracking_utils.evaluation import Evaluator
from falconmot.datasets.dataset.coco_detection import LoadCocoSequencesForTracking


_NORM_TYPES = (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm2d,
)


def _with_lr(opt, lr):
    """Shallow copy of opt with a different base lr (for per-phase optimizer)."""
    o = copy.copy(opt)
    o.lr = lr
    return o


def build_optimizer(model, opt):
    """
    AdamW with 4 param groups:
      - backbone non-norm : lr * 0.05, weight_decay = default
      - backbone norm/bias: lr * 0.05, weight_decay = 0
      - other   norm/bias : lr,        weight_decay = 0
      - everything else   : lr,        weight_decay = default

    Norm layers are detected by module type (not just param name) so that
    BN gamma params inside anonymous nn.Sequential blocks (e.g.
    backbone.sta.stem.1.weight) are correctly assigned weight_decay=0.
    """
    base_lr      = opt.lr
    backbone_lr  = base_lr * 0.05
    weight_decay = opt.weight_decay

    # Collect FQNs of all params that belong to a normalization module.
    norm_param_names: set = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, _NORM_TYPES):
            for p_name, _ in module.named_parameters(recurse=False):
                norm_param_names.add(f'{mod_name}.{p_name}')

    backbone_wd, backbone_no_wd, other_no_wd, default = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = 'backbone' in name
        is_no_wd    = name in norm_param_names or name.endswith('.bias')
        if is_backbone and is_no_wd:
            backbone_no_wd.append(param)
        elif is_backbone:
            backbone_wd.append(param)
        elif is_no_wd:
            other_no_wd.append(param)
        else:
            default.append(param)

    param_groups = [
        pg for pg in [
            {'params': backbone_wd,    'lr': backbone_lr},
            {'params': backbone_no_wd, 'lr': backbone_lr, 'weight_decay': 0.},
            {'params': other_no_wd,                        'weight_decay': 0.},
            {'params': default},
        ]
        if pg['params']
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    return optimizer


def _flat_cosine_lr(total_iter, warmup_iter, flat_iter, no_aug_iter,
                    current_iter, init_lr, min_lr):
    """Quadratic warmup → flat → cosine decay → no-aug. Ported from EdgeCrafter."""
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(max(1, warmup_iter))) ** 2
    if current_iter <= flat_iter:
        return init_lr
    if current_iter >= total_iter - no_aug_iter:
        return min_lr
    cosine = 0.5 * (1.0 + math.cos(
        math.pi * (current_iter - flat_iter) /
        float(max(1, total_iter - flat_iter - no_aug_iter))
    ))
    return min_lr + (init_lr - min_lr) * cosine


class FlatCosineScheduler:
    """
    Flat-cosine LR scheduler compatible with base_trainer's scheduler.step() API.

    Phases (per iteration):
      1. Quadratic warmup  : 0 → init_lr  over warmup_iter steps
      2. Flat              : init_lr constant for flat_epoch epochs
      3. Cosine decay      : init_lr → min_lr
      4. No-aug            : min_lr constant for no_aug_epochs final epochs

    flat_epochs  — epochs where LR stays at init_lr (use stop_epoch or num_epochs//2)
    no_aug_epochs — final constant-LR epochs (EdgeCrafter default: 2)
    lr_gamma      — min_lr = init_lr × lr_gamma  (default 0.01)
    """

    def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
                 warmup_iter: int, flat_epochs: int, no_aug_epochs: int,
                 lr_gamma: float):
        # Snapshot initial LR for each param group
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        self._optimizer   = optimizer
        self._base_lrs    = [g['initial_lr'] for g in optimizer.param_groups]
        self._min_lrs     = [lr * lr_gamma   for lr in self._base_lrs]
        self._cur_iter    = 0

        total_iter   = num_epochs    * steps_per_epoch
        flat_iter    = flat_epochs   * steps_per_epoch
        no_aug_iter  = no_aug_epochs * steps_per_epoch

        self._total_iter  = total_iter
        self._warmup_iter = warmup_iter
        self._flat_iter   = flat_iter
        self._no_aug_iter = no_aug_iter

        print(
            f'[FlatCosineScheduler] total={total_iter} warmup={warmup_iter} '
            f'flat={flat_iter} no_aug={no_aug_iter} '
            f'base_lrs={self._base_lrs} min_lrs={self._min_lrs}'
        )

    def step(self):
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = _flat_cosine_lr(
                self._total_iter, self._warmup_iter,
                self._flat_iter,  self._no_aug_iter,
                self._cur_iter,
                self._base_lrs[i], self._min_lrs[i],
            )
        self._cur_iter += 1

    def fast_forward(self, n_steps: int):
        """Jump scheduler state by n_steps without looping (O(1) instead of O(n))."""
        self._cur_iter = n_steps
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = _flat_cosine_lr(
                self._total_iter, self._warmup_iter,
                self._flat_iter,  self._no_aug_iter,
                self._cur_iter,
                self._base_lrs[i], self._min_lrs[i],
            )

    def state_dict(self):
        return {'cur_iter': self._cur_iter}

    def load_state_dict(self, state: dict):
        self._cur_iter = state.get('cur_iter', 0)


class RFDETRScheduler:
    """
    LR scheduler ported from RF-DETR: linear warmup + cosine annealing or step decay.

    Phases (per step):
      1. Linear warmup : 0 → base_lr  over warmup_steps
      2a. Cosine       : lr_min_factor + (1-lr_min_factor)*0.5*(1+cos(π*progress))
      2b. Step         : base_lr until lr_drop epochs, then base_lr * 0.1

    Activated via --lr_scheduler cosine|step. Params from opt:
      warmup_epochs  (float, default 0.0)
      lr_drop        (int,   default = num_epochs)
      lr_min_factor  (float, default 0.0)
    """

    def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
                 warmup_epochs: float, lr_drop: int, lr_scheduler: str,
                 lr_min_factor: float):
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        self._optimizer     = optimizer
        self._base_lrs      = [g['initial_lr'] for g in optimizer.param_groups]
        self._cur_step      = 0
        self._total_steps   = num_epochs * steps_per_epoch
        self._warmup_steps  = int(steps_per_epoch * warmup_epochs)
        self._lr_drop_steps = lr_drop * steps_per_epoch
        self._lr_scheduler  = lr_scheduler
        self._lr_min_factor = lr_min_factor

        print(
            f'[RFDETRScheduler] type={lr_scheduler} total={self._total_steps} '
            f'warmup={self._warmup_steps} lr_drop_step={self._lr_drop_steps} '
            f'lr_min_factor={lr_min_factor} base_lrs={self._base_lrs}'
        )

    def _compute_lr(self, step: int, base_lr: float) -> float:
        if step < self._warmup_steps:
            return base_lr * float(step) / float(max(1, self._warmup_steps))
        if self._lr_scheduler == 'cosine':
            progress = float(step - self._warmup_steps) / float(
                max(1, self._total_steps - self._warmup_steps))
            factor = self._lr_min_factor + (1.0 - self._lr_min_factor) * 0.5 * (
                1.0 + math.cos(math.pi * progress))
            return base_lr * factor
        return base_lr if step < self._lr_drop_steps else base_lr * 0.1

    def step(self):
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])
        self._cur_step += 1

    def fast_forward(self, n_steps: int):
        self._cur_step = n_steps
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])

    def state_dict(self):
        return {'cur_step': self._cur_step}

    def load_state_dict(self, state: dict):
        self._cur_step = state.get('cur_step', 0)


def build_scheduler(optimizer, opt, steps_per_epoch: int):
    """
    Dispatch on opt.lr_scheduler:
      'cosine' | 'step'  → RFDETRScheduler (linear warmup + cosine/step)
      anything else      → FlatCosineScheduler (quadratic warmup + flat + cosine, legacy)
    """
    lr_scheduler = getattr(opt, 'lr_scheduler', 'flat_cosine')

    if lr_scheduler in ('cosine', 'step'):
        return RFDETRScheduler(
            optimizer,
            steps_per_epoch = steps_per_epoch,
            num_epochs      = opt.num_epochs,
            warmup_epochs   = getattr(opt, 'warmup_epochs',  0.0),
            lr_drop         = getattr(opt, 'lr_drop',        opt.num_epochs),
            lr_scheduler    = lr_scheduler,
            lr_min_factor   = getattr(opt, 'lr_min_factor',  0.0),
        )

    # legacy flat-cosine
    warmup_iter   = getattr(opt, 'warmup_iters',  2000)
    num_epochs    = opt.num_epochs
    no_aug_epochs = getattr(opt, 'no_aug_epochs', 2)
    lr_gamma      = getattr(opt, 'lr_gamma',      0.5)
    stop_epoch    = getattr(opt, 'stop_epoch',    -1)
    flat_epochs   = stop_epoch if stop_epoch > 0 else num_epochs // 2

    return FlatCosineScheduler(
        optimizer, steps_per_epoch, num_epochs,
        warmup_iter, flat_epochs, no_aug_epochs, lr_gamma,
    )


def _print_stage1_banner(opt):
    """Log key settings for --train_single_det stage-1 runs."""
    if not getattr(opt, 'train_single_det', False):
        return
    w, h = opt.input_wh[0], opt.input_wh[1]
    es = getattr(opt, 'eval_spatial_size', [h, w])
    print('=' * 72)
    print('Stage-1 detection-only training  (--train_single_det)')
    print(f'  input {w}x{h}  eval_spatial_size={es}  use_s4={getattr(opt, "use_s4", False)} '
          f'use_s4_aux={getattr(opt, "use_s4_aux", True)}')
    print(f'  losses: cls + bbox + giou'
          f"{' + s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''}  |  "
          f'ReID: OFF')
    print(f'  aug: mosaic={getattr(opt, "mosaic", False)} '
          f'(p={getattr(opt, "mosaic_prob", 0.5)})  '
          f'temporal_mosaic=OFF')
    if getattr(opt, 'deim_pretrained', ''):
        print(f'  deim_pretrained: {opt.deim_pretrained}')
    print('  stage-2: load checkpoint WITHOUT --train_single_det')
    print('=' * 72)


def _summarize_model(model, opt):
    core = model.module if hasattr(model, 'module') else model
    print(f'[model] use_s4={getattr(core, "use_s4", False)}  '
          f'use_s4_aux={getattr(core, "use_s4_aux", True)}  '
          f'use_reid={getattr(core, "use_reid", True)}  '
          f's4_branch={hasattr(core, "s4_branch")}  '
          f'reid_head={hasattr(core, "reid_head")}')
    if getattr(opt, 'train_single_det', False) and getattr(opt, 'use_s4', False) \
            and getattr(opt, 'deim_pretrained', ''):
        print('[model] NOTE: s4_branch/s4_aux_head not in DEIM pretrained — '
              'trained from scratch in stage-1.')


@torch.no_grad()
def run_coco_eval(model, val_loader, opt, ann_file: str = '') -> dict:
    """
    COCO mAP evaluation — DEIMv2-compatible.

    ann_file set  → CocoJsonEvaluator: GT từ COCO JSON, dùng real image_id (COCO format)
    ann_file ''   → JDECocoEvaluator:  GT rebuild từ batch với letterbox inverse (JDE format)
    """
    model.eval()

    net_h, net_w = opt.input_wh[1], opt.input_wh[0]

    postprocessor = FalconJDEPostProcessor(
        num_classes=opt.num_classes,
        num_top_queries=500,
        use_focal_loss=True,
        conf_thres=0.0
    )
    # Plain resize: KHÔNG set_net_hw -> postprocessor dùng nhánh norm*orig (đúng cho resize thẳng)

    if ann_file:
        evaluator = CocoJsonEvaluator(ann_file)
    else:
        evaluator = JDECocoEvaluator(
            num_classes=opt.num_classes,
            net_h=net_h,
            net_w=net_w,
        )

    max_batches = getattr(opt, 'debug_val_batches', 0)

    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(opt.device, non_blocking=True)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        orig_hw    = batch.get('orig_hw')
        orig_sizes = orig_hw if orig_hw is not None else \
                     torch.tensor([[net_h, net_w]] * batch['input'].shape[0],
                                  device=opt.device)

        outputs    = model(batch['input'])
        dt_results = postprocessor(outputs, orig_sizes)   # plain resize -> norm*orig
        evaluator.update(dt_results, batch)

    model.train()
    return evaluator.summarize()


@torch.no_grad()
def run_track_eval(model, opt, val_ann_file: str, val_img_root: str) -> dict:
    """Tracking validation trên các sequence của tập val (motmetrics IDF1/MOTA).

    Chạy ĐÚNG pipeline như scripts/track_5cls.py nhưng:
      * trong KHÔNG GIAN class gốc (opt.num_classes), KHÔNG remap 7->5;
      * TẮT GMC để nhanh + bất biến — không bao giờ gọi tracker.set_image(),
        nên MCJDETracker._curr_img luôn None và khối GMC bị bỏ qua;
      * forward detector tuỳ chọn fp16 (mặc định bật trên CUDA).

    GT đọc thẳng từ val COCO JSON qua CocoGTEvaluator; track-id dự đoán được gắn
    offset toàn cục theo cùng công thức (id + cls*1_000_000) để khớp với GT.

    GIẢ ĐỊNH: category_id trong val JSON == (chỉ số class của model) + 1
    (chuẩn COCO 1-indexed). Nếu val set của bạn dùng scheme class khác model,
    cần map lại trước khi đẩy vào tracker.

    Trả về: {'idf1','mota','num_switches','track_score','fps'}.
    """
    import time

    if not (val_ann_file and val_img_root and os.path.isfile(val_ann_file)):
        print('[track-eval] thiếu val_ann/val_img — bỏ qua tracking eval.')
        return {}

    model.eval()
    net_w, net_h = opt.img_size                      # opt.img_size = (W, H)
    ncls     = opt.num_classes
    min_area = getattr(opt, 'min_box_area', 100)
    use_fp16 = bool(getattr(opt, 'track_val_fp16', 1)) and opt.device.type == 'cuda'
    _OFF     = 1_000_000                             # phải khớp _CLS_ID_OFFSET trong coco_gt_reader

    postproc = FalconJDEPostProcessor(
        num_classes=ncls,
        num_top_queries=getattr(opt, 'K', 300),
        conf_thres=opt.conf_thres,
        use_focal_loss=True,
    )
    # LƯU Ý: KHÔNG gọi set_net_hw(). Loader val dùng plain-resize (preprocess_for_tracking),
    # nên postprocessor phải dùng nhánh "simple scale" (norm*orig) — giống hệt run_coco_eval.
    # (track_5cls gọi set_net_hw + plain-resize là không nhất quán, gây lệch toạ độ box
    #  với ảnh không cùng aspect-ratio mạng -> IoU sai -> IDF1/MOTA bị hạ oan.)

    # Tracker chạy native class space. KHÔNG gọi set_image() => GMC tắt.
    opt_trk = copy.copy(opt)
    opt_trk.num_classes = ncls
    tracker = MCJDETracker(opt_trk, frame_rate=getattr(opt, 'frame_rate', 30))

    src = LoadCocoSequencesForTracking(val_ann_file, val_img_root, img_size=opt.img_size)

    accs, names = [], []
    t0, n_frames = time.time(), 0

    for seq_id in src.seqs:
        tracker.reset()
        ev = CocoGTEvaluator(val_ann_file, seq_id)
        ev.reset_accumulator()

        for frame_id, img, img0 in src.sequence(seq_id):
            orig_h, orig_w = img0.shape[:2]
            sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
            blob  = torch.from_numpy(img[None]).to(opt.device)

            if use_fp16:
                with torch.cuda.amp.autocast():
                    output = model(blob)
            else:
                output = model(blob)
            res = postproc(output, sizes)[0]

            # decode -> per-class MCTrack (native class space)
            dets = _defaultdict(list)
            if len(res['scores']) > 0:
                bxs = res['boxes'].float().cpu().numpy()
                scs = res['scores'].float().cpu().numpy()
                lbs = res['labels'].cpu().numpy()
                rid = res['reid'].float().cpu().numpy() if 'reid' in res else None
                ws  = bxs[:, 2] - bxs[:, 0]
                hs  = bxs[:, 3] - bxs[:, 1]
                for i in np.where((ws > 0) & (hs > 0))[0]:
                    c = int(lbs[i])
                    if c < 0 or c >= ncls:
                        continue
                    tlwh = np.array([bxs[i, 0], bxs[i, 1], ws[i], hs[i]], dtype=np.float32)
                    emb  = rid[i] if rid is not None else np.zeros(1, dtype=np.float32)
                    dets[c].append(MCTrack(tlwh, float(scs[i]), emb, ncls, c))

            # CHỦ Ý: không set_image() -> GMC bị bỏ qua.
            online = tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

            tlwhs, tids = [], []
            for c, tracks in online.items():
                for t in tracks:
                    w, h = t.curr_tlwh[2], t.curr_tlwh[3]
                    if t.track_id < 0 or (w * h) <= min_area:
                        continue
                    tlwhs.append(t.curr_tlwh)
                    tids.append(int(t.track_id) + c * _OFF)
            ev.eval_frame(int(frame_id), tlwhs, tids)
            n_frames += 1

        accs.append(ev.acc)
        names.append(seq_id)

    model.train()

    if not accs:
        return {}

    summary = Evaluator.get_summary(
        accs, names, metrics=('mota', 'num_switches', 'idf1', 'idp', 'idr'))
    row  = summary.loc['OVERALL']
    idf1 = float(row['idf1'])
    mota = float(row['mota'])
    nsw  = int(row['num_switches'])
    w_id = float(getattr(opt, 'track_val_w_idf1', 0.6))
    w_mo = float(getattr(opt, 'track_val_w_mota', 0.4))
    fps  = n_frames / max(1e-6, time.time() - t0)
    return {
        'idf1': idf1, 'mota': mota, 'num_switches': nsw,
        'track_score': w_id * idf1 + w_mo * mota, 'fps': fps,
    }


def run(opt):
    _print_stage1_banner(opt)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset      = get_dataset(opt.dataset, opt.task)
    use_coco_fmt = (opt.dataset == 'coco')

    f = open(opt.data_cfg)
    data_config  = json.load(f)
    dataset_root = data_config['root']
    print("Dataset root: %s" % dataset_root)
    f.close()

    from falconmot.datasets.dataset.coco_detection import VisDroneCocoDataset

    if use_coco_fmt:
        # ── Gom nhiều nguồn vào tập train (vd train + val) ────────────────
        #   Ưu tiên: config['train_sources'] (list) > cờ --merge_val_into_train
        #   > nguồn đơn train_ann/train_img (mặc định, như cũ).
        train_sources = data_config.get('train_sources')
        if not train_sources and getattr(opt, 'merge_val_into_train', False):
            train_sources = [
                {'ann': data_config['train_ann'], 'img': data_config['train_img']},
                {'ann': data_config['val_ann'],   'img': data_config['val_img']},
            ]
            print('[data] merge_val_into_train: gộp train + val vào tập train')

        if train_sources:
            dataset = VisDroneCocoDataset(opt=opt, sources=train_sources, augment=True)
        else:
            train_ann = data_config['train_ann']
            train_img = data_config['train_img']
            dataset   = VisDroneCocoDataset(
                opt=opt, img_root=train_img, ann_file=train_ann, augment=True)
    else:
        trainset_paths = data_config['train']
        transforms     = T.Compose([T.ToTensor()])
        dataset        = Dataset(opt=opt, root=dataset_root,
                                 paths=trainset_paths, img_size=opt.input_wh,
                                 augment=True, transforms=transforms)

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    print("opt.gpus_str: ", opt.gpus_str)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # ── Val dataset (optional) ──────────────────────────────────────────
    # val_loader   = None
    # val_ann_file = ''   # used by CocoJsonEvaluator when use_coco_fmt

    # if getattr(opt, 'val_cfg', ''):
    #     with open(opt.val_cfg) as f:
    #         val_config = json.load(f)
    #     val_dataset = None

    #     if use_coco_fmt:
    #         val_ann_file = val_config.get('val_ann', '')
    #         val_img      = val_config.get('val_img', '')
    #         if val_ann_file and val_img:
    #             val_dataset = VisDroneCocoDataset(
    #                 opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
    #         else:
    #             print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
    #     else:
    #         val_root  = val_config.get('root', dataset_root)
    #         val_paths = val_config.get('val') or val_config.get('test') or []
    #         if val_paths:
    #             val_dataset = Dataset(
    #                 opt=opt, root=val_root, paths=val_paths,
    #                 img_size=opt.input_wh, augment=False,
    #                 transforms=T.Compose([T.ToTensor()]))
    #         else:
    #             print('[warn] val_cfg provided but no val/test paths found.')

    #     if val_dataset is not None:
    #         val_loader = torch.utils.data.DataLoader(
    #             dataset=val_dataset,
    #             batch_size=opt.batch_size,
    #             shuffle=False,
    #             num_workers=opt.num_workers,
    #             pin_memory=True,
    #             drop_last=False,
    #             persistent_workers=opt.num_workers > 0,
    #             prefetch_factor=2 if opt.num_workers > 0 else None,
    #         )
    #         print(f'Val dataset: {len(val_dataset)} images')
    val_loader   = None
    val_ann_file = ''   # used by CocoJsonEvaluator when use_coco_fmt
    val_img_root = ''   # used by run_track_eval (tracking validation)

    if getattr(opt, 'merge_val_into_train', False):
        print('[data] merge_val_into_train=True -> bỏ qua val_loader, không chạy COCO eval.')
    elif getattr(opt, 'val_cfg', ''):
        with open(opt.val_cfg) as f:
            val_config = json.load(f)
        val_dataset = None

        if use_coco_fmt:
            val_ann_file = val_config.get('val_ann', '')
            val_img      = val_config.get('val_img', '')
            val_img_root = val_img
            if val_ann_file and val_img:
                val_dataset = VisDroneCocoDataset(
                    opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
            else:
                print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
        else:
            val_root  = val_config.get('root', dataset_root)
            val_paths = val_config.get('val') or val_config.get('test') or []
            if val_paths:
                val_dataset = Dataset(
                    opt=opt, root=val_root, paths=val_paths,
                    img_size=opt.input_wh, augment=False,
                    transforms=T.Compose([T.ToTensor()]))
            else:
                print('[warn] val_cfg provided but no val/test paths found.')

        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=opt.num_workers > 0,
                prefetch_factor=2 if opt.num_workers > 0 else None,
            )
            print(f'Val dataset: {len(val_dataset)} images')

    print('Creating model...')
    model = create_model(opt.arch, opt)
    _summarize_model(model, opt)

    # ── Load stage-1 weights (weights only) BEFORE applying the freeze policy ──
    start_epoch = 0
    if opt.load_model != '':
        loaded = load_model(model, opt.load_model, None,
                            opt.resume, opt.lr, opt.lr_step)
        # load_model returns just `model` when optimizer is None
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if opt.resume:
            try:
                ckpt = torch.load(opt.load_model, map_location='cpu', weights_only=False)
                start_epoch = int(ckpt.get('epoch', 0))
            except Exception:
                start_epoch = 0

    # ── Training stage policy ─────────────────────────────────────────────────
    #   Stage-1 (--train_single_det): full detector, no ReID head/loss
    #   Stage-2 (default): Phase 0 ReID warmup → Phase 1 joint fine-tune
    det_only  = getattr(opt, 'train_single_det', False)
    warmup_ep = max(0, getattr(opt, 'reid_warmup_epochs', 0))
    p0_lr     = opt.lr if getattr(opt, 'reid_warmup_lr', -1) <= 0 else opt.reid_warmup_lr

    in_phase1 = det_only or start_epoch >= warmup_ep
    if det_only:
        stage_mgr.apply_det_only(model)
        init_lr = opt.lr
    elif in_phase1:
        stage_mgr.apply_phase1(model,
                              keep_backbone_frozen=False,
                              freeze_norm=False)
        init_lr = opt.lr
    else:
        stage_mgr.apply_phase0(model)
        init_lr = p0_lr

    # Optimizer for the current phase. Trainer.__init__ will add criterion params.
    optimizer = build_optimizer(model, _with_lr(opt, init_lr))

    _nw = opt.num_workers
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=_nw > 0,
        prefetch_factor=2 if _nw > 0 else None,
    )
    steps_per_epoch = len(train_loader)

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    # Per-phase scheduler (spans only the current phase's epochs)
    if in_phase1:
        phase_epochs = max(1, opt.num_epochs - warmup_ep)
    else:
        phase_epochs = warmup_ep if warmup_ep > 0 else opt.num_epochs
    scheduler = stage_mgr.build_phase_scheduler(
        optimizer, opt, build_scheduler, steps_per_epoch, phase_epochs,
        warmup_iters=min(getattr(opt, 'warmup_iters', 2000), steps_per_epoch))
    # fast-forward scheduler if resuming mid-phase
    if in_phase1 and start_epoch > warmup_ep:
        scheduler.fast_forward((start_epoch - warmup_ep) * steps_per_epoch)
    elif (not in_phase1) and start_epoch > 0:
        scheduler.fast_forward(start_epoch * steps_per_epoch)

    best_mAP = 0.0
    best_track = 0.0

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        # ── Phase 0 -> Phase 1 transition (stage-2 only) ─────────────────────
        if (not det_only) and (not in_phase1) and epoch > warmup_ep:
            in_phase1 = True
            stage_mgr.apply_phase1(model,
                                   keep_backbone_frozen=getattr(opt, 'keep_backbone_frozen', True),
                                   freeze_norm=getattr(opt, 'freeze_norm_stats', True))
            # set of trainable params changed -> rebuild optimizer (+criterion params)
            optimizer = stage_mgr.build_phase_optimizer(
                model, trainer.loss, opt, build_optimizer, lr=opt.lr)
            trainer.optimizer = optimizer
            trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
            scheduler = stage_mgr.build_phase_scheduler(
                optimizer, opt, build_scheduler, steps_per_epoch,
                max(1, opt.num_epochs - warmup_ep),
                warmup_iters=min(getattr(opt, 'warmup_iters', 2000), steps_per_epoch))
            print(f'[stage] >>> switched to Phase 1 (joint) at epoch {epoch}')

        # ── id_weight schedule (stage-2 only) ────────────────────────────────
        if det_only:
            trainer.loss.id_weight = 0.0
        elif in_phase1:
            ep_in_p1 = epoch - warmup_ep
            stage_mgr.ramp_id_weight(trainer.loss, opt.id_weight,
                                     ep_in_p1, getattr(opt, 'id_warmup_epochs', 0))
        else:
            # Phase 0: detector is frozen, let ReID learn at full strength
            trainer.loss.id_weight = opt.id_weight
        if not det_only:
            logger.write('id_w {:.3f} | '.format(trainer.loss.id_weight))

        # notify dataset of current epoch (0-indexed, matching EdgeCrafter convention)
        train_loader.dataset.set_epoch(epoch - 1)

        log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

        cur_lr = optimizer.param_groups[0]['lr']
        logger.write('epoch: {} |'.format(epoch))
        logger.write('lr {:e} | '.format(cur_lr))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        # ── Periodic checkpoint ────────────────────────────────────────
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
                       epoch, model, optimizer)

        # ── COCO mAP evaluation — CHỈ chạy ở STAGE 1 (detection-only) ────────
        #   Stage 1 không train ReID nên tracking metric vô nghĩa -> chọn
        #   model_best theo detection AP. Stage 2 KHÔNG chạy mAP eval (bỏ qua
        #   hoàn toàn) và chọn model_best theo tracking IDF1/MOTA bên dưới.
        if det_only and val_loader is not None and opt.val_intervals > 0 \
                and epoch % opt.val_intervals == 0:
            print(f'\n[Eval] epoch {epoch} — running COCO mAP (stage-1)...')
            metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)

            log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
            print(f'[Eval] {log_line}')
            logger.write(f'[eval] {log_line} | ')

            for k, v in metrics.items():
                logger.scalar_summary(f'val_{k}', v, epoch)

            cur_mAP = metrics.get('AP', 0.0)   # evaluators return 'AP', not 'mAP'
            if cur_mAP > best_mAP:
                best_mAP = cur_mAP
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model, optimizer)
                print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

        # ── Tracking metric evaluation (IDF1/MOTA) — CHỈ chạy ở STAGE 2 ───────
        #   Chọn model_best theo track_score. Bật bằng --track_val.
        #   Mặc định bỏ qua Phase 0 (detector đóng băng) trừ --track_val_in_phase0.
        _tvi = getattr(opt, 'track_val_intervals', 0) or opt.val_intervals
        do_track = (
            (not det_only)                       # CHỈ Stage 2
            and getattr(opt, 'track_val', False)
            and val_ann_file and val_img_root
            and opt.val_intervals > 0
            and epoch % max(1, _tvi) == 0
        )
        skip_p0 = (not in_phase1) and not getattr(opt, 'track_val_in_phase0', False)
        if do_track and not skip_p0:
            print(f'\n[Track-Eval] epoch {epoch} — running tracking IDF1/MOTA '
                  f'({"fp16" if (getattr(opt,"track_val_fp16",1) and opt.device.type=="cuda") else "fp32"}, GMC off)...')
            tmetrics = run_track_eval(model, opt, val_ann_file, val_img_root)
            if tmetrics:
                tline = (f"IDF1 {tmetrics['idf1']:.4f}  MOTA {tmetrics['mota']:.4f}  "
                         f"IDsw {tmetrics['num_switches']}  "
                         f"score {tmetrics['track_score']:.4f}  "
                         f"({tmetrics['fps']:.1f} fps)")
                print(f'[Track-Eval] {tline}')
                logger.write(f'[track] {tline} | ')
                logger.scalar_summary('val_idf1', tmetrics['idf1'], epoch)
                logger.scalar_summary('val_mota', tmetrics['mota'], epoch)
                logger.scalar_summary('val_idsw', tmetrics['num_switches'], epoch)
                logger.scalar_summary('val_track_score', tmetrics['track_score'], epoch)

                if tmetrics['track_score'] > best_track:
                    best_track = tmetrics['track_score']
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                               epoch, model, optimizer)
                    print(f"[Track-Eval] ★ New best track_score={best_track:.4f} "
                          f"(IDF1={tmetrics['idf1']:.4f}, MOTA={tmetrics['mota']:.4f}) "
                          f"→ model_best.pth")

        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print("opt.gpus: ", opt.gpus)
    print('epoch:', opt.num_epochs)
    run(opt)