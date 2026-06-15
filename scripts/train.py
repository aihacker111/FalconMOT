# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import math
# import os
# import json

# import numpy as np
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# import torch
# import torch.utils.data
# from torchvision.transforms import transforms as T

# import _paths  # noqa: F401  (sys.path bootstrap)
# from falconmot.opts import opts
# from falconmot.models.model import create_model, load_model, save_model
# from falconmot.models.data_parallel import DataParallel
# from falconmot.logger import Logger
# from falconmot.datasets.dataset_factory import get_dataset
# from falconmot.engine.train_factory import train_factory
# from falconmot.utils.jde_eval import CocoJsonEvaluator, JDECocoEvaluator
# from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor


# _NORM_TYPES = (
#     torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
#     torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm2d,
# )


# def build_optimizer(model, opt):
#     """
#     AdamW with 4 param groups:
#       - backbone non-norm : lr * 0.05, weight_decay = default
#       - backbone norm/bias: lr * 0.05, weight_decay = 0
#       - other   norm/bias : lr,        weight_decay = 0
#       - everything else   : lr,        weight_decay = default

#     Norm layers are detected by module type (not just param name) so that
#     BN gamma params inside anonymous nn.Sequential blocks (e.g.
#     backbone.sta.stem.1.weight) are correctly assigned weight_decay=0.
#     """
#     base_lr      = opt.lr
#     backbone_lr  = base_lr * 0.05
#     weight_decay = opt.weight_decay

#     # Collect FQNs of all params that belong to a normalization module.
#     norm_param_names: set = set()
#     for mod_name, module in model.named_modules():
#         if isinstance(module, _NORM_TYPES):
#             for p_name, _ in module.named_parameters(recurse=False):
#                 norm_param_names.add(f'{mod_name}.{p_name}')

#     backbone_wd, backbone_no_wd, other_no_wd, default = [], [], [], []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         is_backbone = 'backbone' in name
#         is_no_wd    = name in norm_param_names or name.endswith('.bias')
#         if is_backbone and is_no_wd:
#             backbone_no_wd.append(param)
#         elif is_backbone:
#             backbone_wd.append(param)
#         elif is_no_wd:
#             other_no_wd.append(param)
#         else:
#             default.append(param)

#     param_groups = [
#         pg for pg in [
#             {'params': backbone_wd,    'lr': backbone_lr},
#             {'params': backbone_no_wd, 'lr': backbone_lr, 'weight_decay': 0.},
#             {'params': other_no_wd,                        'weight_decay': 0.},
#             {'params': default},
#         ]
#         if pg['params']
#     ]

#     optimizer = torch.optim.AdamW(
#         param_groups,
#         lr=base_lr,
#         betas=(0.9, 0.999),
#         weight_decay=weight_decay,
#     )
#     return optimizer


# def _flat_cosine_lr(total_iter, warmup_iter, flat_iter, no_aug_iter,
#                     current_iter, init_lr, min_lr):
#     """Quadratic warmup → flat → cosine decay → no-aug. Ported from EdgeCrafter."""
#     if current_iter <= warmup_iter:
#         return init_lr * (current_iter / float(max(1, warmup_iter))) ** 2
#     if current_iter <= flat_iter:
#         return init_lr
#     if current_iter >= total_iter - no_aug_iter:
#         return min_lr
#     cosine = 0.5 * (1.0 + math.cos(
#         math.pi * (current_iter - flat_iter) /
#         float(max(1, total_iter - flat_iter - no_aug_iter))
#     ))
#     return min_lr + (init_lr - min_lr) * cosine


# class FlatCosineScheduler:
#     """
#     Flat-cosine LR scheduler compatible with base_trainer's scheduler.step() API.

#     Phases (per iteration):
#       1. Quadratic warmup  : 0 → init_lr  over warmup_iter steps
#       2. Flat              : init_lr constant for flat_epoch epochs
#       3. Cosine decay      : init_lr → min_lr
#       4. No-aug            : min_lr constant for no_aug_epochs final epochs

#     flat_epochs  — epochs where LR stays at init_lr (use stop_epoch or num_epochs//2)
#     no_aug_epochs — final constant-LR epochs (EdgeCrafter default: 2)
#     lr_gamma      — min_lr = init_lr × lr_gamma  (default 0.01)
#     """

#     def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
#                  warmup_iter: int, flat_epochs: int, no_aug_epochs: int,
#                  lr_gamma: float):
#         # Snapshot initial LR for each param group
#         for group in optimizer.param_groups:
#             group['initial_lr'] = group['lr']

#         self._optimizer   = optimizer
#         self._base_lrs    = [g['initial_lr'] for g in optimizer.param_groups]
#         self._min_lrs     = [lr * lr_gamma   for lr in self._base_lrs]
#         self._cur_iter    = 0

#         total_iter   = num_epochs    * steps_per_epoch
#         flat_iter    = flat_epochs   * steps_per_epoch
#         no_aug_iter  = no_aug_epochs * steps_per_epoch

#         self._total_iter  = total_iter
#         self._warmup_iter = warmup_iter
#         self._flat_iter   = flat_iter
#         self._no_aug_iter = no_aug_iter

#         print(
#             f'[FlatCosineScheduler] total={total_iter} warmup={warmup_iter} '
#             f'flat={flat_iter} no_aug={no_aug_iter} '
#             f'base_lrs={self._base_lrs} min_lrs={self._min_lrs}'
#         )

#     def step(self):
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = _flat_cosine_lr(
#                 self._total_iter, self._warmup_iter,
#                 self._flat_iter,  self._no_aug_iter,
#                 self._cur_iter,
#                 self._base_lrs[i], self._min_lrs[i],
#             )
#         self._cur_iter += 1

#     def fast_forward(self, n_steps: int):
#         """Jump scheduler state by n_steps without looping (O(1) instead of O(n))."""
#         self._cur_iter = n_steps
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = _flat_cosine_lr(
#                 self._total_iter, self._warmup_iter,
#                 self._flat_iter,  self._no_aug_iter,
#                 self._cur_iter,
#                 self._base_lrs[i], self._min_lrs[i],
#             )

#     def state_dict(self):
#         return {'cur_iter': self._cur_iter}

#     def load_state_dict(self, state: dict):
#         self._cur_iter = state.get('cur_iter', 0)


# class RFDETRScheduler:
#     """
#     LR scheduler ported from RF-DETR: linear warmup + cosine annealing or step decay.

#     Phases (per step):
#       1. Linear warmup : 0 → base_lr  over warmup_steps
#       2a. Cosine       : lr_min_factor + (1-lr_min_factor)*0.5*(1+cos(π*progress))
#       2b. Step         : base_lr until lr_drop epochs, then base_lr * 0.1

#     Activated via --lr_scheduler cosine|step. Params from opt:
#       warmup_epochs  (float, default 0.0)
#       lr_drop        (int,   default = num_epochs)
#       lr_min_factor  (float, default 0.0)
#     """

#     def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
#                  warmup_epochs: float, lr_drop: int, lr_scheduler: str,
#                  lr_min_factor: float):
#         for group in optimizer.param_groups:
#             group['initial_lr'] = group['lr']

#         self._optimizer     = optimizer
#         self._base_lrs      = [g['initial_lr'] for g in optimizer.param_groups]
#         self._cur_step      = 0
#         self._total_steps   = num_epochs * steps_per_epoch
#         self._warmup_steps  = int(steps_per_epoch * warmup_epochs)
#         self._lr_drop_steps = lr_drop * steps_per_epoch
#         self._lr_scheduler  = lr_scheduler
#         self._lr_min_factor = lr_min_factor

#         print(
#             f'[RFDETRScheduler] type={lr_scheduler} total={self._total_steps} '
#             f'warmup={self._warmup_steps} lr_drop_step={self._lr_drop_steps} '
#             f'lr_min_factor={lr_min_factor} base_lrs={self._base_lrs}'
#         )

#     def _compute_lr(self, step: int, base_lr: float) -> float:
#         if step < self._warmup_steps:
#             return base_lr * float(step) / float(max(1, self._warmup_steps))
#         if self._lr_scheduler == 'cosine':
#             progress = float(step - self._warmup_steps) / float(
#                 max(1, self._total_steps - self._warmup_steps))
#             factor = self._lr_min_factor + (1.0 - self._lr_min_factor) * 0.5 * (
#                 1.0 + math.cos(math.pi * progress))
#             return base_lr * factor
#         return base_lr if step < self._lr_drop_steps else base_lr * 0.1

#     def step(self):
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])
#         self._cur_step += 1

#     def fast_forward(self, n_steps: int):
#         self._cur_step = n_steps
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])

#     def state_dict(self):
#         return {'cur_step': self._cur_step}

#     def load_state_dict(self, state: dict):
#         self._cur_step = state.get('cur_step', 0)


# def build_scheduler(optimizer, opt, steps_per_epoch: int):
#     """
#     Dispatch on opt.lr_scheduler:
#       'cosine' | 'step'  → RFDETRScheduler (linear warmup + cosine/step)
#       anything else      → FlatCosineScheduler (quadratic warmup + flat + cosine, legacy)
#     """
#     lr_scheduler = getattr(opt, 'lr_scheduler', 'flat_cosine')

#     if lr_scheduler in ('cosine', 'step'):
#         return RFDETRScheduler(
#             optimizer,
#             steps_per_epoch = steps_per_epoch,
#             num_epochs      = opt.num_epochs,
#             warmup_epochs   = getattr(opt, 'warmup_epochs',  0.0),
#             lr_drop         = getattr(opt, 'lr_drop',        opt.num_epochs),
#             lr_scheduler    = lr_scheduler,
#             lr_min_factor   = getattr(opt, 'lr_min_factor',  0.0),
#         )

#     # legacy flat-cosine
#     warmup_iter   = getattr(opt, 'warmup_iters',  2000)
#     num_epochs    = opt.num_epochs
#     no_aug_epochs = getattr(opt, 'no_aug_epochs', 2)
#     lr_gamma      = getattr(opt, 'lr_gamma',      0.5)
#     stop_epoch    = getattr(opt, 'stop_epoch',    -1)
#     flat_epochs   = stop_epoch if stop_epoch > 0 else num_epochs // 2

#     return FlatCosineScheduler(
#         optimizer, steps_per_epoch, num_epochs,
#         warmup_iter, flat_epochs, no_aug_epochs, lr_gamma,
#     )


# @torch.no_grad()
# def run_coco_eval(model, val_loader, opt, ann_file: str = '') -> dict:
#     """
#     COCO mAP evaluation — DEIMv2-compatible.

#     ann_file set  → CocoJsonEvaluator: GT từ COCO JSON, dùng real image_id (COCO format)
#     ann_file ''   → JDECocoEvaluator:  GT rebuild từ batch với letterbox inverse (JDE format)
#     """
#     model.eval()

#     net_h, net_w = opt.input_wh[1], opt.input_wh[0]

#     postprocessor = FalconJDEPostProcessor(
#         num_classes=opt.num_classes,
#         num_top_queries=500,
#         use_focal_loss=True,
#         conf_thres=0.0
#     )
#     # Plain resize: KHÔNG set_net_hw -> postprocessor dùng nhánh norm*orig (đúng cho resize thẳng)

#     if ann_file:
#         evaluator = CocoJsonEvaluator(ann_file)
#     else:
#         evaluator = JDECocoEvaluator(
#             num_classes=opt.num_classes,
#             net_h=net_h,
#             net_w=net_w,
#         )

#     max_batches = getattr(opt, 'debug_val_batches', 0)

#     for i, batch in enumerate(val_loader):
#         if max_batches > 0 and i >= max_batches:
#             break
#         batch = {k: v.to(opt.device, non_blocking=True)
#                  if isinstance(v, torch.Tensor) else v
#                  for k, v in batch.items()}

#         orig_hw    = batch.get('orig_hw')
#         orig_sizes = orig_hw if orig_hw is not None else \
#                      torch.tensor([[net_h, net_w]] * batch['input'].shape[0],
#                                   device=opt.device)

#         outputs    = model(batch['input'])
#         dt_results = postprocessor(outputs, orig_sizes)   # plain resize -> norm*orig
#         evaluator.update(dt_results, batch)

#     model.train()
#     return evaluator.summarize()


# def run(opt):
#     torch.manual_seed(opt.seed)
#     torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

#     print('Setting up data...')
#     Dataset      = get_dataset(opt.dataset, opt.task)
#     use_coco_fmt = (opt.dataset == 'coco')

#     f = open(opt.data_cfg)
#     data_config  = json.load(f)
#     dataset_root = data_config['root']
#     print("Dataset root: %s" % dataset_root)
#     f.close()

#     from falconmot.datasets.dataset.coco_detection import VisDroneCocoDataset

#     if use_coco_fmt:
#         train_ann = data_config['train_ann']
#         train_img = data_config['train_img']
#         dataset   = VisDroneCocoDataset(
#             opt=opt, img_root=train_img, ann_file=train_ann, augment=True)
#     else:
#         trainset_paths = data_config['train']
#         transforms     = T.Compose([T.ToTensor()])
#         dataset        = Dataset(opt=opt, root=dataset_root,
#                                  paths=trainset_paths, img_size=opt.input_wh,
#                                  augment=True, transforms=transforms)

#     opt = opts().update_dataset_info_and_set_heads(opt, dataset)
#     print("opt:\n", opt)
#     logger = Logger(opt)

#     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     print("opt.gpus_str: ", opt.gpus_str)
#     opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

#     # ── Val dataset (optional) ──────────────────────────────────────────
#     val_loader   = None
#     val_ann_file = ''   # used by CocoJsonEvaluator when use_coco_fmt

#     if getattr(opt, 'val_cfg', ''):
#         with open(opt.val_cfg) as f:
#             val_config = json.load(f)
#         val_dataset = None

#         if use_coco_fmt:
#             val_ann_file = val_config.get('val_ann', '')
#             val_img      = val_config.get('val_img', '')
#             if val_ann_file and val_img:
#                 val_dataset = VisDroneCocoDataset(
#                     opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
#             else:
#                 print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
#         else:
#             val_root  = val_config.get('root', dataset_root)
#             val_paths = val_config.get('val') or val_config.get('test') or []
#             if val_paths:
#                 val_dataset = Dataset(
#                     opt=opt, root=val_root, paths=val_paths,
#                     img_size=opt.input_wh, augment=False,
#                     transforms=T.Compose([T.ToTensor()]))
#             else:
#                 print('[warn] val_cfg provided but no val/test paths found.')

#         if val_dataset is not None:
#             val_loader = torch.utils.data.DataLoader(
#                 dataset=val_dataset,
#                 batch_size=opt.batch_size,
#                 shuffle=False,
#                 num_workers=opt.num_workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 persistent_workers=opt.num_workers > 0,
#                 prefetch_factor=2 if opt.num_workers > 0 else None,
#             )
#             print(f'Val dataset: {len(val_dataset)} images')

#     print('Creating model...')
#     model = create_model(opt.arch, opt)

#     optimizer = build_optimizer(model, opt)

#     start_epoch = 0
#     if opt.load_model != '':
#         model, optimizer, start_epoch = load_model(
#             model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
#         )

#     _nw = opt.num_workers
#     train_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=_nw,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=_nw > 0,
#         prefetch_factor=2 if _nw > 0 else None,
#     )

#     print('Starting training...')
#     Trainer = train_factory[opt.task]
#     trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
#     trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

#     scheduler = build_scheduler(optimizer, opt, steps_per_epoch=len(train_loader))
#     # fast-forward scheduler state if resuming mid-training (O(1), not O(n))
#     if start_epoch > 0:
#         scheduler.fast_forward(start_epoch * len(train_loader))

#     best_mAP = 0.0

#     for epoch in range(start_epoch + 1, opt.num_epochs + 1):
#         mark = epoch if opt.save_all else 'last'

#         # notify dataset of current epoch (0-indexed, matching EdgeCrafter convention)
#         train_loader.dataset.set_epoch(epoch - 1)

#         log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

#         cur_lr = optimizer.param_groups[0]['lr']
#         logger.write('epoch: {} |'.format(epoch))
#         logger.write('lr {:e} | '.format(cur_lr))
#         for k, v in log_dict_train.items():
#             logger.scalar_summary('train_{}'.format(k), v, epoch)
#             logger.write('{} {:8f} | '.format(k, v))

#         # ── Periodic checkpoint ────────────────────────────────────────
#         if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
#                        epoch, model, optimizer)
#         else:
#             save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
#                        epoch, model, optimizer)

#         # ── COCO mAP evaluation ────────────────────────────────────────
#         if val_loader is not None and opt.val_intervals > 0 \
#                 and epoch % opt.val_intervals == 0:
#             print(f'\n[Eval] epoch {epoch} — running COCO mAP...')
#             metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)

#             log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
#             print(f'[Eval] {log_line}')
#             logger.write(f'[eval] {log_line} | ')

#             for k, v in metrics.items():
#                 logger.scalar_summary(f'val_{k}', v, epoch)

#             cur_mAP = metrics.get('AP', 0.0)   # evaluators return 'AP', not 'mAP'
#             if cur_mAP > best_mAP:
#                 best_mAP = cur_mAP
#                 save_model(os.path.join(opt.save_dir, 'model_best.pth'),
#                            epoch, model, optimizer)
#                 print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

#         logger.write('\n')

#         if epoch in opt.lr_step:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#         if epoch % 5 == 0 or epoch >= 25:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#     logger.close()


# if __name__ == '__main__':
#     opt = opts().parse()
#     print("opt.gpus: ", opt.gpus)
#     print('epoch:', opt.num_epochs)
#     run(opt)




# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import math
# import os
# import json

# import numpy as np
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# import torch
# import torch.utils.data
# from torchvision.transforms import transforms as T

# import _paths  # noqa: F401  (sys.path bootstrap)
# from falconmot.opts import opts
# from falconmot.models.model import create_model, load_model, save_model
# from falconmot.models.data_parallel import DataParallel
# from falconmot.logger import Logger
# from falconmot.datasets.dataset_factory import get_dataset
# from falconmot.engine.train_factory import train_factory
# from falconmot.utils.jde_eval import CocoJsonEvaluator, JDECocoEvaluator
# from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor


# _NORM_TYPES = (
#     torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
#     torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm2d,
# )


# def build_optimizer(model, opt):
#     """
#     AdamW with 4 param groups:
#       - backbone non-norm : lr * 0.05, weight_decay = default
#       - backbone norm/bias: lr * 0.05, weight_decay = 0
#       - other   norm/bias : lr,        weight_decay = 0
#       - everything else   : lr,        weight_decay = default

#     Norm layers are detected by module type (not just param name) so that
#     BN gamma params inside anonymous nn.Sequential blocks (e.g.
#     backbone.sta.stem.1.weight) are correctly assigned weight_decay=0.
#     """
#     base_lr      = opt.lr
#     backbone_lr  = base_lr * 0.05
#     weight_decay = opt.weight_decay

#     # Collect FQNs of all params that belong to a normalization module.
#     norm_param_names: set = set()
#     for mod_name, module in model.named_modules():
#         if isinstance(module, _NORM_TYPES):
#             for p_name, _ in module.named_parameters(recurse=False):
#                 norm_param_names.add(f'{mod_name}.{p_name}')

#     backbone_wd, backbone_no_wd, other_no_wd, default = [], [], [], []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         is_backbone = 'backbone' in name
#         is_no_wd    = name in norm_param_names or name.endswith('.bias')
#         if is_backbone and is_no_wd:
#             backbone_no_wd.append(param)
#         elif is_backbone:
#             backbone_wd.append(param)
#         elif is_no_wd:
#             other_no_wd.append(param)
#         else:
#             default.append(param)

#     param_groups = [
#         pg for pg in [
#             {'params': backbone_wd,    'lr': backbone_lr},
#             {'params': backbone_no_wd, 'lr': backbone_lr, 'weight_decay': 0.},
#             {'params': other_no_wd,                        'weight_decay': 0.},
#             {'params': default},
#         ]
#         if pg['params']
#     ]

#     optimizer = torch.optim.AdamW(
#         param_groups,
#         lr=base_lr,
#         betas=(0.9, 0.999),
#         weight_decay=weight_decay,
#     )
#     return optimizer


# def _flat_cosine_lr(total_iter, warmup_iter, flat_iter, no_aug_iter,
#                     current_iter, init_lr, min_lr):
#     """Quadratic warmup → flat → cosine decay → no-aug. Ported from EdgeCrafter."""
#     if current_iter <= warmup_iter:
#         return init_lr * (current_iter / float(max(1, warmup_iter))) ** 2
#     if current_iter <= flat_iter:
#         return init_lr
#     if current_iter >= total_iter - no_aug_iter:
#         return min_lr
#     cosine = 0.5 * (1.0 + math.cos(
#         math.pi * (current_iter - flat_iter) /
#         float(max(1, total_iter - flat_iter - no_aug_iter))
#     ))
#     return min_lr + (init_lr - min_lr) * cosine


# class FlatCosineScheduler:
#     """
#     Flat-cosine LR scheduler compatible with base_trainer's scheduler.step() API.

#     Phases (per iteration):
#       1. Quadratic warmup  : 0 → init_lr  over warmup_iter steps
#       2. Flat              : init_lr constant for flat_epoch epochs
#       3. Cosine decay      : init_lr → min_lr
#       4. No-aug            : min_lr constant for no_aug_epochs final epochs

#     flat_epochs  — epochs where LR stays at init_lr (use stop_epoch or num_epochs//2)
#     no_aug_epochs — final constant-LR epochs (EdgeCrafter default: 2)
#     lr_gamma      — min_lr = init_lr × lr_gamma  (default 0.01)
#     """

#     def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
#                  warmup_iter: int, flat_epochs: int, no_aug_epochs: int,
#                  lr_gamma: float):
#         # Snapshot initial LR for each param group
#         for group in optimizer.param_groups:
#             group['initial_lr'] = group['lr']

#         self._optimizer   = optimizer
#         self._base_lrs    = [g['initial_lr'] for g in optimizer.param_groups]
#         self._min_lrs     = [lr * lr_gamma   for lr in self._base_lrs]
#         self._cur_iter    = 0

#         total_iter   = num_epochs    * steps_per_epoch
#         flat_iter    = flat_epochs   * steps_per_epoch
#         no_aug_iter  = no_aug_epochs * steps_per_epoch

#         self._total_iter  = total_iter
#         self._warmup_iter = warmup_iter
#         self._flat_iter   = flat_iter
#         self._no_aug_iter = no_aug_iter

#         print(
#             f'[FlatCosineScheduler] total={total_iter} warmup={warmup_iter} '
#             f'flat={flat_iter} no_aug={no_aug_iter} '
#             f'base_lrs={self._base_lrs} min_lrs={self._min_lrs}'
#         )

#     def step(self):
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = _flat_cosine_lr(
#                 self._total_iter, self._warmup_iter,
#                 self._flat_iter,  self._no_aug_iter,
#                 self._cur_iter,
#                 self._base_lrs[i], self._min_lrs[i],
#             )
#         self._cur_iter += 1

#     def fast_forward(self, n_steps: int):
#         """Jump scheduler state by n_steps without looping (O(1) instead of O(n))."""
#         self._cur_iter = n_steps
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = _flat_cosine_lr(
#                 self._total_iter, self._warmup_iter,
#                 self._flat_iter,  self._no_aug_iter,
#                 self._cur_iter,
#                 self._base_lrs[i], self._min_lrs[i],
#             )

#     def state_dict(self):
#         return {'cur_iter': self._cur_iter}

#     def load_state_dict(self, state: dict):
#         self._cur_iter = state.get('cur_iter', 0)


# class RFDETRScheduler:
#     """
#     LR scheduler ported from RF-DETR: linear warmup + cosine annealing or step decay.

#     Phases (per step):
#       1. Linear warmup : 0 → base_lr  over warmup_steps
#       2a. Cosine       : lr_min_factor + (1-lr_min_factor)*0.5*(1+cos(π*progress))
#       2b. Step         : base_lr until lr_drop epochs, then base_lr * 0.1

#     Activated via --lr_scheduler cosine|step. Params from opt:
#       warmup_epochs  (float, default 0.0)
#       lr_drop        (int,   default = num_epochs)
#       lr_min_factor  (float, default 0.0)
#     """

#     def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
#                  warmup_epochs: float, lr_drop: int, lr_scheduler: str,
#                  lr_min_factor: float):
#         for group in optimizer.param_groups:
#             group['initial_lr'] = group['lr']

#         self._optimizer     = optimizer
#         self._base_lrs      = [g['initial_lr'] for g in optimizer.param_groups]
#         self._cur_step      = 0
#         self._total_steps   = num_epochs * steps_per_epoch
#         self._warmup_steps  = int(steps_per_epoch * warmup_epochs)
#         self._lr_drop_steps = lr_drop * steps_per_epoch
#         self._lr_scheduler  = lr_scheduler
#         self._lr_min_factor = lr_min_factor

#         print(
#             f'[RFDETRScheduler] type={lr_scheduler} total={self._total_steps} '
#             f'warmup={self._warmup_steps} lr_drop_step={self._lr_drop_steps} '
#             f'lr_min_factor={lr_min_factor} base_lrs={self._base_lrs}'
#         )

#     def _compute_lr(self, step: int, base_lr: float) -> float:
#         if step < self._warmup_steps:
#             return base_lr * float(step) / float(max(1, self._warmup_steps))
#         if self._lr_scheduler == 'cosine':
#             progress = float(step - self._warmup_steps) / float(
#                 max(1, self._total_steps - self._warmup_steps))
#             factor = self._lr_min_factor + (1.0 - self._lr_min_factor) * 0.5 * (
#                 1.0 + math.cos(math.pi * progress))
#             return base_lr * factor
#         return base_lr if step < self._lr_drop_steps else base_lr * 0.1

#     def step(self):
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])
#         self._cur_step += 1

#     def fast_forward(self, n_steps: int):
#         self._cur_step = n_steps
#         for i, group in enumerate(self._optimizer.param_groups):
#             group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])

#     def state_dict(self):
#         return {'cur_step': self._cur_step}

#     def load_state_dict(self, state: dict):
#         self._cur_step = state.get('cur_step', 0)


# def build_scheduler(optimizer, opt, steps_per_epoch: int):
#     """
#     Dispatch on opt.lr_scheduler:
#       'cosine' | 'step'  → RFDETRScheduler (linear warmup + cosine/step)
#       anything else      → FlatCosineScheduler (quadratic warmup + flat + cosine, legacy)
#     """
#     lr_scheduler = getattr(opt, 'lr_scheduler', 'flat_cosine')

#     if lr_scheduler in ('cosine', 'step'):
#         return RFDETRScheduler(
#             optimizer,
#             steps_per_epoch = steps_per_epoch,
#             num_epochs      = opt.num_epochs,
#             warmup_epochs   = getattr(opt, 'warmup_epochs',  0.0),
#             lr_drop         = getattr(opt, 'lr_drop',        opt.num_epochs),
#             lr_scheduler    = lr_scheduler,
#             lr_min_factor   = getattr(opt, 'lr_min_factor',  0.0),
#         )

#     # legacy flat-cosine
#     warmup_iter   = getattr(opt, 'warmup_iters',  2000)
#     num_epochs    = opt.num_epochs
#     no_aug_epochs = getattr(opt, 'no_aug_epochs', 2)
#     lr_gamma      = getattr(opt, 'lr_gamma',      0.5)
#     stop_epoch    = getattr(opt, 'stop_epoch',    -1)
#     flat_epochs   = stop_epoch if stop_epoch > 0 else num_epochs // 2

#     return FlatCosineScheduler(
#         optimizer, steps_per_epoch, num_epochs,
#         warmup_iter, flat_epochs, no_aug_epochs, lr_gamma,
#     )


# @torch.no_grad()
# def run_coco_eval(model, val_loader, opt, ann_file: str = '') -> dict:
#     """
#     COCO mAP evaluation — DEIMv2-compatible.

#     ann_file set  → CocoJsonEvaluator: GT từ COCO JSON, dùng real image_id (COCO format)
#     ann_file ''   → JDECocoEvaluator:  GT rebuild từ batch với letterbox inverse (JDE format)
#     """
#     model.eval()

#     net_h, net_w = opt.input_wh[1], opt.input_wh[0]

#     postprocessor = FalconJDEPostProcessor(
#         num_classes=opt.num_classes,
#         num_top_queries=500,
#         use_focal_loss=True,
#         conf_thres=0.0
#     )
#     # Plain resize: KHÔNG set_net_hw -> postprocessor dùng nhánh norm*orig (đúng cho resize thẳng)

#     if ann_file:
#         evaluator = CocoJsonEvaluator(ann_file)
#     else:
#         evaluator = JDECocoEvaluator(
#             num_classes=opt.num_classes,
#             net_h=net_h,
#             net_w=net_w,
#         )

#     max_batches = getattr(opt, 'debug_val_batches', 0)

#     for i, batch in enumerate(val_loader):
#         if max_batches > 0 and i >= max_batches:
#             break
#         batch = {k: v.to(opt.device, non_blocking=True)
#                  if isinstance(v, torch.Tensor) else v
#                  for k, v in batch.items()}

#         orig_hw    = batch.get('orig_hw')
#         orig_sizes = orig_hw if orig_hw is not None else \
#                      torch.tensor([[net_h, net_w]] * batch['input'].shape[0],
#                                   device=opt.device)

#         outputs    = model(batch['input'])
#         dt_results = postprocessor(outputs, orig_sizes)   # plain resize -> norm*orig
#         evaluator.update(dt_results, batch)

#     model.train()
#     return evaluator.summarize()


# def run(opt):
#     torch.manual_seed(opt.seed)
#     torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

#     print('Setting up data...')
#     Dataset      = get_dataset(opt.dataset, opt.task)
#     use_coco_fmt = (opt.dataset == 'coco')

#     f = open(opt.data_cfg)
#     data_config  = json.load(f)
#     dataset_root = data_config['root']
#     print("Dataset root: %s" % dataset_root)
#     f.close()

#     from falconmot.datasets.dataset.coco_detection import VisDroneCocoDataset

#     if use_coco_fmt:
#         train_ann = data_config['train_ann']
#         train_img = data_config['train_img']
#         dataset   = VisDroneCocoDataset(
#             opt=opt, img_root=train_img, ann_file=train_ann, augment=True)
#     else:
#         trainset_paths = data_config['train']
#         transforms     = T.Compose([T.ToTensor()])
#         dataset        = Dataset(opt=opt, root=dataset_root,
#                                  paths=trainset_paths, img_size=opt.input_wh,
#                                  augment=True, transforms=transforms)

#     opt = opts().update_dataset_info_and_set_heads(opt, dataset)
#     print("opt:\n", opt)
#     logger = Logger(opt)

#     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     print("opt.gpus_str: ", opt.gpus_str)
#     opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

#     # ── Val dataset (optional) ──────────────────────────────────────────
#     val_loader   = None
#     val_ann_file = ''   # used by CocoJsonEvaluator when use_coco_fmt

#     if getattr(opt, 'val_cfg', ''):
#         with open(opt.val_cfg) as f:
#             val_config = json.load(f)
#         val_dataset = None

#         if use_coco_fmt:
#             val_ann_file = val_config.get('val_ann', '')
#             val_img      = val_config.get('val_img', '')
#             if val_ann_file and val_img:
#                 val_dataset = VisDroneCocoDataset(
#                     opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
#             else:
#                 print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
#         else:
#             val_root  = val_config.get('root', dataset_root)
#             val_paths = val_config.get('val') or val_config.get('test') or []
#             if val_paths:
#                 val_dataset = Dataset(
#                     opt=opt, root=val_root, paths=val_paths,
#                     img_size=opt.input_wh, augment=False,
#                     transforms=T.Compose([T.ToTensor()]))
#             else:
#                 print('[warn] val_cfg provided but no val/test paths found.')

#         if val_dataset is not None:
#             val_loader = torch.utils.data.DataLoader(
#                 dataset=val_dataset,
#                 batch_size=opt.batch_size,
#                 shuffle=False,
#                 num_workers=opt.num_workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 persistent_workers=opt.num_workers > 0,
#                 prefetch_factor=2 if opt.num_workers > 0 else None,
#             )
#             print(f'Val dataset: {len(val_dataset)} images')

#     print('Creating model...')
#     model = create_model(opt.arch, opt)

#     optimizer = build_optimizer(model, opt)

#     start_epoch = 0
#     if opt.load_model != '':
#         model, optimizer, start_epoch = load_model(
#             model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
#         )

#     _nw = opt.num_workers
#     train_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=_nw,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=_nw > 0,
#         prefetch_factor=2 if _nw > 0 else None,
#     )

#     print('Starting training...')
#     Trainer = train_factory[opt.task]
#     trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
#     trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

#     scheduler = build_scheduler(optimizer, opt, steps_per_epoch=len(train_loader))
#     # fast-forward scheduler state if resuming mid-training (O(1), not O(n))
#     if start_epoch > 0:
#         scheduler.fast_forward(start_epoch * len(train_loader))

#     best_mAP = 0.0

#     for epoch in range(start_epoch + 1, opt.num_epochs + 1):
#         mark = epoch if opt.save_all else 'last'

#         # notify dataset of current epoch (0-indexed, matching EdgeCrafter convention)
#         train_loader.dataset.set_epoch(epoch - 1)

#         log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

#         cur_lr = optimizer.param_groups[0]['lr']
#         logger.write('epoch: {} |'.format(epoch))
#         logger.write('lr {:e} | '.format(cur_lr))
#         for k, v in log_dict_train.items():
#             logger.scalar_summary('train_{}'.format(k), v, epoch)
#             logger.write('{} {:8f} | '.format(k, v))

#         # ── Periodic checkpoint ────────────────────────────────────────
#         if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
#                        epoch, model, optimizer)
#         else:
#             save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
#                        epoch, model, optimizer)

#         # ── COCO mAP evaluation ────────────────────────────────────────
#         if val_loader is not None and opt.val_intervals > 0 \
#                 and epoch % opt.val_intervals == 0:
#             print(f'\n[Eval] epoch {epoch} — running COCO mAP...')
#             metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)

#             log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
#             print(f'[Eval] {log_line}')
#             logger.write(f'[eval] {log_line} | ')

#             for k, v in metrics.items():
#                 logger.scalar_summary(f'val_{k}', v, epoch)

#             cur_mAP = metrics.get('AP', 0.0)   # evaluators return 'AP', not 'mAP'
#             if cur_mAP > best_mAP:
#                 best_mAP = cur_mAP
#                 save_model(os.path.join(opt.save_dir, 'model_best.pth'),
#                            epoch, model, optimizer)
#                 print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

#         logger.write('\n')

#         if epoch in opt.lr_step:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#         if epoch % 5 == 0 or epoch >= 25:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#     logger.close()


# if __name__ == '__main__':
#     opt = opts().parse()
#     print("opt.gpus: ", opt.gpus)
#     print('epoch:', opt.num_epochs)
#     run(opt)




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from falconmot.utils.jde_eval import CocoJsonEvaluator, JDECocoEvaluator
from falconmot.models.falcon_jde.postprocessor import FalconJDEPostProcessor


_NORM_TYPES = (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm2d,
)

# =============================================================================
# STAGE-2 FREEZE SCHEDULE
# =============================================================================
#
# Chiến lược 3 giai đoạn:
#
#  Epochs [1 .. freeze_backbone_epochs]:
#    - Freeze hoàn toàn backbone (dinov3 + sta)
#    - Chỉ train: backbone.convs/norms, encoder, decoder, reid_head, s4_branch
#    - LR base = ft_lr (nhỏ, ví dụ 5e-5)
#
#  Epochs [freeze_backbone_epochs+1 .. freeze_backbone_epochs+unfreeze_upper_epochs]:
#    - Unfreeze 1/3 cuối backbone ViT (transformer blocks cuối) + sta
#    - Vẫn freeze patch_embed và các block đầu của ViT
#    - LR backbone = ft_lr * 0.1 (rất nhỏ để không phá pretrained weights)
#
#  Epochs [freeze_backbone_epochs+unfreeze_upper_epochs+1 .. num_epochs]:
#    - Unfreeze toàn bộ backbone
#    - LR backbone = ft_lr * 0.05 (theo công thức build_optimizer gốc)
#
# Khuyến nghị hyper-params cho stage 2:
#   --lr 5e-5                        (thay vì 5e-4)
#   --freeze_backbone_epochs 5       (5 epoch đầu chỉ train head)
#   --unfreeze_upper_epochs 5        (tiếp 5 epoch mở block cuối ViT)
#   --num_epochs 20
#   --lr_scheduler cosine
#   --warmup_epochs 0.5
#   --lr_min_factor 0.1
# =============================================================================


def _get_vit_blocks(backbone):
    """Trả về list các transformer block của ViT (dinov3 hoặc VisionTransformer)."""
    dinov3 = backbone.dinov3
    # DinoVisionTransformer dùng .blocks, VisionTransformer dùng ._model.blocks
    if hasattr(dinov3, 'blocks'):
        return list(dinov3.blocks)
    if hasattr(dinov3, '_model') and hasattr(dinov3._model, 'blocks'):
        return list(dinov3._model.blocks)
    return []


def freeze_backbone_fully(model):
    """Freeze toàn bộ backbone (dinov3 + sta). convs/norms vẫn train."""
    backbone = model.backbone
    # Freeze ViT
    for p in backbone.dinov3.parameters():
        p.requires_grad_(False)
    # Freeze SpatialPriorModule (sta)
    if hasattr(backbone, 'sta'):
        for p in backbone.sta.parameters():
            p.requires_grad_(False)
    # backbone.convs và backbone.norms vẫn được train (requires_grad=True mặc định)
    n_frozen = sum(1 for p in backbone.parameters() if not p.requires_grad)
    print(f'[Stage2] Backbone FULLY FROZEN — {n_frozen} params frozen '
          f'(convs/norms still trainable)')


def unfreeze_backbone_upper(model, unfreeze_ratio: float = 0.34):
    """
    Unfreeze phần trên của ViT (unfreeze_ratio cuối) + toàn bộ sta.
    Giữ freeze patch_embed và các block đầu.
    """
    backbone = model.backbone
    blocks = _get_vit_blocks(backbone)
    n_total = len(blocks)
    n_keep_frozen = int(n_total * (1.0 - unfreeze_ratio))

    # Unfreeze từ block n_keep_frozen trở đi
    for i, blk in enumerate(blocks):
        if i >= n_keep_frozen:
            for p in blk.parameters():
                p.requires_grad_(True)

    # Unfreeze sta (spatial prior module)
    if hasattr(backbone, 'sta'):
        for p in backbone.sta.parameters():
            p.requires_grad_(True)

    n_frozen  = sum(1 for p in backbone.parameters() if not p.requires_grad)
    n_train   = sum(1 for p in backbone.parameters() if p.requires_grad)
    print(f'[Stage2] Backbone UPPER UNFROZEN (top {int(unfreeze_ratio*100)}% blocks + sta) '
          f'— frozen={n_frozen}, trainable={n_train}')


def unfreeze_backbone_all(model):
    """Unfreeze hoàn toàn backbone."""
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    n_train = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f'[Stage2] Backbone FULLY UNFROZEN — {n_train} trainable params')


def build_optimizer_stage2(model, opt):
    """
    Optimizer cho stage 2 với 4 param groups:
      - backbone non-norm : backbone_lr = base_lr * backbone_lr_ratio
      - backbone norm/bias: backbone_lr, weight_decay=0
      - other   norm/bias : base_lr, weight_decay=0
      - everything else   : base_lr
    Chỉ include params có requires_grad=True (quan trọng khi có freeze).
    """
    base_lr            = opt.lr
    backbone_lr_ratio  = getattr(opt, 'backbone_lr_ratio', 0.05)
    backbone_lr        = base_lr * backbone_lr_ratio
    weight_decay       = opt.weight_decay

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

    n_backbone = len(backbone_wd) + len(backbone_no_wd)
    n_head     = len(other_no_wd) + len(default)
    print(f'[Stage2] Optimizer built — backbone params={n_backbone} (lr={backbone_lr:.2e}), '
          f'head params={n_head} (lr={base_lr:.2e})')
    return optimizer


def rebuild_optimizer_and_scheduler(model, opt, steps_per_epoch, start_epoch, criterion=None):
    """
    Gọi sau khi thay đổi requires_grad để rebuild optimizer với đúng param groups.
    Scheduler reset về epoch hiện tại.

    QUAN TRỌNG: build_optimizer_stage2 chỉ duyệt model.named_parameters() nên KHÔNG
    gồm param của criterion (vd các ArcFace classifier nằm trong self.loss). Phải
    add lại chúng — nếu thiếu, grad của chúng không bao giờ được zero_grad → tích
    luỹ vô hạn → clip_grad_norm_ (tính trên toàn model_with_loss) nhiễm inf/NaN →
    đầu độc toàn bộ grad → NaN weights → pred NaN → matcher assert.
    """
    optimizer = build_optimizer_stage2(model, opt)
    if criterion is not None:
        extra = [p for p in criterion.parameters() if p.requires_grad]
        if extra:
            # lr mặc định = base_lr (khớp hành vi add_param_group ở BaseTrainer.__init__)
            optimizer.add_param_group({'params': extra})
            print(f'[Stage2] Re-added {len(extra)} criterion (ReID/ArcFace) params to optimizer')
    scheduler = build_scheduler(optimizer, opt, steps_per_epoch)
    if start_epoch > 0:
        scheduler.fast_forward(start_epoch * steps_per_epoch)
    return optimizer, scheduler


def apply_freeze_schedule(model, trainer, opt, epoch, steps_per_epoch,
                          freeze_backbone_epochs, unfreeze_upper_epochs,
                          unfreeze_ratio=0.34):
    """
    Gọi ĐẦU MỖI EPOCH để áp dụng freeze schedule.
    Trả về (optimizer, scheduler) mới nếu cần rebuild, hoặc None nếu không thay đổi.
    """
    rebuild = False

    if epoch == 1:
        # Giai đoạn 1: Freeze toàn bộ backbone
        freeze_backbone_fully(model)
        rebuild = True

    elif epoch == freeze_backbone_epochs + 1:
        # Giai đoạn 2: Unfreeze phần trên ViT + sta
        unfreeze_backbone_upper(model, unfreeze_ratio=unfreeze_ratio)
        rebuild = True

    elif epoch == freeze_backbone_epochs + unfreeze_upper_epochs + 1:
        # Giai đoạn 3: Unfreeze toàn bộ backbone
        unfreeze_backbone_all(model)
        rebuild = True

    if rebuild:
        optimizer, scheduler = rebuild_optimizer_and_scheduler(
            model, opt, steps_per_epoch, start_epoch=epoch - 1,
            criterion=getattr(trainer, 'loss', None),
        )
        # Gán lại vào trainer
        trainer.optimizer = optimizer
        return optimizer, scheduler

    return None, None


# =============================================================================
# LR Schedulers (giữ nguyên từ train.py gốc)
# =============================================================================

def _flat_cosine_lr(total_iter, warmup_iter, flat_iter, no_aug_iter,
                    current_iter, init_lr, min_lr):
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
    def __init__(self, optimizer, steps_per_epoch, num_epochs,
                 warmup_iter, flat_epochs, no_aug_epochs, lr_gamma):
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
        print(f'[FlatCosineScheduler] total={total_iter} warmup={warmup_iter} '
              f'flat={flat_iter} no_aug={no_aug_iter} '
              f'base_lrs={self._base_lrs} min_lrs={self._min_lrs}')

    def step(self):
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = _flat_cosine_lr(
                self._total_iter, self._warmup_iter,
                self._flat_iter,  self._no_aug_iter,
                self._cur_iter,
                self._base_lrs[i], self._min_lrs[i],
            )
        self._cur_iter += 1

    def fast_forward(self, n_steps):
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

    def load_state_dict(self, state):
        self._cur_iter = state.get('cur_iter', 0)


class RFDETRScheduler:
    def __init__(self, optimizer, steps_per_epoch, num_epochs,
                 warmup_epochs, lr_drop, lr_scheduler, lr_min_factor):
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
        print(f'[RFDETRScheduler] type={lr_scheduler} total={self._total_steps} '
              f'warmup={self._warmup_steps} lr_drop_step={self._lr_drop_steps} '
              f'lr_min_factor={lr_min_factor} base_lrs={self._base_lrs}')

    def _compute_lr(self, step, base_lr):
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

    def fast_forward(self, n_steps):
        self._cur_step = n_steps
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = self._compute_lr(self._cur_step, self._base_lrs[i])

    def state_dict(self):
        return {'cur_step': self._cur_step}

    def load_state_dict(self, state):
        self._cur_step = state.get('cur_step', 0)


def build_scheduler(optimizer, opt, steps_per_epoch):
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


# =============================================================================
# Eval (giữ nguyên)
# =============================================================================

@torch.no_grad()
def run_coco_eval(model, val_loader, opt, ann_file=''):
    model.eval()
    net_h, net_w = opt.input_wh[1], opt.input_wh[0]
    postprocessor = FalconJDEPostProcessor(
        num_classes=opt.num_classes, num_top_queries=500,
        use_focal_loss=True, conf_thres=0.0
    )
    evaluator = CocoJsonEvaluator(ann_file) if ann_file else JDECocoEvaluator(
        num_classes=opt.num_classes, net_h=net_h, net_w=net_w)
    max_batches = getattr(opt, 'debug_val_batches', 0)
    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(opt.device, non_blocking=True)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        orig_hw    = batch.get('orig_hw')
        orig_sizes = orig_hw if orig_hw is not None else \
                     torch.tensor([[net_h, net_w]] * batch['input'].shape[0], device=opt.device)
        outputs    = model(batch['input'])
        dt_results = postprocessor(outputs, orig_sizes)
        evaluator.update(dt_results, batch)
    model.train()
    return evaluator.summarize()


# =============================================================================
# Main run — Stage 2 với freeze schedule
# =============================================================================

def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    # --- Freeze schedule params (có thể thêm vào opts.py nếu muốn CLI) ---
    freeze_backbone_epochs  = getattr(opt, 'freeze_backbone_epochs',  5)
    unfreeze_upper_epochs   = getattr(opt, 'unfreeze_upper_epochs',   5)
    unfreeze_ratio          = getattr(opt, 'unfreeze_ratio',          0.34)

    print(f'[Stage2] freeze_backbone_epochs={freeze_backbone_epochs}, '
          f'unfreeze_upper_epochs={unfreeze_upper_epochs}, '
          f'unfreeze_ratio={unfreeze_ratio}')
    print(f'[Stage2] LR={opt.lr} (backbone_lr_ratio={getattr(opt, "backbone_lr_ratio", 0.05)})')

    print('Setting up data...')
    Dataset      = get_dataset(opt.dataset, opt.task)
    use_coco_fmt = (opt.dataset == 'coco')

    with open(opt.data_cfg) as f:
        data_config  = json.load(f)
    dataset_root = data_config['root']
    print("Dataset root: %s" % dataset_root)

    from falconmot.datasets.dataset.coco_detection import VisDroneCocoDataset

    if use_coco_fmt:
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
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # --- Val dataset ---
    val_loader   = None
    val_ann_file = ''
    if getattr(opt, 'val_cfg', ''):
        with open(opt.val_cfg) as f:
            val_config = json.load(f)
        val_dataset = None
        if use_coco_fmt:
            val_ann_file = val_config.get('val_ann', '')
            val_img      = val_config.get('val_img', '')
            if val_ann_file and val_img:
                val_dataset = VisDroneCocoDataset(
                    opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
        else:
            val_root  = val_config.get('root', dataset_root)
            val_paths = val_config.get('val') or val_config.get('test') or []
            if val_paths:
                val_dataset = Dataset(opt=opt, root=val_root, paths=val_paths,
                                      img_size=opt.input_wh, augment=False,
                                      transforms=T.Compose([T.ToTensor()]))
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.num_workers, pin_memory=True, drop_last=False,
                persistent_workers=opt.num_workers > 0,
                prefetch_factor=2 if opt.num_workers > 0 else None,
            )
            print(f'Val dataset: {len(val_dataset)} images')

    print('Creating model...')
    model = create_model(opt.arch, opt)

    start_epoch = 0
    if opt.load_model != '':
        # Load stage-1 best model — KHÔNG load optimizer (fine-tuning từ đầu)
        # model, _, start_epoch = load_model(
        #     model, opt.load_model, optimizer=None, resume=False,
        #     lr=opt.lr, lr_step=opt.lr_step
        # )
        model = load_model(
            model, opt.load_model, optimizer=None, resume=False,
            lr=opt.lr, lr_step=opt.lr_step
        )
        start_epoch = 0   # bắt đầu từ epoch 1 cho stage 2
        print(f'[Stage2] Loaded stage-1 weights from {opt.load_model}, reset epoch to 0')

    _nw = opt.num_workers
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=_nw, pin_memory=True, drop_last=True,
        persistent_workers=_nw > 0,
        prefetch_factor=2 if _nw > 0 else None,
    )
    steps_per_epoch = len(train_loader)

    print('Starting Stage-2 training...')
    Trainer = train_factory[opt.task]

    # Khởi tạo ban đầu với backbone đã freeze (epoch 1 sẽ apply freeze trước khi train)
    freeze_backbone_fully(model)
    optimizer = build_optimizer_stage2(model, opt)

    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    scheduler = build_scheduler(optimizer, opt, steps_per_epoch=steps_per_epoch)

    best_mAP = 0.0

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # --- Apply freeze schedule TRƯỚC khi train ---
        new_opt, new_sched = apply_freeze_schedule(
            model, trainer, opt, epoch, steps_per_epoch,
            freeze_backbone_epochs, unfreeze_upper_epochs, unfreeze_ratio
        )
        if new_opt is not None:
            optimizer = new_opt
            scheduler = new_sched
            # Cập nhật optimizer trong trainer
            trainer.optimizer = optimizer

        mark = epoch if opt.save_all else 'last'
        train_loader.dataset.set_epoch(epoch - 1)

        log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

        cur_lr = optimizer.param_groups[0]['lr']
        logger.write('epoch: {} |'.format(epoch))
        logger.write('lr {:e} | '.format(cur_lr))

        # Log freeze stage
        frozen_count = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        if epoch <= freeze_backbone_epochs:
            stage_tag = f'[freeze_all backbone frozen={frozen_count}]'
        elif epoch <= freeze_backbone_epochs + unfreeze_upper_epochs:
            stage_tag = f'[unfreeze_upper backbone frozen={frozen_count}]'
        else:
            stage_tag = '[full_train]'
        logger.write(f'{stage_tag} | ')

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
                       epoch, model, optimizer)

        if val_loader is not None and opt.val_intervals > 0 \
                and epoch % opt.val_intervals == 0:
            print(f'\n[Eval] epoch {epoch} {stage_tag} — running COCO mAP...')
            metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)
            log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
            print(f'[Eval] {log_line}')
            logger.write(f'[eval] {log_line} | ')
            for k, v in metrics.items():
                logger.scalar_summary(f'val_{k}', v, epoch)
            cur_mAP = metrics.get('AP', 0.0)
            if cur_mAP > best_mAP:
                best_mAP = cur_mAP
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model, optimizer)
                print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

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