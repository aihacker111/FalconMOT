"""
AdamW optimizer builder with discriminative learning rates for FalconMOT.

Param groups (up to 6, empty groups are dropped):
  backbone  non-norm : lr * backbone_lr_factor, weight_decay
  backbone  norm/bias: lr * backbone_lr_factor, weight_decay=0
  reid_head non-norm : lr * reid_lr_factor,     weight_decay
  reid_head norm/bias: lr * reid_lr_factor,     weight_decay=0
  other     norm/bias: lr,                      weight_decay=0
  everything else    : lr,                      weight_decay
"""
import torch
import torch.nn as nn

_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d,
)


def build_optimizer(model: nn.Module, opt) -> torch.optim.Optimizer:
    """Build AdamW with per-group LRs and weight-decay splitting.

    Norm layers are detected by module type (not param name) so that BN gamma
    params inside anonymous nn.Sequential blocks are correctly assigned wd=0.
    """
    base_lr      = opt.lr
    backbone_lr  = base_lr * getattr(opt, 'backbone_lr_factor', 0.05)
    reid_lr      = base_lr * getattr(opt, 'reid_lr_factor', 1.0)
    weight_decay = opt.weight_decay

    norm_param_names: set = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, _NORM_TYPES):
            for p_name, _ in module.named_parameters(recurse=False):
                norm_param_names.add(f'{mod_name}.{p_name}')

    backbone_wd, backbone_no_wd = [], []
    reid_wd, reid_no_wd         = [], []
    other_no_wd, default        = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = 'backbone' in name
        is_reid     = 'reid_head' in name
        is_no_wd    = name in norm_param_names or name.endswith('.bias')
        if is_backbone and is_no_wd:
            backbone_no_wd.append(param)
        elif is_backbone:
            backbone_wd.append(param)
        elif is_reid and is_no_wd:
            reid_no_wd.append(param)
        elif is_reid:
            reid_wd.append(param)
        elif is_no_wd:
            other_no_wd.append(param)
        else:
            default.append(param)

    param_groups = [
        pg for pg in [
            {'params': backbone_wd,    'lr': backbone_lr},
            {'params': backbone_no_wd, 'lr': backbone_lr, 'weight_decay': 0.},
            {'params': reid_wd,        'lr': reid_lr},
            {'params': reid_no_wd,     'lr': reid_lr,     'weight_decay': 0.},
            {'params': other_no_wd,                       'weight_decay': 0.},
            {'params': default},
        ]
        if pg['params']
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
