"""
Two-phase staged fine-tuning for FalconJDE (stage-2 tracking).

Phase 0  (ReID warmup):  freeze backbone + encoder + decoder (+ S4 branch),
                         train ONLY reid_head + ReID classifiers (in criterion).
                         Detection path frozen *and* its BatchNorm running-stats
                         locked → mAP cannot move.
Phase 1  (joint):        unfreeze encoder + decoder (+ S4 branch), keep backbone
                         frozen (weights + BN stats). id_weight ramps 0→target
                         over the first `id_warmup_epochs` of this phase.

The optimizer (+ criterion params) and LR scheduler are rebuilt automatically at
the phase boundary, because the set of trainable params changes.
"""
from __future__ import absolute_import, division, print_function

import copy
import torch
import torch.nn as nn

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d)


def freeze_bn_stats(module, tag=''):
    """
    Lock BatchNorm running-stats (keep them in eval) while leaving the BN
    affine weights (gamma/beta) trainable. Standard detection-finetune trick:
    prevents eval-time degradation when the stage-2 augmentation distribution
    differs from the clean eval distribution.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, _BN_TYPES):
            m.eval()
            m.__dict__['train'] = (lambda _m: (lambda mode=True: _m))(m)  # stay eval
            n += 1
    if tag and n:
        print(f'  [stage] {tag:<10}: BN running-stats LOCKED ({n} BN layers), '
              f'affine weights still trainable')
    return n


# ---------------------------------------------------------------------------
# Freeze / unfreeze a submodule, including its BatchNorm running statistics.
# ---------------------------------------------------------------------------
def _set_module_trainable(module: torch.nn.Module, trainable: bool, tag: str = ''):
    """
    trainable=False -> requires_grad=False, module.eval(), and .train() is
                       neutralised so a parent model.train() can't drag its
                       BatchNorm back into "update running-stats" mode.
    trainable=True  -> restore grad + normal .train() behaviour.
    """
    n = 0
    for p in module.parameters():
        if p.requires_grad != trainable:
            p.requires_grad = trainable
            n += 1

    if trainable:
        # Remove the .train() override installed earlier, if any.
        if 'train' in module.__dict__:
            del module.__dict__['train']
        module.train()
    else:
        module.eval()
        # Make .train(mode) a no-op so frozen BN never updates running_mean/var.
        module.__dict__['train'] = lambda mode=True: module

    if tag:
        state = 'TRAIN' if trainable else 'FROZEN (eval, BN locked)'
        print(f'  [stage] {tag:<10} -> {state}  ({n} params toggled)')
    return n


def _core(model):
    return model.module if hasattr(model, 'module') else model


def _detection_modules(model):
    """Submodules that make up the detection path (everything except reid_head)."""
    m = _core(model)
    mods = [('backbone', m.backbone), ('encoder', m.encoder), ('decoder', m.decoder)]
    if getattr(m, 'use_s4', False):
        mods += [('s4_branch', m.s4_branch), ('s4_aux_head', m.s4_aux_head)]
    return mods


def _has_reid_head(model):
    m = _core(model)
    return getattr(m, 'use_reid', True) and hasattr(m, 'reid_head')


def apply_det_only(model):
    """Stage-1 detection-only: train backbone + encoder + decoder (+ S4), no ReID."""
    print('[stage] === Stage-1: detection-only (full detector trainable) ===')
    for tag, mod in _detection_modules(model):
        _set_module_trainable(mod, True, tag)
    if _has_reid_head(model):
        _set_module_trainable(_core(model).reid_head, False, 'reid_head')
    _report(model)


def apply_phase0(model):
    """Freeze whole detection path; leave reid_head trainable."""
    print('[stage] === Phase 0: ReID warmup (detector frozen) ===')
    for tag, mod in _detection_modules(model):
        _set_module_trainable(mod, False, tag)
    if _has_reid_head(model):
        _set_module_trainable(_core(model).reid_head, True, 'reid_head')
    else:
        print('  [stage] reid_head: SKIPPED (detection-only model)')
    _report(model)


def apply_phase1(model, keep_backbone_frozen: bool = False, freeze_norm: bool = False):
    """Unfreeze encoder + decoder (+ S4). Optionally keep backbone frozen.
    If freeze_norm, BatchNorm running-stats in encoder/S4 stay locked (eval) to
    avoid eval-time drift under heavy augmentation, while their weights train."""
    print('[stage] === Phase 1: joint fine-tune ===')
    m = _core(model)
    _set_module_trainable(m.backbone, not keep_backbone_frozen, 'backbone')
    _set_module_trainable(m.encoder, True, 'encoder')
    _set_module_trainable(m.decoder, True, 'decoder')
    if getattr(m, 'use_s4', False):
        _set_module_trainable(m.s4_branch, True, 's4_branch')
        _set_module_trainable(m.s4_aux_head, True, 's4_aux_head')
    if _has_reid_head(model):
        _set_module_trainable(m.reid_head, True, 'reid_head')

    if freeze_norm:
        # Call AFTER _set_module_trainable (which re-enables .train()); this
        # re-locks only the BN running-stats, leaving affine weights trainable.
        freeze_bn_stats(m.encoder, 'encoder')
        if getattr(m, 'use_s4', False):
            freeze_bn_stats(m.s4_branch, 's4_branch')
            freeze_bn_stats(m.s4_aux_head, 's4_aux_head')
    _report(model)


def _report(model):
    m = _core(model)
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in m.parameters())
    print(f'  [stage] trainable {tr/1e6:.2f}M / {tot/1e6:.2f}M params '
          f'({100*tr/max(tot,1):.1f}%)')


# ---------------------------------------------------------------------------
# Rebuild optimizer + scheduler for the current phase.
# ---------------------------------------------------------------------------
def build_phase_optimizer(model, criterion, opt, build_optimizer_fn, lr: float):
    """
    Reuse the project's build_optimizer (which already skips requires_grad=False
    params and applies the backbone discriminative LR), then re-attach the
    criterion's trainable params (ReID classifiers) — mirroring what
    BaseTrainer.__init__ does once at startup.
    """
    opt_lr = copy.copy(opt)
    opt_lr.lr = lr
    optimizer = build_optimizer_fn(model, opt_lr)

    crit_params = [p for p in criterion.parameters() if p.requires_grad]
    if crit_params:
        optimizer.add_param_group({'params': crit_params})
    return optimizer


def build_phase_scheduler(optimizer, opt, build_scheduler_fn,
                          steps_per_epoch: int, phase_epochs: int,
                          warmup_iters: int):
    """Fresh scheduler spanning only this phase's epochs."""
    opt_ph = copy.copy(opt)
    opt_ph.num_epochs   = phase_epochs
    opt_ph.warmup_iters = warmup_iters
    opt_ph.warmup_epochs = min(getattr(opt, 'warmup_epochs', 0.0),
                               max(phase_epochs - 1, 0))
    opt_ph.lr_drop      = phase_epochs
    return build_scheduler_fn(optimizer, opt_ph, steps_per_epoch)


# ---------------------------------------------------------------------------
# id_weight ramp helper.
# ---------------------------------------------------------------------------
def ramp_id_weight(criterion, target: float, epoch_in_phase: int, warmup_epochs: int):
    """Linear 0 -> target over `warmup_epochs` (epoch_in_phase is 1-indexed)."""
    if warmup_epochs <= 0:
        w = target
    else:
        w = target * min(1.0, epoch_in_phase / float(warmup_epochs))
    criterion.id_weight = w
    return w