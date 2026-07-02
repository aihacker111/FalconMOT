"""
Two-stage fine-tuning for FalconJDE.

Stage 1 (Detection Only): Train Backbone, Encoder, Decoder, S4 branch.
                          ReID head is frozen.
Stage 2 (ReID Only):      Freeze Backbone, Encoder, Decoder, S4 branch (including BN stats).
                          Train ONLY the ReID head.
"""
import copy
import torch
import torch.nn as nn

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d)

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
    """Submodules that make up the detection path (everything except reid_head).
    [OSD] duoc xep vao day de no train cung detector o Stage-1 (subspace det
    hinh thanh som); o Stage-2 no duoc mo rieng trong apply_stage2_mot."""
    m = _core(model)
    mods = [('backbone', m.backbone), ('encoder', m.encoder), ('decoder', m.decoder)]
    if getattr(m, 'use_s4', False):
        mods += [('s4_branch', m.s4_branch), ('s4_aux_head', m.s4_aux_head)]
    if getattr(m, 'use_osd', False) and hasattr(m, 'osd'):
        mods += [('osd', m.osd)]
    return mods

def _has_reid_head(model):
    m = _core(model)
    return getattr(m, 'use_reid', True) and hasattr(m, 'reid_head')

def _report(model):
    m = _core(model)
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in m.parameters())
    print(f'  [stage] trainable {tr/1e6:.2f}M / {tot/1e6:.2f}M params '
          f'({100*tr/max(tot,1):.1f}%)')

# ===========================================================================
# STAGE CONTROL FUNCTIONS
# ===========================================================================

def apply_det_only(model):
    """STAGE 1: Detection-only (Train full detector, freeze ReID)"""
    print('[stage] === STAGE 1: Detection-only (full detector trainable) ===')
    for tag, mod in _detection_modules(model):
        _set_module_trainable(mod, True, tag)
    if _has_reid_head(model):
        _set_module_trainable(_core(model).reid_head, False, 'reid_head')
    _report(model)


def apply_reid_only(model):
    """STAGE 2 (Freeze Detection): Freeze whole detection path; train reid_head ONLY."""
    print('[stage] === STAGE 2: ReID-only (Detector FROZEN, BN locked) ===')
    
    # 1. Khóa toàn bộ mạng Detection (Backbone, Encoder, Decoder)
    # Cơ chế _set_module_trainable(False) sẽ tự động set requires_grad=False,
    # gọi .eval() và vô hiệu hóa hàm .train() để BatchNorm không bao giờ bị cập nhật.
    for tag, mod in _detection_modules(model):
        _set_module_trainable(mod, False, tag)
        
    # 2. Mở khóa mạng ReID
    if _has_reid_head(model):
        _set_module_trainable(_core(model).reid_head, True, 'reid_head')
    else:
        print('  [stage] WARNING: reid_head not found in model!')
    _report(model)


def apply_joint_training(model):
    """FALLBACK (Joint): Unfreeze everything (Det + ReID)."""
    print('[stage] === JOINT TRAINING (All trainable) ===')
    for tag, mod in _detection_modules(model):
        _set_module_trainable(mod, True, tag)
    if _has_reid_head(model):
        _set_module_trainable(_core(model).reid_head, True, 'reid_head')
    _report(model)


def apply_stage2_mot(model):
    """
    STAGE 2 (MOT Fine-tuning): 
    Freeze Backbone & Encoder (giữ nguyên feature extraction).
    Train Decoder & ReID head (thích nghi với tracking queries).
    """
    print('[stage] === STAGE 2: Freeze Backbone/Encoder | Train Decoder + ReID ===')
    m = _core(model)

    # 1. Đóng băng Backbone & Encoder (bao gồm cả việc lock BatchNorm stats)
    _set_module_trainable(m.backbone, False, 'backbone')
    _set_module_trainable(m.encoder, False, 'encoder')

    # Xử lý nhánh S4 (Stride-4): Vì nó đóng vai trò trích xuất đặc trưng độ phân giải cao 
    # cho object nhỏ (giống Encoder), ta cũng nên freeze nó để tránh phá vỡ feature map.
    if getattr(m, 'use_s4', False):
        _set_module_trainable(m.s4_branch, False, 's4_branch')
        if hasattr(m, 's4_aux_head'):
            _set_module_trainable(m.s4_aux_head, False, 's4_aux_head')

    # 2. Mo khoa Decoder — FIX: truoc day bi set False nguoc voi comment,
    #    khien toan bo detection path dung im va MOTA khong the cai thien.
    _set_module_trainable(m.decoder, True, 'decoder')

    # 2b. [OSD] Mo khoa OSD — Q can tiep tuc xoay de dinh tuyen thong tin
    #     identity cua domain video vao subspace ReID.
    if getattr(m, 'use_osd', False) and hasattr(m, 'osd'):
        _set_module_trainable(m.osd, True, 'osd')

    # 3. Mở khóa mạng ReID
    if _has_reid_head(model):
        _set_module_trainable(m.reid_head, True, 'reid_head')
    else:
        print('  [stage] WARNING: reid_head not found in model!')

    _report(model)
# ===========================================================================
# OPTIMIZER / SCHEDULER UTILS
# ===========================================================================

def build_phase_optimizer(model, criterion, opt, build_optimizer_fn, lr: float):
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
    opt_ph = copy.copy(opt)
    opt_ph.num_epochs   = phase_epochs
    opt_ph.warmup_iters = warmup_iters
    opt_ph.warmup_epochs = min(getattr(opt, 'warmup_epochs', 0.0),
                               max(phase_epochs - 1, 0))
    opt_ph.lr_drop      = phase_epochs
    return build_scheduler_fn(optimizer, opt_ph, steps_per_epoch)