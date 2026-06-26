"""
LR schedulers for FalconMOT training.

FlatCosineScheduler — quadratic warmup → flat → cosine decay → no-aug (default)
RFDETRScheduler     — linear warmup → cosine or step decay (ported from RF-DETR)
build_scheduler     — factory dispatching on opt.lr_scheduler
"""
import math
import torch


# ---------------------------------------------------------------------------
# Flat-cosine helpers
# ---------------------------------------------------------------------------

def _flat_cosine_lr(total_iter, warmup_iter, flat_iter, no_aug_iter,
                    current_iter, init_lr, min_lr):
    """Quadratic warmup → flat → cosine decay → no-aug floor."""
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
    """Per-iteration flat-cosine scheduler compatible with base_trainer.step() API.

    Phases:
      1. Quadratic warmup  : 0 → init_lr  (warmup_iter steps)
      2. Flat              : init_lr constant  (flat_epochs epochs)
      3. Cosine decay      : init_lr → min_lr
      4. No-aug floor      : min_lr constant  (no_aug_epochs final epochs)
    """

    def __init__(self, optimizer, steps_per_epoch: int, num_epochs: int,
                 warmup_iter: int, flat_epochs: int, no_aug_epochs: int,
                 lr_gamma: float):
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        self._optimizer  = optimizer
        self._base_lrs   = [g['initial_lr'] for g in optimizer.param_groups]
        self._min_lrs    = [lr * lr_gamma   for lr in self._base_lrs]
        self._cur_iter   = 0

        self._total_iter  = num_epochs    * steps_per_epoch
        self._warmup_iter = warmup_iter
        self._flat_iter   = flat_epochs   * steps_per_epoch
        self._no_aug_iter = no_aug_epochs * steps_per_epoch

        print(
            f'[FlatCosineScheduler] total={self._total_iter} warmup={warmup_iter} '
            f'flat={self._flat_iter} no_aug={self._no_aug_iter} '
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
        """Jump scheduler state by n_steps (O(1))."""
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


# ---------------------------------------------------------------------------
# RF-DETR scheduler (linear warmup + cosine / step)
# ---------------------------------------------------------------------------

class RFDETRScheduler:
    """Linear warmup + cosine annealing or step decay (ported from RF-DETR).

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, opt, steps_per_epoch: int):
    """Dispatch on opt.lr_scheduler to create the appropriate scheduler.

    'cosine' | 'step'  → RFDETRScheduler (linear warmup + cosine/step)
    'flat_cosine'      → FlatCosineScheduler (quadratic warmup + flat + cosine)
    """
    lr_scheduler = getattr(opt, 'lr_scheduler', 'flat_cosine')

    if lr_scheduler in ('cosine', 'step'):
        return RFDETRScheduler(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            num_epochs=opt.num_epochs,
            warmup_epochs=getattr(opt, 'warmup_epochs', 0.0),
            lr_drop=getattr(opt, 'lr_drop', opt.num_epochs),
            lr_scheduler=lr_scheduler,
            lr_min_factor=getattr(opt, 'lr_min_factor', 0.0),
        )

    warmup_iter   = getattr(opt, 'warmup_iters',  2000)
    no_aug_epochs = getattr(opt, 'no_aug_epochs', 2)
    lr_gamma      = getattr(opt, 'lr_gamma',      0.5)
    stop_epoch    = getattr(opt, 'stop_epoch',    -1)
    flat_epochs   = stop_epoch if stop_epoch > 0 else opt.num_epochs // 2

    return FlatCosineScheduler(
        optimizer, steps_per_epoch, opt.num_epochs,
        warmup_iter, flat_epochs, no_aug_epochs, lr_gamma,
    )
