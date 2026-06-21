"""
DEIM-JDE training step.

Losses used:
    loss_cls       — sigmoid focal classification (α=0.25, γ=2.0)
    loss_bbox      — L1 box regression
    loss_giou      — GIoU box regression
    loss_reid      — CE per-class ReID (optional: + triplet)
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from falconmot.models.falcon_jde import FalconJDECriterion, HungarianMatcher
from .base_trainer import BaseTrainer


def _build_criterion(opt) -> FalconJDECriterion:
    matcher = HungarianMatcher(
        weight_dict   = {'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0},
        use_focal_loss = True,
        alpha          = 0.25,
        gamma          = 2.0,
    )

    use_rep      = getattr(opt, 'rep', False)
    rep_weight   = getattr(opt, 'rep_weight', 0.5)
    use_s4       = getattr(opt, 'use_s4', False)
    use_s4_aux   = use_s4 and getattr(opt, 'use_s4_aux', True)

    # DETR-style detection loss: loss_cls + loss_bbox + loss_giou
    weight_dict = {
        'loss_cls':  2.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
    }
    if use_s4_aux:
        weight_dict['loss_s4_aux'] = 0.2
    if use_rep:
        weight_dict['loss_rep'] = rep_weight

    base_losses = ['focal', 'boxes']
    if use_rep:
        base_losses.append('rep')
    if use_s4_aux:
        base_losses.append('s4_aux')
    losses = tuple(base_losses)

    return FalconJDECriterion(
        matcher             = matcher,
        num_classes         = opt.num_classes,
        nid_dict            = opt.nID_dict,
        reid_dim            = getattr(opt, 'reid_dim', 128),
        weight_dict         = weight_dict,
        losses              = losses,
        boxes_weight_format = None,
        use_uni_set         = True,
        use_reid            = getattr(opt, 'use_reid', True),
        id_weight           = getattr(opt, 'id_weight', 1.0),
        use_triplet         = getattr(opt, 'tri', False),
        # Plain CE + emb_scale is the stable FairMOT/AMOT recipe and the default.
        # ArcFace is opt-in (set --use_arcface) — it tends to overfit a low-capacity
        # head, which is exactly what degraded ReID over epochs previously.
        use_arcface         = getattr(opt, 'use_arcface', False),
        # s init ≈ log(initial loss_det) and log(initial loss_reid). Read the
        # first-iteration loss_det / loss_reid from the log and set these.
        s_det_init          = getattr(opt, 's_det_init', 2.5),
        s_id_init           = getattr(opt, 's_id_init', 1.85),
    )


class FalconJDEWithLoss(nn.Module):
    """Wraps model + criterion into a single forward for DataParallel."""

    def __init__(self, model, criterion):
        super().__init__()
        self.model     = model
        self.criterion = criterion

    def forward(self, batch):
        B = batch['input'].shape[0]
        targets = []
        for i in range(B):
            n            = int(batch['detr_num_objs'][i].item())
            valid_labels = batch['detr_labels'][i, :n]
            valid_boxes  = batch['detr_boxes'][i, :n]
            valid_tids   = batch['detr_track_ids'][i, :n]
            keep = valid_labels >= 0
            targets.append({
                'labels':    valid_labels[keep],
                'boxes':     valid_boxes[keep],
                'track_ids': valid_tids[keep],
            })

        outputs   = self.model(batch['input'], targets)
        loss_dict = self.criterion(outputs, targets)
        return outputs, loss_dict['loss'], loss_dict


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None, **kwargs):
        super().__init__(opt, model, optimizer=optimizer, **kwargs)

    def _get_losses(self, opt):
        loss_states = ['loss', 'loss_det']
        if getattr(opt, 'use_reid', True) and getattr(opt, 'id_weight', 1.0) > 0:
            loss_states += ['loss_reid', 's_det', 's_id']
        loss_states += ['loss_cls', 'loss_bbox', 'loss_giou']
        if getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True):
            loss_states.append('loss_s4_aux')
        if getattr(opt, 'rep', False):
            loss_states.append('loss_rep')
        criterion = _build_criterion(opt)
        return loss_states, criterion

    def _build_model_with_loss(self, model, loss):
        return FalconJDEWithLoss(model, loss)

    def save_result(self, output, batch, results):
        pass