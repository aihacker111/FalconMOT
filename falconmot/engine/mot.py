"""
DEIM-JDE training step.

Detection loss = đúng recipe `deimv2_dinov3_s`:
    loss_mal (gamma=1.5) + loss_bbox(5) + loss_giou(2) + loss_fgl(0.15) + loss_ddf(1.5)
ReID loss (mở rộng MOT): loss_reid (+ loss_s4_aux tuỳ chọn).
"""

from __future__ import absolute_import, division, print_function

import torch.nn as nn

from falconmot.models.falcon_jde import FalconJDECriterion, HungarianMatcher
from .base_trainer import BaseTrainer


def _build_criterion(opt) -> FalconJDECriterion:
    # Matcher: cost giống DEIM; chuyển sang IoU-aware ở cuối quá trình train.
    # matcher_change_epoch nên ~76% tổng số epoch (DEIMv2-S: 100/132).
    matcher = HungarianMatcher(
        weight_dict          = {'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0},
        use_focal_loss       = True,
        alpha                = 0.25,
        gamma                = 2.0,
        change_matcher       = getattr(opt, 'change_matcher', True),
        iou_order_alpha      = getattr(opt, 'iou_order_alpha', 4.0),
        matcher_change_epoch = getattr(opt, 'matcher_change_epoch', 50),
    )

    use_s4_aux = getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', False)

    # Trọng số loss detection (deimv2_dinov3_s).
    weight_dict = {
        'loss_mal':  1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
        'loss_fgl':  0.15,
        'loss_ddf':  1.5,
    }
    losses = ['mal', 'boxes', 'local']
    if use_s4_aux:
        weight_dict['loss_s4_aux'] = 0.2
        losses.append('s4_aux')

    return FalconJDECriterion(
        matcher             = matcher,
        num_classes         = opt.num_classes,
        nid_dict            = opt.nID_dict,
        reid_dim            = getattr(opt, 'reid_dim', 128),
        weight_dict         = weight_dict,
        losses              = tuple(losses),
        gamma               = 1.5,    # MAL gamma (deimv2_s)
        mal_alpha           = getattr(opt, 'mal_alpha', None),  # None = DEIM gốc
        reg_max             = getattr(opt, 'reg_max', 32),
        boxes_weight_format = None,
        use_uni_set         = True,
        # ----- ReID -----
        use_reid            = getattr(opt, 'use_reid', True),
        id_weight           = getattr(opt, 'id_weight', 1.0),
        use_triplet         = getattr(opt, 'tri', False),
        use_arcface         = getattr(opt, 'use_arcface', False),
        s_det_init          = getattr(opt, 's_det_init', 2.5),
        s_id_init           = getattr(opt, 's_id_init', 1.85),
    )


class FalconJDEWithLoss(nn.Module):
    """Gộp model + criterion thành một forward cho DataParallel."""

    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, batch, epoch=0):
        B = batch['input'].shape[0]
        targets = []
        for i in range(B):
            n = int(batch['detr_num_objs'][i].item())
            valid_labels = batch['detr_labels'][i, :n]
            valid_boxes = batch['detr_boxes'][i, :n]
            valid_tids = batch['detr_track_ids'][i, :n]
            keep = valid_labels >= 0
            targets.append({
                'labels':    valid_labels[keep],
                'boxes':     valid_boxes[keep],
                'track_ids': valid_tids[keep],
            })

        outputs = self.model(batch['input'], targets)
        loss_dict = self.criterion(outputs, targets, epoch=epoch)
        return outputs, loss_dict['loss'], loss_dict


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None, **kwargs):
        super().__init__(opt, model, optimizer=optimizer, **kwargs)

    def _get_losses(self, opt):
        loss_states = ['loss', 'loss_det']
        if getattr(opt, 'use_reid', True) and getattr(opt, 'id_weight', 1.0) > 0:
            loss_states += ['loss_reid', 's_det', 's_id']
        loss_states += ['loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl', 'loss_ddf']
        if getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', False):
            loss_states.append('loss_s4_aux')
        criterion = _build_criterion(opt)
        return loss_states, criterion

    def _build_model_with_loss(self, model, loss):
        return FalconJDEWithLoss(model, loss)

    def save_result(self, output, batch, results):
        pass