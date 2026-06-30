"""
DEIM-JDE training step.
 
Detection loss = the `deimv2_dinov3_s` recipe:
    loss_mal (gamma=1.5) + loss_bbox(5) + loss_giou(2) + loss_fgl(0.15) + loss_ddf(1.5)
ReID loss (MOT extension): loss_reid (+ optional loss_s4_aux).
"""
 
from __future__ import absolute_import, division, print_function
 
import torch.nn as nn
 
from falconmot.nn.falcon_jde import FalconJDECriterion, HungarianMatcher
from .trainer import BaseTrainer
 
 
def _build_criterion(opt) -> FalconJDECriterion:
    # ----- Resolve SI-WBD config FIRST (matcher cần biết để dùng trong matching) -----
    use_siwbd = getattr(opt, 'use_siwbd', False)
    # Back-compat: --siwbd_replaces_giou ép sang 'replace'.
    box_reg_mode = getattr(opt, 'box_reg_mode', 'blend')
    if getattr(opt, 'siwbd_replaces_giou', False):
        box_reg_mode = 'replace'
 
    # Matcher: DEIM-style cost; switches to IoU-aware near the end of training.
    # matcher_change_epoch should be ~76% of total epochs (DEIMv2-S: 100/132).
    # [FIXED] Wire SI-WBD vào matcher để matching dùng cùng tín hiệu với loss.
    matcher_wd = {'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
    if use_siwbd:
        matcher_wd['use_siwbd']    = True
        matcher_wd['siwbd_C']      = getattr(opt, 'siwbd_C', 0.5)
        matcher_wd['cost_siwbd']   = getattr(opt, 'cost_siwbd', 2.0)
        matcher_wd['box_reg_mode'] = box_reg_mode
        matcher_wd['siwbd_beta']   = getattr(opt, 'siwbd_beta', 1.0)
        matcher_wd['siwbd_gate_center'] = getattr(opt, 'siwbd_gate_center', 0.003)
        matcher_wd['siwbd_gate_scale']  = getattr(opt, 'siwbd_gate_scale', 1.0)
 
    matcher = HungarianMatcher(
        weight_dict          = matcher_wd,
        use_focal_loss       = True,
        alpha                = 0.25,
        gamma                = 2.0,
        change_matcher       = getattr(opt, 'change_matcher', False),
        iou_order_alpha      = getattr(opt, 'iou_order_alpha', 4.0),
        matcher_change_epoch = getattr(opt, 'matcher_change_epoch', 100),
    )
 
    use_s4_aux = getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', False)
 
    # Detection loss weights (deimv2_dinov3_s).
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
 
    # ----- Fovea-MOT: SI-WBD box loss -----
    # (use_siwbd / box_reg_mode đã được resolve ở đầu hàm để wire vào matcher.)
    # loss_siwbd is a *separate* weighted term only in 'add' mode; in replace/blend
    # the overlap signal lives in the loss_giou slot (weight 2.0).
    if use_siwbd and box_reg_mode == 'add':
        weight_dict['loss_siwbd'] = getattr(opt, 'siwbd_weight', 2.0)
 
    # ----- Fovea-MOT: SAFA entropy supervision -----
    use_entropy_aux = getattr(opt, 'use_safa', False) and getattr(opt, 'use_entropy_aux', True)
    if use_entropy_aux:
        weight_dict['loss_entropy'] = getattr(opt, 'entropy_weight', 0.5)
        losses.append('entropy')
 
    return FalconJDECriterion(
        matcher             = matcher,
        num_classes         = opt.num_classes,
        nid_dict            = opt.nID_dict,
        reid_dim            = getattr(opt, 'reid_dim', 128),
        weight_dict         = weight_dict,
        losses              = tuple(losses),
        gamma               = 1.5,    # MAL gamma (deimv2_s)
        mal_alpha           = getattr(opt, 'mal_alpha', None),  # None = original DEIM
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
        # ----- Fovea-MOT -----
        use_siwbd           = use_siwbd,
        siwbd_C             = getattr(opt, 'siwbd_C', 0.5),
        siwbd_replaces_giou = getattr(opt, 'siwbd_replaces_giou', False),
        box_reg_mode        = box_reg_mode,
        siwbd_beta          = getattr(opt, 'siwbd_beta', 1.0),
        siwbd_gate_center   = getattr(opt, 'siwbd_gate_center', 0.003),
        siwbd_gate_scale    = getattr(opt, 'siwbd_gate_scale', 1.0),
        siwbd_gate_dynamic  = getattr(opt, 'siwbd_gate_dynamic', False),
        siwbd_gate_momentum = getattr(opt, 'siwbd_gate_momentum', 0.99),
        use_tucl            = getattr(opt, 'use_tucl', False),
        tucl_lambda         = getattr(opt, 'tucl_lambda', 0.05),
        tucl_detach_emb     = getattr(opt, 'tucl_detach_emb', True),
        tucl_quality_prior  = getattr(opt, 'tucl_quality_prior', True),
        use_entropy_aux     = use_entropy_aux,
    )
 
 
class FalconJDEWithLoss(nn.Module):
    """Combine model + criterion into a single forward for DataParallel."""
 
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
 
        # Detection: report main + each decoder aux layer separately for easier monitoring.
        # (the last/main layer has no loss_ddf because it is the DDF teacher)
        main_keys = ['loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl']
        aux_keys  = ['loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl', 'loss_ddf']
        n_aux = max(getattr(opt, 'num_dec_layers', 4) - 1, 0)
        loss_states += main_keys
        for i in range(n_aux):
            loss_states += [f'{k}_aux_{i}' for k in aux_keys]
 
        if getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', False):
            loss_states.append('loss_s4_aux')
        if getattr(opt, 'use_siwbd', False):
            _brm = getattr(opt, 'box_reg_mode', 'blend')
            if getattr(opt, 'siwbd_replaces_giou', False):
                _brm = 'replace'
            if _brm == 'add':
                loss_states.append('loss_siwbd')
        if getattr(opt, 'use_safa', False) and getattr(opt, 'use_entropy_aux', True):
            loss_states.append('loss_entropy')
        criterion = _build_criterion(opt)
        return loss_states, criterion
 
    def _build_model_with_loss(self, model, loss):
        return FalconJDEWithLoss(model, loss)
 
    def save_result(self, output, batch, results):
        pass





# """
# DEIM-JDE training step.
 
# Detection loss = the `deimv2_dinov3_s` recipe:
#     loss_mal (gamma=1.5) + loss_bbox(5) + loss_giou(2) + loss_fgl(0.15) + loss_ddf(1.5)
# ReID loss (MOT extension): loss_reid (+ optional loss_s4_aux).
# """
 
# from __future__ import absolute_import, division, print_function
 
# import torch.nn as nn
 
# from falconmot.nn.falcon_jde import FalconJDECriterion, HungarianMatcher
# from .trainer import BaseTrainer
 
 
# def _build_criterion(opt) -> FalconJDECriterion:
#     # ----- Resolve SI-WBD config FIRST (matcher cần biết để dùng trong matching) -----
#     use_siwbd = getattr(opt, 'use_siwbd', False)
#     # Back-compat: --siwbd_replaces_giou ép sang 'replace'.
#     box_reg_mode = getattr(opt, 'box_reg_mode', 'blend')
#     if getattr(opt, 'siwbd_replaces_giou', False):
#         box_reg_mode = 'replace'
 
#     # Matcher: DEIM-style cost; switches to IoU-aware near the end of training.
#     # matcher_change_epoch should be ~76% of total epochs (DEIMv2-S: 100/132).
#     # [FIXED] Wire SI-WBD vào matcher để matching dùng cùng tín hiệu với loss.
#     matcher_wd = {'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
#     if use_siwbd:
#         matcher_wd['use_siwbd']    = True
#         matcher_wd['siwbd_C']      = getattr(opt, 'siwbd_C', 0.5)
#         matcher_wd['cost_siwbd']   = getattr(opt, 'cost_siwbd', 2.0)
#         matcher_wd['box_reg_mode'] = box_reg_mode
#         matcher_wd['siwbd_beta']   = getattr(opt, 'siwbd_beta', 1.0)
#         matcher_wd['siwbd_gate_center'] = getattr(opt, 'siwbd_gate_center', 0.003)
#         matcher_wd['siwbd_gate_scale']  = getattr(opt, 'siwbd_gate_scale', 1.0)
 
#     matcher = HungarianMatcher(
#         weight_dict          = matcher_wd,
#         use_focal_loss       = True,
#         alpha                = 0.25,
#         gamma                = 2.0,
#         change_matcher       = getattr(opt, 'change_matcher', False),
#         iou_order_alpha      = getattr(opt, 'iou_order_alpha', 4.0),
#         matcher_change_epoch = getattr(opt, 'matcher_change_epoch', 100),
#     )
 
#     use_s4_aux = getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', False)
 
#     # Detection loss weights (deimv2_dinov3_s).
#     weight_dict = {
#         'loss_mal':  1.0,
#         'loss_bbox': 5.0,
#         'loss_giou': 2.0,
#         'loss_fgl':  0.15,
#         'loss_ddf':  1.5,
#     }
#     losses = ['mal', 'boxes', 'local']
#     if use_s4_aux:
#         weight_dict['loss_s4_aux'] = 0.2
#         losses.append('s4_aux')
 
#     # ----- Fovea-MOT: SI-WBD box loss -----
#     # (use_siwbd / box_reg_mode đã được resolve ở đầu hàm để wire vào matcher.)
#     # loss_siwbd is a *separate* weighted term only in 'add' mode; in replace/blend
#     # the overlap signal lives in the loss_giou slot (weight 2.0).
#     if use_siwbd and box_reg_mode == 'add':
#         weight_dict['loss_siwbd'] = getattr(opt, 'siwbd_weight', 2.0)
 
#     # ----- Fovea-MOT: SAFA entropy supervision -----
#     use_entropy_aux = getattr(opt, 'use_safa', False) and getattr(opt, 'use_entropy_aux', True)
#     if use_entropy_aux:
#         weight_dict['loss_entropy'] = getattr(opt, 'entropy_weight', 0.5)
#         losses.append('entropy')
 
#     return FalconJDECriterion(
#         matcher             = matcher,
#         num_classes         = opt.num_classes,
#         nid_dict            = opt.nID_dict,
#         reid_dim            = getattr(opt, 'reid_dim', 128),
#         weight_dict         = weight_dict,
#         losses              = tuple(losses),
#         gamma               = 1.5,    # MAL gamma (deimv2_s)
#         mal_alpha           = getattr(opt, 'mal_alpha', None),  # None = original DEIM
#         reg_max             = getattr(opt, 'reg_max', 32),
#         boxes_weight_format = None,
#         use_uni_set         = True,
#         # ----- ReID -----
#         use_reid            = getattr(opt, 'use_reid', True),
#         id_weight           = getattr(opt, 'id_weight', 1.0),
#         use_triplet         = getattr(opt, 'tri', False),
#         use_arcface         = getattr(opt, 'use_arcface', False),
#         s_det_init          = getattr(opt, 's_det_init', 2.5),
#         s_id_init           = getattr(opt, 's_id_init', 1.85),
#         # ----- Fovea-MOT -----
#         use_siwbd           = use_siwbd,
#         siwbd_C             = getattr(opt, 'siwbd_C', 0.5),
#         siwbd_replaces_giou = getattr(opt, 'siwbd_replaces_giou', False),
#         box_reg_mode        = box_reg_mode,
#         siwbd_beta          = getattr(opt, 'siwbd_beta', 1.0),
#         siwbd_gate_center   = getattr(opt, 'siwbd_gate_center', 0.003),
#         siwbd_gate_scale    = getattr(opt, 'siwbd_gate_scale', 1.0),
#         siwbd_gate_dynamic  = getattr(opt, 'siwbd_gate_dynamic', False),
#         siwbd_gate_momentum = getattr(opt, 'siwbd_gate_momentum', 0.99),
#         use_tucl            = getattr(opt, 'use_tucl', False),
#         tucl_lambda         = getattr(opt, 'tucl_lambda', 0.05),
#         use_entropy_aux     = use_entropy_aux,
#         # ----- TOD-Head: Orthogonal Feature Loss -----
#         use_tod             = getattr(opt, 'use_tod', False),
#         ortho_weight        = getattr(opt, 'ortho_weight', 0.1),
#         ortho_mode          = getattr(opt, 'ortho_mode', 'channel'),
#         ortho_warmup_epochs = getattr(opt, 'ortho_warmup_epochs', 0),
#     )
 
 
# class FalconJDEWithLoss(nn.Module):
#     """Combine model + criterion into a single forward for DataParallel."""
 
#     def __init__(self, model, criterion):
#         super().__init__()
#         self.model = model
#         self.criterion = criterion
 
#     def forward(self, batch, epoch=0):
#         B = batch['input'].shape[0]
#         targets = []
#         for i in range(B):
#             n = int(batch['detr_num_objs'][i].item())
#             valid_labels = batch['detr_labels'][i, :n]
#             valid_boxes = batch['detr_boxes'][i, :n]
#             valid_tids = batch['detr_track_ids'][i, :n]
#             keep = valid_labels >= 0
#             targets.append({
#                 'labels':    valid_labels[keep],
#                 'boxes':     valid_boxes[keep],
#                 'track_ids': valid_tids[keep],
#             })
 
#         outputs = self.model(batch['input'], targets)
#         loss_dict = self.criterion(outputs, targets, epoch=epoch)
#         return outputs, loss_dict['loss'], loss_dict
 
 
# class MotTrainer(BaseTrainer):
#     def __init__(self, opt, model, optimizer=None, **kwargs):
#         super().__init__(opt, model, optimizer=optimizer, **kwargs)
 
#     def _get_losses(self, opt):
#         loss_states = ['loss', 'loss_det']
#         if getattr(opt, 'use_reid', True) and getattr(opt, 'id_weight', 1.0) > 0:
#             loss_states += ['loss_reid', 's_det', 's_id']
 
#         # Detection: report main + each decoder aux layer separately for easier monitoring.
#         # (the last/main layer has no loss_ddf because it is the DDF teacher)
#         main_keys = ['loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl']
#         aux_keys  = ['loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl', 'loss_ddf']
#         n_aux = max(getattr(opt, 'num_dec_layers', 4) - 1, 0)
#         loss_states += main_keys
#         for i in range(n_aux):
#             loss_states += [f'{k}_aux_{i}' for k in aux_keys]
 
#         if getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', False):
#             loss_states.append('loss_s4_aux')
#         if getattr(opt, 'use_siwbd', False):
#             _brm = getattr(opt, 'box_reg_mode', 'blend')
#             if getattr(opt, 'siwbd_replaces_giou', False):
#                 _brm = 'replace'
#             if _brm == 'add':
#                 loss_states.append('loss_siwbd')
#         # if getattr(opt, 'use_safa', False) and getattr(opt, 'use_entropy_aux', True):
#         #     loss_states.append('loss_entropy')
#         if getattr(opt, 'use_safa', False) and getattr(opt, 'use_entropy_aux', True):
#             loss_states.append('loss_entropy')
#         if getattr(opt, 'use_tod', False) and getattr(opt, 'ortho_weight', 0.1) > 0:
#             loss_states.append('loss_ortho')
#         criterion = _build_criterion(opt)
#         return loss_states, criterion
 
#     def _build_model_with_loss(self, model, loss):
#         return FalconJDEWithLoss(model, loss)
 
#     def save_result(self, output, batch, results):
#         pass