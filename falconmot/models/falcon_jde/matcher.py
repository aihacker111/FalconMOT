# """
# Hungarian Matcher — adapted from DEIMv2, extended with ODLTM-style Gaussian cost.

# ETDMOT (Sci. Reports 2024) ODLTM matching cost, Eq. (1):
#     L_m = L_f(class, focal) + L_1(bbox) + L_g(Gaussian KL divergence)

# Adaptation for the DETR/DEIM query paradigm:
#   - ETDMOT measures KLD between the *effective receptive field* Gaussian and the
#     *trajectory box* Gaussian. Queries have no per-query receptive field, so we
#     model BOTH pred box and GT box as 2D Gaussians (paper Eq. (3)(4)):
#         mu = (cx, cy),  Sigma = diag(w^2/4, h^2/4)
#     and use the closed-form KLD of Eq. (5).
#   - Why this helps VisDrone: GIoU saturates for tiny boxes (a few px offset
#     -> IoU = 0, no ranking signal). The Gaussian KLD stays smooth and finite,
#     so tiny GTs still attract their nearest queries in the assignment.

# Also fixes the `change_matcher` branch (epoch >= matcher_change_epoch):
#   - Original:  C = -(class_score * IoU^alpha)          # pure IoU — starves
#                                                         # small objects of
#                                                         # supervision late in
#                                                         # training
#   - Patched:   C = -(class_score * quality^alpha)
#     where quality = beta * IoU + (1 - beta) * exp(-sqrt(KLD))
#     (beta = gauss_iou_blend, default 0.5; set 1.0 to recover original).

# Backward compatible: if weight_dict has no 'cost_gauss', behaviour of the
# default branch is identical to the original matcher.
# """
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from typing import Dict

# from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


# # ---------------------------------------------------------------------------
# # ODLTM Gaussian KL divergence (ETDMOT Eq. (3)-(5), closed form, diagonal Sigma)
# # ---------------------------------------------------------------------------

# def gaussian_kld(out_bbox: torch.Tensor, tgt_bbox: torch.Tensor,
#                  eps: float = 1e-7) -> torch.Tensor:
#     """Pairwise KL( N_pred || N_gt ) between box Gaussians.

#     Boxes are (cx, cy, w, h), normalized to [0, 1].
#     N = ( mu=(cx,cy), Sigma=diag(w^2/4, h^2/4) )   — paper Eq. (3)(4)

#     KL(N_e||N_g) = 0.5 * [ tr(Sg^-1 Se) + (mu_g-mu_e)^T Sg^-1 (mu_g-mu_e)
#                            + ln(|Sg|/|Se|) ] - 1
#     With diagonal Sigmas this expands to the cheap elementwise form below.

#     Returns: (num_pred, num_gt) tensor, >= 0.
#     """
#     px, py = out_bbox[:, 0:1], out_bbox[:, 1:2]          # (P,1)
#     pw = out_bbox[:, 2:3].clamp(min=eps)
#     ph = out_bbox[:, 3:4].clamp(min=eps)

#     gx, gy = tgt_bbox[None, :, 0], tgt_bbox[None, :, 1]  # (1,G)
#     gw = tgt_bbox[None, :, 2].clamp(min=eps)
#     gh = tgt_bbox[None, :, 3].clamp(min=eps)

#     # tr(Sg^-1 Se) = pw^2/gw^2 + ph^2/gh^2
#     trace = (pw ** 2) / (gw ** 2) + (ph ** 2) / (gh ** 2)
#     # Mahalanobis: Sg^-1 = diag(4/gw^2, 4/gh^2)
#     maha = 4.0 * ((px - gx) ** 2) / (gw ** 2) + 4.0 * ((py - gy) ** 2) / (gh ** 2)
#     # log-det ratio: ln(|Sg|/|Se|) = 2*ln(gw*gh / (pw*ph))
#     logdet = 2.0 * (torch.log(gw * gh) - torch.log(pw * ph))

#     kld = 0.5 * (trace + maha + logdet) - 1.0
#     return kld.clamp(min=0.0)


# def gaussian_similarity(out_bbox: torch.Tensor, tgt_bbox: torch.Tensor) -> torch.Tensor:
#     """Bounded similarity in (0, 1]:  exp(-sqrt(KLD)).
#     sqrt() flattens the heavy tail of KLD so the exp() does not collapse
#     moderately-distant pairs to exactly 0 (keeps ranking signal alive).
#     """
#     return torch.exp(-torch.sqrt(gaussian_kld(out_bbox, tgt_bbox) + 1e-7))


# # ---------------------------------------------------------------------------
# # Paired (elementwise) versions — for use in criterion losses (MAL / FGL),
# # where pred boxes are already matched 1-1 with GT boxes.
# # ---------------------------------------------------------------------------

# def gaussian_kld_paired(src: torch.Tensor, tgt: torch.Tensor,
#                         eps: float = 1e-7) -> torch.Tensor:
#     """Elementwise KL( N_src || N_tgt ) for matched box pairs. (N,4) cxcywh -> (N,)"""
#     pw = src[:, 2].clamp(min=eps); ph = src[:, 3].clamp(min=eps)
#     gw = tgt[:, 2].clamp(min=eps); gh = tgt[:, 3].clamp(min=eps)
#     trace  = (pw ** 2) / (gw ** 2) + (ph ** 2) / (gh ** 2)
#     maha   = 4.0 * ((src[:, 0] - tgt[:, 0]) ** 2) / (gw ** 2) \
#            + 4.0 * ((src[:, 1] - tgt[:, 1]) ** 2) / (gh ** 2)
#     logdet = 2.0 * (torch.log(gw * gh) - torch.log(pw * ph))
#     return (0.5 * (trace + maha + logdet) - 1.0).clamp(min=0.0)


# def gaussian_similarity_paired(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
#     """Elementwise exp(-sqrt(KLD)) in (0, 1] for matched pairs."""
#     return torch.exp(-torch.sqrt(gaussian_kld_paired(src, tgt) + 1e-7))


# def blended_quality_paired(src: torch.Tensor, tgt: torch.Tensor,
#                            ious: torch.Tensor,
#                            blend='adaptive',
#                            s_small: float = 0.03,
#                            s_large: float = 0.09) -> torch.Tensor:
#     """Scale-adaptive quality  q = beta*IoU + (1-beta)*GaussSim  for matched pairs.

#     blend:
#       - 'adaptive' : beta grows linearly with GT size s = sqrt(w*h) (normalized
#                      coords), from 0.2 at s<=s_small to 0.8 at s>=s_large.
#                      With ~1100px input, s_small/s_large ~ 32px/96px (COCO
#                      small/medium boundary). Tiny boxes lean on the smooth
#                      Gaussian signal; large boxes keep the true-overlap IoU.
#       - float in [0,1] : fixed beta (1.0 = pure IoU = original behaviour).
#     """
#     gauss = gaussian_similarity_paired(src, tgt)
#     if blend == 'adaptive':
#         s = torch.sqrt((tgt[:, 2] * tgt[:, 3]).clamp(min=0))
#         beta = (0.2 + 0.6 * ((s - s_small) / (s_large - s_small)).clamp(0, 1))
#     else:
#         beta = float(blend)
#         if beta >= 1.0:
#             return ious
#     return beta * ious + (1.0 - beta) * gauss


# class HungarianMatcher(nn.Module):
#     """Bipartite matcher between predictions and ground-truth boxes."""

#     def __init__(
#         self,
#         weight_dict: dict,
#         use_focal_loss: bool = True,
#         alpha: float = 0.25,
#         gamma: float = 2.0,
#         change_matcher: bool = False,
#         iou_order_alpha: float = 1.0,
#         matcher_change_epoch: int = 10000,
#         gauss_iou_blend: float = 0.5,   # beta: 1.0 = pure IoU (original), 0.0 = pure Gaussian
#     ):
#         super().__init__()
#         self.cost_class = weight_dict['cost_class']
#         self.cost_bbox  = weight_dict['cost_bbox']
#         self.cost_giou  = weight_dict['cost_giou']
#         # ODLTM Gaussian cost weight (L_g in ETDMOT Eq. (1)). 0.0 disables.
#         self.cost_gauss = weight_dict.get('cost_gauss', 0.0)
#         self.use_focal_loss = use_focal_loss
#         self.alpha = alpha
#         self.gamma = gamma
#         self.change_matcher       = change_matcher
#         self.iou_order_alpha      = iou_order_alpha
#         self.matcher_change_epoch = matcher_change_epoch
#         self.gauss_iou_blend      = gauss_iou_blend
#         assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0

#     @torch.no_grad()
#     def forward(self, outputs: Dict[str, torch.Tensor], targets, epoch: int = 0, **kwargs):
#         bs, num_queries = outputs['pred_logits'].shape[:2]

#         if self.use_focal_loss:
#             out_prob = F.sigmoid(outputs['pred_logits'].flatten(0, 1))
#         else:
#             out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)

#         out_bbox = outputs['pred_boxes'].flatten(0, 1)
#         tgt_ids  = torch.cat([v['labels'] for v in targets])
#         tgt_bbox = torch.cat([v['boxes']  for v in targets])

#         if self.change_matcher and epoch >= self.matcher_change_epoch:
#             # --- IoU-order branch, FIXED: blend IoU with Gaussian similarity ---
#             class_score = out_prob[:, tgt_ids]
#             bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
#             if self.gauss_iou_blend < 1.0:
#                 gauss_sim = gaussian_similarity(out_bbox, tgt_bbox)
#                 quality = (self.gauss_iou_blend * bbox_iou
#                            + (1.0 - self.gauss_iou_blend) * gauss_sim)
#             else:
#                 quality = bbox_iou      # original behaviour
#             C = (-1) * (class_score * torch.pow(quality, self.iou_order_alpha))
#         else:
#             if self.use_focal_loss:
#                 p = out_prob[:, tgt_ids]
#                 neg_cost = (1 - self.alpha) * (p ** self.gamma) * (-(1 - p + 1e-8).log())
#                 pos_cost = self.alpha * ((1 - p) ** self.gamma) * (-(p + 1e-8).log())
#                 cost_class = pos_cost - neg_cost
#             else:
#                 cost_class = -out_prob[:, tgt_ids]

#             cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
#             cost_giou = -generalized_box_iou(
#                 box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
#             C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

#             # --- ODLTM Gaussian cost (L_g): smooth signal where GIoU saturates ---
#             if self.cost_gauss > 0:
#                 cost_gauss = 1.0 - gaussian_similarity(out_bbox, tgt_bbox)  # in [0,1)
#                 C = C + self.cost_gauss * cost_gauss

#         C = C.view(bs, num_queries, -1).cpu()
#         C = torch.nan_to_num(C, nan=1.0)
#         sizes = [len(v['boxes']) for v in targets]
#         indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
#         indices = [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices_pre
#         ]
#         return {'indices': indices}










"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

# from ..core import register
import numpy as np



# @register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score
            class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]

            # # Compute iou
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list