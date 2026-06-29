# """
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modules to compute the matching cost and solve the corresponding LSAP.

# Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from scipy.optimize import linear_sum_assignment
# from typing import Dict

# from ..ops.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

# # from ..core import register
# import numpy as np



# # @register()
# class HungarianMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network

#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """

#     __share__ = ['use_focal_loss', ]

#     def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
#                 change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000):
#         """Creates the matcher

#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
#             cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
#         """
#         super().__init__()
#         self.cost_class = weight_dict['cost_class']
#         self.cost_bbox = weight_dict['cost_bbox']
#         self.cost_giou = weight_dict['cost_giou']

#         self.change_matcher = change_matcher
#         self.iou_order_alpha = iou_order_alpha
#         self.matcher_change_epoch = matcher_change_epoch
#         if self.change_matcher:
#             print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

#         self.use_focal_loss = use_focal_loss
#         self.alpha = alpha
#         self.gamma = gamma

#         assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

#     @torch.no_grad()
#     def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
#         """ Performs the matching

#         Params:
#             outputs: This is a dict that contains at least these entries:
#                  "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
#                  "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                  "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                            objects in the target) containing the class labels
#                  "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#         bs, num_queries = outputs["pred_logits"].shape[:2]

#         # We flatten to compute the cost matrices in a batch
#         if self.use_focal_loss:
#             out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
#         else:
#             out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

#         out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

#         # Also concat the target labels and boxes
#         tgt_ids = torch.cat([v["labels"] for v in targets])
#         tgt_bbox = torch.cat([v["boxes"] for v in targets])

#         if self.change_matcher and epoch >= self.matcher_change_epoch:
#             # Compute the class_score
#             class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]

#             # # Compute iou
#             bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

#             # Final cost matrix
#             C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
#         else:
#             # Compute the classification cost. Contrary to the loss, we don't use the NLL,
#             # but approximate it in 1 - proba[target class].
#             # The 1 is a constant that doesn't change the matching, it can be ommitted.
#             if self.use_focal_loss:
#                 out_prob = out_prob[:, tgt_ids]
#                 neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
#                 pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
#                 cost_class = pos_cost_class - neg_cost_class
#             else:
#                 cost_class = -out_prob[:, tgt_ids]

#             # Compute the L1 cost between boxes
#             cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

#             # Compute the giou cost betwen boxes
#             cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

#             # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
#             C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

#         C = C.view(bs, num_queries, -1).cpu()

#         sizes = [len(v["boxes"]) for v in targets]
#         C = torch.nan_to_num(C, nan=1.0)
#         indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
#         indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

#         # Compute topk indices
#         if return_topk:
#             return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

#         return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

#     def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
#         indices_list = []
#         # C_original = C.clone()
#         for i in range(k):
#             indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
#             indices_list.append([
#                 (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#                 for i, j in indices_k
#             ])
#             for c, idx_k in zip(C.split(sizes, -1), indices_k):
#                 idx_k = np.stack(idx_k)
#                 c[:, idx_k] = 1e6
#         indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
#                         torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
#         # C.copy_(C_original)








"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.

[MODIFIED] Matcher gờ hỗ trợ SI-WBD trong matching cost, NHẤT QUÁN với criterion:
  - box_reg_mode='replace' : overlap = SI-WBD (bỏ GIoU)
  - box_reg_mode='add'     : overlap = GIoU + SI-WBD (hai tín hiệu cộng riêng)
  - box_reg_mode='blend'   : overlap = size-gated (1-λ)·GIoU + λ·SI-WBD
                             (mặc định; vật nhỏ -> SI-WBD, vật lớn -> GIoU)
Không còn zero GIoU một cách cứng nhắc -> matcher và loss nói cùng một ngôn ngữ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from ..ops.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from .siwbd import gaussian_w2_sq  # hàm tính 2-Wasserstein^2 closed-form
import numpy as np


class HungarianMatcher(nn.Module):
    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                 change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000):
        super().__init__()
        self.cost_class = weight_dict.get('cost_class', 1.0)
        self.cost_bbox  = weight_dict.get('cost_bbox', 5.0)

        # ----- SI-WBD config (đọc từ weight_dict; mặc định an toàn) -----
        self.use_siwbd = weight_dict.get('use_siwbd', False)
        self.siwbd_C   = weight_dict.get('siwbd_C', 0.5)

        # [MODIFIED] Luôn giữ cost_giou (không zero nữa). Khi không dùng SI-WBD
        # thì cost_siwbd đơn giản là không được tham chiếu.
        self.cost_giou  = weight_dict.get('cost_giou', 2.0)
        self.cost_siwbd = weight_dict.get('cost_siwbd', 2.0)

        # [NEW] Chế độ trộn overlap, mirror đúng criterion.box_reg_mode.
        self.box_reg_mode = weight_dict.get('box_reg_mode', 'blend')
        assert self.box_reg_mode in ('add', 'replace', 'blend'), \
            f"box_reg_mode must be add|replace|blend, got {self.box_reg_mode}"

        # [NEW] Tham số size-gate cho blend (mirror criterion).
        self.siwbd_beta         = weight_dict.get('siwbd_beta', 1.0)
        self.siwbd_logstd_floor = weight_dict.get('siwbd_logstd_floor', 0.5)

        self.change_matcher       = change_matcher
        self.iou_order_alpha      = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = "
                  f"{iou_order_alpha} at epoch {matcher_change_epoch}")
        if self.use_siwbd:
            print(f"[Matcher] SI-WBD ON | mode={self.box_reg_mode} | "
                  f"C={self.siwbd_C} cost_siwbd={self.cost_siwbd} cost_giou={self.cost_giou}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    # -----------------------------------------------------------------
    # [NEW] Overlap cost dùng chung cho 3 mode. Mọi ma trận đều quy ước
    #       "thấp hơn = khớp tốt hơn" (dạng distance), để blend với SI-WBD
    #       cùng thang đo (giống criterion: giou = 1 - GIoU, siwbd in (0,1)).
    # -----------------------------------------------------------------
    def _overlap_cost(self, out_bbox: torch.Tensor, tgt_bbox: torch.Tensor) -> torch.Tensor:
        # GIoU ở dạng distance: 1 - GIoU  -> [N_q, N_gt], thấp = chồng tốt
        giou_cost = 1.0 - generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # SI-WBD cost: 1 - exp(-W2^2 / (C * area_t)) -> (0,1), thấp = gần
        w2_sq  = gaussian_w2_sq(out_bbox.unsqueeze(1), tgt_bbox.unsqueeze(0))   # [N_q, N_gt]
        area_t = (tgt_bbox[:, 2].clamp(min=0) * tgt_bbox[:, 3].clamp(min=0)).unsqueeze(0)
        norm   = self.siwbd_C * area_t + 1e-7
        siwbd_cost = 1.0 - torch.exp(-w2_sq / norm)                              # [N_q, N_gt]

        if self.box_reg_mode == 'replace':
            # Chỉ SI-WBD gánh slot overlap.
            return self.cost_siwbd * siwbd_cost

        if self.box_reg_mode == 'add':
            # GIoU và SI-WBD là hai tín hiệu cộng riêng, mỗi cái một trọng số.
            return self.cost_giou * giou_cost + self.cost_siwbd * siwbd_cost

        # ----- blend: size-gated convex blend, gate theo GT (per-column) -----
        # λ tự hiệu chỉnh trong không gian log-area: tách tại trung vị batch,
        # độ sắc tỉ lệ với độ phân tán kích thước. Vật nhỏ -> λ→1 (SI-WBD).
        area = (tgt_bbox[:, 2].clamp(min=0) * tgt_bbox[:, 3].clamp(min=0))       # [N_gt]
        la   = torch.log(area.clamp(min=1e-8))                                   # [N_gt]
        if la.numel() == 0:
            lam = la                                                            # rỗng, an toàn
        else:
            center = la.median()
            scale  = la.std() if la.numel() > 1 else torch.as_tensor(
                float('nan'), device=la.device)
            if (not torch.isfinite(scale)) or scale < self.siwbd_logstd_floor:
                scale = torch.as_tensor(self.siwbd_logstd_floor, device=la.device)
            lam = torch.sigmoid((center - la) / (self.siwbd_beta * scale))       # [N_gt]
        overlap = (1.0 - lam)[None, :] * giou_cost + lam[None, :] * siwbd_cost   # [N_q, N_gt]
        # Reuse trọng số cost_giou cho slot overlap đã trộn (mirror criterion,
        # nơi blend được ghi vào loss_giou slot weight 2.0).
        return self.cost_giou * overlap

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids  = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # NOTE: nhánh này dùng IoU thuần -> "vực" IoU cho vật tí hon.
            # Chỉ kích hoạt ở cuối training (matcher_change_epoch lớn).
            class_score = out_prob[:, tgt_ids]
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # [MODIFIED] Phân luồng overlap cost theo cấu hình.
            if self.use_siwbd:
                overlap_cost = self._overlap_cost(out_bbox, tgt_bbox)
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + overlap_cost
            else:
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                   for i, j in indices_pre]

        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(
                C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] \
                if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                         torch.cat([indices_list[i][j][1] for i in range(k)], dim=0))
                        for j in range(len(sizes))]
        return indices_list