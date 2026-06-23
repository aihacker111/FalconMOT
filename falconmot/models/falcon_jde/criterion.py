# """
# FalconJDECriterion — DETR-style detection criterion + per-class ArcFace ReID loss.

# Detection losses (DETR-style):
#     loss_cls   — sigmoid focal classification
#     loss_bbox  — L1 box regression
#     loss_giou  — GIoU box regression

# ReID losses:
#     loss_reid  — ArcFace CE per class, optionally + TripletLoss
# """

# import math
# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision

# from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
# from .matcher import HungarianMatcher
# from .feat_fusion import build_center_heatmaps, gaussian_focal_loss

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _get_world_size():
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         return torch.distributed.get_world_size()
#     return 1

# def _is_dist():
#     return torch.distributed.is_available() and torch.distributed.is_initialized()


# class ArcFace(nn.Module):
#     """
#     ArcFace Margin Loss được tinh chỉnh cho Tracking Drone.
#     Hạ s (scale) và m (margin) giúp model dễ hội tụ hơn khi vật thể nhỏ và nhòe.
#     """
#     def __init__(self, in_features: int, num_ids: int, s: float = 16.0, m: float = 0.15):
#         super().__init__()
#         self.s = s
#         self.weight = nn.Parameter(torch.FloatTensor(num_ids, in_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th    = math.cos(math.pi - m)
#         self.mm    = math.sin(math.pi - m) * m

#     def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         # X đã được qua LayerNorm Bottleneck nên normalize ở đây sẽ rất ổn định
#         cosine = F.linear(F.normalize(x), F.normalize(self.weight))
#         sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
#         phi    = cosine * self.cos_m - sine * self.sin_m
#         phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
#         one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1.0)
#         return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


# class TripletLoss(nn.Module):
#     def __init__(self, margin: float = 0.3):
#         super().__init__()
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         n = inputs.size(0)
#         if n < 2:
#             return inputs.sum() * 0
#         dist = torch.pow(inputs, 2).sum(1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
#         dist = dist.clamp(min=1e-12).sqrt()
#         # Batch-hard mining — VECTORIZED. Bản cũ dùng vòng `for i in range(n)`
#         # với boolean-index `dist[i][mask[i]]` (kích thước phụ thuộc dữ liệu trên
#         # GPU) → ép đồng bộ CPU↔GPU mỗi anchor; với class dày đặc (n vài trăm) là
#         # nghẽn chính khi bật --tri. Phiên bản này không vòng lặp, không sync,
#         # cho kết quả số HỆT bản cũ.
#         mask_pos = targets.unsqueeze(0).eq(targets.unsqueeze(1))   # (n, n), gồm self
#         mask_neg = ~mask_pos
#         dist_ap = dist.masked_fill(mask_neg, float('-inf')).max(dim=1).values  # hardest pos
#         dist_an = dist.masked_fill(mask_pos, float('inf')).min(dim=1).values   # hardest neg
#         valid   = mask_neg.any(dim=1)   # chỉ giữ anchor có ≥1 negative (như bản cũ)
#         dist_ap = dist_ap[valid]
#         dist_an = dist_an[valid]
#         if dist_ap.numel() == 0:
#             return inputs.sum() * 0
#         y = torch.ones_like(dist_an)
#         return self.ranking_loss(dist_an, dist_ap, y)


# # ---------------------------------------------------------------------------
# # Main criterion
# # ---------------------------------------------------------------------------

# class FalconJDECriterion(nn.Module):
#     """
#     Combined detection + ReID criterion for FalconJDE.

#     detection losses: 'focal' (-> loss_cls) and 'boxes' (-> loss_bbox + loss_giou)
#     """

#     def __init__(
#         self,
#         matcher:             HungarianMatcher,
#         num_classes:         int,
#         nid_dict:            dict,
#         reid_dim:            int   = 128,
#         weight_dict:         dict  = None,
#         losses:              tuple = ('focal', 'boxes'),
#         alpha:               float = 0.25,
#         gamma:               float = 2.0,
#         reg_max:             int   = 32,
#         boxes_weight_format: str   = None,
#         use_uni_set:         bool  = True,
#         use_reid:            bool  = False,
#         id_weight:           float = 1.0,
#         use_triplet:         bool  = False,
#         use_arcface:         bool  = True,
#         s_det_init:          float = 2.5,
#         s_id_init:           float = 1.85,
#     ):
#         super().__init__()
#         self.matcher             = matcher
#         self.num_classes         = num_classes
#         self.nid_dict            = nid_dict
#         self.reid_dim            = reid_dim
#         self.losses              = losses
#         self.alpha               = alpha
#         self.gamma               = gamma
#         self.reg_max             = reg_max
#         self.boxes_weight_format = boxes_weight_format
#         self.use_uni_set         = use_uni_set
#         self.use_reid            = use_reid
#         self.id_weight           = id_weight
#         self.use_triplet         = use_triplet
#         self.use_arcface         = use_arcface

#         self.weight_dict = weight_dict or {
#             'loss_cls':  2.0,
#             'loss_bbox': 5.0,
#             'loss_giou': 2.0,
#             'loss_s4_aux': 1.0
#         }

#         # Per-class classifiers: ArcFace (margin-based) or Linear (plain CE+Triplet)
#         if use_reid:
#             if use_arcface:
#                 self.classifiers = nn.ModuleDict({
#                     str(cls_id): ArcFace(reid_dim, nid)
#                     for cls_id, nid in nid_dict.items()
#                 })
#             else:
#                 self.linear_classifiers = nn.ModuleDict({
#                     str(cls_id): nn.Linear(reid_dim, nid, bias=False)
#                     for cls_id, nid in nid_dict.items()
#                 })
#             self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
#             self.triplet = TripletLoss(margin=0.3)

#             # Per-class embedding scale (FairMOT / AMOT recipe). Sharpens the
#             # softmax temperature according to the number of identities in the
#             # class, so plain CE stays well-conditioned across classes whose
#             # nID differ by orders of magnitude.
#             self.emb_scale_dict = {
#                 cls_id: math.sqrt(2) * math.log(max(nid - 1, 2))
#                 for cls_id, nid in nid_dict.items()
#             }

#         # Learnable homoscedastic-uncertainty weights (Kendall et al.) that
#         # balance detection vs ReID automatically. Init should be ≈ log of the
#         # initial raw loss magnitudes (s* = log L), so each weighted loss
#         # exp(-s)·L starts near 1. Defaults are tuned for this DETR-JDE loss
#         # scale (L_det≈12, L_reid≈6); FairMOT's −1.85/−1.05 assume L<1 and are
#         # wrong here — they leave s too far from the optimum to ever reach it.
#         self.s_det = nn.Parameter(torch.tensor(float(s_det_init)))
#         self.s_id  = nn.Parameter(torch.tensor(float(s_id_init)))

#     # ------------------------------------------------------------------
#     # Dense heatmap loss (CenterNet-style, on S4 feature map)
#     # ------------------------------------------------------------------
#     # Detection losses (mirror DEIMCriterion)
#     # ------------------------------------------------------------------

#     def loss_labels_focal(self, outputs, targets, indices, num_boxes):
#         src_logits = outputs['pred_logits']
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#         target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()
#         loss = torchvision.ops.sigmoid_focal_loss(
#             src_logits, target, self.alpha, self.gamma, reduction='none')
#         loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
#         return {'loss_cls': loss}

#     # def loss_labels_focal(self, outputs, targets, indices, num_boxes):
#     #     """Classification loss — IoU-aware BCE (giống RF-DETR ia_bce_loss, nhánh mặc định của RF-DETR).

#     #     Trọng số positive được điều biến bởi IoU giữa box dự đoán và box GT:
#     #         t = prob^alpha * iou^(1 - alpha)   (clamp >= 0.01, detach)
#     #     và BCE được viết lại bằng logsigmoid cho ổn định số học, chuẩn hoá bằng num_boxes.
#     #     """
#     #     src_logits = outputs['pred_logits']
#     #     idx = self._get_src_permutation_idx(indices)
#     #     target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

#     #     alpha = self.alpha
#     #     gamma = self.gamma

#     #     src_boxes = outputs['pred_boxes'][idx]
#     #     target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#     #     iou_targets = torch.diag(box_iou(
#     #         box_cxcywh_to_xyxy(src_boxes.detach()),
#     #         box_cxcywh_to_xyxy(target_boxes),
#     #     )[0])
#     #     pos_ious = iou_targets.clone().detach()

#     #     prob = src_logits.sigmoid()
#     #     # khởi tạo positive/negative weights
#     #     pos_weights = torch.zeros_like(src_logits)
#     #     neg_weights = prob ** gamma

#     #     pos_ind = [i for i in idx]
#     #     pos_ind.append(target_classes_o)

#     #     t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
#     #     t = torch.clamp(t, 0.01).detach()

#     #     pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
#     #     neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)
#     #     # tương đương loss = -pos_weights*log(prob) - neg_weights*log(1-prob),
#     #     # viết lại bằng logsigmoid cho ổn định số học
#     #     loss = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
#     #     loss = loss.sum() / num_boxes
#     #     return {'loss_cls': loss}

#     def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
#         loss_giou = 1 - torch.diag(generalized_box_iou(
#             box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
#         if boxes_weight is not None:
#             loss_giou = loss_giou * boxes_weight
#         return {
#             'loss_bbox': loss_bbox.sum() / num_boxes,
#             'loss_giou': loss_giou.sum() / num_boxes,
#         }

#     # ------------------------------------------------------------------
#     # ReID loss
#     # ------------------------------------------------------------------

#     def loss_reid(self, outputs, targets, indices) -> dict:
#         """Per-class ReID loss on the queries matched to ground-truth objects.

#         Two representations come from the head (BNNeck principle):
#           • ``pred_reid``     — post-neck embedding → CE / ArcFace + inference
#           • ``pred_reid_raw`` — pre-neck embedding  → TripletLoss

#         For the (recommended) plain-CE path we follow FairMOT / AMOT exactly:
#         scale * L2-normalize the embedding, then a per-class linear classifier.
#         Inference L2-normalizes the *same* post-neck vector, so the direction
#         CE optimises is the direction the tracker matches on (cosine).
#         """
#         if 'pred_reid' not in outputs:
#             return {}

#         pred_reid = outputs['pred_reid']                          # (B, N, D) post-neck
#         pred_reid_raw = outputs.get('pred_reid_raw', pred_reid)   # (B, N, D) pre-neck
#         dev = pred_reid.device
#         reid_loss = pred_reid.sum() * 0.0

#         # Gather matched embeddings per ReID class.
#         cls_emb     = {cid: [] for cid in self.nid_dict}   # post-neck (CE)
#         cls_emb_raw = {cid: [] for cid in self.nid_dict}   # pre-neck  (triplet)
#         cls_ids     = {cid: [] for cid in self.nid_dict}

#         for b_idx, (src_idx, tgt_idx) in enumerate(indices):
#             if len(src_idx) == 0:
#                 continue
#             t       = targets[b_idx]
#             src_idx = src_idx.to(dev)
#             tgt_idx = tgt_idx.to(dev)
#             labels  = t['labels'].to(dev)[tgt_idx]
#             tids    = t['track_ids'].to(dev)[tgt_idx]
#             valid   = tids >= 0
#             if not valid.any():
#                 continue

#             src_v   = src_idx[valid]
#             emb_b   = pred_reid[b_idx][src_v]
#             emb_b_r = pred_reid_raw[b_idx][src_v]
#             lbl_b   = labels[valid]
#             ids_b   = tids[valid]

#             for cls_id in self.nid_dict:
#                 mask = (lbl_b == cls_id)
#                 if not mask.any():
#                     continue
#                 cls_emb[cls_id].append(emb_b[mask])
#                 cls_emb_raw[cls_id].append(emb_b_r[mask])
#                 cls_ids[cls_id].append(ids_b[mask])

#         n_active = 0
#         for cls_id in self.nid_dict:
#             if not cls_emb[cls_id]:
#                 continue
#             emb     = torch.cat(cls_emb[cls_id], dim=0)        # (n, D) post-neck
#             emb_raw = torch.cat(cls_emb_raw[cls_id], dim=0)    # (n, D) pre-neck
#             ids     = torch.cat(cls_ids[cls_id], dim=0)        # (n,)

#             # ---- classification (ID) loss ----
#             if self.use_arcface:
#                 logits = self.classifiers[str(cls_id)](emb, ids)   # ArcFace normalizes internally
#             else:
#                 emb_id = self.emb_scale_dict[cls_id] * F.normalize(emb, dim=1)
#                 logits = self.linear_classifiers[str(cls_id)](emb_id)
#             reid_loss = reid_loss + self.ce_loss(logits, ids)

#             # ---- optional metric (triplet) loss on the un-normalized vector ----
#             if self.use_triplet and emb_raw.shape[0] >= 2:
#                 reid_loss = reid_loss + self.triplet(emb_raw, ids)

#             n_active += 1

#         if n_active > 1:
#             reid_loss = reid_loss / n_active
#         return {'loss_reid': reid_loss}

#     # def loss_s4_aux(self, outputs, targets, indices, num_boxes):
#     #     """
#     #     TẠO MỤC TIÊU ĐỘC LẬP & TÍNH LOSS CHO NHÁNH PHỤ STRIDE-4 (S4 AUXILIARY HEAD)
#     #     Cơ chế tiêm mãnh liệt Gradient để duy trì độ nhạy cho các vật thể siêu nhỏ.
#     #     """
#     #     losses = {'loss_s4_aux': torch.tensor(0.0, device=outputs['pred_logits'].device)}
#     #     if 'pred_s4_aux' not in outputs:
#     #         return losses

#     #     pred_heatmap = outputs['pred_s4_aux'] # Kích thước hình học: (B, 1, H/4, W/4)
#     #     B, _, H_s4, W_s4 = pred_heatmap.shape
        
#     #     # Khởi tạo bản đồ nhị phân mục tiêu rỗng
#     #     target_heatmap = torch.zeros_like(pred_heatmap)

#     #     for b in range(B):
#     #         tgt_boxes = targets[b]['boxes'] # Lấy tọa độ chuẩn hóa [cx, cy, w, h] trong khoảng [0, 1]
#     #         if len(tgt_boxes) == 0:
#     #             continue

#     #         # Ánh xạ tọa độ chuẩn hóa về chiều không gian lưới rời rạc của bản đồ Stride-4
#     #         cx = tgt_boxes[:, 0] * W_s4
#     #         cy = tgt_boxes[:, 1] * H_s4
#     #         w  = tgt_boxes[:, 2] * W_s4
#     #         h  = tgt_boxes[:, 3] * H_s4

#     #         x1 = (cx - w * 0.5).long().clamp(0, W_s4 - 1)
#     #         y1 = (cy - h * 0.5).long().clamp(0, H_s4 - 1)
#     #         x2 = (cx + w * 0.5).long().clamp(0, W_s4 - 1)
#     #         y2 = (cy + h * 0.5).long().clamp(0, H_s4 - 1)

#     #         # Điền giá trị 1.0 vào các pixel nằm bên trong vùng bounding box của vật thể
#     #         for idx in range(len(tgt_boxes)):
#     #             target_heatmap[b, 0, y1[idx]:y2[idx] + 1, x1[idx]:x2[idx] + 1] = 1.0

#     #     # Áp dụng hàm phạt BCE Loss có trọng số ổn định số học cao
#     #     losses['loss_s4_aux'] = F.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap, reduction='mean')
#     #     return losses
#     def loss_s4_aux(self, outputs, targets, indices, num_boxes):
#         dev = outputs['pred_logits'].device
#         losses = {'loss_s4_aux': torch.tensor(0.0, device=dev)}
#         if 'pred_s4_aux' not in outputs:
#             return losses
#         pred = outputs['pred_s4_aux']                 # [B,1,H4,W4] logit
#         _, _, H, W = pred.shape
#         gt = build_center_heatmaps(targets, H, W, dev)
#         losses['loss_s4_aux'] = gaussian_focal_loss(pred, gt)
#         return losses
#     # ------------------------------------------------------------------
#     # Mask loss (AMOT DiceLoss adaptation for DETR query-based paradigm)
#     # ------------------------------------------------------------------

#     # ------------------------------------------------------------------
#     # Matching utilities
#     # ------------------------------------------------------------------

#     def _get_src_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx   = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_go_indices(self, indices, aux_indices_list):
#         for aux in aux_indices_list:
#             indices = [(torch.cat([i1[0], i2[0]]), torch.cat([i1[1], i2[1]]))
#                        for i1, i2 in zip(indices, aux)]
#         results = []
#         for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
#             unique, counts = torch.unique(ind, return_counts=True, dim=0)
#             sort_idx = torch.argsort(counts, descending=True)
#             col2row = {}
#             for pair in unique[sort_idx]:
#                 r, c = pair[0].item(), pair[1].item()
#                 if r not in col2row:
#                     col2row[r] = c
#             fr = torch.tensor(list(col2row.keys()),   device=ind.device)
#             fc = torch.tensor(list(col2row.values()), device=ind.device)
#             results.append((fr.long(), fc.long()))
#         return results

#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'focal': self.loss_labels_focal,   # -> loss_cls
#             'boxes': self.loss_boxes,          # -> loss_bbox + loss_giou
#             's4_aux': self.loss_s4_aux
#         }
#         assert loss in loss_map, f'Unknown loss: {loss}'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     def _get_meta(self, loss, outputs, targets, indices):
#         if self.boxes_weight_format is None:
#             return {}
#         idx = self._get_src_permutation_idx(indices)
#         src = outputs['pred_boxes'][idx]
#         tgt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
#         if self.boxes_weight_format == 'iou':
#             iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src.detach()), box_cxcywh_to_xyxy(tgt))[0])
#         elif self.boxes_weight_format == 'giou':
#             iou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src.detach()), box_cxcywh_to_xyxy(tgt)))
#         else:
#             raise ValueError(self.boxes_weight_format)
#         if loss == 'boxes':
#             return {'boxes_weight': iou}
#         return {}

#     @staticmethod
#     def get_cdn_matched_indices(dn_meta, targets):
#         dn_positive_idx = dn_meta['dn_positive_idx']
#         dn_num_group    = dn_meta['dn_num_group']
#         num_gts  = [len(t['labels']) for t in targets]
#         device   = targets[0]['labels'].device
#         result   = []
#         for i, ng in enumerate(num_gts):
#             if ng > 0:
#                 gt_idx = torch.arange(ng, dtype=torch.int64, device=device).tile(dn_num_group)
#                 result.append((dn_positive_idx[i], gt_idx))
#             else:
#                 result.append((torch.zeros(0, dtype=torch.int64, device=device),
#                                torch.zeros(0, dtype=torch.int64, device=device)))
#         return result

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------

#     def forward(self, outputs, targets, epoch: int = 0):
#         outputs_no_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

#         # Main matching
#         indices = self.matcher(outputs_no_aux, targets, epoch=epoch)['indices']

#         # Aux matching for GO (union) set
#         aux_list     = outputs.get('aux_outputs', [])
#         pre_list     = [outputs['pre_outputs']] if 'pre_outputs' in outputs else []
#         enc_list     = outputs.get('enc_aux_outputs', [])
#         all_aux      = list(aux_list) + pre_list + list(enc_list)

#         cached_indices = []
#         for aux in all_aux:
#             cached_indices.append(self.matcher(aux, targets, epoch=epoch)['indices'])

#         indices_go = self._get_go_indices(indices, cached_indices) if cached_indices else indices

#         # num_boxes normalization
#         num_boxes = sum(len(t['labels']) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
#                                     device=next(iter(outputs.values())).device)
#         if _is_dist():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / _get_world_size(), min=1).item()

#         num_boxes_go = sum(len(x[0]) for x in indices_go)
#         num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
#                                        device=next(iter(outputs.values())).device)
#         if _is_dist():
#             torch.distributed.all_reduce(num_boxes_go)
#         num_boxes_go = torch.clamp(num_boxes_go / _get_world_size(), min=1).item()

#         losses = {}
#         _up = outputs.get('up')
#         _rs = outputs.get('reg_scale')

#         def _apply(out, tgts, idx_main, idx_go, nb, nb_go, suffix=''):
#             for loss in self.losses:
#                 # Tránh tính toán lặp lại nhánh S4 Aux trên các tầng phụ (aux layers/dn layers)
#                 if loss == 's4_aux' and suffix != '':
#                     continue
                
#                 use_go = self.use_uni_set and loss == 'boxes'
#                 idx_in = idx_go if use_go else idx_main
#                 nb_in  = nb_go  if use_go else nb
#                 meta   = self._get_meta(loss, out, tgts, idx_in)
#                 l_dict = self.get_loss(loss, out, tgts, idx_in, nb_in, **meta)
#                 l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
#                           for k in l_dict if k in self.weight_dict}
#                 if suffix:
#                     l_dict = {k + suffix: v for k, v in l_dict.items()}
#                 losses.update(l_dict)

#         # Áp dụng cho đầu ra chính (bao gồm cả loss_s4_aux nếu có trong self.losses)
#         _apply(outputs, targets, indices, indices_go, num_boxes, num_boxes_go)

#         # Aux detection losses
#         for i, aux in enumerate(aux_list):
#             aux = {**aux, 'up': _up, 'reg_scale': _rs}
#             _apply(aux, targets, cached_indices[i], indices_go,
#                    num_boxes, num_boxes_go, f'_aux_{i}')

#         if pre_list:
#             pre_idx = len(aux_list)
#             _apply(pre_list[0], targets, cached_indices[pre_idx], indices_go,
#                    num_boxes, num_boxes_go, '_pre')

#         # Enc aux losses — use zero labels when class-agnostic encoder
#         enc_start = len(aux_list) + len(pre_list)
#         class_agnostic = outputs.get('enc_meta', {}).get('class_agnostic', False)
#         if class_agnostic:
#             enc_targets = [{**t, 'labels': torch.zeros_like(t['labels'])} for t in targets]
#             orig_nc = self.num_classes
#             self.num_classes = 1
#         else:
#             enc_targets = targets
#         for i, aux in enumerate(enc_list):
#             _apply(aux, enc_targets, cached_indices[enc_start + i], indices_go,
#                    num_boxes, num_boxes_go, f'_enc_{i}')
#         if class_agnostic:
#             self.num_classes = orig_nc

#         # DN losses
#         if 'dn_outputs' in outputs:
#             indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
#             dn_nb = max(num_boxes * outputs['dn_meta']['dn_num_group'], 1)
#             for i, aux in enumerate(outputs['dn_outputs']):
#                 aux = {**aux, 'up': _up, 'reg_scale': _rs}
#                 _apply(aux, targets, indices_dn, indices_dn, dn_nb, dn_nb, f'_dn_{i}')
#             if 'dn_pre_outputs' in outputs:
#                 aux = {**outputs['dn_pre_outputs'], 'up': _up, 'reg_scale': _rs}
#                 _apply(aux, targets, indices_dn, indices_dn, dn_nb, dn_nb, '_dn_pre')

#         # ReID loss
#         if self.use_reid and self.id_weight > 0:
#             reid_dict = self.loss_reid(outputs, targets, indices)
#             losses.update({k: v * self.id_weight for k, v in reid_dict.items()})

#         losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

#         # ------------------------------------------------------------------
#         # Total loss — learnable uncertainty weighting (Kendall et al.).
#         # det_loss aggregates EVERY detection term (main + aux + dn + enc + s4)
#         # so the auxiliary supervision is preserved; only ReID is weighted
#         # separately. The (s_det + s_id) term is the regulariser that stops the
#         # weights collapsing to zero.
#         # ------------------------------------------------------------------
#         reid_loss = losses.get('loss_reid', None)
#         det_loss  = sum(v for k, v in losses.items() if k != 'loss_reid')

#         if self.use_reid and self.id_weight > 0 and reid_loss is not None:
#             total = (torch.exp(-self.s_det) * det_loss
#                      + torch.exp(-self.s_id) * reid_loss
#                      + (self.s_det + self.s_id))
#         else:
#             # total = torch.exp(-self.s_det) * det_loss + self.s_det
#             total = det_loss
#         total = total

#         # Detached scalars for logging only.
#         losses['loss_det'] = det_loss.detach()
#         losses['s_det']    = self.s_det.detach()
#         losses['s_id']     = self.s_id.detach()
#         losses['loss']     = total
#         return losses



"""
FalconJDECriterion — DETR-style detection criterion + per-class ArcFace ReID loss.

Detection losses (DETR-style):
    loss_cls   — sigmoid focal classification
    loss_bbox  — L1 box regression
    loss_giou  — GIoU box regression

ReID losses:
    loss_reid  — ArcFace CE per class, optionally + TripletLoss
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .matcher import HungarianMatcher
from .feat_fusion import build_center_heatmaps, gaussian_focal_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

def _is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class ArcFace(nn.Module):
    """
    ArcFace Margin Loss được tinh chỉnh cho Tracking Drone.
    Hạ s (scale) và m (margin) giúp model dễ hội tụ hơn khi vật thể nhỏ và nhòe.
    """
    def __init__(self, in_features: int, num_ids: int, s: float = 16.0, m: float = 0.15):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_ids, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # X đã được qua LayerNorm Bottleneck nên normalize ở đây sẽ rất ổn định
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = inputs.size(0)
        if n < 2:
            return inputs.sum() * 0
        dist = torch.pow(inputs, 2).sum(1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        # Batch-hard mining — VECTORIZED. Bản cũ dùng vòng `for i in range(n)`
        # với boolean-index `dist[i][mask[i]]` (kích thước phụ thuộc dữ liệu trên
        # GPU) → ép đồng bộ CPU↔GPU mỗi anchor; với class dày đặc (n vài trăm) là
        # nghẽn chính khi bật --tri. Phiên bản này không vòng lặp, không sync,
        # cho kết quả số HỆT bản cũ.
        mask_pos = targets.unsqueeze(0).eq(targets.unsqueeze(1))   # (n, n), gồm self
        mask_neg = ~mask_pos
        dist_ap = dist.masked_fill(mask_neg, float('-inf')).max(dim=1).values  # hardest pos
        dist_an = dist.masked_fill(mask_pos, float('inf')).min(dim=1).values   # hardest neg
        valid   = mask_neg.any(dim=1)   # chỉ giữ anchor có ≥1 negative (như bản cũ)
        dist_ap = dist_ap[valid]
        dist_an = dist_an[valid]
        if dist_ap.numel() == 0:
            return inputs.sum() * 0
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
                  # [n, D]
# ---------------------------------------------------------------------------
# Main criterion
# ---------------------------------------------------------------------------

class FalconJDECriterion(nn.Module):
    """
    Combined detection + ReID criterion for FalconJDE.

    detection losses: 'focal' (-> loss_cls) and 'boxes' (-> loss_bbox + loss_giou)
    """

    def __init__(
        self,
        matcher:             HungarianMatcher,
        num_classes:         int,
        nid_dict:            dict,
        reid_dim:            int   = 128,
        weight_dict:         dict  = None,
        losses:              tuple = ('mal', 'boxes'),
        alpha:               float = 0.25,
        gamma:               float = 2.0,
        reg_max:             int   = 32,
        boxes_weight_format: str   = None,
        use_uni_set:         bool  = True,
        use_reid:            bool  = False,
        id_weight:           float = 1.0,
        use_triplet:         bool  = False,
        use_arcface:         bool  = True,
        s_det_init:          float = 2.5,
        s_id_init:           float = 1.85,
        mal_alpha:           float = None,
        w_dense_ce: float = 0.5,     
        w_cons: float = 0.1
    ):
        super().__init__()
        self.matcher             = matcher
        self.num_classes         = num_classes
        self.nid_dict            = nid_dict
        self.reid_dim            = reid_dim
        self.losses              = losses
        self.alpha               = alpha
        self.gamma               = gamma
        self.reg_max             = reg_max
        self.boxes_weight_format = boxes_weight_format
        self.use_uni_set         = use_uni_set
        self.use_reid            = use_reid
        self.id_weight           = id_weight
        self.use_triplet         = use_triplet
        self.use_arcface         = use_arcface
        self.mal_alpha           = mal_alpha   # DEIM MAL: hệ số trọng số nhánh negative (None = theo repo gốc)
        self.w_dense_ce = w_dense_ce
        self.w_cons     = w_cons
        self.weight_dict = weight_dict or {
            'loss_mal':  1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
            'loss_s4_aux': 1.0
        }

        # Per-class classifiers: ArcFace (margin-based) or Linear (plain CE+Triplet)
        if use_reid:
            if use_arcface:
                self.classifiers = nn.ModuleDict({
                    str(cls_id): ArcFace(reid_dim, nid)
                    for cls_id, nid in nid_dict.items()
                })
            else:
                self.linear_classifiers = nn.ModuleDict({
                    str(cls_id): nn.Linear(reid_dim, nid, bias=False)
                    for cls_id, nid in nid_dict.items()
                })
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.triplet = TripletLoss(margin=0.3)

            # Per-class embedding scale (FairMOT / AMOT recipe). Sharpens the
            # softmax temperature according to the number of identities in the
            # class, so plain CE stays well-conditioned across classes whose
            # nID differ by orders of magnitude.
            self.emb_scale_dict = {
                cls_id: math.sqrt(2) * math.log(max(nid - 1, 2))
                for cls_id, nid in nid_dict.items()
            }

        # Learnable homoscedastic-uncertainty weights (Kendall et al.) that
        # balance detection vs ReID automatically. Init should be ≈ log of the
        # initial raw loss magnitudes (s* = log L), so each weighted loss
        # exp(-s)·L starts near 1. Defaults are tuned for this DETR-JDE loss
        # scale (L_det≈12, L_reid≈6); FairMOT's −1.85/−1.05 assume L<1 and are
        # wrong here — they leave s too far from the optimum to ever reach it.
        self.s_det = nn.Parameter(torch.tensor(float(s_det_init)))
        self.s_id  = nn.Parameter(torch.tensor(float(s_id_init)))

    # ------------------------------------------------------------------
    # Dense heatmap loss (CenterNet-style, on S4 feature map)
    # ------------------------------------------------------------------
    # Detection losses (mirror DEIMCriterion)
    # ------------------------------------------------------------------

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_cls': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """DEIM Matchability-Aware Loss (thay cho focal) — port nguyên từ repo DEIM.

        Ý tưởng: target phân loại mềm = IoU(box dự đoán, box GT)^gamma tại lớp GT
        (thay vì one-hot cứng). Mẫu khớp tốt (IoU cao) bị ép có score cao hơn ->
        siết tương quan score<->localization, đẩy AP ở IoU cao và vật nhỏ.

        Tự tính IoU nếu `values` không được truyền vào (an toàn với mọi điểm gọi).
        """
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()),
                              box_cxcywh_to_xyxy(tgt_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = (target_score_o.unsqueeze(-1) * target).pow(self.gamma)

        pred_score = F.sigmoid(src_logits).detach()
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score,
                                                  weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    # def loss_labels_focal(self, outputs, targets, indices, num_boxes):
    #     """Classification loss — IoU-aware BCE (giống RF-DETR ia_bce_loss, nhánh mặc định của RF-DETR).

    #     Trọng số positive được điều biến bởi IoU giữa box dự đoán và box GT:
    #         t = prob^alpha * iou^(1 - alpha)   (clamp >= 0.01, detach)
    #     và BCE được viết lại bằng logsigmoid cho ổn định số học, chuẩn hoá bằng num_boxes.
    #     """
    #     src_logits = outputs['pred_logits']
    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

    #     alpha = self.alpha
    #     gamma = self.gamma

    #     src_boxes = outputs['pred_boxes'][idx]
    #     target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    #     iou_targets = torch.diag(box_iou(
    #         box_cxcywh_to_xyxy(src_boxes.detach()),
    #         box_cxcywh_to_xyxy(target_boxes),
    #     )[0])
    #     pos_ious = iou_targets.clone().detach()

    #     prob = src_logits.sigmoid()
    #     # khởi tạo positive/negative weights
    #     pos_weights = torch.zeros_like(src_logits)
    #     neg_weights = prob ** gamma

    #     pos_ind = [i for i in idx]
    #     pos_ind.append(target_classes_o)

    #     t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
    #     t = torch.clamp(t, 0.01).detach()

    #     pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
    #     neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)
    #     # tương đương loss = -pos_weights*log(prob) - neg_weights*log(1-prob),
    #     # viết lại bằng logsigmoid cho ổn định số học
    #     loss = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
    #     loss = loss.sum() / num_boxes
    #     return {'loss_cls': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

    # ------------------------------------------------------------------
    # ReID loss
    # ------------------------------------------------------------------

    # def loss_reid(self, outputs, targets, indices) -> dict:
    #     """Per-class ReID loss on the queries matched to ground-truth objects.

    #     Two representations come from the head (BNNeck principle):
    #       • ``pred_reid``     — post-neck embedding → CE / ArcFace + inference
    #       • ``pred_reid_raw`` — pre-neck embedding  → TripletLoss

    #     For the (recommended) plain-CE path we follow FairMOT / AMOT exactly:
    #     scale * L2-normalize the embedding, then a per-class linear classifier.
    #     Inference L2-normalizes the *same* post-neck vector, so the direction
    #     CE optimises is the direction the tracker matches on (cosine).
    #     """
    #     if 'pred_reid' not in outputs:
    #         return {}

    #     pred_reid = outputs['pred_reid']                          # (B, N, D) post-neck
    #     pred_reid_raw = outputs.get('pred_reid_raw', pred_reid)   # (B, N, D) pre-neck
    #     dev = pred_reid.device
    #     reid_loss = pred_reid.sum() * 0.0

    #     # Gather matched embeddings per ReID class.
    #     cls_emb     = {cid: [] for cid in self.nid_dict}   # post-neck (CE)
    #     cls_emb_raw = {cid: [] for cid in self.nid_dict}   # pre-neck  (triplet)
    #     cls_ids     = {cid: [] for cid in self.nid_dict}

    #     for b_idx, (src_idx, tgt_idx) in enumerate(indices):
    #         if len(src_idx) == 0:
    #             continue
    #         t       = targets[b_idx]
    #         src_idx = src_idx.to(dev)
    #         tgt_idx = tgt_idx.to(dev)
    #         labels  = t['labels'].to(dev)[tgt_idx]
    #         tids    = t['track_ids'].to(dev)[tgt_idx]
    #         valid   = tids >= 0
    #         if not valid.any():
    #             continue

    #         src_v   = src_idx[valid]
    #         emb_b   = pred_reid[b_idx][src_v]
    #         emb_b_r = pred_reid_raw[b_idx][src_v]
    #         lbl_b   = labels[valid]
    #         ids_b   = tids[valid]

    #         for cls_id in self.nid_dict:
    #             mask = (lbl_b == cls_id)
    #             if not mask.any():
    #                 continue
    #             cls_emb[cls_id].append(emb_b[mask])
    #             cls_emb_raw[cls_id].append(emb_b_r[mask])
    #             cls_ids[cls_id].append(ids_b[mask])

    #     n_active = 0
    #     for cls_id in self.nid_dict:
    #         if not cls_emb[cls_id]:
    #             continue
    #         emb     = torch.cat(cls_emb[cls_id], dim=0)        # (n, D) post-neck
    #         emb_raw = torch.cat(cls_emb_raw[cls_id], dim=0)    # (n, D) pre-neck
    #         ids     = torch.cat(cls_ids[cls_id], dim=0)        # (n,)

    #         # ---- classification (ID) loss ----
    #         if self.use_arcface:
    #             logits = self.classifiers[str(cls_id)](emb, ids)   # ArcFace normalizes internally
    #         else:
    #             emb_id = self.emb_scale_dict[cls_id] * F.normalize(emb, dim=1)
    #             logits = self.linear_classifiers[str(cls_id)](emb_id)
    #         reid_loss = reid_loss + self.ce_loss(logits, ids)

    #         # ---- optional metric (triplet) loss on the un-normalized vector ----
    #         if self.use_triplet and emb_raw.shape[0] >= 2:
    #             reid_loss = reid_loss + self.triplet(emb_raw, ids)

    #         n_active += 1

    #     if n_active > 1:
    #         reid_loss = reid_loss / n_active
    #     return {'loss_reid': reid_loss}


    @staticmethod
    def _sample_emb_map(emb_map_b: torch.Tensor, centers_xy: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample emb_map [D,H,W] tại các tâm GT (cx,cy ∈ [0,1]) -> [n, D]."""
        D = emb_map_b.shape[0]
        if centers_xy.numel() == 0:
            return emb_map_b.new_zeros((0, D))
        grid = (centers_xy * 2.0 - 1.0).view(1, -1, 1, 2)   # grid_sample: (x,y) ∈ [-1,1]
        s = F.grid_sample(emb_map_b.unsqueeze(0), grid,
                        mode='bilinear', align_corners=False)   # [1, D, n, 1]
        return s.view(D, -1).t().contiguous()   

    def loss_reid(self, outputs, targets, indices) -> dict:
        """ReID loss per-class (CE + Triplet) + DENSE alignment.
    
        Sparse:
        • pred_reid     (post-neck) -> CE (linear_classifier sau emb_scale * L2-norm)
        • pred_reid_raw (pre-neck)  -> TripletLoss
        Dense (MỚI, chỉ khi có pred_reid_map):
        • lấy emb_map tại TÂM GT thật -> qua CHÍNH linear_classifier đó -> CE
            => cấp tín hiệu identity cho pixel đúng vị trí + kéo dense về cùng metric.
        • consistency: vector dense tại tâm GT kéo về emb sparse của chính instance
            (cosine; sparse.detach() làm teacher để ổn định).
        """
        if 'pred_reid' not in outputs:
            return {}
    
        pred_reid     = outputs['pred_reid']                       # (B,N,D) post-neck
        pred_reid_raw = outputs.get('pred_reid_raw', pred_reid)    # (B,N,D) pre-neck
        pred_reid_map = outputs.get('pred_reid_map', None)         # (B,D,H,W) hoặc None
        dev = pred_reid.device
        reid_loss = pred_reid.sum() * 0.0
    
        cls_emb       = {cid: [] for cid in self.nid_dict}   # post-neck (CE)
        cls_emb_raw   = {cid: [] for cid in self.nid_dict}   # pre-neck  (triplet)
        cls_emb_dense = {cid: [] for cid in self.nid_dict}   # dense (CE + consistency)
        cls_ids       = {cid: [] for cid in self.nid_dict}
    
        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            t       = targets[b_idx]
            src_idx = src_idx.to(dev)
            tgt_idx = tgt_idx.to(dev)
            labels  = t['labels'].to(dev)[tgt_idx]
            tids    = t['track_ids'].to(dev)[tgt_idx]
            valid   = tids >= 0
            if not valid.any():
                continue
    
            src_v   = src_idx[valid]
            emb_b   = pred_reid[b_idx][src_v]
            emb_b_r = pred_reid_raw[b_idx][src_v]
            lbl_b   = labels[valid]
            ids_b   = tids[valid]
    
            dense_b = None
            if pred_reid_map is not None:
                centers = t['boxes'].to(dev)[tgt_idx][valid][:, :2]    # (cx,cy) ∈ [0,1]
                dense_b = self._sample_emb_map(pred_reid_map[b_idx], centers)   # [n, D]
    
            for cls_id in self.nid_dict:
                mask = (lbl_b == cls_id)
                if not mask.any():
                    continue
                cls_emb[cls_id].append(emb_b[mask])
                cls_emb_raw[cls_id].append(emb_b_r[mask])
                cls_ids[cls_id].append(ids_b[mask])
                if dense_b is not None:
                    cls_emb_dense[cls_id].append(dense_b[mask])
    
        n_active = 0
        for cls_id in self.nid_dict:
            if not cls_emb[cls_id]:
                continue
            emb     = torch.cat(cls_emb[cls_id], dim=0)        # (n, D)
            emb_raw = torch.cat(cls_emb_raw[cls_id], dim=0)
            ids     = torch.cat(cls_ids[cls_id], dim=0)
    
            # ---- sparse classification (ID) loss ----
            if self.use_arcface:
                logits = self.classifiers[str(cls_id)](emb, ids)
            else:
                emb_id = self.emb_scale_dict[cls_id] * F.normalize(emb, dim=1)
                logits = self.linear_classifiers[str(cls_id)](emb_id)
            reid_loss = reid_loss + self.ce_loss(logits, ids)
    
            # ---- sparse triplet (pre-neck, Euclidean tự do) ----
            if self.use_triplet and emb_raw.shape[0] >= 2:
                reid_loss = reid_loss + self.triplet(emb_raw, ids)
    
            # ---- DENSE: CE qua CHÍNH classifier + consistency về sparse ----
            if cls_emb_dense[cls_id]:
                dense = torch.cat(cls_emb_dense[cls_id], dim=0)     # (n, D)
                if self.use_arcface:
                    logits_d = self.classifiers[str(cls_id)](dense, ids)
                else:
                    emb_id_d = self.emb_scale_dict[cls_id] * F.normalize(dense, dim=1)
                    logits_d = self.linear_classifiers[str(cls_id)](emb_id_d)
                reid_loss = reid_loss + self.w_dense_ce * self.ce_loss(logits_d, ids)
    
                cons = 1.0 - (F.normalize(dense, dim=1)
                            * F.normalize(emb.detach(), dim=1)).sum(dim=1)
                reid_loss = reid_loss + self.w_cons * cons.mean()
    
            n_active += 1
    
        if n_active > 1:
            reid_loss = reid_loss / n_active
        return {'loss_reid': reid_loss}

    # def loss_s4_aux(self, outputs, targets, indices, num_boxes):
    #     """
    #     TẠO MỤC TIÊU ĐỘC LẬP & TÍNH LOSS CHO NHÁNH PHỤ STRIDE-4 (S4 AUXILIARY HEAD)
    #     Cơ chế tiêm mãnh liệt Gradient để duy trì độ nhạy cho các vật thể siêu nhỏ.
    #     """
    #     losses = {'loss_s4_aux': torch.tensor(0.0, device=outputs['pred_logits'].device)}
    #     if 'pred_s4_aux' not in outputs:
    #         return losses

    #     pred_heatmap = outputs['pred_s4_aux'] # Kích thước hình học: (B, 1, H/4, W/4)
    #     B, _, H_s4, W_s4 = pred_heatmap.shape
        
    #     # Khởi tạo bản đồ nhị phân mục tiêu rỗng
    #     target_heatmap = torch.zeros_like(pred_heatmap)

    #     for b in range(B):
    #         tgt_boxes = targets[b]['boxes'] # Lấy tọa độ chuẩn hóa [cx, cy, w, h] trong khoảng [0, 1]
    #         if len(tgt_boxes) == 0:
    #             continue

    #         # Ánh xạ tọa độ chuẩn hóa về chiều không gian lưới rời rạc của bản đồ Stride-4
    #         cx = tgt_boxes[:, 0] * W_s4
    #         cy = tgt_boxes[:, 1] * H_s4
    #         w  = tgt_boxes[:, 2] * W_s4
    #         h  = tgt_boxes[:, 3] * H_s4

    #         x1 = (cx - w * 0.5).long().clamp(0, W_s4 - 1)
    #         y1 = (cy - h * 0.5).long().clamp(0, H_s4 - 1)
    #         x2 = (cx + w * 0.5).long().clamp(0, W_s4 - 1)
    #         y2 = (cy + h * 0.5).long().clamp(0, H_s4 - 1)

    #         # Điền giá trị 1.0 vào các pixel nằm bên trong vùng bounding box của vật thể
    #         for idx in range(len(tgt_boxes)):
    #             target_heatmap[b, 0, y1[idx]:y2[idx] + 1, x1[idx]:x2[idx] + 1] = 1.0

    #     # Áp dụng hàm phạt BCE Loss có trọng số ổn định số học cao
    #     losses['loss_s4_aux'] = F.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap, reduction='mean')
    #     return losses
    def loss_s4_aux(self, outputs, targets, indices, num_boxes):
        dev = outputs['pred_logits'].device
        losses = {'loss_s4_aux': torch.tensor(0.0, device=dev)}
        if 'pred_s4_aux' not in outputs:
            return losses
        pred = outputs['pred_s4_aux']                 # [B,1,H4,W4] logit
        _, _, H, W = pred.shape
        gt = build_center_heatmaps(targets, H, W, dev)
        losses['loss_s4_aux'] = gaussian_focal_loss(pred, gt)
        return losses
    # ------------------------------------------------------------------
    # Mask loss (AMOT DiceLoss adaptation for DETR query-based paradigm)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Matching utilities
    # ------------------------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_go_indices(self, indices, aux_indices_list):
        for aux in aux_indices_list:
            indices = [(torch.cat([i1[0], i2[0]]), torch.cat([i1[1], i2[1]]))
                       for i1, i2 in zip(indices, aux)]
        results = []
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            sort_idx = torch.argsort(counts, descending=True)
            col2row = {}
            for pair in unique[sort_idx]:
                r, c = pair[0].item(), pair[1].item()
                if r not in col2row:
                    col2row[r] = c
            fr = torch.tensor(list(col2row.keys()),   device=ind.device)
            fc = torch.tensor(list(col2row.values()), device=ind.device)
            results.append((fr.long(), fc.long()))
        return results

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'focal': self.loss_labels_focal,   # -> loss_cls (giữ lại để A/B)
            'mal':   self.loss_labels_mal,     # -> loss_mal (DEIM, mặc định)
            'boxes': self.loss_boxes,          # -> loss_bbox + loss_giou
            's4_aux': self.loss_s4_aux
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _get_meta(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}
        idx = self._get_src_permutation_idx(indices)
        src = outputs['pred_boxes'][idx]
        tgt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        if self.boxes_weight_format == 'iou':
            iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src.detach()), box_cxcywh_to_xyxy(tgt))[0])
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src.detach()), box_cxcywh_to_xyxy(tgt)))
        else:
            raise ValueError(self.boxes_weight_format)
        if loss == 'boxes':
            return {'boxes_weight': iou}
        return {}

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group    = dn_meta['dn_num_group']
        num_gts  = [len(t['labels']) for t in targets]
        device   = targets[0]['labels'].device
        result   = []
        for i, ng in enumerate(num_gts):
            if ng > 0:
                gt_idx = torch.arange(ng, dtype=torch.int64, device=device).tile(dn_num_group)
                result.append((dn_positive_idx[i], gt_idx))
            else:
                result.append((torch.zeros(0, dtype=torch.int64, device=device),
                               torch.zeros(0, dtype=torch.int64, device=device)))
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, outputs, targets, epoch: int = 0):
        outputs_no_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Main matching
        indices = self.matcher(outputs_no_aux, targets, epoch=epoch)['indices']

        # Aux matching for GO (union) set
        aux_list     = outputs.get('aux_outputs', [])
        pre_list     = [outputs['pre_outputs']] if 'pre_outputs' in outputs else []
        enc_list     = outputs.get('enc_aux_outputs', [])
        all_aux      = list(aux_list) + pre_list + list(enc_list)

        cached_indices = []
        for aux in all_aux:
            cached_indices.append(self.matcher(aux, targets, epoch=epoch)['indices'])

        indices_go = self._get_go_indices(indices, cached_indices) if cached_indices else indices

        # num_boxes normalization
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if _is_dist():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / _get_world_size(), min=1).item()

        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
                                       device=next(iter(outputs.values())).device)
        if _is_dist():
            torch.distributed.all_reduce(num_boxes_go)
        num_boxes_go = torch.clamp(num_boxes_go / _get_world_size(), min=1).item()

        losses = {}
        _up = outputs.get('up')
        _rs = outputs.get('reg_scale')

        def _apply(out, tgts, idx_main, idx_go, nb, nb_go, suffix=''):
            for loss in self.losses:
                # Tránh tính toán lặp lại nhánh S4 Aux trên các tầng phụ (aux layers/dn layers)
                if loss == 's4_aux' and suffix != '':
                    continue
                
                use_go = self.use_uni_set and loss == 'boxes'
                idx_in = idx_go if use_go else idx_main
                nb_in  = nb_go  if use_go else nb
                meta   = self._get_meta(loss, out, tgts, idx_in)
                l_dict = self.get_loss(loss, out, tgts, idx_in, nb_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                          for k in l_dict if k in self.weight_dict}
                if suffix:
                    l_dict = {k + suffix: v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Áp dụng cho đầu ra chính (bao gồm cả loss_s4_aux nếu có trong self.losses)
        _apply(outputs, targets, indices, indices_go, num_boxes, num_boxes_go)

        # Aux detection losses
        for i, aux in enumerate(aux_list):
            aux = {**aux, 'up': _up, 'reg_scale': _rs}
            _apply(aux, targets, cached_indices[i], indices_go,
                   num_boxes, num_boxes_go, f'_aux_{i}')

        if pre_list:
            pre_idx = len(aux_list)
            _apply(pre_list[0], targets, cached_indices[pre_idx], indices_go,
                   num_boxes, num_boxes_go, '_pre')

        # Enc aux losses — use zero labels when class-agnostic encoder
        enc_start = len(aux_list) + len(pre_list)
        class_agnostic = outputs.get('enc_meta', {}).get('class_agnostic', False)
        if class_agnostic:
            enc_targets = [{**t, 'labels': torch.zeros_like(t['labels'])} for t in targets]
            orig_nc = self.num_classes
            self.num_classes = 1
        else:
            enc_targets = targets
        for i, aux in enumerate(enc_list):
            _apply(aux, enc_targets, cached_indices[enc_start + i], indices_go,
                   num_boxes, num_boxes_go, f'_enc_{i}')
        if class_agnostic:
            self.num_classes = orig_nc

        # DN losses
        if 'dn_outputs' in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_nb = max(num_boxes * outputs['dn_meta']['dn_num_group'], 1)
            for i, aux in enumerate(outputs['dn_outputs']):
                aux = {**aux, 'up': _up, 'reg_scale': _rs}
                _apply(aux, targets, indices_dn, indices_dn, dn_nb, dn_nb, f'_dn_{i}')
            if 'dn_pre_outputs' in outputs:
                aux = {**outputs['dn_pre_outputs'], 'up': _up, 'reg_scale': _rs}
                _apply(aux, targets, indices_dn, indices_dn, dn_nb, dn_nb, '_dn_pre')

        # ReID loss
        if self.use_reid and self.id_weight > 0:
            reid_dict = self.loss_reid(outputs, targets, indices)
            losses.update({k: v * self.id_weight for k, v in reid_dict.items()})

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        # ------------------------------------------------------------------
        # Total loss — learnable uncertainty weighting (Kendall et al.).
        # det_loss aggregates EVERY detection term (main + aux + dn + enc + s4)
        # so the auxiliary supervision is preserved; only ReID is weighted
        # separately. The (s_det + s_id) term is the regulariser that stops the
        # weights collapsing to zero.
        # ------------------------------------------------------------------
        reid_loss = losses.get('loss_reid', None)
        det_loss  = sum(v for k, v in losses.items() if k != 'loss_reid')

        if self.use_reid and self.id_weight > 0 and reid_loss is not None:
            total = (torch.exp(-self.s_det) * det_loss
                     + torch.exp(-self.s_id) * reid_loss
                     + (self.s_det + self.s_id))
        else:
            # total = torch.exp(-self.s_det) * det_loss + self.s_det
            total = det_loss
        total = total

        # Detached scalars for logging only.
        losses['loss_det'] = det_loss.detach()
        losses['s_det']    = self.s_det.detach()
        losses['s_id']     = self.s_id.detach()
        losses['loss']     = total
        return losses