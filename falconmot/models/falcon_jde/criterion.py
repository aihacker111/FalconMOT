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
    """ArcFace margin loss head for ReID."""

    def __init__(self, in_features: int, num_ids: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_ids, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
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
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            pos = dist[i][mask[i]]
            neg = dist[i][~mask[i]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            dist_ap.append(pos.max().unsqueeze(0))
            dist_an.append(neg.min().unsqueeze(0))
        if not dist_ap:
            return inputs.sum() * 0
        return self.ranking_loss(
            torch.cat(dist_an), torch.cat(dist_ap),
            torch.ones(len(dist_an), device=inputs.device)
        )


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
        losses:              tuple = ('focal', 'boxes'),
        alpha:               float = 0.25,
        gamma:               float = 2.0,
        reg_max:             int   = 32,
        boxes_weight_format: str   = None,
        use_uni_set:         bool  = True,
        id_weight:           float = 1.0,
        use_triplet:         bool  = False,
        use_arcface:         bool  = True,
        # Spatial-Proximity ReID loss
        use_prox_reid:       bool  = False,
        prox_dist_thresh:    float = 0.10,
        prox_base_margin:    float = 0.30,
        prox_margin_scale:   float = 0.40,
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
        self.id_weight           = id_weight
        self.use_triplet         = use_triplet
        self.use_arcface         = use_arcface
        self.use_prox_reid       = use_prox_reid
        self.prox_dist_thresh    = prox_dist_thresh
        self.prox_base_margin    = prox_base_margin
        self.prox_margin_scale   = prox_margin_scale


        self.weight_dict = weight_dict or {
            'loss_cls':  2.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
            'loss_s4_aux':  0.5
        }

        # Per-class classifiers: ArcFace (margin-based) or Linear (plain CE+Triplet)
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

    def loss_reid(self, outputs, targets, indices) -> dict:
        if 'pred_reid' not in outputs:
            return {}

        pred_reid = outputs['pred_reid']   # (B, N_det, D)
        dev = pred_reid.device
        reid_loss = pred_reid.sum() * 0.0

        cls_emb = {cid: [] for cid in self.nid_dict}
        cls_ids = {cid: [] for cid in self.nid_dict}

        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            t      = targets[b_idx]
            labels = t['labels'].to(dev)[tgt_idx.to(dev)]
            tids   = t['track_ids'].to(dev)[tgt_idx.to(dev)]
            valid  = tids >= 0
            if not valid.any():
                continue
            emb_all = pred_reid[b_idx][src_idx.to(dev)[valid]]
            lbl_all = labels[valid]
            ids_all = tids[valid]

            for cls_id in self.nid_dict:
                mask = (lbl_all == cls_id)
                if not mask.any():
                    continue
                cls_emb[cls_id].append(emb_all[mask])
                cls_ids[cls_id].append(ids_all[mask])

        n_active = 0
        for cls_id in self.nid_dict:
            if not cls_emb[cls_id]:
                continue
            emb_cat = torch.cat(cls_emb[cls_id], dim=0)
            ids_cat = torch.cat(cls_ids[cls_id], dim=0)
            if self.use_arcface:
                pred = self.classifiers[str(cls_id)](emb_cat, ids_cat)
            else:
                pred = self.linear_classifiers[str(cls_id)](emb_cat)
            reid_loss = reid_loss + self.ce_loss(pred, ids_cat)
            if self.use_triplet and emb_cat.shape[0] >= 2:
                reid_loss = reid_loss + self.triplet(F.normalize(emb_cat, dim=-1), ids_cat)
            n_active += 1

        if n_active > 1:
            reid_loss = reid_loss / n_active
        return {'loss_reid': reid_loss}


    def loss_s4_aux(self, outputs, targets, indices, num_boxes):
        """
        TẠO MỤC TIÊU ĐỘC LẬP & TÍNH LOSS CHO NHÁNH PHỤ STRIDE-4 (S4 AUXILIARY HEAD)
        Cơ chế tiêm mãnh liệt Gradient để duy trì độ nhạy cho các vật thể siêu nhỏ.
        """
        losses = {'loss_s4_aux': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        if 'pred_s4_aux' not in outputs:
            return losses

        pred_heatmap = outputs['pred_s4_aux'] # Kích thước hình học: (B, 1, H/4, W/4)
        B, _, H_s4, W_s4 = pred_heatmap.shape
        
        # Khởi tạo bản đồ nhị phân mục tiêu rỗng
        target_heatmap = torch.zeros_like(pred_heatmap)

        for b in range(B):
            tgt_boxes = targets[b]['boxes'] # Lấy tọa độ chuẩn hóa [cx, cy, w, h] trong khoảng [0, 1]
            if len(tgt_boxes) == 0:
                continue

            # Ánh xạ tọa độ chuẩn hóa về chiều không gian lưới rời rạc của bản đồ Stride-4
            cx = tgt_boxes[:, 0] * W_s4
            cy = tgt_boxes[:, 1] * H_s4
            w  = tgt_boxes[:, 2] * W_s4
            h  = tgt_boxes[:, 3] * H_s4

            x1 = (cx - w * 0.5).long().clamp(0, W_s4 - 1)
            y1 = (cy - h * 0.5).long().clamp(0, H_s4 - 1)
            x2 = (cx + w * 0.5).long().clamp(0, W_s4 - 1)
            y2 = (cy + h * 0.5).long().clamp(0, H_s4 - 1)

            # Điền giá trị 1.0 vào các pixel nằm bên trong vùng bounding box của vật thể
            for idx in range(len(tgt_boxes)):
                target_heatmap[b, 0, y1[idx]:y2[idx] + 1, x1[idx]:x2[idx] + 1] = 1.0

        # Áp dụng hàm phạt BCE Loss có trọng số ổn định số học cao
        losses['loss_s4_aux'] = F.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap, reduction='mean')
        return losses
    # ------------------------------------------------------------------
    # Mask loss (AMOT DiceLoss adaptation for DETR query-based paradigm)
    # ------------------------------------------------------------------

    def loss_prox_reid(self, outputs, targets, indices) -> dict:
        """Spatial-Proximity ReID loss (analog của CenterNet offset loss cho ID collision).

        Vấn đề: TripletLoss dùng margin cố định cho tất cả negative pairs,
        kể cả hai object cách nhau rất xa — không cần push mạnh.
        Hai object gần nhau mới là nguồn gốc ID collision, cần margin lớn hơn.

        Với mỗi image: tìm cặp (i, j) khác track_id có center distance < prox_dist_thresh,
        apply: loss = ReLU(margin(dist_ij) - ||emb_i - emb_j||)
        trong đó margin tỉ lệ nghịch với khoảng cách: gần hơn → margin lớn hơn.
        """
        if 'pred_reid' not in outputs:
            return {}

        pred_reid = outputs['pred_reid']   # (B, N_queries, D)
        dev       = pred_reid.device
        total     = pred_reid.sum() * 0.0
        n_pairs   = 0

        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) < 2:
                continue

            t        = targets[b_idx]
            tids     = t['track_ids'].to(dev)[tgt_idx.to(dev)]   # (N,)
            gt_boxes = t['boxes'].to(dev)[tgt_idx.to(dev)]       # (N, 4) cxcywh norm
            valid    = tids >= 0
            if valid.sum() < 2:
                continue

            embs  = F.normalize(pred_reid[b_idx][src_idx.to(dev)[valid]], dim=-1)
            tids  = tids[valid]
            boxes = gt_boxes[valid]  # (M, 4)

            # Center distance matrix (normalized coords, range [0, sqrt(2)])
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            dist_mat = ((cx.unsqueeze(0) - cx.unsqueeze(1)) ** 2
                      + (cy.unsqueeze(0) - cy.unsqueeze(1)) ** 2).sqrt()  # (M, M)

            # Hard-negative proximity mask: khác ID + đủ gần + upper-triangle (no dup)
            diff_id  = tids.unsqueeze(0) != tids.unsqueeze(1)
            proximal = dist_mat < self.prox_dist_thresh
            upper    = torch.triu(torch.ones_like(diff_id, dtype=torch.bool), diagonal=1)
            mask     = diff_id & proximal & upper

            if not mask.any():
                continue

            ii, jj   = mask.nonzero(as_tuple=True)
            emb_dist = (embs[ii] - embs[jj]).norm(dim=-1)          # (P,)

            # Adaptive margin: proximity_score=1 khi hoàn toàn chồng lên nhau, =0 tại ngưỡng
            proximity_score = 1.0 - dist_mat[ii, jj] / self.prox_dist_thresh
            margin = self.prox_base_margin + self.prox_margin_scale * proximity_score

            total   = total + F.relu(margin - emb_dist).mean()
            n_pairs += 1

        return {'loss_prox_reid': total / max(n_pairs, 1)}

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
            'focal': self.loss_labels_focal,   # -> loss_cls
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
        if self.id_weight > 0:
            reid_dict = self.loss_reid(outputs, targets, indices)
            losses.update({k: v * self.id_weight for k, v in reid_dict.items()})

            # Spatial-Proximity ReID loss
            if self.use_prox_reid:
                prox_dict = self.loss_prox_reid(outputs, targets, indices)
                w = self.weight_dict.get('loss_prox_reid', 0.5)
                losses.update({k: v * w for k, v in prox_dict.items()})

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        # Aggregate main-output losses for logging
        _det_keys = {'loss_cls', 'loss_bbox', 'loss_giou', 'loss_s4_aux'}
        losses['loss_det'] = sum(losses[k] for k in _det_keys if k in losses)
        losses['loss']     = sum(v for k, v in losses.items() if k not in ('loss_det', 'loss'))
        return losses