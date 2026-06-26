"""
FalconJDECriterion — DETR-style detection loss (DEIMv2 recipe) + per-class ReID loss.

Detection (matches the `deimv2_dinov3_s` config):
    loss_mal             — Matchability-Aware Loss (DEIM), gamma=1.5, mal_alpha=None
    loss_bbox, loss_giou — L1 + GIoU
    loss_fgl, loss_ddf   — Fine-Grained Localization + Decoupled Distillation Focal (D-FINE)

ReID (this project's MOT extension):
    loss_reid            — per-class CE/ArcFace (+ Triplet, + optional dense alignment)
    loss_s4_aux          — auxiliary stride-4 heatmap (optional)

The total loss balances det/ReID via homoscedastic uncertainty (Kendall et al.).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from falconmot.nn.falcon_jde.loss.matcher import HungarianMatcher
from ..ops.utils import bbox2distance
from ..ops.feat_fusion import build_center_heatmaps, gaussian_focal_loss


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# ---------------------------------------------------------------------------
# ReID sub-modules
# ---------------------------------------------------------------------------
class ArcFace(nn.Module):
    """ArcFace margin loss with reduced s/m for small, blurry (drone) objects."""

    def __init__(self, in_features: int, num_ids: int, s: float = 16.0, m: float = 0.15):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_ids, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
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
        mask_pos = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        mask_neg = ~mask_pos
        dist_ap = dist.masked_fill(mask_neg, float('-inf')).max(dim=1).values
        dist_an = dist.masked_fill(mask_pos, float('inf')).min(dim=1).values
        valid = mask_neg.any(dim=1)
        dist_ap, dist_an = dist_ap[valid], dist_an[valid]
        if dist_ap.numel() == 0:
            return inputs.sum() * 0
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


# ---------------------------------------------------------------------------
# Main criterion
# ---------------------------------------------------------------------------
class FalconJDECriterion(nn.Module):

    def __init__(
        self,
        matcher:             HungarianMatcher,
        num_classes:         int,
        nid_dict:            dict,
        reid_dim:            int   = 128,
        weight_dict:         dict  = None,
        losses:              tuple = ('mal', 'boxes', 'local'),
        gamma:               float = 1.5,            # MAL gamma (deimv2_s)
        mal_alpha:           float = None,           # None = original DEIM default (deimv2_s)
        reg_max:             int   = 32,
        boxes_weight_format: str   = None,
        use_uni_set:         bool  = True,
        # ----- ReID / multi-task -----
        use_reid:            bool  = False,
        id_weight:           float = 1.0,
        use_triplet:         bool  = False,
        use_arcface:         bool  = True,
        s_det_init:          float = 2.5,
        s_id_init:           float = 1.85,
        w_dense_ce:          float = 0.5,
        w_cons:              float = 0.1,
    ):
        super().__init__()
        self.matcher             = matcher
        self.num_classes         = num_classes
        self.nid_dict            = nid_dict
        self.reid_dim            = reid_dim
        self.losses              = losses
        self.gamma               = gamma
        self.mal_alpha           = mal_alpha
        self.reg_max             = reg_max
        self.boxes_weight_format = boxes_weight_format
        self.use_uni_set         = use_uni_set
        self.use_reid            = use_reid
        self.id_weight           = id_weight
        self.use_triplet         = use_triplet
        self.use_arcface         = use_arcface
        self.w_dense_ce          = w_dense_ce
        self.w_cons              = w_cons

        self.weight_dict = weight_dict or {
            'loss_mal':  1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
            'loss_fgl':  0.15,
            'loss_ddf':  1.5,
        }

        # Cache for FGL/DDF, reset every forward.
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

        # Per-class ReID classifiers.
        if use_reid:
            if use_arcface:
                self.classifiers = nn.ModuleDict(
                    {str(c): ArcFace(reid_dim, nid) for c, nid in nid_dict.items()})
            else:
                self.linear_classifiers = nn.ModuleDict(
                    {str(c): nn.Linear(reid_dim, nid, bias=False) for c, nid in nid_dict.items()})
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.triplet = TripletLoss(margin=0.3)
            # FairMOT/AMOT embedding scale based on the number of IDs per class.
            self.emb_scale_dict = {
                c: math.sqrt(2) * math.log(max(nid - 1, 2)) for c, nid in nid_dict.items()}

        # Learnable uncertainty weights (det vs ReID). Init ≈ log(initial raw loss).
        self.s_det = nn.Parameter(torch.tensor(float(s_det_init)))
        self.s_id  = nn.Parameter(torch.tensor(float(s_id_init)))

    # ==================================================================
    # Detection losses
    # ==================================================================
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """Matchability-Aware Loss (DEIM): soft target = IoU^gamma at the GT class."""
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

    @staticmethod
    def unimodal_distribution_focal_loss(pred, label, weight_right, weight_left,
                                         weight=None, avg_factor=None):
        """CE over the two neighboring bins (left/right) with interpolation weights — the FGL kernel of D-FINE."""
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = (F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1)
                + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1))
        if weight is not None:
            loss = loss * weight.float()
        return loss.sum() / avg_factor if avg_factor is not None else loss.sum()

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """FGL (Fine-Grained Localization) + DDF (Decoupled Distillation Focal).

        Directly supervises the discrete distribution `pred_corners` (4*(reg_max+1)).
        Skips branches without corners (enc/pre)."""
        losses = {}
        if 'pred_corners' not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        pred_corners = outputs['pred_corners'][idx].reshape(-1, self.reg_max + 1)
        ref_points = outputs['ref_points'][idx].detach()

        # Distribution target: computed once per forward (denoising branch handled separately).
        with torch.no_grad():
            if self.fgl_targets_dn is None and 'is_dn' in outputs:
                self.fgl_targets_dn = bbox2distance(
                    ref_points, box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max, outputs['reg_scale'], outputs['up'])
            if self.fgl_targets is None and 'is_dn' not in outputs:
                self.fgl_targets = bbox2distance(
                    ref_points, box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max, outputs['reg_scale'], outputs['up'])
        target_corners, weight_right, weight_left = (
            self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets)

        ious = torch.diag(box_iou(
            box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),
            box_cxcywh_to_xyxy(target_boxes))[0])
        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left,
            weight=weight_targets, avg_factor=num_boxes)

        # DDF: distill the last-layer distribution (teacher) -> earlier layers.
        if 'teacher_corners' in outputs:
            pred_all = outputs['pred_corners'].reshape(-1, self.reg_max + 1)
            teacher_all = outputs['teacher_corners'].reshape(-1, self.reg_max + 1)
            if not torch.equal(pred_all, teacher_all):
                weight_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]
                mask = torch.zeros_like(weight_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                weight_local[idx] = ious.reshape_as(weight_local[idx]).to(weight_local.dtype)
                weight_local = weight_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                loss_match = weight_local * (T ** 2) * nn.KLDivLoss(reduction='none')(
                    F.log_softmax(pred_all / T, dim=1),
                    F.softmax(teacher_all.detach() / T, dim=1)).sum(-1)

                # num_pos/num_neg are computed once on the non-dn branch and reused for dn.
                if 'is_dn' not in outputs:
                    batch_scale = 8 / outputs['pred_boxes'].shape[0]
                    self.num_pos = (mask.sum() * batch_scale) ** 0.5
                    self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
                l_pos = loss_match[mask].mean() if mask.any() else 0
                l_neg = loss_match[~mask].mean() if (~mask).any() else 0
                losses['loss_ddf'] = (l_pos * self.num_pos + l_neg * self.num_neg) \
                    / (self.num_pos + self.num_neg)

        return losses

    # ==================================================================
    # ReID loss
    # ==================================================================
    @staticmethod
    def _sample_emb_map(emb_map_b: torch.Tensor, centers_xy: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample emb_map [D,H,W] at the GT center (cx,cy in [0,1]) -> [n, D]."""
        D = emb_map_b.shape[0]
        if centers_xy.numel() == 0:
            return emb_map_b.new_zeros((0, D))
        grid = (centers_xy * 2.0 - 1.0).view(1, -1, 1, 2)
        s = F.grid_sample(emb_map_b.unsqueeze(0), grid, mode='bilinear', align_corners=False)
        return s.view(D, -1).t().contiguous()

    def loss_reid(self, outputs, targets, indices) -> dict:
        """Per-class ReID: sparse CE(+Triplet) + dense alignment (if pred_reid_map is given)."""
        if 'pred_reid' not in outputs:
            return {}

        pred_reid = outputs['pred_reid']                       # (B,N,D) post-neck
        pred_reid_raw = outputs.get('pred_reid_raw', pred_reid)  # (B,N,D) pre-neck
        pred_reid_map = outputs.get('pred_reid_map', None)       # (B,D,H,W) | None
        dev = pred_reid.device
        reid_loss = pred_reid.sum() * 0.0

        cls_emb       = {c: [] for c in self.nid_dict}
        cls_emb_raw   = {c: [] for c in self.nid_dict}
        cls_emb_dense = {c: [] for c in self.nid_dict}
        cls_emb_app   = {c: [] for c in self.nid_dict}          # NEW
        cls_ids       = {c: [] for c in self.nid_dict}
        pred_reid_app = outputs.get('pred_reid_app', None)

        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            t = targets[b_idx]
            src_idx, tgt_idx = src_idx.to(dev), tgt_idx.to(dev)
            labels = t['labels'].to(dev)[tgt_idx]
            tids = t['track_ids'].to(dev)[tgt_idx]
            valid = tids >= 0
            if not valid.any():
                continue

            src_v = src_idx[valid]
            emb_b = pred_reid[b_idx][src_v]
            emb_b_r = pred_reid_raw[b_idx][src_v]
            app_b = pred_reid_app[b_idx][src_v] if pred_reid_app is not None else None
            lbl_b = labels[valid]
            ids_b = tids[valid]

            dense_b = None
            if pred_reid_map is not None:
                centers = t['boxes'].to(dev)[tgt_idx][valid][:, :2]
                dense_b = self._sample_emb_map(pred_reid_map[b_idx], centers)

            for cls_id in self.nid_dict:
                mask = (lbl_b == cls_id)
                if not mask.any():
                    continue
                cls_emb[cls_id].append(emb_b[mask])
                cls_emb_raw[cls_id].append(emb_b_r[mask])
                if app_b is not None:                                  # NEW
                    cls_emb_app[cls_id].append(app_b[mask])
                cls_ids[cls_id].append(ids_b[mask])
                if dense_b is not None:
                    cls_emb_dense[cls_id].append(dense_b[mask])

        n_active = 0
        for cls_id in self.nid_dict:
            if not cls_emb[cls_id]:
                continue
            emb = torch.cat(cls_emb[cls_id], dim=0)
            emb_raw = torch.cat(cls_emb_raw[cls_id], dim=0)
            ids = torch.cat(cls_ids[cls_id], dim=0)

            # sparse classification
            if self.use_arcface:
                logits = self.classifiers[str(cls_id)](emb, ids)
            else:
                emb_id = self.emb_scale_dict[cls_id] * F.normalize(emb, dim=1)
                logits = self.linear_classifiers[str(cls_id)](emb_id)
            reid_loss = reid_loss + self.ce_loss(logits, ids)

            # sparse triplet (pre-neck)
            if self.use_triplet and emb_raw.shape[0] >= 2:
                reid_loss = reid_loss + self.triplet(emb_raw, ids)

            # dense CE + consistency with the sparse embedding
            if cls_emb_dense[cls_id]:
                dense = torch.cat(cls_emb_dense[cls_id], dim=0)
                if self.use_arcface:
                    logits_d = self.classifiers[str(cls_id)](dense, ids)
                else:
                    emb_id_d = self.emb_scale_dict[cls_id] * F.normalize(dense, dim=1)
                    logits_d = self.linear_classifiers[str(cls_id)](emb_id_d)
                reid_loss = reid_loss + self.w_dense_ce * self.ce_loss(logits_d, ids)

                if cls_emb_app[cls_id]:
                    app_t = torch.cat(cls_emb_app[cls_id], dim=0)
                    cons = 1.0 - (F.normalize(dense, dim=1)
                                  * F.normalize(app_t.detach(), dim=1)).sum(dim=1)
                    reid_loss = reid_loss + self.w_cons * cons.mean()

            n_active += 1

        if n_active > 1:
            reid_loss = reid_loss / n_active
        return {'loss_reid': reid_loss}

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

    # ==================================================================
    # Utilities
    # ==================================================================
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_go_indices(self, indices, aux_indices_list):
        """Union set (Dense O2O) across all decoder layers."""
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
            fr = torch.tensor(list(col2row.keys()), device=ind.device)
            fc = torch.tensor(list(col2row.values()), device=ind.device)
            results.append((fr.long(), fc.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'mal':    self.loss_labels_mal,
            'boxes':  self.loss_boxes,
            'local':  self.loss_local,
            's4_aux': self.loss_s4_aux,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _get_meta(self, loss, outputs, targets, indices):
        """boxes_weight (IoU) for loss_boxes when boxes_weight_format is enabled."""
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
        return {'boxes_weight': iou} if loss == 'boxes' else {}

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group = dn_meta['dn_num_group']
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        result = []
        for i, ng in enumerate(num_gts):
            if ng > 0:
                gt_idx = torch.arange(ng, dtype=torch.int64, device=device).tile(dn_num_group)
                result.append((dn_positive_idx[i], gt_idx))
            else:
                result.append((torch.zeros(0, dtype=torch.int64, device=device),
                               torch.zeros(0, dtype=torch.int64, device=device)))
        return result

    # ==================================================================
    # Forward
    # ==================================================================
    def forward(self, outputs, targets, epoch: int = 0):
        outputs_no_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(outputs_no_aux, targets, epoch=epoch)['indices']
        self._clear_cache()

        # Matching cho union (GO) set.
        aux_list = outputs.get('aux_outputs', [])
        pre_list = [outputs['pre_outputs']] if 'pre_outputs' in outputs else []
        enc_list = outputs.get('enc_aux_outputs', [])
        all_aux = list(aux_list) + pre_list + list(enc_list)

        cached_indices = [self.matcher(aux, targets, epoch=epoch)['indices'] for aux in all_aux]
        indices_go = self._get_go_indices(indices, cached_indices) if cached_indices else indices

        # Normalize num_boxes.
        dev = next(iter(outputs.values())).device

        def _normalize(n):
            n = torch.as_tensor([n], dtype=torch.float, device=dev)
            if _is_dist():
                torch.distributed.all_reduce(n)
            return torch.clamp(n / _get_world_size(), min=1).item()

        num_boxes = _normalize(sum(len(t['labels']) for t in targets))
        num_boxes_go = _normalize(sum(len(x[0]) for x in indices_go))

        losses = {}
        _up = outputs.get('up')
        _rs = outputs.get('reg_scale')

        def _apply(out, tgts, idx_main, idx_go, nb, nb_go, suffix=''):
            for loss in self.losses:
                # s4_aux is computed only once on the main branch.
                if loss == 's4_aux' and suffix != '':
                    continue
                # GO/union set applied to both boxes and local (per DEIM).
                use_go = self.use_uni_set and loss in ('boxes', 'local')
                idx_in = idx_go if use_go else idx_main
                nb_in = nb_go if use_go else nb
                meta = self._get_meta(loss, out, tgts, idx_in)
                l_dict = self.get_loss(loss, out, tgts, idx_in, nb_in, **meta)
                l_dict = {k: v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict}
                if suffix:
                    l_dict = {k + suffix: v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Main branch (last layer) — set self.fgl_targets for FGL.
        _apply(outputs, targets, indices, indices_go, num_boxes, num_boxes_go)

        # Aux layers.
        for i, aux in enumerate(aux_list):
            aux = {**aux, 'up': _up, 'reg_scale': _rs}
            _apply(aux, targets, cached_indices[i], indices_go, num_boxes, num_boxes_go, f'_aux_{i}')

        # Pre head (D-FINE), no corners -> only mal/boxes.
        if pre_list:
            _apply(pre_list[0], targets, cached_indices[len(aux_list)], indices_go,
                   num_boxes, num_boxes_go, '_pre')

        # Encoder aux (class-agnostic if needed), no corners.
        enc_start = len(aux_list) + len(pre_list)
        class_agnostic = outputs.get('enc_meta', {}).get('class_agnostic', False)
        if class_agnostic:
            enc_targets = [{**t, 'labels': torch.zeros_like(t['labels'])} for t in targets]
            orig_nc, self.num_classes = self.num_classes, 1
        else:
            enc_targets = targets
        for i, aux in enumerate(enc_list):
            _apply(aux, enc_targets, cached_indices[enc_start + i], indices_go,
                   num_boxes, num_boxes_go, f'_enc_{i}')
        if class_agnostic:
            self.num_classes = orig_nc

        # Denoising (CDN) — flagged is_dn so FGL uses its own cache.
        if 'dn_outputs' in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_nb = max(num_boxes * outputs['dn_meta']['dn_num_group'], 1)
            for i, aux in enumerate(outputs['dn_outputs']):
                aux = {**aux, 'up': _up, 'reg_scale': _rs, 'is_dn': True}
                _apply(aux, targets, indices_dn, indices_dn, dn_nb, dn_nb, f'_dn_{i}')
            if 'dn_pre_outputs' in outputs:
                aux = {**outputs['dn_pre_outputs'], 'up': _up, 'reg_scale': _rs, 'is_dn': True}
                _apply(aux, targets, indices_dn, indices_dn, dn_nb, dn_nb, '_dn_pre')

        # ReID.
        if self.use_reid and self.id_weight > 0:
            reid_dict = self.loss_reid(outputs, targets, indices)
            losses.update({k: v * self.id_weight for k, v in reid_dict.items()})

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        reid_loss = losses.get('loss_reid', None)
        det_loss = sum(v for k, v in losses.items() if k != 'loss_reid')

        if self.use_reid and self.id_weight > 0 and reid_loss is not None:
            total = det_loss + self.id_weight * reid_loss
        else:
            total = det_loss

        losses['loss_det'] = det_loss.detach()
        losses['loss'] = total

        # ------------------------------------------------------------------
        # Logging: report each component separately for easier monitoring.
        #   loss_mal, loss_mal_aux_0, loss_mal_aux_1, ...
        #   loss_fgl, loss_fgl_aux_0, ...   (not summed together)
        # These keys are already created by _apply() with suffixes; here we only detach
        # them for printing (without breaking the graph — the total 'loss' is kept for backward).
        # ------------------------------------------------------------------
        for k in list(losses.keys()):
            if k != 'loss' and isinstance(losses[k], torch.Tensor):
                losses[k] = losses[k].detach()

        # Ensure the main keys always exist so the Trainer never hits a KeyError.
        # (the main branch has no loss_ddf because the last layer is the teacher -> defaults to 0)
        for key in ('loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl', 'loss_ddf'):
            losses.setdefault(key, torch.tensor(0.0, device=total.device))

        return losses