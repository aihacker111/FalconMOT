"""
FalconJDEPostProcessor — dual-mode postprocessor.

eval mode (for COCO mAP):
    Returns list[dict{labels, boxes (xyxy pixel), scores}]
    boxes scaled directly by orig_size (no letterbox inverse needed when using
    DEIMv2-style resize)

tracking mode (conf_thres > 0):
    Same as eval but also returns 'reid' (L2-normed embeddings), applies the
    conf_thres filter and an embedding-aware NMS. Includes inverse-letterbox for
    JDE-style letterbox preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _mod(a: torch.Tensor, b: int) -> torch.Tensor:
    return a - a // b * b


class FalconJDEPostProcessor(nn.Module):
    """
    Converts model outputs → final detections.

    Two operating modes controlled by conf_thres:
      conf_thres == 0.0 → eval mode: COCO mAP compatible output
      conf_thres  > 0.0 → tracking mode: filtered + ReID output + letterbox inverse

    Args:
        num_classes:     number of object classes
        num_top_queries: top-K selections from N*C flattened scores
        conf_thres:      0.0 = eval (no filter), >0 = tracking filter threshold
        use_focal_loss:  True for sigmoid scoring, False for softmax
        nms_iou:         overlap above which a same-class pair *may* be a duplicate
        nms_emb_hi:      appearance similarity required at IoU == nms_iou (strictest)
        nms_emb_relax:   how far that requirement loosens as IoU → 1
                         (0.0 reproduces the old fixed two-threshold rule)
        nms_soft:        decay duplicate scores instead of hard-removing them
        nms_soft_sigma:  Gaussian width for the soft decay
    """

    def __init__(
        self,
        num_classes:     int,
        num_top_queries: int   = 300,
        conf_thres:      float = 0.0,
        use_focal_loss:  bool  = True,
        nms_iou:         float = 0.6,
        nms_emb_hi:      float = 0.85,
        nms_emb_relax:   float = 0.45,
        nms_soft:        bool  = False,
        nms_soft_sigma:  float = 0.5,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.num_top_queries = num_top_queries
        self.conf_thres      = conf_thres
        self.use_focal_loss  = use_focal_loss
        self._net_hw         = None   # (net_H, net_W) — set for letterbox tracking mode

        # embedding-aware NMS config (tunable per dataset, no code edits)
        self.nms_iou        = nms_iou
        self.nms_emb_hi     = nms_emb_hi
        self.nms_emb_relax  = nms_emb_relax
        self.nms_soft       = nms_soft
        self.nms_soft_sigma = nms_soft_sigma

    def set_net_hw(self, net_h: int, net_w: int):
        """Call once before tracking inference to enable letterbox-inverse."""
        self._net_hw = (net_h, net_w)

    @torch.no_grad()
    def embedding_aware_nms(self, boxes, scores, labels, reid_feats=None,
                            iou_thr=None, emb_hi=None, emb_relax=None,
                            soft=None, soft_sigma=None, score_floor=0.0):
        """Suppress boxes that pile onto a single object, appearance-aware.

        Vectorised (no Python loop). A box is suppressed iff some higher-score
        box of the same class overlaps it past `iou_thr` with appearance >= a
        bar that *slides with overlap*:

            needed_sim(iou) = emb_hi − emb_relax · (iou − iou_thr)/(1 − iou_thr)

        so heavy overlap needs little appearance agreement to merge, while a
        weak overlap demands near-identical appearance — protecting two distinct
        objects sitting close together. `emb_relax = 0` recovers the old fixed
        two-threshold rule. `soft=True` decays scores instead of removing.

        Args:
            boxes      : (N, 4) xyxy.
            scores     : (N,)
            labels     : (N,) integer class ids.
            reid_feats : (N, C) embeddings (L2-normed; re-normalised for safety).
        Returns:
            (boxes, scores, labels, reid_feats) filtered.
        """
        iou_thr    = self.nms_iou        if iou_thr    is None else iou_thr
        emb_hi     = self.nms_emb_hi     if emb_hi     is None else emb_hi
        emb_relax  = self.nms_emb_relax  if emb_relax  is None else emb_relax
        soft       = self.nms_soft       if soft       is None else soft
        soft_sigma = self.nms_soft_sigma if soft_sigma is None else soft_sigma

        n = boxes.shape[0]
        if n <= 1:
            return boxes, scores, labels, reid_feats

        # The triangular suppression below assumes index order == score order.
        # Sort defensively so the function is correct regardless of caller.
        order = torch.argsort(scores, descending=True)
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        if reid_feats is not None:
            reid_feats = reid_feats[order]

        iou = torchvision.ops.box_iou(boxes, boxes)                 # (N, N)
        same = labels.unsqueeze(1) == labels.unsqueeze(0)           # (N, N)

        if reid_feats is not None:
            feat = F.normalize(reid_feats, dim=1, eps=1e-6)
            sim = feat @ feat.t()                                   # (N, N) cosine
        else:
            sim = torch.ones_like(iou)

        # appearance bar that slides with overlap
        over = ((iou - iou_thr) / max(1.0 - iou_thr, 1e-6)).clamp_(0.0, 1.0)
        needed = emb_hi - emb_relax * over                          # (N, N)

        # i suppresses j ⇔ same class, overlap past gate, appearance meets bar
        dup = same & (iou > iou_thr) & (sim >= needed)
        dup = torch.triu(dup, diagonal=1)        # higher-score → lower-score only

        if not soft:
            keep = ~dup.any(dim=0)
            return (boxes[keep], scores[keep], labels[keep],
                    reid_feats[keep] if reid_feats is not None else None)

        # soft mode: keep the lightest decay from any higher duplicate
        decay = torch.where(dup, torch.exp(-(iou ** 2) / soft_sigma),
                            torch.ones_like(iou))
        new_scores = scores * decay.min(dim=0).values
        keep = new_scores > score_floor
        return (boxes[keep], new_scores[keep], labels[keep],
                reid_feats[keep] if reid_feats is not None else None)

    @torch.no_grad()
    def forward(self, outputs: dict, orig_target_sizes: torch.Tensor,
                lb_pad: torch.Tensor = None) -> list:
        """
        Args:
            outputs:           model output dict with 'pred_logits' and 'pred_boxes'
            orig_target_sizes: (B, 2) int tensor [orig_H, orig_W]
            lb_pad:            (B, 2) int tensor [pad_w, pad_h] exact pixel offsets
                               from letterbox dataset. Eliminates ~0.33px rounding
                               mismatch vs recalculating (nw-ow*ratio)/2.

        Returns list[dict] per image:
            eval mode:     {'labels', 'boxes' (xyxy pixel), 'scores'}
            tracking mode: {'labels', 'boxes' (xyxy pixel), 'scores', 'reid'}
        """
        logits = outputs['pred_logits']   # (B, N, C)
        boxes  = outputs['pred_boxes']    # (B, N, 4) cxcywh normalized (letterbox space)
        reid   = outputs.get('pred_reid') # (B, N, D) or None
        B, N, C = logits.shape

        # ── 1. Score selection ────────────────────────────────────────────
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            K = min(self.num_top_queries, N * C)
            topk_scores, topk_idx = torch.topk(scores.flatten(1), K, dim=-1)
            labels    = _mod(topk_idx, C)         # (B, K)
            query_idx = topk_idx // C             # (B, K)
            sel_boxes = boxes.gather(
                1, query_idx.unsqueeze(-1).expand(-1, -1, 4))   # (B, K, 4)
            sel_reid = None
            if reid is not None:
                sel_reid = F.normalize(
                    reid.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, reid.shape[-1])),
                    dim=-1)

        else:
            scores = F.softmax(logits, dim=-1)[..., :-1]
            topk_scores, labels = scores.max(dim=-1)
            if topk_scores.shape[1] > self.num_top_queries:
                topk_scores, idx = torch.topk(topk_scores, self.num_top_queries, dim=-1)
                labels    = labels.gather(1, idx)
                sel_boxes = boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
            else:
                sel_boxes = boxes
            sel_reid = reid  # already (B, N, D) or None

        # ── 2. Box decode: cxcywh → xyxy in original pixel space ─────────
        results = []
        for b in range(B):
            oh = float(orig_target_sizes[b, 0])
            ow = float(orig_target_sizes[b, 1])
            bx = sel_boxes[b]   # (K, 4) cxcywh normalized

            if self._net_hw is not None:
                # Letterbox inverse
                nh, nw = self._net_hw
                ratio = min(nh / oh, nw / ow)
                if lb_pad is not None:
                    pad_w = float(lb_pad[b, 0].item())
                    pad_h = float(lb_pad[b, 1].item())
                else:
                    pad_w = (nw - ow * ratio) * 0.5
                    pad_h = (nh - oh * ratio) * 0.5
                cx = bx[:, 0] * nw;  cy = bx[:, 1] * nh
                bw = bx[:, 2] * nw;  bh = bx[:, 3] * nh
                x1 = (cx - bw * 0.5 - pad_w) / ratio
                y1 = (cy - bh * 0.5 - pad_h) / ratio
                x2 = (cx + bw * 0.5 - pad_w) / ratio
                y2 = (cy + bh * 0.5 - pad_h) / ratio
            else:
                # Simple scale (no letterbox)
                x1 = (bx[:, 0] - bx[:, 2] * 0.5) * ow
                y1 = (bx[:, 1] - bx[:, 3] * 0.5) * oh
                x2 = (bx[:, 0] + bx[:, 2] * 0.5) * ow
                y2 = (bx[:, 1] + bx[:, 3] * 0.5) * oh

            boxes_xyxy = torch.stack([
                x1.clamp(0, ow), y1.clamp(0, oh),
                x2.clamp(0, ow), y2.clamp(0, oh),
            ], dim=-1)

            sc  = topk_scores[b]
            lbl = labels[b]

            # ── 3. Confidence filter + embedding-aware NMS (tracking mode) ──
            if self.conf_thres > 0:
                keep       = sc >= self.conf_thres
                sc         = sc[keep]
                lbl        = lbl[keep]
                boxes_xyxy = boxes_xyxy[keep]
                reid_b     = sel_reid[b][keep] if sel_reid is not None else None

                # run after conf filter to keep the matrices small
                boxes_xyxy, sc, lbl, reid_b = self.embedding_aware_nms(
                    boxes_xyxy, sc, lbl, reid_b)
            else:
                reid_b = sel_reid[b] if sel_reid is not None else None

            res = {'labels': lbl, 'boxes': boxes_xyxy, 'scores': sc}
            if reid_b is not None:
                res['reid'] = reid_b
            results.append(res)

        return results

    def deploy(self):
        self.eval()
        return self
