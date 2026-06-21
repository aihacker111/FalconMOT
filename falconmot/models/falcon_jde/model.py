"""
FalconJDEModel — DINOv3STAs + HybridEncoder + DEIMTransformer + ReID head.

Updated with Deep-embedded 4-scale S4 Encoder & Auxiliary Gradient Injector Head.

ReID design (FairMOT/AMOT idea, adapted to a query-based detector):
  • A single appearance ReID head samples the SHARED feature map at each
    predicted box via deformable attention. The feature map is NOT detached,
    so the encoder/backbone receive appearance gradient — this restores the
    "joint" coupling of JDE.
  • The object query and predicted box are passed in detached: they act only
    as POINTERS (where to look), shielding the decoder's localisation and
    classification semantics from the ReID gradient.
  • Detection vs ReID are balanced by learnable uncertainty weights inside the
    criterion — not by a hard stop-gradient.
"""
import torch
import torch.nn as nn

from .backbone import DINOv3STAs
from .hybrid_encoder import HybridEncoder
from .decoder import DEIMTransformer
from .dfine_decoder import MSDeformableAttention
from .feat_fusion import FeatFusion, S4AuxiliaryHeadV2


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

class _GradScale(torch.autograd.Function):
    """Identity in the forward pass; scales the gradient by `scale` in backward.

    Lets the ReID branch couple to the shared trunk while optionally damping
    how strongly its gradient perturbs detection features (scale in [0, 1]).
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


def _largest_divisor(dim: int, candidates=(8, 6, 4, 3, 2, 1)) -> int:
    """Pick the largest head-count in `candidates` that divides `dim`."""
    for c in candidates:
        if dim % c == 0:
            return c
    return 1


class ReIDHead(nn.Module):
    """Appearance ReID head for a query-based (DETR / D-FINE) JDE tracker.

    Pipeline
    --------
    Each object query is a *pointer* saying WHERE to look; the appearance
    *content* is read from the shared feature map by deformable attention:

        query (detached) ─┐
                          ├─► deform-attn(sample feat at box) ─► appearance
        box   (detached) ─┘                                          │
        query (detached) ───────────────────────────────────────────┤
                                                                     ▼
                                            fuse([query, appearance]) → emb_raw
                                                                     │
                                              LayerNorm neck (LNNeck)│
                                                                     ▼
                                                                    emb

    Gradient policy (set by the model, not here):
      • `query` and `box` arrive **detached** → pointers only, so the decoder's
        localisation / classification semantics are shielded from ReID gradient.
      • `feat` arrives **connected** → appearance gradient flows into the
        encoder / backbone, giving the shared trunk identity-aware features
        (the "joint" coupling of JDE). Detection vs ReID are balanced later by
        learnable uncertainty weights in the criterion.

    Dual output (BNNeck principle) keeps the two ReID objectives from fighting
    over one vector:
      • ``emb_raw`` (pre-neck)  → TripletLoss  (free Euclidean space)
      • ``emb``     (post-neck) → CE / ArcFace + inference (stable manifold)

    A per-sample LayerNorm neck is used instead of BatchNorm because the number
    of matched objects per image varies a lot in DETR-style training, which
    makes batch statistics unreliable.
    """

    def __init__(self, hidden_dim: int, reid_dim: int,
                 num_heads: int = 8, num_points: int = 8):
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = _largest_divisor(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads

        # --- appearance sampling (single-scale deformable attention) ---
        self.value_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.deform_attn = MSDeformableAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            num_levels=1, num_points=num_points, method='default',
        )
        self.norm_q    = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # --- fuse [query, appearance] -> reid embedding ---
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        )

        # --- LNNeck: per-sample normalisation, batch-size independent ---
        # affine=False keeps the vector on a stable manifold for cosine / CE
        # and does not distort the sphere used by the angular objective.
        self.neck = nn.LayerNorm(reid_dim, elementwise_affine=False)

    def _build_value(self, feat: torch.Tensor):
        """feat [B,C,H,W] -> (value_list, spatial_shapes) for MSDeformableAttention.

        Produces value[0] of shape [B, n_head, head_dim, H*W], which is the
        layout expected by `deformable_attention_core_func_v2` (value_shape
        'default').
        """
        B, C, H, W = feat.shape
        v = feat.flatten(2).permute(0, 2, 1)          # [B, HW, C]
        v = self.value_proj(v)                        # [B, HW, C]
        head_dim = C // self.num_heads
        v = v.reshape(B, H * W, self.num_heads, head_dim)
        v = v.permute(0, 2, 3, 1).contiguous()        # [B, n_head, head_dim, HW]
        return [v], [[H, W]]

    @torch.no_grad()
    def dense_appearance(self, feat: torch.Tensor) -> torch.Tensor:
        """Per-pixel appearance map in the SAME space the deform-attn samples.

        Applies the head's `value_proj` densely so that a track template
        (bilinearly sampled from this map) and the dense map live in one metric
        space — the prerequisite for the cross-frame correlation in
        `appearance_motion.predict_centers`.

        Args:
            feat : [B, C, H, W] shared appearance feature map (`reid_feat`).
        Returns:
            [B, C, H, W] value-projected dense appearance map.
        """
        B, C, H, W = feat.shape
        v = feat.flatten(2).permute(0, 2, 1)          # [B, HW, C]
        v = self.value_proj(v)                        # [B, HW, C]
        return v.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, query: torch.Tensor, boxes: torch.Tensor,
                feat: torch.Tensor) -> dict:
        """
        Args:
            query : [B, N, C]    detached decoder hidden state (pointer)
            boxes : [B, N, 4]    detached predicted boxes, cxcywh in [0, 1]
            feat  : [B, C, H, W] shared feature map (kept connected)
        Returns:
            {'emb': post-neck embedding (CE + eval),
             'emb_raw': pre-neck embedding (triplet)}
        """
        if feat is None:
            raise ValueError("ReIDHead requires the shared feature map `feat`.")

        value_list, spatial_shapes = self._build_value(feat)
        q   = self.norm_q(query)
        ref = boxes.unsqueeze(2)                       # [B, N, 1, 4]

        appearance = self.deform_attn(q, ref, value_list, spatial_shapes)
        appearance = self.norm_attn(appearance)

        emb_raw = self.fuse(torch.cat([q, appearance], dim=-1))
        emb     = self.neck(emb_raw)
        return {'emb': emb, 'emb_raw': emb_raw}


class S4AuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, in_channels // 2), in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class S4LightBranch(nn.Module):
    def __init__(self, c1_ch: int, hidden_dim: int):
        super().__init__()
        self.lateral = nn.Conv2d(c1_ch, hidden_dim, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, c1: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
        x = self.lateral(c1)
        x = x + F.interpolate(s8, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(x)


class FalconJDEModel(nn.Module):
    def __init__(
        self,
        backbone: DINOv3STAs,
        encoder:  HybridEncoder,
        decoder:  DEIMTransformer,
        reid_dim: int  = 128,
        use_s4:   bool = False,
        use_s4_aux: bool = True,
        sta_dim:  int  = 0,
        use_reid: bool = True,
        reid_num_points: int = 8,
        reid_grad_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone   = backbone
        self.encoder    = encoder
        self.decoder    = decoder
        # Set True by the tracking script to emit the dense appearance map
        # (Query Appearance-Motion). Off by default → training / eval-mAP unchanged.
        self.return_reid_dense = False
        self.use_s4     = use_s4
        self.use_s4_aux = use_s4_aux
        self.use_reid   = use_reid
        # Strength of the ReID gradient that reaches the shared trunk via the
        # feature map. 1.0 = full JDE coupling; lower it (e.g. 0.1) only if
        # detection visibly degrades once coupling is enabled.
        self.reid_grad_scale = reid_grad_scale

        if use_reid:
            self.reid_head = ReIDHead(
                decoder.hidden_dim, reid_dim,
                num_heads=8, num_points=reid_num_points,
            )

        if use_s4:
            self.s4_branch   = FeatFusion(sta_dim, decoder.hidden_dim, n_blocks=2)
            self.s4_aux_head = S4AuxiliaryHeadV2(decoder.hidden_dim)

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)            
        feats = self.encoder(feats)         

        if self.use_s4:
            c1 = getattr(self.backbone, '_s4_feat', None)
            p2 = self.s4_branch(c1, feats[0])
            dec_feats = [p2, feats[0], feats[1]]
            reid_feat = p2
        else:
            dec_feats = feats
            reid_feat = feats[0]            

        out = self.decoder(dec_feats, targets)

        if self.use_s4 and self.use_s4_aux and self.training:
            out['pred_s4_aux'] = self.s4_aux_head(p2)   

        if 'eval_hs' in out and self.use_reid:
            hs = out.pop('eval_hs')
            pred_boxes = out['pred_boxes']

            # Gradient policy — the heart of conflict-free JDE:
            #   • query (hs) and boxes are DETACHED → pointers only, so the
            #     decoder's localisation / classification semantics are shielded
            #     from the ReID gradient.
            #   • reid_feat stays CONNECTED (optionally gradient-scaled) → the
            #     appearance gradient flows into the encoder / backbone so the
            #     shared trunk learns identity features. Detection vs ReID are
            #     balanced by uncertainty weights in the criterion, not here.
            # hs_det    = hs.detach()
            # boxes_det = pred_boxes.detach()
            reid_feat = grad_scale(reid_feat, self.reid_grad_scale)

            reid_out = self.reid_head(hs, pred_boxes, reid_feat)
            out['pred_reid']     = reid_out['emb']      # post-neck → CE + eval
            out['pred_reid_raw'] = reid_out['emb_raw']  # pre-neck  → triplet

            # Dense appearance map for Query Appearance-Motion (tracking only).
            # Cheap (one Linear over H*W); gated so training/eval-mAP are untouched.
            if getattr(self, 'return_reid_dense', False) and not self.training:
                out['reid_dense']        = self.reid_head.dense_appearance(reid_feat)
                out['reid_dense_stride'] = 4 if self.use_s4 else 8
        elif 'eval_hs' in out:
            out.pop('eval_hs')

        return out

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy') and m is not self:
                m.convert_to_deploy()
        return self


# ---------------------------------------------------------------------------
# Factory (Phần này giữ nguyên hoàn toàn như code của bạn)
# ---------------------------------------------------------------------------

def load_pretrained(model, ckpt_path, verbose=True):
    import os
    from collections import defaultdict
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        if verbose:
            print(f'[load_pretrained] no checkpoint at "{ckpt_path}" — skipping')
        return {'loaded': 0, 'total_model': len(model.state_dict())}

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'ema', 'model_ema'):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
        if 'module' in ckpt and isinstance(ckpt['module'], dict) and len(ckpt) <= 2:
            ckpt = ckpt['module']
    state = ckpt

    def _strip(k):
        for p in ('module.', 'model.', 'deim.', 'ema.'):
            if k.startswith(p):
                k = k[len(p):]
        return k
    state = {_strip(k): v for k, v in state.items() if hasattr(v, 'shape')}

    model_sd = model.state_dict()
    matched, shape_mismatch = {}, []
    used = set()

    for k, v in state.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                matched[k] = v; used.add(k)
            else:
                shape_mismatch.append(k)

    unmatched_model = {k: t for k, t in model_sd.items() if k not in matched}
    free_ckpt = {k: v for k, v in state.items() if k not in used}
    remapped = 0
    if unmatched_model and free_ckpt:
        def suffix(k, n=4):
            return '.'.join(k.split('.')[-n:])
        ck_by_suf = defaultdict(list)
        for k in free_ckpt:
            ck_by_suf[suffix(k)].append(k)
        md_by_suf = defaultdict(list)
        for k in unmatched_model:
            md_by_suf[suffix(k)].append(k)
        for suf, mkeys in md_by_suf.items():
            ckeys = ck_by_suf.get(suf, [])
            if len(mkeys) == 1 and len(ckeys) == 1:
                mk, ckk = mkeys[0], ckeys[0]
                if model_sd[mk].shape == free_ckpt[ckk].shape:
                    matched[mk] = free_ckpt[ckk]; used.add(ckk); remapped += 1

    missing    = [k for k in model_sd if k not in matched]
    unexpected = [k for k in state if k not in used and k not in shape_mismatch]
    model.load_state_dict(matched, strict=False)

    tot, got = defaultdict(int), defaultdict(int)
    for k in model_sd:
        g = k.split('.')[0]; tot[g] += 1
        if k in matched:
            got[g] += 1

    stats = {
        'loaded': len(matched), 'total_model': len(model_sd),
        'exact': len(matched) - remapped, 'remapped': remapped,
        'shape_mismatch': len(shape_mismatch), 'missing': len(missing),
        'unexpected': len(unexpected), 'per_module': dict(got),
    }
    if verbose:
        print(f'[load_pretrained] {ckpt_path}')
        print(f'  loaded {len(matched)}/{len(model_sd)} tensors '
              f'(exact={stats["exact"]}, suffix-remapped={remapped}, '
              f'shape-mismatch={len(shape_mismatch)}, missing={len(missing)}, '
              f'unexpected-in-ckpt={len(unexpected)})')
        for g in sorted(tot):
            flag = '   <-- NOT LOADED' if got[g] == 0 and tot[g] > 0 else ''
            print(f'    {g:<12} {got[g]:>4}/{tot[g]:<4}{flag}')
    return stats


def build_falcon_jde(opt) -> FalconJDEModel:
    num_classes = opt.num_classes
    reid_dim    = getattr(opt, 'reid_dim', 128)
    eval_size   = getattr(opt, 'eval_spatial_size', None)
    use_s4      = getattr(opt, 'use_s4', False)

    backbone = DINOv3STAs(
        name                = getattr(opt, 'dinov3_name',              'vit_tiny'),
        weights_path        = getattr(opt, 'dinov3_weights',           ''),
        interaction_indexes = getattr(opt, 'dinov3_interaction_indexes', [3, 7, 11]),
        embed_dim           = getattr(opt, 'dinov3_embed_dim',         192),
        num_heads           = getattr(opt, 'dinov3_num_heads',         3),
        patch_size          = 16,
        use_sta             = getattr(opt, 'use_sta',                  True),
        conv_inplane        = getattr(opt, 'conv_inplane',             16),
        hidden_dim          = getattr(opt, 'hidden_dim',               192),
        finetune            = True,
    )

    hidden_dim = backbone.hidden_dim
    sta_dim  = getattr(opt, 'conv_inplane', 16) if use_s4 else 0

    encoder_in_channels  = [hidden_dim] * 3
    encoder_feat_strides = [8, 16, 32]
    encoder_use_idx      = [2]

    encoder = HybridEncoder(
        in_channels       = encoder_in_channels,
        feat_strides      = encoder_feat_strides,
        hidden_dim        = hidden_dim,
        nhead             = 8,
        dim_feedforward   = getattr(opt, 'enc_dim_ff',    512),
        expansion         = getattr(opt, 'enc_expansion', 0.34),
        depth_mult        = getattr(opt, 'enc_depth_mult', 0.67),
        use_encoder_idx   = encoder_use_idx,
        num_encoder_layers= 1,
        fuse_op           = 'sum',
        version           = 'deim',
    )

    if use_s4:
        feat_channels = [hidden_dim] * 3
        feat_strides  = [4, 8, 16]
        num_levels    = 3
        num_points    = [6, 4, 4]
    else:
        feat_channels = [hidden_dim] * 3
        feat_strides  = [8, 16, 32]
        num_levels    = 3
        num_points    = [3, 6, 3]

    decoder = DEIMTransformer(
        num_classes       = num_classes,
        hidden_dim        = hidden_dim,
        num_queries       = getattr(opt, 'num_queries',   300),
        feat_channels     = feat_channels,
        feat_strides      = feat_strides,
        num_levels        = num_levels,
        num_points        = num_points,
        nhead             = 8,
        num_layers        = getattr(opt, 'num_dec_layers', 4),
        dim_feedforward   = getattr(opt, 'dec_dim_ff',    512),
        activation        = 'silu',
        mlp_act           = 'silu',
        num_denoising     = getattr(opt, 'num_denoising', 100),
        label_noise_ratio = 0.5,
        box_noise_scale   = 1.0,
        eval_spatial_size = tuple(eval_size) if eval_size else None,
        eval_idx          = -1,
        aux_loss          = True,
        reg_max           = getattr(opt, 'reg_max', 32),
        reg_scale         = 4.0,
    )

    model = FalconJDEModel(
        backbone, encoder, decoder,
        reid_dim=reid_dim,
        use_s4=use_s4,
        use_s4_aux=getattr(opt, 'use_s4_aux', True),
        use_reid=getattr(opt, 'use_reid', True),
        sta_dim=sta_dim,
        reid_num_points=getattr(opt, 'reid_num_points', 8),
        reid_grad_scale=getattr(opt, 'reid_grad_scale', 1.0),
    )

    ckpt_path = getattr(opt, 'deim_pretrained', '')
    if ckpt_path:
        load_pretrained(model, ckpt_path, verbose=True)

    return model






# """
# FalconJDEModel — DINOv3STAs + HybridEncoder + DEIMTransformer + ReID head.

# Updated with Deep-embedded 4-scale S4 Encoder & Auxiliary Gradient Injector Head.

# ReID design (FairMOT/AMOT idea, adapted to a query-based detector):
#   • A single appearance ReID head samples the SHARED feature map at each
#     predicted box via deformable attention. The feature map is NOT detached,
#     so the encoder/backbone receive appearance gradient — this restores the
#     "joint" coupling of JDE.
#   • The object query and predicted box are passed in detached: they act only
#     as POINTERS (where to look), shielding the decoder's localisation and
#     classification semantics from the ReID gradient.
#   • Detection vs ReID are balanced by learnable uncertainty weights inside the
#     criterion — not by a hard stop-gradient.
# """
# import torch
# import torch.nn as nn

# from .backbone import DINOv3STAs
# from .hybrid_encoder import HybridEncoder
# from .decoder import DEIMTransformer
# from .dfine_decoder import MSDeformableAttention
# from .feat_fusion import FeatFusion, S4AuxiliaryHeadV2


# # ---------------------------------------------------------------------------
# # Gradient utilities
# # ---------------------------------------------------------------------------

# class _GradScale(torch.autograd.Function):
#     """Identity in the forward pass; scales the gradient by `scale` in backward.

#     Lets the ReID branch couple to the shared trunk while optionally damping
#     how strongly its gradient perturbs detection features (scale in [0, 1]).
#     """
#     @staticmethod
#     def forward(ctx, x, scale):
#         ctx.scale = scale
#         return x

#     @staticmethod
#     def backward(ctx, grad):
#         return grad * ctx.scale, None


# def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
#     if scale == 1.0:
#         return x
#     return _GradScale.apply(x, scale)


# def _largest_divisor(dim: int, candidates=(8, 6, 4, 3, 2, 1)) -> int:
#     """Pick the largest head-count in `candidates` that divides `dim`."""
#     for c in candidates:
#         if dim % c == 0:
#             return c
#     return 1


# class ReIDHead(nn.Module):
#     """Appearance ReID head for a query-based (DETR / D-FINE) JDE tracker.

#     Pipeline
#     --------
#     Each object query is a *pointer* saying WHERE to look; the appearance
#     *content* is read from the shared feature map by deformable attention:

#         query (detached) ─┐
#                           ├─► deform-attn(sample feat at box) ─► appearance
#         box   (detached) ─┘                                          │
#         query (detached) ───────────────────────────────────────────┤
#                                                                      ▼
#                                             fuse([query, appearance]) → emb_raw
#                                                                      │
#                                               LayerNorm neck (LNNeck)│
#                                                                      ▼
#                                                                     emb

#     Gradient policy (set by the model, not here):
#       • `query` and `box` arrive **detached** → pointers only, so the decoder's
#         localisation / classification semantics are shielded from ReID gradient.
#       • `feat` arrives **connected** → appearance gradient flows into the
#         encoder / backbone, giving the shared trunk identity-aware features
#         (the "joint" coupling of JDE). Detection vs ReID are balanced later by
#         learnable uncertainty weights in the criterion.

#     Dual output (BNNeck principle) keeps the two ReID objectives from fighting
#     over one vector:
#       • ``emb_raw`` (pre-neck)  → TripletLoss  (free Euclidean space)
#       • ``emb``     (post-neck) → CE / ArcFace + inference (stable manifold)

#     A per-sample LayerNorm neck is used instead of BatchNorm because the number
#     of matched objects per image varies a lot in DETR-style training, which
#     makes batch statistics unreliable.
#     """

#     def __init__(self, hidden_dim: int, reid_dim: int,
#                  num_heads: int = 8, num_points: int = 8):
#         super().__init__()
#         if hidden_dim % num_heads != 0:
#             num_heads = _largest_divisor(hidden_dim)
#         self.hidden_dim = hidden_dim
#         self.num_heads  = num_heads

#         # --- appearance sampling (single-scale deformable attention) ---
#         self.value_proj  = nn.Linear(hidden_dim, hidden_dim)
#         self.deform_attn = MSDeformableAttention(
#             embed_dim=hidden_dim, num_heads=num_heads,
#             num_levels=1, num_points=num_points, method='default',
#         )
#         self.norm_q    = nn.LayerNorm(hidden_dim)
#         self.norm_attn = nn.LayerNorm(hidden_dim)

#         # --- fuse [query, appearance] -> reid embedding ---
#         self.fuse = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.SiLU(inplace=True),
#             nn.Linear(hidden_dim, reid_dim),
#         )

#         # --- LNNeck: per-sample normalisation, batch-size independent ---
#         # affine=False keeps the vector on a stable manifold for cosine / CE
#         # and does not distort the sphere used by the angular objective.
#         self.neck = nn.LayerNorm(reid_dim, elementwise_affine=False)

#     def _build_value(self, feat: torch.Tensor):
#         """feat [B,C,H,W] -> (value_list, spatial_shapes) for MSDeformableAttention.

#         Produces value[0] of shape [B, n_head, head_dim, H*W], which is the
#         layout expected by `deformable_attention_core_func_v2` (value_shape
#         'default').
#         """
#         B, C, H, W = feat.shape
#         v = feat.flatten(2).permute(0, 2, 1)          # [B, HW, C]
#         v = self.value_proj(v)                        # [B, HW, C]
#         head_dim = C // self.num_heads
#         v = v.reshape(B, H * W, self.num_heads, head_dim)
#         v = v.permute(0, 2, 3, 1).contiguous()        # [B, n_head, head_dim, HW]
#         return [v], [[H, W]]

#     @torch.no_grad()
#     def dense_appearance(self, feat: torch.Tensor) -> torch.Tensor:
#         """Per-pixel appearance map in the SAME space the deform-attn samples.

#         Applies the head's `value_proj` densely so that a track template
#         (bilinearly sampled from this map) and the dense map live in one metric
#         space — the prerequisite for the cross-frame correlation in
#         `appearance_motion.predict_centers`.

#         Args:
#             feat : [B, C, H, W] shared appearance feature map (`reid_feat`).
#         Returns:
#             [B, C, H, W] value-projected dense appearance map.
#         """
#         B, C, H, W = feat.shape
#         v = feat.flatten(2).permute(0, 2, 1)          # [B, HW, C]
#         v = self.value_proj(v)                        # [B, HW, C]
#         return v.permute(0, 2, 1).reshape(B, C, H, W)

#     def forward(self, query: torch.Tensor, boxes: torch.Tensor,
#                 feat: torch.Tensor) -> dict:
#         """
#         Args:
#             query : [B, N, C]    detached decoder hidden state (pointer)
#             boxes : [B, N, 4]    detached predicted boxes, cxcywh in [0, 1]
#             feat  : [B, C, H, W] shared feature map (kept connected)
#         Returns:
#             {'emb': post-neck embedding (CE + eval),
#              'emb_raw': pre-neck embedding (triplet)}
#         """
#         if feat is None:
#             raise ValueError("ReIDHead requires the shared feature map `feat`.")

#         value_list, spatial_shapes = self._build_value(feat)
#         q   = self.norm_q(query)
#         ref = boxes.unsqueeze(2)                       # [B, N, 1, 4]

#         appearance = self.deform_attn(q, ref, value_list, spatial_shapes)
#         appearance = self.norm_attn(appearance)

#         emb_raw = self.fuse(torch.cat([q, appearance], dim=-1))
#         emb     = self.neck(emb_raw)
#         return {'emb': emb, 'emb_raw': emb_raw}


# class S4AuxiliaryHead(nn.Module):
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(min(32, in_channels // 2), in_channels // 2),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(in_channels // 2, 1, kernel_size=1)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.conv(x)


# class S4LightBranch(nn.Module):
#     def __init__(self, c1_ch: int, hidden_dim: int):
#         super().__init__()
#         self.lateral = nn.Conv2d(c1_ch, hidden_dim, 1, bias=False)
#         self.refine = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.SiLU(inplace=True),
#         )

#     def forward(self, c1: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
#         x = self.lateral(c1)
#         x = x + F.interpolate(s8, size=x.shape[-2:], mode='bilinear', align_corners=False)
#         return self.refine(x)


# class FalconJDEModel(nn.Module):
#     def __init__(
#         self,
#         backbone: DINOv3STAs,
#         encoder:  HybridEncoder,
#         decoder:  DEIMTransformer,
#         reid_dim: int  = 128,
#         use_s4:   bool = False,
#         use_s4_aux: bool = True,
#         sta_dim:  int  = 0,
#         use_reid: bool = True,
#         reid_num_points: int = 8,
#         reid_grad_scale: float = 1.0,
#     ):
#         super().__init__()
#         self.backbone   = backbone
#         self.encoder    = encoder
#         self.decoder    = decoder
#         # Set True by the tracking script to emit the dense appearance map
#         # (Query Appearance-Motion). Off by default → training / eval-mAP unchanged.
#         self.return_reid_dense = False
#         self.use_s4     = use_s4
#         self.use_s4_aux = use_s4_aux
#         self.use_reid   = use_reid
#         # Strength of the ReID gradient that reaches the shared trunk via the
#         # feature map. 1.0 = full JDE coupling; lower it (e.g. 0.1) only if
#         # detection visibly degrades once coupling is enabled.
#         self.reid_grad_scale = reid_grad_scale

#         if use_reid:
#             self.reid_head = ReIDHead(
#                 decoder.hidden_dim, reid_dim,
#                 num_heads=8, num_points=reid_num_points,
#             )

#         if use_s4:
#             self.s4_branch   = FeatFusion(sta_dim, decoder.hidden_dim, n_blocks=2)
#             self.s4_aux_head = S4AuxiliaryHeadV2(decoder.hidden_dim)

#     def forward(self, x: torch.Tensor, targets=None):
#         feats = self.backbone(x)            
#         feats = self.encoder(feats)         

#         if self.use_s4:
#             c1 = getattr(self.backbone, '_s4_feat', None)
#             p2 = self.s4_branch(c1, feats[0])
#             dec_feats = [p2, feats[0], feats[1]]
#             reid_feat = p2
#         else:
#             dec_feats = feats
#             reid_feat = feats[0]            

#         out = self.decoder(dec_feats, targets)

#         if self.use_s4 and self.use_s4_aux and self.training:
#             out['pred_s4_aux'] = self.s4_aux_head(p2)   

#         if 'eval_hs' in out and self.use_reid:
#             hs = out.pop('eval_hs')
#             pred_boxes = out['pred_boxes']

#             # Gradient policy — the heart of conflict-free JDE:
#             #   • query (hs) and boxes are DETACHED → pointers only, so the
#             #     decoder's localisation / classification semantics are shielded
#             #     from the ReID gradient.
#             #   • reid_feat stays CONNECTED (optionally gradient-scaled) → the
#             #     appearance gradient flows into the encoder / backbone so the
#             #     shared trunk learns identity features. Detection vs ReID are
#             #     balanced by uncertainty weights in the criterion, not here.
#             # hs_det    = hs.detach()
#             # boxes_det = pred_boxes.detach()
#             reid_feat = grad_scale(reid_feat, self.reid_grad_scale)

#             reid_out = self.reid_head(hs, pred_boxes, reid_feat)
#             out['pred_reid']     = reid_out['emb']      # post-neck → CE + eval
#             out['pred_reid_raw'] = reid_out['emb_raw']  # pre-neck  → triplet

#             # Dense appearance map for Query Appearance-Motion (tracking only).
#             # Cheap (one Linear over H*W); gated so training/eval-mAP are untouched.
#             if getattr(self, 'return_reid_dense', False) and not self.training:
#                 out['reid_dense']        = self.reid_head.dense_appearance(reid_feat)
#                 out['reid_dense_stride'] = 4 if self.use_s4 else 8
#         elif 'eval_hs' in out:
#             out.pop('eval_hs')

#         return out

#     def deploy(self):
#         self.eval()
#         for m in self.modules():
#             if hasattr(m, 'convert_to_deploy') and m is not self:
#                 m.convert_to_deploy()
#         return self


# # ---------------------------------------------------------------------------
# # Factory (Phần này giữ nguyên hoàn toàn như code của bạn)
# # ---------------------------------------------------------------------------

# def load_pretrained(model, ckpt_path, verbose=True):
#     import os
#     from collections import defaultdict
#     if not (ckpt_path and os.path.isfile(ckpt_path)):
#         if verbose:
#             print(f'[load_pretrained] no checkpoint at "{ckpt_path}" — skipping')
#         return {'loaded': 0, 'total_model': len(model.state_dict())}

#     try:
#         ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     except TypeError:
#         ckpt = torch.load(ckpt_path, map_location='cpu')

#     if isinstance(ckpt, dict):
#         for key in ('model', 'state_dict', 'ema', 'model_ema'):
#             if key in ckpt and isinstance(ckpt[key], dict):
#                 ckpt = ckpt[key]
#                 break
#         if 'module' in ckpt and isinstance(ckpt['module'], dict) and len(ckpt) <= 2:
#             ckpt = ckpt['module']
#     state = ckpt

#     def _strip(k):
#         for p in ('module.', 'model.', 'deim.', 'ema.'):
#             if k.startswith(p):
#                 k = k[len(p):]
#         return k
#     state = {_strip(k): v for k, v in state.items() if hasattr(v, 'shape')}

#     model_sd = model.state_dict()
#     matched, shape_mismatch = {}, []
#     used = set()

#     for k, v in state.items():
#         if k in model_sd:
#             if model_sd[k].shape == v.shape:
#                 matched[k] = v; used.add(k)
#             else:
#                 shape_mismatch.append(k)

#     unmatched_model = {k: t for k, t in model_sd.items() if k not in matched}
#     free_ckpt = {k: v for k, v in state.items() if k not in used}
#     remapped = 0
#     if unmatched_model and free_ckpt:
#         def suffix(k, n=4):
#             return '.'.join(k.split('.')[-n:])
#         ck_by_suf = defaultdict(list)
#         for k in free_ckpt:
#             ck_by_suf[suffix(k)].append(k)
#         md_by_suf = defaultdict(list)
#         for k in unmatched_model:
#             md_by_suf[suffix(k)].append(k)
#         for suf, mkeys in md_by_suf.items():
#             ckeys = ck_by_suf.get(suf, [])
#             if len(mkeys) == 1 and len(ckeys) == 1:
#                 mk, ckk = mkeys[0], ckeys[0]
#                 if model_sd[mk].shape == free_ckpt[ckk].shape:
#                     matched[mk] = free_ckpt[ckk]; used.add(ckk); remapped += 1

#     missing    = [k for k in model_sd if k not in matched]
#     unexpected = [k for k in state if k not in used and k not in shape_mismatch]
#     model.load_state_dict(matched, strict=False)

#     tot, got = defaultdict(int), defaultdict(int)
#     for k in model_sd:
#         g = k.split('.')[0]; tot[g] += 1
#         if k in matched:
#             got[g] += 1

#     stats = {
#         'loaded': len(matched), 'total_model': len(model_sd),
#         'exact': len(matched) - remapped, 'remapped': remapped,
#         'shape_mismatch': len(shape_mismatch), 'missing': len(missing),
#         'unexpected': len(unexpected), 'per_module': dict(got),
#     }
#     if verbose:
#         print(f'[load_pretrained] {ckpt_path}')
#         print(f'  loaded {len(matched)}/{len(model_sd)} tensors '
#               f'(exact={stats["exact"]}, suffix-remapped={remapped}, '
#               f'shape-mismatch={len(shape_mismatch)}, missing={len(missing)}, '
#               f'unexpected-in-ckpt={len(unexpected)})')
#         for g in sorted(tot):
#             flag = '   <-- NOT LOADED' if got[g] == 0 and tot[g] > 0 else ''
#             print(f'    {g:<12} {got[g]:>4}/{tot[g]:<4}{flag}')
#     return stats


# def build_falcon_jde(opt) -> FalconJDEModel:
#     num_classes = opt.num_classes
#     reid_dim    = getattr(opt, 'reid_dim', 128)
#     eval_size   = getattr(opt, 'eval_spatial_size', None)
#     use_s4      = getattr(opt, 'use_s4', False)

#     backbone = DINOv3STAs(
#         name                = getattr(opt, 'dinov3_name',              'vit_tiny'),
#         weights_path        = getattr(opt, 'dinov3_weights',           ''),
#         interaction_indexes = getattr(opt, 'dinov3_interaction_indexes', [3, 7, 11]),
#         embed_dim           = getattr(opt, 'dinov3_embed_dim',         192),
#         num_heads           = getattr(opt, 'dinov3_num_heads',         3),
#         patch_size          = 16,
#         use_sta             = getattr(opt, 'use_sta',                  True),
#         conv_inplane        = getattr(opt, 'conv_inplane',             16),
#         hidden_dim          = getattr(opt, 'hidden_dim',               192),
#         finetune            = True,
#     )

#     hidden_dim = backbone.hidden_dim
#     sta_dim  = getattr(opt, 'conv_inplane', 16) if use_s4 else 0

#     encoder_in_channels  = [hidden_dim] * 3
#     encoder_feat_strides = [8, 16, 32]
#     encoder_use_idx      = [2]

#     encoder = HybridEncoder(
#         in_channels       = encoder_in_channels,
#         feat_strides      = encoder_feat_strides,
#         hidden_dim        = hidden_dim,
#         nhead             = 8,
#         dim_feedforward   = getattr(opt, 'enc_dim_ff',    512),
#         expansion         = getattr(opt, 'enc_expansion', 0.34),
#         depth_mult        = getattr(opt, 'enc_depth_mult', 0.67),
#         use_encoder_idx   = encoder_use_idx,
#         num_encoder_layers= 1,
#         fuse_op           = 'sum',
#         version           = 'deim',
#     )

#     if use_s4:
#         feat_channels = [hidden_dim] * 3
#         feat_strides  = [4, 8, 16]
#         num_levels    = 3
#         num_points    = [6, 4, 4]
#     else:
#         feat_channels = [hidden_dim] * 3
#         feat_strides  = [8, 16, 32]
#         num_levels    = 3
#         num_points    = [3, 6, 3]

#     decoder = DEIMTransformer(
#         num_classes       = num_classes,
#         hidden_dim        = hidden_dim,
#         num_queries       = getattr(opt, 'num_queries',   300),
#         feat_channels     = feat_channels,
#         feat_strides      = feat_strides,
#         num_levels        = num_levels,
#         num_points        = num_points,
#         nhead             = 8,
#         num_layers        = getattr(opt, 'num_dec_layers', 4),
#         dim_feedforward   = getattr(opt, 'dec_dim_ff',    512),
#         activation        = 'silu',
#         mlp_act           = 'silu',
#         num_denoising     = getattr(opt, 'num_denoising', 100),
#         label_noise_ratio = 0.5,
#         box_noise_scale   = 1.0,
#         eval_spatial_size = tuple(eval_size) if eval_size else None,
#         eval_idx          = -1,
#         aux_loss          = True,
#         reg_max           = getattr(opt, 'reg_max', 32),
#         reg_scale         = 4.0,
#     )

#     model = FalconJDEModel(
#         backbone, encoder, decoder,
#         reid_dim=reid_dim,
#         use_s4=use_s4,
#         use_s4_aux=getattr(opt, 'use_s4_aux', True),
#         sta_dim=sta_dim,
#         reid_num_points=getattr(opt, 'reid_num_points', 8),
#         reid_grad_scale=getattr(opt, 'reid_grad_scale', 1.0),
#     )

#     ckpt_path = getattr(opt, 'deim_pretrained', '')
#     if ckpt_path:
#         load_pretrained(model, ckpt_path, verbose=True)

#     return model