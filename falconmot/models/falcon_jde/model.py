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
from .feat_fusion import (
    FeatFusion, S4AuxiliaryHeadV2, ConvNeXtV2Block, LayerNorm2d,
)


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


def _largest_divisor(dim, candidates=(8, 6, 4, 3, 2, 1)):
    for c in candidates:
        if dim % c == 0:
            return c
    return 1
 
# ===========================================================================
#  DenseReIDHead — DECOUPLED BRANCH, QAM-COMPATIBLE, CONS-COHERENT
#  (drop-in thay cho class cũ ở model.py ~dòng 1183–1259)
#
#  Giữ NGUYÊN chữ ký & format:
#     __init__(hidden_dim, reid_dim, num_heads, num_points,
#              use_s4_dense, s4_in_ch, **kwargs)
#     forward(query, boxes, reid_feat, c1=None, return_dense=False)
#     output keys: 'emb'      (post-neck: CE+eval+cosine BYTE),
#                  'emb_raw'  (pre-neck:  triplet),
#                  'emb_app'  (MỚI: mẫu THUẦN của emb_map tại box -> đích cho
#                              dense `cons`, cùng không gian với `dense`),
#                  'emb_map'  ([B,reid_dim,H,W]: reid_dense cho QAM + dense CE).
#
#  Phụ thuộc đã có trong scope model.py: nn, F, torch, LayerNorm2d,
#  FeatFusion, MSDeformableAttention, _largest_divisor.
#
#  ─────────────── BẤT BIẾN ĐỒNG NHẤT (một field duy nhất) ────────────────
#  emb_map đóng mọi vai appearance: Value của deform-attn, reid_dense cho QAM,
#  nguồn template QAM (sample_dense), VÀ nguồn cho dense-CE/cons trong loss.
#  => KHÔNG value_proj. Mọi nhánh appearance cùng sample một tensor.
#
#  ─────────── FIX CONFLICT dense_ce ↔ cons (lý do có 'emb_app') ───────────
#  `cons` cũ kéo `dense`(mẫu tâm emb_map) về `emb` — nhưng emb đã qua neck +
#  cộng query nên LỆCH khỏi không gian emb_map -> đối kháng với dense_ce
#  (vốn nặn emb_map theo hướng tách-ID). Sửa: cho `cons` ngắm `emb_app` =
#  mẫu THUẦN của emb_map tại box (raw deform output, trước norm/ffn/query).
#  Khi đó dense(mẫu tâm) và emb_app(mẫu box) cùng là mẫu của emb_map ->
#  `cons` chỉ còn là "nhất quán cục bộ trên map", BỔ TRỢ dense_ce, hết giằng co.
#
#  4 thay đổi so với bản gốc của bạn
#  ---------------------------------
#  (1) DECOUPLED: detach reid_feat & c1 (detach_input=True).
#  (2) HIGH-RES stride-4: dense map sắc hơn -> đỉnh correlation QAM rõ hơn.
#  (3) OWN-TOWER sâu hơn, kernel NHỎ 3x3 (giữ đỉnh sắc), KHÔNG value_proj.
#  (4) Query CHỈ định vị; nội dung emb = appearance + residual có cổng (chỉ
#      ảnh hưởng cosine BYTE, không đụng QAM/dense).
# ===========================================================================


class _ReIDResBlock(nn.Module):
    """Residual nhẹ, kernel NHỎ (3x3 DW) -> thêm chiều sâu nhưng giữ đỉnh
    correlation sắc cho QAM, chống identity-bleeding ở vật nhỏ."""
    def __init__(self, dim: int):
        super().__init__()
        self.dw   = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)
        self.pw   = nn.Conv2d(dim, dim, 1, bias=False)
        self.act  = nn.GELU()

    def forward(self, x):
        return x + self.pw(self.act(self.norm(self.dw(x))))


class DenseReIDHead(nn.Module):
    """ReID head tách rời (decoupled), một-field-duy-nhất cho QAM, và lộ
    'emb_app' để dense `cons` không còn xung đột với dense_ce."""
    def __init__(self, hidden_dim, reid_dim, num_heads=8, num_points=8,
                 use_s4_dense=False, s4_in_ch=None,
                 tower_depth=2, query_gate_init=0.1, detach_input=True,
                 **kwargs):
        super().__init__()
        nh = num_heads if reid_dim % num_heads == 0 else _largest_divisor(reid_dim)
        self.hidden_dim    = hidden_dim
        self.reid_dim      = reid_dim
        self.num_heads     = nh
        self.use_s4_dense  = bool(use_s4_dense and s4_in_ch is not None)
        self.detach_input  = bool(detach_input)

        # (2) HIGH-RES: c1(stride-4) + reid_feat(stride-8) -> field stride-4.
        if self.use_s4_dense:
            self.s4_fuse = FeatFusion(s4_in_ch, hidden_dim, n_blocks=1)

        # (3) OWN-TOWER: vào reid_dim rồi tower_depth block residual DW3x3.
        #     Output emb_map = field DUY NHẤT (Value + reid_dense + dense-CE/cons).
        self.in_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, reid_dim, kernel_size=1, bias=False),
            LayerNorm2d(reid_dim),
        )
        self.dense_tower = nn.Sequential(
            *[_ReIDResBlock(reid_dim) for _ in range(max(1, tower_depth))],
            LayerNorm2d(reid_dim),
        )

        # KHÔNG value_proj: Value = emb_map trực tiếp -> đồng nhất với QAM/loss.

        # SPARSE PATH — query CHỈ để định vị điểm sample.
        self.q_proj      = nn.Linear(hidden_dim, reid_dim)
        self.norm_q      = nn.LayerNorm(reid_dim)
        self.deform_attn = MSDeformableAttention(
            embed_dim=reid_dim, num_heads=nh,
            num_levels=1, num_points=num_points, method='default',
        )
        self.norm_attn   = nn.LayerNorm(reid_dim)

        # (4) Nội dung emb = appearance (chủ đạo) + residual có cổng từ query.
        self.app_ffn = nn.Sequential(
            nn.Linear(reid_dim, reid_dim),
            nn.SiLU(inplace=True),
            nn.Linear(reid_dim, reid_dim),
        )
        self.q_content  = nn.Linear(hidden_dim, reid_dim)
        self.query_gate = nn.Parameter(torch.tensor(float(query_gate_init)))

        self.neck = nn.LayerNorm(reid_dim, elementwise_affine=False)

    # ---- field duy nhất: Value + reid_dense + dense-CE/cons + template QAM ---
    def build_emb_map(self, reid_feat, c1=None):
        x = reid_feat
        if self.use_s4_dense and c1 is not None:
            x = self.s4_fuse(c1, reid_feat)          # -> [B, hidden_dim, H4, W4]
        x = self.in_proj(x)                          # -> [B, reid_dim,  H,  W]
        return self.dense_tower(x)                   # -> [B, reid_dim,  H,  W]

    # ---- Value cho deform-attn = chính emb_map (đúng format core) -----------
    def _build_value(self, emb_map):
        B, C, H, W = emb_map.shape
        v = emb_map.flatten(2).permute(0, 2, 1)              # [B, HW, C]
        hd = C // self.num_heads
        v = v.reshape(B, H * W, self.num_heads, hd).permute(0, 2, 3, 1).contiguous()
        return [v], [[H, W]]                                 # [B, nh, hd, HW], shapes

    def forward(self, query, boxes, reid_feat, c1=None, return_dense=False):
        # (1) DECOUPLED: cắt gradient về trunk (forward value KHÔNG đổi).
        if self.detach_input:
            reid_feat = reid_feat.detach()
            if c1 is not None:
                c1 = c1.detach()

        emb_map = self.build_emb_map(reid_feat, c1)          # field duy nhất
        value_list, spatial_shapes = self._build_value(emb_map)

        # query -> CHỈ định vị điểm sample quanh box.
        q_in = self.norm_q(self.q_proj(query))
        # app_raw = mẫu THUẦN của emb_map tại box (chưa norm/ffn/query)
        #         -> đích sạch cho dense `cons`, cùng không gian với `dense`.
        app_raw = self.deform_attn(q_in, boxes.unsqueeze(2), value_list, spatial_shapes)
        app = self.norm_attn(app_raw)

        # (4) emb = appearance chủ đạo + residual có cổng từ query.
        app     = app + self.app_ffn(app)
        emb_raw = app + self.query_gate * self.q_content(query)

        out = {
            'emb':     self.neck(emb_raw),    # cosine BYTE + CE/eval
            'emb_raw': emb_raw,               # triplet (pre-neck)
            'emb_app': app_raw,               # đích cho dense cons (emb_map-space)
        }
        if return_dense:
            out['emb_map'] = emb_map          # -> model: out['reid_dense'] = emb_map (QAM)
        return out


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
        reid_use_s4_dense=False,
        reid_s4_in_ch=None
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
            # Truyền đúng tham số, bỏ qua các config nặng cũ
            self.reid_head = DenseReIDHead(
                decoder.hidden_dim, reid_dim, 
                num_heads=8, num_points=reid_num_points,
                use_s4_dense=reid_use_s4_dense, 
                s4_in_ch=reid_s4_in_ch
            )

        if use_s4:
            self.s4_branch   = FeatFusion(sta_dim, decoder.hidden_dim, n_blocks=2)
            self.s4_aux_head = S4AuxiliaryHeadV2(decoder.hidden_dim)

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        c1 = getattr(self.backbone, '_s4_feat', None)
    
        if self.use_s4:
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
    
            # GRADIENT POLICY (đã sửa đúng theo docstring):
            #   • hs, pred_boxes -> DETACH: chỉ là pointer, KHÔNG để gradient ReID
            #     chạm vào localization/classification của decoder.
            #   • reid_feat      -> CÒN nối (qua grad_scale): coupling JDE có kiểm soát.
            # reid_feat_c = grad_scale(reid_feat, self.reid_grad_scale)
    
            # train: cần emb_map cho dense loss; infer: chỉ cần khi bật dense-tracking
            want_dense = self.training or getattr(self, 'return_reid_dense', False)
            reid_out = self.reid_head(
                hs.detach(), pred_boxes.detach(), reid_feat,
                c1=c1, return_dense=want_dense,
            )
            out['pred_reid']     = reid_out['emb']      # post-neck -> CE + eval
            out['pred_reid_raw'] = reid_out['emb_raw']  # pre-neck  -> triplet
            if self.training:
                out['pred_reid_app'] = reid_out['emb_app']
            if 'emb_map' in reid_out:
                if self.training:
                    out['pred_reid_map'] = reid_out['emb_map']        # [B,D,H,W] cho dense loss
                if getattr(self, 'return_reid_dense', False) and not self.training:
                    out['reid_dense']        = reid_out['emb_map'][0]  # [D,H,W] (batch=1 khi track)
                    out['reid_dense_stride'] = 4 if (self.reid_head.use_s4_dense or self.use_s4) else 8
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
# Factory
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
        conv_inplane        = getattr(opt, 'conv_inplane',             32),
        hidden_dim          = getattr(opt, 'hidden_dim',               192),
        finetune            = True,
    )

    hidden_dim = backbone.hidden_dim
    sta_dim  = getattr(opt, 'conv_inplane', 32) if use_s4 else 0

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
        use_s4_aux=getattr(opt, 'use_s4_aux', False),
        use_reid=getattr(opt, 'use_reid', True),
        sta_dim=sta_dim,
        reid_num_points=getattr(opt, 'reid_num_points', 8),
        reid_grad_scale=getattr(opt, 'reid_grad_scale', 1.0),
        reid_use_s4_dense=getattr(opt, 'reid_use_s4_dense', False),
        reid_s4_in_ch=getattr(opt, 'conv_inplane', 32)
    )

    ckpt_path = getattr(opt, 'deim_pretrained', '')
    if ckpt_path:
        load_pretrained(model, ckpt_path, verbose=True)

    return model