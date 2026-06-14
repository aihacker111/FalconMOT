"""
FalconJDEModel — DINOv3STAs + HybridEncoder + DEIMTransformer + ReID head.

Updated with Deep-embedded 4-scale S4 Encoder & Auxiliary Gradient Injector Head.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3STAs
from .hybrid_encoder import HybridEncoder
from .decoder import DEIMTransformer


class ReIDHead(nn.Module):
    """Maps per-query hidden state → ReID embedding vector."""
    def __init__(self, hidden_dim: int, reid_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class S4AuxiliaryHead(nn.Module):
    """
    Auxiliary Objectness Head tác động trực tiếp lên nhánh đặc trưng S4 sau Encoder.
    Ép mô hình kích hoạt vùng không gian vật thể nhỏ và duy trì Gradient mạnh mẽ, tránh starvation.
    """
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
    """Nhánh stride-4 nhẹ chỉ phục vụ decoder (không qua encoder nặng).
    Chi tiết từ backbone c1 (stride-4) + ngữ nghĩa từ S8 encoder (bilinear, 0 param)
    -> 1x1 lateral + depthwise-separable refine -> P2 (hidden_dim, stride 4).
    """
    def __init__(self, c1_ch: int, hidden_dim: int):
        super().__init__()
        self.lateral = nn.Conv2d(c1_ch, hidden_dim, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),  # DW
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),                                 # PW
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, c1: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
        x = self.lateral(c1)
        x = x + F.interpolate(s8, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(x)


class FalconJDEModel(nn.Module):
    """
    DEIM-JDE detection model with Native Deep-Fused S4 Scale.

    backbone → (S8, S16, S32) + c1 (S4)
    encoder  → Natively receives 4 scales [S4, S8, S16, S32] and runs Per-Pixel Gated Fusion
    decoder  → Multi-scale Query Prediction
    """
    def __init__(
        self,
        backbone: DINOv3STAs,
        encoder:  HybridEncoder,
        decoder:  DEIMTransformer,
        reid_dim: int  = 128,
        use_s4:   bool = False,
        sta_dim:  int  = 0,
    ):
        super().__init__()
        self.backbone  = backbone
        self.encoder   = encoder
        self.decoder   = decoder
        self.reid_head = ReIDHead(decoder.hidden_dim, reid_dim)
        self.use_s4    = use_s4

        if use_s4:
            # Nhánh P2 nhẹ: c1 (stride-4) + S8 -> hidden_dim, dùng làm level đầu cho decoder
            self.s4_branch   = S4LightBranch(sta_dim, decoder.hidden_dim)
            # Aux objectness trên P2 -> giữ gradient mạnh cho vật thể nhỏ
            self.s4_aux_head = S4AuxiliaryHead(decoder.hidden_dim)

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)            # (S8, S16, S32)
        feats = self.encoder(feats)         # encoder 3-scale (rẻ) -> [S8, S16, S32]

        if self.use_s4:
            # P2 từ nhánh nhẹ (KHÔNG qua encoder); decoder dùng [S4, S8, S16] (bỏ S32)
            c1 = getattr(self.backbone, '_s4_feat', None)
            p2 = self.s4_branch(c1, feats[0])
            dec_feats = [p2, feats[0], feats[1]]
        else:
            dec_feats = feats

        out = self.decoder(dec_feats, targets)

        if self.use_s4 and self.training:
            out['pred_s4_aux'] = self.s4_aux_head(p2)   # aux objectness trên P2
        if 'eval_hs' in out:
            hs = out.pop('eval_hs')  
            out['pred_reid'] = self.reid_head(hs)

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
        conv_inplane        = getattr(opt, 'conv_inplane',             16),
        hidden_dim          = getattr(opt, 'hidden_dim',               192),
        finetune            = True,
    )

    hidden_dim = backbone.hidden_dim
    sta_dim  = getattr(opt, 'conv_inplane', 16) if use_s4 else 0

    # Encoder LUÔN chạy 3-scale [S8,S16,S32]; S4 tách ra thành nhánh nhẹ ở model.forward
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
        # Decoder lấy [S4, S8, S16]: thêm S4, bỏ S32; dồn điểm lấy mẫu nhiều nhất cho S4
        feat_channels = [hidden_dim] * 3
        feat_strides  = [4, 8, 16]
        num_levels    = 3
        num_points    = [6, 4, 3]
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
        reid_dim=reid_dim, use_s4=use_s4, sta_dim=sta_dim,
    )

    ckpt_path = getattr(opt, 'deim_pretrained', '')
    if ckpt_path:
        load_pretrained(model, ckpt_path, verbose=True)

    return model