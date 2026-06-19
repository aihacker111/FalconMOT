"""
FalconJDEModel — DINOv3STAs + HybridEncoder + DEIMTransformer + ReID head.

Updated with Deep-embedded 4-scale S4 Encoder & Auxiliary Gradient Injector Head.
Thêm: ContextAwareReIDHead (Spatial-aware Self-Attention) và cơ chế detach() bảo vệ nhánh Detection.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3STAs
from .hybrid_encoder import HybridEncoder
from .decoder import DEIMTransformer
from .dfine_decoder import MSDeformableAttention
from .deim_utils import RMSNorm
from .feat_fusion import FeatFusion, S4AuxiliaryHeadV2


class ReIDHead(nn.Module):
    """Maps per-query hidden state → ReID embedding vector (baseline MLP)."""
    def __init__(self, hidden_dim: int, reid_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.net(x)


def _largest_divisor(dim: int, candidates=(8, 6, 4, 3, 2, 1)) -> int:
    """Pick the largest head-count in `candidates` that divides `dim`."""
    for c in candidates:
        if dim % c == 0:
            return c
    return 1


class TransformerReIDHead(nn.Module):
    """
    Appearance-aware ReID head: Lấy mẫu đặc trưng thực tế từ Feature Map quanh box.
    """
    def __init__(self, hidden_dim: int, reid_dim: int,
                 num_heads: int = 8, num_points: int = 8):
        super().__init__()
        num_heads = _largest_divisor(hidden_dim) if hidden_dim % num_heads else num_heads
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads

        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.deform_attn = MSDeformableAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            num_levels=1, num_points=num_points, method='default',
        )
        # self.norm_q    = RMSNorm(hidden_dim)
        # self.norm_attn = RMSNorm(hidden_dim)
        self.norm_q    = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        )
        
        # BẮT BUỘC: LayerNorm Bottleneck giúp ArcFace hội tụ. 
        # elementwise_affine=False để không làm lệch phân phối vector trên mặt cầu.
        self.bottleneck = nn.LayerNorm(reid_dim, elementwise_affine=False)

    def _build_value(self, feat: torch.Tensor):
        B, C, H, W = feat.shape
        v = feat.flatten(2).permute(0, 2, 1)          
        v = self.value_proj(v)                        
        head_dim = C // self.num_heads
        v = v.reshape(B, H * W, self.num_heads, head_dim)
        v = v.permute(0, 2, 3, 1).contiguous()        
        return [v], [[H, W]]

    def forward(self, det_hs, pred_boxes, feat, **kwargs) -> torch.Tensor:
        # det_hs: [B, N, C] - Đã được detach() từ Forward
        # pred_boxes: [B, N, 4] - Đã được detach() từ Forward
        # feat: [B, C, H, W] - KHÔNG detach để Encoder học ReID

        value_list, spatial_shapes = self._build_value(feat)
        q   = self.norm_q(det_hs)
        ref = pred_boxes.unsqueeze(2) # [B, N, 1, 4]

        # Lấy mẫu đặc trưng ngoại quan dựa trên tọa độ box dự đoán
        appearance = self.deform_attn(q, ref, value_list, spatial_shapes) 
        appearance = self.norm_attn(appearance)

        fused = torch.cat([det_hs, appearance], dim=-1)
        emb = self.fuse(fused)
        
        return self.bottleneck(emb) # Trả về vector đã chuẩn hóa


class ContextAwareReIDHead(nn.Module):
    """
    Spatial Context-Aware ReID Head.
    Sử dụng Self-Attention để các Object Queries giao tiếp với nhau,
    kết hợp với Spatial Embedding từ tọa độ bounding box dự đoán.
    Phù hợp xử lý tracking đám đông (drone) khi ngoại quan vật thể nghèo nàn.
    """
    def __init__(self, hidden_dim: int, reid_dim: int, num_heads: int = 8, dim_feedforward: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reid_dim = reid_dim

        # 1. MLP map Bounding Box (4) -> Spatial Embedding (hidden_dim)
        self.bbox_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2. Một tầng Transformer Encoder Layer tiêu chuẩn
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 3. Projection cuối cùng ra reid_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim)
        )

    def forward(self, det_hs: torch.Tensor, pred_boxes: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # det_hs: [B, N, hidden_dim]
        # pred_boxes: [B, N, 4]
        
        spatial_emb = self.bbox_embed(pred_boxes)
        combined_hs = det_hs + spatial_emb 
        
        context_hs = self.transformer_layer(combined_hs)
        reid_emb = self.proj(context_hs) 
        
        return reid_emb


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
        reid_head_type: str = 'context_aware', # Mặc định trỏ sang head mới
        reid_num_points: int = 8,
    ):
        super().__init__()
        self.backbone  = backbone
        self.encoder   = encoder
        self.decoder   = decoder
        self.use_s4    = use_s4
        self.use_s4_aux = use_s4_aux
        self.use_reid  = use_reid
        self.reid_head_type = reid_head_type

        if use_reid:
            if reid_head_type == 'transformer':
                self.reid_head = TransformerReIDHead(
                    decoder.hidden_dim, reid_dim, num_heads=8, num_points=reid_num_points)
            elif reid_head_type == 'context_aware':
                self.reid_head = ContextAwareReIDHead(
                    decoder.hidden_dim, reid_dim)
            else:
                self.reid_head = ReIDHead(decoder.hidden_dim, reid_dim)

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

        if 'eval_hs' in out:
            hs = out.pop('eval_hs')
            pred_boxes = out['pred_boxes']
            
            # QUAN TRỌNG: Tất cả các head giờ đều nhận input đã được .detach()
            # để đảm bảo Gradient của loss_reid không chạy ngược phá hỏng Decoder.
            hs_det = hs.detach()
            boxes_det = pred_boxes.detach()
            
            if self.reid_head_type == 'transformer':
                out['pred_reid'] = self.reid_head(hs_det, boxes_det, reid_feat)
            elif self.reid_head_type == 'context_aware':
                out['pred_reid'] = self.reid_head(hs_det, boxes_det)
            else:
                out['pred_reid'] = self.reid_head(hs_det)

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
        sta_dim=sta_dim,
        reid_head_type=getattr(opt, 'reid_head_type', 'context_aware'), # Cập nhật default tại đây
        reid_num_points=getattr(opt, 'reid_num_points', 8),
    )

    ckpt_path = getattr(opt, 'deim_pretrained', '')
    if ckpt_path:
        load_pretrained(model, ckpt_path, verbose=True)

    return model