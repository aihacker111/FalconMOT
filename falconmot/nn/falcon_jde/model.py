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

import math
import torch
import torch.nn as nn

from .backbone import DINOv3STAs
from .neck.hybrid_encoder import HybridEncoder
from .head.decoder import DEIMTransformer
from .head.dfine_decoder import MSDeformableAttention
from .ops.feat_fusion import (
    FeatFusion, S4AuxiliaryHeadV2, ConvNeXtV2Block, LayerNorm2d,
)
from .ops.safa import SparseFeatFusion


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
#  DenseReIDHead — decoupled ReID branch, QAM-compatible, cons-coherent.
# ===========================================================================


class _ReIDResBlock(nn.Module):
    """Light residual with a small (3x3 DW) kernel -> adds depth while keeping the
    correlation peak sharp for QAM, preventing identity bleeding on small objects.

    drop_path (stochastic depth): chong overfit single-frame o TOWER — jitter/shrink
    chi augment o input sampling, con tower van co the hoc thuoc texture nen cuc bo
    cua tung sequence train; drop-path ngau nhien ca residual branch xu ly dung cho do."""
    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dw   = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)
        self.pw   = nn.Conv2d(dim, dim, 1, bias=False)
        self.act  = nn.GELU()
        self.drop_path = float(drop_path)

    def forward(self, x):
        y = self.pw(self.act(self.norm(self.dw(x))))
        if self.training and self.drop_path > 0:
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
                    >= self.drop_path).to(x.dtype)
            y = y * keep / (1.0 - self.drop_path)
        return x + y

# ===========================================================================
#  DenseReIDHead — deformable-sampling ReID head (jitter + shrink, DropPath tower)
# ===========================================================================

class DenseReIDHead(nn.Module):
    """Decoupled ReID head: sample appearance tu shared feature map bang
    deformable attention tai vi tri box; query (hs.detach()) chi la POINTER.

    Cac chinh sua so voi ban truoc:
      * SHRINK box la DETERMINISTIC -> ap o CA train va eval (fix train/eval
        mismatch: truoc day eval sample tren full box trong khi train chi
        thay vung loi 0.8x, khien inference "nhin" them mot vanh background
        chua tung duoc train).
      * JITTER la STOCHASTIC -> chi ap khi training.
      * BO context_layer (self-attention giua cac embedding): no lam embedding
        cua object A phu thuoc vao hang xom trong frame -> hang xom doi giua
        cac frame => temporal instability, hai cho EMA smooth_feat + cosine
        matching cua tracker.
      * Tower dung DropPath (stochastic depth) chong overfit single-frame.

    NOTE (use_s4_dense): chi danh cho use_s4=False (reid_feat = feats[0],
    stride-8). Khi use_s4=True, reid_feat = p2 DA la stride-4 va c1 da duoc
    fuse trong s4_branch — bat use_s4_dense se fuse c1 hai lan (co hai).
    """
    def __init__(self, hidden_dim, reid_dim, num_heads=8, num_points=8,
                 use_s4_dense=False, s4_in_ch=None,
                 tower_depth=2, query_gate_init=0.1, detach_input=True,
                 box_shrink_factor=0.8, jitter_scale=0.05,
                 tower_drop_path=0.1,
                 **kwargs):
        super().__init__()
        nh = num_heads if reid_dim % num_heads == 0 else _largest_divisor(reid_dim)
        self.hidden_dim    = hidden_dim
        self.reid_dim      = reid_dim
        self.num_heads     = nh
        self.use_s4_dense  = bool(use_s4_dense and s4_in_ch is not None)
        self.detach_input  = bool(detach_input)
        self.box_shrink_factor = box_shrink_factor
        self.jitter_scale  = jitter_scale

        # (1) HIGH-RES (chi khi use_s4=False): c1(stride-4) + reid_feat(stride-8)
        #     -> field stride-4. Xem NOTE o docstring.
        if self.use_s4_dense:
            self.s4_fuse = FeatFusion(s4_in_ch, hidden_dim, n_blocks=1)

        # (2) OWN tower: project to reid_dim, then residual DW3x3 blocks (DropPath).
        self.in_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, reid_dim, kernel_size=1, bias=False),
            LayerNorm2d(reid_dim),
        )
        self.dense_tower = nn.Sequential(
            *[_ReIDResBlock(reid_dim, drop_path=tower_drop_path)
              for _ in range(max(1, tower_depth))],
            LayerNorm2d(reid_dim),
        )

        # (3) SPARSE PATH — query lam pointer cho deformable sampling.
        self.q_proj      = nn.Linear(hidden_dim, reid_dim)
        self.norm_q      = nn.LayerNorm(reid_dim)
        self.deform_attn = MSDeformableAttention(
            embed_dim=reid_dim, num_heads=nh,
            num_levels=1, num_points=num_points, method='default',
        )
        self.norm_attn   = nn.LayerNorm(reid_dim)

        # (4) Content = appearance (dominant) + gated residual tu query content.
        #     Voi query = hs.detach(): day la semantic content DONG (hop le),
        #     mang thong tin class-level; gate init 0.1 giu appearance dominant.
        #     KHONG tang gate — hs khong giup tach car-A khoi car-B.
        self.app_ffn = nn.Sequential(
            nn.Linear(reid_dim, reid_dim),
            nn.SiLU(inplace=True),
            nn.Linear(reid_dim, reid_dim),
        )
        self.q_content  = nn.Linear(hidden_dim, reid_dim)
        self.query_gate = nn.Parameter(torch.tensor(float(query_gate_init)))

        self.neck = nn.LayerNorm(reid_dim, elementwise_affine=False)

    def build_emb_map(self, reid_feat, c1=None):
        x = reid_feat
        if self.use_s4_dense and c1 is not None:
            x = self.s4_fuse(c1, reid_feat)
        x = self.in_proj(x)
        return self.dense_tower(x)

    def _build_value(self, emb_map):
        B, C, H, W = emb_map.shape
        v = emb_map.flatten(2).permute(0, 2, 1)
        hd = C // self.num_heads
        v = v.reshape(B, H * W, self.num_heads, hd).permute(0, 2, 3, 1).contiguous()
        return [v], [[H, W]]

    def forward(self, query, boxes, reid_feat, c1=None, return_dense=False):
        # Gradient isolation (tuy chon; mac dinh False khi dung OSD — projector
        # truc giao da co lap conflict, gradient ReID nuoi trunk qua subspace rieng).
        if self.detach_input:
            reid_feat = reid_feat.detach()
            if c1 is not None:
                c1 = c1.detach()

        emb_map = self.build_emb_map(reid_feat, c1)
        value_list, spatial_shapes = self._build_value(emb_map)

        q_in = self.norm_q(self.q_proj(query))

        reid_boxes = boxes.clone()

        # SHRINK — deterministic: ap o CA train va eval de phan phoi sampling
        # points nhat quan giua hai pha (fix train/eval mismatch).
        reid_boxes[..., 2:] = reid_boxes[..., 2:] * self.box_shrink_factor

        if self.training:
            # JITTER — stochastic: chi train; lac nhe tam box de ReID khong
            # hoc vet background / vi tri tuyet doi.
            noise_xy = (torch.rand_like(reid_boxes[..., :2]) - 0.5) \
                       * self.jitter_scale * reid_boxes[..., 2:]
            reid_boxes[..., :2] = (reid_boxes[..., :2] + noise_xy).clamp(0.0, 1.0)

        app_raw = self.deform_attn(q_in, reid_boxes.unsqueeze(2), value_list, spatial_shapes)
        app = self.norm_attn(app_raw)

        app     = app + self.app_ffn(app)
        emb_raw = app + self.query_gate * self.q_content(query)

        out = {
            'emb':     self.neck(emb_raw),
            'emb_raw': emb_raw,
            'emb_app': app_raw,
        }
        if return_dense:
            out['emb_map'] = emb_map
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
        import torch.nn.functional as F
        x = self.lateral(c1)
        x = x + F.interpolate(s8, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(x)


# ===========================================================================
#  OSD — Orthogonal Subspace Decoupling
# ===========================================================================

import torch.nn.functional as Fn


class OrthoSubspaceSplit(nn.Module):
    """OSD — Orthogonal Subspace Decoupling.

    Hoc mot phep xoay truc giao Q thuoc O(C) tren khong gian channel, roi tach
    thanh hai khong gian con bu truc giao:
        F_det = P_det . F = Q^T . M_det . Q . F      (detection doc)
        F_id  = P_id  . F = Q^T . M_id  . Q . F      (ReID doc)
    voi M_det + M_id = I (mask cheo 0/1). Vi range(P_det) vuong goc range(P_id):
        <dL_det/dF, dL_id/dF> = 0   (chinh xac, moi batch, moi thoi diem)
    => conflict tai diem re nhanh bi triet tieu bang hinh hoc, khong can
    uncertainty weighting hay grad scale. Q duoc giu tren manifold O(C)
    bang torch parametrization (khong can re-project thu cong).
    """

    def __init__(self, dim: int, id_ratio: float = 0.25):
        super().__init__()
        assert 0.0 < id_ratio < 1.0
        self.dim  = dim
        self.c_id = max(8, int(round(dim * id_ratio)))
        # Q luu duoi dang Linear(C, C) khong bias, parametrize truc giao.
        self.rot = nn.utils.parametrizations.orthogonal(
            nn.Linear(dim, dim, bias=False))

    def _Q(self):
        return self.rot.weight                      # (C, C), Q^T Q = I

    def forward(self, x: torch.Tensor):
        """x: (B, C, H, W) -> (f_det, f_id, decorr_loss | None)

        f_det, f_id giu nguyen so channel C (chinh la P_det.x, P_id.x) nen
        decoder va reid head KHONG can doi kien truc. decorr_loss chi tinh
        khi training.
        """
        B, C, H, W = x.shape
        Q  = self._Q()                              # (C, C)
        w  = Q.unsqueeze(-1).unsqueeze(-1)          # 1x1 conv kernel  (Q.F)
        wT = Q.t().unsqueeze(-1).unsqueeze(-1)      # Q^T . F

        z = Fn.conv2d(x, w)                         # z = Q.x  (norm-preserving)
        z_det, z_id = z[:, :-self.c_id], z[:, -self.c_id:]

        zeros_id  = torch.zeros_like(z_id)
        zeros_det = torch.zeros_like(z_det)
        f_det = Fn.conv2d(torch.cat([z_det, zeros_id],  dim=1), wT)  # P_det.x
        f_id  = Fn.conv2d(torch.cat([zeros_det, z_id],  dim=1), wT)  # P_id.x

        decorr = self._decorr(z_det, z_id) if self.training else None
        return f_det, f_id, decorr

    @staticmethod
    def _decorr(z_det: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        """Barlow-style cross-covariance penalty giua hai subspace (da xoay),
        chuan hoa theo std de scale-invariant. Ngan mang "lach" bang tuong
        quan thong ke du da truc giao hinh hoc."""
        B, Cd, H, W = z_det.shape
        a = z_det.flatten(2)                        # (B, Cd, HW)
        b = z_id.flatten(2)                         # (B, Ci, HW)
        a = (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-5)
        b = (b - b.mean(-1, keepdim=True)) / (b.std(-1, keepdim=True) + 1e-5)
        cov = torch.bmm(a, b.transpose(1, 2)) / (H * W)   # (B, Cd, Ci)
        return (cov ** 2).mean()


class FalconJDEModel(nn.Module):
    def __init__(
        self,
        backbone,           # DINOv3STAs
        encoder,            # HybridEncoder
        decoder,            # DEIMTransformer
        reid_dim: int  = 128,
        use_s4:   bool = False,
        use_s4_aux: bool = True,
        sta_dim:  int  = 0,
        use_reid: bool = True,
        reid_num_points: int = 8,
        reid_grad_scale: float = 1.0,   # giu API; 1.0 = khong dung grad scale
        reid_use_s4_dense=False,
        reid_s4_in_ch=None,
        use_safa: bool = False,
        safa_keep_ratio: float = 0.25,
        use_osd: bool = True,           # [OSD] bat Orthogonal Subspace Decoupling
        osd_id_ratio: float = 0.33,     # [OSD] ty le channel cho subspace ReID (rank 64 @ dim 192)
    ):
        super().__init__()
        self.backbone   = backbone
        self.encoder    = encoder
        self.decoder    = decoder
        self.return_reid_dense = False
        self.use_s4     = use_s4
        self.use_s4_aux = use_s4_aux
        self.use_reid   = use_reid
        self.use_safa   = use_safa
        self.reid_grad_scale = reid_grad_scale
        self.use_osd    = use_osd

        if use_osd:
            self.osd = OrthoSubspaceSplit(decoder.hidden_dim, id_ratio=osd_id_ratio)

        if use_reid:
            # Guard: use_s4=True nghia la reid_feat = p2 (stride-4, c1 da fuse
            # trong s4_branch) — use_s4_dense luc nay se fuse c1 LAN THU HAI.
            if use_s4 and reid_use_s4_dense:
                print('[FalconJDE][warn] use_s4=True: reid_feat (p2) da la stride-4; '
                      'tat reid_use_s4_dense de tranh fuse c1 hai lan.')
                reid_use_s4_dense = False

            self.reid_head = DenseReIDHead(
                decoder.hidden_dim, reid_dim,
                num_heads=8, num_points=reid_num_points,
                use_s4_dense=reid_use_s4_dense,
                s4_in_ch=reid_s4_in_ch,
                tower_drop_path=0.1,
                # OSD da co lap conflict bang projector truc giao, nen feature
                # KHONG detach — giu co-adaptation kieu JDE/MOTIP: gradient ReID
                # nuoi trunk qua subspace rieng, khong dung subspace detection.
                detach_input=False,
                jitter_scale=0.05,          # giam tu 0.1: object VisDrone rat nho
                box_shrink_factor=0.8,
            )

        if use_s4:
            if use_safa:
                # SAFA: entropy-gated sparse S4 fusion (replaces dense FeatFusion).
                self.s4_branch = SparseFeatFusion(
                    sta_dim, decoder.hidden_dim, n_blocks=2,
                    scorer_in_ch=decoder.hidden_dim, keep_ratio=safa_keep_ratio)
            else:
                self.s4_branch = FeatFusion(sta_dim, decoder.hidden_dim, n_blocks=2)
            self.s4_aux_head = S4AuxiliaryHeadV2(decoder.hidden_dim)

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        c1 = getattr(self.backbone, '_s4_feat', None)

        if self.use_s4:
            if self.use_safa:
                p2, ent_logit = self.s4_branch(c1, feats[0])
            else:
                p2 = self.s4_branch(c1, feats[0])
                ent_logit = None
            shared = p2
        else:
            shared = feats[0]
            ent_logit = None

        # ---------------- [OSD] tach shared map thanh 2 subspace truc giao ----
        decorr = None
        if self.use_osd:
            f_det, f_id, decorr = self.osd(shared)
        else:
            f_det, f_id = shared, shared

        if self.use_s4:
            dec_feats = [f_det, feats[0], feats[1]]
        else:
            dec_feats = [f_det, feats[1], feats[2]]
        reid_feat = f_id
        # ----------------------------------------------------------------------

        out = self.decoder(dec_feats, targets)

        if self.training and decorr is not None:
            out['pred_decorr'] = decorr
        if self.use_s4 and self.use_s4_aux and self.training:
            out['pred_s4_aux'] = self.s4_aux_head(f_det)
        if self.use_safa and ent_logit is not None and self.training:
            out['pred_entropy'] = ent_logit

        if 'eval_hs' in out and self.use_reid:
            hs = out.pop('eval_hs')
            pred_boxes = out['pred_boxes']
            want_dense = self.training or getattr(self, 'return_reid_dense', False)

            # Thiet ke chuan (JDE/MOTIP-consensus):
            #   * hs.detach()    — query chi la POINTER: ID loss khong sua
            #                      semantics phan loai / dinh vi cua decoder.
            #   * boxes.detach() — ID gradient khong dung box regression.
            #   * reid_feat      — KHONG detach; gradient chay ve trunk nhung
            #                      nam tron trong subspace P_id ⟂ P_det (OSD).
            if self.reid_grad_scale != 1.0:
                reid_feat = grad_scale(reid_feat, self.reid_grad_scale)

            reid_out = self.reid_head(
                query=hs.detach(),
                boxes=pred_boxes.detach(),
                reid_feat=reid_feat,
                c1=c1,
                return_dense=want_dense,
            )
            out['pred_reid']     = reid_out['emb']
            out['pred_reid_raw'] = reid_out['emb_raw']
            if self.training:
                out['pred_reid_app'] = reid_out['emb_app']
            if 'emb_map' in reid_out:
                if self.training:
                    out['pred_reid_map'] = reid_out['emb_map']
                if getattr(self, 'return_reid_dense', False) and not self.training:
                    out['reid_dense']        = reid_out['emb_map'][0]
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
        scale_adaptive    = getattr(opt, 'use_safa', False) and getattr(opt, 'safa_scale_adaptive', True),
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
        reid_s4_in_ch=getattr(opt, 'conv_inplane', 32),
        use_safa=getattr(opt, 'use_safa', False),
        safa_keep_ratio=getattr(opt, 'safa_keep_ratio', 1.0),
        use_osd=getattr(opt, 'use_osd', True),
        osd_id_ratio=getattr(opt, 'osd_id_ratio', 0.33),
    )

    ckpt_path = getattr(opt, 'deim_pretrained', '')
    if ckpt_path:
        load_pretrained(model, ckpt_path, verbose=True)

    return model