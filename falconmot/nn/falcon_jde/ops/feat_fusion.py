"""
s4_module_v2.py
================
Redesigned S4 (stride-4) branch + S4 auxiliary loss for FalconJDE.

Goal: few parameters but a strong feature map, balancing speed/accuracy,
with a focus on extremely small objects (VisDrone / UAVDT).

Two parts:
  (A) S4FusionBranch  : thay cho S4LightBranch
        - Gated fusion (c1 detail + S8 semantics) instead of a raw bilinear add
        - 1-2 ConvNeXtV2 blocks (DW7x7 + GRN) -> strong feature mixing, almost 0 params
  (B) Gaussian-Focal center heatmap aux loss : thay cho box-fill BCE
        - Splat a Gaussian at the center with a radius based on box size
        - Penalty-reduced focal loss (CenterNet) -> focused gradients, no starvation
        - Supports inverse-size weighting -> pushes the model to be sensitive to small objects

Integration: see the INTEGRATION section at the end of this file.
"""
from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# (A) S4 FUSION BRANCH  —  ConvNeXtV2 + GRN + Gated Fusion
# =====================================================================

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (ConvNeXt channels_first style)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXtV2).
    Increases contrast/diversity across channels to prevent feature collapse.
    Costs only 2*C parameters -> nearly free, yet noticeably improves feature-map quality.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)          # [B,C,1,1] per-channel energy
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)            # normalize across channels
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """DW(7x7) -> LN -> PW-expand -> GELU -> GRN -> PW-project, residual.
    Large receptive field (7x7 depthwise) + GRN, far fewer params than a standard conv.
    """
    def __init__(self, dim: int, expand: float = 2.0, k: int = 3):
        super().__init__()
        self.dw   = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=True)
        self.norm = LayerNorm2d(dim)
        hidden    = int(dim * expand)
        self.pw1  = nn.Conv2d(dim, hidden, 1)
        self.act  = nn.GELU()
        self.grn  = GRN(hidden)
        self.pw2  = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        return r + x

class FeatFusion(nn.Module):
    """Ultra-light Stride-4 feature-fusion branch (FLOPs & VRAM optimized).

    Instead of a heavy Concat + 1x1 Conv, this version uses deep semantics (S8)
    through a Depthwise Spatial Gate to suppress background noise on the detail level (C1),
    followed by an element-wise addition.
    """
    def __init__(self, c1_ch: int, hidden_dim: int,
                 mid_dim: int = None, n_blocks: int = 2, expand: float = 2.0):
        super().__init__()
        mid = mid_dim or hidden_dim // 2

        # Channel-reduction (projection) layers, unchanged
        self.detail = nn.Conv2d(c1_ch, mid, 1, bias=False)        # detail from c1
        self.sem    = nn.Conv2d(hidden_dim, mid, 1, bias=False)   # semantics from S8

        # IMPROVEMENT 1: replace the Concat gate with a Depthwise Spatial Gate
        # Costs only mid * 3 * 3 params (tiny) instead of (mid * 2) * mid of the old 1x1 conv
        self.gate_spatial = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=True),
            nn.Sigmoid()
        )

        # IMPROVEMENT 3: a learnable scalar controls how much semantics is added to detail.
        # init=1.0 -> balanced at first (x = d + g*s); the model dials it up/down on its own.
        # (Set init to 0.0 to force a "pure-detail-first" start.)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # IMPROVEMENT 4: depthwise 7x7 kernel (as stated in the docstring) -> a large
        # receptive field at S4 at almost no extra param/FLOP cost. Compensates for p2
        # having no global context.
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(mid, expand, k=7) for _ in range(n_blocks)])

        self.out    = nn.Conv2d(mid, hidden_dim, 1, bias=False)

        # IMPROVEMENT 2: switch to LayerNorm2d instead of BatchNorm2d
        # Stabilizes the gradient flow when training with small batch sizes (2, 4, 8)
        self.out_norm = LayerNorm2d(hidden_dim)

    def forward(self, c1: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
        # 1. Project geometric and semantic features to the same channel count `mid`
        d = self.detail(c1)                                       # [B, mid, H4, W4]
        s = self.sem(s8)                                          # [B, mid, H8, W8]

        # 2. Upsample S8 to the same resolution as C1
        s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False) # [B, mid, H4, W4]

        # 3. Build the gate from the S8 semantics, which localizes objects well
        g = self.gate_spatial(s)                                  # [B, mid, H4, W4] trong (0, 1)

        # 4. CORE IMPROVEMENT: detail (d) is the high-res backbone, keeping sharp edges for
        #    small objects; semantics (s, blurred after upsampling) is only ADDED as a gated,
        #    alpha-scaled residual, instead of a plain add that would dilute localization.
        #    This is an FPN-style top-down design optimized for extremely small objects.
        x = d + self.alpha * (g * s)

        # 5. Pass through the refinement blocks and expand to the output channels
        x = self.blocks(x)
        return self.out_norm(self.out(x))

class S4AuxiliaryHeadV2(nn.Module):
    """Lightweight objectness head on P2 (stride-4). Outputs a 1-channel logit (pre-sigmoid).
    Deep enough to shape features while staying cheap (DW + PW).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),  # DW
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),                              # PW
            nn.GroupNorm(min(32, in_channels // 2), in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),                                                    # logit
        )
        # Focal-prior bias: sigmoid(-2.19) ~ 0.1 -> stabilizes the negative loss early on
        nn.init.constant_(self.conv[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =====================================================================
# (B) GAUSSIAN-FOCAL CENTER-HEATMAP AUX LOSS  (CenterNet-style)
# =====================================================================

def _gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> int:
    """Gaussian radius guaranteeing IoU >= min_overlap (CornerNet/CenterNet)."""
    a1 = 1.0
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / 2

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))
    r2 = (b2 - sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2
    return int(max(0, min(r1, r2, r3)))


def _draw_gaussian(hmap: torch.Tensor, dmap: torch.Tensor, cx: int, cy: int, radius: int):
    """Splat a Gaussian (element-wise max) into hmap[H,W] and sum into dmap[H,W]. In-place on GPU."""
    if radius < 1:
        radius = 1
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    H, W = hmap.shape
    ys = torch.arange(-radius, radius + 1, device=hmap.device, dtype=hmap.dtype)
    g1 = torch.exp(-(ys * ys) / (2 * sigma * sigma))
    g = g1[:, None] * g1[None, :]                                # [d,d]

    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    if right <= -left or bottom <= -top:
        return
        
    masked_h = hmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_d = dmap[cy - top:cy + bottom, cx - left:cx + right]  # [THÊM MỚI]
    masked_g = g[radius - top:radius + bottom, radius - left:radius + right]
    
    # 1. Hmap dùng cho Classification (Lấy Max) - Giữ nguyên của bạn
    torch.maximum(masked_h, masked_g, out=masked_h)
    
    # 2. Dmap dùng cho DMFR (Cộng dồn Density) - Thêm cho Paper của bạn
    masked_d += masked_g


@torch.no_grad()
def build_center_heatmaps(targets, H: int, W: int, device,
                          min_overlap: float = 0.7):
    """Generate the target Gaussian heatmap and Density Map for the whole batch."""
    B = len(targets)
    hm = torch.zeros((B, 1, H, W), device=device)
    density = torch.zeros((B, 1, H, W), device=device) # [THÊM MỚI]
    
    for b in range(B):
        boxes = targets[b]['boxes']
        if boxes.numel() == 0:
            continue
        cx = (boxes[:, 0] * W).clamp(0, W - 1)
        cy = (boxes[:, 1] * H).clamp(0, H - 1)
        bw = (boxes[:, 2] * W).clamp(min=1)
        bh = (boxes[:, 3] * H).clamp(min=1)
        for i in range(boxes.shape[0]):
            r = _gaussian_radius(float(bh[i]), float(bw[i]), min_overlap)
            # Truyền cả hm và density vào hàm vẽ
            _draw_gaussian(hm[b, 0], density[b, 0], int(cx[i]), int(cy[i]), r)
            
    return hm, density # [TRẢ VỀ CẢ HAI]



def gaussian_focal_loss(pred_logits: torch.Tensor, gt_heatmap: torch.Tensor,
                        density_map: torch.Tensor = None, # [THÊM MỚI]
                        alpha: float = 2.0, beta: float = 4.0,
                        gamma_crowd: float = 0.5, # Cường độ tập trung vào đám đông
                        eps: float = 1e-6) -> torch.Tensor:
    """Penalty-reduced focal loss with Density-Modulated Foveal Routing (DMFR)"""
    pred = pred_logits.sigmoid().clamp(eps, 1 - eps)
    pos = gt_heatmap.eq(1.0).float() # Giờ dòng này chạy HOÀN HẢO vì dùng code gốc
    neg = 1.0 - pos
    neg_weights = (1.0 - gt_heatmap).pow(beta)

    # Nếu có bản đồ mật độ, khuếch đại loss ở những vùng chồng chéo
    if density_map is not None:
        # Dùng log1p để chuẩn hóa: 10 objects đè nhau -> log(1+10) = 2.39 -> Hệ số ~ 2.2
        crowd_amplifier = 1.0 + gamma_crowd * torch.log1p(density_map)
    else:
        crowd_amplifier = 1.0

    # DMFR: Khuếch đại loss tại vị trí pos (Nhân với crowd_amplifier)
    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos * crowd_amplifier
    
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * neg_weights * neg

    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos