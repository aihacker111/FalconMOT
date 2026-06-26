"""
s4_module_v2.py
================
Thiết kế lại nhánh S4 (stride-4) + S4 auxiliary loss cho FalconJDE.

Mục tiêu: ÍT THAM SỐ nhưng MẠNH trên feature map, cân bằng speed/accuracy,
đặc biệt cho vật thể siêu nhỏ (VisDrone / UAVDT).

Hai phần:
  (A) S4FusionBranch  : thay cho S4LightBranch
        - Gated fusion (chi tiết c1 ⊕ ngữ nghĩa S8) thay vì cộng bilinear thô
        - 1-2 block ConvNeXtV2 (DW7x7 + GRN) -> trộn đặc trưng mạnh, gần như 0 param
  (B) Gaussian-Focal center heatmap aux loss : thay cho box-fill BCE
        - Splat Gaussian theo tâm + bán kính theo kích thước box
        - Penalty-reduced focal loss (CenterNet) -> gradient tập trung, hết starvation
        - Hỗ trợ trọng số nghịch theo size -> ép model nhạy vật thể nhỏ

Tích hợp: xem phần INTEGRATION ở cuối file.
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
    """LayerNorm trên channel cho tensor NCHW (kiểu ConvNeXt channels_first)."""
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
    Tăng độ tương phản / đa dạng giữa các kênh, chống feature-collapse.
    Chỉ tốn 2*C tham số -> gần như free nhưng cải thiện chất lượng feature map rõ rệt.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)          # [B,C,1,1] năng lượng mỗi kênh
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)            # chuẩn hoá theo kênh
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """DW(7x7) -> LN -> PW-expand -> GELU -> GRN -> PW-project, residual.
    Receptive field lớn (7x7 depthwise) + GRN, rất ít param so với conv thường.
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
    """Nhánh kết hợp đặc trưng Stride-4 siêu nhẹ (Đã tối ưu FLOPs & VRAM).
    
    Thay vì Concat + Conv 1x1 nặng nề, phiên bản này dùng ngữ nghĩa tầng sâu (S8) 
    qua một bộ lọc Depthwise Spatial Gate để dập nhiễu nền trên tầng chi tiết (C1), 
    sau đó thực hiện phép cộng (Element-wise Addition).
    """
    def __init__(self, c1_ch: int, hidden_dim: int,
                 mid_dim: int = None, n_blocks: int = 2, expand: float = 2.0):
        super().__init__()
        mid = mid_dim or hidden_dim // 2

        # Lớp giảm kênh (Projection) giữ nguyên
        self.detail = nn.Conv2d(c1_ch, mid, 1, bias=False)        # Chi tiết từ c1
        self.sem    = nn.Conv2d(hidden_dim, mid, 1, bias=False)   # Ngữ nghĩa từ S8

        # CẢI TIẾN 1: Thay thế Gate Concat bằng Depthwise Spatial Gate
        # Chỉ tốn mid * 3 * 3 tham số (cực kỳ ít) thay vì (mid * 2) * mid của Conv 1x1 cũ
        self.gate_spatial = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=True),
            nn.Sigmoid()
        )

        # CẢI TIẾN 3: scalar học được điều tiết mức bồi semantic vào detail.
        # init=1.0 -> ban đầu cân bằng (x = d + g*s); model tự dial lên/xuống.
        # (Muốn ép "pure-detail-first" thì init 0.0.)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # CẢI TIẾN 4: depthwise kernel 7x7 (đúng như docstring) -> receptive field
        # lớn ở S4 mà gần như không tốn thêm param/FLOP. Bù cho việc p2 không có
        # global context nào.
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(mid, expand, k=7) for _ in range(n_blocks)])

        self.out    = nn.Conv2d(mid, hidden_dim, 1, bias=False)
        
        # CẢI TIẾN 2: Đồng nhất sang LayerNorm2d thay vì BatchNorm2d
        # Giúp ổn định luồng gradient khi train với Batch Size nhỏ (2, 4, 8)
        self.out_norm = LayerNorm2d(hidden_dim)

    def forward(self, c1: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
        # 1. Trích xuất đặc trưng hình học và ngữ nghĩa về cùng số kênh `mid`
        d = self.detail(c1)                                       # [B, mid, H4, W4]
        s = self.sem(s8)                                          # [B, mid, H8, W8]
        
        # 2. Nội suy S8 lên cùng độ phân giải với C1
        s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False) # [B, mid, H4, W4]
        
        # 3. Sinh màng lọc (gate) từ Semantic S8 — nơi định vị vật thể tốt
        g = self.gate_spatial(s)                                  # [B, mid, H4, W4] trong (0, 1)

        # 4. CẢI TIẾN CỐT LÕI: detail (d) làm xương sống high-res, giữ cạnh sắc cho
        #    vật nhỏ; semantic (s, đã up-sample nên mờ) chỉ BỒI vào như residual có
        #    gate + scale alpha — thay vì cộng nguyên si làm loãng localization.
        #    Đây là dạng top-down kiểu FPN nhưng tối ưu cho object siêu nhỏ.
        x = d + self.alpha * (g * s)
        
        # 5. Đi qua các block tinh lọc đặc trưng và bung kênh đầu ra
        x = self.blocks(x)
        return self.out_norm(self.out(x))

class S4AuxiliaryHeadV2(nn.Module):
    """Head objectness nhẹ trên P2 (stride-4). Xuất logit 1 kênh (chưa sigmoid).
    Sâu vừa đủ để định hình feature mà vẫn rẻ (DW + PW).
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
        # Focal-prior bias: sigmoid(-2.19) ≈ 0.1 -> ổn định loss âm ở bước đầu
        nn.init.constant_(self.conv[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =====================================================================
# (B) GAUSSIAN-FOCAL CENTER-HEATMAP AUX LOSS  (CenterNet-style)
# =====================================================================

def _gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> int:
    """Bán kính Gaussian đảm bảo IoU >= min_overlap (CornerNet/CenterNet)."""
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


def _draw_gaussian(hmap: torch.Tensor, cx: int, cy: int, radius: int):
    """Vẽ Gaussian (lấy max) vào hmap[H,W] tại tâm (cx,cy). In-place trên GPU."""
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
    masked_g = g[radius - top:radius + bottom, radius - left:radius + right]
    torch.maximum(masked_h, masked_g, out=masked_h)


@torch.no_grad()
def build_center_heatmaps(targets, H: int, W: int, device,
                          min_overlap: float = 0.7):
    """Sinh target heatmap Gaussian cho cả batch.
    targets[b]['boxes'] : [Ni,4] cxcywh chuẩn hoá [0,1].
    Trả về [B,1,H,W]. (Khuyến nghị: chuyển hàm này vào dataloader/collate để bỏ
    hẳn khỏi bước GPU -> nhanh hơn nữa.)
    """
    B = len(targets)
    hm = torch.zeros((B, 1, H, W), device=device)
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
            _draw_gaussian(hm[b, 0], int(cx[i]), int(cy[i]), r)
    return hm


def gaussian_focal_loss(pred_logits: torch.Tensor, gt_heatmap: torch.Tensor,
                        alpha: float = 2.0, beta: float = 4.0,
                        eps: float = 1e-6) -> torch.Tensor:
    """Penalty-reduced focal loss (CenterNet).
    - pred_logits : [B,1,H,W] (logit, sẽ sigmoid bên trong)
    - gt_heatmap  : [B,1,H,W] (Gaussian soft target, đỉnh = 1.0)
    Gradient tập trung quanh tâm, hạ trọng số negative dễ -> hết 'starvation'.
    """
    pred = pred_logits.sigmoid().clamp(eps, 1 - eps)
    pos = gt_heatmap.eq(1.0).float()
    neg = 1.0 - pos
    neg_weights = (1.0 - gt_heatmap).pow(beta)

    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * neg_weights * neg

    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos