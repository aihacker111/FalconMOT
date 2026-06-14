"""
DINOv3STAs backbone adapter.
Adapted from DEIMv2 — removed @register decorator, replaced SyncBatchNorm with BatchNorm2d.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer


class SpatialPriorModule(nn.Module):
    """Lightweight CNN that extracts dense spatial priors at S8/S16/S32."""

    def __init__(self, inplanes: int = 16):
        super().__init__()
        # S4
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # S8
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
        )
        # S16
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )
        # S32
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)   # S8
        c3 = self.conv3(c2)   # S16
        c4 = self.conv4(c3)   # S32
        return c1, c2, c3, c4  # c1 = stride-4 spatial detail, used by S4Branch


class DINOv3STAs(nn.Module):
    """
    DINOv3 backbone with Spatial-aware Temporal Adapters (STAs).

    Outputs three feature maps: (C2=S8, C3=S16, C4=S32) all projected to hidden_dim.
    Compatible with HybridEncoder expecting in_channels=[hidden_dim]*3.
    """

    def __init__(
        self,
        name: str = 'vit_tiny',
        weights_path: str = None,
        interaction_indexes: list = None,
        finetune: bool = True,
        embed_dim: int = 192,
        num_heads: int = 3,
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 16,
        hidden_dim: int = None,
    ):
        super().__init__()
        interaction_indexes = interaction_indexes or []

        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path and os.path.exists(weights_path):
                print(f'[DINOv3STAs] Loading weights from {weights_path}')
                self.dinov3.load_state_dict(torch.load(weights_path, map_location='cpu'))
            else:
                print('[DINOv3STAs] Training DINOv3 from scratch')
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes
            )
            if weights_path and os.path.exists(weights_path):
                print(f'[DINOv3STAs] Loading ViT weights from {weights_path}')
                self.dinov3._model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            else:
                print('[DINOv3STAs] Training ViT from scratch')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModule(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.hidden_dim = hidden_dim

        # Project fused features to hidden_dim
        sta_c2 = conv_inplane * 2
        sta_c3 = conv_inplane * 4
        sta_c4 = conv_inplane * 4
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + sta_c2, hidden_dim, 1, bias=False),
            nn.Conv2d(embed_dim + sta_c3, hidden_dim, 1, bias=False),
            nn.Conv2d(embed_dim + sta_c4, hidden_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
        ])

    def forward(self, x):
        H_c = x.shape[2] // 16
        W_c = x.shape[3] // 16
        bs  = x.shape[0]

        if self.interaction_indexes and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        # Repeat last layer if fewer than 3 scales returned
        if len(all_layers) == 1:
            all_layers = [all_layers[0]] * 3

        # Reshape tokens → feature maps, upsample/downsample to S8/S16/S32
        # Use scale_factor (not size) so the op is traceable without int(Tensor) calls.
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, (feat, _) in enumerate(all_layers):
            feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c)
            scale_exp = num_scales - i   # 1, 0, -1 for 3 layers
            if scale_exp != 0:
                feat = F.interpolate(feat, scale_factor=2 ** scale_exp,
                                     mode='bilinear', align_corners=False)
            sem_feats.append(feat)

        # Fuse with spatial priors
        if self.use_sta:
            c1_detail, *detail_feats = self.sta(x)  # c1_detail = stride-4 (S4)
            self._s4_feat = c1_detail               # stash for S4Branch in model.py
            fused = [torch.cat([s, d], dim=1) for s, d in zip(sem_feats, detail_feats)]
        else:
            self._s4_feat = None
            fused = sem_feats

        c2 = self.norms[0](self.convs[0](fused[0]))  # S8
        c3 = self.norms[1](self.convs[1](fused[1]))  # S16
        c4 = self.norms[2](self.convs[2](fused[2]))  # S32
        return c2, c3, c4
