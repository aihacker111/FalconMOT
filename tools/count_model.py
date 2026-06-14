"""
Count params and GFLOPs of DEIMJDEModel (DINOv3STAs + HybridEncoder + DEIMTransformer + ReID).

S4Branch (when --use_s4):
  Path 1 — bilinear(S8_enc) + PixelShuffle residual (expand PW×4 → PixelShuffle×2 → pw_out zero-init)
  Path 2 — detail_proj: STA stride-4 feature (c1, conv_inplane ch) → hidden_dim  [NEW]
  Path 3 — refine: DWConv3×3 + PW1×1 spatial sharpening                          [NEW]

Usage:
    python tools/count_model.py                        # 3-scale baseline
    python tools/count_model.py --use_s4               # 4-scale + S4Branch
    python tools/count_model.py --compare              # side-by-side 3-scale vs 4-scale
    python tools/count_model.py --load_pretrained path/to/ckpt.pth
    python tools/count_model.py --conv_inplane 32      # larger STA (affects S4Branch detail_proj)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import torch
from falconmot.models.deim_jde.backbone import DINOv3STAs
from falconmot.models.deim_jde.hybrid_encoder import HybridEncoder
from falconmot.models.deim_jde.decoder import DEIMTransformer
from falconmot.models.deim_jde.model import DEIMJDEModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_classes',     type=int,   default=10)
    p.add_argument('--reid_dim',        type=int,   default=128)
    p.add_argument('--input_h',         type=int,   default=480)
    p.add_argument('--input_w',         type=int,   default=864)
    p.add_argument('--conv_inplane',    type=int,   default=16,
                   help='SpatialPriorModule base channels (affects S4Branch detail_proj size)')
    p.add_argument('--use_s4',          action='store_true', default=False,
                   help='add S4Branch (4-scale decoder: S4+S8+S16+S32)')
    p.add_argument('--compare',         action='store_true', default=False,
                   help='print side-by-side 3-scale vs 4-scale table')
    p.add_argument('--device',          default='cpu')
    p.add_argument('--load_pretrained', default='',
                   help='pretrained checkpoint (.pth) — loads matching weights')
    return p.parse_args()


def build_model(num_classes, reid_dim, input_h, input_w,
                use_s4=False, conv_inplane=16, device='cpu') -> DEIMJDEModel:
    backbone = DINOv3STAs(
        name='vit_tiny', embed_dim=192, num_heads=3,
        interaction_indexes=[5, 8, 11], use_sta=True,
        conv_inplane=conv_inplane, hidden_dim=192,
    )
    encoder = HybridEncoder(
        in_channels=[192]*3, feat_strides=[8, 16, 32], hidden_dim=192,
        nhead=8, dim_feedforward=512, expansion=0.34, depth_mult=0.67,
        use_encoder_idx=[2], num_encoder_layers=1, fuse_op='sum', version='deim',
    )

    if use_s4:
        # S4 gets 4 sampling points (largest feature map, most small objects)
        feat_channels = [192] * 4
        feat_strides  = [4, 8, 16, 32]
        num_levels    = 4
        num_points    = [4, 3, 6, 3]
    else:
        feat_channels = [192] * 3
        feat_strides  = [8, 16, 32]
        num_levels    = 3
        num_points    = [3, 6, 3]

    decoder = DEIMTransformer(
        num_classes=num_classes, hidden_dim=192, num_queries=300,
        feat_channels=feat_channels, feat_strides=feat_strides,
        num_levels=num_levels, num_points=num_points,
        nhead=8, num_layers=4, dim_feedforward=512,
        activation='silu', mlp_act='silu', num_denoising=0,
        eval_spatial_size=(input_h, input_w), eval_idx=-1,
        aux_loss=False, reg_max=32, reg_scale=4.0,
    )

    sta_dim = conv_inplane if use_s4 else 0
    model = DEIMJDEModel(
        backbone, encoder, decoder,
        reid_dim=reid_dim, use_s4=use_s4, sta_dim=sta_dim,
    )
    return model.to(device).eval()


# ── Pretrained loading ────────────────────────────────────────────────────────

_EXPECTED_SKIP = (
    'score_head', 'anchors', 'valid_mask',
    'denoising_class_embed', 'reid_head', 's4_branch',
)


def load_pretrained(model, ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    for key in ('model', 'state_dict', 'ema'):
        if key in state:
            state = state[key]
            break
    state    = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model_sd = model.state_dict()
    matched, skip_ok, skip_warn = {}, 0, []

    for k, v in state.items():
        expected = any(s in k for s in _EXPECTED_SKIP)
        if k not in model_sd or model_sd[k].shape != v.shape:
            if not expected:
                skip_warn.append(f'    {k}')
            else:
                skip_ok += 1
        else:
            matched[k] = v

    model.load_state_dict(matched, strict=False)
    print(f'  Pretrained : {os.path.basename(ckpt_path)}')
    print(f'    Loaded {len(matched):,}  skipped {skip_ok} (expected)')
    if skip_warn:
        print(f'    WARNING — unexpected mismatches:')
        for s in skip_warn:
            print(s)


# ── Count helpers ─────────────────────────────────────────────────────────────

def _params(model):
    n = lambda m: sum(p.numel() for p in m.parameters())
    rows = [
        ('backbone (DINOv3STAs)',      n(model.backbone)),
        ('encoder  (HybridEncoder)',   n(model.encoder)),
        ('decoder  (DEIMTransformer)', n(model.decoder)),
        ('reid_head',                  n(model.reid_head)),
    ]
    if model.s4_branch is not None:
        s4 = model.s4_branch
        rows.append(('s4_branch  (total)',               n(s4)))
        rows.append(('  ├ expand (PW×4+BN+SiLU)',        n(s4.expand)))
        rows.append(('  ├ shuffle (PixelShuffle×2)',      n(s4.shuffle)))
        rows.append(('  ├ pw_out  (zero-init residual)',  n(s4.pw_out)))
        if s4.has_detail:
            rows.append(('  ├ detail_proj (c1→dim) [NEW]', n(s4.detail_proj)))
        else:
            rows.append(('  ├ detail_proj              ', 0))
        rows.append(('  └ refine  (DWConv+PW) [NEW]',    n(s4.refine)))
    return rows, n(model)


def _gflops(model, input_h, input_w, device):
    try:
        from fvcore.nn import FlopCountAnalysis
        x  = torch.zeros(1, 3, input_h, input_w, device=device)
        fa = FlopCountAnalysis(model, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    except ImportError:
        return None


def _print_table(args, use_s4, model, gf):
    scales = '4 [S4,S8,S16,S32]' if use_s4 else '3 [S8,S16,S32]'
    tag    = '4-scale + S4Branch (enhanced)' if use_s4 else '3-scale'
    W      = 62
    print(f'\n{"="*W}')
    print(f'  DEIMJDEModel  [{tag}]  {args.input_h}×{args.input_w}')
    print(f'  Classes={args.num_classes}  ReID={args.reid_dim}  conv_inplane={args.conv_inplane}')
    print(f'  Scales={scales}')
    print(f'{"─"*W}')
    rows, total = _params(model)
    for name, cnt in rows:
        marker = ' ←' if '[NEW]' in name else ''
        print(f'  {name:<36} {cnt/1e6:>7.3f} M{marker}')
    print(f'{"─"*W}')
    print(f'  {"Total":<36} {total/1e6:>7.3f} M')
    if gf is not None:
        print(f'  {"GFLOPs (fvcore)*":<36} {gf:>7.2f} G')
        print(f'  * deformable-attn ops counted as 0')
    print(f'{"="*W}')


# ── Entry ─────────────────────────────────────────────────────────────────────

def count(args):
    if args.compare:
        m3  = build_model(args.num_classes, args.reid_dim,
                          args.input_h, args.input_w,
                          use_s4=False, conv_inplane=args.conv_inplane, device=args.device)
        m4  = build_model(args.num_classes, args.reid_dim,
                          args.input_h, args.input_w,
                          use_s4=True,  conv_inplane=args.conv_inplane, device=args.device)
        gf3 = _gflops(m3, args.input_h, args.input_w, args.device)
        gf4 = _gflops(m4, args.input_h, args.input_w, args.device)
        t3  = sum(p.numel() for p in m3.parameters())
        t4  = sum(p.numel() for p in m4.parameters())

        # Per-component delta
        n   = lambda model, attr: sum(p.numel() for p in getattr(model, attr).parameters())
        s4  = m4.s4_branch
        ns4 = lambda attr: sum(p.numel() for p in getattr(s4, attr).parameters())

        W = 62
        print(f'\n{"="*W}')
        print(f'  {"Config":<28} {"Params":>9}  {"GFLOPs":>8}  {"ΔParams":>9}')
        print(f'{"─"*W}')
        gf3_str = f'{gf3:>7.2f}G' if gf3 else '    n/a  '
        gf4_str = f'{gf4:>7.2f}G' if gf4 else '    n/a  '
        dg_str  = f'{gf4-gf3:>+8.2f}G' if (gf3 and gf4) else '     n/a '
        dp      = (t4 - t3) / 1e6
        print(f'  {"3-scale [S8,S16,S32]":<28} {t3/1e6:>8.3f}M  {gf3_str}  {"—":>9}')
        print(f'  {"4-scale [S4,S8,S16,S32]":<28} {t4/1e6:>8.3f}M  {gf4_str}  {dp:>+8.3f}M')
        print(f'{"─"*W}')
        print(f'  S4Branch breakdown (new additions):')
        print(f'    expand + shuffle + pw_out        {(ns4("expand")+ns4("pw_out"))/1e3:>7.1f} K  (original)')
        if s4.has_detail:
            print(f'    detail_proj (c1→dim)  [NEW]  {ns4("detail_proj")/1e3:>7.1f} K')
        print(f'    refine (DWConv+PW)    [NEW]  {ns4("refine")/1e3:>7.1f} K')
        print(f'    num_points[S4]: 3 → 4  (affects decoder sampling_offsets shape)')
        print(f'  GFLOPs delta                       {"":>9}  {dg_str}')
        print(f'{"="*W}')
        return

    model = build_model(args.num_classes, args.reid_dim,
                        args.input_h, args.input_w,
                        use_s4=args.use_s4, conv_inplane=args.conv_inplane,
                        device=args.device)
    if args.load_pretrained:
        load_pretrained(model, args.load_pretrained)

    gf = _gflops(model, args.input_h, args.input_w, args.device)
    _print_table(args, args.use_s4, model, gf)


if __name__ == '__main__':
    count(parse_args())
