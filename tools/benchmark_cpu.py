"""
benchmark_cpu.py — measure ECDetJDE inference latency on CPU.

Runs a configurable number of warmup + timed forward passes and reports
mean latency, std, min/max, and FPS.

Usage:
    # random weights (no checkpoint needed):
    python tools/benchmark_cpu.py

    # with trained weights:
    python tools/benchmark_cpu.py --load_model exp/mot/run/model_best.pth

    # change resolution / backbone:
    python tools/benchmark_cpu.py --input_h 480 --input_w 864 --ecvit_name ecvitt

    # compare multiple resolutions:
    python tools/benchmark_cpu.py --compare

    # control CPU threads (default: all available):
    python tools/benchmark_cpu.py --num_threads 4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import time

import torch

from lib.models.ecdet_jde.ecvit import ViTAdapter
from lib.models.ecdet_jde.hybrid_encoder import HybridEncoder
from lib.models.ecdet_jde.decoder import ECTransformer
from lib.models.ecdet_jde.model import (
    ECDetJDE,
    _ECVIT_CONFIGS,
    _ECDET_NHEAD,
    _ECDET_NUM_QUERIES,
    _ECDET_NUM_LAYERS,
    _ECDET_NUM_DENOISING,
    _ECDET_REG_MAX,
    _ECDET_NUM_POINTS,
    _ECDET_NUM_POINTS_S4,
)


def parse_args():
    p = argparse.ArgumentParser(description='CPU latency benchmark for ECDetJDE')
    p.add_argument('--ecvit_name',  default='ecvitt',
                   choices=list(_ECVIT_CONFIGS.keys()))
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--reid_dim',    type=int, default=128)
    p.add_argument('--input_h',     type=int, default=480)
    p.add_argument('--input_w',     type=int, default=864)
    p.add_argument('--use_s4',      action='store_true')
    p.add_argument('--refine_s8',   action='store_true')
    p.add_argument('--load_model',  default='',
                   help='path to .pth checkpoint (optional; random weights if omitted)')
    p.add_argument('--warmup',      type=int, default=1,
                   help='warmup forward passes (not timed)')
    p.add_argument('--runs',        type=int, default=1,
                   help='timed forward passes')
    p.add_argument('--num_threads', type=int, default=0,
                   help='PyTorch intra-op threads (0 = all available)')
    p.add_argument('--compare',     action='store_true',
                   help='benchmark multiple resolutions side by side')
    return p.parse_args()


def build_model(args, input_h=None, input_w=None,
                use_s4=None, refine_s8=None) -> ECDetJDE:
    input_h   = input_h   if input_h   is not None else args.input_h
    input_w   = input_w   if input_w   is not None else args.input_w
    use_s4    = use_s4    if use_s4    is not None else args.use_s4
    refine_s8 = refine_s8 if refine_s8 is not None else args.refine_s8

    vcfg       = _ECVIT_CONFIGS[args.ecvit_name]
    hidden_dim = vcfg['proj_dim'] if vcfg['proj_dim'] else vcfg['embed_dim']

    backbone = ViTAdapter(
        name               = args.ecvit_name,
        weights_path       = None,
        skip_load_backbone = True,
    )
    encoder = HybridEncoder(
        in_channels     = [hidden_dim] * 3,
        feat_strides    = [8, 16, 32],
        hidden_dim      = hidden_dim,
        use_encoder_idx = [2],
        nhead           = _ECDET_NHEAD,
        dim_feedforward = vcfg['enc_dim_ff'],
        expansion       = vcfg['expansion'],
        depth_mult      = vcfg['depth_mult'],
    )

    if use_s4:
        dec_feat_channels = [hidden_dim] * 4
        dec_feat_strides  = [4, 8, 16, 32]
        dec_num_levels    = 4
        dec_num_points    = _ECDET_NUM_POINTS_S4
    else:
        dec_feat_channels = [hidden_dim] * 3
        dec_feat_strides  = [8, 16, 32]
        dec_num_levels    = 3
        dec_num_points    = _ECDET_NUM_POINTS

    decoder = ECTransformer(
        num_classes       = args.num_classes,
        hidden_dim        = hidden_dim,
        num_queries       = _ECDET_NUM_QUERIES,
        feat_channels     = dec_feat_channels,
        feat_strides      = dec_feat_strides,
        num_levels        = dec_num_levels,
        num_points        = dec_num_points,
        nhead             = _ECDET_NHEAD,
        num_layers        = _ECDET_NUM_LAYERS,
        dim_feedforward   = vcfg['dec_dim_ff'],
        activation        = 'silu',
        num_denoising     = 0,
        eval_spatial_size = (input_h, input_w),
        eval_idx          = -1,
        aux_loss          = False,
        reg_max           = _ECDET_REG_MAX,
        reid_dim          = args.reid_dim,
        mask_downsample_ratio = None,
    )

    s8_dim = hidden_dim if refine_s8 else None
    s4_dim = hidden_dim if use_s4    else None
    model  = ECDetJDE(backbone, encoder, decoder, s8_dim=s8_dim, s4_dim=s4_dim)

    if args.load_model:
        checkpoint  = torch.load(args.load_model, map_location='cpu')
        state_dict_ = checkpoint.get('state_dict', checkpoint)
        state_dict  = {k[7:] if k.startswith('module') else k: v
                       for k, v in state_dict_.items()}
        model_sd    = model.state_dict()
        for k in list(state_dict):
            if k in model_sd and state_dict[k].shape != model_sd[k].shape:
                state_dict[k] = model_sd[k]
        for k in model_sd:
            if k not in state_dict:
                state_dict[k] = model_sd[k]
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded weights from {args.load_model}')

    return model


def run_benchmark(model: ECDetJDE, input_h: int, input_w: int,
                  warmup: int, runs: int) -> dict:
    model.eval()
    x = torch.zeros(1, 3, input_h, input_w)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _  = model(x)
            times.append((time.perf_counter() - t0) * 1000)   # ms

    times = sorted(times)
    mean  = sum(times) / len(times)
    std   = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    # trim 5% outliers only when enough samples exist
    trim  = int(len(times) * 0.05) if len(times) >= 20 else 0
    return {
        'mean_ms': mean,
        'std_ms':  std,
        'min_ms':  times[trim],
        'max_ms':  times[-(trim + 1)],
        'fps':     1000.0 / mean,
    }


def print_result(label: str, r: dict, runs: int, threads: int):
    print(f'\n  Config   : {label}')
    print(f'  Threads  : {threads}')
    print(f'  Runs     : {runs}')
    print(f'  ─────────────────────────────')
    print(f'  Mean     : {r["mean_ms"]:7.1f} ms   ({r["fps"]:.2f} FPS)')
    print(f'  Std      : {r["std_ms"]:7.1f} ms')
    print(f'  Min/Max  : {r["min_ms"]:.1f} / {r["max_ms"]:.1f} ms  (5% trimmed)')


def _compare(args, threads: int):
    resolutions = [
        (320, 576,  False, False),
        (480, 864,  False, False),
        (480, 864,  False, True),   # +S4Branch
        (608, 1088, False, False),
    ]

    W = 74
    print(f'\n{"="*W}')
    print(f'  {"Resolution":<16} {"S4":>4}  {"Mean(ms)":>9}  '
          f'{"Std":>6}  {"Min":>7}  {"Max":>7}  {"FPS":>8}')
    print('─' * W)

    for h, w, rs, s4 in resolutions:
        model = build_model(args, input_h=h, input_w=w, use_s4=s4, refine_s8=rs)
        r     = run_benchmark(model, h, w, args.warmup, args.runs)
        s4tag = '✓' if s4 else ' '
        print(f'  {w}×{h:<10}      {s4tag}   '
              f'{r["mean_ms"]:>8.1f}  {r["std_ms"]:>6.1f}  '
              f'{r["min_ms"]:>6.1f}  {r["max_ms"]:>6.1f}  {r["fps"]:>7.2f}')

    print(f'{"="*W}')
    print(f'  Backbone: {args.ecvit_name.upper()}  |  '
          f'Threads: {threads}  |  '
          f'Warmup: {args.warmup}  Runs: {args.runs}')
    print(f'{"="*W}')


def main():
    args    = parse_args()
    threads = args.num_threads or torch.get_num_threads()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    print(f'\n{"="*60}')
    print(f'  ECDetJDE CPU Benchmark — {args.ecvit_name.upper()}')
    print(f'  PyTorch {torch.__version__}  |  Threads: {threads}')
    print(f'{"="*60}')

    if args.compare:
        _compare(args, threads)
        return

    model = build_model(args)
    label = (f'{args.input_w}×{args.input_h}'
             + (' +S4' if args.use_s4 else '')
             + (' +refine_s8' if args.refine_s8 else ''))
    r = run_benchmark(model, args.input_h, args.input_w, args.warmup, args.runs)
    print_result(label, r, args.runs, threads)
    print()


if __name__ == '__main__':
    main()
