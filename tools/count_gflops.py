"""
Đếm params và GFLOPs của FalconJDEModel — dùng CHÍNH opts.py làm nguồn cấu hình.

Mọi default (backbone/encoder/decoder, num_classes, eval_spatial_size, ...) lấy
trực tiếp từ falconmot/opts.py, nên cấu hình khớp tuyệt đối với lúc train.

Cách dùng (đặt file ở gốc repo, cạnh thư mục falconmot/, hoặc trong tools/):
    python count_flops.py                       # mặc định opts.py (3-scale)
    python count_flops.py --use_s4              # 4-scale (S4+S8+S16+S32)
    python count_flops.py --num_denoising 0     # đếm "thuần inference" (bỏ denoising embed)
    # mọi cờ khác của opts.py đều truyền thẳng được, ví dụ --conv_inplane 32
"""
import os, sys, argparse
from contextlib import redirect_stdout

_ROOT = os.path.dirname(os.path.abspath(__file__))
for cand in (_ROOT, os.path.dirname(_ROOT)):
    if os.path.isdir(os.path.join(cand, "falconmot")):
        sys.path.insert(0, cand)
        break

import torch
from falconmot.opts import opts
from falconmot.models.model import create_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = {}
    for name, p in model.named_parameters():
        g = name.split(".")[0]
        groups[g] = groups.get(g, 0) + p.numel()
    return total, train, groups


def count_gflops(model, h, w, device):
    x = torch.zeros(1, 3, h, w, device=device)
    # try:
    #     from fvcore.nn import FlopCountAnalysis
    #     fa = FlopCountAnalysis(model, x)
    #     fa.unsupported_ops_warnings(False)
    #     fa.uncalled_modules_warnings(False)
    #     return fa.total() / 1e9, "fvcore"
    # except ImportError:
    #     pass
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return 2 * macs / 1e9, "thop (2*MACs)"
    except ImportError:
        return None, None


def main():
    # Chỉ giữ riêng --device; mọi cờ còn lại chuyển nguyên cho opts.py
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--device", default="cpu")
    known, passthrough = ap.parse_known_args()

    # Truyền 1 list (kể cả rỗng) để opts.py KHÔNG đọc sys.argv và dùng default thật
    with open(os.devnull, "w") as null, redirect_stdout(null):  # nuốt log train của opts
        opt = opts().init(passthrough)

    model = create_model(opt.arch, opt).to(known.device).eval()
    h, w  = opt.input_h, opt.input_w
    total, train, groups = count_params(model)

    print(f"\nFalconJDEModel  |  {h}x{w}  |  "
          f"{'4-scale (S4)' if getattr(opt, 'use_s4', False) else '3-scale'}  |  "
          f"num_classes={opt.num_classes}  num_denoising={opt.num_denoising}")
    print("-" * 52)
    for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:<28} {n/1e6:>8.3f} M")
    print("-" * 52)
    print(f"  {'Total params':<28} {total/1e6:>8.3f} M")
    print(f"  {'Trainable params':<28} {train/1e6:>8.3f} M")

    with torch.no_grad():
        gf, tool = count_gflops(model, h, w, known.device)
    if gf is not None:
        print(f"  {'GFLOPs (' + tool + ')':<28} {gf:>8.2f} G")
        print("  * op deformable-attention có thể bị đếm = 0")
    else:
        print("  GFLOPs: cần cài 'fvcore' hoặc 'thop' (pip install fvcore)")
    print()


if __name__ == "__main__":
    main()