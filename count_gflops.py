#!/usr/bin/env python3
"""
count_gflops.py
===============
GFLOPs / params counter for the Fovea-MOT (FalconMOT / DEIMv2) model.

This tool reuses the project's REAL argument parser (`falconmot.cfg.args.opts`),
so every training flag is available and the model is built EXACTLY as in training
-- including DEIMv2 pretrained loading. Pass `--deim_pretrained path/to/deimv2.pth`
(handled inside `build_falcon_jde`) or `--load_model path/to/ckpt.pth` to count a
finetuned checkpoint with a realistic (trained) entropy mask.

Why a custom counter?
---------------------
Two parts of this model defeat off-the-shelf counters (thop / ptflops):
  * SparseFeatFusion runs a *data-dependent* window at inference, so the real
    FLOPs depend on the entropy mask. A hook-based counter that reads the actual
    tensor shapes flowing through each layer captures this correctly.
  * The deformable cross-attention uses F.grid_sample, which most counters skip.

Reports:
  * total GFLOPs + params,
  * per-top-level-module breakdown (backbone / encoder / s4_branch / decoder /
    reid_head ...),
  * for SAFA models, an *analytic* S4-branch line: dense vs keep_ratio-charged
    (the reproducible number to put in a paper, independent of whether the
    entropy scorer is trained).
  * with --gflops_compare, also builds the no-S4 and dense-S4 baselines.

Usage
-----
    # count the SAFA model with the DEIMv2 pretrained loaded, at eval resolution
    python tools/count_gflops.py --use_s4 --use_safa \
        --reid_cls_ids 0,1,2,3,4,5,6 --eval_spatial_size 480 864 \
        --deim_pretrained weights/deimv2_dinov3_s.pth

    # count a finetuned MOT checkpoint (realistic sparse mask) + averaging
    python tools/count_gflops.py --use_s4 --use_safa \
        --load_model exp/mot/run/model_last.pth --gflops_avg 16

    # full comparison table (no-S4 vs dense-S4 vs SAFA-sparse)
    python tools/count_gflops.py --use_s4 --use_safa --gflops_compare

Convention: 1 MAC = 2 FLOPs. Numbers below are FLOPs.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Hook-based FLOP counting (pure torch; no external deps)
# ===========================================================================
def _conv_flops(m, inp, out):
    out_h, out_w = out.shape[-2:]
    k = 1
    for s in m.kernel_size:
        k *= s
    cin_per_group = m.in_channels // m.groups
    macs = out.shape[0] * m.out_channels * out_h * out_w * cin_per_group * k
    if m.bias is not None:
        macs += out.shape[0] * m.out_channels * out_h * out_w
    return 2 * macs


def _linear_flops(m, inp, out):
    tokens = out.numel() / m.out_features
    macs = tokens * m.in_features * m.out_features
    if m.bias is not None:
        macs += tokens * m.out_features
    return 2 * int(macs)


def _mha_flops(m, inp, out):
    q = inp[0]
    Lq = q.shape[0] if m.batch_first is False else q.shape[1]
    B = q.shape[1] if m.batch_first is False else q.shape[0]
    E = m.embed_dim
    proj = 4 * B * Lq * E * E
    attn = 2 * B * Lq * Lq * E
    return 2 * (proj + attn)


_HOOK_TYPES = {
    nn.Conv2d: _conv_flops,
    nn.ConvTranspose2d: _conv_flops,
    nn.Linear: _linear_flops,
    nn.MultiheadAttention: _mha_flops,
}


class FlopCounter:
    """Accumulate FLOPs per leaf module via forward hooks."""

    def __init__(self, model, count_grid_sample=False):
        self.model = model
        self.count_grid_sample = count_grid_sample
        self.per_module = {}
        self._handles = []
        self._name_of = {m: n for n, m in model.named_modules()}
        self._gs_flops = 0
        self._orig_grid_sample = None

    def _hook(self, m, inp, out):
        fn = _HOOK_TYPES.get(type(m))
        if fn is None:
            return
        try:
            f = fn(m, inp, out)
        except Exception:
            f = 0
        name = self._name_of.get(m, repr(m))
        self.per_module[name] = self.per_module.get(name, 0) + f

    def __enter__(self):
        for m in self.model.modules():
            if type(m) in _HOOK_TYPES:
                self._handles.append(m.register_forward_hook(self._hook))
        if self.count_grid_sample:
            self._orig_grid_sample = F.grid_sample
            counter = self

            def _gs(input, grid, *a, **kw):
                o = counter._orig_grid_sample(input, grid, *a, **kw)
                counter._gs_flops += 2 * o.numel() * 4
                return o
            F.grid_sample = _gs
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._orig_grid_sample is not None:
            F.grid_sample = self._orig_grid_sample
            self._orig_grid_sample = None

    def total(self):
        return sum(self.per_module.values()) + self._gs_flops

    def by_top_module(self):
        agg = {}
        for name, f in self.per_module.items():
            top = name.split('.')[0] if '.' in name else name
            agg[top] = agg.get(top, 0) + f
        if self._gs_flops:
            agg['grid_sample'] = agg.get('grid_sample', 0) + self._gs_flops
        return agg


@torch.no_grad()
def measure_flops(model, x, count_grid_sample=False, avg=1):
    model.eval()
    totals, breakdown = [], {}
    for i in range(avg):
        xi = x if avg == 1 else torch.randn_like(x)
        with FlopCounter(model, count_grid_sample) as fc:
            model(xi)
        totals.append(fc.total())
        for k, v in fc.by_top_module().items():
            breakdown.setdefault(k, []).append(v)
    mean_total = sum(totals) / len(totals)
    mean_break = {k: sum(v) / len(v) for k, v in breakdown.items()}
    return mean_total, mean_break


# ===========================================================================
# Analytic S4 line: dense vs keep_ratio-charged (reproducible paper number)
# ===========================================================================
@torch.no_grad()
def analytic_s4(model, x, keep_ratio):
    from falconmot.nn.falcon_jde.ops.safa import SparseFeatFusion
    s4, s4_name = None, None
    for n, m in model.named_modules():
        if isinstance(m, SparseFeatFusion):
            s4, s4_name = m, n
            break
    if s4 is None:
        return None

    was_training = s4.training
    s4.train()                              # SparseFeatFusion runs dense in train()
    name_of = {mm: nn_ for nn_, mm in s4.named_modules()}
    cheap, heavy = 0, 0
    handles = []

    def mk_hook():
        def hook(m, inp, out):
            fn = _HOOK_TYPES.get(type(m))
            if fn is None:
                return
            f = fn(m, inp, out)
            local = name_of.get(m, '')
            nonlocal cheap, heavy
            if local.startswith('blocks') or local.startswith('out'):
                heavy += f
            else:
                cheap += f
        return hook

    for m in s4.modules():
        if type(m) in _HOOK_TYPES:
            handles.append(m.register_forward_hook(mk_hook()))

    model.eval()
    model(x)
    for h in handles:
        h.remove()
    if not was_training:
        s4.eval()

    dense = cheap + heavy
    sparse = cheap + keep_ratio * heavy
    return {
        'module': s4_name, 'keep_ratio': keep_ratio,
        'cheap': cheap, 'heavy_dense': heavy,
        's4_dense': dense, 's4_sparse_analytic': sparse,
        'saved': dense - sparse,
    }


# ===========================================================================
# Model building (uses the project's real build path -> loads DEIMv2 pretrained)
# ===========================================================================
def build(opt, load_weights=True):
    """Build the model via the project factory. `build_falcon_jde` loads
    opt.deim_pretrained internally; we additionally load opt.load_model if set."""
    from falconmot.nn.falcon_jde.model import build_falcon_jde, load_pretrained
    o = opt if load_weights else _strip_weights(copy.copy(opt))
    model = build_falcon_jde(o).eval()
    lm = getattr(o, 'load_model', '')
    if load_weights and lm:
        load_pretrained(model, lm, verbose=True)
    return model


def _strip_weights(o):
    o.deim_pretrained = ''
    o.load_model = ''
    return o


def variant(opt, **flags):
    """Clone opt with flags overridden and weight paths stripped (structural)."""
    o = copy.copy(opt)
    for k, v in flags.items():
        setattr(o, k, v)
    return _strip_weights(o)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _fmt(flops):
    return f"{flops / 1e9:8.3f} GFLOPs"


def report(model, x, title, keep_ratio=None, count_grid_sample=False, avg=1):
    total, brk = measure_flops(model, x, count_grid_sample, avg)
    print(f"\n=== {title} ===")
    print(f"  params : {n_params(model) / 1e6:7.2f} M")
    print(f"  total  : {_fmt(total)}  (measured, eval, avg={avg})")
    for k in sorted(brk, key=lambda z: -brk[z]):
        print(f"    {k:14s} {_fmt(brk[k])}  ({100 * brk[k] / max(total, 1):4.1f}%)")
    if keep_ratio is not None:
        a = analytic_s4(model, x, keep_ratio)
        if a:
            print(f"  analytic S4 branch (keep_ratio={a['keep_ratio']}):")
            print(f"    dense  : {_fmt(a['s4_dense'])}")
            print(f"    sparse : {_fmt(a['s4_sparse_analytic'])}  "
                  f"(saves {_fmt(a['saved'])}, "
                  f"{100 * a['saved'] / max(a['s4_dense'], 1):.1f}% of S4 branch)")
    return total


# ===========================================================================
# Main
# ===========================================================================
def _add_gflops_args(parser):
    parser.add_argument('--gflops_avg', type=int, default=1,
                        help='average measured FLOPs over N forwards (sparse path)')
    parser.add_argument('--gflops_grid_sample', action='store_true',
                        help='also count deformable grid_sample (minor; cancels in comparisons)')
    parser.add_argument('--gflops_compare', action='store_true',
                        help='build no-S4 / dense-S4 / SAFA-sparse and tabulate')


def main():
    from falconmot.cfg.args import opts
    op = opts()
    _add_gflops_args(op.parser)
    opt = op.init()                 # reads sys.argv via the REAL parser (loads all flags)

    H, W = opt.eval_spatial_size
    x = torch.randn(1, 3, H, W)
    avg = getattr(opt, 'gflops_avg', 1)
    gs = getattr(opt, 'gflops_grid_sample', False)
    use_safa = getattr(opt, 'use_safa', False)
    keep = getattr(opt, 'safa_keep_ratio', 0.25)

    if getattr(opt, 'gflops_compare', False):
        m0 = build(variant(opt, use_s4=False, use_safa=False), load_weights=False)
        t0 = report(m0, x, "baseline (no S4)", count_grid_sample=gs)
        m1 = build(variant(opt, use_s4=True, use_safa=False), load_weights=False)
        t1 = report(m1, x, "dense S4 (no SAFA)", count_grid_sample=gs)
        # SAFA variant: load the real weights so the sparse mask is realistic
        m2 = build(variant(opt, use_s4=True, use_safa=True,
                           deim_pretrained=getattr(opt, 'deim_pretrained', ''),
                           load_model=getattr(opt, 'load_model', '')),
                   load_weights=True)
        t2 = report(m2, x, "SAFA sparse S4", keep_ratio=keep, count_grid_sample=gs, avg=avg)
        print("\n=== summary ===")
        print(f"  no-S4         : {_fmt(t0)}")
        print(f"  dense-S4      : {_fmt(t1)}   (+{_fmt(t1 - t0)} vs no-S4)")
        print(f"  SAFA-sparse   : {_fmt(t2)}   "
              f"({100 * (t1 - t2) / max(t1, 1):+.1f}% vs dense-S4, measured)")
        if not (getattr(opt, 'deim_pretrained', '') or getattr(opt, 'load_model', '')):
            print("  note: no weights loaded -> sparse mask is random; measured SAFA is an")
            print("        upper bound. Use the analytic S4 line / load a checkpoint for the paper figure.")
    else:
        model = build(opt, load_weights=True)
        kr = keep if use_safa else None
        report(model, x, "Fovea-MOT", keep_ratio=kr, count_grid_sample=gs, avg=avg)


if __name__ == '__main__':
    main()