"""
export_onnx.py — Convert FalconJDEModel to ONNX.

Usage (trained checkpoint, 4-scale + S4):
    python tools/export_onnx.py \
        --use_s4 \
        --load_model exp/mot/track_stage2/model_best.pth \
        --onnx_path falcon_jde.onnx \
        --img_h 480 --img_w 864 \
        --opset 17

Usage (dummy / random weights — verify ONNX graph converts):
    python tools/export_onnx.py \
        --dummy --use_s4 \
        --onnx_path falcon_jde_dummy.onnx \
        --img_h 480 --img_w 864

Detection-only model (stage-1, no ReID head):
    python tools/export_onnx.py \
        --dummy --train_single_det --use_s4 \
        --onnx_path falcon_jde_det.onnx

Note: pass the same flags as training (--use_s4, --train_single_det, --reid_head_type, …)
so the exported graph matches your checkpoint architecture.

Outputs (full model):
    pred_logits  (1, 300, num_classes)  raw class logits
    pred_boxes   (1, 300, 4)            cxcywh normalised [0,1] in letterbox space
    pred_reid    (1, 300, reid_dim)     ReID embeddings

Outputs (detection-only, --train_single_det):
    pred_logits, pred_boxes only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _paths  # noqa: F401  (sys.path bootstrap)

import torch
import torch.nn as nn

from falconmot.opts import opts
from falconmot.models.model import create_model, load_model


# ---------------------------------------------------------------------------
# Wrapper: dict → tuple  (ONNX does not support dict outputs)
# ---------------------------------------------------------------------------
class FalconJDEModelONNX(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.use_reid = getattr(model, 'use_reid', True)
        self.use_s4   = getattr(model, 'use_s4', False)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        logits = out['pred_logits']
        boxes  = out['pred_boxes']
        if self.use_reid:
            return logits, boxes, out['pred_reid']
        return logits, boxes


def _output_names(use_reid: bool):
    names = ['pred_logits', 'pred_boxes']
    if use_reid:
        names.append('pred_reid')
    return names


def _core(model):
    return model.module if hasattr(model, 'module') else model


def _summarize_model(model: nn.Module) -> str:
    m = _core(model)
    use_s4   = getattr(m, 'use_s4', False)
    use_reid = getattr(m, 'use_reid', True)
    scales   = '[S4,S8,S16] decoder' if use_s4 else '[S8,S16,S32] decoder'
    parts = [
        f'scales={scales}',
        f'reid={"ON" if use_reid else "OFF"}',
        f's4_branch={hasattr(m, "s4_branch")}',
        f's4_aux_head={hasattr(m, "s4_aux_head")}',
        f'reid_head={hasattr(m, "reid_head")}',
    ]
    return '  '.join(parts)


def _ckpt_has_s4(state_dict: dict) -> bool:
    return any(k.startswith('s4_branch.') or k.startswith('s4_aux_head.')
               for k in state_dict)


def _load_ckpt_state(path: str) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()
            if hasattr(v, 'shape')}


def _validate_arch(model: nn.Module, ckpt_path: str):
    """Warn when checkpoint and built model disagree on S4 / ReID."""
    sd = _load_ckpt_state(ckpt_path)
    m  = _core(model)
    ckpt_s4   = _ckpt_has_s4(sd)
    model_s4  = getattr(m, 'use_s4', False)
    ckpt_reid = any(k.startswith('reid_head.') for k in sd)
    model_reid = getattr(m, 'use_reid', True) and hasattr(m, 'reid_head')

    if ckpt_s4 and not model_s4:
        raise ValueError(
            f'Checkpoint "{ckpt_path}" contains s4_branch weights but model was '
            f'built without --use_s4. Re-run with --use_s4.')
    if model_s4 and not ckpt_s4 and not getattr(model, '_dummy', False):
        print('[export] WARNING: --use_s4 model but checkpoint has no s4_branch keys '
              '(weights will stay random for S4 modules).')
    if ckpt_reid and not model_reid:
        print('[export] WARNING: checkpoint has reid_head weights but model has ReID OFF '
              '(e.g. --train_single_det); those keys will be skipped.')
    if model_reid and not ckpt_reid:
        print('[export] NOTE: model has reid_head but checkpoint has no reid_head keys '
              '(expected for stage-1 det checkpoint → stage-2 export).')


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(opt, output: str, img_h: int, img_w: int, opset: int,
           dummy: bool, deploy: bool):
    use_s4 = getattr(opt, 'use_s4', False)
    if use_s4 and not getattr(opt, 'use_sta', True):
        raise ValueError('--use_s4 requires STA backbone (--use_sta, default ON) '
                         'to produce stride-4 features.')

    print(f'[export] Building model: {opt.arch} / {getattr(opt, "dinov3_name", "vit_tiny")}')
    print(f'[export] opts: use_s4={use_s4}  use_reid={getattr(opt, "use_reid", True)}  '
          f'reid_head_type={getattr(opt, "reid_head_type", "transformer")}')
    model = create_model(opt.arch, opt)
    print(f'[export] arch: {_summarize_model(model)}')

    if dummy:
        model._dummy = True
        print('[export] --dummy: skipping checkpoint load (random weights)')
    else:
        if not getattr(opt, 'load_model', ''):
            raise ValueError('Provide --load_model or use --dummy for untrained export.')
        _validate_arch(model, opt.load_model)
        print(f'[export] Loading weights: {opt.load_model}')
        model = load_model(model, opt.load_model)

    # if deploy:
        # model = model.deploy()      # fuse Conv-BN in HybridEncoder for speed
    model.eval().cpu()

    wrapper = FalconJDEModelONNX(model)
    wrapper.eval()
    use_reid = wrapper.use_reid
    out_names = _output_names(use_reid)

    dummy_in = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    print(f'[export] Dry-run forward  input={tuple(dummy_in.shape)}  '
          f's4={"ON" if wrapper.use_s4 else "OFF"}  '
          f'reid={"ON" if use_reid else "OFF"} ...')
    with torch.no_grad():
        dry_out = wrapper(dummy_in)
        if use_reid:
            logits, boxes, reid = dry_out
            print(f'  pred_logits {tuple(logits.shape)}  '
                  f'pred_boxes {tuple(boxes.shape)}  '
                  f'pred_reid {tuple(reid.shape)}')
        else:
            logits, boxes = dry_out
            print(f'  pred_logits {tuple(logits.shape)}  '
                  f'pred_boxes {tuple(boxes.shape)}')

    print(f'[export] ONNX export → {output}  (opset {opset}) ...')
    torch.onnx.export(
        wrapper,
        dummy_in,
        output,
        opset_version=opset,
        input_names=['image'],
        output_names=out_names,
        dynamic_axes=None,          # fixed H×W — simpler graph, better ORT perf
        do_constant_folding=True,
        verbose=False,
    )

    import onnx
    model_onnx = onnx.load(output)
    onnx.checker.check_model(model_onnx)

    for node in model_onnx.graph.output:
        dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
        print(f'  {node.name:20s} {dims}')

    size_mb = os.path.getsize(output) / 1e6
    print(f'[export] Saved → {output}  ({size_mb:.1f} MB)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    eparser = argparse.ArgumentParser(
        description='Export FalconJDEModel to ONNX.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    eparser.add_argument('--onnx_path', default='falcon_jde.onnx',
                         help='output .onnx file path')
    eparser.add_argument('--img_h', type=int, default=608,
                         help='input height (must match training letterbox H)')
    eparser.add_argument('--img_w', type=int, default=1088,
                         help='input width (must match training letterbox W)')
    eparser.add_argument('--opset', type=int, default=17,
                         help='ONNX opset version')
    eparser.add_argument('--dummy', action='store_true',
                         help='export with random weights (no --load_model); '
                              'use to verify ONNX conversion only')
    eparser.add_argument('--no_deploy', action='store_true',
                         help='skip model.deploy() Conv-BN fusion before export')
    eargs, remaining = eparser.parse_known_args()

    opt = opts().init(remaining)
    export(opt, eargs.onnx_path, eargs.img_h, eargs.img_w, eargs.opset,
           eargs.dummy, deploy=not eargs.no_deploy)
