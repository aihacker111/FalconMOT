"""
export_onnx.py — Convert FalconJDEModel to ONNX.

Usage:
    python export_onnx.py \
        --arch falcon_jde \
        --load_model checkpoints/falcon_jde.pth \
        --onnx_path falcon_jde.onnx \
        --img_h 608 --img_w 1088 \
        --opset 17

Outputs:
    pred_logits  (1, 300, num_classes)  raw class logits
    pred_boxes   (1, 300, 4)            cxcywh normalised [0,1] in letterbox space
    pred_reid    (1, 300, reid_dim)     L2-normed ReID embeddings
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

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        # Return fixed-order tuple so ONNX output names are stable
        return out['pred_logits'], out['pred_boxes'], out['pred_reid']


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(opt, output: str, img_h: int, img_w: int, opset: int):
    print(f'[export] Building model: {opt.arch} / {getattr(opt, "dinov3_name", "vit_tiny")}')
    model = create_model(opt.arch, opt)
    model = load_model(model, opt.load_model)
    model = model.deploy()          # fuse Conv-BN in HybridEncoder for speed
    model.eval().cpu()

    wrapper = FalconJDEModelONNX(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    print(f'[export] Tracing with dummy input {tuple(dummy.shape)} ...')
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        opset_version=opset,
        input_names=['image'],
        output_names=['pred_logits', 'pred_boxes', 'pred_reid'],
        dynamic_axes=None,          # fixed H×W — simpler graph, better ORT perf
        do_constant_folding=True,
        verbose=False,
    )

    # Verify
    import onnx
    model_onnx = onnx.load(output)
    onnx.checker.check_model(model_onnx)

    # Print output shapes
    for node in model_onnx.graph.output:
        dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
        print(f'  {node.name:20s} {dims}')

    size_mb = os.path.getsize(output) / 1e6
    print(f'[export] Saved → {output}  ({size_mb:.1f} MB)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Parse export-specific args first; pass the rest to opts
    eparser = argparse.ArgumentParser(add_help=False)
    eparser.add_argument('--onnx_path', default='falcon_jde.onnx')
    eparser.add_argument('--img_h',  type=int, default=608)
    eparser.add_argument('--img_w',  type=int, default=1088)
    eparser.add_argument('--opset',  type=int, default=17)
    eargs, remaining = eparser.parse_known_args()

    opt = opts().init(remaining)        # opts only sees its own args
    export(opt, eargs.onnx_path, eargs.img_h, eargs.img_w, eargs.opset)
