"""
count_gflops.py — đếm Params + GFLOPs cho FalconJDEModel (kèm TransformerReIDHead).

Hai vấn đề được xử lý ở đây:

  1) Lỗi anchor mismatch khi profiling
     -------------------------------------------------------------------
     Ở eval mode, decoder dùng anchor cache sinh sẵn theo `eval_spatial_size`.
     Nếu ảnh đầu vào (HxW lúc profiling) KHÁC eval_spatial_size, độ dài memory
     (flatten feature) != độ dài valid_mask -> RuntimeError:
         "The size of tensor a (34020) must match the size of tensor b (42840)"
     (vd anchors cho 480x864 nhưng input 544x960).
     FIX: ép decoder sinh anchor động từ feature thật bằng cách đặt
          `decoder.eval_spatial_size = None`. Khi đó forward chạy đúng với
          BẤT KỲ kích thước nào, không phụ thuộc eval_spatial_size.

  2) Đếm thiếu FLOPs của TransformerReIDHead
     -------------------------------------------------------------------
     thop tự đếm các nn.Linear (value_proj, sampling_offsets, attention_weights,
     fuse) nhưng KHÔNG đếm bước lấy mẫu grid_sample bên trong deformable
     attention (custom op -> mặc định = 0). Ta thêm custom hook cho
     MSDeformableAttention (và RMSNorm) để không có khoảng trống "= 0" âm thầm.
     Lưu ý: phần lớn cost của reid_head đến từ value_proj chiếu trên toàn bộ
     feature map mịn, không phải từ bước sampling.

Cách dùng (full model + S4):
    python tools/count_gflops.py --task mot --arch falcon_jde \
        --use_s4 --num_queries 300 --reid_dim 128 \
        --reid_head_type transformer --reid_num_points 8 \
        --input-wh 960 544 --eval_spatial_size 544 960 \
        --load_model exp/mot/falcon_stage2_mot/model_last.pth --gpus 0

Stage-1 detection-only (--train_single_det: no ReID head, no ReID FLOPs):
    python tools/count_gflops.py --task mot --arch falcon_jde \
        --train_single_det --use_s4 \
        --input-wh 864 480 --gpus -1
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as osp
import sys

import torch

# --- sys.path bootstrap: thêm repo root để `import falconmot` chạy được khi
#     gọi trực tiếp `python tools/count_gflops.py` mà chưa pip install -e . ---
_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))   # repo root (chứa falconmot/)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from falconmot.models.model import create_model, load_model
from falconmot.models.falcon_jde.dfine_decoder import MSDeformableAttention
from falconmot.models.falcon_jde.deim_utils import RMSNorm
from falconmot.opts import opts


# ---------------------------------------------------------------------------
# Custom thop hooks (đếm phần ops mà thop không tự nhận diện)
# ---------------------------------------------------------------------------
def _deform_attn_flops(module, x, y):
    """grid_sample bilinear (~4 taps) + weighted aggregation (~1) trên mỗi
    phần tử được lấy mẫu. Đây là phần deformable mà thop bỏ sót (các Linear con
    đã được thop đếm riêng nên KHÔNG cộng lại ở đây để tránh double-count)."""
    query = x[0]                                   # [bs, Len_q, C]
    bs, len_q = int(query.shape[0]), int(query.shape[1])
    n_head    = int(module.num_heads)
    head_dim  = int(module.head_dim)
    total_pts = int(sum(module.num_points_list))
    macs = bs * len_q * n_head * head_dim * total_pts * 5
    module.total_ops += torch.DoubleTensor([float(macs)])


def _rmsnorm_flops(module, x, y):
    """RMSNorm ~ vài phép elementwise / phần tử (nhỏ, chỉ để tránh cảnh báo)."""
    module.total_ops += torch.DoubleTensor([float(y.numel() * 2)])


CUSTOM_OPS = {
    MSDeformableAttention: _deform_attn_flops,
    RMSNorm:               _rmsnorm_flops,
}


# ---------------------------------------------------------------------------
# Đếm GFLOPs
# ---------------------------------------------------------------------------
def count_gflops(model, h, w, device, breakdown=True):
    """Trả về (gflops, params_M, per_layer_dict). KHÔNG ném lỗi anchor mismatch."""
    try:
        from thop import profile
    except ImportError:
        raise ImportError("Cần cài thop:  pip install thop")

    model = model.to(device).eval()

    # --- FIX anchor mismatch: ép sinh anchor động theo input thật ---
    dec = getattr(model, 'decoder', None)
    if dec is not None and hasattr(dec, 'eval_spatial_size'):
        dec.eval_spatial_size = None

    x = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        macs, params, ret = profile(
            model, inputs=(x,),
            custom_ops=CUSTOM_OPS,
            ret_layer_info=True,
            verbose=False,
        )

    gflops = macs / 1e9 * 2          # MACs -> FLOPs
    params_m = params / 1e6

    per_layer = {}
    if breakdown:
        for name, info in ret.items():
            per_layer[name] = (info[0] / 1e9 * 2, info[1] / 1e6)

    return gflops, params_m, per_layer


# ---------------------------------------------------------------------------
def _summarize_flags(opt, model):
    use_s4   = getattr(opt, 'use_s4', False)
    use_reid = getattr(model, 'use_reid', getattr(opt, 'use_reid', True))
    scales   = '4-scale (S4)' if use_s4 else '3-scale'
    mode     = 'det-only (stage-1)' if getattr(opt, 'train_single_det', False) \
               else ('tracking' if use_reid else 'det-only')
    head     = getattr(opt, 'reid_head_type', 'transformer') if use_reid else 'N/A'
    return scales, mode, head, use_reid


def main():
    opt = opts().init()
    opt.device = (f'cuda:{opt.gpus[0]}'
                  if getattr(opt, 'gpus', [-1])[0] >= 0 and torch.cuda.is_available()
                  else 'cpu')

    if getattr(opt, 'use_s4', False) and not getattr(opt, 'use_sta', True):
        raise ValueError('--use_s4 requires STA backbone (--use_sta, default ON).')

    # opt.img_size = (W, H) theo quy ước repo (track.py: net_w, net_h = opt.img_size)
    w, h = opt.img_size

    print('Creating model...')
    model = create_model(opt.arch, opt)
    if getattr(opt, 'load_model', ''):
        model = load_model(model, opt.load_model)

    scales, mode, head, use_reid = _summarize_flags(opt, model)
    print(f'  mode={mode}  scales={scales}  use_reid={use_reid}  '
          f's4_branch={hasattr(model, "s4_branch")}  '
          f'reid_head={hasattr(model, "reid_head")}')

    gflops, params_m, per_layer = count_gflops(model, h, w, opt.device)

    print('-' * 60)
    print(f'{type(model).__name__}  |  {h}x{w}  |  {scales}  |  {mode}  |  reid_head={head}')
    print('-' * 60)
    order = ['backbone', 'encoder', 'decoder', 's4_branch', 's4_aux_head']
    if use_reid:
        order.insert(3, 'reid_head')
    seen = set()
    for name in order + [k for k in per_layer if k not in order]:
        if name in per_layer and name not in seen:
            seen.add(name)
            g, p = per_layer[name]
            print(f'  {name:<14} {g:8.3f} GFLOPs   {p:7.3f} M')
    print('-' * 60)
    print(f'  {"TOTAL":<14} {gflops:8.3f} GFLOPs   {params_m:7.3f} M')


if __name__ == '__main__':
    main()