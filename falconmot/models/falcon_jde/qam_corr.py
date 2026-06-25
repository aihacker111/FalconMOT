"""Cross-frame QAM training losses (A + B + C).

Tối ưu TRỰC TIẾP cơ chế QAM dùng ở inference: lấy template (mẫu dense map tại
tâm object ở frame t), correlate lên dense map frame t+1, và bắt response:

  A (correlation)  : soft-argmax của response rơi ĐÚNG tâm object ở t+1
                     + cosine tại đúng vị trí cao (template transfer qua thời gian).
  B (distractor)   : response CAO ở vị trí object đó, THẤP ở các object CÙNG LỚP
                     khác (InfoNCE theo vị trí) -> map đơn-đỉnh, hết look-alike.
  C (entropy)      : response của object hiện diện phải SẮC (entropy thấp)
                     -> calibrate confidence cho am_beta/am_peak_thr.

Tất cả ở KHÔNG GIAN normalized [0,1] (grid_sample align_corners=True), nên không
cần image-transform. Map = `pred_reid_map` (= emb_map của head), CÙNG field mà QAM
inference dùng -> những gì học ở đây chuyển thẳng sang tracking.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def _norm_map(m: torch.Tensor) -> torch.Tensor:
    """L2-normalise [C,H,W] theo kênh tại mỗi pixel (giống normalize_dense)."""
    return F.normalize(m, dim=0, eps=1e-6)


def _sample_vec(m: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Sample [C,H,W] tại centers [n,2] (norm [0,1], xy) -> [n,C]."""
    C = m.shape[0]
    if centers.numel() == 0:
        return m.new_zeros((0, C))
    grid = (centers * 2.0 - 1.0).view(1, -1, 1, 2)
    s = F.grid_sample(m.unsqueeze(0), grid, mode='bilinear', align_corners=True)
    return s.view(C, -1).t().contiguous()                       # [n,C]


def _sample_scalar(sim_hw: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Mỗi object sample TRÊN sim-map riêng của nó.

    sim_hw  : [n,H,W]  (n response maps).
    centers : [n,2]    (norm [0,1], xy) — vị trí cần lấy của từng object.
    return  : [n]      giá trị sim tại center tương ứng.
    """
    n = sim_hw.shape[0]
    if n == 0:
        return sim_hw.new_zeros((0,))
    grid = (centers * 2.0 - 1.0).view(n, 1, 1, 2)               # [n,1,1,2]
    s = F.grid_sample(sim_hw.unsqueeze(1), grid,
                      mode='bilinear', align_corners=True)      # [n,1,1,1]
    return s.view(n)


def _common_pairs(tid_t: torch.Tensor, tid_1: torch.Tensor):
    """Trả về (idx_t, idx_1): cặp chỉ số của track_id xuất hiện ở CẢ hai frame."""
    a = tid_t.detach().cpu().tolist()
    b = {int(v): j for j, v in enumerate(tid_1.detach().cpu().tolist()) if v >= 0}
    it, i1 = [], []
    for i, v in enumerate(a):
        v = int(v)
        if v >= 0 and v in b:
            it.append(i)
            i1.append(b[v])
    dev = tid_t.device
    return (torch.tensor(it, dtype=torch.long, device=dev),
            torch.tensor(i1, dtype=torch.long, device=dev))


@torch.no_grad()
def _grid_norm(H: int, W: int, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    gx = (xs.reshape(-1) / max(W - 1, 1))                       # [HW] in [0,1]
    gy = (ys.reshape(-1) / max(H - 1, 1))
    return torch.stack([gx, gy], dim=1)                         # [HW,2]


def qam_cross_frame_loss(out_t, out_t1, targets_t, targets_t1,
                         tau: float = 0.07,
                         w_corr: float = 1.0,
                         w_distr: float = 0.5,
                         w_ent: float = 0.1):
    """A+B+C giữa frame t (template) và frame t+1 (search field).

    out_t, out_t1   : dict output của model (cần 'pred_reid_map' [B,C,H,W]).
    targets_t/t1    : list per-batch dict {'labels','boxes'(cxcywh norm),'track_ids'}.
    Trả về dict {'loss_qam_corr','loss_qam_distr','loss_qam_ent'} (đã nhân trọng số).
    """
    map_t  = out_t.get('pred_reid_map', None)
    map_t1 = out_t1.get('pred_reid_map', None)
    ref = map_t1 if map_t1 is not None else map_t
    if map_t is None or map_t1 is None:
        z = (ref.sum() * 0.0) if ref is not None else torch.zeros(())
        return {'loss_qam_corr': z, 'loss_qam_distr': z, 'loss_qam_ent': z}

    B, C, H, W = map_t1.shape
    dev, dt = map_t1.device, map_t1.dtype
    grid_xy = _grid_norm(H, W, dev, dt)                         # [HW,2] in [0,1]

    L_corr = map_t1.sum() * 0.0
    L_distr = L_corr.clone()
    L_ent = L_corr.clone()
    n_obj = 0

    for b in range(B):
        tt, t1 = targets_t[b], targets_t1[b]
        tid_t = tt['track_ids'].to(dev)
        tid_1 = t1['track_ids'].to(dev)
        if tid_t.numel() == 0 or tid_1.numel() == 0:
            continue
        it, i1 = _common_pairs(tid_t, tid_1)
        if it.numel() == 0:
            continue

        ct  = tt['boxes'].to(dev)[it][:, :2]                   # [n,2] tâm t (norm)
        c1  = t1['boxes'].to(dev)[i1][:, :2]                   # [n,2] tâm t+1 (norm)
        lbl = tt['labels'].to(dev)[it]                        # [n] lớp
        lbl1_all = t1['labels'].to(dev)                       # [n1]
        c1_all   = t1['boxes'].to(dev)[:, :2]                 # [n1,2]

        m_t  = _norm_map(map_t[b])                            # [C,H,W]
        m_t1 = _norm_map(map_t1[b])
        R = m_t1.reshape(C, H * W)                            # [C,HW]

        tmpl = _sample_vec(m_t, ct)                           # [n,C]
        tmpl = F.normalize(tmpl, dim=1, eps=1e-6)
        sim = tmpl @ R                                        # [n,HW] cosine
        A = torch.softmax(sim / tau, dim=1)                  # [n,HW]

        # ── A: soft-argmax -> tâm t+1 (normalized) ──
        pred = A @ grid_xy                                   # [n,2] in [0,1]
        L_corr = L_corr + F.smooth_l1_loss(pred, c1, reduction='sum')

        # ── C: entropy thấp + cosine cao tại đúng vị trí (template transfer) ──
        ent = -(A * A.clamp_min(1e-12).log()).sum(dim=1) \
              / float(torch.log(torch.tensor(float(H * W))))
        sim_hw = sim.reshape(-1, H, W)                       # [n,H,W]
        peak_at_tgt = _sample_scalar(sim_hw, c1)             # [n] cosine tại tâm t+1
        L_ent = L_ent + ent.sum() + (1.0 - peak_at_tgt).sum()

        # ── B: InfoNCE theo vị trí — cao tại object o, thấp tại object cùng lớp khác ──
        for k in range(it.numel()):
            same = (lbl1_all == lbl[k]).nonzero(as_tuple=True)[0]   # idx object cùng lớp ở t+1
            if same.numel() < 2:
                continue                                     # không có distractor -> bỏ
            centers_k = c1_all[same]                          # [m,2]
            sim_k = sim[k].reshape(1, H, W).expand(centers_k.shape[0], H, W)
            logits = _sample_scalar(sim_k, centers_k) / tau  # [m] cosine tại từng object cùng lớp
            pos = (same == i1[k]).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            L_distr = L_distr + F.cross_entropy(
                logits.unsqueeze(0), pos[:1])

        n_obj += it.numel()

    if n_obj > 0:
        inv = 1.0 / float(n_obj)
        L_corr = L_corr * inv
        L_distr = L_distr * inv
        L_ent = L_ent * inv

    return {
        'loss_qam_corr':  w_corr  * L_corr,
        'loss_qam_distr': w_distr * L_distr,
        'loss_qam_ent':   w_ent   * L_ent,
    }