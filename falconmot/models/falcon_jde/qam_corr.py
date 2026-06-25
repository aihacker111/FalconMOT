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
    """Sample đặc trưng vector từ feature map m [C,H,W] tại các tâm [N, 2] (đơn vị [0,1]).

    Trả về: [N, C]
    """
    C, H, W = m.shape
    N = centers.shape[0]
    if N == 0:
        return torch.empty((0, C), device=m.device)
    
    # grid_sample nhận tọa độ dạng [-1, 1], định dạng (x, y)
    grid = centers.view(1, N, 1, 2) * 2.0 - 1.0
    out = F.grid_sample(m.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return out.squeeze(0).squeeze(-1).permute(1, 0) # [N, C]


def _sample_scalar(m: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Sample các giá trị scalar từ batch các bản đồ tương đồng m [N,H,W] tại các tâm [N, 2].

    Trả về: [N]
    """
    N, H, W = m.shape
    if N == 0:
        return torch.empty((0,), device=m.device)
    
    grid = centers.view(N, 1, 1, 2) * 2.0 - 1.0
    out = F.grid_sample(m.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return out.view(N)


def qam_cross_frame_loss(outputs_t: dict, outputs_t1: dict, targets_t: list, targets_t1: list,
                         tau: float = 0.07,
                         tau_spatial: float = 1.0,  # Thêm cấu hình nhiệt độ không gian để tránh sụp đổ softmax
                         w_corr: float = 1.0,
                         w_distr: float = 1.0,
                         w_ent: float = 1.0) -> dict[str, torch.Tensor]:
    """Tính toán bộ 3 loss QAM (A+B+C) dựa trên mối tương quan giữa frame t và frame t+1.

    Args:
        outputs_t/t1: Output của model chứa 'pred_reid_map' shape [B, C, H, W]
        targets_t/t1: List chứa dict thông tin nhãn (boxes [0,1], labels, track_ids) của từng ảnh trong batch
    """
    device = outputs_t['pred_reid_map'].device
    map_t = outputs_t['pred_reid_map']
    map_t1 = outputs_t1['pred_reid_map']
    
    B, C, H, W = map_t.shape
    
    # Tạo lưới tọa độ pixel quy đổi về đoạn [0, 1] phục vụ cho Soft-argmax
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(H * W, 2) # [HW, 2]
    
    L_corr = torch.tensor(0.0, device=device)
    L_distr = torch.tensor(0.0, device=device)
    L_ent = torch.tensor(0.0, device=device)
    
    total_matches = 0
    
    for b in range(B):
        m_t = _norm_map(map_t[b])   # [C, H, W]
        m_t1 = _norm_map(map_t1[b]) # [C, H, W]
        
        tgt_t = targets_t[b]
        tgt_t1 = targets_t1[b]
        
        # Đảm bảo có track_ids/ids để thực hiện bắt cặp thực thể qua thời gian
        ids_key = 'track_ids' if 'track_ids' in tgt_t else 'ids'
        if ids_key not in tgt_t or ids_key not in tgt_t1:
            continue
            
        if len(tgt_t['boxes']) == 0 or len(tgt_t1['boxes']) == 0:
            continue
            
        ids_t = tgt_t[ids_key]
        ids_t1 = tgt_t1[ids_key]
        boxes_t = tgt_t['boxes']   # [num_t, 4] dạng (cx, cy, w, h)
        boxes_t1 = tgt_t1['boxes'] # [num_t1, 4]
        lbl_t = tgt_t['labels']     # [num_t]
        lbl_t1 = tgt_t1['labels']   # [num_t1]
        
        # ── BẮT CẶP CÁC OBJECT XUẤT HIỆN Ở CẢ 2 FRAME (MATCHING) ──
        matches = []
        for i, id_t in enumerate(ids_t):
            if id_t == -1: # Bỏ qua background hoặc vật thể không có ID tracking hợp lệ
                continue
            for j, id_t1 in enumerate(ids_t1):
                if id_t == id_t1:
                    matches.append((i, j))
                    break
                    
        if len(matches) == 0:
            continue
            
        # it: index của các object khớp ở frame t, i1: index tương ứng ở frame t+1
        it = torch.tensor([m[0] for m in matches], dtype=torch.long, device=device)
        i1 = torch.tensor([m[1] for m in matches], dtype=torch.long, device=device)
        
        total_matches += it.numel()
        
        # Trích xuất tâm (cx, cy) của các object
        c0_all = boxes_t[:, :2]  # [num_t, 2]
        c1_all = boxes_t1[:, :2] # [num_t1, 2]
        
        c0 = c0_all[it] # Các tâm tại frame t làm Template
        c1 = c1_all[i1] # Các tâm GT mục tiêu tại frame t+1
        
        lbl = lbl_t[it]
        lbl1_all = lbl_t1
        
        # Lấy đặc trưng mẫu (Template vector) từ frame t
        tmpl = _sample_vec(m_t, c0) # [n, C]
        
        # Trải phẳng bản đồ frame t+1 để thực hiện phép nhân ma trận (Correlation)
        R = m_t1.reshape(C, H * W) # [C, HW]
        
        tmpl = F.normalize(tmpl, dim=1, eps=1e-6)
        sim = tmpl @ R             # [n, HW] ma trận Cosine Similarity
        
        # ── A: Soft-argmax ĐÚNG VỊ TRÍ TÂM TẠI FRAME T+1 (Dùng tau_spatial lớn) ──
        A_spatial = torch.softmax(sim / tau_spatial, dim=1)  # [n, HW]
        pred = A_spatial @ grid_xy                           # [n, 2] dự đoán tọa độ (x, y)
        L_corr = L_corr + F.smooth_l1_loss(pred, c1, reduction='sum')

        # ── C: ENTROPY SẮC NÉT + COSINE CAO TẠI ĐÚNG VỊ TRÍ ──
        ent = -(A_spatial * A_spatial.clamp_min(1e-12).log()).sum(dim=1) \
              / float(torch.log(torch.tensor(float(H * W), device=device)))
        sim_hw = sim.reshape(-1, H, W)                       # [n, H, W]
        peak_at_tgt = _sample_scalar(sim_hw, c1)             # [n] lấy độ tương đồng ngay tại tâm GT
        L_ent = L_ent + ent.sum() + (1.0 - peak_at_tgt).sum()

        # ── B: INFONCE THEO VỊ TRÍ (Giữ nguyên tau=0.07 nhỏ để phân biệt id cực đoan) ──
        for k in range(it.numel()):
            same = (lbl1_all == lbl[k]).nonzero(as_tuple=True)[0]   # Các đối tượng cùng class ở frame t+1
            if same.numel() < 2:
                continue                                     # Không có distractor cùng loại thì bỏ qua
                
            centers_k = c1_all[same]                         # [m, 2]
            sim_k = sim[k].reshape(1, H, W).expand(centers_k.shape[0], H, W)
            logits = _sample_scalar(sim_k, centers_k) / tau  # [m]
            
            # Tìm vị trí chính xác của object k hiện tại trong danh sách vật thể cùng class ở frame t+1
            pos = (same == i1[k]).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
                
            L_distr = L_distr + F.cross_entropy(logits.unsqueeze(0), pos)
            
    # Chuẩn hóa tổng loss theo số lượng mẫu bắt cặp thành công
    if total_matches > 0:
        L_corr = L_corr / total_matches
        L_distr = L_distr / total_matches
        L_ent = L_ent / total_matches
        
    return {
        'loss_qam_corr': L_corr * w_corr,
        'loss_qam_distr': L_distr * w_distr,
        'loss_qam_ent': L_ent * w_ent
    }