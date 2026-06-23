import _paths  # noqa: F401
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch

from falconmot.models.falcon_jde import FalconJDEPostProcessor
from falconmot.models.model import create_model, load_model
from falconmot.opts import opts

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Định nghĩa danh sách nhãn chuẩn của VisDrone
# VISDRONE_CLASSES = [
#     "pedestrian",
#     "people",
#     "bicycle",
#     "car",
#     "van",
#     "truck",
#     "tricycle",
#     "awning-tricycle",
#     "bus",
#     "motor",
# ]
VISDRONE_CLASSES = [
    "pedestrian",
    "bicycle",
    "car",
    "van",
    "truck",
    "bus",
    "motor",
]

# Tạo màu ngẫu nhiên cố định cho từng lớp để vẽ
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(VISDRONE_CLASSES), 3), dtype=np.uint8)


def resize_plain(img, net_h, net_w):
  # Plain resize về (net_w, net_h) — không letterbox/pad
  return cv2.resize(img, (net_w, net_h), interpolation=cv2.INTER_AREA)


def to_tensor(img_bgr):
  """Tiền xử lý: Chuyển BGR sang RGB, chia 255.0 để chuẩn hóa về [0, 1].

  KHÔNG áp dụng Mean/Std normalization theo cấu hình mô hình hiện tại.
  """
  rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  # Đổi trục từ (H, W, C) sang (C, H, W) và thêm batch dimension (1, C, H, W)
  return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _nms(boxes, scores, iou_thr=0.6):
  if len(boxes) == 0:
    return np.empty((0,), int)
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  areas = (x2 - x1) * (y2 - y1)
  order = scores.argsort()[::-1]
  keep = []
  while order.size:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])
    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h
    iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
    order = order[1:][iou <= iou_thr]
  return np.array(keep, int)


@torch.no_grad()
def infer_image(model, post, img_bgr, net_h, net_w, device):
  H0, W0 = img_bgr.shape[:2]
  lb = resize_plain(img_bgr, net_h, net_w)
  x = to_tensor(lb).to(device)
  out = model(x)
  res = post(out, torch.tensor([[H0, W0]], device=device))[0]
  return (
      res["boxes"].cpu().numpy(),
      res["scores"].cpu().numpy(),
      res["labels"].cpu().numpy(),
  )


@torch.no_grad()
def infer_tiled(
    model, post, img_bgr, net_h, net_w, device, rows=2, cols=2, overlap=0.2
):
  H0, W0 = img_bgr.shape[:2]
  th, tw = int(H0 / rows), int(W0 / cols)
  oy, ox = int(th * overlap), int(tw * overlap)
  all_b, all_s, all_l = [], [], []

  crops = [(0, 0, W0, H0)]
  for r in range(rows):
    for c in range(cols):
      x1 = max(0, c * tw - ox)
      y1 = max(0, r * th - oy)
      x2 = min(W0, (c + 1) * tw + ox)
      y2 = min(H0, (r + 1) * th + oy)
      crops.append((x1, y1, x2, y2))

  for x1, y1, x2, y2 in crops:
    b, s, l = infer_image(
        model, post, img_bgr[y1:y2, x1:x2], net_h, net_w, device
    )
    if len(b):
      b = b.copy()
      b[:, [0, 2]] += x1
      b[:, [1, 3]] += y1
      all_b.append(b)
      all_s.append(s)
      all_l.append(l)

  if not all_b:
    return np.empty((0, 4)), np.empty((0,)), np.empty((0,), int)

  boxes = np.concatenate(all_b)
  scores = np.concatenate(all_s)
  labels = np.concatenate(all_l)

  keep_all = []
  for cls in np.unique(labels):
    idx = np.where(labels == cls)[0]
    keep = _nms(boxes[idx], scores[idx])
    keep_all.extend(idx[keep])
  keep_all = np.array(keep_all, int)
  return boxes[keep_all], scores[keep_all], labels[keep_all]


def visualize(img, boxes, scores, labels, conf_thres=0.3):
  vis_img = img.copy()
  for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, labels):
    if score < conf_thres:
      continue

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cls_id = int(cls_id)

    if cls_id < len(VISDRONE_CLASSES):
      class_name = VISDRONE_CLASSES[cls_id]
    else:
      class_name = f"cls_{cls_id}"

    color = [int(c) for c in COLORS[cls_id % len(COLORS)]]

    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
    label_text = f"{class_name} {score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        vis_img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1
    )
    cv2.putText(
        vis_img,
        label_text,
        (x1, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

  return vis_img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--input_path",
      required=True,
      help="Đường dẫn tới 1 ảnh hoặc thư mục chứa ảnh",
  )
  ap.add_argument(
      "--output_dir", default="out/inference_results", help="Thư mục lưu ảnh kết quả"
  )
  ap.add_argument(
      "--conf_thres",
      type=float,
      default=0.4,
      help="Ngưỡng tự tin để hiển thị box",
  )
  ap.add_argument(
      "--max_dets",
      type=int,
      default=300,
      help="Số lượng object tối đa giữ lại",
  )
  ap.add_argument(
      "--tile",
      action="store_true",
      help="Bật SAHI-style sliced inference cho vật thể nhỏ",
  )
  ap.add_argument("--tile_grid", type=int, nargs=2, default=[2, 2])
  a, unknown = ap.parse_known_args()

  opt = opts().init(unknown)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net_w, net_h = opt.input_wh[0], opt.input_wh[1]

  print(f"Đang khởi tạo mô hình {opt.arch}...")
  model = create_model(opt.arch, opt)
  assert opt.load_model, "Yêu cầu tham số --load_model"
  model = load_model(model, opt.load_model)
  model = model.to(device).eval()

  post = FalconJDEPostProcessor(
      num_classes=opt.num_classes, num_top_queries=a.max_dets
  )
  # Plain resize -> không set_net_hw (postprocessor đảo ngược bằng norm*orig)

  if os.path.isdir(a.input_path):
    img_files = sorted(
        p
        for p in glob.glob(os.path.join(a.input_path, "*"))
        if p.lower().endswith(_IMG_EXTS)
    )
  else:
    img_files = [a.input_path] if a.input_path.lower().endswith(_IMG_EXTS) else []

  print(
      f"Tìm thấy {len(img_files)} ảnh. Kích thước mạng: {net_w}x{net_h} |"
      f" Tiled Mode: {a.tile}"
  )
  if not img_files:
    print("Không tìm thấy ảnh hợp lệ!")
    return

  os.makedirs(a.output_dir, exist_ok=True)

  for n, path in enumerate(img_files):
    img = cv2.imread(path)
    if img is None:
      print(f"Không thể đọc ảnh: {path}")
      continue

    t0 = time.time()
    if a.tile:
      boxes, scores, labels = infer_tiled(
          model,
          post,
          img,
          net_h,
          net_w,
          device,
          rows=a.tile_grid[0],
          cols=a.tile_grid[1],
      )
    else:
      boxes, scores, labels = infer_image(model, post, img, net_h, net_w, device)

    if len(scores) > a.max_dets:
      top = scores.argsort()[::-1][: a.max_dets]
      boxes, scores, labels = boxes[top], scores[top], labels[top]

    print(
        f"[{n+1}/{len(img_files)}] {os.path.basename(path)} | Thời gian xử lý:"
        f" {(time.time() - t0)*1000:.1f}ms"
    )

    vis_img = visualize(img, boxes, scores, labels, conf_thres=a.conf_thres)
    out_path = os.path.join(a.output_dir, os.path.basename(path))
    cv2.imwrite(out_path, vis_img)

  print(f"Đã hoàn thành! Kết quả trực quan được lưu tại: {a.output_dir}")


if __name__ == "__main__":
  main()