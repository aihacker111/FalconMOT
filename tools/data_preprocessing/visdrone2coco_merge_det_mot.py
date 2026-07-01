"""
Script gộp dữ liệu từ VisDrone2019-DET và VisDrone2019-MOT thành một bộ dataset COCO duy nhất.
Mục tiêu: Dành riêng cho huấn luyện Object Detection (Stage 1).
"""

import os
import json
import argparse
import glob
from collections import defaultdict
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Bản đồ Class (10 -> 7 class) ─────────────────────────────────────────────
CLASS_MAPPING = {
    1: 1, 2: 1, 
    3: 2, 
    4: 3, 
    5: 4, 
    6: 5, 
    7: -1, 8: -1, 
    9: 6, 
    10: 7
}

TARGET_CATEGORIES = [
    {'id': 1, 'name': 'pedestrian'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'van'},
    {'id': 5, 'name': 'truck'},
    {'id': 6, 'name': 'bus'},
    {'id': 7, 'name': 'motor'},
]

def process_image_worker(task):
    """
    Hàm thực thi đa luồng: Đọc ảnh, bôi đen vùng ignore và lưu lại.
    """
    src_path = task['src_path']
    dst_path = task['dst_path']
    ignore_boxes = task['ignore_boxes']
    
    img = cv2.imread(src_path)
    if img is None:
        return task, None # Lỗi đọc ảnh
    
    h, w = img.shape[:2]
    
    # Bôi đen các vùng cần bỏ qua (nhãn 0 và 11)
    for box in ignore_boxes:
        bx, by, bw, bh = box
        x1, y1 = max(0, int(bx)), max(0, int(by))
        x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
    cv2.imwrite(dst_path, img)
    return task, (w, h)

def main():
    parser = argparse.ArgumentParser(description="Gộp VisDrone DET và MOT sang COCO Detection.")
    parser.add_argument('--det_root', required=True, help='Đường dẫn tới thư mục gốc VisDrone2019-DET')
    parser.add_argument('--mot_root', required=True, help='Đường dẫn tới thư mục gốc VisDrone2019-MOT')
    parser.add_argument('--output_root', required=True, help='Đường dẫn xuất dữ liệu gộp')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='Các tập cần chia (vd: train val)')
    parser.add_argument('--workers', type=int, default=12, help='Số luồng xử lý song song')
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n[{split.upper()}] Bắt đầu thu thập dữ liệu...")
        
        # Sửa lại đường dẫn cho khớp với Terminal của bạn
        det_img_dir = os.path.join(args.det_root, split, 'images')
        det_ann_dir = os.path.join(args.det_root, split, 'annotations')
        
        # MOT thường để ảnh trong folder 'sequences', nếu bạn để trong 'images' thì code sẽ tự thích ứng
        mot_seq_dir = os.path.join(args.mot_root, split, 'sequences')
        if not os.path.exists(mot_seq_dir):
            mot_seq_dir = os.path.join(args.mot_root, split, 'images')
        mot_ann_dir = os.path.join(args.mot_root, split, 'annotations')
        
        out_split_dir = os.path.join(args.output_root, split)
        out_img_dir = os.path.join(out_split_dir, 'images')
        out_json_path = os.path.join(out_split_dir, f'instances_{split}.json')
        os.makedirs(out_img_dir, exist_ok=True)
        
        tasks = [] # Chứa các công việc cần xử lý đa luồng
        
        # ─────────────────────────────────────────────────────────────
        # 1. PARSE DỮ LIỆU DET (Ảnh tĩnh - 8 Cột)
        # ─────────────────────────────────────────────────────────────
        if os.path.isdir(det_img_dir) and os.path.isdir(det_ann_dir):
            det_images = sorted(glob.glob(os.path.join(det_img_dir, '*.jpg')))
            print(f"  > Đang parse {len(det_images)} ảnh DET...")
            
            for img_path in det_images:
                img_name = os.path.basename(img_path)
                ann_path = os.path.join(det_ann_dir, img_name.replace('.jpg', '.txt'))
                if not os.path.exists(ann_path): continue
                
                ignore_boxes, valid_anns = [], []
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = [int(float(x)) for x in line.strip().split(',') if x.strip()]
                        if len(parts) < 8: continue
                        
                        x, y, w, h, score, cat, tr, occ = parts[:8]
                        if cat in [0, 11]:
                            ignore_boxes.append([x, y, w, h])
                        else:
                            new_cat = CLASS_MAPPING.get(cat, -1)
                            if new_cat != -1:
                                valid_anns.append({'cat': new_cat, 'bbox': [x, y, w, h], 'tr': tr, 'occ': occ})
                
                tasks.append({
                    'src_path': img_path,
                    'dst_path': os.path.join(out_img_dir, f"det_{img_name}"),
                    'file_name': f"det_{img_name}",
                    'ignore_boxes': ignore_boxes,
                    'valid_anns': valid_anns
                })
        
        # ─────────────────────────────────────────────────────────────
        # 2. PARSE DỮ LIỆU MOT (Video Sequences - 10 Cột)
        # ─────────────────────────────────────────────────────────────
        if os.path.isdir(mot_seq_dir) and os.path.isdir(mot_ann_dir):
            seqs = sorted(os.listdir(mot_seq_dir))
            print(f"  > Đang parse {len(seqs)} sequences MOT...")
            
            for seq_name in seqs:
                seq_dir = os.path.join(mot_seq_dir, seq_name)
                if not os.path.isdir(seq_dir): continue # Đảm bảo nó là folder
                
                ann_path = os.path.join(mot_ann_dir, f"{seq_name}.txt")
                if not os.path.exists(ann_path): continue
                
                frame_dict = defaultdict(lambda: {'ignore': [], 'valid': []})
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = [int(float(x)) for x in line.strip().split(',') if x.strip()]
                        if len(parts) < 10: continue
                        
                        frame_idx = parts[0]
                        bbox = parts[2:6]
                        cat, tr, occ = parts[7], parts[8], parts[9]
                        
                        if cat in [0, 11]:
                            frame_dict[frame_idx]['ignore'].append(bbox)
                        else:
                            new_cat = CLASS_MAPPING.get(cat, -1)
                            if new_cat != -1:
                                frame_dict[frame_idx]['valid'].append({'cat': new_cat, 'bbox': bbox, 'tr': tr, 'occ': occ})
                
                for img_path in sorted(glob.glob(os.path.join(seq_dir, '*.jpg'))):
                    frame_idx = int(os.path.basename(img_path).replace('.jpg', ''))
                    
                    tasks.append({
                        'src_path': img_path,
                        'dst_path': os.path.join(out_img_dir, f"mot_{seq_name}_{os.path.basename(img_path)}"),
                        'file_name': f"mot_{seq_name}_{os.path.basename(img_path)}",
                        'ignore_boxes': frame_dict[frame_idx]['ignore'],
                        'valid_anns': frame_dict[frame_idx]['valid']
                    })

        # ─────────────────────────────────────────────────────────────
        # 3. THỰC THI & BUILD JSON ĐỒNG BỘ
        # ─────────────────────────────────────────────────────────────
        print(f"  > Bắt đầu xử lý ảnh và bôi đen (Total tasks: {len(tasks)})...")
        
        coco_images = []
        coco_anns = []
        img_id_counter = 0
        ann_id_counter = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Gửi toàn bộ task vào luồng
            futures = [executor.submit(process_image_worker, t) for t in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Images"):
                task, res = future.result()
                if res is not None:
                    width, height = res
                    img_id_counter += 1
                    
                    # 1. Ghi Image Meta
                    coco_images.append({
                        'id': img_id_counter,
                        'file_name': task['file_name'],
                        'width': width,
                        'height': height
                    })
                    
                    # 2. Ghi Annotations
                    for ann in task['valid_anns']:
                        ann_id_counter += 1
                        bbox = ann['bbox']
                        coco_anns.append({
                            'id': ann_id_counter,
                            'image_id': img_id_counter,
                            'category_id': ann['cat'],
                            'bbox': bbox,
                            'area': bbox[2] * bbox[3],
                            'iscrowd': 0,
                            'truncation': ann['tr'],
                            'occlusion': ann['occ']
                        })

        # 4. Lưu File
        coco_data = {'images': coco_images, 'annotations': coco_anns, 'categories': TARGET_CATEGORIES}
        with open(out_json_path, 'w') as f:
            json.dump(coco_data, f)
            
        print(f"  [Hoàn thành {split}] Ảnh hợp lệ: {len(coco_images)} | Bboxes: {len(coco_anns)}")

if __name__ == '__main__':
    main()