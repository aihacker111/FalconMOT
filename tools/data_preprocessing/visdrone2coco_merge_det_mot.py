"""
Script gộp dữ liệu từ VisDrone2019-DET và VisDrone2019-MOT thành một bộ dataset COCO duy nhất.
Mục tiêu: Dành riêng cho huấn luyện Object Detection (Stage 1).
Đặc điểm:
- Đưa định dạng của MOT (10 cột) về cùng format chuẩn với DET (8 cột).
- Loại bỏ hoàn toàn các khái niệm của tracking (track_id, frame_id) trong file JSON đầu ra.
- Gộp chung hình ảnh của DET và MOT vào một thư mục (có tiền tố det_ và mot_ để chống trùng).
- Bôi đen (Blackout) các vùng ignore (nhãn 0 và 11).
- Áp dụng chuẩn 7 class (gộp pedestrian+people, bỏ qua tricycle).
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
# 1: pedestrian, 2: people -> 1 (pedestrian)
# 3: bicycle               -> 2 (bicycle)
# 4: car                   -> 3 (car)
# 5: van                   -> 4 (van)
# 6: truck                 -> 5 (truck)
# 7: tricycle              -> BỎ QUA (-1)
# 8: awning-tricycle       -> BỎ QUA (-1)
# 9: bus                   -> 6 (bus)
# 10: motor                -> 7 (motor)
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

def process_image_worker(src_path, dst_path, ignore_boxes):
    """
    Hàm thực thi đa luồng: Đọc ảnh, bôi đen vùng ignore và lưu lại.
    Trả về chiều rộng (w) và chiều cao (h) của ảnh để ghi vào JSON.
    """
    img = cv2.imread(src_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Bôi đen các vùng cần bỏ qua (nhãn 0 và 11)
    for box in ignore_boxes:
        bx, by, bw, bh = box
        # Giới hạn bbox không vượt quá kích thước ảnh
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
    cv2.imwrite(dst_path, img)
    return w, h

def main():
    parser = argparse.ArgumentParser(description="Gộp VisDrone DET và MOT sang định dạng COCO Detection chuẩn cho Stage 1.")
    parser.add_argument('--det_root', required=True, help='Đường dẫn tới thư mục gốc VisDrone2019-DET')
    parser.add_argument('--mot_root', required=True, help='Đường dẫn tới thư mục gốc VisDrone2019-MOT')
    parser.add_argument('--output_root', required=True, help='Đường dẫn xuất dữ liệu gộp')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='Các tập cần chia (vd: train val)')
    parser.add_argument('--workers', type=int, default=12, help='Số luồng xử lý song song')
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n[{split.upper()}] Bắt đầu gộp và xử lý dữ liệu...")
        
        # Đường dẫn nguồn
        det_img_dir = os.path.join(args.det_root, f'VisDrone2019-DET-{split}', 'images')
        det_ann_dir = os.path.join(args.det_root, f'VisDrone2019-DET-{split}', 'annotations')
        
        mot_seq_dir = os.path.join(args.mot_root, f'VisDrone2019-MOT-{split}', 'sequences')
        mot_ann_dir = os.path.join(args.mot_root, f'VisDrone2019-MOT-{split}', 'annotations')
        
        # Đường dẫn đích
        out_split_dir = os.path.join(args.output_root, split)
        out_img_dir = os.path.join(out_split_dir, 'images')
        out_json_path = os.path.join(out_split_dir, f'instances_{split}.json')
        
        os.makedirs(out_img_dir, exist_ok=True)
        
        coco_images = []
        coco_anns = []
        img_id_counter = 0
        ann_id_counter = 0
        
        tasks = [] # Danh sách các tác vụ gửi vào luồng
        img_meta_map = {} # Mapping từ img_id sang thông tin để cập nhật w, h sau này
        
        # ─────────────────────────────────────────────────────────────
        # 1. PARSE DỮ LIỆU DET (Ảnh tĩnh - 8 Cột)
        # ─────────────────────────────────────────────────────────────
        if os.path.isdir(det_img_dir) and os.path.isdir(det_ann_dir):
            det_images = sorted(glob.glob(os.path.join(det_img_dir, '*.jpg')))
            print(f"  > Tìm thấy {len(det_images)} ảnh DET.")
            
            for img_path in det_images:
                img_name = os.path.basename(img_path)
                ann_path = os.path.join(det_ann_dir, img_name.replace('.jpg', '.txt'))
                
                if not os.path.exists(ann_path):
                    continue
                
                img_id_counter += 1
                dst_img_name = f"det_{img_name}" # Thêm tiền tố chống trùng
                dst_img_path = os.path.join(out_img_dir, dst_img_name)
                
                ignore_boxes = []
                
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = [int(float(x)) for x in line.strip().split(',') if x.strip()]
                        if len(parts) < 8: continue
                        
                        # DET Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                        x, y, w, h, score, cat, tr, occ = parts[:8]
                        
                        if cat in [0, 11]:
                            ignore_boxes.append([x, y, w, h])
                        else:
                            new_cat = CLASS_MAPPING.get(cat, -1)
                            if new_cat != -1:
                                ann_id_counter += 1
                                # Chuẩn COCO Detection tĩnh, KHÔNG CÓ track_id
                                coco_anns.append({
                                    'id': ann_id_counter,
                                    'image_id': img_id_counter,
                                    'category_id': new_cat,
                                    'bbox': [x, y, w, h],
                                    'area': w * h,
                                    'iscrowd': 0,
                                    'truncation': tr,
                                    'occlusion': occ
                                })
                
                img_meta_map[img_id_counter] = {
                    'id': img_id_counter,
                    'file_name': dst_img_name
                }
                
                tasks.append((img_id_counter, img_path, dst_img_path, ignore_boxes))
        else:
            print(f"  > Bỏ qua DET: Không tìm thấy thư mục {det_img_dir}")

        # ─────────────────────────────────────────────────────────────
        # 2. PARSE DỮ LIỆU MOT (Video Sequences - 10 Cột)
        # ─────────────────────────────────────────────────────────────
        if os.path.isdir(mot_seq_dir) and os.path.isdir(mot_ann_dir):
            seqs = sorted(os.listdir(mot_seq_dir))
            print(f"  > Tìm thấy {len(seqs)} sequences MOT.")
            
            for seq_name in seqs:
                seq_dir = os.path.join(mot_seq_dir, seq_name)
                ann_path = os.path.join(mot_ann_dir, f"{seq_name}.txt")
                
                if not os.path.exists(ann_path):
                    continue
                
                frame_dict = defaultdict(list)
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = [int(float(x)) for x in line.strip().split(',') if x.strip()]
                        if len(parts) < 10: continue
                        
                        # MOT Format: <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                        frame_idx = parts[0]
                        # Bỏ qua parts[1] (target_id) vì không cần thiết cho Detection
                        bbox = parts[2:6]
                        cat = parts[7]
                        tr, occ = parts[8], parts[9]
                        
                        if cat in [0, 11]:
                            frame_dict[frame_idx].append({'type': 'ignore', 'bbox': bbox})
                        else:
                            new_cat = CLASS_MAPPING.get(cat, -1)
                            if new_cat != -1:
                                frame_dict[frame_idx].append({
                                    'type': 'valid',
                                    'cat': new_cat,
                                    'bbox': bbox,
                                    'tr': tr,
                                    'occ': occ
                                })
                
                img_files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
                for img_path in img_files:
                    frame_idx = int(os.path.basename(img_path).replace('.jpg', ''))
                    
                    items = frame_dict.get(frame_idx, [])
                    ignore_boxes = [x['bbox'] for x in items if x['type'] == 'ignore']
                    valid_items = [x for x in items if x['type'] == 'valid']
                    
                    img_id_counter += 1
                    dst_img_name = f"mot_{seq_name}_{os.path.basename(img_path)}"
                    dst_img_path = os.path.join(out_img_dir, dst_img_name)
                    
                    for item in valid_items:
                        ann_id_counter += 1
                        # Chuẩn COCO Detection tĩnh, KHÔNG CÓ track_id
                        coco_anns.append({
                            'id': ann_id_counter,
                            'image_id': img_id_counter,
                            'category_id': item['cat'],
                            'bbox': item['bbox'],
                            'area': item['bbox'][2] * item['bbox'][3],
                            'iscrowd': 0,
                            'truncation': item['tr'],
                            'occlusion': item['occ']
                        })
                    
                    img_meta_map[img_id_counter] = {
                        'id': img_id_counter,
                        'file_name': dst_img_name
                    }
                    
                    tasks.append((img_id_counter, img_path, dst_img_path, ignore_boxes))
        else:
            print(f"  > Bỏ qua MOT: Không tìm thấy thư mục {mot_seq_dir}")

        # ─────────────────────────────────────────────────────────────
        # 3. THỰC THI XỬ LÝ ẢNH ĐA LUỒNG
        # ─────────────────────────────────────────────────────────────
        print(f"  > Bắt đầu blackout và copy {len(tasks)} ảnh với {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_img_id = {
                executor.submit(process_image_worker, t[1], t[2], t[3]): t[0] 
                for t in tasks
            }
            
            for future in tqdm(as_completed(future_to_img_id), total=len(tasks), desc="Processing Images"):
                img_id = future_to_img_id[future]
                res = future.result()
                if res is not None:
                    w, h = res
                    img_meta_map[img_id]['width'] = w
                    img_meta_map[img_id]['height'] = h
                    coco_images.append(img_meta_map[img_id])
                else:
                    print(f"  [Lỗi] Không đọc được ảnh cho ID: {img_id}")

        # ─────────────────────────────────────────────────────────────
        # 4. GHI FILE JSON
        # ─────────────────────────────────────────────────────────────
        coco_data = {
            'images': coco_images,
            'annotations': coco_anns,
            'categories': TARGET_CATEGORIES
        }
        
        with open(out_json_path, 'w') as f:
            json.dump(coco_data, f)
            
        print(f"  [Hoàn thành] Đã tạo thành công bộ dữ liệu gộp {split}!")
        print(f"    - Tổng số hình ảnh: {len(coco_images)}")
        print(f"    - Tổng số Bboxes : {len(coco_anns)}")
        print(f"    - File JSON lưu tại: {out_json_path}\n")

if __name__ == '__main__':
    main()