"""
Script Gộp 2 bộ dữ liệu VisDrone (đã ở định dạng COCO JSON) thành 1 bộ COCO duy nhất.
Tốc độ cực nhanh vì chỉ copy ảnh (không decode/encode bằng cv2) và xử lý JSON trong RAM.
"""

import os
import json
import argparse
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ── Bản đồ Class (10 -> 7 class) ─────────────────────────────────────────────
CLASS_MAPPING = {
    1: 1, 2: 1, 
    3: 2, 
    4: 3, 
    5: 4, 
    6: 5, 
    7: -1, 8: -1,  # Bỏ qua tricycle, awning-tricycle
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

def find_json_file(directory):
    """Tìm file JSON đầu tiên trong thư mục"""
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file .json nào trong {directory}")
    return json_files[0]

def copy_worker(src, dst):
    """Worker copy file"""
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_root', required=True, help='Đường dẫn gốc VisDrone2019-DET-COCO')
    parser.add_argument('--mot_root', required=True, help='Đường dẫn gốc VisDrone2019-COCO (MOT)')
    parser.add_argument('--output_root', required=True, help='Đường dẫn xuất dữ liệu gộp')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n{'='*50}\n[XỬ LÝ TẬP {split.upper()}]\n{'='*50}")
        
        # 1. Setup đường dẫn
        out_split_dir = os.path.join(args.output_root, split)
        out_img_dir = os.path.join(out_split_dir, 'images')
        os.makedirs(out_img_dir, exist_ok=True)
        
        # 2. Định vị file JSON
        try:
            det_json_path = find_json_file(os.path.join(args.det_root, split, 'annotations'))
            mot_json_path = find_json_file(os.path.join(args.mot_root, split, 'annotations'))
        except Exception as e:
            print(f"BỎ QUA {split}: {e}")
            continue

        print(f"Đọc DET JSON: {det_json_path}")
        with open(det_json_path, 'r') as f:
            det_data = json.load(f)
            
        print(f"Đọc MOT JSON: {mot_json_path}")
        with open(mot_json_path, 'r') as f:
            mot_data = json.load(f)

        merged_images = []
        merged_anns = []
        copy_tasks = []
        
        global_img_id = 0
        global_ann_id = 0
        
        # ─────────────────────────────────────────────────────────────────
        # HÀM XỬ LÝ CHUNG CHO TỪNG BỘ DỮ LIỆU
        # ─────────────────────────────────────────────────────────────────
        def process_dataset(dataset_data, root_dir, prefix):
            nonlocal global_img_id, global_ann_id
            
            # Mapping image_id cũ -> image_id mới
            img_id_map = {}
            
            # Xử lý Images
            for img in dataset_data['images']:
                old_id = img['id']
                old_file_name = img['file_name']
                
                global_img_id += 1
                img_id_map[old_id] = global_img_id
                
                # Tạo tên file mới để tránh trùng lặp, ví dụ: "det_0000001_02999_d_0000001.jpg"
                # (Thay thế dấu gạch chéo thư mục bằng gạch dưới để làm phẳng thư mục)
                safe_name = old_file_name.replace('/', '_').replace('\\', '_')
                new_file_name = f"{prefix}_{safe_name}"
                
                # Đường dẫn copy
                src_path = os.path.join(root_dir, split, 'images', old_file_name)
                # MOT COCO đôi khi ảnh nằm trong thư mục 'sequences'
                if not os.path.exists(src_path):
                    src_path = os.path.join(root_dir, split, old_file_name)
                    
                dst_path = os.path.join(out_img_dir, new_file_name)
                
                if os.path.exists(src_path):
                    copy_tasks.append((src_path, dst_path))
                    
                # Lưu vào danh sách ảnh merged
                merged_images.append({
                    'id': global_img_id,
                    'file_name': new_file_name,
                    'width': img['width'],
                    'height': img['height']
                })
            
            # Xử lý Annotations
            valid_anns = 0
            for ann in dataset_data['annotations']:
                # Bỏ qua các annotation trỏ tới ảnh không tồn tại
                if ann['image_id'] not in img_id_map:
                    continue
                
                old_cat = ann['category_id']
                new_cat = CLASS_MAPPING.get(old_cat, -1)
                
                # Bỏ qua Ignore regions (0, 11) và các class bị drop (7, 8)
                if new_cat != -1:
                    global_ann_id += 1
                    valid_anns += 1
                    
                    new_ann = ann.copy()
                    new_ann['id'] = global_ann_id
                    new_ann['image_id'] = img_id_map[ann['image_id']]
                    new_ann['category_id'] = new_cat
                    
                    # Dọn dẹp các field rác của tracking (nếu có)
                    new_ann.pop('track_id', None)
                    new_ann.pop('seq_id', None)
                    
                    merged_anns.append(new_ann)
                    
            print(f"  [{prefix.upper()}] Thêm {len(img_id_map)} ảnh và {valid_anns} bounding boxes hợp lệ.")

        # ─────────────────────────────────────────────────────────────────
        # THỰC THI
        # ─────────────────────────────────────────────────────────────────
        process_dataset(det_data, args.det_root, 'det')
        process_dataset(mot_data, args.mot_root, 'mot')
        
        # Tiến hành Copy ảnh bằng đa luồng (Tốc độ bàn thờ)
        print(f"Đang copy {len(copy_tasks)} hình ảnh vào thư mục chung...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(lambda x: copy_worker(*x), copy_tasks), total=len(copy_tasks)))
            
        # Lưu file JSON
        out_json_path = os.path.join(out_split_dir, f'instances_{split}.json')
        merged_json = {
            'images': merged_images,
            'annotations': merged_anns,
            'categories': TARGET_CATEGORIES
        }
        
        print(f"Lưu file JSON gộp tại: {out_json_path}")
        with open(out_json_path, 'w') as f:
            json.dump(merged_json, f)
            
        print(f"Hoàn thành {split}! Tổng ảnh: {len(merged_images)} | Tổng Bbox: {len(merged_anns)}")

if __name__ == '__main__':
    main()