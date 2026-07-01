"""
Script Gộp 2 bộ dữ liệu VisDrone (DET và MOT) ĐÃ ĐƯỢC CONVERT SANG COCO (7 Class).
[UPDATE]: Dùng RANDOM SAMPLING cho MOT để triệt tiêu tính liên tục thời gian.
Giữ nguyên Category ID (Vì 2 file JSON đầu vào đã là 7 class).
"""

import os
import json
import argparse
import shutil
import glob
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Cấu trúc Category 7-Class (Phải khớp hoàn toàn với output của 2 script trước)
TARGET_CATEGORIES = [
    {'id': 1, 'name': 'pedestrian', 'supercategory': 'object'},
    {'id': 2, 'name': 'bicycle', 'supercategory': 'object'},
    {'id': 3, 'name': 'car', 'supercategory': 'object'},
    {'id': 4, 'name': 'van', 'supercategory': 'object'},
    {'id': 5, 'name': 'truck', 'supercategory': 'object'},
    {'id': 6, 'name': 'bus', 'supercategory': 'object'},
    {'id': 7, 'name': 'motor', 'supercategory': 'object'},
]

def find_json_file(directory):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file .json nào trong {directory}")
    return json_files[0]

def copy_worker(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_root', required=True, help='Đường dẫn gốc VisDrone-DET (ĐÃ CONVERT COCO)')
    parser.add_argument('--mot_root', required=True, help='Đường dẫn gốc VisDrone-MOT (ĐÃ CONVERT COCO)')
    parser.add_argument('--output_root', required=True, help='Đường dẫn xuất dữ liệu gộp')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max_frames_per_seq', type=int, default=15, 
                        help='Số ảnh lấy RANDOM tối đa từ 1 video MOT.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    for split in args.splits:
        print(f"\n{'='*60}\n[XỬ LÝ TẬP {split.upper()}]\n{'='*60}")
        
        out_split_dir = os.path.join(args.output_root, split)
        out_img_dir = os.path.join(out_split_dir, 'images')
        os.makedirs(out_img_dir, exist_ok=True)
        
        try:
            det_json_path = find_json_file(os.path.join(args.det_root, split, 'annotations'))
            mot_json_path = find_json_file(os.path.join(args.mot_root, split, 'annotations'))
        except Exception as e:
            print(f"BỎ QUA {split}: {e}")
            continue

        print(f"Đọc DET: {det_json_path}")
        with open(det_json_path, 'r') as f:
            det_data = json.load(f)
            
        print(f"Đọc MOT: {mot_json_path}")
        with open(mot_json_path, 'r') as f:
            mot_data = json.load(f)

        merged_images = []
        merged_anns = []
        copy_tasks = []
        
        global_img_id = 0
        global_ann_id = 0
        
        def process_dataset(dataset_data, root_dir, prefix, is_mot=False):
            nonlocal global_img_id, global_ann_id
            img_id_map = {}
            images_to_process = []
            
            # [RANDOM SAMPLING LỌC ẢNH CHO MOT]
            if is_mot and args.max_frames_per_seq > 0:
                seq_dict = defaultdict(list)
                for img in dataset_data['images']:
                    # Tùy thuộc vào cấu trúc thư mục của bạn, file_name có dạng: 'seq_name/0000001.jpg'
                    seq_name = os.path.dirname(img['file_name']) 
                    seq_dict[seq_name].append(img)
                
                print(f"  [{prefix.upper()}] Có {len(seq_dict)} videos. Bốc RANDOM tối đa {args.max_frames_per_seq} ảnh/video.")
                
                for seq_name, imgs in seq_dict.items():
                    if len(imgs) <= args.max_frames_per_seq:
                        sampled_imgs = imgs
                    else:
                        sampled_imgs = random.sample(imgs, args.max_frames_per_seq)
                        sampled_imgs = sorted(sampled_imgs, key=lambda x: x['file_name'])
                    images_to_process.extend(sampled_imgs)
            else:
                images_to_process = dataset_data['images']
            
            # XỬ LÝ ẢNH
            for img in images_to_process:
                old_id = img['id']
                # Lấy tên file gốc (ví dụ: 'uav0000013_00000_v/0000001.jpg' -> 'uav0000013_00000_v_0000001.jpg')
                old_file_name = img['file_name']
                
                global_img_id += 1
                img_id_map[old_id] = global_img_id
                
                safe_name = old_file_name.replace('/', '_').replace('\\', '_')
                new_file_name = f"{prefix}_{safe_name}"
                
                # Copy ảnh
                src_path = os.path.join(root_dir, split, 'images', old_file_name)
                dst_path = os.path.join(out_img_dir, new_file_name)
                
                if os.path.exists(src_path):
                    copy_tasks.append((src_path, dst_path))
                    
                merged_images.append({
                    'id': global_img_id,
                    'file_name': new_file_name,
                    'width': img['width'],
                    'height': img['height']
                })
            
            # XỬ LÝ ANNOTATIONS
            valid_anns = 0
            for ann in dataset_data['annotations']:
                if ann['image_id'] not in img_id_map:
                    continue
                
                # KHÔNG MAP CLASS NỮA! (Giữ nguyên Category 7-class)
                cat_id = ann['category_id']
                if cat_id < 1 or cat_id > 7:
                    continue  # Đề phòng có class lạ (dù 2 file trước đã dọn rồi)
                
                global_ann_id += 1
                valid_anns += 1
                
                new_ann = ann.copy()
                new_ann['id'] = global_ann_id
                new_ann['image_id'] = img_id_map[ann['image_id']]
                
                # Xóa ID Tracking (Để trở thành tập thuần Detection)
                new_ann.pop('track_id', None)
                new_ann.pop('seq_id', None)
                
                merged_anns.append(new_ann)
                    
            print(f"  [{prefix.upper()}] Xong! Góp {len(img_id_map)} ảnh | {valid_anns} boxes.")

        # CHẠY 2 TẬP
        process_dataset(det_data, args.det_root, 'det', is_mot=False)
        process_dataset(mot_data, args.mot_root, 'mot', is_mot=True)
        
        print(f"\nĐang copy {len(copy_tasks)} hình ảnh...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(lambda x: copy_worker(*x), copy_tasks), total=len(copy_tasks)))
            
        out_json_path = os.path.join(out_split_dir, 'annotations', f'instances_{split}.json')
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        
        merged_json = {
            'images': merged_images,
            'annotations': merged_anns,
            'categories': TARGET_CATEGORIES
        }
        
        with open(out_json_path, 'w') as f:
            json.dump(merged_json, f)
            
        print(f"Hoàn thành {split}! Tổng ảnh: {len(merged_images)} | Tổng Bbox: {len(merged_anns)}")

if __name__ == '__main__':
    main()