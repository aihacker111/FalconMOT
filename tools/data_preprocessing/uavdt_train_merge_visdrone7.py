"""
uavdt_train_merge_visdrone7.py
Convert the UAV-DT *train* split (official format) into the SAME 7-class COCO that
visdrone2coco_7cls_mot.py produces, and MERGE it into an existing VisDrone train
JSON so you can train on VisDrone + UAVDT jointly.

Why: UAVDT contributes many high-altitude / bird-view vehicle scenes that VisDrone
under-represents, which should help the model on those conditions.

UAV-DT has only vehicles -> mapped into the VisDrone 7-class space:
    UAVDT car(1)   -> VisDrone car   (3)
    UAVDT truck(2) -> VisDrone truck (5)
    UAVDT bus(3)   -> VisDrone bus   (6)
    (VisDrone pedestrian/bicycle/van/motor get NO new boxes from UAVDT.)

Merge guarantees:
    * image ids / annotation ids continue after VisDrone's max (no collision).
    * track_ids continue PER CLASS after VisDrone's max for that class, so
      (category_id, track_id) stays globally unique across the merged set.
    * UAVDT ignore regions are BLACKED OUT in the written frames (same as the
      VisDrone training pipeline).
    * UAVDT images are written/symlinked INTO the VisDrone images root as
      <SEQ>/<frame:07d>.jpg, so the merged JSON has a single image root.

Input (official UAVDT):
    <uavdt_root>/UAV-benchmark-M/<SEQ>/img000001.jpg
    <uavdt_root>/GT/<SEQ>_gt.txt , <SEQ>_gt_ignore.txt
Train sequences: from --attr_dir M_attr/train (authoritative), else all M**** in
UAV-benchmark-M MINUS the 20 official test sequences (never train on test).

Output:
    <visdrone_images>/<SEQ>/<frame:07d>.jpg          (UAVDT frames added here)
    --output_json (default: instances_train_merged.json next to --visdrone_coco)
"""

import os
import re
import json
import glob
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# Official UAVDT-MOT test split -> excluded from training by default.
UAVDT_TEST_SEQS = {
    'M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
    'M0701', 'M0801', 'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101',
    'M1301', 'M1302', 'M1303', 'M1401',
}
# UAVDT category (1=car,2=truck,3=bus) -> VisDrone 7-class id
UAVDT_TO_VISDRONE7 = {1: 3, 2: 5, 3: 6}
V7_NAMES = {1: 'pedestrian', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'bus', 7: 'motor'}


def _find_dir(root, name):
    if os.path.basename(root.rstrip('/')) == name and os.path.isdir(root):
        return root
    hits = [d for d in glob.glob(os.path.join(root, '**', name), recursive=True) if os.path.isdir(d)]
    return hits[0] if hits else None


def _read_attr_seqs(attr_dir):
    if not attr_dir or not os.path.isdir(attr_dir):
        return []
    return sorted({os.path.basename(f)[:-len('_attr.txt')]
                   for f in glob.glob(os.path.join(attr_dir, '*_attr.txt'))})


def _frame_index_from_name(fname):
    m = re.findall(r'(\d+)', os.path.basename(fname))
    return int(m[-1]) if m else -1


def _parse_uavdt_gt(gt_dir, seq, suffix_order=('_gt.txt', '_gt_whole.txt')):
    path = None
    for suf in suffix_order:
        c = glob.glob(os.path.join(gt_dir, '**', f'{seq}{suf}'), recursive=True)
        if c:
            path = c[0]; break
    if path is None:
        return {}
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = re.split(r'[,\s]+', line.strip())
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0])); tid = int(float(p[1]))
                x, y, w, h = map(float, p[2:6])
                cat = int(float(p[8])) if len(p) > 8 else 0
            except ValueError:
                continue
            if w > 0 and h > 0:
                out[fr].append((tid, cat, (x, y, w, h)))
    return dict(out)


def _load_gt_ignore(gt_dir, seq):
    c = glob.glob(os.path.join(gt_dir, '**', f'{seq}_gt_ignore.txt'), recursive=True)
    if not c:
        return {}
    out = defaultdict(list)
    with open(c[0]) as f:
        for line in f:
            p = re.split(r'[,\s]+', line.strip())
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0])); x, y, w, h = map(float, p[2:6])
            except ValueError:
                continue
            if w > 0 and h > 0:
                out[fr].append([x, y, w, h])
    return dict(out)


def _place_image(src, dst, mode, overwrite):
    if os.path.islink(dst) or os.path.isfile(dst):
        if not overwrite:
            return
        os.remove(dst)
    if mode == 'symlink':
        os.symlink(os.path.abspath(src), dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description='Merge UAVDT-train into VisDrone 7-class train COCO.')
    ap.add_argument('--visdrone_coco', required=True,
                    help='Existing VisDrone train JSON (instances_train.json, 7-class).')
    ap.add_argument('--uavdt_root', required=True,
                    help='Official UAVDT root containing UAV-benchmark-M/ and GT/.')
    ap.add_argument('--visdrone_images', default=None,
                    help='VisDrone train images root (single root of the merged JSON). '
                         'Default: inferred as <coco>/../images.')
    ap.add_argument('--attr_dir', default=None,
                    help='M_attr/train folder -> authoritative train sequence list. '
                         'If omitted, uses all M**** minus the 20 official test sequences.')
    ap.add_argument('--output_json', default=None,
                    help='Merged JSON path (default: instances_train_merged.json beside --visdrone_coco).')
    ap.add_argument('--link', choices=['copy', 'symlink', 'hardlink'], default='symlink')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    # ── 1. load VisDrone train json, compute offsets ──
    with open(args.visdrone_coco) as f:
        vd = json.load(f)
    vd_imgs, vd_anns = vd['images'], vd['annotations']
    cats = vd.get('categories') or [{'id': k, 'name': v, 'supercategory': 'object'}
                                    for k, v in V7_NAMES.items()]
    base_img_id = (max((im['id'] for im in vd_imgs), default=-1) + 1)
    base_ann_id = (max((a['id'] for a in vd_anns), default=-1) + 1)
    # per-class next track_id (VisDrone track_ids are contiguous 0..N-1 per class)
    next_tid = defaultdict(int)
    for a in vd_anns:
        c = a['category_id']; next_tid[c] = max(next_tid[c], a.get('track_id', -1) + 1)
    print(f"[visdrone] {len(vd_imgs):,} images  {len(vd_anns):,} anns  "
          f"next_img_id={base_img_id}  next_ann_id={base_ann_id}")
    print(f"[visdrone] per-class next track_id: "
          f"{{{', '.join(f'{V7_NAMES[c]}:{next_tid[c]}' for c in sorted(next_tid))}}}")

    images_root = args.visdrone_images or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.visdrone_coco))), 'images')
    os.makedirs(images_root, exist_ok=True)

    # ── 2. resolve UAVDT train sequences ──
    img_root = _find_dir(args.uavdt_root, 'UAV-benchmark-M') or args.uavdt_root
    gt_dir = _find_dir(args.uavdt_root, 'GT')
    if gt_dir is None:
        raise FileNotFoundError(f"GT/ not found under {args.uavdt_root}")
    all_seqs = sorted(d for d in os.listdir(img_root)
                      if os.path.isdir(os.path.join(img_root, d)) and d.upper().startswith('M'))
    if args.attr_dir:
        want = set(_read_attr_seqs(args.attr_dir))
        train_seqs = [s for s in all_seqs if s in want]
        print(f"[uavdt] train seqs from M_attr ({len(train_seqs)}): {train_seqs}")
    else:
        train_seqs = [s for s in all_seqs if s not in UAVDT_TEST_SEQS]
        print(f"[uavdt] train seqs = all M**** minus 20 test ({len(train_seqs)}).")
    if not train_seqs:
        raise RuntimeError("No UAVDT train sequences found.")

    # ── 3. convert + merge ──
    add_imgs, add_anns = [], []
    img_id, ann_id = base_img_id, base_ann_id
    n_box = n_ign = 0
    cls_box = defaultdict(int)

    for seq in tqdm(train_seqs, desc='[uavdt-train]'):
        seq_img_dir = os.path.join(img_root, seq)
        paths = []
        for ext in ('*.jpg', '*.JPG', '*.png', '*.jpeg'):
            paths += glob.glob(os.path.join(seq_img_dir, ext))
        frame_to_path = {}
        for p in sorted(paths):
            fi = _frame_index_from_name(p)
            if fi >= 0:
                frame_to_path[fi] = p
        if not frame_to_path:
            continue
        gt = _parse_uavdt_gt(gt_dir, seq)
        ignore_by_frame = _load_gt_ignore(gt_dir, seq)

        # per-seq local track map: (uavdt_tid, v7_cat) -> global track id (per-class continued)
        keys_by_cls = defaultdict(set)
        for rows in gt.values():
            for (tid, cat, _bb) in rows:
                v7 = UAVDT_TO_VISDRONE7.get(cat)
                if v7 is not None:
                    keys_by_cls[v7].add(tid)
        local_map = {}
        for v7, tids in keys_by_cls.items():
            for rank, tid in enumerate(sorted(tids)):
                local_map[(v7, tid)] = next_tid[v7] + rank
            next_tid[v7] += len(tids)

        # one size read per sequence (fixed resolution)
        probe = cv2.imread(next(iter(frame_to_path.values())))
        H0, W0 = (probe.shape[:2] if probe is not None else (0, 0))
        dst_seq_dir = os.path.join(images_root, seq); os.makedirs(dst_seq_dir, exist_ok=True)

        def _write(fr):
            dst = os.path.join(dst_seq_dir, f'{fr:07d}.jpg')
            ign = ignore_by_frame.get(fr, [])
            if ign:
                im = cv2.imread(frame_to_path[fr])
                if im is None:
                    return fr, None
                for (x, y, w, h) in ign:
                    x, y = max(0, int(x)), max(0, int(y))
                    im[y:y + int(h), x:x + int(w)] = 0
                if os.path.islink(dst) or os.path.isfile(dst):
                    os.remove(dst)
                cv2.imwrite(dst, im)
                return fr, (im.shape[0], im.shape[1])
            _place_image(frame_to_path[fr], dst, args.link, args.overwrite)
            return fr, (H0, W0)

        sizes = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(_write, fr) for fr in sorted(frame_to_path)]):
                fr, hw = fut.result()
                if hw is not None:
                    sizes[fr] = hw
        for fr in frame_to_path:
            if ignore_by_frame.get(fr):
                n_ign += len(ignore_by_frame[fr])

        for fr in sorted(frame_to_path):
            if fr not in sizes:
                continue
            H, W = sizes[fr]
            add_imgs.append({'id': img_id, 'file_name': f'{seq}/{fr:07d}.jpg',
                             'height': H, 'width': W, 'seq_id': seq, 'frame_id': fr})
            for (tid, cat, (x, y, w, h)) in gt.get(fr, []):
                v7 = UAVDT_TO_VISDRONE7.get(cat)
                if v7 is None or w <= 0 or h <= 0:
                    continue
                add_anns.append({'id': ann_id, 'image_id': img_id, 'category_id': v7,
                                 'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0,
                                 'track_id': local_map[(v7, tid)]})
                ann_id += 1; n_box += 1; cls_box[v7] += 1
            img_id += 1

    # ── 4. write merged json ──
    merged = {'images': vd_imgs + add_imgs,
              'annotations': vd_anns + add_anns,
              'categories': cats}
    out_json = args.output_json or os.path.join(
        os.path.dirname(os.path.abspath(args.visdrone_coco)), 'instances_train_merged.json')
    with open(out_json, 'w') as f:
        json.dump(merged, f, separators=(',', ':'))

    print(f"\n[merge] added {len(add_imgs):,} UAVDT images, {n_box:,} boxes "
          f"({n_ign:,} ignore boxes blacked).")
    print('  UAVDT boxes per class:', {V7_NAMES[c]: cls_box[c] for c in sorted(cls_box)})
    print(f"[merged] total {len(merged['images']):,} images  {len(merged['annotations']):,} anns")
    print('  per-class total track IDs:',
          {V7_NAMES[c]: next_tid[c] for c in sorted(next_tid)})
    print(f"  -> {out_json}")
    print(f"  (train with --track_ann_file {out_json} and image root {images_root})")


if __name__ == '__main__':
    main()