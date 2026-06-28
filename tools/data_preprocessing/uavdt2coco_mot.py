"""
uavdt2coco_mot.py
Convert UAV-DT -> COCO JSON in the SAME format FalconMOT's VisDrone 5-class
benchmark uses, so the tracker + tools/eval_mot_uavdt.py + hota.py work unchanged.

Supports TWO source layouts (auto-detected, or force with --source_format):

(A) OFFICIAL UAVDT  (--source_format uavdt)   << your current data
    <root>/UAV-benchmark-M/<SEQ>/img000001.jpg ...
    <root>/GT/<SEQ>_gt.txt            (MOT ground truth)
    <root>/GT/<SEQ>_gt_whole.txt      (DET ground truth; fallback)
    <root>/GT/<SEQ>_gt_ignore.txt     (ignore regions)
    GT line format (9 cols):
        frame_index, target_id, x, y, w, h, out-of-view, occlusion, category
        category: 1=car, 2=truck, 3=bus
    ignore line format:
        frame_index, id, x, y, w, h, flag, -1, -1     (ignore box = cols 2..5)

(B) SUPERVISELY  (--source_format supervisely)  (Datasets-Ninja / Kaggle export)
    <root>/<split>/img/<SEQ>_img<NNNNNN>.jpg
    <root>/<split>/ann/<SEQ>_img<NNNNNN>.jpg.json   (per-image Supervisely JSON)

OUTPUT (identical schema to visdrone2coco_5cls_benchmark_mot.py):
    <out>/<split>/images/<SEQ>/<frame:07d>.jpg
    <out>/<split>/annotations/instances_<split>.json
        images[]:      {id, file_name='<SEQ>/<frame:07d>.jpg', height, width, seq_id, frame_id}
        annotations[]: {id, image_id, category_id, bbox=[x,y,w,h], area, iscrowd, track_id}
        categories[]:  per --class_scheme

IGNORE: in BOTH modes, official UAVDT ignore regions are BLACKED OUT in the output
frames (matching the VisDrone pipeline), so detections there are not penalised as FPs.
Ignore is read from GT/<SEQ>_gt_ignore.txt automatically in uavdt mode, or from
--gt_ignore_dir in supervisely mode.
"""

import os
import re
import json
import glob
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


# ── UAV-DT classes (official category ids) ────────────────────────────────────
UAVDT_CAT_NAMES = {1: 'car', 2: 'truck', 3: 'bus'}
CLASSTITLE_TO_UAVDT = {'car': 1, 'truck': 2, 'bus': 3}          # supervisely classTitle
IGNORE_TITLES = {'ignore', 'ignored', 'ignored region', 'ignored_region', 'region'}
TRACKID_TAG_NAMES = {'trackid', 'track', 'objectid', 'targetid', 'gtid', 'tid', 'instanceid', 'id'}

_NAME_RE = re.compile(r'^(?P<seq>.+?)_img0*(?P<frame>\d+)\.(?:jpg|jpeg|png)(?:\.json)?$',
                      re.IGNORECASE)

# Official UAVDT-MOT test split (verified against M_attr/test/*_attr.txt). 20 sequences.
UAVDT_TEST_SEQS = [
    'M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
    'M0701', 'M0801', 'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101',
    'M1301', 'M1302', 'M1303', 'M1401',
]


def _read_attr_seqs(attr_dir):
    """Read the authoritative sequence list from an M_attr/{train,test} folder.

    Each test/train sequence is one '<SEQ>_attr.txt' file; the filename stem (minus
    the '_attr' suffix) is the sequence name. Returns a sorted list, or [] if none.
    """
    if not attr_dir or not os.path.isdir(attr_dir):
        return []
    seqs = []
    for f in glob.glob(os.path.join(attr_dir, '*_attr.txt')):
        name = os.path.basename(f)[:-len('_attr.txt')]
        if name:
            seqs.append(name)
    return sorted(set(seqs))


# ── Target class schemes ──────────────────────────────────────────────────────
#   map: uavdt category (1=car,2=truck,3=bus) -> output category id
SCHEMES = {
    'vehicle1':  {'categories': {1: 'vehicle'}, 'map': {1: 1, 2: 1, 3: 1}},
    'visdrone5': {'categories': {1: 'pedestrian', 2: 'car', 3: 'van', 4: 'truck', 5: 'bus'},
                  'map': {1: 2, 2: 4, 3: 5}},
    'uavdt3':    {'categories': {1: 'car', 2: 'truck', 3: 'bus'}, 'map': {1: 1, 2: 2, 3: 3}},
}


class TrackIDManager:
    """Globally-unique, per-output-class track ids (mirrors the VisDrone converter)."""
    def __init__(self, num_target_classes):
        self._start = [0] * num_target_classes

    def build_seq_map(self, raw_keys_by_newcat):
        seq_map = {}
        for new_cat, keys in raw_keys_by_newcat.items():
            idx = new_cat - 1
            for rank, key in enumerate(sorted(keys, key=lambda k: str(k))):
                seq_map[(new_cat, key)] = rank + self._start[idx]
            self._start[idx] += len(keys)
        return seq_map

    def totals(self):
        return {i + 1: n for i, n in enumerate(self._start) if n > 0}


# ── small helpers ─────────────────────────────────────────────────────────────
def parse_name(fname):
    m = _NAME_RE.match(os.path.basename(fname))
    return (m.group('seq'), int(m.group('frame'))) if m else (None, None)


def _frame_index_from_name(fname):
    m = re.findall(r'(\d+)', os.path.basename(fname))
    return int(m[-1]) if m else -1


def bbox_from_exterior(ext):
    (x1, y1), (x2, y2) = ext[0], ext[1]
    return float(min(x1, x2)), float(min(y1, y2)), float(abs(x2 - x1)), float(abs(y2 - y1))


def resolve_track_key(obj, source='auto'):
    tags = obj.get('tags') or []
    if source.startswith('tag:'):
        want = source[4:].lower()
        for t in tags:
            if str(t.get('name', '')).lower() == want:
                return t.get('value')
        return None
    if source in ('objectKey', 'key', 'id'):
        return obj.get(source)
    for t in tags:
        nm = str(t.get('name', '')).lower().replace(' ', '').replace('_', '')
        if nm in TRACKID_TAG_NAMES and t.get('value') is not None:
            return t.get('value')
    for k in ('objectKey', 'key'):
        if obj.get(k) is not None:
            return obj.get(k)
    return None


def _find_dir(root, name):
    """Find a sub-directory called <name> anywhere under root (incl. root itself)."""
    if os.path.basename(root.rstrip('/')) == name and os.path.isdir(root):
        return root
    hits = [d for d in glob.glob(os.path.join(root, '**', name), recursive=True)
            if os.path.isdir(d)]
    return hits[0] if hits else None


def _load_gt_ignore(gt_dir, seq):
    """GT/<seq>_gt_ignore.txt -> {frame_1based: [[x,y,w,h], ...]}."""
    if not gt_dir:
        return {}
    cands = glob.glob(os.path.join(gt_dir, '**', f'{seq}_gt_ignore.txt'), recursive=True)
    if not cands:
        return {}
    out = defaultdict(list)
    with open(cands[0]) as f:
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


def _parse_uavdt_gt(gt_dir, seq, gt_suffix_order=('_gt.txt', '_gt_whole.txt')):
    """Official UAVDT GT -> {frame: [(track_id, uavdt_cat, (x,y,w,h)), ...]}, plus the file used."""
    path = None
    for suf in gt_suffix_order:
        cands = glob.glob(os.path.join(gt_dir, '**', f'{seq}{suf}'), recursive=True)
        if cands:
            path = cands[0]; break
    if path is None:
        return {}, None
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
    return dict(out), path


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


# ── per-sequence extraction: each returns
#     frame_to_path : {frame -> src image path}
#     dets_by_frame : {frame -> [(new_cat, raw_track_key, (x,y,w,h)), ...]}
#     ignore_by_frame : {frame -> [[x,y,w,h], ...]}
#     wh_by_frame  : {frame -> (H, W)} or {} (then read from image)
def _seq_official(seq, img_seq_dir, gt_dir, scheme, debug_holder, gt_suffix_order):
    cls_map, cats = scheme['map'], scheme['categories']
    single = (len(cats) == 1)
    # images
    paths = []
    for ext in ('*.jpg', '*.JPG', '*.png', '*.jpeg'):
        paths += glob.glob(os.path.join(img_seq_dir, ext))
    frame_to_path = {}
    for p in sorted(paths):
        fi = _frame_index_from_name(p)
        if fi >= 0:
            frame_to_path[fi] = p
    # gt
    gt, gt_path = _parse_uavdt_gt(gt_dir, seq, gt_suffix_order)
    dets_by_frame = defaultdict(list)
    raw_keys_by_newcat = defaultdict(set)
    for fr, rows in gt.items():
        for (tid, cat, bb) in rows:
            new_cat = cls_map.get(cat)
            if new_cat is None:
                if single:
                    new_cat = 1               # vehicle1: keep all vehicles even if cat col is odd
                else:
                    continue
            raw_key = ('t', tid)
            dets_by_frame[fr].append((new_cat, raw_key, bb))
            raw_keys_by_newcat[new_cat].add(raw_key)
    ignore_by_frame = _load_gt_ignore(gt_dir, seq)
    if debug_holder.get('print'):
        sample = next(iter(gt.items()), None)
        print(f"\n[debug] seq={seq}  gt={gt_path}")
        print(f"[debug] images={len(frame_to_path)}  gt_frames={len(gt)}  "
              f"ignore_frames={len(ignore_by_frame)}")
        print(f"[debug] sample GT (frame {sample[0] if sample else '-'}): "
              f"{sample[1][:2] if sample else None}")
        debug_holder['print'] = False
    return frame_to_path, dict(dets_by_frame), ignore_by_frame, raw_keys_by_newcat, {}


def _seq_supervisely(seq, ann_files, img_dir, gt_dir, scheme, track_src, debug_holder):
    cls_map, cats = scheme['map'], scheme['categories']
    dets_by_frame = defaultdict(list)
    ignore_by_frame = defaultdict(list)
    wh_by_frame = {}
    frame_to_path = {}
    raw_keys_by_newcat = defaultdict(set)
    n_noid = 0
    for fr, af in sorted(ann_files):
        src_img = os.path.join(img_dir, os.path.basename(af)[:-5])
        if not os.path.isfile(src_img):
            cand = glob.glob(os.path.join(img_dir, os.path.splitext(os.path.basename(af)[:-5])[0] + '.*'))
            if cand:
                src_img = cand[0]
        frame_to_path[fr] = src_img
        try:
            ann = json.load(open(af))
        except Exception:
            continue
        wh_by_frame[fr] = (int(ann.get('size', {}).get('height', 0)),
                           int(ann.get('size', {}).get('width', 0)))
        for obj in ann.get('objects', []):
            title = str(obj.get('classTitle', '')).strip().lower()
            pts = (obj.get('points') or {}).get('exterior') or []
            if len(pts) < 2:
                continue
            bb = bbox_from_exterior(pts)
            if title in IGNORE_TITLES:
                ignore_by_frame[fr].append(list(bb)); continue
            uav_cat = CLASSTITLE_TO_UAVDT.get(title)
            new_cat = cls_map.get(uav_cat) if uav_cat else None
            if new_cat is None or bb[2] <= 0 or bb[3] <= 0:
                continue
            rk = resolve_track_key(obj, track_src)
            if rk is None:
                n_noid += 1; rk = ('_figid', obj.get('id'), fr)
            dets_by_frame[fr].append((new_cat, rk, bb))
            raw_keys_by_newcat[new_cat].add(rk)
    # merge official ignore (if a GT dir is also given in supervisely mode)
    for fr, regs in _load_gt_ignore(gt_dir, seq).items():
        ignore_by_frame[fr].extend(regs)
    if debug_holder.get('print'):
        print(f"\n[debug] seq={seq}  frames={len(frame_to_path)}  "
              f"ignore_frames={len(ignore_by_frame)}  no_persistent_id={n_noid}")
        debug_holder['print'] = False
    return frame_to_path, dict(dets_by_frame), dict(ignore_by_frame), raw_keys_by_newcat, wh_by_frame


# ── main conversion (shared backend) ──────────────────────────────────────────
def convert(uavdt_root, dst_root, split_name, scheme, source_format, workers,
            overwrite, link_mode, seq_prefix, track_src, gt_dir_override, debug,
            seqs_filter=None, gt_suffix_order=('_gt.txt', '_gt_whole.txt')):
    cats, _ = scheme['categories'], scheme['map']
    tid_mgr = TrackIDManager(len(cats))

    # locate inputs per format
    if source_format == 'uavdt':
        img_root = (_find_dir(uavdt_root, 'UAV-benchmark-M')
                    or _find_dir(uavdt_root, 'UAV-benchmark-M1') or uavdt_root)
        gt_dir = gt_dir_override or _find_dir(uavdt_root, 'GT')
        if gt_dir is None:
            raise FileNotFoundError(f"GT/ folder not found under {uavdt_root}")
        seq_list = sorted(d for d in os.listdir(img_root)
                          if os.path.isdir(os.path.join(img_root, d)))
        seq_to_anns = {s: None for s in seq_list}        # not used in uavdt mode
        print(f"[uavdt] images: {img_root}\n[uavdt] GT: {gt_dir}")
    else:  # supervisely
        split_root = os.path.join(uavdt_root, split_name)
        img_dir = os.path.join(split_root, 'img'); ann_dir = os.path.join(split_root, 'ann')
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"ann/ not found under {split_root}")
        seq_to_anns = defaultdict(list)
        for af in sorted(glob.glob(os.path.join(ann_dir, '*.json'))):
            s, fr = parse_name(af)
            if s is not None:
                seq_to_anns[s].append((fr, af))
        seq_list = sorted(seq_to_anns)
        gt_dir = gt_dir_override
        print(f"[supervisely] img: {img_dir}\n[supervisely] ann: {ann_dir}")

    # filter sequences (drop SOT S****; optional explicit list)
    if seqs_filter:
        want = {s.strip() for s in seqs_filter}
        present = set(seq_list)
        missing = sorted(want - present)
        if missing:
            print(f"  [WARN] {len(missing)} requested sequence(s) NOT found in the data: "
                  f"{missing} -- check your UAV-benchmark-M folder is complete.")
        seq_list = [s for s in seq_list if s in want]
        print(f"  [seqs] converting {len(seq_list)}/{len(want)} requested sequences.")
    if seq_prefix:
        before = list(seq_list)
        seq_list = [s for s in seq_list if s.upper().startswith(seq_prefix.upper())]
        skipped = sorted(set(before) - set(seq_list))
        if skipped:
            print(f"  [seq_prefix={seq_prefix!r}] keeping {len(seq_list)}, "
                  f"skipping {len(skipped)} (e.g. {skipped[:5]}) -- SOT/non-MOT clips.")
    if not seq_list:
        raise RuntimeError("No sequences to convert after filtering.")

    dst_img_root = os.path.join(dst_root, 'images')
    dst_ann_dir = os.path.join(dst_root, 'annotations')
    os.makedirs(dst_img_root, exist_ok=True); os.makedirs(dst_ann_dir, exist_ok=True)

    images_list, anns_list = [], []
    img_id = ann_id = n_box = n_ign_boxes = n_ign_frames = 0
    class_box_counts = {k: 0 for k in cats}
    debug_holder = {'print': bool(debug)}

    for seq in tqdm(seq_list, desc=f'[{split_name}]'):
        if source_format == 'uavdt':
            frame_to_path, dets_by_frame, ignore_by_frame, raw_keys, wh = _seq_official(
                seq, os.path.join(img_root, seq), gt_dir, scheme, debug_holder, gt_suffix_order)
        else:
            frame_to_path, dets_by_frame, ignore_by_frame, raw_keys, wh = _seq_supervisely(
                seq, seq_to_anns[seq], img_dir, gt_dir, scheme, track_src, debug_holder)
        if not frame_to_path:
            continue

        seq_map = tid_mgr.build_seq_map(raw_keys)
        dst_seq_dir = os.path.join(dst_img_root, seq); os.makedirs(dst_seq_dir, exist_ok=True)
        frame_ids = sorted(frame_to_path)

        def _write(fr):
            src_img = frame_to_path[fr]
            dst_img = os.path.join(dst_seq_dir, f'{fr:07d}.jpg')
            ign = ignore_by_frame.get(fr, [])
            if ign:
                im = cv2.imread(src_img)
                if im is None:
                    return fr, None
                for (x, y, w, h) in ign:
                    x, y = max(0, int(x)), max(0, int(y))
                    im[y:y + int(h), x:x + int(w)] = 0
                if os.path.islink(dst_img) or os.path.isfile(dst_img):
                    os.remove(dst_img)
                cv2.imwrite(dst_img, im)
                return fr, (im.shape[0], im.shape[1])
            if os.path.isfile(src_img):
                _place_image(src_img, dst_img, link_mode, overwrite)
            if fr in wh and all(wh[fr]):
                return fr, wh[fr]
            im = cv2.imread(dst_img)
            return fr, (im.shape[:2] if im is not None else None)

        sizes = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_write, fr) for fr in frame_ids]):
                fr, hw = fut.result()
                if hw is not None:
                    sizes[fr] = hw
        for fr in frame_ids:
            if ignore_by_frame.get(fr):
                n_ign_frames += 1; n_ign_boxes += len(ignore_by_frame[fr])

        for fr in frame_ids:
            if fr not in sizes:
                continue
            H, W = sizes[fr]
            images_list.append({'id': img_id, 'file_name': f'{seq}/{fr:07d}.jpg',
                                'height': H, 'width': W, 'seq_id': seq, 'frame_id': fr})
            for new_cat, rk, (x, y, w, h) in dets_by_frame.get(fr, []):
                anns_list.append({'id': ann_id, 'image_id': img_id, 'category_id': new_cat,
                                  'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0,
                                  'track_id': seq_map[(new_cat, rk)]})
                ann_id += 1; n_box += 1; class_box_counts[new_cat] += 1
            img_id += 1

    coco = {'images': images_list, 'annotations': anns_list,
            'categories': [{'id': k, 'name': v, 'supercategory': 'object'} for k, v in cats.items()]}
    out_json = os.path.join(dst_ann_dir, f'instances_{split_name}.json')
    json.dump(coco, open(out_json, 'w'), separators=(',', ':'))

    print(f"\n[{split_name}] {len(seq_list)} sequences  {len(images_list):,} images  {n_box:,} annotations")
    print('  Per-class boxes:', {cats[c]: n for c, n in class_box_counts.items()})
    print('  Unique track IDs:', {cats[c]: n for c, n in tid_mgr.totals().items()})
    if n_ign_boxes:
        print(f"  Ignore regions blacked out: {n_ign_boxes:,} boxes / {n_ign_frames:,} frames")
    print(f'  -> {out_json}')


def main():
    ap = argparse.ArgumentParser(description='UAV-DT -> COCO MOT (FalconMOT format).')
    ap.add_argument('--uavdt_root', required=True,
                    help='uavdt mode: dir containing UAV-benchmark-M/ and GT/. '
                         'supervisely mode: dir containing <split>/{img,ann}.')
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--source_format', choices=['auto', 'uavdt', 'supervisely'], default='auto')
    ap.add_argument('--split_name', default='test', help='Output split name (json/folder).')
    ap.add_argument('--splits', nargs='+', default=None,
                    help='supervisely mode only: input split folder names.')
    ap.add_argument('--class_scheme', choices=list(SCHEMES), default='vehicle1',
                    help="vehicle1 (single 'vehicle', standard UAVDT MOT, default), "
                         "visdrone5, or uavdt3.")
    ap.add_argument('--uavdt_split', choices=['test', 'all'], default=None,
                    help="Restrict to the official UAVDT-MOT split. 'test' = the 20 "
                         "official test sequences (hardcoded & verified); 'all' = no "
                         "restriction. Overridden by --attr_dir or --seqs if given.")
    ap.add_argument('--attr_dir', default=None,
                    help="Path to M_attr/test (or train). The <SEQ>_attr.txt filenames "
                         "there are read as the AUTHORITATIVE sequence list to convert. "
                         "Recommended: point at your M_attr/test to guarantee the exact "
                         "official test split.")
    ap.add_argument('--seqs', default=None,
                    help='Optional comma-separated list to restrict sequences '
                         '(e.g. M0203,M0205,...). Use the 20 official MOT test seqs.')
    ap.add_argument('--seq_prefix', default='M',
                    help="Keep only sequences with this prefix (UAVDT MOT='M', SOT='S'). "
                         "Pass '' to keep all.")
    ap.add_argument('--gt_suffix', default='_gt.txt',
                    help="uavdt mode GT file to prefer (_gt.txt MOT, or _gt_whole.txt DET).")
    ap.add_argument('--gt_ignore_dir', default=None,
                    help='Override path to the GT folder with <seq>_gt_ignore.txt. In uavdt '
                         'mode this is auto-found under --uavdt_root/GT.')
    ap.add_argument('--track_id_source', default='auto',
                    help='supervisely mode: auto|objectKey|key|id|tag:<name>.')
    ap.add_argument('--link', choices=['copy', 'symlink', 'hardlink'], default='copy',
                    help='copy (default), symlink (saves disk on Kaggle), or hardlink. '
                         'Frames with ignore regions are always written out (blacked).')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    scheme = SCHEMES[args.class_scheme]

    # Resolve which sequences to convert (priority: --seqs > --attr_dir > --uavdt_split).
    seqs_filter = None
    if args.seqs:
        seqs_filter = [s.strip() for s in args.seqs.split(',') if s.strip()]
        print(f"[seqs] explicit list ({len(seqs_filter)}): {seqs_filter}")
    elif args.attr_dir:
        seqs_filter = _read_attr_seqs(args.attr_dir)
        if not seqs_filter:
            raise FileNotFoundError(f"No *_attr.txt found in --attr_dir {args.attr_dir}")
        print(f"[seqs] from M_attr ({len(seqs_filter)} seqs in {args.attr_dir}): {seqs_filter}")
    elif args.uavdt_split == 'test':
        seqs_filter = list(UAVDT_TEST_SEQS)
        print(f"[seqs] official UAVDT-MOT test split ({len(seqs_filter)}): {seqs_filter}")
    gt_suffix_order = (args.gt_suffix, '_gt_whole.txt', '_gt.txt')
    # dedup preserve order
    seen = set(); gt_suffix_order = tuple(s for s in gt_suffix_order if not (s in seen or seen.add(s)))

    fmt = args.source_format
    if fmt == 'auto':
        if _find_dir(args.uavdt_root, 'GT') and (_find_dir(args.uavdt_root, 'UAV-benchmark-M')
                                                 or glob.glob(os.path.join(args.uavdt_root, 'M[0-9]*'))):
            fmt = 'uavdt'
        else:
            fmt = 'supervisely'
    print(f"[source_format={fmt}] [class_scheme={args.class_scheme}] "
          f"categories={scheme['categories']}")

    if fmt == 'uavdt':
        convert(args.uavdt_root, os.path.join(args.output_root, args.split_name),
                args.split_name, scheme, 'uavdt', args.workers, args.overwrite, args.link,
                args.seq_prefix, args.track_id_source, args.gt_ignore_dir, args.debug,
                seqs_filter, gt_suffix_order)
    else:
        for split in (args.splits or ['test']):
            convert(args.uavdt_root, os.path.join(args.output_root, split), split, scheme,
                    'supervisely', args.workers, args.overwrite, args.link, args.seq_prefix,
                    args.track_id_source, args.gt_ignore_dir, args.debug, seqs_filter, gt_suffix_order)


if __name__ == '__main__':
    main()