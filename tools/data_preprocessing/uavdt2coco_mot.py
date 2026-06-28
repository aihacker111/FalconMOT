"""
uavdt2coco_mot.py
Convert UAV-DT in the **Supervisely image format** (e.g. the Kaggle / Datasets-Ninja
export: flat  <split>/{img,ann,meta}  with per-image JSONs) into COCO JSON in the
SAME format FalconMOT's VisDrone 5-class benchmark uses, so the tracker +
tools/eval_mot_5cls.py + hota.py work unchanged.

SOURCE LAYOUT (what this expects)
    <root>/<split>/img/<SEQ>_img<NNNNNN>.jpg            (flat; seq encoded in name)
    <root>/<split>/ann/<SEQ>_img<NNNNNN>.jpg.json       (one Supervisely JSON per image)
    <root>/<split>/meta/...                             (optional; ignored)
  Each ann JSON:
    {"size": {"height": H, "width": W},
     "objects": [ {"classTitle": "car"|"truck"|"bus",
                   "objectKey": "<uuid persistent across frames>",   # <- track identity
                   "tags": [ {"name": ..., "value": ...}, ... ],
                   "points": {"exterior": [[x1,y1],[x2,y2]], "interior": []}}, ... ]}

OUTPUT (identical schema to visdrone2coco_5cls_benchmark_mot.py)
    <out>/<split>/images/<SEQ>/<frame:07d>.jpg
    <out>/<split>/annotations/instances_<split>.json
        images[]:      {id, file_name='<SEQ>/<frame:07d>.jpg', height, width, seq_id, frame_id}
        annotations[]: {id, image_id, category_id, bbox=[x,y,w,h], area, iscrowd, track_id}
        categories[]:  per --class_scheme

TRACK-ID SOURCE (critical for MOT). Resolved per object, priority (auto):
    1) an object tag whose name looks like a track id  -> use its integer value
    2) objectKey  (Supervisely's persistent per-object UUID)   <- usual case
    3) key
  The per-figure field `id` is NOT used as identity (it changes every frame).
  Use --debug to dump the first object so you can confirm where the id lives, and
  --track_id_source to force it (e.g. --track_id_source objectKey or tag:track_id).
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


# ── UAV-DT class titles -> canonical uavdt category id (1=car,2=truck,3=bus) ──
CLASSTITLE_TO_UAVDT = {'car': 1, 'truck': 2, 'bus': 3}
UAVDT_CAT_NAMES = {1: 'car', 2: 'truck', 3: 'bus'}
# classTitles treated as ignore regions -> blacked out, not annotated
IGNORE_TITLES = {'ignore', 'ignored', 'ignored region', 'ignored_region', 'region', 'others'}
# tag names that may carry the integer track id
TRACKID_TAG_NAMES = {'trackid', 'track', 'objectid', 'targetid', 'gtid', 'tid', 'instanceid', 'id'}

# filename like  M0203_img000001.jpg.json  ->  seq=M0203  frame=1
_NAME_RE = re.compile(r'^(?P<seq>.+?)_img0*(?P<frame>\d+)\.(?:jpg|jpeg|png)(?:\.json)?$',
                      re.IGNORECASE)


# ── Target class schemes (choose with --class_scheme) ─────────────────────────
SCHEMES = {
    'visdrone5': {'categories': {1: 'pedestrian', 2: 'car', 3: 'van', 4: 'truck', 5: 'bus'},
                  'map': {1: 2, 2: 4, 3: 5}},        # car->2, truck->4, bus->5
    'vehicle1':  {'categories': {1: 'vehicle'}, 'map': {1: 1, 2: 1, 3: 1}},
    'uavdt3':    {'categories': {1: 'car', 2: 'truck', 3: 'bus'}, 'map': {1: 1, 2: 2, 3: 3}},
}


class TrackIDManager:
    """Globally-unique, per-target-class track ids (mirrors the VisDrone converter)."""
    def __init__(self, num_target_classes):
        self._start = [0] * num_target_classes

    def build_seq_map(self, raw_keys_by_newcat):
        """raw_keys_by_newcat: {new_cat -> set(raw_key)} -> {(new_cat, raw_key): global_id}."""
        seq_map = {}
        for new_cat, keys in raw_keys_by_newcat.items():
            idx = new_cat - 1
            for rank, key in enumerate(sorted(keys, key=lambda k: str(k))):
                seq_map[(new_cat, key)] = rank + self._start[idx]
            self._start[idx] += len(keys)
        return seq_map

    def totals(self):
        return {i + 1: n for i, n in enumerate(self._start) if n > 0}


def parse_name(fname):
    m = _NAME_RE.match(os.path.basename(fname))
    if not m:
        return None, None
    return m.group('seq'), int(m.group('frame'))


def resolve_track_key(obj, source='auto'):
    """Return a hashable key that is STABLE for the same object across frames."""
    tags = obj.get('tags') or []
    if source.startswith('tag:'):
        want = source[4:].lower()
        for t in tags:
            if str(t.get('name', '')).lower() == want:
                return t.get('value')
        return None
    if source in ('objectKey', 'key', 'id'):
        return obj.get(source)
    # auto
    for t in tags:                                   # 1) id-like tag value
        nm = str(t.get('name', '')).lower().replace(' ', '').replace('_', '')
        if nm in TRACKID_TAG_NAMES and t.get('value') is not None:
            return t.get('value')
    for k in ('objectKey', 'key'):                   # 2/3) persistent uuid
        if obj.get(k) is not None:
            return obj.get(k)
    return None


def bbox_from_exterior(ext):
    (x1, y1), (x2, y2) = ext[0], ext[1]
    x, y = min(x1, x2), min(y1, y2)
    return float(x), float(y), float(abs(x2 - x1)), float(abs(y2 - y1))


def convert_split(split_root, dst_root, split, scheme, workers, overwrite,
                  track_src, debug):
    img_dir = os.path.join(split_root, 'img')
    ann_dir = os.path.join(split_root, 'ann')
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f'ann/ not found under {split_root}')

    cats, cls_map = scheme['categories'], scheme['map']
    tid_mgr = TrackIDManager(len(cats))

    # group annotation files by sequence
    ann_files = sorted(glob.glob(os.path.join(ann_dir, '*.json')))
    by_seq = defaultdict(list)
    for af in ann_files:
        seq, fr = parse_name(af)
        if seq is not None:
            by_seq[seq].append((fr, af))
    if not by_seq:
        raise RuntimeError(f'No <SEQ>_img<N>.jpg.json files parsed in {ann_dir}')

    images_list, anns_list = [], []
    img_id = ann_id = 0
    n_box = n_noid = 0
    class_box_counts = {k: 0 for k in cats}

    dst_img_root = os.path.join(dst_root, 'images')
    dst_ann_dir = os.path.join(dst_root, 'annotations')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    printed = False
    for seq in tqdm(sorted(by_seq), desc=f'[{split}]'):
        frames = sorted(by_seq[seq])
        # PASS 1: read all jsons, collect raw track keys per new-cat for stable global ids
        parsed = {}                       # frame -> (H, W, [ (new_cat, raw_key, bbox) ], ignore_boxes)
        raw_keys_by_newcat = defaultdict(set)
        for fr, af in frames:
            try:
                with open(af) as f:
                    ann = json.load(f)
            except Exception:
                continue
            H = int(ann.get('size', {}).get('height', 0))
            W = int(ann.get('size', {}).get('width', 0))
            dets, ignores = [], []
            for obj in ann.get('objects', []):
                title = str(obj.get('classTitle', '')).strip().lower()
                pts = (obj.get('points') or {}).get('exterior') or []
                if len(pts) < 2:
                    continue
                bb = bbox_from_exterior(pts)
                if title in IGNORE_TITLES:
                    ignores.append(bb)
                    continue
                uav_cat = CLASSTITLE_TO_UAVDT.get(title)
                if uav_cat is None or uav_cat not in cls_map or bb[2] <= 0 or bb[3] <= 0:
                    continue
                new_cat = cls_map[uav_cat]
                raw_key = resolve_track_key(obj, track_src)
                if raw_key is None:
                    n_noid += 1
                    raw_key = ('_figid', obj.get('id'), fr)   # last-resort, non-persistent
                raw_keys_by_newcat[new_cat].add(raw_key)
                dets.append((new_cat, raw_key, bb))
            parsed[fr] = (H, W, dets, ignores)

            if debug and not printed:
                ex = next((o for o in ann.get('objects', []) if o.get('points')), None)
                print(f'\n[debug] seq={seq} frame={fr}  file={os.path.basename(af)}')
                print('[debug] object[0] keys:', sorted(ex.keys()) if ex else None)
                print('[debug] object[0]:', json.dumps(ex)[:600] if ex else None)
                print('[debug] resolved track key ->', resolve_track_key(ex, track_src) if ex else None)
                printed = True

        seq_map = tid_mgr.build_seq_map(raw_keys_by_newcat)

        # PASS 2: write images (copy; blackout only if the frame has ignore regions)
        dst_seq_dir = os.path.join(dst_img_root, seq)
        os.makedirs(dst_seq_dir, exist_ok=True)

        def _write(fr_af):
            fr, af = fr_af
            src_img = os.path.join(img_dir, os.path.basename(af)[:-5])   # strip '.json'
            if not os.path.isfile(src_img):
                # try common extensions
                base = os.path.splitext(os.path.basename(af)[:-5])[0]
                cand = glob.glob(os.path.join(img_dir, base + '.*'))
                src_img = cand[0] if cand else src_img
            dst_img = os.path.join(dst_seq_dir, f'{fr:07d}.jpg')
            H, W, dets, ignores = parsed.get(fr, (0, 0, [], []))
            if ignores:
                im = cv2.imread(src_img)
                if im is None:
                    return fr, None
                for (x, y, w, h) in ignores:
                    x, y = max(0, int(x)), max(0, int(y))
                    im[y:y + int(h), x:x + int(w)] = 0
                if overwrite or not os.path.isfile(dst_img):
                    cv2.imwrite(dst_img, im)
                return fr, (im.shape[0], im.shape[1])
            else:
                if overwrite or not os.path.isfile(dst_img):
                    if os.path.isfile(src_img):
                        shutil.copy2(src_img, dst_img)
                if H and W:
                    return fr, (H, W)
                im = cv2.imread(dst_img)
                return fr, (im.shape[:2] if im is not None else None)

        sizes = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_write, fa): fa[0] for fa in frames}
            for fut in as_completed(futs):
                fr, hw = fut.result()
                if hw is not None:
                    sizes[fr] = hw

        for fr, _af in frames:
            if fr not in sizes:
                continue
            H, W = sizes[fr]
            images_list.append({
                'id': img_id, 'file_name': f'{seq}/{fr:07d}.jpg',
                'height': H, 'width': W, 'seq_id': seq, 'frame_id': fr,
            })
            for new_cat, raw_key, (x, y, w, h) in parsed[fr][2]:
                gtid = seq_map[(new_cat, raw_key)]
                anns_list.append({
                    'id': ann_id, 'image_id': img_id, 'category_id': new_cat,
                    'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0, 'track_id': gtid,
                })
                ann_id += 1
                n_box += 1
                class_box_counts[new_cat] += 1
            img_id += 1

    coco = {'images': images_list, 'annotations': anns_list,
            'categories': [{'id': k, 'name': v, 'supercategory': 'object'} for k, v in cats.items()]}
    out_json = os.path.join(dst_ann_dir, f'instances_{split}.json')
    with open(out_json, 'w') as f:
        json.dump(coco, f, separators=(',', ':'))

    print(f'\n[{split}] {len(by_seq)} sequences  {len(images_list):,} images  {n_box:,} annotations')
    if n_noid:
        print(f'  [WARN] {n_noid:,} objects had no persistent track key (tag/objectKey/key) -> '
              f'check --debug and set --track_id_source; identities may be unreliable.')
    print('  Per-class box counts:')
    for cid in sorted(class_box_counts):
        print(f'    [{cid}] {cats[cid]:<12s}: {class_box_counts[cid]:,}')
    print('  Unique track IDs per class:')
    for cid, n in tid_mgr.totals().items():
        print(f'    [{cid}] {cats[cid]:<12s}: {n:,}')
    print(f'  -> {out_json}')


def main():
    ap = argparse.ArgumentParser(description='UAV-DT (Supervisely format) -> COCO MOT (FalconMOT).')
    ap.add_argument('--uavdt_root', required=True,
                    help='Parent dir that CONTAINS the split folder, so '
                         '<uavdt_root>/<split>/{img,ann} exists.')
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--class_scheme', choices=list(SCHEMES), default='vehicle1',
                    help="vehicle1 (single 'vehicle' class -- standard UAVDT MOT protocol, default), "
                         "visdrone5 (car->2,truck->4,bus->5; drop-in for eval_mot_5cls), "
                         "uavdt3 (native car/truck/bus).")
    ap.add_argument('--track_id_source', default='auto',
                    help="auto | objectKey | key | id | tag:<tagname>  (default auto).")
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--debug', action='store_true',
                    help='Dump the first object so you can confirm the track-id field.')
    args = ap.parse_args()

    scheme = SCHEMES[args.class_scheme]
    print(f'[class_scheme={args.class_scheme}] categories={scheme["categories"]} '
          f'| track_id_source={args.track_id_source}')
    for split in args.splits:
        split_root = os.path.join(args.uavdt_root, split)
        if not os.path.isdir(split_root):
            print(f'[Error] not found: {split_root}')
            continue
        convert_split(split_root, os.path.join(args.output_root, split), split,
                      scheme, args.workers, args.overwrite, args.track_id_source, args.debug)


if __name__ == '__main__':
    main()