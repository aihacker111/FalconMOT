# import copy
# import os
# import json
# import collections
# from concurrent.futures import ThreadPoolExecutor

# import cv2
# import numpy as np

# # Progress bar — optional dependency. Falls back to a no-op wrapper if tqdm
# # is not installed so eval still runs (just without the live bar).
# try:
#     from tqdm import tqdm
# except Exception:  # pragma: no cover
#     def tqdm(iterable=None, **kwargs):
#         return iterable if iterable is not None else []
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# import torch
# import torch.utils.data
# from torchvision.transforms import transforms as T

# import _paths  # noqa: F401  (sys.path bootstrap)
# from falconmot.cfg.args import opts
# from falconmot.nn import create_model, load_model, save_model
# from falconmot.nn.parallel import DataParallel
# from falconmot.utils.log import Logger
# from falconmot.data.factory import get_dataset
# from falconmot.engine.factory import train_factory
# from falconmot.engine import stage as stage_mgr
# from falconmot.optim import build_optimizer, build_scheduler
# from falconmot.utils.eval import CocoJsonEvaluator
# from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor
# from collections import defaultdict as _defaultdict
# from falconmot.tracker.multitracker import MCJDETracker, MCTrack
# from falconmot.tracker.utils.coco_gt_reader import load_all_coco_gt
# from falconmot.tracker.utils.hota import HOTACollector, box_iou_matrix
# from falconmot.data.dataset import LoadCocoSequencesForTracking, preprocess_for_tracking


# def _with_lr(opt, lr):
#     """Shallow copy of opt with a different base lr (for per-phase optimizer)."""
#     o = copy.copy(opt)
#     o.lr = lr
#     return o


# def _print_stage1_banner(opt):
#     if not getattr(opt, 'train_single_det', False):
#         return
#     w, h = opt.input_wh[0], opt.input_wh[1]
#     es = getattr(opt, 'eval_spatial_size', [h, w])
#     print('=' * 72)
#     print('Stage-1 detection-only training  (--train_single_det)')
#     print(f'  input {w}x{h}  eval_spatial_size={es}  use_s4={getattr(opt, "use_s4", False)} '
#           f'use_s4_aux={getattr(opt, "use_s4_aux", True)}')
#     print(f'  losses: cls + bbox + giou'
#           f"{' + s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''}  |  "
#           f'ReID: OFF')
#     print(f'  aug: mosaic={getattr(opt, "mosaic", False)} '
#           f'(p={getattr(opt, "mosaic_prob", 0.5)})  '
#           f'temporal_mosaic=OFF')
#     if getattr(opt, 'deim_pretrained', ''):
#         print(f'  deim_pretrained: {opt.deim_pretrained}')
#     print('  stage-2: load checkpoint WITHOUT --train_single_det')
#     print('=' * 72)


# def _summarize_model(model, opt):
#     core = model.module if hasattr(model, 'module') else model
#     print(f'[model] use_s4={getattr(core, "use_s4", False)}  '
#           f'use_s4_aux={getattr(core, "use_s4_aux", False)}  '
#           f'use_reid={getattr(core, "use_reid", False)}  '
#           f's4_branch={hasattr(core, "s4_branch")}  '
#           f'reid_head={hasattr(core, "reid_head")}')
#     if getattr(opt, 'train_single_det', False) and getattr(opt, 'use_s4', False) \
#             and getattr(opt, 'deim_pretrained', ''):
#         print('[model] NOTE: s4_branch/s4_aux_head not in DEIM pretrained — '
#               'trained from scratch in stage-1.')


# @torch.no_grad()
# def run_coco_eval(model, val_loader, opt, ann_file: str = '') -> dict:
#     """COCO mAP evaluation on the val split."""
#     model.eval()

#     postprocessor = FalconJDEPostProcessor(
#         num_classes=opt.num_classes,
#         num_top_queries=500,
#         use_focal_loss=True,
#         conf_thres=0.0
#     )

#     evaluator = CocoJsonEvaluator(ann_file) if ann_file else JDECocoEvaluator(
#         num_classes=opt.num_classes,
#         net_h=opt.input_wh[1],
#         net_w=opt.input_wh[0],
#     )

#     max_batches = getattr(opt, 'debug_val_batches', 0)

#     for i, batch in enumerate(val_loader):
#         if max_batches > 0 and i >= max_batches:
#             break
#         batch = {k: v.to(opt.device, non_blocking=True)
#                  if isinstance(v, torch.Tensor) else v
#                  for k, v in batch.items()}

#         orig_hw    = batch.get('orig_hw')
#         orig_sizes = orig_hw if orig_hw is not None else \
#                      torch.tensor([[opt.input_wh[1], opt.input_wh[0]]] * batch['input'].shape[0],
#                                   device=opt.device)

#         outputs    = model(batch['input'])
#         dt_results = postprocessor(outputs, orig_sizes)
#         evaluator.update(dt_results, batch)

#     model.train()
#     return evaluator.summarize()


# class _ParallelFrameReader:
#     """Read + preprocess a sequence's frames in worker threads, yield IN ORDER.

#     The tracker is stateful (Kalman + ReID memory) so frames must reach it in
#     order — but the *loading* of frame N+1..N+k (disk read + resize + normalize)
#     can overlap with the GPU forward of frame N. cv2.imread and the numpy
#     resize/normalize release the GIL, so threads give real parallelism here and
#     keep the GPU fed instead of idling on disk I/O.

#     Emits the same (frame_id, img_chw, img0_bgr) tuples as _CocoSeqIterator.
#     """

#     def __init__(self, frames, width, height, num_workers=4, prefetch=8):
#         self.frames      = frames
#         self.width       = width
#         self.height      = height
#         self.num_workers = max(1, int(num_workers))
#         # Keep at least num_workers reads in flight; deeper window hides latency.
#         self.prefetch    = max(self.num_workers, int(prefetch))

#     @staticmethod
#     def _load(args):
#         frame_id, path, w, h = args
#         img0 = cv2.imread(path)
#         img  = preprocess_for_tracking(img0, w, h)
#         return frame_id, img, img0

#     def __iter__(self):
#         tasks = ((fid, p, self.width, self.height) for fid, p in self.frames)
#         with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
#             inflight = collections.deque()
#             # Prime the sliding window.
#             for _ in range(self.prefetch):
#                 try:
#                     inflight.append(ex.submit(self._load, next(tasks)))
#                 except StopIteration:
#                     break
#             # Yield oldest-first while topping the window back up.
#             while inflight:
#                 fut = inflight.popleft()
#                 try:
#                     inflight.append(ex.submit(self._load, next(tasks)))
#                 except StopIteration:
#                     pass
#                 yield fut.result()


# def _collect_hota_frame(coll, class_names, seq_id, gt_objs, pred_tlwhs, pred_tids, offset):
#     """Feed one frame's GT + tracker output into the HOTACollector, split per class.

#     The class is baked into the (global) id as ``id // offset`` on both sides, so
#     GT global_id and tracker global_id (track_id + c*offset) split identically.
#     Frames that are empty for a class (no GT, no pred) contribute nothing to HOTA
#     and are skipped — this shrinks the per-(class,seq) timestep list and speeds up
#     the final TrackEval pass without changing the score.
#     """
#     ncls = len(class_names)
#     gt_b = [[] for _ in range(ncls)]
#     gt_i = [[] for _ in range(ncls)]
#     for tlwh, gid, _vis in gt_objs:
#         c = int(gid) // offset
#         if 0 <= c < ncls:
#             gt_b[c].append(tlwh)
#             gt_i[c].append(int(gid))

#     pr_b = [[] for _ in range(ncls)]
#     pr_i = [[] for _ in range(ncls)]
#     for tlwh, tid in zip(pred_tlwhs, pred_tids):
#         c = int(tid) // offset
#         if 0 <= c < ncls:
#             pr_b[c].append(tlwh)
#             pr_i[c].append(int(tid))

#     for c in range(ncls):
#         if not gt_i[c] and not pr_i[c]:
#             continue
#         g = np.asarray(gt_b[c], dtype=np.float64).reshape(-1, 4)
#         p = np.asarray(pr_b[c], dtype=np.float64).reshape(-1, 4)
#         sim = box_iou_matrix(g, p)
#         coll.add_frame(class_names[c], seq_id, gt_i[c], pr_i[c], sim)


# @torch.inference_mode()
# def run_track_eval(model, opt, val_ann_file: str, val_img_root: str) -> dict:
#     """Tracking validation over val sequences — HOTA metrics only.

#     Uses the native class space without any class remap. GMC is disabled for
#     speed and determinism (set_image is never called). Returns a dict with the
#     HOTA-family scalars from hota.py (hota/deta/assa/loca/detre/assre) plus
#     track_score (= HOTA, drives model_best) and fps.

#     Speed: frame loading is pipelined across threads (overlaps disk I/O with
#     GPU inference), the detector forward runs in fp16 autocast when
#     --track_val_fp16 is set, and HOTA is computed in-memory (no result .txt
#     round-trip) with a parallel TrackEval pass. A tqdm bar shows live progress.
#     """
#     import time

#     if not (val_ann_file and val_img_root and os.path.isfile(val_ann_file)):
#         print('[track-eval] missing val_ann/val_img -- skipping tracking eval.')
#         return {}

#     model.eval()
#     net_w, net_h = opt.img_size
#     ncls     = opt.num_classes
#     min_area = getattr(opt, 'min_box_area', 100)
#     _OFF     = 1_000_000

#     postproc = FalconJDEPostProcessor(
#         num_classes=ncls,
#         num_top_queries=getattr(opt, 'K', 300),
#         conf_thres=opt.conf_thres,
#         use_focal_loss=True,
#     )

#     opt_trk = copy.copy(opt)
#     opt_trk.num_classes = ncls
#     tracker = MCJDETracker(opt_trk, frame_rate=getattr(opt, 'frame_rate', 30))

#     src = LoadCocoSequencesForTracking(val_ann_file, val_img_root, img_size=opt.img_size)

#     # ── Class names for HOTA (native class space, no remap) ─────────────────
#     # HOTA is computed per class then class-averaged (VisDrone convention).
#     try:
#         from falconmot.tracker.class_remap import CLS7_NAMES, CLS5_NAMES, CLS4_NAMES
#         _name_tables = {7: CLS7_NAMES, 5: CLS5_NAMES, 4: CLS4_NAMES}
#     except Exception:
#         _name_tables = {}
#     _names_map = _name_tables.get(ncls, {})
#     class_names = {c: _names_map.get(c, f'class_{c}') for c in range(ncls)}
#     hota_names  = [class_names[c] for c in range(ncls)]
#     coll = HOTACollector(hota_names)

#     # ── Parse GT ONCE (instead of re-parsing the whole json per sequence) ───
#     gt_by_seq = load_all_coco_gt(val_ann_file)

#     # ── fp16 autocast for the detector forward (wires up --track_val_fp16) ──
#     use_fp16 = bool(int(getattr(opt, 'track_val_fp16', 1))) and opt.device.type == 'cuda'

#     # ── Threaded frame prefetch params (overlap disk I/O with GPU) ──────────
#     n_threads  = int(getattr(opt, 'track_val_loader_threads', 4))
#     prefetch   = int(getattr(opt, 'track_val_prefetch', 8))

#     t0, n_frames = time.time(), 0

#     total_frames = sum(src.num_frames(s) for s in src.seqs)
#     pbar = tqdm(total=total_frames, desc='[track-eval]', unit='f', dynamic_ncols=True)

#     for seq_id in src.seqs:
#         tracker.reset()
#         gt_frames = gt_by_seq.get(seq_id, {})

#         frames = src._seq_frames[seq_id]
#         reader = _ParallelFrameReader(frames, net_w, net_h,
#                                       num_workers=n_threads, prefetch=prefetch)

#         for frame_id, img, img0 in reader:
#             orig_h, orig_w = img0.shape[:2]
#             sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
#             blob  = torch.from_numpy(img[None]).to(opt.device, non_blocking=True)

#             with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
#                 output = model(blob)
#                 res = postproc(output, sizes)[0]

#             dets = _defaultdict(list)
#             if len(res['scores']) > 0:
#                 bxs = res['boxes'].float().cpu().numpy()
#                 scs = res['scores'].float().cpu().numpy()
#                 lbs = res['labels'].cpu().numpy()
#                 rid = res['reid'].float().cpu().numpy() if 'reid' in res else None
#                 ws  = bxs[:, 2] - bxs[:, 0]
#                 hs  = bxs[:, 3] - bxs[:, 1]
#                 for i in np.where((ws > 0) & (hs > 0))[0]:
#                     c = int(lbs[i])
#                     if c < 0 or c >= ncls:
#                         continue
#                     tlwh = np.array([bxs[i, 0], bxs[i, 1], ws[i], hs[i]], dtype=np.float32)
#                     emb  = rid[i] if rid is not None else np.zeros(1, dtype=np.float32)
#                     dets[c].append(MCTrack(tlwh, float(scs[i]), emb, ncls, c))

#             online = tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

#             tlwhs, tids = [], []
#             for c, tracks in online.items():
#                 for t in tracks:
#                     w, h = t.curr_tlwh[2], t.curr_tlwh[3]
#                     if t.track_id < 0 or (w * h) <= min_area:
#                         continue
#                     tlwhs.append(t.curr_tlwh)
#                     tids.append(int(t.track_id) + c * _OFF)

#             _collect_hota_frame(coll, class_names, seq_id,
#                                 gt_frames.get(int(frame_id), []),
#                                 tlwhs, tids, _OFF)
#             n_frames += 1
#             pbar.update(1)
#             if n_frames % 20 == 0:
#                 pbar.set_postfix_str(f'{n_frames / max(1e-6, time.time() - t0):.1f} fps')

#     pbar.close()
#     infer_elapsed = time.time() - t0   # pure inference time (excludes HOTA compute)
#     model.train()

#     if n_frames == 0:
#         return {}

#     # ── HOTA (the only metric — drives model_best selection) ────────────────
#     hota_workers = int(getattr(opt, 'track_val_hota_workers', 0)) or min(8, os.cpu_count() or 4)
#     try:
#         hres = coll.compute(num_workers=hota_workers)
#     except Exception as e:
#         print(f'[track-eval] HOTA unavailable ({e}); install TrackEval to enable it. '
#               f'Skipping best-model update this round.')
#         return {'fps': n_frames / max(1e-6, infer_elapsed)}

#     ov  = hres['overall']
#     fps = n_frames / max(1e-6, infer_elapsed)
#     out = {k.lower(): float(ov.get(k, 0.0))
#            for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe')}
#     out['track_score'] = out['hota']     # best model selected by HOTA
#     out['fps'] = fps
#     return out


# def run(opt):
#     _print_stage1_banner(opt)
#     torch.manual_seed(opt.seed)
#     torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

#     print('Setting up data...')
#     Dataset      = get_dataset(opt.dataset, opt.task)
#     use_coco_fmt = (opt.dataset == 'coco')

#     with open(opt.data_cfg) as f:
#         data_config  = json.load(f)
#     dataset_root = data_config['root']
#     print("Dataset root: %s" % dataset_root)

#     from falconmot.data.dataset import VisDroneCocoDataset

#     if use_coco_fmt:
#         train_sources = data_config.get('train_sources')
#         if not train_sources and getattr(opt, 'merge_val_into_train', False):
#             train_sources = [
#                 {'ann': data_config['train_ann'], 'img': data_config['train_img']},
#                 {'ann': data_config['val_ann'],   'img': data_config['val_img']},
#             ]
#             print('[data] merge_val_into_train: merging train + val into the train set')

#         if train_sources:
#             dataset = VisDroneCocoDataset(opt=opt, sources=train_sources, augment=True)
#         else:
#             dataset = VisDroneCocoDataset(
#                 opt=opt, img_root=data_config['train_img'],
#                 ann_file=data_config['train_ann'], augment=True)
#     else:
#         dataset = Dataset(opt=opt, root=dataset_root,
#                           paths=data_config['train'], img_size=opt.input_wh,
#                           augment=True, transforms=T.Compose([T.ToTensor()]))
#     opt = opts().init()
#     opt = opts().update_dataset_info_and_set_heads(opt, dataset)
#     print("opt:\n", opt)
#     logger = Logger(opt)

#     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     print("opt.gpus_str: ", opt.gpus_str)
#     opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

#     # ── Val dataset (optional) ──────────────────────────────────────────────
#     val_loader   = None
#     val_ann_file = ''
#     val_img_root = ''

#     if getattr(opt, 'merge_val_into_train', False):
#         print('[data] merge_val_into_train=True -> skipping val_loader, no COCO eval.')
#     elif getattr(opt, 'val_cfg', ''):
#         with open(opt.val_cfg) as f:
#             val_config = json.load(f)
#         val_dataset = None

#         if use_coco_fmt:
#             val_ann_file = val_config.get('val_ann', '')
#             val_img      = val_config.get('val_img', '')
#             val_img_root = val_img
#             if val_ann_file and val_img:
#                 val_dataset = VisDroneCocoDataset(
#                     opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
#             else:
#                 print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
#         else:
#             val_root  = val_config.get('root', dataset_root)
#             val_paths = val_config.get('val') or val_config.get('test') or []
#             if val_paths:
#                 val_dataset = Dataset(
#                     opt=opt, root=val_root, paths=val_paths,
#                     img_size=opt.input_wh, augment=False,
#                     transforms=T.Compose([T.ToTensor()]))
#             else:
#                 print('[warn] val_cfg provided but no val/test paths found.')

#         if val_dataset is not None:
#             val_loader = torch.utils.data.DataLoader(
#                 dataset=val_dataset,
#                 batch_size=opt.batch_size,
#                 shuffle=False,
#                 num_workers=opt.num_workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 persistent_workers=opt.num_workers > 0,
#                 prefetch_factor=2 if opt.num_workers > 0 else None,
#             )
#             print(f'Val dataset: {len(val_dataset)} images')

#     print('Creating model...')
#     model = create_model(opt.arch, opt)
#     _summarize_model(model, opt)

#     # ── Load checkpoint BEFORE applying freeze policy ───────────────────────
#     start_epoch = 0
#     if opt.load_model != '':
#         loaded = load_model(model, opt.load_model, None, opt.resume, opt.lr, opt.lr_step)
#         model = loaded[0] if isinstance(loaded, tuple) else loaded
#         if opt.resume:
#             try:
#                 ckpt = torch.load(opt.load_model, map_location='cpu', weights_only=False)
#                 start_epoch = int(ckpt.get('epoch', 0))
#             except Exception:
#                 start_epoch = 0

#     # # ── Training stage policy ───────────────────────────────────────────────
#     # det_only  = getattr(opt, 'train_single_det', False)
#     # warmup_ep = max(0, getattr(opt, 'reid_warmup_epochs', 0))
#     # p0_lr     = opt.lr if getattr(opt, 'reid_warmup_lr', -1) <= 0 else opt.reid_warmup_lr

#     # in_phase1 = det_only or start_epoch >= warmup_ep
#     # if det_only:
#     #     stage_mgr.apply_det_only(model)
#     #     init_lr = opt.lr
#     # elif in_phase1:
#     #     stage_mgr.apply_phase1(model, keep_backbone_frozen=False, freeze_norm=False)
#     #     init_lr = opt.lr
#     # else:
#     #     stage_mgr.apply_phase0(model)
#     #     init_lr = p0_lr
#     # ========================================================================
#     # ── Training stage policy (Stage 1 vs Stage 2) ──────────────────────────
#     # ========================================================================
#     det_only  = getattr(opt, 'train_single_det', False)
#     reid_only = getattr(opt, 'train_reid_only', False)

#     if det_only and reid_only:
#         raise ValueError("Cannot set both --train_single_det and --train_reid_only!")

#     if det_only:
#         # QUÁ TRÌNH 1
#         stage_mgr.apply_det_only(model)
#     elif reid_only:
#         # QUÁ TRÌNH 2: Gọi hàm khóa detection
#         stage_mgr.apply_reid_only(model)
#     else:
#         # Fallback (Phòng trường hợp bạn muốn train chung cả 2 cùng lúc)
#         stage_mgr.apply_joint_training(model)

#     optimizer = build_optimizer(model, _with_lr(opt, opt.lr))

#     _nw = opt.num_workers
#     train_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=_nw,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=_nw > 0,
#         prefetch_factor=2 if _nw > 0 else None,
#     )
#     steps_per_epoch = len(train_loader)

#     print('Starting training...')
#     Trainer = train_factory[opt.task]
#     trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
#     trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

#     if in_phase1:
#         phase_epochs = max(1, opt.num_epochs - warmup_ep)
#     else:
#         phase_epochs = warmup_ep if warmup_ep > 0 else opt.num_epochs
#     scheduler = stage_mgr.build_phase_scheduler(
#         optimizer, opt, build_scheduler, steps_per_epoch, phase_epochs,
#         warmup_iters=min(getattr(opt, 'warmup_iters', 2000), steps_per_epoch))

#     if in_phase1 and start_epoch > warmup_ep:
#         scheduler.fast_forward((start_epoch - warmup_ep) * steps_per_epoch)
#     elif (not in_phase1) and start_epoch > 0:
#         scheduler.fast_forward(start_epoch * steps_per_epoch)

#     best_mAP   = 0.0
#     best_track = 0.0

#     for epoch in range(start_epoch + 1, opt.num_epochs + 1):
#         mark = epoch if opt.save_all else 'last'

#         # ── Phase 0 → Phase 1 transition ───────────────────────────────────
#         if (not det_only) and (not in_phase1) and epoch > warmup_ep:
#             in_phase1 = True
#             stage_mgr.apply_phase1(
#                 model,
#                 keep_backbone_frozen=getattr(opt, 'keep_backbone_frozen', False),
#                 freeze_norm=getattr(opt, 'freeze_norm_stats', False))
#             optimizer = stage_mgr.build_phase_optimizer(
#                 model, trainer.loss, opt, build_optimizer, lr=opt.lr)
#             trainer.optimizer = optimizer
#             trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
#             scheduler = stage_mgr.build_phase_scheduler(
#                 optimizer, opt, build_scheduler, steps_per_epoch,
#                 max(1, opt.num_epochs - warmup_ep),
#                 warmup_iters=min(getattr(opt, 'warmup_iters', 2000), steps_per_epoch))
#             print(f'[stage] >>> switched to Phase 1 (joint) at epoch {epoch}')

#         # ── id_weight schedule ──────────────────────────────────────────────
#         if det_only:
#             trainer.loss.id_weight = 0.0
#         elif in_phase1:
#             stage_mgr.ramp_id_weight(trainer.loss, opt.id_weight,
#                                      epoch - warmup_ep,
#                                      getattr(opt, 'id_warmup_epochs', 0))
#         else:
#             trainer.loss.id_weight = opt.id_weight
#         if not det_only:
#             logger.write('id_w {:.3f} | '.format(trainer.loss.id_weight))

#         train_loader.dataset.set_epoch(epoch - 1)

#         log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

#         cur_lr = optimizer.param_groups[0]['lr']
#         logger.write('epoch: {} |'.format(epoch))
#         logger.write('lr {:e} | '.format(cur_lr))
#         for k, v in log_dict_train.items():
#             logger.scalar_summary('train_{}'.format(k), v, epoch)
#             logger.write('{} {:8f} | '.format(k, v))

#         # ── Periodic checkpoint ─────────────────────────────────────────────
#         if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
#                        epoch, model, optimizer)
#         else:
#             save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
#                        epoch, model, optimizer)

#         # ── COCO mAP eval (stage-1 only) ───────────────────────────────────
#         if det_only and val_loader is not None and opt.val_intervals > 0 \
#                 and epoch % opt.val_intervals == 0:
#             print(f'\n[Eval] epoch {epoch} — running COCO mAP (stage-1)...')
#             metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)

#             log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
#             print(f'[Eval] {log_line}')
#             logger.write(f'[eval] {log_line} | ')
#             for k, v in metrics.items():
#                 logger.scalar_summary(f'val_{k}', v, epoch)

#             cur_mAP = metrics.get('AP', 0.0)
#             if cur_mAP > best_mAP:
#                 best_mAP = cur_mAP
#                 save_model(os.path.join(opt.save_dir, 'model_best.pth'),
#                            epoch, model, optimizer)
#                 print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

#         # ── Tracking eval (stage-2 only) ────────────────────────────────────
#         _tvi = getattr(opt, 'track_val_intervals', 0) or opt.val_intervals
#         do_track = (
#             (not det_only)
#             and getattr(opt, 'track_val', False)
#             and val_ann_file and val_img_root
#             and opt.val_intervals > 0
#             and epoch % max(1, _tvi) == 0
#         )
#         skip_p0 = (not in_phase1) and not getattr(opt, 'track_val_in_phase0', False)
#         if do_track and not skip_p0:
#             print(f'\n[Track-Eval] epoch {epoch} — running HOTA tracking eval (GMC off)...')
#             tmetrics = run_track_eval(model, opt, val_ann_file, val_img_root)
#             if tmetrics and 'hota' in tmetrics:
#                 tline = (f"HOTA {tmetrics['hota']:.4f}  DetA {tmetrics['deta']:.4f}  "
#                          f"AssA {tmetrics['assa']:.4f}  LocA {tmetrics['loca']:.4f}  "
#                          f"DetRe {tmetrics['detre']:.4f}  AssRe {tmetrics['assre']:.4f}  "
#                          f"({tmetrics['fps']:.1f} fps)")
#                 print(f'[Track-Eval] {tline}')
#                 logger.write(f'[track] {tline} | ')
#                 logger.scalar_summary('val_hota', tmetrics['hota'], epoch)
#                 logger.scalar_summary('val_deta', tmetrics['deta'], epoch)
#                 logger.scalar_summary('val_assa', tmetrics['assa'], epoch)
#                 logger.scalar_summary('val_loca', tmetrics['loca'], epoch)
#                 logger.scalar_summary('val_track_score', tmetrics['track_score'], epoch)

#                 if tmetrics['track_score'] > best_track:
#                     best_track = tmetrics['track_score']
#                     save_model(os.path.join(opt.save_dir, 'model_best.pth'),
#                                epoch, model, optimizer)
#                     print(f"[Track-Eval] ★ New best HOTA={best_track:.4f} "
#                           f"(DetA={tmetrics['deta']:.4f}, AssA={tmetrics['assa']:.4f}) "
#                           f"→ model_best.pth")

#         logger.write('\n')

#         if epoch in opt.lr_step:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#         if epoch % 5 == 0 or epoch >= 25:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#     logger.close()


# if __name__ == '__main__':
#     opt = opts().parse()
#     print("opt.gpus: ", opt.gpus)
#     print('epoch:', opt.num_epochs)
#     run(opt)







# import copy
# import os
# import json
# import collections
# from concurrent.futures import ThreadPoolExecutor

# import cv2
# import numpy as np

# # Progress bar — optional dependency. Falls back to a no-op wrapper if tqdm
# # is not installed so eval still runs (just without the live bar).
# try:
#     from tqdm import tqdm
# except Exception:  # pragma: no cover
#     def tqdm(iterable=None, **kwargs):
#         return iterable if iterable is not None else []
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# import torch
# import torch.utils.data
# from torchvision.transforms import transforms as T

# import _paths  # noqa: F401  (sys.path bootstrap)
# from falconmot.cfg.args import opts
# from falconmot.nn import create_model, load_model, save_model
# from falconmot.nn.parallel import DataParallel
# from falconmot.utils.log import Logger
# from falconmot.data.factory import get_dataset
# from falconmot.engine.factory import train_factory
# from falconmot.engine import stage as stage_mgr
# from falconmot.optim import build_optimizer, build_scheduler
# from falconmot.utils.eval import CocoJsonEvaluator
# from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor
# from collections import defaultdict as _defaultdict
# from falconmot.tracker.multitracker import MCJDETracker, MCTrack
# from falconmot.tracker.utils.coco_gt_reader import load_all_coco_gt
# from falconmot.tracker.utils.hota import HOTACollector, box_iou_matrix
# from falconmot.data.dataset import LoadCocoSequencesForTracking, preprocess_for_tracking


# def _with_lr(opt, lr):
#     """Shallow copy of opt with a different base lr (for per-phase optimizer)."""
#     o = copy.copy(opt)
#     o.lr = lr
#     return o


# def _print_stage1_banner(opt):
#     if getattr(opt, 'train_single_det', False):
#         w, h = opt.input_wh[0], opt.input_wh[1]
#         es = getattr(opt, 'eval_spatial_size', [h, w])
#         print('=' * 72)
#         print('STAGE 1: Detection-only training (--train_single_det)')
#         print(f'  input {w}x{h}  eval_spatial_size={es}  use_s4={getattr(opt, "use_s4", False)} '
#               f'use_s4_aux={getattr(opt, "use_s4_aux", True)}')
#         print(f'  losses: cls + bbox + giou'
#               f"{' + s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''}  |  "
#               f'ReID: OFF')
#         print(f'  aug: mosaic={getattr(opt, "mosaic", False)} '
#               f'(p={getattr(opt, "mosaic_prob", 0.5)})  '
#               f'temporal_mosaic=OFF')
#         if getattr(opt, 'deim_pretrained', ''):
#             print(f'  deim_pretrained: {opt.deim_pretrained}')
#         print('=' * 72)
#     elif getattr(opt, 'train_reid_only', False):
#         print('=' * 72)
#         print('STAGE 2: ReID-only training (--train_reid_only)')
#         print('  Detector is completely FROZEN (including BatchNorm stats).')
#         print('  Training ONLY the ReID head and Orthogonal Feature Loss.')
#         print('=' * 72)


# def _summarize_model(model, opt):
#     core = model.module if hasattr(model, 'module') else model
#     print(f'[model] use_s4={getattr(core, "use_s4", False)}  '
#           f'use_s4_aux={getattr(core, "use_s4_aux", False)}  '
#           f'use_reid={getattr(core, "use_reid", False)}  '
#           f's4_branch={hasattr(core, "s4_branch")}  '
#           f'reid_head={hasattr(core, "reid_head")}')
#     if getattr(opt, 'train_single_det', False) and getattr(opt, 'use_s4', False) \
#             and getattr(opt, 'deim_pretrained', ''):
#         print('[model] NOTE: s4_branch/s4_aux_head not in DEIM pretrained — '
#               'trained from scratch in stage-1.')


# @torch.no_grad()
# def run_coco_eval(model, val_loader, opt, ann_file: str = '') -> dict:
#     """COCO mAP evaluation on the val split."""
#     model.eval()

#     postprocessor = FalconJDEPostProcessor(
#         num_classes=opt.num_classes,
#         num_top_queries=500,
#         use_focal_loss=True,
#         conf_thres=0.0
#     )

#     evaluator = CocoJsonEvaluator(ann_file) if ann_file else JDECocoEvaluator(
#         num_classes=opt.num_classes,
#         net_h=opt.input_wh[1],
#         net_w=opt.input_wh[0],
#     )

#     max_batches = getattr(opt, 'debug_val_batches', 0)

#     for i, batch in enumerate(val_loader):
#         if max_batches > 0 and i >= max_batches:
#             break
#         batch = {k: v.to(opt.device, non_blocking=True)
#                  if isinstance(v, torch.Tensor) else v
#                  for k, v in batch.items()}

#         orig_hw    = batch.get('orig_hw')
#         orig_sizes = orig_hw if orig_hw is not None else \
#                      torch.tensor([[opt.input_wh[1], opt.input_wh[0]]] * batch['input'].shape[0],
#                                   device=opt.device)

#         outputs    = model(batch['input'])
#         dt_results = postprocessor(outputs, orig_sizes)
#         evaluator.update(dt_results, batch)

#     model.train()
#     return evaluator.summarize()


# class _ParallelFrameReader:
#     """Read + preprocess a sequence's frames in worker threads, yield IN ORDER.

#     The tracker is stateful (Kalman + ReID memory) so frames must reach it in
#     order — but the *loading* of frame N+1..N+k (disk read + resize + normalize)
#     can overlap with the GPU forward of frame N. cv2.imread and the numpy
#     resize/normalize release the GIL, so threads give real parallelism here and
#     keep the GPU fed instead of idling on disk I/O.

#     Emits the same (frame_id, img_chw, img0_bgr) tuples as _CocoSeqIterator.
#     """

#     def __init__(self, frames, width, height, num_workers=4, prefetch=8):
#         self.frames      = frames
#         self.width       = width
#         self.height      = height
#         self.num_workers = max(1, int(num_workers))
#         # Keep at least num_workers reads in flight; deeper window hides latency.
#         self.prefetch    = max(self.num_workers, int(prefetch))

#     @staticmethod
#     def _load(args):
#         frame_id, path, w, h = args
#         img0 = cv2.imread(path)
#         img  = preprocess_for_tracking(img0, w, h)
#         return frame_id, img, img0

#     def __iter__(self):
#         tasks = ((fid, p, self.width, self.height) for fid, p in self.frames)
#         with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
#             inflight = collections.deque()
#             # Prime the sliding window.
#             for _ in range(self.prefetch):
#                 try:
#                     inflight.append(ex.submit(self._load, next(tasks)))
#                 except StopIteration:
#                     break
#             # Yield oldest-first while topping the window back up.
#             while inflight:
#                 fut = inflight.popleft()
#                 try:
#                     inflight.append(ex.submit(self._load, next(tasks)))
#                 except StopIteration:
#                     pass
#                 yield fut.result()


# def _collect_hota_frame(coll, class_names, seq_id, gt_objs, pred_tlwhs, pred_tids, offset):
#     """Feed one frame's GT + tracker output into the HOTACollector, split per class.

#     The class is baked into the (global) id as ``id // offset`` on both sides, so
#     GT global_id and tracker global_id (track_id + c*offset) split identically.
#     Frames that are empty for a class (no GT, no pred) contribute nothing to HOTA
#     and are skipped — this shrinks the per-(class,seq) timestep list and speeds up
#     the final TrackEval pass without changing the score.
#     """
#     ncls = len(class_names)
#     gt_b = [[] for _ in range(ncls)]
#     gt_i = [[] for _ in range(ncls)]
#     for tlwh, gid, _vis in gt_objs:
#         c = int(gid) // offset
#         if 0 <= c < ncls:
#             gt_b[c].append(tlwh)
#             gt_i[c].append(int(gid))

#     pr_b = [[] for _ in range(ncls)]
#     pr_i = [[] for _ in range(ncls)]
#     for tlwh, tid in zip(pred_tlwhs, pred_tids):
#         c = int(tid) // offset
#         if 0 <= c < ncls:
#             pr_b[c].append(tlwh)
#             pr_i[c].append(int(tid))

#     for c in range(ncls):
#         if not gt_i[c] and not pr_i[c]:
#             continue
#         g = np.asarray(gt_b[c], dtype=np.float64).reshape(-1, 4)
#         p = np.asarray(pr_b[c], dtype=np.float64).reshape(-1, 4)
#         sim = box_iou_matrix(g, p)
#         coll.add_frame(class_names[c], seq_id, gt_i[c], pr_i[c], sim)


# # @torch.inference_mode()
# # def run_track_eval(model, opt, val_ann_file: str, val_img_root: str) -> dict:
# #     """Tracking validation over val sequences — HOTA metrics only.

# #     Uses the native class space without any class remap. GMC is disabled for
# #     speed and determinism (set_image is never called). Returns a dict with the
# #     HOTA-family scalars from hota.py (hota/deta/assa/loca/detre/assre) plus
# #     track_score (= HOTA, drives model_best) and fps.

# #     Speed: frame loading is pipelined across threads (overlaps disk I/O with
# #     GPU inference), the detector forward runs in fp16 autocast when
# #     --track_val_fp16 is set, and HOTA is computed in-memory (no result .txt
# #     round-trip) with a parallel TrackEval pass. A tqdm bar shows live progress.
# #     """
# #     import time

# #     if not (val_ann_file and val_img_root and os.path.isfile(val_ann_file)):
# #         print('[track-eval] missing val_ann/val_img -- skipping tracking eval.')
# #         return {}

# #     model.eval()
# #     net_w, net_h = opt.img_size
# #     ncls     = opt.num_classes
# #     min_area = getattr(opt, 'min_box_area', 100)
# #     _OFF     = 1_000_000

# #     postproc = FalconJDEPostProcessor(
# #         num_classes=ncls,
# #         num_top_queries=getattr(opt, 'K', 300),
# #         conf_thres=opt.conf_thres,
# #         use_focal_loss=True,
# #     )

# #     opt_trk = copy.copy(opt)
# #     opt_trk.num_classes = ncls
# #     tracker = MCJDETracker(opt_trk, frame_rate=getattr(opt, 'frame_rate', 30))

# #     src = LoadCocoSequencesForTracking(val_ann_file, val_img_root, img_size=opt.img_size)

# #     # ── Class names for HOTA (native class space, no remap) ─────────────────
# #     # HOTA is computed per class then class-averaged (VisDrone convention).
# #     try:
# #         from falconmot.tracker.class_remap import CLS7_NAMES, CLS5_NAMES, CLS4_NAMES
# #         _name_tables = {7: CLS7_NAMES, 5: CLS5_NAMES, 4: CLS4_NAMES}
# #     except Exception:
# #         _name_tables = {}
# #     _names_map = _name_tables.get(ncls, {})
# #     class_names = {c: _names_map.get(c, f'class_{c}') for c in range(ncls)}
# #     hota_names  = [class_names[c] for c in range(ncls)]
# #     coll = HOTACollector(hota_names)

# #     # ── Parse GT ONCE (instead of re-parsing the whole json per sequence) ───
# #     gt_by_seq = load_all_coco_gt(val_ann_file)

# #     # ── fp16 autocast for the detector forward (wires up --track_val_fp16) ──
# #     use_fp16 = bool(int(getattr(opt, 'track_val_fp16', 1))) and opt.device.type == 'cuda'

# #     # ── Threaded frame prefetch params (overlap disk I/O with GPU) ──────────
# #     n_threads  = int(getattr(opt, 'track_val_loader_threads', 4))
# #     prefetch   = int(getattr(opt, 'track_val_prefetch', 8))

# #     t0, n_frames = time.time(), 0

# #     total_frames = sum(src.num_frames(s) for s in src.seqs)
# #     pbar = tqdm(total=total_frames, desc='[track-eval]', unit='f', dynamic_ncols=True)

# #     for seq_id in src.seqs:
# #         tracker.reset()
# #         gt_frames = gt_by_seq.get(seq_id, {})

# #         frames = src._seq_frames[seq_id]
# #         reader = _ParallelFrameReader(frames, net_w, net_h,
# #                                       num_workers=n_threads, prefetch=prefetch)

# #         for frame_id, img, img0 in reader:
# #             orig_h, orig_w = img0.shape[:2]
# #             sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
# #             blob  = torch.from_numpy(img[None]).to(opt.device, non_blocking=True)

# #             with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
# #                 output = model(blob)
# #                 res = postproc(output, sizes)[0]

# #             dets = _defaultdict(list)
# #             if len(res['scores']) > 0:
# #                 bxs = res['boxes'].float().cpu().numpy()
# #                 scs = res['scores'].float().cpu().numpy()
# #                 lbs = res['labels'].cpu().numpy()
# #                 rid = res['reid'].float().cpu().numpy() if 'reid' in res else None
# #                 ws  = bxs[:, 2] - bxs[:, 0]
# #                 hs  = bxs[:, 3] - bxs[:, 1]
# #                 for i in np.where((ws > 0) & (hs > 0))[0]:
# #                     c = int(lbs[i])
# #                     if c < 0 or c >= ncls:
# #                         continue
# #                     tlwh = np.array([bxs[i, 0], bxs[i, 1], ws[i], hs[i]], dtype=np.float32)
# #                     emb  = rid[i] if rid is not None else np.zeros(1, dtype=np.float32)
# #                     dets[c].append(MCTrack(tlwh, float(scs[i]), emb, ncls, c))

# #             online = tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

# #             tlwhs, tids = [], []
# #             for c, tracks in online.items():
# #                 for t in tracks:
# #                     w, h = t.curr_tlwh[2], t.curr_tlwh[3]
# #                     if t.track_id < 0 or (w * h) <= min_area:
# #                         continue
# #                     tlwhs.append(t.curr_tlwh)
# #                     tids.append(int(t.track_id) + c * _OFF)

# #             _collect_hota_frame(coll, class_names, seq_id,
# #                                 gt_frames.get(int(frame_id), []),
# #                                 tlwhs, tids, _OFF)
# #             n_frames += 1
# #             pbar.update(1)
# #             if n_frames % 20 == 0:
# #                 pbar.set_postfix_str(f'{n_frames / max(1e-6, time.time() - t0):.1f} fps')

# #     pbar.close()
# #     infer_elapsed = time.time() - t0   # pure inference time (excludes HOTA compute)
# #     model.train()

# #     if n_frames == 0:
# #         return {}

# #     # ── HOTA (the only metric — drives model_best selection) ────────────────
# #     hota_workers = int(getattr(opt, 'track_val_hota_workers', 0)) or min(8, os.cpu_count() or 4)
# #     try:
# #         hres = coll.compute(num_workers=hota_workers)
# #     except Exception as e:
# #         print(f'[track-eval] HOTA unavailable ({e}); install TrackEval to enable it. '
# #               f'Skipping best-model update this round.')
# #         return {'fps': n_frames / max(1e-6, infer_elapsed)}

# #     ov  = hres['overall']
# #     fps = n_frames / max(1e-6, infer_elapsed)
# #     out = {k.lower(): float(ov.get(k, 0.0))
# #            for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe')}
# #     out['track_score'] = out['hota']     # best model selected by HOTA
# #     out['fps'] = fps
# #     return out


# @torch.no_grad()
# def run_reid_eval(model, val_loader, criterion, opt) -> dict:
#     """Đánh giá ReID bằng cách tính ReID Loss trên tập Validation (Ảnh rời rạc).
#     Model có ReID Loss thấp nhất sẽ được Save Best.
#     """
#     import time
#     model.eval()
#     criterion.eval()

#     total_reid_loss = 0.0
#     num_batches = 0

#     t0 = time.time()
#     for batch_i, batch in enumerate(val_loader):
#         # Đưa data lên GPU
#         batch = {k: v.to(opt.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
#                  for k, v in batch.items()}
        
#         # Tạo Target y hệt lúc Train
#         B = batch['input'].shape[0]
#         targets = []
#         for i in range(B):
#             n = int(batch['detr_num_objs'][i].item())
#             valid_labels = batch['detr_labels'][i, :n]
#             valid_boxes = batch['detr_boxes'][i, :n]
#             valid_tids = batch['detr_track_ids'][i, :n]
#             keep = valid_labels >= 0
#             targets.append({
#                 'labels':    valid_labels[keep],
#                 'boxes':     valid_boxes[keep],
#                 'track_ids': valid_tids[keep],
#             })

#         # Chạy Model
#         outputs = model(batch['input'], targets)
        
#         # Chỉ tính Loss (Bỏ qua Detection)
#         loss_dict = criterion(outputs, targets, epoch=0, compute_det_loss=False)
        
#         # Cộng dồn ReID Loss
#         if 'loss_reid' in loss_dict:
#             total_reid_loss += loss_dict['loss_reid'].item()
#             num_batches += 1

#     model.train()
#     criterion.train()

#     avg_reid_loss = (total_reid_loss / num_batches) if num_batches > 0 else float('inf')
    
#     elapsed = time.time() - t0
#     return {'val_reid_loss': avg_reid_loss, 'eval_time': elapsed}


# @torch.inference_mode()
# def run_track_eval(model, opt, val_ann_file: str, val_img_root: str) -> dict:
#     """Tracking validation over val sequences — HOTA metrics only.
#     Uses Plain Resize logic (no letterbox) for QAM Dense Map feeding.
#     """
#     import time
#     from collections import defaultdict as _defaultdict

#     if not (val_ann_file and val_img_root and os.path.isfile(val_ann_file)):
#         print('[track-eval] missing val_ann/val_img -- skipping tracking eval.')
#         return {}

#     model.eval()
    
#     # [QUAN TRỌNG 1]: Ép mô hình xuất Dense Map (emb_map) khi đang chạy eval()
#     core_model = model.module if hasattr(model, 'module') else model
#     core_model.return_reid_dense = True  

#     net_w, net_h = opt.img_size
#     ncls     = opt.num_classes
#     min_area = getattr(opt, 'min_box_area', 100)
#     _OFF     = 1_000_000

#     postproc = FalconJDEPostProcessor(
#         num_classes=ncls,
#         num_top_queries=getattr(opt, 'K', 300),
#         conf_thres=opt.conf_thres,
#         use_focal_loss=True,
#     )
#     # KHÔNG gọi postproc.set_net_hw() vì ta dùng Plain Resize

#     opt_trk = copy.copy(opt)
#     opt_trk.num_classes = ncls
#     tracker = MCJDETracker(opt_trk, frame_rate=getattr(opt, 'frame_rate', 30))

#     src = LoadCocoSequencesForTracking(val_ann_file, val_img_root, img_size=opt.img_size)

#     try:
#         from falconmot.tracker.class_remap import CLS7_NAMES, CLS5_NAMES, CLS4_NAMES
#         _name_tables = {7: CLS7_NAMES, 5: CLS5_NAMES, 4: CLS4_NAMES}
#     except Exception:
#         _name_tables = {}
#     _names_map = _name_tables.get(ncls, {})
#     class_names = {c: _names_map.get(c, f'class_{c}') for c in range(ncls)}
#     hota_names  = [class_names[c] for c in range(ncls)]
#     coll = HOTACollector(hota_names)

#     gt_by_seq = load_all_coco_gt(val_ann_file)
#     use_fp16 = bool(int(getattr(opt, 'track_val_fp16', 1))) and opt.device.type == 'cuda'

#     n_threads  = int(getattr(opt, 'track_val_loader_threads', 4))
#     prefetch   = int(getattr(opt, 'track_val_prefetch', 8))

#     t0, n_frames = time.time(), 0
#     total_frames = sum(src.num_frames(s) for s in src.seqs)
#     pbar = tqdm(total=total_frames, desc='[track-eval]', unit='f', dynamic_ncols=True)

#     for seq_id in src.seqs:
#         tracker.reset()
#         gt_frames = gt_by_seq.get(seq_id, {})

#         frames = src._seq_frames[seq_id]
#         reader = _ParallelFrameReader(frames, net_w, net_h,
#                                       num_workers=n_threads, prefetch=prefetch)

#         for frame_id, img, img0 in reader:
#             orig_h, orig_w = img0.shape[:2]
#             sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
#             blob  = torch.from_numpy(img[None]).to(opt.device, non_blocking=True)

#             with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
#                 output = model(blob)
#                 res = postproc(output, sizes)[0]
                
#             tracker.set_image(img0)

#             # ====================================================================
#             # [QUAN TRỌNG 2]: Set Dense Map cho Tracker (Logic Plain Resize)
#             # ====================================================================
#             if 'reid_dense' in output and getattr(opt, 'use_appearance_motion', False):
#                 # Tỷ lệ scale độc lập theo 2 trục X và Y
#                 rx = net_w / orig_w
#                 ry = net_h / orig_h
                
#                 tracker.set_dense(
#                     output['reid_dense'],  # Tensor [128, H/4, W/4]
#                     stride=output['reid_dense_stride'],
#                     ratio_x=rx, 
#                     ratio_y=ry, 
#                     pad_w=0.0, 
#                     pad_h=0.0
#                 )
#             else:
#                 tracker.set_dense(None, 1.0, 1.0, 1.0)
#             # ====================================================================

#             dets = _defaultdict(list)
#             if len(res['scores']) > 0:
#                 bxs = res['boxes'].float().cpu().numpy()
#                 scs = res['scores'].float().cpu().numpy()
#                 lbs = res['labels'].cpu().numpy()
#                 rid = res['reid'].float().cpu().numpy() if 'reid' in res else None
#                 ws  = bxs[:, 2] - bxs[:, 0]
#                 hs  = bxs[:, 3] - bxs[:, 1]
#                 for i in np.where((ws > 0) & (hs > 0))[0]:
#                     c = int(lbs[i])
#                     if c < 0 or c >= ncls:
#                         continue
#                     tlwh = np.array([bxs[i, 0], bxs[i, 1], ws[i], hs[i]], dtype=np.float32)
#                     emb  = rid[i] if rid is not None else np.zeros(1, dtype=np.float32)
#                     dets[c].append(MCTrack(tlwh, float(scs[i]), emb, ncls, c))

#             # Update tracker (Không truyền dense_map vào đây nữa, vì đã gọi set_dense ở trên)
#             online = tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

#             tlwhs, tids = [], []
#             for c, tracks in online.items():
#                 for t in tracks:
#                     w, h = t.curr_tlwh[2], t.curr_tlwh[3]
#                     if t.track_id < 0 or (w * h) <= min_area:
#                         continue
#                     tlwhs.append(t.curr_tlwh)
#                     tids.append(int(t.track_id) + c * _OFF)

#             _collect_hota_frame(coll, class_names, seq_id,
#                                 gt_frames.get(int(frame_id), []),
#                                 tlwhs, tids, _OFF)
#             n_frames += 1
#             pbar.update(1)
#             if n_frames % 20 == 0:
#                 pbar.set_postfix_str(f'{n_frames / max(1e-6, time.time() - t0):.1f} fps')

#     pbar.close()
#     infer_elapsed = time.time() - t0   
    
#     # [QUAN TRỌNG 3]: Trả mô hình về trạng thái cũ để không sinh thừa DenseMap khi train
#     core_model.return_reid_dense = False
#     model.train()

#     if n_frames == 0:
#         return {}

#     hota_workers = int(getattr(opt, 'track_val_hota_workers', 0)) or min(8, os.cpu_count() or 4)
#     try:
#         hres = coll.compute(num_workers=hota_workers)
#     except Exception as e:
#         print(f'[track-eval] HOTA unavailable ({e}); install TrackEval to enable it. '
#               f'Skipping best-model update this round.')
#         return {'fps': n_frames / max(1e-6, infer_elapsed)}

#     ov  = hres['overall']
#     fps = n_frames / max(1e-6, infer_elapsed)
#     out = {k.lower(): float(ov.get(k, 0.0))
#            for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe')}
#     out['track_score'] = out['hota']     # best model selected by HOTA
#     out['fps'] = fps
#     return out


# def run(opt):
#     _print_stage1_banner(opt)
#     torch.manual_seed(opt.seed)
#     torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

#     print('Setting up data...')
#     Dataset      = get_dataset(opt.dataset, opt.task)
#     use_coco_fmt = (opt.dataset == 'coco')

#     with open(opt.data_cfg) as f:
#         data_config  = json.load(f)
#     dataset_root = data_config['root']
#     print("Dataset root: %s" % dataset_root)

#     from falconmot.data.dataset import VisDroneCocoDataset

#     if use_coco_fmt:
#         train_sources = data_config.get('train_sources')
#         if not train_sources and getattr(opt, 'merge_val_into_train', False):
#             train_sources = [
#                 {'ann': data_config['train_ann'], 'img': data_config['train_img']},
#                 {'ann': data_config['val_ann'],   'img': data_config['val_img']},
#             ]
#             print('[data] merge_val_into_train: merging train + val into the train set')

#         if train_sources:
#             dataset = VisDroneCocoDataset(opt=opt, sources=train_sources, augment=True)
#         else:
#             dataset = VisDroneCocoDataset(
#                 opt=opt, img_root=data_config['train_img'],
#                 ann_file=data_config['train_ann'], augment=True)
#     else:
#         dataset = Dataset(opt=opt, root=dataset_root,
#                           paths=data_config['train'], img_size=opt.input_wh,
#                           augment=True, transforms=T.Compose([T.ToTensor()]))
#     opt = opts().init()
#     opt = opts().update_dataset_info_and_set_heads(opt, dataset)
#     print("opt:\n", opt)
#     logger = Logger(opt)

#     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     print("opt.gpus_str: ", opt.gpus_str)
#     opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

#     # ── Val dataset (optional) ──────────────────────────────────────────────
#     val_loader   = None
#     val_ann_file = ''
#     val_img_root = ''

#     if getattr(opt, 'merge_val_into_train', False):
#         print('[data] merge_val_into_train=True -> skipping val_loader, no COCO eval.')
#     elif getattr(opt, 'val_cfg', ''):
#         with open(opt.val_cfg) as f:
#             val_config = json.load(f)
#         val_dataset = None

#         if use_coco_fmt:
#             val_ann_file = val_config.get('val_ann', '')
#             val_img      = val_config.get('val_img', '')
#             val_img_root = val_img
#             if val_ann_file and val_img:
#                 val_dataset = VisDroneCocoDataset(
#                     opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
#             else:
#                 print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
#         else:
#             val_root  = val_config.get('root', dataset_root)
#             val_paths = val_config.get('val') or val_config.get('test') or []
#             if val_paths:
#                 val_dataset = Dataset(
#                     opt=opt, root=val_root, paths=val_paths,
#                     img_size=opt.input_wh, augment=False,
#                     transforms=T.Compose([T.ToTensor()]))
#             else:
#                 print('[warn] val_cfg provided but no val/test paths found.')

#         if val_dataset is not None:
#             val_loader = torch.utils.data.DataLoader(
#                 dataset=val_dataset,
#                 batch_size=opt.batch_size,
#                 shuffle=False,
#                 num_workers=opt.num_workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 persistent_workers=opt.num_workers > 0,
#                 prefetch_factor=2 if opt.num_workers > 0 else None,
#             )
#             print(f'Val dataset: {len(val_dataset)} images')

#     print('Creating model...')
#     model = create_model(opt.arch, opt)
#     _summarize_model(model, opt)

#     # ── Load checkpoint BEFORE applying freeze policy ───────────────────────
#     start_epoch = 0
#     if opt.load_model != '':
#         loaded = load_model(model, opt.load_model, None, opt.resume, opt.lr, opt.lr_step)
#         model = loaded[0] if isinstance(loaded, tuple) else loaded
#         if opt.resume:
#             try:
#                 ckpt = torch.load(opt.load_model, map_location='cpu', weights_only=False)
#                 start_epoch = int(ckpt.get('epoch', 0))
#             except Exception:
#                 start_epoch = 0

#     # ========================================================================
#     # ── Training stage policy (Stage 1 vs Stage 2) ──────────────────────────
#     # ========================================================================
#     det_only  = getattr(opt, 'train_single_det', False)
#     reid_only = getattr(opt, 'train_stage2_mot', False)

#     if det_only and reid_only:
#         raise ValueError("Cannot set both --train_single_det and --train_reid_only!")

#     if det_only:
#         # STAGE 1: Detection only
#         stage_mgr.apply_det_only(model)
#     elif reid_only:
#         # STAGE 2: Freeze Detection, Train ReID
#         stage_mgr.apply_stage2_mot(model)
#     else:
#         # Fallback / Joint Training
#         stage_mgr.apply_joint_training(model)

#     optimizer = build_optimizer(model, _with_lr(opt, opt.lr))

#     _nw = opt.num_workers
#     train_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=_nw,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=_nw > 0,
#         prefetch_factor=2 if _nw > 0 else None,
#     )
#     steps_per_epoch = len(train_loader)

#     print('Starting training...')
#     Trainer = train_factory[opt.task]
#     trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
#     trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

#     scheduler = stage_mgr.build_phase_scheduler(
#         optimizer, opt, build_scheduler, steps_per_epoch, opt.num_epochs,
#         warmup_iters=min(getattr(opt, 'warmup_iters', 2000), steps_per_epoch))

#     if start_epoch > 0:
#         scheduler.fast_forward(start_epoch * steps_per_epoch)

#     best_mAP   = 0.0
#     best_track = 0.0

#     for epoch in range(start_epoch + 1, opt.num_epochs + 1):
#         mark = epoch if opt.save_all else 'last'

#         # ── id_weight setup ────────────────────────────────────────────────
#         if det_only:
#             trainer.loss.id_weight = 0.0
#         else:
#             # Stage 2 (reid_only) or joint: use constant id_weight
#             trainer.loss.id_weight = opt.id_weight
#             logger.write('id_w {:.3f} | '.format(trainer.loss.id_weight))

#         train_loader.dataset.set_epoch(epoch - 1)

#         log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

#         cur_lr = optimizer.param_groups[0]['lr']
#         logger.write('epoch: {} |'.format(epoch))
#         logger.write('lr {:e} | '.format(cur_lr))
#         for k, v in log_dict_train.items():
#             logger.scalar_summary('train_{}'.format(k), v, epoch)
#             logger.write('{} {:8f} | '.format(k, v))

#         # ── Periodic checkpoint ─────────────────────────────────────────────
#         if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
#                        epoch, model, optimizer)
#         else:
#             save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
#                        epoch, model, optimizer)

#         # ── COCO mAP eval (stage-1 only) ───────────────────────────────────
#         if det_only and val_loader is not None and opt.val_intervals > 0 \
#                 and epoch % opt.val_intervals == 0:
#             print(f'\n[Eval] epoch {epoch} — running COCO mAP (stage-1)...')
#             metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)

#             log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
#             print(f'[Eval] {log_line}')
#             logger.write(f'[eval] {log_line} | ')
#             for k, v in metrics.items():
#                 logger.scalar_summary(f'val_{k}', v, epoch)

#             cur_mAP = metrics.get('AP', 0.0)
#             if cur_mAP > best_mAP:
#                 best_mAP = cur_mAP
#                 save_model(os.path.join(opt.save_dir, 'model_best.pth'),
#                            epoch, model, optimizer)
#                 print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

#         # ── Tracking eval (stage-2 only) ────────────────────────────────────
#         _tvi = getattr(opt, 'track_val_intervals', 0) or opt.val_intervals
#         do_track = (
#             (not det_only)
#             and getattr(opt, 'track_val', False)
#             and val_ann_file and val_img_root
#             and opt.val_intervals > 0
#             and epoch % max(1, _tvi) == 0
#         )
#         if do_track:
#             print(f'\n[Track-Eval] epoch {epoch} — running HOTA tracking eval (GMC off)...')
#             tmetrics = run_track_eval(model, opt, val_ann_file, val_img_root)
#             if tmetrics and 'hota' in tmetrics:
#                 tline = (f"HOTA {tmetrics['hota']:.4f}  DetA {tmetrics['deta']:.4f}  "
#                          f"AssA {tmetrics['assa']:.4f}  LocA {tmetrics['loca']:.4f}  "
#                          f"DetRe {tmetrics['detre']:.4f}  AssRe {tmetrics['assre']:.4f}  "
#                          f"({tmetrics['fps']:.1f} fps)")
#                 print(f'[Track-Eval] {tline}')
#                 logger.write(f'[track] {tline} | ')
#                 logger.scalar_summary('val_hota', tmetrics['hota'], epoch)
#                 logger.scalar_summary('val_deta', tmetrics['deta'], epoch)
#                 logger.scalar_summary('val_assa', tmetrics['assa'], epoch)
#                 logger.scalar_summary('val_loca', tmetrics['loca'], epoch)
#                 logger.scalar_summary('val_track_score', tmetrics['track_score'], epoch)

#                 if tmetrics['track_score'] > best_track:
#                     best_track = tmetrics['track_score']
#                     save_model(os.path.join(opt.save_dir, 'model_best.pth'),
#                                epoch, model, optimizer)
#                     print(f"[Track-Eval] ★ New best HOTA={best_track:.4f} "
#                           f"(DetA={tmetrics['deta']:.4f}, AssA={tmetrics['assa']:.4f}) "
#                           f"→ model_best.pth")

#         logger.write('\n')

#         if epoch in opt.lr_step:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#         if epoch % 5 == 0 or epoch >= 25:
#             save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
#                        epoch, model, optimizer)

#     logger.close()


# if __name__ == '__main__':
#     opt = opts().parse()
#     print("opt.gpus: ", opt.gpus)
#     print('epoch:', opt.num_epochs)
#     run(opt)





import copy
import os
import json
import collections
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# Progress bar — optional dependency. Falls back to a no-op wrapper if tqdm
# is not installed so eval still runs (just without the live bar).
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.utils.data
from torchvision.transforms import transforms as T

import _paths  # noqa: F401  (sys.path bootstrap)
from falconmot.cfg.args import opts
from falconmot.nn import create_model, load_model, save_model
from falconmot.nn.parallel import DataParallel
from falconmot.utils.log import Logger
from falconmot.data.factory import get_dataset
from falconmot.engine.factory import train_factory
from falconmot.engine import stage as stage_mgr
from falconmot.optim import build_optimizer, build_scheduler
from falconmot.utils.eval import CocoJsonEvaluator
from falconmot.nn.falcon_jde.postprocessor import FalconJDEPostProcessor
from collections import defaultdict as _defaultdict
from falconmot.tracker.multitracker import MCJDETracker, MCTrack
from falconmot.tracker.utils.coco_gt_reader import load_all_coco_gt
from falconmot.tracker.utils.hota import HOTACollector, box_iou_matrix
from falconmot.data.dataset import LoadCocoSequencesForTracking, preprocess_for_tracking


def _with_lr(opt, lr):
    """Shallow copy of opt with a different base lr (for per-phase optimizer)."""
    o = copy.copy(opt)
    o.lr = lr
    return o


def _print_stage1_banner(opt):
    if getattr(opt, 'train_single_det', False):
        w, h = opt.input_wh[0], opt.input_wh[1]
        es = getattr(opt, 'eval_spatial_size', [h, w])
        print('=' * 72)
        print('STAGE 1: Detection-only training (--train_single_det)')
        print(f'  input {w}x{h}  eval_spatial_size={es}  use_s4={getattr(opt, "use_s4", False)} '
              f'use_s4_aux={getattr(opt, "use_s4_aux", True)}')
        print(f'  losses: cls + bbox + giou'
              f"{' + s4_aux' if (getattr(opt, 'use_s4', False) and getattr(opt, 'use_s4_aux', True)) else ''}  |  "
              f'ReID: OFF')
        print(f'  aug: mosaic={getattr(opt, "mosaic", False)} '
              f'(p={getattr(opt, "mosaic_prob", 0.5)})  '
              f'temporal_mosaic=OFF')
        if getattr(opt, 'deim_pretrained', ''):
            print(f'  deim_pretrained: {opt.deim_pretrained}')
        print('=' * 72)
    elif getattr(opt, 'train_reid_only', False):
        print('=' * 72)
        print('STAGE 2: ReID-only training (--train_reid_only)')
        print('  Detector is completely FROZEN (including BatchNorm stats).')
        print('  Training ONLY the ReID head and Orthogonal Feature Loss.')
        print('=' * 72)


def _summarize_model(model, opt):
    core = model.module if hasattr(model, 'module') else model
    print(f'[model] use_s4={getattr(core, "use_s4", False)}  '
          f'use_s4_aux={getattr(core, "use_s4_aux", False)}  '
          f'use_reid={getattr(core, "use_reid", False)}  '
          f's4_branch={hasattr(core, "s4_branch")}  '
          f'reid_head={hasattr(core, "reid_head")}')
    if getattr(opt, 'train_single_det', False) and getattr(opt, 'use_s4', False) \
            and getattr(opt, 'deim_pretrained', ''):
        print('[model] NOTE: s4_branch/s4_aux_head not in DEIM pretrained — '
              'trained from scratch in stage-1.')


@torch.no_grad()
def run_coco_eval(model, val_loader, opt, ann_file: str = '') -> dict:
    """COCO mAP evaluation on the val split."""
    model.eval()

    postprocessor = FalconJDEPostProcessor(
        num_classes=opt.num_classes,
        num_top_queries=500,
        use_focal_loss=True,
        conf_thres=0.0
    )

    evaluator = CocoJsonEvaluator(ann_file) if ann_file else JDECocoEvaluator(
        num_classes=opt.num_classes,
        net_h=opt.input_wh[1],
        net_w=opt.input_wh[0],
    )

    max_batches = getattr(opt, 'debug_val_batches', 0)

    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(opt.device, non_blocking=True)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        orig_hw    = batch.get('orig_hw')
        orig_sizes = orig_hw if orig_hw is not None else \
                     torch.tensor([[opt.input_wh[1], opt.input_wh[0]]] * batch['input'].shape[0],
                                  device=opt.device)

        outputs    = model(batch['input'])
        dt_results = postprocessor(outputs, orig_sizes)
        evaluator.update(dt_results, batch)

    model.train()
    return evaluator.summarize()


class _ParallelFrameReader:
    """Read + preprocess a sequence's frames in worker threads, yield IN ORDER.

    The tracker is stateful (Kalman + ReID memory) so frames must reach it in
    order — but the *loading* of frame N+1..N+k (disk read + resize + normalize)
    can overlap with the GPU forward of frame N. cv2.imread and the numpy
    resize/normalize release the GIL, so threads give real parallelism here and
    keep the GPU fed instead of idling on disk I/O.

    Emits the same (frame_id, img_chw, img0_bgr) tuples as _CocoSeqIterator.
    """

    def __init__(self, frames, width, height, num_workers=4, prefetch=8):
        self.frames      = frames
        self.width       = width
        self.height      = height
        self.num_workers = max(1, int(num_workers))
        # Keep at least num_workers reads in flight; deeper window hides latency.
        self.prefetch    = max(self.num_workers, int(prefetch))

    @staticmethod
    def _load(args):
        frame_id, path, w, h = args
        img0 = cv2.imread(path)
        img  = preprocess_for_tracking(img0, w, h)
        return frame_id, img, img0

    def __iter__(self):
        tasks = ((fid, p, self.width, self.height) for fid, p in self.frames)
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            inflight = collections.deque()
            # Prime the sliding window.
            for _ in range(self.prefetch):
                try:
                    inflight.append(ex.submit(self._load, next(tasks)))
                except StopIteration:
                    break
            # Yield oldest-first while topping the window back up.
            while inflight:
                fut = inflight.popleft()
                try:
                    inflight.append(ex.submit(self._load, next(tasks)))
                except StopIteration:
                    pass
                yield fut.result()


def _collect_hota_frame(coll, class_names, seq_id, gt_objs, pred_tlwhs, pred_tids, offset):
    """Feed one frame's GT + tracker output into the HOTACollector, split per class.

    The class is baked into the (global) id as ``id // offset`` on both sides, so
    GT global_id and tracker global_id (track_id + c*offset) split identically.
    Frames that are empty for a class (no GT, no pred) contribute nothing to HOTA
    and are skipped — this shrinks the per-(class,seq) timestep list and speeds up
    the final TrackEval pass without changing the score.
    """
    ncls = len(class_names)
    gt_b = [[] for _ in range(ncls)]
    gt_i = [[] for _ in range(ncls)]
    for tlwh, gid, _vis in gt_objs:
        c = int(gid) // offset
        if 0 <= c < ncls:
            gt_b[c].append(tlwh)
            gt_i[c].append(int(gid))

    pr_b = [[] for _ in range(ncls)]
    pr_i = [[] for _ in range(ncls)]
    for tlwh, tid in zip(pred_tlwhs, pred_tids):
        c = int(tid) // offset
        if 0 <= c < ncls:
            pr_b[c].append(tlwh)
            pr_i[c].append(int(tid))

    for c in range(ncls):
        if not gt_i[c] and not pr_i[c]:
            continue
        g = np.asarray(gt_b[c], dtype=np.float64).reshape(-1, 4)
        p = np.asarray(pr_b[c], dtype=np.float64).reshape(-1, 4)
        sim = box_iou_matrix(g, p)
        coll.add_frame(class_names[c], seq_id, gt_i[c], pr_i[c], sim)


# @torch.inference_mode()
# def run_track_eval(model, opt, val_ann_file: str, val_img_root: str) -> dict:
#     """Tracking validation over val sequences — HOTA metrics only.

#     Uses the native class space without any class remap. GMC is disabled for
#     speed and determinism (set_image is never called). Returns a dict with the
#     HOTA-family scalars from hota.py (hota/deta/assa/loca/detre/assre) plus
#     track_score (= HOTA, drives model_best) and fps.

#     Speed: frame loading is pipelined across threads (overlaps disk I/O with
#     GPU inference), the detector forward runs in fp16 autocast when
#     --track_val_fp16 is set, and HOTA is computed in-memory (no result .txt
#     round-trip) with a parallel TrackEval pass. A tqdm bar shows live progress.
#     """
#     import time

#     if not (val_ann_file and val_img_root and os.path.isfile(val_ann_file)):
#         print('[track-eval] missing val_ann/val_img -- skipping tracking eval.')
#         return {}

#     model.eval()
#     net_w, net_h = opt.img_size
#     ncls     = opt.num_classes
#     min_area = getattr(opt, 'min_box_area', 100)
#     _OFF     = 1_000_000

#     postproc = FalconJDEPostProcessor(
#         num_classes=ncls,
#         num_top_queries=getattr(opt, 'K', 300),
#         conf_thres=opt.conf_thres,
#         use_focal_loss=True,
#     )

#     opt_trk = copy.copy(opt)
#     opt_trk.num_classes = ncls
#     tracker = MCJDETracker(opt_trk, frame_rate=getattr(opt, 'frame_rate', 30))

#     src = LoadCocoSequencesForTracking(val_ann_file, val_img_root, img_size=opt.img_size)

#     # ── Class names for HOTA (native class space, no remap) ─────────────────
#     # HOTA is computed per class then class-averaged (VisDrone convention).
#     try:
#         from falconmot.tracker.class_remap import CLS7_NAMES, CLS5_NAMES, CLS4_NAMES
#         _name_tables = {7: CLS7_NAMES, 5: CLS5_NAMES, 4: CLS4_NAMES}
#     except Exception:
#         _name_tables = {}
#     _names_map = _name_tables.get(ncls, {})
#     class_names = {c: _names_map.get(c, f'class_{c}') for c in range(ncls)}
#     hota_names  = [class_names[c] for c in range(ncls)]
#     coll = HOTACollector(hota_names)

#     # ── Parse GT ONCE (instead of re-parsing the whole json per sequence) ───
#     gt_by_seq = load_all_coco_gt(val_ann_file)

#     # ── fp16 autocast for the detector forward (wires up --track_val_fp16) ──
#     use_fp16 = bool(int(getattr(opt, 'track_val_fp16', 1))) and opt.device.type == 'cuda'

#     # ── Threaded frame prefetch params (overlap disk I/O with GPU) ──────────
#     n_threads  = int(getattr(opt, 'track_val_loader_threads', 4))
#     prefetch   = int(getattr(opt, 'track_val_prefetch', 8))

#     t0, n_frames = time.time(), 0

#     total_frames = sum(src.num_frames(s) for s in src.seqs)
#     pbar = tqdm(total=total_frames, desc='[track-eval]', unit='f', dynamic_ncols=True)

#     for seq_id in src.seqs:
#         tracker.reset()
#         gt_frames = gt_by_seq.get(seq_id, {})

#         frames = src._seq_frames[seq_id]
#         reader = _ParallelFrameReader(frames, net_w, net_h,
#                                       num_workers=n_threads, prefetch=prefetch)

#         for frame_id, img, img0 in reader:
#             orig_h, orig_w = img0.shape[:2]
#             sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
#             blob  = torch.from_numpy(img[None]).to(opt.device, non_blocking=True)

#             with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
#                 output = model(blob)
#                 res = postproc(output, sizes)[0]

#             dets = _defaultdict(list)
#             if len(res['scores']) > 0:
#                 bxs = res['boxes'].float().cpu().numpy()
#                 scs = res['scores'].float().cpu().numpy()
#                 lbs = res['labels'].cpu().numpy()
#                 rid = res['reid'].float().cpu().numpy() if 'reid' in res else None
#                 ws  = bxs[:, 2] - bxs[:, 0]
#                 hs  = bxs[:, 3] - bxs[:, 1]
#                 for i in np.where((ws > 0) & (hs > 0))[0]:
#                     c = int(lbs[i])
#                     if c < 0 or c >= ncls:
#                         continue
#                     tlwh = np.array([bxs[i, 0], bxs[i, 1], ws[i], hs[i]], dtype=np.float32)
#                     emb  = rid[i] if rid is not None else np.zeros(1, dtype=np.float32)
#                     dets[c].append(MCTrack(tlwh, float(scs[i]), emb, ncls, c))

#             online = tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

#             tlwhs, tids = [], []
#             for c, tracks in online.items():
#                 for t in tracks:
#                     w, h = t.curr_tlwh[2], t.curr_tlwh[3]
#                     if t.track_id < 0 or (w * h) <= min_area:
#                         continue
#                     tlwhs.append(t.curr_tlwh)
#                     tids.append(int(t.track_id) + c * _OFF)

#             _collect_hota_frame(coll, class_names, seq_id,
#                                 gt_frames.get(int(frame_id), []),
#                                 tlwhs, tids, _OFF)
#             n_frames += 1
#             pbar.update(1)
#             if n_frames % 20 == 0:
#                 pbar.set_postfix_str(f'{n_frames / max(1e-6, time.time() - t0):.1f} fps')

#     pbar.close()
#     infer_elapsed = time.time() - t0   # pure inference time (excludes HOTA compute)
#     model.train()

#     if n_frames == 0:
#         return {}

#     # ── HOTA (the only metric — drives model_best selection) ────────────────
#     hota_workers = int(getattr(opt, 'track_val_hota_workers', 0)) or min(8, os.cpu_count() or 4)
#     try:
#         hres = coll.compute(num_workers=hota_workers)
#     except Exception as e:
#         print(f'[track-eval] HOTA unavailable ({e}); install TrackEval to enable it. '
#               f'Skipping best-model update this round.')
#         return {'fps': n_frames / max(1e-6, infer_elapsed)}

#     ov  = hres['overall']
#     fps = n_frames / max(1e-6, infer_elapsed)
#     out = {k.lower(): float(ov.get(k, 0.0))
#            for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe')}
#     out['track_score'] = out['hota']     # best model selected by HOTA
#     out['fps'] = fps
#     return out


@torch.no_grad()
def run_reid_eval(model, val_loader, criterion, opt) -> dict:
    """Đánh giá ReID bằng cách tính ReID Loss trên tập Validation (Ảnh rời rạc).
    Model có ReID Loss thấp nhất sẽ được Save Best.
    """
    import time
    model.eval()
    criterion.eval()

    total_reid_loss = 0.0
    num_batches = 0

    t0 = time.time()
    for batch_i, batch in enumerate(val_loader):
        # Đưa data lên GPU
        batch = {k: v.to(opt.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Tạo Target y hệt lúc Train
        B = batch['input'].shape[0]
        targets = []
        for i in range(B):
            n = int(batch['detr_num_objs'][i].item())
            valid_labels = batch['detr_labels'][i, :n]
            valid_boxes = batch['detr_boxes'][i, :n]
            valid_tids = batch['detr_track_ids'][i, :n]
            keep = valid_labels >= 0
            targets.append({
                'labels':    valid_labels[keep],
                'boxes':     valid_boxes[keep],
                'track_ids': valid_tids[keep],
            })

        # Chạy Model
        outputs = model(batch['input'], targets)
        
        # Chỉ tính Loss (Bỏ qua Detection)
        loss_dict = criterion(outputs, targets, epoch=0, compute_det_loss=False)
        
        # Cộng dồn ReID Loss
        if 'loss_reid' in loss_dict:
            total_reid_loss += loss_dict['loss_reid'].item()
            num_batches += 1

    model.train()
    criterion.train()

    avg_reid_loss = (total_reid_loss / num_batches) if num_batches > 0 else float('inf')
    
    elapsed = time.time() - t0
    return {'val_reid_loss': avg_reid_loss, 'eval_time': elapsed}


@torch.inference_mode()
def run_track_eval(model, opt, val_ann_file: str, val_img_root: str) -> dict:
    """Tracking validation over val sequences — HOTA metrics only.
    Uses Plain Resize logic (no letterbox) for QAM Dense Map feeding.
    """
    import time
    from collections import defaultdict as _defaultdict

    if not (val_ann_file and val_img_root and os.path.isfile(val_ann_file)):
        print('[track-eval] missing val_ann/val_img -- skipping tracking eval.')
        return {}

    model.eval()
    
    # [QUAN TRỌNG 1]: Ép mô hình xuất Dense Map (emb_map) khi đang chạy eval()
    core_model = model.module if hasattr(model, 'module') else model
    core_model.return_reid_dense = True  

    net_w, net_h = opt.img_size
    ncls     = opt.num_classes
    min_area = getattr(opt, 'min_box_area', 100)
    _OFF     = 1_000_000

    postproc = FalconJDEPostProcessor(
        num_classes=ncls,
        num_top_queries=getattr(opt, 'K', 300),
        conf_thres=opt.conf_thres,
        use_focal_loss=True,
    )
    # KHÔNG gọi postproc.set_net_hw() vì ta dùng Plain Resize

    opt_trk = copy.copy(opt)
    opt_trk.num_classes = ncls
    tracker = MCJDETracker(opt_trk, frame_rate=getattr(opt, 'frame_rate', 30))

    src = LoadCocoSequencesForTracking(val_ann_file, val_img_root, img_size=opt.img_size)

    try:
        from falconmot.tracker.class_remap import CLS7_NAMES, CLS5_NAMES, CLS4_NAMES
        _name_tables = {7: CLS7_NAMES, 5: CLS5_NAMES, 4: CLS4_NAMES}
    except Exception:
        _name_tables = {}
    _names_map = _name_tables.get(ncls, {})
    class_names = {c: _names_map.get(c, f'class_{c}') for c in range(ncls)}
    hota_names  = [class_names[c] for c in range(ncls)]
    coll = HOTACollector(hota_names)

    gt_by_seq = load_all_coco_gt(val_ann_file)
    use_fp16 = bool(int(getattr(opt, 'track_val_fp16', 1))) and opt.device.type == 'cuda'

    n_threads  = int(getattr(opt, 'track_val_loader_threads', 4))
    prefetch   = int(getattr(opt, 'track_val_prefetch', 8))

    t0, n_frames = time.time(), 0
    total_frames = sum(src.num_frames(s) for s in src.seqs)
    pbar = tqdm(total=total_frames, desc='[track-eval]', unit='f', dynamic_ncols=True)

    for seq_id in src.seqs:
        tracker.reset()
        gt_frames = gt_by_seq.get(seq_id, {})

        frames = src._seq_frames[seq_id]
        reader = _ParallelFrameReader(frames, net_w, net_h,
                                      num_workers=n_threads, prefetch=prefetch)

        for frame_id, img, img0 in reader:
            orig_h, orig_w = img0.shape[:2]
            sizes = torch.tensor([[orig_h, orig_w]], device=opt.device)
            blob  = torch.from_numpy(img[None]).to(opt.device, non_blocking=True)

            with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
                output = model(blob)
                res = postproc(output, sizes)[0]
                
            tracker.set_image(img0)

            # ====================================================================
            # [QUAN TRỌNG 2]: Set Dense Map cho Tracker (Logic Plain Resize)
            # ====================================================================
            if 'reid_dense' in output and getattr(opt, 'use_appearance_motion', False):
                # Tỷ lệ scale độc lập theo 2 trục X và Y
                rx = net_w / orig_w
                ry = net_h / orig_h
                
                tracker.set_dense(
                    output['reid_dense'],  # Tensor [128, H/4, W/4]
                    stride=output['reid_dense_stride'],
                    ratio_x=rx, 
                    ratio_y=ry, 
                    pad_w=0.0, 
                    pad_h=0.0
                )
            else:
                tracker.set_dense(None, 1.0, 1.0, 1.0)
            # ====================================================================

            dets = _defaultdict(list)
            if len(res['scores']) > 0:
                bxs = res['boxes'].float().cpu().numpy()
                scs = res['scores'].float().cpu().numpy()
                lbs = res['labels'].cpu().numpy()
                rid = res['reid'].float().cpu().numpy() if 'reid' in res else None
                ws  = bxs[:, 2] - bxs[:, 0]
                hs  = bxs[:, 3] - bxs[:, 1]
                for i in np.where((ws > 0) & (hs > 0))[0]:
                    c = int(lbs[i])
                    if c < 0 or c >= ncls:
                        continue
                    tlwh = np.array([bxs[i, 0], bxs[i, 1], ws[i], hs[i]], dtype=np.float32)
                    emb  = rid[i] if rid is not None else np.zeros(1, dtype=np.float32)
                    dets[c].append(MCTrack(tlwh, float(scs[i]), emb, ncls, c))

            # Update tracker (Không truyền dense_map vào đây nữa, vì đã gọi set_dense ở trên)
            online = tracker.update(dets, h_orig=orig_h, w_orig=orig_w)

            tlwhs, tids = [], []
            for c, tracks in online.items():
                for t in tracks:
                    w, h = t.curr_tlwh[2], t.curr_tlwh[3]
                    if t.track_id < 0 or (w * h) <= min_area:
                        continue
                    tlwhs.append(t.curr_tlwh)
                    tids.append(int(t.track_id) + c * _OFF)

            _collect_hota_frame(coll, class_names, seq_id,
                                gt_frames.get(int(frame_id), []),
                                tlwhs, tids, _OFF)
            n_frames += 1
            pbar.update(1)
            if n_frames % 20 == 0:
                pbar.set_postfix_str(f'{n_frames / max(1e-6, time.time() - t0):.1f} fps')

    pbar.close()
    infer_elapsed = time.time() - t0   
    
    # [QUAN TRỌNG 3]: Trả mô hình về trạng thái cũ để không sinh thừa DenseMap khi train
    core_model.return_reid_dense = False
    model.train()

    if n_frames == 0:
        return {}

    hota_workers = int(getattr(opt, 'track_val_hota_workers', 0)) or min(8, os.cpu_count() or 4)
    try:
        hres = coll.compute(num_workers=hota_workers)
    except Exception as e:
        print(f'[track-eval] HOTA unavailable ({e}); install TrackEval to enable it. '
              f'Skipping best-model update this round.')
        return {'fps': n_frames / max(1e-6, infer_elapsed)}

    ov  = hres['overall']
    fps = n_frames / max(1e-6, infer_elapsed)
    out = {k.lower(): float(ov.get(k, 0.0))
           for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe')}
    out['track_score'] = out['hota']     # best model selected by HOTA
    out['fps'] = fps
    return out


def run(opt):
    _print_stage1_banner(opt)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset      = get_dataset(opt.dataset, opt.task)
    use_coco_fmt = (opt.dataset == 'coco')

    with open(opt.data_cfg) as f:
        data_config  = json.load(f)
    dataset_root = data_config['root']
    print("Dataset root: %s" % dataset_root)

    from falconmot.data.dataset import VisDroneCocoDataset

    if use_coco_fmt:
        train_sources = data_config.get('train_sources')
        if not train_sources and getattr(opt, 'merge_val_into_train', False):
            train_sources = [
                {'ann': data_config['train_ann'], 'img': data_config['train_img']},
                {'ann': data_config['val_ann'],   'img': data_config['val_img']},
            ]
            print('[data] merge_val_into_train: merging train + val into the train set')

        if train_sources:
            dataset = VisDroneCocoDataset(opt=opt, sources=train_sources, augment=True)
        else:
            dataset = VisDroneCocoDataset(
                opt=opt, img_root=data_config['train_img'],
                ann_file=data_config['train_ann'], augment=True)
    else:
        dataset = Dataset(opt=opt, root=dataset_root,
                          paths=data_config['train'], img_size=opt.input_wh,
                          augment=True, transforms=T.Compose([T.ToTensor()]))
    opt = opts().init()
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    print("opt.gpus_str: ", opt.gpus_str)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # ── Val dataset (optional) ──────────────────────────────────────────────
    val_loader   = None
    val_ann_file = ''
    val_img_root = ''

    if getattr(opt, 'merge_val_into_train', False):
        print('[data] merge_val_into_train=True -> skipping val_loader, no COCO eval.')
    elif getattr(opt, 'val_cfg', ''):
        with open(opt.val_cfg) as f:
            val_config = json.load(f)
        val_dataset = None

        if use_coco_fmt:
            val_ann_file = val_config.get('val_ann', '')
            val_img      = val_config.get('val_img', '')
            val_img_root = val_img
            if val_ann_file and val_img:
                val_dataset = VisDroneCocoDataset(
                    opt=opt, img_root=val_img, ann_file=val_ann_file, augment=False)
            else:
                print('[warn] val_cfg missing val_ann/val_img keys for COCO format.')
        else:
            val_root  = val_config.get('root', dataset_root)
            val_paths = val_config.get('val') or val_config.get('test') or []
            if val_paths:
                val_dataset = Dataset(
                    opt=opt, root=val_root, paths=val_paths,
                    img_size=opt.input_wh, augment=False,
                    transforms=T.Compose([T.ToTensor()]))
            else:
                print('[warn] val_cfg provided but no val/test paths found.')

        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=opt.num_workers > 0,
                prefetch_factor=2 if opt.num_workers > 0 else None,
            )
            print(f'Val dataset: {len(val_dataset)} images')

    print('Creating model...')
    model = create_model(opt.arch, opt)
    _summarize_model(model, opt)

    # ── Load checkpoint BEFORE applying freeze policy ───────────────────────
    start_epoch = 0
    if opt.load_model != '':
        loaded = load_model(model, opt.load_model, None, opt.resume, opt.lr, opt.lr_step)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if opt.resume:
            try:
                ckpt = torch.load(opt.load_model, map_location='cpu', weights_only=False)
                start_epoch = int(ckpt.get('epoch', 0))
            except Exception:
                start_epoch = 0

    # ========================================================================
    # ── Training stage policy (Stage 1 vs Stage 2) ──────────────────────────
    # ========================================================================
    det_only  = getattr(opt, 'train_single_det', False)
    reid_only = getattr(opt, 'train_stage2_mot', False)

    if det_only and reid_only:
        raise ValueError("Cannot set both --train_single_det and --train_reid_only!")

    if det_only:
        # STAGE 1: Detection only
        stage_mgr.apply_det_only(model)
    elif reid_only:
        # STAGE 2: Freeze Detection, Train ReID
        stage_mgr.apply_stage2_mot(model)
    else:
        # Fallback / Joint Training
        stage_mgr.apply_joint_training(model)

    optimizer = build_optimizer(model, _with_lr(opt, opt.lr))

    _nw = opt.num_workers
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=_nw > 0,
        prefetch_factor=2 if _nw > 0 else None,
    )
    steps_per_epoch = len(train_loader)

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    scheduler = stage_mgr.build_phase_scheduler(
        optimizer, opt, build_scheduler, steps_per_epoch, opt.num_epochs,
        warmup_iters=min(getattr(opt, 'warmup_iters', 2000), steps_per_epoch))

    if start_epoch > 0:
        scheduler.fast_forward(start_epoch * steps_per_epoch)

    best_mAP       = 0.0
    best_reid_loss = float('inf')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        # ── id_weight setup ────────────────────────────────────────────────
        if det_only:
            trainer.loss.id_weight = 0.0
        else:
            # Stage 2 (reid_only) or joint: use constant id_weight
            trainer.loss.id_weight = opt.id_weight
            logger.write('id_w {:.3f} | '.format(trainer.loss.id_weight))

        train_loader.dataset.set_epoch(epoch - 1)

        log_dict_train, _ = trainer.train(epoch, train_loader, scheduler=scheduler)

        cur_lr = optimizer.param_groups[0]['lr']
        logger.write('epoch: {} |'.format(epoch))
        logger.write('lr {:e} | '.format(cur_lr))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        # ── Periodic checkpoint ─────────────────────────────────────────────
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
                       epoch, model, optimizer)

        # ── COCO mAP eval (stage-1 only) ───────────────────────────────────
        if det_only and val_loader is not None and opt.val_intervals > 0 \
                and epoch % opt.val_intervals == 0:
            print(f'\n[Eval] epoch {epoch} — running COCO mAP (stage-1)...')
            metrics = run_coco_eval(model, val_loader, opt, ann_file=val_ann_file)

            log_line = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items())
            print(f'[Eval] {log_line}')
            logger.write(f'[eval] {log_line} | ')
            for k, v in metrics.items():
                logger.scalar_summary(f'val_{k}', v, epoch)

            cur_mAP = metrics.get('AP', 0.0)
            if cur_mAP > best_mAP:
                best_mAP = cur_mAP
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model, optimizer)
                print(f'[Eval] ★ New best AP={best_mAP:.4f} → model_best.pth')

        # ── ReID val-loss eval (stage-2 only) — drives best-model selection ──
        _rvi = getattr(opt, 'reid_val_intervals', 0) or opt.val_intervals
        do_reid = (
            (not det_only)
            and val_loader is not None
            and opt.val_intervals > 0
            and epoch % max(1, _rvi) == 0
        )
        if do_reid:
            print(f'\n[ReID-Eval] epoch {epoch} — running ReID validation loss...')
            rmetrics = run_reid_eval(model, val_loader, trainer.loss, opt)
            if rmetrics and 'val_reid_loss' in rmetrics:
                rline = (f"val_reid_loss {rmetrics['val_reid_loss']:.4f}  "
                         f"({rmetrics['eval_time']:.1f}s)")
                print(f'[ReID-Eval] {rline}')
                logger.write(f'[reid] {rline} | ')
                logger.scalar_summary('val_reid_loss', rmetrics['val_reid_loss'], epoch)

                if rmetrics['val_reid_loss'] < best_reid_loss:
                    best_reid_loss = rmetrics['val_reid_loss']
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                               epoch, model, optimizer)
                    print(f"[ReID-Eval] ★ New best val_reid_loss={best_reid_loss:.4f} "
                          f"→ model_best.pth")

        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print("opt.gpus: ", opt.gpus)
    print('epoch:', opt.num_epochs)
    run(opt)