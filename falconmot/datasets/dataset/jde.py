import glob
import math
import os
import os.path as osp
import random
import copy
import time
import warnings

import cv2
import numpy as np
import torch

from collections import OrderedDict, defaultdict
from falconmot.utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
# id2cls = {
#     0: 'pedestrian', 1: 'people',   2: 'bicycle',        3: 'car',
#     4: 'van',        5: 'truck',    6: 'tricycle',        7: 'awning-tricycle',
#     8: 'bus',        9: 'motor',
# }
id2cls = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor'
}
from falconmot.datasets.augment import (
    augment_hsv, cxcywh_to_xyxy,
    sanitize_boxes,
    random_bias_crop,
    random_zoom_out,
    apply_appearance_augments,
    copy_paste_small_objects,
    mosaic_with_scale_bias,
    random_affine as _random_affine_amot,
)

# ImageNet mean/std (matching EdgeCrafter's Normalize op)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# for inference
class LoadImages:
    def __init__(self, path, img_size=(1088, 608)):
        """
        :param path:
        :param img_size:
        """
        self.frame_rate = 10  # no actual meaning here

        if type(path) == str:
            if os.path.isdir(path):
                image_format = ['.jpg', '.jpeg', '.png', '.tif']
                self.files = sorted(glob.glob('%s/*.*' % path))
                self.files = list(filter(lambda x: os.path.splitext(x)[
                                                       1].lower() in image_format, self.files))
            elif os.path.isfile(path):
                self.files = [path]
        elif type(path) == list:
            self.files = path

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if self.count == self.nF:
            raise StopIteration

        img_path = self.files[self.count]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # BGR → RGB, scale to [0, 1] — no ImageNet normalize (matches training)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return img_path, img, img_0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # BGR → RGB, scale to [0, 1] — no ImageNet normalize (matches training)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return img_path, img, img_0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self,
                 path,
                 img_size=(1088, 608)):
        """
        :param path:
        :param img_size:
        """
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080  # 设置(输出的分辨率)
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration

        # Read image
        res, img_0 = self.cap.read()  # BGR
        assert img_0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img_0 = cv2.resize(img_0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # BGR → RGB, scale to [0, 1] — no ImageNet normalize (matches training)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return self.count, img, img_0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self,
                 path,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None):
        """
        :param path:
        :param img_size:
        :param augment:
        :param transforms:
        """
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids')
                            .replace('.png', '.txt')
                            .replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files

        self.width = img_size[0]
        self.height = img_size[1]

        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, width=None, height=None):
        """
        图像数据格式转换, 增强; 标签格式化
        :param img_path:
        :param label_path:
        :param height:
        :param width:
        :return:
        """
        # 输入网络的图像分辨率
        if height is None or width is None:
            height = self.height
            width = self.width

        # 读取图片数据为numpy array格式, 3通道顺序为BGR
        img = cv2.imread(img_path)  # cv(numpy): BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, pad_w, pad_h = letterbox(img, height=height, width=width)  # resizing and padding

        # Load labels
        if os.path.isfile(label_path):
            with warnings.catch_warnings():  # No warnings for empty label file(txt)
                warnings.simplefilter("ignore")
                labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

                # reformat xywh to pixel xyxy(x1, y1, x2, y2) format
                labels = labels_0.copy()  # deep copy
                labels[:, 2] = ratio * w * (labels_0[:, 2] - labels_0[:, 4] / 2) + pad_w  # x1
                labels[:, 3] = ratio * h * (labels_0[:, 3] - labels_0[:, 5] / 2) + pad_h  # y1
                labels[:, 4] = ratio * w * (labels_0[:, 2] + labels_0[:, 4] / 2) + pad_w  # x2
                labels[:, 5] = ratio * h * (labels_0[:, 3] + labels_0[:, 5] / 2) + pad_h  # y2
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(img, labels,
                                           degrees=(-5, 5),
                                           translate=(0.10, 0.10),
                                           scale=(0.50, 1.20))

        plot_flag = False
        if plot_flag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [2, 4, 4, 2, 2]].T,
                     labels[:, [3, 3, 5, 5, 3]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        num_labels = len(labels)
        if num_labels > 0:
            # convert xyxy to xywh(center_x, center_y, b_w, b_h)
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())

            # normalize to 0~1
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if num_labels > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img,
              height=608,
              width=1088,
              color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded rectangular
    :param img:
    :param height:
    :param width:
    :param color:
    :return:
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) * 0.5  # width padding
    dh = (height - new_shape[1]) * 0.5  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    # Return the ACTUAL integer pixel offsets applied to the image (left, top),
    # not the fractional dw/dh — _letterbox_labels must use these exact values
    # to avoid up-to-0.5px bbox drift from rounding mismatch.
    return img, ratio, left, top


def random_affine(img, targets=None,
                  degrees=(-10, 10),
                  translate=(.1, .1),
                  scale=(.9, 1.1),
                  shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
        img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
              img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
              img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * \
                    (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)),
                            abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)




# ----------

class JointDataset(LoadImagesAndLabels):  # for training
    """
    joint detection and embedding dataset
    """
    mean = None
    std = None

    def __init__(self,
                 opt,
                 root,
                 paths,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None):
        """
        :param opt:
        :param root:
        :param paths:
        :param img_size:
        :param augment:
        :param transforms:
        """
        self.opt = opt
        # dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = len(opt.reid_cls_ids.split(','))  # C5: car, bicycle, person, cyclist, tricycle

        # make sure img_size equal to opt.input_wh
        if opt.input_wh[0] != img_size[0] or opt.input_wh[1] != img_size[1]:
            opt.input_wh[0], opt.input_wh[1] = img_size[0], img_size[1]

        # default input width and height
        self.default_input_wh = opt.input_wh

        # net input width and height
        self.width = self.default_input_wh[0]
        self.height = self.default_input_wh[1]

        # ----- generate img and label file path lists
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [x.replace('images', 'labels_with_ids')
                                    .replace('.png', '.txt')
                                    .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

            print('Total {} image files in {} dataset.'.format(len(self.label_files[ds]), ds))

        if opt.id_weight > 0:  # If do ReID calculation
            # @even: for MCMOT training
            for ds, label_paths in self.label_files.items():  # 每个子数据集
                max_ids_dict = defaultdict(int)  # cls_id => max track id

                # 子数据集中每个label
                for lp in label_paths:
                    if not os.path.isfile(lp):
                        print('[Warning]: invalid label file {}.'.format(lp))
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        lb = np.loadtxt(lp)
                        if len(lb) < 1:  # 空标签文件
                            continue

                        lb = lb.reshape(-1, 6)
                        for item in lb:  # label中每一个item(检测目标)
                            if item[1] > max_ids_dict[int(item[0])]:  # item[0]: cls_id, item[1]: track id
                                max_ids_dict[int(item[0])] = item[1]

                # track id number
                self.tid_num[ds] = max_ids_dict  # 每个子数据集按照需要reid的cls_id组织成dict

            # @even: for MCMOT training
            self.tid_start_idx_of_cls_ids = defaultdict(dict)
            last_idx_dict = defaultdict(int)  # 从0开始
            for k, v in self.tid_num.items():  # 统计每一个子数据集
                for cls_id, id_num in v.items():  # 统计这个子数据集的每一个类别, v是一个max_ids_dict
                    self.tid_start_idx_of_cls_ids[k][cls_id] = last_idx_dict[cls_id]
                    last_idx_dict[cls_id] += id_num

            # @even: for MCMOT training
            self.nID_dict = defaultdict(int)
            for k, v in last_idx_dict.items():
                self.nID_dict[k] = int(v)  # 每个类别的tack ids数量

        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        # ---- AMOT-style augmentation (optional epoch cutoff) ----
        self.cur_epoch    = 0
        stop_epoch        = getattr(opt, 'stop_epoch', -1)
        self.stop_epoch   = opt.num_epochs if stop_epoch < 0 else stop_epoch
        self.hsv_fraction = getattr(opt, 'hsv_fraction', 0.5)
        self.affine_degrees    = (-5, 5)
        self.affine_translate  = (0.10, 0.10)
        self.affine_scale      = (0.50, 1.20)

        # ---- AMOT-exact augmentation mode ----
        self.use_amot_aug = getattr(opt, 'amot_aug', False)

        # ---- Small-object augmentation flags ----
        self.use_copy_paste        = getattr(opt, 'copy_paste',            False)
        self.copy_paste_prob       = getattr(opt, 'copy_paste_prob',       0.5)
        self.copy_paste_max_area   = getattr(opt, 'copy_paste_max_area',   0.01)
        self.copy_paste_n          = getattr(opt, 'copy_paste_n',          5)

        self.use_mosaic            = getattr(opt, 'mosaic',                False)
        self.mosaic_prob           = getattr(opt, 'mosaic_prob',           0.5)
        self.mosaic_scale_bias_prob= getattr(opt, 'mosaic_scale_bias_prob',0.5)
        self.mosaic_scale_min      = getattr(opt, 'mosaic_scale_min',      0.3)
        self.mosaic_scale_max      = getattr(opt, 'mosaic_scale_max',      0.6)


        print('dataset summary')
        print(self.tid_num)

        if opt.id_weight > 0:  # If do ReID calculation
            # print('total # identities:', self.nID)
            for k, v in self.nID_dict.items():
                print('Total {:d} IDs of {}'.format(v, id2cls[k]))

            # print('start index', self.tid_start_index)
            for k, v in self.tid_start_idx_of_cls_ids.items():
                for cls_id, start_idx in v.items():
                    print('Start index of dataset {} class {:d} is {:d}'
                          .format(k, int(cls_id), int(start_idx)))

    # ------------------------------------------------------------------
    # Epoch-aware augmentation schedule (call once per epoch in train loop)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        """Set current epoch (0-indexed) for augmentation schedule."""
        self.cur_epoch = epoch

    def __len__(self):
        return self.nF   # rotation mode: same steps/epoch as single-chip

    # ------------------------------------------------------------------
    # Raw loader — no augmentation, returns cxcywh-norm labels
    # ------------------------------------------------------------------

    def _load_raw(self, img_path, label_path):
        """Load raw BGR image + normalized cxcywh labels. No resize, no aug."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'File corrupt {img_path}')

        labels = np.zeros((0, 6), dtype=np.float32)
        if os.path.isfile(label_path):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                raw = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
                if len(raw) > 0:
                    labels = raw   # [cls, tid, cx, cy, w, h] already normalized

        return img, labels

    # ------------------------------------------------------------------
    # Mosaic loader — picks 3 extra random images from the same dataset
    # ------------------------------------------------------------------

    def _load_mosaic(self, anchor_img, anchor_labels, ds):
        """Build a mosaic from anchor + 3 random images in dataset `ds`."""
        n = len(self.img_files[ds])
        extra_indices = [random.randint(0, n - 1) for _ in range(3)]
        imgs_labels = [(anchor_img, anchor_labels)]
        for ei in extra_indices:
            ei_img, ei_labels = self._load_raw(
                self.img_files[ds][ei],
                self.label_files[ds][ei],
            )
            imgs_labels.append((ei_img, ei_labels))
        random.shuffle(imgs_labels)  # randomize tile positions
        return mosaic_with_scale_bias(
            imgs_labels,
            output_w=self.width,
            output_h=self.height,
            scale_bias_prob=self.mosaic_scale_bias_prob,
            scale_min=self.mosaic_scale_min,
            scale_max=self.mosaic_scale_max,
        )

    # ------------------------------------------------------------------
    # Letterbox helper (adjusts labels for the added padding/scaling)
    # ------------------------------------------------------------------

    def _letterbox_labels(self, labels, orig_w, orig_h, ratio, pad_w, pad_h,
                          lbw=None, lbh=None):
        """Adjust cxcywh-norm labels from original image to letterboxed image."""
        if len(labels) == 0:
            return labels
        lbw = lbw if lbw is not None else self.width
        lbh = lbh if lbh is not None else self.height
        out = labels.copy()
        out[:, 2] = (labels[:, 2] * orig_w * ratio + pad_w) / lbw
        out[:, 3] = (labels[:, 3] * orig_h * ratio + pad_h) / lbh
        out[:, 4] = labels[:, 4] * orig_w * ratio / lbw
        out[:, 5] = labels[:, 5] * orig_h * ratio / lbh
        return out

    def _affine_labels(self, img, labels):
        """Random affine on letterboxed image; labels are cxcywh normalized."""
        if len(labels) == 0:
            img = random_affine(
                img, None,
                degrees=self.affine_degrees,
                translate=self.affine_translate,
                scale=self.affine_scale,
            )
            return img, labels

        targets = labels.copy()
        targets[:, 2:6] = cxcywh_to_xyxy(labels[:, 2:6], self.width, self.height)
        img, targets, _ = random_affine(
            img, targets,
            degrees=self.affine_degrees,
            translate=self.affine_translate,
            scale=self.affine_scale,
        )
        if len(targets) == 0:
            return img, targets

        out = targets.copy()
        out[:, 2:6] = xyxy2xywh(targets[:, 2:6].copy())
        out[:, 2] /= self.width
        out[:, 3] /= self.height
        out[:, 4] /= self.width
        out[:, 5] /= self.height
        return img, out

    # ------------------------------------------------------------------
    # AMOT-exact augmentation pipeline
    # Replicates exactly what AMOT/src/lib/datasets/dataset/jde.py::get_data does:
    #   1. HSV (S+V only, fraction=0.50) on raw image
    #   2. Letterbox → (width, height) gray fill 127.5
    #   3. Convert labels: cxcywh norm → pixel xyxy in letterboxed image
    #   4. random_affine(deg=(-5,5), trans=0.10, scale=(0.50,1.20), shear=(-2,2))
    #   5. Convert back: pixel xyxy → cxcywh normalized
    #   6. Horizontal flip (50%) — NO vertical flip
    #   7. BGR → RGB + ImageNet normalize
    # ------------------------------------------------------------------

    def _amot_aug_sample(self, img, labels):
        """Apply AMOT's exact augmentation pipeline.

        img    : BGR uint8 numpy array (raw, before any resize)
        labels : (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh

        Returns
        -------
        img_tensor : (3, H, W) float32 torch.Tensor, ImageNet normalized
        labels     : (M, 6) [cls, tid, cx, cy, w, h] normalized cxcywh
        orig_h     : int  original image height
        orig_w     : int  original image width
        """
        orig_h, orig_w = img.shape[:2]

        # 1. HSV augmentation (S+V only, fraction=0.50) — always when augmenting
        if self.augment:
            augment_hsv(img, fraction=0.50)   # in-place, BGR

        # 2. Letterbox to network size, gray fill (matches AMOT default 127.5)
        img, ratio, pad_w, pad_h = letterbox(img, height=self.height, width=self.width)

        # 3. Convert cxcywh normalized → pixel xyxy in the letterboxed image
        #    Mirrors AMOT get_data label conversion exactly.
        if len(labels) > 0:
            lbs = labels.copy()
            lbs_xyxy = labels.copy()
            lbs_xyxy[:, 2] = ratio * orig_w * (labels[:, 2] - labels[:, 4] * 0.5) + pad_w  # x1
            lbs_xyxy[:, 3] = ratio * orig_h * (labels[:, 3] - labels[:, 5] * 0.5) + pad_h  # y1
            lbs_xyxy[:, 4] = ratio * orig_w * (labels[:, 2] + labels[:, 4] * 0.5) + pad_w  # x2
            lbs_xyxy[:, 5] = ratio * orig_h * (labels[:, 3] + labels[:, 5] * 0.5) + pad_h  # y2
        else:
            lbs_xyxy = np.zeros((0, 6), dtype=np.float32)
            lbs      = lbs_xyxy

        # 4. Random affine (exact AMOT params)
        if self.augment:
            img, lbs_xyxy, _ = _random_affine_amot(
                img, lbs_xyxy,
                degrees=(-5, 5),
                translate=(0.10, 0.10),
                scale=(0.50, 1.20),
                shear=(-2, 2),
            )

        # 5. Convert pixel xyxy → cxcywh normalized (AMOT post-affine conversion)
        if len(lbs_xyxy) > 0:
            out = lbs_xyxy.copy()
            out[:, 2] = (lbs_xyxy[:, 2] + lbs_xyxy[:, 4]) * 0.5 / self.width   # cx
            out[:, 3] = (lbs_xyxy[:, 3] + lbs_xyxy[:, 5]) * 0.5 / self.height  # cy
            out[:, 4] = (lbs_xyxy[:, 4] - lbs_xyxy[:, 2]) / self.width          # w
            out[:, 5] = (lbs_xyxy[:, 5] - lbs_xyxy[:, 3]) / self.height         # h
            labels = out
        else:
            labels = lbs_xyxy   # empty (0, 6)

        # 6. Horizontal flip (50%) — AMOT does NOT do vertical flip
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            if len(labels) > 0:
                labels[:, 2] = 1.0 - labels[:, 2]

        # 7. BGR → RGB, ImageNet normalize (same as extended pipeline)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        return img, labels, orig_h, orig_w

    # ------------------------------------------------------------------
    # Main __getitem__:
    # letterbox → PhotometricDistort → ZoomOut → IoUCrop → sanitize
    #           → letterbox(re-fit) → flip → sanitize
    # Spatial augs run on the SMALL letterboxed image (~1088×608) not the raw
    # large image, so intermediate arrays are 10-20× smaller → fast data loading.
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        for i, c in enumerate(self.cds):
            if idx >= c:
                ds          = list(self.label_files.keys())[i]
                start_index = c
        img_path   = self.img_files[ds][idx - start_index]
        label_path = self.label_files[ds][idx - start_index]

        img, labels = self._load_raw(img_path, label_path)

        do_aug = self.augment and self.cur_epoch < self.stop_epoch

        # ----------------------------------------------------------------
        # [1] Copy-Paste small objects (on raw image, before any resize)
        # ----------------------------------------------------------------
        if do_aug and self.use_copy_paste:
            img, labels = copy_paste_small_objects(
                img, labels,
                max_area=self.copy_paste_max_area,
                max_paste=self.copy_paste_n,
                p=self.copy_paste_prob,
            )

        # ----------------------------------------------------------------
        # [2] Mosaic OR single-image spatial augmentation
        #     Mosaic already outputs at (self.width, self.height) so we
        #     skip letterbox for that path.
        # ----------------------------------------------------------------
        mosaic_used = False
        if do_aug and self.use_mosaic and random.random() < self.mosaic_prob:
            img, labels = self._load_mosaic(img, labels, ds)
            img, labels = self._affine_labels(img, labels)
            labels = sanitize_boxes(labels, self.width, self.height)
            mosaic_used = True
        else:
            # ---- [2b] Standard spatial aug (bias_crop / zoom_out / full-scene) ----
            # Probs tuned from VisDrone stats: all scales ≤55% make 100% objects
            # detectable (≥16px), so full-scene path is now 40% to expose model
            # to real density (median 41 objects/frame, max 148).
            if do_aug:
                _r = random.random()
                if _r < 0.40:
                    img, labels = random_bias_crop(img, labels, p=1.0)
                elif _r < 0.60:
                    img, labels = random_zoom_out(img, labels, max_scale=2.0, p=1.0)
                # else 40%: full scene — model sees full-density distribution

        # ----------------------------------------------------------------
        # Letterbox to network size (skipped when mosaic already sized).
        # ----------------------------------------------------------------
        if mosaic_used:
            pre_lb_h, pre_lb_w = self.height, self.width
            ratio, pad_w, pad_h = 1.0, 0, 0
        else:
            pre_lb_h, pre_lb_w = img.shape[:2]
            img, ratio, pad_w, pad_h = letterbox(img, height=self.height, width=self.width)
            labels = self._letterbox_labels(labels, pre_lb_w, pre_lb_h, ratio, pad_w, pad_h)

        if do_aug:
            img = apply_appearance_augments(img)

            labels = sanitize_boxes(labels, self.width, self.height)

            if random.random() > 0.5:
                img = np.fliplr(img)
                if len(labels) > 0:
                    labels[:, 2] = 1 - labels[:, 2]

            # vertical flip — valid for UAV top-down view
            if random.random() > 0.5:
                img = np.flipud(img)
                if len(labels) > 0:
                    labels[:, 3] = 1 - labels[:, 3]

            labels = sanitize_boxes(labels, self.width, self.height)

        # ---- store pre-letterbox size for COCO eval coordinate conversion ----
        orig_h_for_eval = pre_lb_h
        orig_w_for_eval = pre_lb_w

        # ---- BGR → RGB, normalize (ImageNet mean/std) ----
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # ---- remap track IDs to global offsets (single, authoritative remap) ----
        if self.opt.id_weight > 0 and len(labels) > 0:
            for i in range(len(labels)):
                if labels[i, 1] > -1:
                    row_ds    = ds
                    cls_id    = int(labels[i][0])
                    start_idx = self.tid_start_idx_of_cls_ids[row_ds].get(cls_id, 0)
                    labels[i, 1] += start_idx
                    # Sanity check: remapped ID must be within the known pool
                    assert labels[i, 1] <= self.nID_dict.get(cls_id, 0), (
                        f"Remapped track id {int(labels[i, 1])} exceeds "
                        f"nID_dict[{cls_id}]={self.nID_dict.get(cls_id, 0)} "
                        f"(row_ds={row_ds}, raw_tid={int(labels[i,1]) - start_idx}, "
                        f"start_idx={start_idx})"
                    )

        # ---- pack DETR-format targets ----
        num_objs       = min(len(labels), self.max_objs)
        detr_boxes     = np.zeros((self.max_objs, 4), dtype=np.float32)
        detr_labels    = np.full((self.max_objs,), -1, dtype=np.int64)
        detr_track_ids = np.full((self.max_objs,), -1, dtype=np.int64)

        for k in range(num_objs):
            lb = labels[k]
            detr_boxes[k]     = lb[2:6]
            detr_labels[k]    = int(lb[0])
            detr_track_ids[k] = int(lb[1]) - 1   # 1-indexed → 0-indexed

        return {
            'input':          img,
            'detr_boxes':     detr_boxes,
            'detr_labels':    detr_labels,
            'detr_track_ids': detr_track_ids,
            'detr_num_objs':  np.array(num_objs, dtype=np.int64),
            'orig_hw':        np.array([orig_h_for_eval, orig_w_for_eval], dtype=np.int64),
        }

