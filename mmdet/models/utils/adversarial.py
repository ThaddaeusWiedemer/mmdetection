from operator import ne
from typing_extensions import get_type_hints
import warnings
import math
import random
import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from . import GradReverse


class AdversarialHead(BaseModule):
    """Adversarial domain adaptation module with domain classifier and gradient reverse layer.
    """
    def __init__(self, cfg, in_shape, init_cfg=None):
        super(AdversarialHead, self).__init__(init_cfg)

        # get config info
        self.feat = cfg.get('feat', 'roi')
        self.tag = cfg.get('tag', None)  # to allow for multiple heads on same features
        self.tag = f'_{self.tag}' if self.tag is not None else ''
        self.lambd_mode = cfg.get('lambd', 'coupled')
        assert self.lambd_mode in ['incr', 'coupled'] or self._is_float(
            self.lambd_mode
        ), f"adversarial lambda-mode has to be one of ['const', 'incr', (float)], but is '{self.lambd_mode}'"
        if self._is_float(self.lambd_mode):
            self.lambd = self.lambd_mode
        self.lambd_weight = cfg.get('lambd_weight', 1)
        self.trafo = cfg.get('transform', 'none')
        self.mode = cfg.get('mode', 'none')
        self.n_sample = cfg.get('n_sample', 16)
        self.sample_shape = cfg.get('sample_shape', (7, 7))
        if isinstance(self.sample_shape, int):
            self.sample_shape = (self.sample_shape, self.sample_shape)
        self.only_fg = cfg.get('only_fg', False)
        self.schedule = cfg.get('schedule', None)
        self.gt_iou_thrs = cfg.get('gt_iou_thrs', (.9, .1))

        # keep track of loss and iteration to compute lambda
        self.prev_loss = 0.
        self.iter = 0

        # build layer
        self.classifier = nn.Sequential()
        self.transform = nn.Sequential()

        # input shape differs depending on mode
        if self.mode == 'channel':
            in_shape[0] = 1

        # first fc-layer depends on input transformation
        if self.trafo == 'none':
            # assume constant input size in both domains
            warnings.warn(
                f"Transform set to 'none' for adversarial adaptation head on {self.feat}. Make sure input features are of constant size, e.g. through setting resize.keep_ratio=True"
            )
            self.classifier.add_module(f'dcls_{self.feat}_fc0', nn.Linear(math.prod(in_shape), 128))
        elif self.trafo == 'pool2d':
            # pool inputs of any size (a, b) to (7, 7)
            # h = math.ceil(in_shape[1] / 2)
            # w = math.ceil(in_shape[2] / 2)
            h, w = 7, 7
            self.transform.add_module(f'dcls_{self.feat}_pool', nn.AdaptiveAvgPool2d((h, w)))
            self.classifier.add_module(f'dcls_{self.feat}_fc0', nn.Linear(in_shape[0] * h * w, 128))
        elif self.trafo == 'conv':
            # one conve layer and global average pooling
            self.transform.add_module(f'dcls_{self.feat}_pool', nn.Conv2d(in_shape[0], in_shape[0], 7))
            self.transform.add_module(f'dcls_{self.feat}_pool', nn.AdaptiveAvgPool2d((1, 1)))
            self.classifier.add_module(f'dcls_{self.feat}_fc0', nn.Linear(in_shape[0], 128))
        elif self.trafo == 'sample':
            # take crops from feature map
            self.transform.add_module(
                f'dcls_{self.feat}_crop',
                RandomCrop(self.n_sample, self.sample_shape, feat=self.feat, thrs=self.gt_iou_thrs))
            self.classifier.add_module(f'dcls_{self.feat}_fc0',
                                       nn.Linear(in_shape[0] * math.prod(self.sample_shape), 128))
        elif self.trafo == 'sample_gt':
            # take crops from feature map and label them as foreground or background trough ground-truth information
            self.transform.add_module(
                f'dcls_{self.feat}_crop',
                RandomCrop(self.n_sample, self.sample_shape, use_gt=True, feat=self.feat, thrs=self.gt_iou_thrs))
            self.classifier.add_module(f'dcls_{self.feat}_fc0',
                                       nn.Linear(in_shape[0] * math.prod(self.sample_shape), 128))
            if not self.only_fg:
                self.classifier_bg = nn.Sequential()
                self.classifier_bg.add_module(f'dcls_{self.feat}_fc0_bg',
                                              nn.Linear(in_shape[0] * math.prod(self.sample_shape), 128))
                self.classifier_bg.add_module(f'dcls_{self.feat}_bn0_bg', nn.BatchNorm1d(128))
                self.classifier_bg.add_module(f'dcls_{self.feat}_relu0_bg', nn.ReLU(True))
                # second fc-layer
                self.classifier_bg.add_module(f'dcls_{self.feat}_fc1_bg', nn.Linear(128, 32))
                self.classifier_bg.add_module(f'dcls_{self.feat}_bn1_bg', nn.BatchNorm1d(32))
                self.classifier_bg.add_module(f'dcls_{self.feat}_relu1_bg', nn.ReLU(True))
                # output
                self.classifier_bg.add_module(f'dcls_{self.feat}_fc2_bg', nn.Linear(32, 2))
                self.classifier_bg.add_module(f'dcls_{self.feat}_softmax_bg', nn.LogSoftmax(dim=1))

        self.classifier.add_module(f'dcls_{self.feat}_bn0', nn.BatchNorm1d(128))
        self.classifier.add_module(f'dcls_{self.feat}_relu0', nn.ReLU(True))
        # second fc-layer
        self.classifier.add_module(f'dcls_{self.feat}_fc1', nn.Linear(128, 32))
        self.classifier.add_module(f'dcls_{self.feat}_bn1', nn.BatchNorm1d(32))
        self.classifier.add_module(f'dcls_{self.feat}_relu1', nn.ReLU(True))
        # output
        self.classifier.add_module(f'dcls_{self.feat}_fc2', nn.Linear(32, 2))
        self.classifier.add_module(f'dcls_{self.feat}_softmax', nn.LogSoftmax(dim=1))

    def forward(self, inputs):
        # get feature maps to align
        try:
            x_src, x_tgt = inputs[self.feat]
        except KeyError:
            print(f"'{self.feat}' is not a valid input for an adaptation module")

        # apply input transformation -> gradient-reverse layer -> domain classifier
        if self.trafo == 'sample_gt':
            # get ground-truths as list(Tensor): (B, (N, 4))
            gt_src, gt_tgt = inputs['gt_bboxes']

            # get image metas
            img_metas_src, img_metas_tgt = inputs['img_metas']

            x_src, x_src_bg = self.transform((x_src, gt_src, img_metas_src))
            x_tgt, x_tgt_bg = self.transform((x_tgt, gt_tgt, img_metas_tgt))

            loss = self._classify_domain(x_src, x_tgt)
            if not self.only_fg:
                loss += self._classify_domain(x_src_bg, x_tgt_bg, background=True)
                loss /= 2

        else:
            x_src = self.transform(x_src)
            x_tgt = self.transform(x_tgt)

            loss = self._classify_domain(x_src, x_tgt)

        if self.lambd_mode == 'coupled':
            self.prev_loss = loss

        return {f'loss_{self.feat}_adversarial{self.tag}': loss}

    def _classify_domain(self, x_src, x_tgt, background=False):
        # flatten inputs for fc-layers
        # features have size (N, C, F) where N is batch-size or number of regions and C is channels
        if self.mode == 'channel':
            # fold channels in batch-dimension as (N*C, F)
            x_src = x_src.view(x_src.size(0) * x_src.size(1), -1).flatten(1)
            x_tgt = x_tgt.view(x_tgt.size(0) * x_tgt.size(1), -1).flatten(1)
        else:
            # view as (N, F)
            x_src = x_src.flatten(1)
            x_tgt = x_tgt.flatten(1)

        # features have shape (N, F), where N is either batch size or number of regions. We combine them into a single tensor with shape (2*N, F)
        feats = torch.cat((x_src, x_tgt), dim=0)

        # set lambda (weight of gradient after reversal) according to config
        if self.lambd_mode == 'incr':
            p = float(self.iter) / 40 / 2
            self.lambd = 2. / (1. + np.exp(-10 * p)) - 1
            self.iter += 1
        elif self.lambd_mode == 'coupled':
            self.lambd = math.exp(-self.prev_loss)
        # apply weight to lambda independent of mode
        self.lambd *= self.lambd_weight

        if self.schedule is not None:
            if self.iter % self.schedule[0] in self.schedule[1]:
                self.lambd = 0

        # apply gradient reverse layer and domain classifier
        if background:
            out = self.classifier_bg(GradReverse.apply(feats, self.lambd))
        else:
            out = self.classifier(GradReverse.apply(feats, self.lambd))

        # build classification targets of shape (2*N) with entries 0: source, 1: target
        target = torch.cat((torch.zeros(x_src.size(0)), torch.ones(x_tgt.size(0))), dim=0).long().cuda()

        # calculate loss
        loss = nn.NLLLoss()(out, target)

        return loss

    def _is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False


class RandomCrop(BaseModule):
    def __init__(self, n, shape, concat_out=True, init_cfg=None, use_gt=False, feat=None, thrs=None):
        """Take `n` random crops of shape (h, w) from input tensor with shape (B, C, H, W).

        Args:
            n (int): number of random crops
            shape (tuple(int, int)): shape (h, w) of crops, must be smaller than input shape (H, W)
            concat_out (bool, optional): whether to concat outputs in dimension or return as list, defaults to True
            use_gt (bool, optional): whether to use ground-truth information to sort cropped samples in foreground and
                background, defaults to False

        Returns:
            Tensor: of shape (B×n, C, shape[0], shape[1]) or (G, B×n, C, shape[0], shape[1]), where G is forground/
                background, B is batch size, and C is number of channels
        """
        super(RandomCrop, self).__init__(init_cfg=init_cfg)
        self.n = n
        assert isinstance(shape, tuple), 'shape must be a tuple of (int, int)'
        self.shape = shape
        self.concat_out = concat_out
        self.use_gt = use_gt
        self.feat = feat
        self.thrs = thrs

    def forward(self, x):
        if self.use_gt:
            x, gts, img_metas = x
            x = self._collect_crops_gt(x, gts, img_metas)
        else:
            x = self._collect_crops(x)

        return x

    def _collect_crops(self, x):
        # we can just generate all random crop indices up-front
        w, h = self.shape
        crop_x = np.random.randint(0, x.size(2) - w, self.n)
        crop_y = np.random.randint(0, x.size(3) - h, self.n)

        # collect crops
        crops = []

        for _x, _y in zip(crop_x, crop_y):
            crops.append(x[:, :, _x:_x + w, _y:_y + h])

        # stack crops in batch-dimension
        if self.concat_out:
            x = torch.cat(crops, 0)
        else:
            x = crops

        return x

    def _collect_crops_gt(self, x, gts, img_metas):
        w, h = self.shape

        # find x, y coordinates of ground-truths (using their center point)
        gts_x, gts_y = [], []
        for gts_batch, meta in zip(gts, img_metas):
            # ground-truth center point in image coordinates
            x_img = (gts_batch[:, 2] + gts_batch[:, 0]) / 2
            y_img = (gts_batch[:, 3] + gts_batch[:, 1]) / 2
            # ground-truth center point relative to image size
            h_img, w_img, _ = meta['img_shape']
            x_rel = x_img / h_img
            y_rel = y_img / w_img
            # ground-truth center point on current feature map
            gts_x.append(x_rel * x.size(2))
            gts_y.append(y_rel * x.size(3))

        # collect crops
        # we can't generate indices up-front, since indices must
        crops = []
        crops_bg = []

        for i in range(x.size(0)):
            crops_batch = []
            crops_bg_batch = []

            run_limit = 10000
            run = 0
            while len(crops_batch) < self.n and run < run_limit:
                run += 1
                _x = random.randint(0, x.size(2) - w)
                _y = random.randint(0, x.size(3) - h)
                # for any ground-truth: |gx - cx| <= d * w && |gy - cy| <= d * w
                # where gx, gy, cx, cy are the coordinates of ground-truth and crop center points and d is distance
                if torch.logical_and(torch.le(torch.abs(gts_x[i] - (_x + w / 2)), self.thrs[1] * w),
                                     torch.le(torch.abs(gts_y[i] - (_y + h / 2)), self.thrs[1] * h)).any():
                    crops_batch.append(x[i, :, _x:_x + w, _y:_y + h])
            # print('ran', run, 'times')

            run = 0
            while len(crops_bg_batch) < self.n and run < run_limit:
                run += 1
                _x = random.randint(0, x.size(2) - w)
                _y = random.randint(0, x.size(3) - h)
                # for no ground-truth: |gx - cx| <= d * w && |gy - cy| <= d * w
                if torch.logical_not(
                        torch.logical_and(torch.le(torch.abs(gts_x[i] - (_x + w / 2)), self.thrs[0] * w),
                                          torch.le(torch.abs(gts_y[i] - (_y + h / 2)), self.thrs[0] * h))).all():
                    crops_bg_batch.append(x[i, :, _x:_x + w, _y:_y + h])
            # print('ran', run, 'times')

            crops.extend(crops_batch)
            crops_bg.extend(crops_bg_batch)

        # stack crops in batch-dimension
        if self.concat_out:
            # in case no foreground/background crops were found withing the run limit, pass on a 0-dimensional tensor
            if crops and crops_bg:
                x = (torch.stack(crops, 0), torch.stack(crops_bg, 0))
            elif not crops:
                warnings.warn(f'-- no foreground crops found on {self.feat} in this iteration --')
                x = (torch.empty((0, x.size(1), w, h)).cuda(), torch.stack(crops_bg, 0))
            elif not crops_bg:
                warnings.warn(f'-- no background crops found on {self.feat} in this iteration --')
                x = (torch.stack(crops, 0), torch.empty((0, x.size(1), w, h)).cuda())
        else:
            x = (crops, crops_bg)

        return x