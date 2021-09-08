from operator import ne
import warnings
import math
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
        self.lambd_mode = cfg.get('lambd', 1.0)
        assert self.lambd_mode in ['incr', 'coupled'] or self._is_float(
            self.lambd_mode
        ), f"adversarial lambda-mode has to be one of ['const', 'incr', (float)], but is '{self.lambd_mode}'"
        if self._is_float(self.lambd_mode):
            self.lambd = self.lambd_mode
        self.trafo = cfg.get('transform', 'none')
        self.mode = cfg.get('mode', 'none')
        self.n_sample = cfg.get('n_sample', 16)
        self.sample_shape = cfg.get('sample_shape', (7, 7))

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
            self.transform.add_module(f'dcls_{self.feat}_crop', RandomCrop(self.n_sample, self.sample_shape))
            self.classifier.add_module(f'dcls_{self.feat}_fc0',
                                       nn.Linear(in_shape[0] * math.prod(self.sample_shape), 128))

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

        # apply input transformation
        x_src = self.transform(x_src)
        x_tgt = self.transform(x_tgt)
        # if self.trafo == 'sample':
        #     # generate random crop indices
        #     dh, dw = self.sample_shape
        #     h_src = np.random.randint(0, x_src.size(2) - dh, self.n_sample)
        #     w_src = np.random.randint(0, x_src.size(3) - dw, self.n_sample)
        #     h_tgt = np.random.randint(0, x_tgt.size(2) - dh, self.n_sample)
        #     w_tgt = np.random.randint(0, x_tgt.size(3) - dw, self.n_sample)
        #     # collect crops
        #     crops_src = []
        #     crops_tgt = []
        #     for h_s, w_s, h_t, w_t in zip(h_src, w_src, h_tgt, w_tgt):
        #         crops_src.append(x_src[:, :, h_s:h_s + dh, w_s:w_s + dw])
        #         crops_tgt.append(x_tgt[:, :, h_t:h_t + dh, w_t:w_t + dw])
        #     # stack crops in batch-dimension
        #     x_src = torch.cat(crops_src, 0)
        #     x_tgt = torch.cat(crops_tgt, 0)
        # print('crops', crops_src[0].shape, crops_tgt[0].shape)
        # print('after', x_src.shape, x_tgt.shape)

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

        # apply gradient reverse layer and domain classifier
        out = self.classifier(GradReverse.apply(feats, self.lambd))

        # build classification targets of shape (2*N) with entries 0: source, 1: target
        target = torch.cat((torch.zeros(x_src.size(0)), torch.ones(x_tgt.size(0))), dim=0).long().cuda()

        # calculate loss
        loss = nn.NLLLoss()(out, target)

        if self.lambd_mode == 'coupled':
            self.prev_loss = loss

        return {f'loss_{self.feat}_adversarial': loss}

    def _is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False


class RandomCrop(BaseModule):
    def __init__(self, n, shape, concat_out=True, init_cfg=None):
        """Take `n` random crops of shape (h, w) from input tensor with shape (B, C, H, W).

        Args:
            n (int): number of random crops
            shape (tuple(int, int)): shape (h, w) of crops, must be smaller than input shape (H, W)
            concat_out (bool, optional): whether to concat outputs in dimension or return as list
        """
        super(RandomCrop, self).__init__(init_cfg=init_cfg)
        self.n = n
        self.shape = shape
        self.concat_out = concat_out

    def forward(self, x):
        # generate random crop indices
        dh, dw = self.shape
        h = np.random.randint(0, x.size(2) - dh, self.n)
        w = np.random.randint(0, x.size(3) - dw, self.n)

        # collect crops
        crops = []
        for _h, _w in zip(h, w):
            crops.append(x[:, :, _h:_h + dh, _w:_w + dw])

        # stack crops in batch-dimension
        if self.concat_out:
            x = torch.cat(crops, 0)
        else:
            x = crops

        return x