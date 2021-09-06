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
        self.mode = cfg.get('lambd', 1.0)
        assert self.mode in ['incr', 'coupled'] or self._is_float(
            self.mode), f"adversarial lambda-mode has to be one of ['const', 'incr', (float)], but is '{self.mode}'"
        if self._is_float(self.mode):
            self.lambd = self.mode

        # keep track of loss and iteration to compute lambda
        self.prev_loss = 0.
        self.iter = 0

        # build layer
        self.classifier = nn.Sequential()
        # first fc-layer
        self.classifier.add_module(f'dcls_{self.feat}_fc0', nn.Linear(math.prod(in_shape), 128))
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

        # features have shape (N, F), where N is either batch size or number of regions. We combine them into a single tensor with shape (2*N, F)
        x_src = x_src.view(x_src.size(0), -1)
        x_tgt = x_tgt.view(x_tgt.size(0), -1)
        feats = torch.cat((x_src, x_tgt), dim=0)

        # set lambda (weight of gradient after reversal) according to config
        if self.mode == 'incr':
            p = float(self.iter) / 40 / 2
            self.lambd = 2. / (1. + np.exp(-10 * p)) - 1
            self.iter += 1
        elif self.mode == 'coupled':
            self.lambd = math.exp(-self.prev_loss)

        # apply gradient reverse layer and domain classifier
        out = self.classifier(GradReverse.apply(feats, self.lambd))

        # build classification targets of shape (2*N) with entries 0: source, 1: target
        target = torch.cat((torch.zeros(x_src.size(0)), torch.ones(x_tgt.size(0))), dim=0).long().cuda()

        # calculate loss
        loss = nn.NLLLoss()(out, target)

        if self.mode == 'coupled':
            self.prev_loss = loss

        return {f'loss_{self.feat}_adversarial': loss}

    def _is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False