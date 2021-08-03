import warnings

import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage_adaptive import TwoStageDetectorAdaptive


@DETECTORS.register_module()
class TwoStageDetectorAdaptiveAutoBalance(TwoStageDetectorAdaptive):
    """Two-stage detectors with domain adaptation using the Method in
    [Liebel and Körner, “Auxiliary Tasks in Multi-Task Learning.”] to automatically balance multi-task losses.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetectorAdaptiveAutoBalance, self).__init__(backbone, neck=neck, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg)
        if roi_head is not None:
            # learnable parameters for loss-balancing
            self.coeff_faster_rcnn = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.coeff_roi_intra = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.coeff_roi_inter = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.coeff_rcnn_intra = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.coeff_rcnn_inter = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)

    def _loss_coeff(self, a):
        """The actual coefficient of each loss is 1/(2·param²), where param is the learnable parameter."""
        return 1 / (2 * torch.pow(a, 2))

    def _loss_penalty(self, a):
        """The regularization term for each coefficient is ln(1 + param²), where param is the learnable parameter."""
        return torch.log(1 + torch.pow(a, 2))

    def _gpa_balance_losses(self, roi_loss_intra, roi_loss_inter, rcnn_loss_intra, rcnn_loss_inter):
        roi_intra = self.gpa_cfg.get('loss_roi_intra', 1.0)
        roi_inter = self.gpa_cfg.get('loss_roi_inter', 1.0)
        rcnn_intra = self.gpa_cfg.get('loss_rcnn_intra', 1.0)
        rcnn_inter = self.gpa_cfg.get('loss_rcnn_inter', 1.0)

        losses = {}

        losses.update({'roi_loss_intra': roi_loss_intra * roi_intra * self._loss_coeff(self.coeff_roi_intra)})
        losses.update({'roi_loss_inter': roi_loss_inter * roi_inter * self._loss_coeff(self.coeff_roi_inter)})

        losses.update({'rcnn_loss_intra': rcnn_loss_intra * rcnn_intra * self._loss_coeff(self.coeff_rcnn_intra)})
        losses.update({'rcnn_loss_inter': rcnn_loss_inter * rcnn_inter * self._loss_coeff(self.coeff_rcnn_inter)})

        penalty = self._loss_penalty(self.coeff_roi_intra) + self._loss_penalty(self.coeff_roi_inter)\
             + self._loss_penalty(self.coeff_rcnn_intra) + self._loss_penalty(self.coeff_rcnn_inter)\
             + self._loss_penalty(self.coeff_faster_rcnn)
        losses.update({'penalty': penalty})

        print('DEBUG INFO:', \
            f'ROI intra: {roi_loss_intra.item()},', \
            f'ROI inter: {roi_loss_inter.item()},', \
            f'RCNN intra: {rcnn_loss_intra.item()},', \
            f'RCNN inter: {rcnn_loss_inter.item()},', \
            f'w old: {self._loss_coeff(self.coeff_faster_rcnn).item()},', \
            f'w ROI intra: {self._loss_coeff(self.coeff_roi_intra).item()},', \
            f'w ROI inter: {self._loss_coeff(self.coeff_roi_inter).item()},', \
            f'w RCNN intra: {self._loss_coeff(self.coeff_rcnn_intra).item()},', \
            f'w RCNN inter: {self._loss_coeff(self.coeff_rcnn_inter).item()}')

        return losses
    
    def _balance_losses(self, losses):
        _losses = {}
        for key, value in losses.items():
            if isinstance(value, list):
                _losses.update({key: [v * self._loss_coeff(self.coeff_faster_rcnn) for v in value]})
            else:
                _losses.update({key: value * self._loss_coeff(self.coeff_faster_rcnn)})
        return _losses