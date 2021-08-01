import warnings

import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage_adaptive import TwoStageDetectorAdaptive


@DETECTORS.register_module()
class TwoStageDetectorAdaptiveAutoBalance(TwoStageDetectorAdaptive):
    """Base class for two-stage detectors with domain adaptation.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
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
            self.da_weight_old = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.da_weight_roi_intra = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.da_weight_roi_inter = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.da_weight_rcnn_intra = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)
            self.da_weight_rcnn_inter = nn.Parameter(torch.ones(1) * 0.2, requires_grad=True)

    
    def forward_train(self,
                      img,
                      img_metas,
                      img_tgt,
                      img_metas_tgt,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_tgt=None,
                      gt_labels_tgt=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            img_tgt (Tensor): ``img`` for target domain

            img_metas_tgt (list[dict]): ``img_metas`` for target domain

            gt_bboxes_tgt (list[Tensor]): ``gt_bboxes`` for target domain

            gt_labels_tgt (list[Tensor]): ``gt_labels`` for target domain

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert gt_masks is None, 'domain adaptation cannot handle gt_masks as of now'

        batch_size = img.size(0)

        # extract features in both domains
        x_src = self.extract_feat(img)
        x_tgt = self.extract_feat(img_tgt)

        losses = dict()

        def coeff(a):
            return 1 / (2 * torch.pow(a, 2))

        def reg(a):
            return torch.log(1 + torch.pow(a, 2))

        # RPN forward and loss in both domains
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses_src, proposal_list_src = self.rpn_head.forward_train(
                x_src,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            rpn_losses_tgt, proposal_list_tgt = self.rpn_head.forward_train(
                x_tgt,
                img_metas_tgt,
                gt_bboxes_tgt,
                gt_labels_tgt=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            # we only want to improve on the target domain
            for key, value in rpn_losses_tgt.items():
                if isinstance(value, list):
                    losses.update({key: [v * coeff(self.da_weight_old) for v in value]})
                else:
                    losses.update({key: value * coeff(self.da_weight_old)}) # RPN_loss_cls + RPN_loss_bbox
        else:
            raise NotImplementedError('Two stage domain-adaptive detector only works with RPN for now')
            proposal_list = proposals

        # get ROI losses and sampled ROIs in both domains
        roi_losses_src, rois_src, feat_roi_src, feat_rcnn_src, cls_score_src = self.roi_head.forward_train(
            x_src, img_metas, proposal_list_src, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        roi_losses_tgt, rois_tgt, feat_roi_tgt, feat_rcnn_tgt, cls_score_tgt = self.roi_head.forward_train(
            x_tgt, img_metas_tgt, proposal_list_tgt, gt_bboxes_tgt, gt_labels_tgt, gt_bboxes_ignore, gt_masks, **kwargs)
        # we only want to improve on the target domain
        # losses.update(roi_losses_src) # RCNN_loss_cls + RCNN_loss_bbox
        for key, value in roi_losses_tgt.items():
            if isinstance(value, list):
                losses.update({key: [v * coeff(self.da_weight_old) for v in value]})
            else:
                losses.update({key: value * coeff(self.da_weight_old)}) # RCNN_loss_cls + RCNN_loss_bbox

        # adapting on features after ROI and after RCNN only makes a difference when the ROI-head has shared
        # layers
        if feat_roi_src.size() == feat_rcnn_src.size():
            if torch.eq(feat_roi_src, feat_rcnn_src).all():
                warnings.warn('The features for domain adaptation after ROI and RCNN are the same, the model might not be\
                    using a shared head')

        # feed all features used for domain adaptation through fc layer (2 different ones for ROI and RCNN features)
        # the dimensions of the features are
        #   ROI:  (samples_per_gpu * roi_sampler_num, roi_out_channels, roi_output_size, roi_output_size)
        #   RCNN: (samples_per_gpu * roi_sampler_num, head_fc_out_channels)
        feat_roi_src = self.da_fc_roi(feat_roi_src.flatten(1))
        feat_roi_tgt = self.da_fc_roi(feat_roi_tgt.flatten(1))
        feat_rcnn_src = self.da_fc_rcnn(feat_rcnn_src)
        feat_rcnn_tgt = self.da_fc_rcnn(feat_rcnn_tgt)

        # compute intra-class and inter-class loss after ROI and after RCNN
        da_weight = self.train_cfg.get('loss_weight_da', 1.0)
        da_weight_roi = self.train_cfg.get('loss_weight_da_roi', 1.0)
        da_weight_rcnn = self.train_cfg.get('loss_weight_da_rcnn', 1.0)
        da_weight_intra = self.train_cfg.get('loss_weight_da_intra', 1.0)
        da_weight_inter = self.train_cfg.get('loss_weight_da_inter', 1.0)

        roi_loss_intra, roi_loss_inter = self._gpa_loss(feat_roi_src, cls_score_src, rois_src, gt_bboxes, gt_labels, feat_roi_tgt, cls_score_tgt, rois_tgt, gt_bboxes_tgt, gt_labels_tgt, batch_size)
        losses.update({'roi_loss_intra': roi_loss_intra * da_weight * da_weight_roi * da_weight_intra * coeff(self.da_weight_roi_intra)})
        losses.update({'roi_loss_inter': roi_loss_inter * da_weight * da_weight_roi * da_weight_inter * coeff(self.da_weight_roi_inter)})

        rcnn_loss_intra, rcnn_loss_inter = self._gpa_loss(feat_rcnn_src, cls_score_src, rois_src, gt_bboxes, gt_labels, feat_rcnn_tgt, cls_score_tgt, rois_tgt, gt_bboxes_tgt, gt_labels_tgt, batch_size)
        losses.update({'rcnn_loss_intra': rcnn_loss_intra * da_weight * da_weight_rcnn * da_weight_intra * coeff(self.da_weight_rcnn_intra)})
        losses.update({'rcnn_loss_inter': rcnn_loss_inter * da_weight * da_weight_rcnn * da_weight_inter * coeff(self.da_weight_rcnn_inter)})

        coeff_reg = reg(self.da_weight_roi_intra) + reg(self.da_weight_roi_inter) + reg(self.da_weight_rcnn_intra) + reg(self.da_weight_rcnn_inter) + reg(self.da_weight_old)
        losses.update({'balance': coeff_reg})

        print('DEBUG INFO', end=': ')
        print('ROI intra:', roi_loss_intra.item(), end=', ')
        print('ROI inter:', roi_loss_inter.item(), end=', ')
        print('RCNN intra:', rcnn_loss_intra.item(), end=', ')
        print('RCNN inter:', rcnn_loss_inter.item(), end=', ')
        print('w old:', coeff(self.da_weight_old).item(), end=', ')
        print('w ROI intra:', coeff(self.da_weight_roi_intra).item(), end=', ')
        print('w ROI inter:', coeff(self.da_weight_roi_inter).item(), end=', ')
        print('w RCNN intra:', coeff(self.da_weight_rcnn_intra).item(), end=', ')
        print('w RCNN inter:', coeff(self.da_weight_rcnn_inter).item())

        return losses