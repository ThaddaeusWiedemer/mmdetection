import warnings

import torch
import torch.nn as nn
import numpy as np

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage_adaptive import TwoStageDetectorAdaptive
from ..utils import GradReverse


@DETECTORS.register_module()
class TwoStageDetectorAdaptiveAdversarial(TwoStageDetectorAdaptive):
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
        super(TwoStageDetectorAdaptiveAdversarial, self).__init__(backbone,
                                                                  neck=neck,
                                                                  rpn_head=rpn_head,
                                                                  roi_head=roi_head,
                                                                  train_cfg=train_cfg,
                                                                  test_cfg=test_cfg,
                                                                  pretrained=pretrained,
                                                                  init_cfg=init_cfg)

        # self.grad_reverse = GradReverse(1.0)

        self.domain_classifier_roi = nn.Sequential()
        self.domain_classifier_roi.add_module('d_fc1', nn.Linear(128, 64))
        self.domain_classifier_roi.add_module('d_bn1', nn.BatchNorm1d(64))
        self.domain_classifier_roi.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier_roi.add_module('d_fc2', nn.Linear(64, 2))
        self.domain_classifier_roi.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier_rcnn = nn.Sequential()
        self.domain_classifier_rcnn.add_module('d_fc1', nn.Linear(64, 32))
        self.domain_classifier_rcnn.add_module('d_bn1', nn.BatchNorm1d(32))
        self.domain_classifier_rcnn.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier_rcnn.add_module('d_fc2', nn.Linear(32, 2))
        self.domain_classifier_rcnn.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.iter = 0

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

        # RPN forward and loss in both domains
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses_src, proposal_list_src = self.rpn_head.forward_train(x_src,
                                                                            img_metas,
                                                                            gt_bboxes,
                                                                            gt_labels=None,
                                                                            gt_bboxes_ignore=gt_bboxes_ignore,
                                                                            proposal_cfg=proposal_cfg)
            rpn_losses_tgt, proposal_list_tgt = self.rpn_head.forward_train(x_tgt,
                                                                            img_metas_tgt,
                                                                            gt_bboxes_tgt,
                                                                            gt_labels_tgt=None,
                                                                            gt_bboxes_ignore=gt_bboxes_ignore,
                                                                            proposal_cfg=proposal_cfg)
            # we only want to improve on the target domain
            losses.update(self._balance_losses(rpn_losses_tgt))  # RPN_loss_cls + RPN_loss_bbox
        else:
            raise NotImplementedError('Two stage domain-adaptive detector only works with RPN for now')
            proposal_list = proposals

        # get ROI losses and sampled ROIs in both domains
        roi_losses_src, rois_src, feat_roi_src, feat_rcnn_src, cls_score_src = self.roi_head.forward_train(
            x_src, img_metas, proposal_list_src, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        roi_losses_tgt, rois_tgt, feat_roi_tgt, feat_rcnn_tgt, cls_score_tgt = self.roi_head.forward_train(
            x_tgt, img_metas_tgt, proposal_list_tgt, gt_bboxes_tgt, gt_labels_tgt, gt_bboxes_ignore, gt_masks, **kwargs)
        # we only want to improve on the target domain
        losses.update(self._balance_losses(roi_losses_tgt))  # RCNN_loss_cls + RCNN_loss_bbox

        # adapting on features after ROI and after RCNN only makes a difference when the ROI-head has shared
        # layers
        if feat_roi_src.size() == feat_rcnn_src.size() and torch.eq(feat_roi_src, feat_rcnn_src).all():
            warnings.warn('The features for domain adaptation after ROI and RCNN are the same, the model might not be\
                using a shared head')

        # GPA
        if self.gpa_cfg is not None:
            # feed all features used for domain adaptation through fc layer (2 different ones for ROI and RCNN features)
            # the dimensions of the features are
            #   ROI:  (samples_per_gpu * roi_sampler_num, roi_out_channels, roi_output_size, roi_output_size)
            #   RCNN: (samples_per_gpu * roi_sampler_num, head_fc_out_channels)
            if 'fc_layer' in self.gpa_layer:
                # interpret 'fc_layer' as fc-layer for both, but 'fc_layer_roi' as fc-layer only for ROI and vice-versa
                if 'rcnn' in self.gpa_layer:
                    feat_roi_src = feat_roi_src.flatten(1)
                    feat_roi_tgt = feat_roi_tgt.flatten(1)
                else:
                    feat_roi_src = self.gpa_layer_roi(feat_roi_src.flatten(1))
                    feat_roi_tgt = self.gpa_layer_roi(feat_roi_tgt.flatten(1))
                if 'roi' in self.gpa_layer:
                    feat_rcnn_src = feat_rcnn_src
                    feat_rcnn_tgt = feat_rcnn_tgt
                else:
                    feat_rcnn_src = self.gpa_layer_rcnn(feat_rcnn_src)
                    feat_rcnn_tgt = self.gpa_layer_rcnn(feat_rcnn_tgt)
            elif self.gpa_layer in ['avgpool', 'maxpool']:
                feat_roi_src = self.gpa_layer_roi(feat_roi_src.flatten(1).unsqueeze(1)).squeeze(1)
                feat_roi_tgt = self.gpa_layer_roi(feat_roi_tgt.flatten(1).unsqueeze(1)).squeeze(1)
                feat_rcnn_src = self.gpa_layer_rcnn(feat_rcnn_src.unsqueeze(1)).squeeze(1)
                feat_rcnn_tgt = self.gpa_layer_rcnn(feat_rcnn_tgt.unsqueeze(1)).squeeze(1)
            elif self.gpa_layer == 'none':
                feat_roi_src = feat_roi_src.flatten(1)
                feat_roi_tgt = feat_roi_tgt.flatten(1)
                feat_rcnn_src = feat_rcnn_src
                feat_rcnn_tgt = feat_rcnn_tgt

            # compute intra-class and inter-class loss after ROI and RCNN
            roi_loss = self._gpa_loss(feat_roi_src, cls_score_src, rois_src, gt_bboxes, gt_labels, feat_roi_tgt,
                                      cls_score_tgt, rois_tgt, gt_bboxes_tgt, gt_labels_tgt, batch_size,
                                      self.domain_classifier_roi)
            rcnn_loss = self._gpa_loss(feat_rcnn_src, cls_score_src, rois_src, gt_bboxes, gt_labels, feat_rcnn_tgt,
                                       cls_score_tgt, rois_tgt, gt_bboxes_tgt, gt_labels_tgt, batch_size,
                                       self.domain_classifier_rcnn)

            gpa_losses = self._gpa_balance_losses(roi_loss, rcnn_loss)
            losses.update(gpa_losses)

        return losses

    def _gpa_loss(self,
                  feat,
                  cls_prob,
                  rois,
                  gt_bboxes,
                  gt_labels,
                  feat_tgt,
                  cls_prob_tgt,
                  rois_tgt,
                  gt_bboxes_tgt,
                  gt_labels_tgt,
                  batch_size,
                  classifier,
                  epsilon=1e-6):
        """Graph-based prototpye aggregation using adversaril loss instead of inter- and intra-class loss.
        """
        use_graph = self.gpa_cfg.get('use_graph', True)
        normalize = self.gpa_cfg.get('normalize', False)

        # view inputs as (batch_size, roi_sampler_num, )
        # with dimensions (batch_size, roi_sampler_num, num_feat)
        feat = feat.view(batch_size, feat.size(0) // batch_size, feat.size(1))
        feat_tgt = feat_tgt.view(batch_size, feat_tgt.size(0) // batch_size, feat_tgt.size(1))

        # get the class probability of every class for source and target domains
        # with dimensions (batch_size, roi_sampler_num, num_feat)
        cls_prob = cls_prob.view(batch_size, cls_prob.size(0) // batch_size, cls_prob.size(1))
        cls_prob_tgt = cls_prob_tgt.view(batch_size, cls_prob_tgt.size(0) // batch_size, cls_prob_tgt.size(1))

        # view rois as (batch_size, roi_sampler_num, 5)
        rois = rois.view(batch_size, rois.size(0) // batch_size, rois.size(1))
        rois_tgt = rois_tgt.view(batch_size, rois_tgt.size(0) // batch_size, rois_tgt.size(1))

        num_classes = cls_prob.size(2)
        class_ptt = list()
        tgt_class_ptt = list()

        for i in range(num_classes):
            tmp_cls_prob = cls_prob[:, :, i].view(cls_prob.size(0), cls_prob.size(1), 1)
            tmp_class_feat = feat * tmp_cls_prob
            tmp_feat = list()
            tmp_weight = list()

            for j in range(batch_size):
                tmp_batch_feat_ = tmp_class_feat[j, :, :]
                tmp_batch_weight_ = tmp_cls_prob[j, :, :]

                # graph-based aggregation
                if use_graph:
                    tmp_batch_adj = self._gpa_get_adj(rois[j, :, :])
                    tmp_batch_feat = torch.mm(tmp_batch_adj, tmp_batch_feat_)
                    tmp_batch_weight = torch.mm(tmp_batch_adj, tmp_batch_weight_)

                    # divide by sum of edge weights as in paper
                    if normalize:
                        weight_sum = tmp_batch_adj.sum(dim=1).unsqueeze(1)
                        tmp_batch_feat = torch.div(tmp_batch_feat, weight_sum)
                        tmp_batch_weight = torch.div(tmp_batch_weight, weight_sum)

                    tmp_feat.append(tmp_batch_feat)
                    tmp_weight.append(tmp_batch_weight)
                else:
                    tmp_feat.append(tmp_batch_feat_)
                    tmp_weight.append(tmp_batch_weight_)

            tmp_class_feat_ = torch.stack(tmp_feat, dim=0)
            tmp_class_weight = torch.stack(tmp_weight, dim=0)
            tmp_class_feat = torch.sum(torch.sum(tmp_class_feat_, dim=1),
                                       dim=0) / (torch.sum(tmp_class_weight) + epsilon)
            class_ptt.append(tmp_class_feat)

            tmp_tgt_cls_prob = cls_prob_tgt[:, :, i].view(cls_prob_tgt.size(0), cls_prob_tgt.size(1), 1)
            tmp_tgt_class_ptt = feat_tgt * tmp_tgt_cls_prob
            tmp_tgt_feat = list()
            tmp_tgt_weight = list()

            for j in range(batch_size):
                tmp_tgt_batch_feat_ = tmp_tgt_class_ptt[j, :, :]
                tmp_tgt_batch_weight_ = tmp_tgt_cls_prob[j, :, :]

                if use_graph:
                    # graph-based aggregation
                    tmp_tgt_batch_adj = self._gpa_get_adj(rois_tgt[j, :, :])
                    tmp_tgt_batch_feat = torch.mm(tmp_tgt_batch_adj, tmp_tgt_batch_feat_)
                    tmp_tgt_batch_weight = torch.mm(tmp_tgt_batch_adj, tmp_tgt_batch_weight_)

                    # divide by sum of edge weights as in paper
                    if normalize:
                        weight_sum = tmp_tgt_batch_adj.sum(dim=1).unsqueeze(1)
                        tmp_tgt_batch_feat = torch.div(tmp_tgt_batch_feat, weight_sum)
                        tmp_tgt_batch_weight = torch.div(tmp_tgt_batch_weight, weight_sum)

                    tmp_tgt_feat.append(tmp_tgt_batch_feat)
                    tmp_tgt_weight.append(tmp_tgt_batch_weight)
                else:
                    tmp_tgt_feat.append(tmp_tgt_batch_feat_)
                    tmp_tgt_weight.append(tmp_tgt_batch_weight_)

            # build prototypes from all samples in batch. results doesn't have batch dimension
            tmp_tgt_class_ptt_ = torch.stack(tmp_tgt_feat, dim=0)
            tmp_tgt_class_weight = torch.stack(tmp_tgt_weight, dim=0)
            tmp_tgt_class_ptt = torch.sum(torch.sum(tmp_tgt_class_ptt_, dim=1),
                                          dim=0) / (torch.sum(tmp_tgt_class_weight) + epsilon)
            tgt_class_ptt.append(tmp_tgt_class_ptt)

        class_ptt = torch.stack(class_ptt, dim=0)
        tgt_class_ptt = torch.stack(tgt_class_ptt, dim=0)

        # source/target prototypes have shape (C, F). We combine them into a single tensor with shape (2*C, F)
        ptts = torch.cat((class_ptt, tgt_class_ptt), dim=0)

        # calculate lambda
        # this is just a crude implementation for 100 epochs with batch-size 16
        # p = float(self.iter) / 100 / 17
        # lambd = 2. / (1. + np.exp(-10 * p)) - 1
        # self.iter += 1
        lambd = 1.0

        # put the prototypes through the gradient reverse layer and the domain classifier
        res = classifier(GradReverse.apply(ptts, lambd))

        # target should have shape (2*C) and have entries (0, 0, 0, ..., 0, 1, ..., 1, 1, 1), i.e. 0 for source and 1
        # for target domain
        target = torch.cat((torch.zeros(num_classes), torch.ones(num_classes)), dim=0).long().cuda()

        # just build negative log-likelihood loss
        loss = nn.NLLLoss()(res, target)

        # for i in range(num_classes):
        #     tmp_src_feat_1 = class_ptt[i, :]
        #     tmp_tgt_feat_1 = tgt_class_ptt[i, :]

        #     # intra-class loss is just distance of features
        #     loss_intra = loss_intra + self._gpa_distance(tmp_src_feat_1, tmp_tgt_feat_1)

        #     # inter-class loss is distance between all 4 source-target pairs
        #     for j in range(i + 1, num_classes):
        #         tmp_src_feat_2 = class_ptt[j, :]
        #         tmp_tgt_feat_2 = tgt_class_ptt[j, :]

        #         loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_src_feat_1, tmp_src_feat_2)
        #         loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_tgt_feat_1, tmp_tgt_feat_2)
        #         loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_src_feat_1, tmp_tgt_feat_2)
        #         loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_tgt_feat_1, tmp_src_feat_2)

        # # normalize losses
        # loss_intra = loss_intra / class_ptt.size(0)
        # loss_inter = loss_inter / (class_ptt.size(0) * (class_ptt.size(0) - 1) * 2)

        return loss

    def _gpa_balance_losses(self, roi_loss, rcnn_loss):
        """Combine the GPA losses with their individual weights"""
        roi = self.gpa_cfg.get('loss_roi_intra', 1.0)
        rcnn = self.gpa_cfg.get('loss_rcnn_intra', 1.0)

        losses = {}

        losses.update({'roi_loss': roi_loss * roi})

        losses.update({'rcnn_loss': rcnn_loss * rcnn})

        return losses