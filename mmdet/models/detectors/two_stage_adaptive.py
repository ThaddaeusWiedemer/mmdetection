import warnings

import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base_adaptive import BaseDetectorAdaptive


@DETECTORS.register_module()
class TwoStageDetectorAdaptive(BaseDetectorAdaptive):
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
        super(TwoStageDetectorAdaptive, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

            # fc layers for domain adaptation
            roi_out_size = roi_head['bbox_roi_extractor']['roi_layer']['output_size']
            roi_out_size *= roi_out_size
            roi_out_size *= roi_head['bbox_roi_extractor']['out_channels']
            self.da_fc_roi = nn.Linear(roi_out_size, 128)
            self.da_fc_rcnn = nn.Linear(2048, 64)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def _gpa_loss(self,
                         feat,
                         cls_prob,
                         rois,
                         feat_tgt,
                         cls_prob_tgt,
                         rois_tgt,
                         batch_size,
                         margin=1,
                         epsilon=1e-6):
        """Graph-based prototpye daptation loss as in https://github.com/ChrisAllenMing/GPA-detection.
        
        """
        # TODO check if this is working correctly for only 1 class
        def distance(feat_a, feat_b):
            """use this to compute distances between features for this loss"""
            return torch.pow(feat_a - feat_b, 2.0).mean()

        def get_adj(rois, epsilon=epsilon):
            """use this to calculate adjacency matrix of region proposals based on IoU
            
            Arguments:
                rois (Tensor): of shape (num_rois, 5), where the second dimension contains the values
                    in the [batch_idx, x_min, y_min, x_max, y_max] format
            """
            # compute the area of every bbox
            area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
            area = area + (area == 0).float() * epsilon

            # compute iou
            x_min = rois[:,1]
            x_min_copy = torch.stack([x_min] * rois.size(0), dim=0)
            x_min_copy_ = x_min_copy.permute((1,0))
            x_min_matrix = torch.max(torch.stack([x_min_copy, x_min_copy_], dim=-1), dim=-1)[0]
            x_max = rois[:,3]
            x_max_copy = torch.stack([x_max] * rois.size(0), dim=0)
            x_max_copy_ = x_max_copy.permute((1, 0))
            x_max_matrix = torch.min(torch.stack([x_max_copy, x_max_copy_], dim=-1), dim=-1)[0]
            y_min = rois[:,2]
            y_min_copy = torch.stack([y_min] * rois.size(0), dim=0)
            y_min_copy_ = y_min_copy.permute((1, 0))
            y_min_matrix = torch.max(torch.stack([y_min_copy, y_min_copy_], dim=-1), dim=-1)[0]
            y_max = rois[:,4]
            y_max_copy = torch.stack([y_max] * rois.size(0), dim=0)
            y_max_copy_ = y_max_copy.permute((1, 0))
            y_max_matrix = torch.min(torch.stack([y_max_copy, y_max_copy_], dim=-1), dim=-1)[0]
            
            w = torch.max(torch.stack([(x_max_matrix - x_min_matrix), torch.zeros_like(x_min_matrix)], dim = -1), dim = -1)[0]
            h = torch.max(torch.stack([(y_max_matrix - y_min_matrix), torch.zeros_like(y_min_matrix)], dim = -1), dim = -1)[0]
            intersection = w * h
            area_copy = torch.stack([area] * rois.size(0), dim = 0)
            area_copy_ = area_copy.permute((1,0))
            area_sum = area_copy + area_copy_
            union = area_sum - intersection
            iou = intersection / union

            return iou

        def inter_class_loss(feat_a, feat_b):
            # could use F.relu to write more concisely
            out = torch.pow((margin - torch.sqrt(distance(feat_a, feat_b))) / margin, 2) \
                * torch.pow(torch.max(margin - torch.sqrt(distance(feat_a, feat_b)), torch.tensor(0).float().cuda()), 2.0)
            return out

        # get the feature embedding of every class for source and target domains with GCN
        feat = feat.view(batch_size, feat.size(0) // batch_size, feat.size(1))
        feat_tgt = feat_tgt.view(batch_size, feat_tgt.size(0) // batch_size, feat_tgt.size(1))

        # get the class probability of every class for source and target domains
        cls_prob = cls_prob.view(batch_size, cls_prob.size(0) // batch_size, cls_prob.size(1))
        cls_prob_tgt = cls_prob_tgt.view(batch_size, cls_prob_tgt.size(0) // batch_size, cls_prob_tgt.size(1))

        num_classes = cls_prob.size(2)
        class_feat = list()
        tgt_class_feat = list()

        for i in range(num_classes):
            tmp_cls_prob = cls_prob[:, :, i].view(cls_prob.size(0), cls_prob.size(1), 1)
            tmp_class_feat = feat * tmp_cls_prob
            tmp_feat = list()
            tmp_weight = list()

            for j in range(batch_size):
                tmp_batch_feat_ = tmp_class_feat[j, :, :]
                tmp_batch_weight_ = tmp_cls_prob[j, :, :]
                tmp_batch_adj = get_adj(rois[j, :, :])

                # graph-based aggregation
                tmp_batch_feat = torch.mm(tmp_batch_adj, tmp_batch_feat_)
                tmp_batch_weight = torch.mm(tmp_batch_adj, tmp_batch_weight_)

                tmp_feat.append(tmp_batch_feat)
                tmp_weight.append(tmp_batch_weight)

            tmp_class_feat_ = torch.stack(tmp_feat, dim = 0)
            tmp_class_weight = torch.stack(tmp_weight, dim = 0)
            tmp_class_feat = torch.sum(torch.sum(tmp_class_feat_, dim=1), dim = 0) / (torch.sum(tmp_class_weight) + epsilon)
            class_feat.append(tmp_class_feat)

            tmp_tgt_cls_prob = cls_prob_tgt[:, :, i].view(cls_prob_tgt.size(0), cls_prob_tgt.size(1), 1)
            tmp_tgt_class_feat = feat_tgt * tmp_tgt_cls_prob
            tmp_tgt_feat = list()
            tmp_tgt_weight = list()

            for j in range(batch_size):
                tmp_tgt_batch_feat_ = tmp_tgt_class_feat[j, :, :]
                tmp_tgt_batch_weight_ = tmp_tgt_cls_prob[j, :, :]
                tmp_tgt_batch_adj = get_adj(rois_tgt[j, :, :])

                # graph-based aggregation
                tmp_tgt_batch_feat = torch.mm(tmp_tgt_batch_adj, tmp_tgt_batch_feat_)
                tmp_tgt_batch_weight = torch.mm(tmp_tgt_batch_adj, tmp_tgt_batch_weight_)

                tmp_tgt_feat.append(tmp_tgt_batch_feat)
                tmp_tgt_weight.append(tmp_tgt_batch_weight)

            tmp_tgt_class_feat_ = torch.stack(tmp_tgt_feat, dim = 0)
            tmp_tgt_class_weight = torch.stack(tmp_tgt_weight, dim = 0)
            tmp_tgt_class_feat = torch.sum(torch.sum(tmp_tgt_class_feat_, dim=1), dim = 0) / (torch.sum(tmp_tgt_class_weight) + epsilon)
            tgt_class_feat.append(tmp_tgt_class_feat)

        class_feat = torch.stack(class_feat, dim = 0)
        tgt_class_feat = torch.stack(tgt_class_feat, dim = 0)

        # get the intra-class and inter-class adaptation loss
        loss_intra = 0
        loss_inter = 0

        # TODO replace class_feat.size(0) with num_classes for better readability
        for i in range(class_feat.size(0)):
            tmp_src_feat_1 = class_feat[i, :]
            tmp_tgt_feat_1 = tgt_class_feat[i, :]

            # intra-class loss is just distance of features
            loss_intra = loss_intra + distance(tmp_src_feat_1, tmp_tgt_feat_1)

            # inter-class loss is distance between all 4 source-target pairs
            for j in range(i+1, class_feat.size(0)):
                tmp_src_feat_2 = class_feat[j, :]
                tmp_tgt_feat_2 = tgt_class_feat[j, :]

                loss_inter = loss_inter + inter_class_loss(tmp_src_feat_1, tmp_src_feat_2)
                loss_inter = loss_inter + inter_class_loss(tmp_tgt_feat_1, tmp_tgt_feat_2)
                loss_inter = loss_inter + inter_class_loss(tmp_src_feat_1, tmp_tgt_feat_2)
                loss_inter = loss_inter + inter_class_loss(tmp_tgt_feat_1, tmp_src_feat_2)

        # normalize losses
        loss_intra = loss_intra / class_feat.size(0)
        loss_inter = loss_inter / (class_feat.size(0) * (class_feat.size(0) - 1) * 2)

        return loss_intra, loss_inter
    
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
            losses.update(rpn_losses_tgt) # RPN_loss_cls + RPN_loss_bbox
        else:
            raise NotImplementedError('Two stage domain-adaptive detector only works with RPN for now')
            proposal_list = proposals

        # get ROI losses and sampled ROIs in both domains
        roi_losses_src, rois_src, feat_roi_src, feat_rcnn_src, cls_score_src = self.roi_head.forward_train(
            x_src, img_metas, proposal_list_src, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        roi_losses_tgt, rois_tgt, feat_roi_tgt, feat_rcnn_tgt, cls_score_tgt = self.roi_head.forward_train(
            x_tgt, img_metas_tgt, proposal_list_tgt, gt_bboxes_tgt, gt_labels_tgt, gt_bboxes_ignore, gt_masks, **kwargs)
        # we only want to improve on the target domain
        losses.update(roi_losses_tgt) # RCNN_loss_cls + RCNN_loss_bbox

        # adapting on features after ROI and after RCNN only makes a difference when the ROI-head has shared
        # layers
        assert not torch.eq(feat_roi_src, feat_rcnn_src).all(), 'The ROI-head has no shared layers!'

        # feed all features used for domain adaptation through fc layer (2 different ones for ROI and RCNN features)
        feat_roi_src = self.da_fc_roi(feat_roi_src)
        feat_roi_tgt = self.da_fc_roi(feat_roi_tgt)
        feat_rcnn_src = self.da_fc_rcnn(feat_rcnn_src)
        feat_rcnn_tgt = self.da_fc_rcnn(feat_rcnn_tgt)

        # compute intra-class and inter-class loss after ROI and after RCNN
        da_weight = self.train_cfg.get('loss_weight_da', 1.0)
        da_weight_rpn = self.train_cfg.get('loss_weight_da_rpn', 1.0)
        
        roi_loss_intra, roi_loss_inter = self._gpa_loss(feat_roi_src, cls_score_src, rois_src, feat_roi_tgt, cls_score_tgt, rois_tgt, batch_size)
        losses.update({'roi_loss_intra': roi_loss_intra * da_weight_rpn})
        losses.update({'roi_loss_inter': roi_loss_inter * da_weight_rpn})

        rcnn_loss_intra, rcnn_loss_inter = self._gpa_loss(feat_rcnn_src, cls_score_src, rois_src, feat_rcnn_tgt, cls_score_tgt, rois_tgt, batch_size)
        losses.update({'rcnn_loss_intra': rcnn_loss_intra * da_weight})
        losses.update({'rcnn_loss_inter': rcnn_loss_inter * da_weight})

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        return self.roi_head.onnx_export(x, proposals, img_metas)
