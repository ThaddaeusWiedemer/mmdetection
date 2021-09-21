import warnings
import math
import numpy as np
import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base_adaptive import BaseDetectorAdaptive
from ..utils import AdversarialHead, GPAHead


@DETECTORS.register_module()
class TwoStageDetectorDA(BaseDetectorAdaptive):
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
        super(TwoStageDetectorDA, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, ' 'please use "init_cfg" instead')
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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # all domain adaptation is only relevant during training
        if self.train_cfg is not None:
            self.da_cfg = self.train_cfg.get('da', None)
            self.with_da = self.da_cfg is not None
            self.train_source = self.train_cfg.get('train_source', False)

            self.first_iter = True
            self.iter = 0
            self.prev_loss = dict()

            # TODO get backbone stages and neck output shape
            feat_shapes = dict()
            # feat_shapes.update([(f'backbone_{i}', [ch] for i, ch in enumerate([neck['in_channels']]))])
            neck_out = neck['out_channels']
            feat_shapes.update({
                'feat_neck_0': [neck_out, 200, 200],
                'feat_neck_1': [neck_out, 100, 100],
                'feat_neck_2': [neck_out, 50, 50],
                'feat_neck_3': [neck_out, 25, 25],
                'feat_neck_4': [neck_out, 13, 13]
            })

            roi_out_channels = roi_head['bbox_roi_extractor']['out_channels']
            roi_out_size = roi_head['bbox_roi_extractor']['roi_layer']['output_size']
            feat_shapes.update({'feat_roi': [roi_out_channels, roi_out_size, roi_out_size]})

            feat_shapes.update({'feat_rcnn_shared': [roi_head['bbox_head']['fc_out_channels'], 1, 1]})
            feat_shapes.update({'feat_rcnn_cls': [roi_head['bbox_head']['fc_out_channels'], 1, 1]})
            feat_shapes.update({'feat_rcnn_bbox': [roi_head['bbox_head']['fc_out_channels'], 1, 1]})

            # define all domain adaptation modules
            self.da_heads = nn.ModuleDict()  # use ModuleDict instead of dict to register all layers to the model
            for module in self.da_cfg:
                name = module.get('type', None)
                feat = module.get('feat', None)
                assert name is not None, 'a type must be specified for each domain adaptation module'
                assert feat is not None, f'domain adaptation module `{name}` did not specify input features'

                # GPA uses one layer to reduce feature dimension
                if name == 'gpa':
                    da_head = GPAHead(module, feat_shapes[feat])

                # adversarial domain adaptation needs a domain classifier
                elif name == 'adversarial':
                    da_head = AdversarialHead(module, feat_shapes[feat])

                self.da_heads.update({f'{feat}_{name}': da_head})

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

        # save all intermediary features as tuples (source, target) in this dict to potentially deploy domain adaptation on them
        feats = dict()
        feats.update({
            'img': (img, img_tgt),
            'gt_bboxes': (gt_bboxes, gt_bboxes_tgt),
            'gt_labels': (gt_labels, gt_labels_tgt)
        })

        # save all losses in this dict to balance later
        losses = dict()

        # extract features in both domains
        # TODO get backbone features before neck?
        x_src = self.extract_feat(img)
        x_tgt = self.extract_feat(img_tgt)
        for i, (_x_src, _x_tgt) in enumerate(zip(x_src, x_tgt)):
            # print(f'features in neck {i}: {_x_src.size()}')
            feats.update({f'feat_neck_{i}': (_x_src, _x_tgt)})

        # RPN forward and loss (class and bbox) in both domains
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses_src, proposals_src = self.rpn_head.forward_train(x_src,
                                                                        img_metas,
                                                                        gt_bboxes,
                                                                        gt_labels=None,
                                                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                                                        proposal_cfg=proposal_cfg)
            rpn_losses_tgt, proposals_tgt = self.rpn_head.forward_train(x_tgt,
                                                                        img_metas_tgt,
                                                                        gt_bboxes_tgt,
                                                                        gt_labels_tgt=None,
                                                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                                                        proposal_cfg=proposal_cfg)
            # save everything for domain adaptation
            feats.update({'proposals': (proposals_src, proposals_tgt)})
            if self.train_source:
                losses.update([(f'{k}_src', v) for k, v in rpn_losses_src.items()])
            losses.update([(f'{k}_tgt', v) for k, v in rpn_losses_tgt.items()])
        else:
            raise NotImplementedError('Two stage domain-adaptive detector only works with RPN for now')
            proposal_list = proposals

        # get sampled ROIs, features in head, class score, and losses (class and bbox) in both domains
        roi_losses_src, bbox_results_src = self.roi_head.forward_train(x_src, img_metas, proposals_src, gt_bboxes,
                                                                       gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        roi_losses_tgt, bbox_results_tgt = self.roi_head.forward_train(x_tgt, img_metas_tgt, proposals_tgt,
                                                                       gt_bboxes_tgt, gt_labels_tgt, gt_bboxes_ignore,
                                                                       gt_masks, **kwargs)
        # save everything for domain adaptation
        for key in bbox_results_src.keys():
            feats.update({key: (bbox_results_src[key], bbox_results_tgt[key])})
        if self.train_source:
            losses.update([(f'{k}_src', v) for k, v in roi_losses_src.items()])
        losses.update([(f'{k}_tgt', v) for k, v in roi_losses_tgt.items()])

        if self.with_da:
            # do domain adaptation
            losses_da = self._align_domains(feats)
            losses.update(losses_da)

            # balance losses
            losses = self._balance_losses(losses)

        return losses

    def _align_domains(self, feats):
        """Compute all domain adaptation losses as specified in config.

        Args:
            feats (dict[str, tuple[Tensor, Tensor]]): All input, features, and outputs of the current training step in
                the form {key, (value_source, value_target)}
        
        Returns:
            dict[str, Tensor]: all domain adaptation losses 
        """
        losses = dict()

        # call every module function and collect losses
        for module in self.da_cfg:
            losses.update(self.da_heads[f"{module['feat']}_{module['type']}"](feats))

        return losses

    def _balance_losses(self, losses):
        """Rebalance all losses.
        
        Args:
            losses (dict[str, Tensor]): network losses
            
        Returns:
            dict[str, Tensor]: re-weighted network losses
        """
        if self.first_iter:
            print('LOSSES:')
            print(losses)

        for module in self.da_cfg:
            name = module['type']
            feat = module['feat']
            for loss, weight in module.loss_weights.items():
                # ignore if weight = 1.0
                if weight == 1.0:
                    break
                # update all losses that match the 'location_type_loss' substring
                _losses = [(key, value * weight) for key, value in losses.items()
                           if f'{feat}_{name}_{loss}' in key.lower()]
                losses.update(_losses)

                if self.first_iter:
                    print(f'updated {feat}_{loss} by factor {weight}:')
                    print(_losses)

        self.first_iter = False
        return losses

    # def _adversarial(self, inputs, classifier, cfg):
    #     # get feature maps to align
    #     feat = cfg['feat']
    #     try:
    #         feat_src, feat_tgt = inputs[feat]
    #     except KeyError:
    #         print(f"`{feat} is not a valid input for an adaptation module")
    #     # features have shape (N, F), where N is either batch size or number of regions. We combine them into a single tensor with shape (2*N, F)
    #     feat_src = feat_src.view(feat_src.size(0), -1)
    #     feat_tgt = feat_tgt.view(feat_tgt.size(0), -1)
    #     feats = torch.cat((feat_src, feat_tgt), dim=0)

    #     # set lambda (weight of gradient after reversal) according to config
    #     def isfloat(value):
    #         try:
    #             float(value)
    #             return True
    #         except ValueError:
    #             return False

    #     mode = cfg.get('lambd', 1.0)
    #     if mode == 'incr':
    #         p = float(self.iter) / 40 / 2
    #         lambd = 2. / (1. + np.exp(-10 * p)) - 1
    #         self.iter += 1
    #     elif mode == 'coupled':
    #         lambd = math.exp(-self.prev_loss[feat])
    #     elif isfloat(mode):
    #         lambd = mode
    #     else:
    #         raise KeyError(f'adversarial lambda-mode has to be one of [`const`, `incr`, (float)], but is `{mode}`')

    #     # apply gradient reverse layer and domain classifier
    #     out = classifier(GradReverse.apply(feats, lambd))

    #     # build classification targets of shape (2*N) with entries 0: source, 1: target
    #     target = torch.cat((torch.zeros(feat_src.size(0)), torch.ones(feat_tgt.size(0))), dim=0).long().cuda()

    #     # calculate loss
    #     loss = nn.NLLLoss()(out, target)

    #     if mode == 'coupled':
    #         self.prev_loss[feat] = loss
    #     return {f'loss_{feat}_adversarial': loss}

    async def async_simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        return self.roi_head.onnx_export(x, proposals, img_metas)
