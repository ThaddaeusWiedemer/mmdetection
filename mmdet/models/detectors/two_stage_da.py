import warnings
import math
import numpy as np
import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base_adaptive import BaseDetectorAdaptive
from ..utils import GradReverse


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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg is not None:
            self.da_cfg = train_cfg.get('da', None)
            self.with_da = self.da_cfg is not None

        self.first_iter = True
        self.iter = 0
        self.prev_loss = dict()

        # TODO get backbone stages and neck output shape
        feat_shapes = dict()
        # feat_shapes.update([(f'backbone_{i}', [ch] for i, ch in enumerate([neck['in_channels']]))])
        neck_out = neck['out_channels']
        feat_shapes.update({
            'neck_0': [neck_out, 200, 200],
            'neck_1': [neck_out, 100, 100],
            'neck_2': [neck_out, 50, 50],
            'neck_3': [neck_out, 25, 25],
            'neck_4': [neck_out, 13, 13]
        })

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

            roi_out_channels = roi_head['bbox_roi_extractor']['out_channels']
            roi_out_size = roi_head['bbox_roi_extractor']['roi_layer']['output_size']
            feat_shapes.update({'roi': [roi_out_channels, roi_out_size, roi_out_size]})

            feat_shapes.update({'rcnn': [roi_head['bbox_head']['fc_out_channels']]})

        # define all layers for the domain adaptation modules
        self.da_layers = nn.ModuleDict()  # use ModuleDict instead of dict to register all layers to the model
        for module in self.da_cfg:
            name = module.get('type', None)
            feat = module.get('feat', None)
            assert name is not None, 'a type must be specified for each domain adaptation module'
            assert feat is not None, f'domain adaptation module `{name}` did not specify input features'

            # GPA uses one layer to reduce feature dimension
            if name == 'gpa':
                assert feat in ['roi',
                                'rcnn'], f'GPA can only be used for ROI or RCNN features, but was defined for `{feat}`'

                layer_type = module.get('layer', 'fc_layer')
                in_shapes = {'roi': 128, 'rcnn': 64}

                if layer_type == 'fc_layer':
                    layer = nn.Linear(math.prod(feat_shapes[feat]), in_shapes[feat])
                elif layer_type == 'avgpool':
                    reduce = math.ceil(math.prod(feat_shapes[feat]) / in_shapes[feat])
                    layer = nn.AvgPool1d(reduce)
                elif layer_type == 'maxpool':
                    reduce = math.ceil(math.prod(feat_shapes[feat]) / in_shapes[feat])
                    layer = nn.MaxPool1d(reduce)
                elif layer_type == 'none':
                    layer = None
                else:
                    raise KeyError(
                        f'Layer type `{layer_type}` in domain adaptation module `{name}` on `{feat}` does not exist!')

            # adversarial domain adaptation needs a domain classifier
            elif name == 'adversarial':
                self.prev_loss.update({feat: 0.0})
                layer = nn.Sequential()
                # first fc-layer
                layer.add_module(f'dcls_{feat}_fc0', nn.Linear(math.prod(feat_shapes[feat]), 128))
                layer.add_module(f'dcls_{feat}_bn0', nn.BatchNorm1d(128))
                layer.add_module(f'dcls_{feat}_relu0', nn.ReLU(True))
                # second fc-layer
                layer.add_module(f'dcls_{feat}_fc1', nn.Linear(128, 32))
                layer.add_module(f'dcls_{feat}_bn1', nn.BatchNorm1d(32))
                layer.add_module(f'dcls_{feat}_relu1', nn.ReLU(True))
                # output
                layer.add_module(f'dcls_{feat}_fc2', nn.Linear(32, 2))
                layer.add_module(f'dcls_{feat}_softmax', nn.LogSoftmax(dim=1))

            self.da_layers.update({f'{feat}_{name}': layer})

            # if self.da_cfg is not None:
            #     # layers for GPA heads
            #     roi_out_size = roi_head['bbox_roi_extractor']['roi_layer']['output_size']
            #     roi_out_size *= roi_out_size
            #     roi_out_size *= roi_head['bbox_roi_extractor']['out_channels']

            #     self.gpa_layer = self.gpa_cfg.get('fc_layer', 'fc_layer')
            #     assert self.gpa_layer in ['fc_layer', 'fc_layer_roi', 'fc_layer_rcnn', 'avgpool', 'maxpool', 'none']

            #     if 'fc_layer' in self.gpa_layer:
            #         # interpret 'fc_layer' as fc-layer for both, but 'fc_layer_roi' as fc-layer only for ROI and vice-
            #         # versa
            #         if 'rcnn' not in self.gpa_layer:
            #             self.gpa_layer_roi = nn.Linear(roi_out_size, 128)
            #         if 'roi' not in self.gpa_layer:
            #             self.gpa_layer_rcnn = nn.Linear(roi_head['bbox_head']['fc_out_channels'], 64)
            #     elif self.gpa_layer == 'avgpool':
            #         # reduce 256*7*7 to 128:
            #         self.gpa_layer_roi = nn.AvgPool1d(98)
            #         # reduce 1024 to 64
            #         self.gpa_layer_rcnn = nn.AvgPool1d(16)
            #     elif self.gpa_layer == 'maxpool':
            #         # reduce 256*7*7 to 128:
            #         self.gpa_layer_roi = nn.MaxPool1d(98)
            #         # reduce 1024 to 64
            #         self.gpa_layer_rcnn = nn.MaxPool1d(16)
            #     elif self.gpa_layer == 'none':
            #         pass

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
            feats.update({f'neck_{i}': (_x_src, _x_tgt)})

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
            losses.update(rpn_losses_src)
            losses.update(rpn_losses_tgt)
        else:
            raise NotImplementedError('Two stage domain-adaptive detector only works with RPN for now')
            proposal_list = proposals

        # get sampled ROIs, features in head, class score, and losses (class and bbox) in both domains
        roi_losses_src, rois_src, roi_src, rcnn_src, cls_src = self.roi_head.forward_train(
            x_src, img_metas, proposals_src, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        roi_losses_tgt, rois_tgt, roi_tgt, rcnn_tgt, cls_tgt = self.roi_head.forward_train(
            x_tgt, img_metas_tgt, proposals_tgt, gt_bboxes_tgt, gt_labels_tgt, gt_bboxes_ignore, gt_masks, **kwargs)
        # save everything for domain adaptation
        # TODO also save final bbox
        feats.update({
            'rois': (rois_src, rois_tgt),
            'roi': (roi_src, roi_tgt),
            'rcnn': (rcnn_src, rcnn_tgt),
            'cls': (cls_src, cls_tgt)
        })
        losses.update(roi_losses_src)
        losses.update(roi_losses_tgt)

        # adapting on features after ROI and after RCNN only makes a difference when the ROI-head has shared
        # layers
        if roi_src.size() == rcnn_src.size() and torch.eq(roi_src, rcnn_src).all():
            warnings.warn('The features for domain adaptation after ROI and RCNN are the same, the model might not be\
                using a shared head')

        if self.with_da:
            # do domain adaptation
            losses_da = self._align_domains(feats)
            losses.update(losses_da)

            # balance losses
            losses = self._balance_losses(losses)

        # GPA
        # if self.gpa_cfg is not None:
        #     # feed all features used for domain adaptation through fc layer (2 different ones for ROI and RCNN features)
        #     # the dimensions of the features are
        #     #   ROI:  (samples_per_gpu * roi_sampler_num, roi_out_channels, roi_output_size, roi_output_size)
        #     #   RCNN: (samples_per_gpu * roi_sampler_num, head_fc_out_channels)
        #     if 'fc_layer' in self.gpa_layer:
        #         # interpret 'fc_layer' as fc-layer for both, but 'fc_layer_roi' as fc-layer only for ROI and vice-versa
        #         if 'rcnn' in self.gpa_layer:
        #             feat_roi_src = feat_roi_src.flatten(1)
        #             feat_roi_tgt = feat_roi_tgt.flatten(1)
        #         else:
        #             feat_roi_src = self.gpa_layer_roi(feat_roi_src.flatten(1))
        #             feat_roi_tgt = self.gpa_layer_roi(feat_roi_tgt.flatten(1))
        #         if 'roi' in self.gpa_layer:
        #             feat_rcnn_src = feat_rcnn_src
        #             feat_rcnn_tgt = feat_rcnn_tgt
        #         else:
        #             feat_rcnn_src = self.gpa_layer_rcnn(feat_rcnn_src)
        #             feat_rcnn_tgt = self.gpa_layer_rcnn(feat_rcnn_tgt)
        #     elif self.gpa_layer in ['avgpool', 'maxpool']:
        #         feat_roi_src = self.gpa_layer_roi(feat_roi_src.flatten(1).unsqueeze(1)).squeeze(1)
        #         feat_roi_tgt = self.gpa_layer_roi(feat_roi_tgt.flatten(1).unsqueeze(1)).squeeze(1)
        #         feat_rcnn_src = self.gpa_layer_rcnn(feat_rcnn_src.unsqueeze(1)).squeeze(1)
        #         feat_rcnn_tgt = self.gpa_layer_rcnn(feat_rcnn_tgt.unsqueeze(1)).squeeze(1)
        #     elif self.gpa_layer == 'none':
        #         feat_roi_src = feat_roi_src.flatten(1)
        #         feat_roi_tgt = feat_roi_tgt.flatten(1)
        #         feat_rcnn_src = feat_rcnn_src
        #         feat_rcnn_tgt = feat_rcnn_tgt

        #     # compute intra-class and inter-class loss after ROI and RCNN
        #     roi_loss_intra, roi_loss_inter = self._gpa_loss(feat_roi_src, cls_score_src, rois_src, gt_bboxes, gt_labels,
        #                                                     feat_roi_tgt, cls_score_tgt, rois_tgt, gt_bboxes_tgt,
        #                                                     gt_labels_tgt, batch_size)
        #     rcnn_loss_intra, rcnn_loss_inter = self._gpa_loss(feat_rcnn_src, cls_score_src, rois_src, gt_bboxes,
        #                                                       gt_labels, feat_rcnn_tgt, cls_score_tgt, rois_tgt,
        #                                                       gt_bboxes_tgt, gt_labels_tgt, batch_size)

        #     gpa_losses = self._gpa_balance_losses(roi_loss_intra, roi_loss_inter, rcnn_loss_intra, rcnn_loss_inter)
        #     losses.update(gpa_losses)

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
            losses.update(
                getattr(self, f"_{module['type']}")(feats, self.da_layers[f"{module['feat']}_{module['type']}"],
                                                    module))

        return losses

    def _balance_losses(self, losses):
        """Rebalance all losses.
        
        Args:
            losses (dict[str, Tensor]): network losses
            
        Returns:
            dict[str, Tensor]: re-weighted network losses
        """
        # if self.first_iter:
        #     print('LOSSES:')
        #     print(losses)

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

                # if self.first_iter:
                #     print(f'updated {feat}_{loss} by factor {weight}:')
                #     print(_losses)

        self.first_iter = False
        return losses

    def _gpa(self, inputs, layer, cfg):
        # get feature maps to align and flatten
        feat = cfg['feat']
        try:
            feat_src, feat_tgt = inputs[feat]
        except KeyError:
            print(f"`{feat}` is not a valid input for an adaptation module")
        feat_src = feat_src.flatten(1)
        feat_tgt = feat_tgt.flatten(1)

        # TODO is this part correct?
        # apply layer
        if isinstance(layer, nn.Linear):
            feat_src = layer(feat_src)
            feat_tgt = layer(feat_tgt)
        elif isinstance(layer, nn.MaxPool1d) or isinstance(layer, nn.MaxPool1d):
            feat_src = layer(feat_src.unsqueeze(1)).squeeze(1)
            feat_tgt = layer(feat_tgt.unsqueeze(1)).squeeze(1)
        elif layer is None:
            pass

        # get gpa losses
        loss_intra, loss_inter = self._gpa_loss(feat_src, feat_tgt, inputs, cfg)

        return {f'loss_{feat}_gpa_intra': loss_intra, f'loss_{feat}_gpa_inter': loss_inter}

    def _adversarial(self, inputs, classifier, cfg):
        # get feature maps to align
        feat = cfg['feat']
        try:
            feat_src, feat_tgt = inputs[feat]
        except KeyError:
            print(f"`{feat} is not a valid input for an adaptation module")
        # features have shape (N, F), where N is either batch size or number of regions. We combine them into a single tensor with shape (2*N, F)
        feat_src = feat_src.view(feat_src.size(0), -1)
        feat_tgt = feat_tgt.view(feat_tgt.size(0), -1)
        feats = torch.cat((feat_src, feat_tgt), dim=0)

        # set lambda (weight of gradient after reversal) according to config
        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        mode = cfg.get('lambd', 1.0)
        if mode == 'incr':
            p = float(self.iter) / 40 / 2
            lambd = 2. / (1. + np.exp(-10 * p)) - 1
            self.iter += 1
        elif mode == 'coupled':
            lambd = math.exp(-self.prev_loss[feat])
        elif isfloat(mode):
            lambd = mode
        else:
            raise KeyError(f'adversarial lambda-mode has to be one of [`const`, `incr`, (float)], but is `{mode}`')

        # apply gradient reverse layer and domain classifier
        out = classifier(GradReverse.apply(feats, lambd))

        # build classification targets of shape (2*N) with entries 0: source, 1: target
        target = torch.cat((torch.zeros(feat_src.size(0)), torch.ones(feat_tgt.size(0))), dim=0).long().cuda()

        # calculate loss
        loss = nn.NLLLoss()(out, target)

        if mode == 'coupled':
            self.prev_loss[feat] = loss
        return {f'loss_{feat}_adversarial': loss}

    def _gpa_distance(self, feat_a, feat_b, cfg):
        """use this to compute distances between features for this loss"""
        distances = ['mean_squared', 'euclidean', 'cosine']
        distance = cfg.get('distance', 'mean_squared')
        assert distance in distances, f'distance for GPA must be one of {distances}, but got {distance}'

        if distance == 'mean_squared':
            return torch.pow(feat_a - feat_b, 2.0).mean()
        if distance == 'euclidean':
            return torch.pow(feat_a - feat_b, 2.0).sum().sqrt()
        cos = nn.CosineSimilarity(dim=0)
        return 1 - cos(feat_a, feat_b)

    def _gpa_get_adj(self, rois, epsilon=1e-6):
        """use this to calculate adjacency matrix of region proposals based on IoU
        
        Arguments:
            rois (Tensor): of shape (num_rois, 5), where the second dimension contains the values
                in the [batch_idx, x_min, y_min, x_max, y_max] format
        """
        # compute the area of every bbox
        area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        area = area + (area == 0).float() * epsilon

        # compute iou
        # x_min_ab of the overlap of a and b is max{x_min_a, x_min_b}
        # x_max_ab of the overlap of a and b is min{x_max_a, x_max_b}
        # same for y
        # these values can be computed for all pairs (a,b) at the same time using matrices
        x_min = rois[:, 1]
        x_min_copy = torch.stack([x_min] * rois.size(0), dim=0)
        x_min_copy_ = x_min_copy.permute((1, 0))
        x_min_matrix = torch.max(torch.stack([x_min_copy, x_min_copy_], dim=-1), dim=-1)[0]
        x_max = rois[:, 3]
        x_max_copy = torch.stack([x_max] * rois.size(0), dim=0)
        x_max_copy_ = x_max_copy.permute((1, 0))
        x_max_matrix = torch.min(torch.stack([x_max_copy, x_max_copy_], dim=-1), dim=-1)[0]
        y_min = rois[:, 2]
        y_min_copy = torch.stack([y_min] * rois.size(0), dim=0)
        y_min_copy_ = y_min_copy.permute((1, 0))
        y_min_matrix = torch.max(torch.stack([y_min_copy, y_min_copy_], dim=-1), dim=-1)[0]
        y_max = rois[:, 4]
        y_max_copy = torch.stack([y_max] * rois.size(0), dim=0)
        y_max_copy_ = y_max_copy.permute((1, 0))
        y_max_matrix = torch.min(torch.stack([y_max_copy, y_max_copy_], dim=-1), dim=-1)[0]

        w = torch.max(torch.stack([(x_max_matrix - x_min_matrix), torch.zeros_like(x_min_matrix)], dim=-1), dim=-1)[0]
        h = torch.max(torch.stack([(y_max_matrix - y_min_matrix), torch.zeros_like(y_min_matrix)], dim=-1), dim=-1)[0]
        intersection = w * h
        area_copy = torch.stack([area] * rois.size(0), dim=0)
        area_copy_ = area_copy.permute((1, 0))
        area_sum = area_copy + area_copy_
        union = area_sum - intersection
        iou = intersection / union

        # intuitively, all diagonal entries of the adjacency matrix should be 1.0. However, some ROIs have 0 width
        # or height, so we we set the diagonal here
        iou.fill_diagonal_(1.)
        # adjacancy matrix should have 1.0 on diagonal
        assert iou.diagonal().numel() - iou.diagonal().nonzero().size(0) == 0, 'some diagonal entries were 0'

        return iou

    def _gpa_get_adj_gt(self, rois, gts, epsilon=1e-6):
        """calculate adjacency matrix between ROIs and ground truth bboxes
        
        Arguments:
            rois (Tensor): of shape (num_rois, 5), where the second dimension contains the values
                in the [batch_idx, x_min, y_min, x_max, y_max] format
            gt_bboxes (Tensor): of shape (num_gts, 4), where the second dimension contains the values
                [x_min, y_min, x_max, y_may]
        
        Returns:
            Tensor: of shape (num_gts, num_rois), where each entry corresponds to the IoU of a ground truth with the
                corresponding region
        """
        # compute the area of every bbox
        area_rois = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        area_rois = area_rois + (area_rois == 0).float() * epsilon
        area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        area_gts = area_gts + (area_gts == 0).float() * epsilon

        # compute iou as in get_adj()
        x_min_rois = torch.stack([rois[:, 1]] * gts.size(0), dim=0)
        x_min_gts = torch.stack([gts[:, 0]] * rois.size(0), dim=0).permute((1, 0))
        x_min_matrix = torch.max(torch.stack([x_min_rois, x_min_gts], dim=-1), dim=-1)[0]
        x_max_rois = torch.stack([rois[:, 3]] * gts.size(0), dim=0)
        x_max_gts = torch.stack([gts[:, 2]] * rois.size(0), dim=0).permute((1, 0))
        x_max_matrix = torch.min(torch.stack([x_max_rois, x_max_gts], dim=-1), dim=-1)[0]
        y_min_rois = torch.stack([rois[:, 2]] * gts.size(0), dim=0)
        y_min_gts = torch.stack([gts[:, 1]] * rois.size(0), dim=0).permute((1, 0))
        y_min_matrix = torch.max(torch.stack([y_min_rois, y_min_gts], dim=-1), dim=-1)[0]
        y_max_rois = torch.stack([rois[:, 4]] * gts.size(0), dim=0)
        y_max_gts = torch.stack([gts[:, 3]] * rois.size(0), dim=0).permute((1, 0))
        y_max_matrix = torch.min(torch.stack([y_max_rois, y_max_gts], dim=-1), dim=-1)[0]

        w = torch.max(torch.stack([(x_max_matrix - x_min_matrix), torch.zeros_like(x_min_matrix)], dim=-1), dim=-1)[0]
        h = torch.max(torch.stack([(y_max_matrix - y_min_matrix), torch.zeros_like(y_min_matrix)], dim=-1), dim=-1)[0]
        intersection = w * h
        _area_rois = torch.stack([area_rois] * gts.size(0), dim=0)
        _area_gts = torch.stack([area_gts] * rois.size(0), dim=0).permute((1, 0))
        area_sum = _area_rois + _area_gts
        union = area_sum - intersection
        iou = intersection / union

        return iou

    def _gpa_inter_class_loss(self, feat_a, feat_b, cfg, margin=1.):
        # could use F.relu to write more concisely
        out = torch.pow((margin - torch.sqrt(self._gpa_distance(feat_a, feat_b, cfg))) / margin, 2) \
            * torch.pow(torch.max(margin - torch.sqrt(self._gpa_distance(feat_a, feat_b, cfg)), torch.tensor(0).float().cuda()), 2.0)
        return out

    # def _gpa_loss(self,
    #               feat,
    #               cls_prob,
    #               rois,
    #               gt_bboxes,
    #               gt_labels,
    #               feat_tgt,
    #               cls_prob_tgt,
    #               rois_tgt,
    #               gt_bboxes_tgt,
    #               gt_labels_tgt,
    #               batch_size,
    #               epsilon=1e-6):
    def _gpa_loss(self, feat_src, feat_tgt, inputs, cfg):
        """Graph-based prototpye daptation loss as in https://github.com/ChrisAllenMing/GPA-detection.
        """
        use_graph = cfg.get('use_graph', True)
        normalize = cfg.get('normalize', False)
        epsilon = cfg.get('epsilon', 1e-6)

        batch_size = inputs['img'][0].size(0)

        # view inputs as (batch_size, roi_sampler_num, )
        # with dimensions (batch_size, roi_sampler_num, num_feat)
        feat_src = feat_src.view(batch_size, -1, feat_src.size(1))
        feat_tgt = feat_tgt.view(batch_size, -1, feat_tgt.size(1))

        # get the class probability of every class for source and target domains
        # with dimensions (batch_size, roi_sampler_num, num_feat)
        cls_src, cls_tgt = inputs['cls']
        cls_src = cls_src.view(batch_size, -1, cls_src.size(1))
        cls_tgt = cls_tgt.view(batch_size, -1, cls_tgt.size(1))

        # view ROIs as (batch_size, roi_sampler_num, 5), since each ROI has 5 parameters
        rois_src, rois_tgt = inputs['rois']
        rois_src = rois_src.view(batch_size, -1, rois_src.size(1))
        rois_tgt = rois_tgt.view(batch_size, -1, rois_tgt.size(1))

        num_classes = cls_src.size(2)
        class_ptt = list()
        tgt_class_ptt = list()

        for i in range(num_classes):
            tmp_cls_prob = cls_src[:, :, i].view(cls_src.size(0), cls_src.size(1), 1)
            # if invert_cls_prob:
            #     tmp_cls_prob = 1 - tmp_cls_prob  # this is useless with 2 classes, it just switches the classes
            # TODO instead of using class probability to assing ROIs to each class, use 25% and 75% IoU with ground-truth
            tmp_class_feat = feat_src * tmp_cls_prob  # weigh features with class probability
            tmp_feat = list()
            tmp_weight = list()

            # build per-image class prototypes
            for j in range(batch_size):
                tmp_batch_feat_ = tmp_class_feat[j, :, :]
                tmp_batch_weight_ = tmp_cls_prob[j, :, :]

                # graph-based aggregation
                if use_graph:
                    tmp_batch_adj = self._gpa_get_adj(rois_src[j, :, :])
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

            # build final class prototypes and normalize by class probability
            tmp_class_feat_ = torch.stack(tmp_feat, dim=0)
            tmp_class_weight = torch.stack(tmp_weight, dim=0)
            tmp_class_feat = torch.sum(torch.sum(tmp_class_feat_, dim=1),
                                       dim=0) / (torch.sum(tmp_class_weight) + epsilon)
            class_ptt.append(tmp_class_feat)

            tmp_tgt_cls_prob = cls_tgt[:, :, i].view(cls_tgt.size(0), cls_tgt.size(1), 1)
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

        # get the intra-class and inter-class adaptation loss
        loss_intra = 0
        loss_inter = 0

        for i in range(num_classes):
            tmp_src_feat_1 = class_ptt[i, :]
            tmp_tgt_feat_1 = tgt_class_ptt[i, :]

            # intra-class loss is just distance of features
            loss_intra = loss_intra + self._gpa_distance(tmp_src_feat_1, tmp_tgt_feat_1, cfg)

            # inter-class loss is distance between all 4 source-target pairs
            for j in range(i + 1, num_classes):
                tmp_src_feat_2 = class_ptt[j, :]
                tmp_tgt_feat_2 = tgt_class_ptt[j, :]

                loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_src_feat_1, tmp_src_feat_2, cfg)
                loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_tgt_feat_1, tmp_tgt_feat_2, cfg)
                loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_src_feat_1, tmp_tgt_feat_2, cfg)
                loss_inter = loss_inter + self._gpa_inter_class_loss(tmp_tgt_feat_1, tmp_src_feat_2, cfg)

        # normalize losses
        loss_intra = loss_intra / class_ptt.size(0)
        loss_inter = loss_inter / (class_ptt.size(0) * (class_ptt.size(0) - 1) * 2)

        # intra-loss with ground-truths
        # TODO weigh features with foreground class-probabilities --> which class is foreground class?
        # TODO then the weight also has to be accumulated in every batch and can be used to normalize after aggregation
        # class_ptts = []
        # class_ptts_tgt = []
        # for j in range(batch_size):
        #     batch_feat = feat[j, :, :]
        #     batch_adj = self._gpa_get_adj_gt(rois[j, :, :], gt_bboxes[j])
        #     batch_feat = torch.mm(batch_adj, batch_feat) # these are the instance prototypes for 1 image
        #     batch_feat = batch_feat.mean(dim=0) # this is the class prototype for 1 image
        #     class_ptts.append(batch_feat)

        #     batch_feat_tgt = feat_tgt[j, :, :]
        #     batch_adj_tgt = self._gpa_get_adj_gt(rois_tgt[j, :, :], gt_bboxes_tgt[j])
        #     batch_feat_tgt = torch.mm(batch_adj_tgt, batch_feat_tgt)
        #     batch_feat_tgt = batch_feat_tgt.mean(dim=0)
        #     class_ptts_tgt.append(batch_feat_tgt)

        # # aggregate over batch to get final class prototype
        # class_ptt = torch.stack(class_ptts, dim=0).mean(dim=0)
        # class_ptt_tgt = torch.stack(class_ptts_tgt, dim=0).mean(dim=0)

        # loss_intra_gt = self._gpa_distance(class_ptt, class_ptt_tgt, d)

        # if self.train_cfg.get('da_gt', False):
        #     return loss_intra_gt, loss_inter

        return loss_intra, loss_inter

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
