import math
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from . import GradDampening


class GPAHead(BaseModule):
    """Graph-based prototpye daptation loss as in https://github.com/ChrisAllenMing/GPA-detection.
    """
    def __init__(self, cfg, in_shape, init_cfg=None):
        super(GPAHead, self).__init__(init_cfg)

        # get config info
        self.feat = cfg.get('feat', 'roi')  # which feature to adapt
        self.tag = cfg.get('tag', None)  # to allow for multiple heads on same features
        self.tag = f'_{self.tag}' if self.tag is not None else ''
        assert self.feat in ['feat_roi', 'feat_rcnn_shared', 'feat_rcnn_cls', 'feat_rcnn_bbox'
                             ], f'GPA can only be used for ROI or RCNN features, but was defined for `{self.feat}`'
        self.use_graph = cfg.get('use_graph', True)  # whether to use adjacency-graph-based aggregation where applicable
        self.normalize = cfg.get('normalize', False)  # whether to normalize aggregated features as in paper
        self.epsilon = cfg.get('epsilon', 1e-6)  # small value to avoid zero-divisions
        self.margin = cfg.get('margin', 1.)  # margin parameter for contrastive loss
        distances = ['mean_squared', 'euclidean', 'cosine']
        self.distance = cfg.get('distance', 'mean_squared')  # distance function for contrastive loss
        assert self.distance in distances, f'distance for GPA must be one of {distances}, but got {self.distance}'
        self.mode = cfg.get('mode', 'prediction')  # what information to use to build prototypes
        self.gt_iou_thrs = cfg.get('gt_iou_thrs', (.75, .25))  # IoU thresholds for ground-truth-based prototypes
        self.thr_mode = cfg.get('thr_mode', 'cut_off')  # how to treat thresholding to generate weights
        self.gt_use_cls = cfg.get('gt_use_cls', False)  # whether GPA with gt should use class prediction
        self.gt_bbox_dim = cfg.get('gt_bbox_dim', 'x')  # which dimension to use for bbox-based prototypes
        self.lambd = cfg.get('lambd', 1)  # factor on gradient

        # build input layer to reduce feature size
        layer_type = cfg.get('layer', 'fc_layer')
        in_shapes = {'feat_roi': 128, 'feat_rcnn_shared': 64, 'feat_rcnn_cls': 64, 'feat_rcnn_bbox': 64}
        if layer_type == 'fc_layer':
            self.layer = nn.Linear(math.prod(in_shape), in_shapes[self.feat])
        elif layer_type == 'avgpool':
            reduce = math.ceil(math.prod(in_shape) / in_shapes[self.feat])
            self.layer = nn.AvgPool1d(reduce)
        elif layer_type == 'maxpool':
            reduce = math.ceil(math.prod(in_shape) / in_shapes[self.feat])
            self.layer = nn.MaxPool1d(reduce)
        elif layer_type == 'none':
            self.layer = None
        else:
            raise KeyError(
                f"GPA layer type must be one of ['fc_layer', 'avgpool', 'maxpool', 'none'], but is '{layer_type}' on '{self.feat}'."
            )

    def forward(self, inputs):
        try:
            x_src, x_tgt = inputs[self.feat]
        except KeyError:
            print(f"Can't find input '{self.feat}' for GPA module.")
        x_src = GradDampening.apply(x_src, self.lambd).flatten(1)
        x_tgt = GradDampening.apply(x_tgt, self.lambd).flatten(1)

        # TODO is this part correct?
        # apply layer
        if isinstance(self.layer, nn.Linear):
            x_src = self.layer(x_src)
            x_tgt = self.layer(x_tgt)
        elif isinstance(self.layer, nn.MaxPool1d) or isinstance(self.layer, nn.MaxPool1d):
            x_src = self.layer(x_src.unsqueeze(1)).squeeze(1)
            x_tgt = self.layer(x_tgt.unsqueeze(1)).squeeze(1)
        elif self.layer is None:
            pass

        # get gpa losses
        loss_intra, loss_inter = self._compute_loss(x_src, x_tgt, inputs)

        return {f'loss_{self.feat}_gpa{self.tag}_intra': loss_intra, f'loss_{self.feat}_gpa{self.tag}_inter': loss_inter}

    def _compute_loss(self, x_src, x_tgt, inputs):
        self.batch_size = inputs['img'][0].size(0)

        # view inputs as (B, R, F)
        # with batch size B, number of ROIs R, and feature size F
        x_src = x_src.view(self.batch_size, -1, x_src.size(1))
        x_tgt = x_tgt.view(self.batch_size, -1, x_tgt.size(1))

        # view ROIs as (B, R, 5), since each ROI has 5 parameters
        rois_src, rois_tgt = inputs['rois']
        rois_src = rois_src.view(self.batch_size, -1, rois_src.size(1))
        rois_tgt = rois_tgt.view(self.batch_size, -1, rois_tgt.size(1))

        # collect prototypes for each category on both domains
        # get the class probability of every class for source and target domains
        # with dimensions (B, R, C) with number of classes C
        cls_src, cls_tgt = inputs['cls_score']
        cls_src = cls_src.view(self.batch_size, -1, cls_src.size(1))
        cls_tgt = cls_tgt.view(self.batch_size, -1, cls_tgt.size(1))

        # get ground-truths as list(Tensor(B, G, 4)) with G number of ground-truths
        gt_src, gt_tgt = inputs['gt_bboxes']

        if self.mode == 'prediction':
            ptt_src = self._build_ptt_cls_pred(x_src, cls_src, rois_src)
            ptt_tgt = self._build_ptt_cls_pred(x_tgt, cls_tgt, rois_tgt)

        elif self.mode == 'ground_truth':
            ptt_src = self._build_ptt_cls_gt(x_src, cls_src, gt_src, rois_src, self.gt_iou_thrs)
            ptt_tgt = self._build_ptt_cls_gt(x_tgt, cls_tgt, gt_tgt, rois_tgt, self.gt_iou_thrs, is_tgt=True)

        elif self.mode == 'gt_bbox':
            ptt_src = self._build_ptt_bbox_gt(x_src, gt_src, rois_src, self.gt_bbox_dim, self.gt_iou_thrs)
            ptt_tgt = self._build_ptt_bbox_gt(x_tgt, gt_tgt, rois_tgt, self.gt_bbox_dim, self.gt_iou_thrs)

        # get the intra-category and inter-category adaptation loss
        loss_intra = 0
        loss_inter = 0

        for i in range(ptt_src.size(0)):
            ptt_src_1 = ptt_src[i, :]
            ptt_tgt_1 = ptt_tgt[i, :]

            # intra-class loss is just distance of features
            loss_intra = loss_intra + self._distance(ptt_src_1, ptt_tgt_1)

            # inter-class loss is distance between all 4 source-target pairs
            for j in range(i + 1, ptt_src.size(0)):
                ptt_src_2 = ptt_src[j, :]
                ptt_tgt_2 = ptt_tgt[j, :]

                loss_inter = loss_inter + self._loss_inter(ptt_src_1, ptt_src_2)
                loss_inter = loss_inter + self._loss_inter(ptt_tgt_1, ptt_tgt_2)
                loss_inter = loss_inter + self._loss_inter(ptt_src_1, ptt_tgt_2)
                loss_inter = loss_inter + self._loss_inter(ptt_tgt_1, ptt_src_2)

        # normalize losses
        loss_intra = loss_intra / ptt_src.size(0)
        loss_inter = loss_inter / (ptt_src.size(0) * (ptt_src.size(0) - 1) * 2)

        return loss_intra, loss_inter

    def _build_ptt_cls_pred(self, x, cls, rois):
        """Build prototypes for each class based on class prediction. This works for unsupervised training.

        Args:
            x (Tensor): features of size (batch size, number of ROIs, feature size)
            cls (Tensor): class prediction for each ROI as (batch size, number of ROIs, number of classes)
            rois (Tensor): meta information of each ROI as (batch size, number of ROIs, coordinates)

        Returns:
            Tensor: class prototypes as (number of classes, feature size)
        """
        n_cls = cls.size(2)
        cls_ptts = list()

        for cls_idx in range(n_cls):
            # prototypes for each class only differ in initial re-weighting of ROIs by their class prediction
            _cls = cls[:, :, cls_idx].view(cls.size(0), cls.size(1), 1)
            _x = x * _cls
            ptts = list()
            ptt_weights = list()

            # build per-image prototypes for current class
            for j in range(self.batch_size):
                _x_batch = _x[j, :, :]
                _cls_batch = _cls[j, :, :]

                # graph-based aggregation
                if self.use_graph:
                    adj = self._get_adj(rois[j, :, :])
                    ptt = torch.mm(adj, _x_batch)
                    _cls_batch = torch.mm(adj, _cls_batch)

                    # divide by sum of edge weights as in paper
                    if self.normalize:
                        weight_sum = adj.sum(dim=1).unsqueeze(1)
                        ptt = torch.div(ptt, weight_sum)
                        _cls_batch = torch.div(_cls_batch, weight_sum)

                    ptts.append(ptt)
                    ptt_weights.append(_cls_batch)
                else:
                    ptts.append(_x_batch)
                    ptt_weights.append(_cls_batch)

            # build final class prototype and normalize by total weight
            ptts = torch.stack(ptts, dim=0)
            ptt_weights = torch.stack(ptt_weights, dim=0)
            cls_ptt = torch.sum(torch.sum(ptts, dim=1), dim=0) / (torch.sum(ptt_weights) + self.epsilon)
            cls_ptts.append(cls_ptt)

        cls_ptts = torch.stack(cls_ptts, dim=0)
        return cls_ptts

    def _build_ptt_cls_gt(self, x, cls, gts, rois, iou_thrs=(.75, .25), is_tgt=False):
        """Build prototypes for each class based on overlap with ground-truth. Only works for supervised training.

        Overlap greater than `iou_thrs[0]` is regarded as belonging to a class, overlap smaller than `iou_thrs[1]` is
        regarded as belonging to background.

        Currently only works for 2 classes. For multiple objects in an image, since all objects are of the same class,
        sufficient overlap with any is regarded as belonging to that class.

        Args:
            x (Tensor): features of size (batch size, number of ROIs, feature size)
            gts (Tensor): ground-truths as (batch size, num of gts, coordinates)
            rois (Tensor): meta information of each ROI as (batch size, number of ROIs, coordinates)
            is_tgt (bool, optional): whether the prototypes are built for the target domain, defaults to False

        Returns:
            Tensor: class prototypes as (number of classes, feature size)
        """
        n_cls = 2
        is_fg = (True, False)
        cls_ptts = list()

        for cls_idx in range(n_cls):
            # prototypes for each class only differ in initial re-weighting of ROIs by their overlap with the ground-truth
            ptts = list()
            weights = list()

            # cls_idx 0 is foreground, 1 is background
            if self.gt_use_cls and is_tgt:
                _cls = cls[:, :, 1 - cls_idx].view(cls.size(0), cls.size(1), 1)
                _x = x * _cls
            else:
                _x = x

            # build per-image prototypes for current class
            # this can't be done in parallel since the number of ground-truths per image varies
            for j in range(self.batch_size):
                x_batch = _x[j, :, :]
                weights_batch, _ = self._get_adj_gt(rois[j, :, :],
                                                    gts[j],
                                                    iou_thrs[cls_idx],
                                                    is_fg[cls_idx],
                                                    is_tgt=is_tgt)
                ptt = x_batch * weights_batch.unsqueeze(1)
                ptts.append(ptt)
                weights.append(weights_batch)

                # graph-based aggregation
                # if self.use_graph:
                #     adj = self._get_adj(rois[j, :, :])
                #     ptt = torch.mm(adj, _x_batch)
                #     _cls_batch = torch.mm(adj, _cls_batch)

                #     # divide by sum of edge weights as in paper
                #     if self.normalize:
                #         weight_sum = adj.sum(dim=1).unsqueeze(1)
                #         ptt = torch.div(ptt, weight_sum)
                #         _cls_batch = torch.div(_cls_batch, weight_sum)

                #     ptts.append(ptt)
                #     ptt_weights.append(_cls_batch)
                # else:
                #     ptts.append(_x_batch)
                #     ptt_weights.append(_cls_batch)

            # build final class prototype and normalize by total weight
            ptts = torch.stack(ptts, dim=0)
            weights = torch.stack(weights, dim=0)
            cls_ptt = torch.sum(torch.sum(ptts, dim=1), dim=0) / (torch.sum(weights) + self.epsilon)
            cls_ptts.append(cls_ptt)

        cls_ptts = torch.stack(cls_ptts, dim=0)
        return cls_ptts

    def _build_ptt_bbox_gt(self, x, gts, rois, dim='x', iou_thr=0.5):
        """Build prototypes for each class based on bbox error to ground-truth. Only works for supervised training.

        Each RoI is paired to the ground-truth with the maximum IoU greater than `iou_thrs`. Unpaired RoIs are ignored.
        Along the chosen dimension RoIs with a positive/negative delta to the ground-truth are aggregated into a prototype.

        Currently only works for 2 classes. For multiple objects in an image, since all objects are of the same class,
        sufficient overlap with any is regarded as belonging to that class.

        Args:
            x (Tensor): features of size (batch size, number of ROIs, feature size)
            gts (Tensor): ground-truths as (batch size, num of gts, coordinates)
            rois (Tensor): meta information of each ROI as (batch size, number of ROIs, coordinates)
            dim (str, optional): which dimension to compute the prototypes for. One of ['x', 'y', 'w', 'h']. Defaults to
                'x'
            iou_thr (float, optional): necessary minimum IoU between a RoI and a ground-truth to include the RoI in
                prototype calculation, defaults to 0.5

        Returns:
            Tensor: class prototypes as (number of classes, feature size)
        """
        # collect per-image prototypes
        ptts_pos = []
        ptts_neg = []
        for i in range(self.batch_size):
            # get IoU of all RoIs on this image with all ground-truths in this image
            _, max_iou_idx = self._get_adj_gt(rois[i, :, :], gts[i], iou_thr, keep_greater=True)

            # find the delta for each region along the specified dimension
            # and collect corresponding features
            feat_pos = []
            feat_neg = []
            for j in range(rois.size(1)):
                roi = rois[i, j, :]
                if max_iou_idx[j] == -1:
                    continue  # max IoU was below threshold
                gt = gts[i][max_iou_idx[j]]

                rx = (roi[1] + roi[3]) * 0.5
                ry = (roi[2] + roi[4]) * 0.5
                rw = roi[3] - roi[1]
                rh = roi[4] - roi[2]

                gx = (gt[0] + gt[2]) * 0.5
                gy = (gt[1] + gt[3]) * 0.5
                gw = gt[2] - gt[0]
                gh = gt[3] - gt[1]

                if dim == 'x':
                    delta = (gx - rx) / rw
                elif dim == 'y':
                    delta = (gy - ry) / rh
                elif dim == 'w':
                    delta = torch.log(gw / rw)
                elif dim == 'h':
                    delta = torch.log(gh / rh)

                if delta >= 0:
                    feat_pos.append(x[i, j, :])
                else:
                    feat_neg.append(x[i, j, :])

            # build per-image prototypes. be careful since either list might be empty
            if feat_pos:
                ptts_pos.append(torch.stack(feat_pos, dim=0).mean(dim=0))
            if feat_neg:
                ptts_neg.append(torch.stack(feat_neg, dim=0).mean(dim=0))

        # TODO with lots of bad luck, it could still happen that either list is empty. How to treat that case?
        ptt_pos = torch.stack(ptts_pos, dim=0).mean(dim=0)
        ptt_neg = torch.stack(ptts_neg, dim=0).mean(dim=0)

        return torch.stack([ptt_pos, ptt_neg], dim=0)

    def _get_adj(self, rois):
        """use this to calculate adjacency matrix of region proposals based on IoU
        
        Arguments:
            rois (Tensor): of shape (num_rois, 5), where the second dimension contains the values
                in the [batch_idx, x_min, y_min, x_max, y_max] format
        """
        # compute the area of every bbox
        area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        area = area + (area == 0).float() * self.epsilon

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

    def _get_adj_gt(self, rois, gts, iou_thr, keep_greater, is_tgt=False):
        """calculate adjacency matrix between ROIs and ground truth bboxes
        
        Arguments:
            rois (Tensor): of shape (number of ROIs, 5), where the third dimension contains the values
                in the [batch_idx, x_min, y_min, x_max, y_max] format
            gts (Tensor): of shape (num of gts, 4), where the third dimension contains the values
                [x_min, y_min, x_max, y_may]
            iou_thr (float): threshold for IoU
            keep_greater (str): whether to keep anything `greater` or `smaller` than threshold
            is_tgt (bool, optional): whether the computation is for the target domain or not, defaults to False
        
        Returns:
            (Tensor, Tensor): of shape ((R), (R)) where R is the number of ROIs, where each entry in the first tensor corresponds to the max IoU of a ROI with any ground-truth, and the second tensor records the index of the ground-truth
        """
        # compute the area of every bbox
        # replace 0 with ε
        area_rois = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        area_rois = area_rois + (area_rois == 0).float() * self.epsilon
        area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        area_gts = area_gts + (area_gts == 0).float() * self.epsilon

        # compute coordinates of intersection
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

        # compute area of intersection
        w = torch.max(torch.stack([(x_max_matrix - x_min_matrix), torch.zeros_like(x_min_matrix)], dim=-1), dim=-1)[0]
        h = torch.max(torch.stack([(y_max_matrix - y_min_matrix), torch.zeros_like(y_min_matrix)], dim=-1), dim=-1)[0]
        intersection = w * h

        # compute union
        _area_rois = torch.stack([area_rois] * gts.size(0), dim=0)
        _area_gts = torch.stack([area_gts] * rois.size(0), dim=0).permute((1, 0))
        area_sum = _area_rois + _area_gts
        union = area_sum - intersection

        # compute IoU
        iou = intersection / union

        # result has shape (number of gts, number of ROIs), but we only want the greatest IoU with any gt
        # we also return the index of the ground-truth with the largest IoU
        iou, idx = torch.max(iou, dim=0)

        # keep only values greater/smaller than threshold, set other indices to -1
        if keep_greater:
            iou[iou < iou_thr] = 0
            idx[iou < iou_thr] = -1
        else:
            iou[iou > iou_thr] = 0
            idx[iou > iou_thr] = -1

        if self.thr_mode == 'step':
            iou[iou != 0] = 1
        elif self.thr_mode == 'invers' or (self.thr_mode == 'invers_tgt' and is_tgt):
            iou = 1 - iou
            iou[iou == 1] = 0

        return iou, idx

    def _loss_inter(self, x1, x2):
        # could use F.relu to write more concisely
        loss = torch.pow((self.margin - torch.sqrt(self._distance(x1, x2))) / self.margin, 2) \
            * torch.pow(torch.max(self.margin - torch.sqrt(self._distance(x1, x2)), torch.tensor(0).float().cuda()), 2.0)
        return loss

    def _distance(self, x1, x2):
        """use this to compute distances between features for this loss"""
        if self.distance == 'mean_squared':
            return torch.pow(x1 - x2, 2.0).mean()
        if self.distance == 'euclidean':
            return torch.pow(x1 - x2, 2.0).sum().sqrt()
        cos = nn.CosineSimilarity(dim=0)
        return 1 - cos(x1, x2)