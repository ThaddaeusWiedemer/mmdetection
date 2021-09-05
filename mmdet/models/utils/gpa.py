import math
import torch
import torch.nn as nn

from mmcv.runner import BaseModule


class GPAHead(BaseModule):
    """Graph-based prototpye daptation loss as in https://github.com/ChrisAllenMing/GPA-detection.
    """
    def __init__(self, cfg, in_shape, init_cfg=None):
        super(GPAHead, self).__init__(init_cfg)

        # get config info
        self.feat = cfg.get('feat', 'roi')
        assert self.feat in ['roi', 'rcnn'
                             ], f'GPA can only be used for ROI or RCNN features, but was defined for `{self.feat}`'
        self.use_graph = cfg.get('use_graph', True)
        self.normalize = cfg.get('normalize', False)
        self.epsilon = cfg.get('epsilon', 1e-6)
        self.margin = cfg.get('margin', 1.)
        distances = ['mean_squared', 'euclidean', 'cosine']
        self.distance = cfg.get('distance', 'mean_squared')
        assert self.distance in distances, f'distance for GPA must be one of {distances}, but got {self.distance}'

        # build layer
        layer_type = cfg.get('layer', 'fc_layer')
        in_shapes = {'roi': 128, 'rcnn': 64}
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
        x_src = x_src.flatten(1)
        x_tgt = x_tgt.flatten(1)

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

        return {f'loss_{self.feat}_gpa_intra': loss_intra, f'loss_{self.feat}_gpa_inter': loss_inter}

    def _compute_loss(self, x_src, x_tgt, inputs):
        self.batch_size = inputs['img'][0].size(0)

        # view inputs as (batch_size, roi_sampler_num, )
        # with dimensions (batch_size, roi_sampler_num, num_feat)
        x_src = x_src.view(self.batch_size, -1, x_src.size(1))
        x_tgt = x_tgt.view(self.batch_size, -1, x_tgt.size(1))

        # get the class probability of every class for source and target domains
        # with dimensions (batch_size, roi_sampler_num, num_feat)
        cls_src, cls_tgt = inputs['cls']
        cls_src = cls_src.view(self.batch_size, -1, cls_src.size(1))
        cls_tgt = cls_tgt.view(self.batch_size, -1, cls_tgt.size(1))

        # view ROIs as (batch_size, roi_sampler_num, 5), since each ROI has 5 parameters
        rois_src, rois_tgt = inputs['rois']
        rois_src = rois_src.view(self.batch_size, -1, rois_src.size(1))
        rois_tgt = rois_tgt.view(self.batch_size, -1, rois_tgt.size(1))

        # collect prototypes for each class
        num_classes = cls_src.size(2)
        ptt_src = list()
        ptt_tgt = list()

        for i in range(num_classes):
            ptt_src.append(self._build_prototype(x_src, cls_src, rois_src, i))
            ptt_tgt.append(self._build_prototype(x_tgt, cls_tgt, rois_tgt, i))

        ptt_src = torch.stack(ptt_src, dim=0)
        ptt_tgt = torch.stack(ptt_tgt, dim=0)

        # get the intra-class and inter-class adaptation loss
        loss_intra = 0
        loss_inter = 0

        for i in range(num_classes):
            ptt_src_1 = ptt_src[i, :]
            ptt_tgt_1 = ptt_tgt[i, :]

            # intra-class loss is just distance of features
            loss_intra = loss_intra + self._distance(ptt_src_1, ptt_tgt_1)

            # inter-class loss is distance between all 4 source-target pairs
            for j in range(i + 1, num_classes):
                ptt_src_2 = ptt_src[j, :]
                ptt_tgt_2 = ptt_tgt[j, :]

                loss_inter = loss_inter + self._inter_class_loss(ptt_src_1, ptt_src_2)
                loss_inter = loss_inter + self._inter_class_loss(ptt_tgt_1, ptt_tgt_2)
                loss_inter = loss_inter + self._inter_class_loss(ptt_src_1, ptt_tgt_2)
                loss_inter = loss_inter + self._inter_class_loss(ptt_tgt_1, ptt_src_2)

        # normalize losses
        loss_intra = loss_intra / ptt_src.size(0)
        loss_inter = loss_inter / (ptt_src.size(0) * (ptt_src.size(0) - 1) * 2)

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

    def _build_prototype(self, x, cls, rois, class_idx):
        _cls = cls[:, :, class_idx].view(cls.size(0), cls.size(1), 1)
        # if invert_cls_prob:
        #     tmp_cls_prob = 1 - tmp_cls_prob  # this is useless with 2 classes, it just switches the classes
        # TODO instead of using class probability to assing ROIs to each class, use 25% and 75% IoU with ground-truth
        _x = x * _cls  # weigh features with class probability
        ptts = list()
        ptt_weights = list()

        # build per-image class prototypes
        for j in range(self.batch_size):
            _x_batch = _x[j, :, :]
            _cls_batch = _cls[j, :, :]

            # graph-based aggregation
            if self.use_graph:
                adj = self._get_adj(rois[j, :, :])
                _x_batch = torch.mm(adj, _x_batch)
                _cls_batch = torch.mm(adj, _cls_batch)

                # divide by sum of edge weights as in paper
                if self.normalize:
                    weight_sum = adj.sum(dim=1).unsqueeze(1)
                    _x_batch = torch.div(_x_batch, weight_sum)
                    _cls_batch = torch.div(_cls_batch, weight_sum)

                ptts.append(_x_batch)
                ptt_weights.append(_cls_batch)
            else:
                ptts.append(_x_batch)
                ptt_weights.append(_cls_batch)

        # build final class prototype and normalize by total weight
        ptts = torch.stack(ptts, dim=0)
        ptt_weights = torch.stack(ptt_weights, dim=0)
        prototype = torch.sum(torch.sum(ptts, dim=1), dim=0) / (torch.sum(ptt_weights) + self.epsilon)
        return prototype

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

    def _get_adj_gt(self, rois, gts):
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
        area_rois = area_rois + (area_rois == 0).float() * self.epsilon
        area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        area_gts = area_gts + (area_gts == 0).float() * self.epsilon

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

    def _inter_loss(self, x1, x2):
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