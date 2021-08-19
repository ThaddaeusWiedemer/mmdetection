# model
model = dict(
    type='TwoStageDetectorAdaptive',  # use adaptive model
    # TODO consider moving this to init_cfg
    pretrained='open-mmlab://detectron2/resnet50_caffe',  # only for backbone, overwritten by `load_from=...`
    backbone=dict(type='ResNet',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  norm_eval=True,
                  style='caffe'),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(type='RPNHead',
                  in_channels=256,
                  feat_channels=256,
                  anchor_generator=dict(type='AnchorGenerator',
                                        scales=[8],
                                        ratios=[0.5, 1.0, 2.0],
                                        strides=[4, 8, 16, 32, 64]),
                  bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                  target_means=[0.0, 0.0, 0.0, 0.0],
                                  target_stds=[1.0, 1.0, 1.0, 1.0]),
                  loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                  loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHeadAdaptive',  # special head that returns more information
        bbox_roi_extractor=dict(type='SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHeadAdaptive',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # for pedestrian detection
            bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                            target_means=[0.0, 0.0, 0.0, 0.0],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        gpa=dict(
            loss_roi_intra=10.0,  # loss for domain adaptation after roi
            loss_roi_inter=0.1,  # loss for domain adaptation in rcnn head
            loss_rcnn_intra=100.0,  # loss for domain adaptation intra-class loss compared to inter-class loss
            loss_rcnn_inter=0.1,  # loss for domain adaptation inter-class loss compared to intra-class loss
            distance='mean_squared',  # distance function for domain adaptation losses
            normalize=False,  # normalize weights when building instance prototypes
            use_graph=True,  # whether to use adjacency matrix for aggregation of instance prototypes
            fc_layer='fc_layer'),
        rpn=dict(assigner=dict(type='MaxIoUAssigner',
                               pos_iou_thr=0.7,
                               neg_iou_thr=0.3,
                               min_pos_iou=0.3,
                               match_low_quality=True,
                               ignore_iof_thr=-1),
                 sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1,
                              add_gt_as_proposals=False),
                 allowed_border=-1,
                 pos_weight=-1,
                 debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner',
                                pos_iou_thr=0.5,
                                neg_iou_thr=0.5,
                                min_pos_iou=0.5,
                                match_low_quality=False,
                                ignore_iof_thr=-1),
                  sampler=dict(type='RandomSampler',
                               num=512,
                               pos_fraction=0.25,
                               neg_pos_ub=-1,
                               add_gt_as_proposals=True),
                  pos_weight=-1,
                  debug=False)),
    test_cfg=dict(rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
                  rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)))

# data
dataset_type = 'CocoDataset'
data_root_src = 'data/PIROPO/'
data_root_tgt = 'data/MW-18Mar/'
classes = ('person', )

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline_src = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_pipeline_tgt = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Albu',
    #      transforms=[dict(type='RandomRotate90', p=1.0)],
    #      bbox_params=dict(type='BboxParams',
    #                       format='pascal_voc',
    #                       label_fields=['gt_labels'],
    #                       min_visibility=0.0,
    #                       filter_lost_elements=True),
    #      keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
    #      update_pad_shape=False,
    #      skip_img_without_anno=False),
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(
    #             type='ShiftScaleRotate',
    #             shift_limit=0.0,
    #             scale_limit=0.0,
    #             rotate_limit=180,
    #             interpolation=1,
    #             p=1.0)
    #     ],
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
    #     update_pad_shape=False,
    #     skip_img_without_anno=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img'])
         ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train_src=dict(  # source domain training set
        type=dataset_type,
        ann_file=data_root_src + 'omni_training.json',
        img_prefix=data_root_src,
        pipeline=train_pipeline_src,
        classes=classes),
    train_tgt=dict(  # target domain training set
        type=dataset_type,
        ann_file=data_root_tgt + 'training.json',
        img_prefix=data_root_tgt,
        pipeline=train_pipeline_tgt,
        classes=classes),
    val=dict(  # validation set from target domain
        type=dataset_type,
        ann_file=data_root_tgt + 'test.json',
        img_prefix=data_root_tgt,
        pipeline=test_pipeline,
        classes=classes),
    test=dict(  # test set from target domain
        type=dataset_type,
        ann_file=data_root_tgt + 'test.json',
        img_prefix=data_root_tgt,
        pipeline=test_pipeline,
        classes=classes))

# training and optimizer
# fine-tuning: smaller lr, freeze FPN (neck), freeze RPN
evaluation = dict(interval=5, save_best='bbox_mAP_50', metric='bbox')
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    # uncomment to freeze neck and RPN
    # paramwise_cfg=dict(
    #     custom_keys=dict({
    #         'neck': dict(lr_mult=0.0),
    #         'rpn_head.cls_convs': dict(lr_mult=0.0)
    #     }))
    # paramwise_cfg=dict(custom_keys=dict({'gpa_layer_roi': dict(lr_mult=0.1), 'gpa_layer_rcnn': dict(lr_mult=0.1)}))
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    # warmup_iters=500,
    # warmup_ratio=0.001,
    step=[6])
# change to iteration based runner with 1776*x iterations for training on PIROPO
runner = dict(type='EpochBasedRunnerAdaptive', max_epochs=12)  # use adaptive runner that loads 2 datasets
checkpoint_config = dict(interval=1)  # for iter-based runner use 1776 or similar
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = 'work_dirs/da'
