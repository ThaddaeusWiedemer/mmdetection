_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
# _base_ = [
#     '../_base_/models/faster_rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]

# model
# model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
# classes = ('person', )

# training
# caffe img norm
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # original coco pedestrian detector uses multi-scale resizing
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# testing
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

# datasets
data_root = 'data/PIROPO/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        # classes=classes,
        ann_file=data_root + 'omni_training.json',
        img_prefix=data_root,
        pipeline=train_pipeline
    ),
    val=dict(
        # classes=classes,
        ann_file=data_root + 'omni_test3.json',
        img_prefix=data_root
    ),
    test=dict(
        # classes=classes,
        ann_file=data_root + 'omni_test2.json',
        img_prefix=data_root
    ))

# optimizer
# changed lr from 0.02 to 0.01 for finetuning
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
# changed steps to 8, 11 and decreased number of epochs for finetuning
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# checkpoint_config = dict(interval=1)

# yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# yapf:enable

# custom_hooks = [dict(type='NumClassCheckHook')]

# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]

load_from = 'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
work_dir = 'work_dirs/PIROPO'
