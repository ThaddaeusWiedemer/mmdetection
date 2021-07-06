_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
# _base_ = [
#     '../_base_/models/faster_rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]

# model
# freeze backbone completely
model = dict(backbone=dict(frozen_stages=4))

# training
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0,
        scale_limit=0.0,
        rotate_limit=180,
        interpolation=1,
        p=0.5)]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False) # caffee image norm
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # original coco pedestrian detector uses multi-scale resizing
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=False),
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
    # with 4 GPUs batch size = 4*4 = 16
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
# fine-tuning: smaller lr, freeze FPN (neck), freeze RPN
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'neck': dict(lr_mult=0.0),
            'rpn_head.cls_convs': dict(lr_mult=0.0)
        }
    ))
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
# we don't need warmup
    warmup=None,
    policy='step',
    step=[10000])
# full dataset has 2357 imgs ---(batch-size=16)--> 148 iterations * 12 epochs = 1776 total iterations
# few-shot fine-tuning paper uses anywhere between 500 and 160000 iterations
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=1776)

# evaluate every 500 iterations
evaluation = dict(interval=500, metric='bbox')
checkpoint_config = dict(interval=1776)

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
