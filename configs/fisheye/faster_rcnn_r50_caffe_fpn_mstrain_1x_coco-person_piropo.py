# _base_ = '/home/thaddaus/MasterthesisCode/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'

data_root = 'data/PIROPO/'

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

data = dict(
    test=dict(
        ann_file=data_root + 'omni_all.json',
        img_prefix=data_root))

load_from = '../../checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
work_dir = 'work_dirs/blub'
