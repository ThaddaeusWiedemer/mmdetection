# concats multiple COCO like datasets to a single dataset

from pathlib import Path
import json
import pycocotools.coco as coco
import random
from typing import List
import math
import string


def merge(images, annotations, next_img_id, next_ann_id, other, img_path=None, nimgs=None):
    print(next_img_id)
    print(next_ann_id)
    other_images = other.loadImgs(other.getImgIds())
    other_annotations = other.loadAnns(other.getAnnIds())
    # print([i['id'] for i in other_images])
    # print([a['image_id'] for a in other_annotations])
    # if number is specified, draw random samples from Images
    if nimgs is not None:
        other_images = random.sample(other_images, nimgs)
    # print([i['id'] for i in other_images])
    img_id_mapping = {}
    for img in other_images:
        img_id_mapping[img['id']] = next_img_id
        img['id'] = next_img_id
        next_img_id += 1
        if img_path is not None:
            img['file_name'] = img_path + img['file_name']
        images.append(img)
    for ann in other_annotations:
        if not ann['image_id'] in img_id_mapping:
            continue
        ann['id'] = next_ann_id
        next_ann_id += 1
        ann['image_id'] = img_id_mapping[ann['image_id']]
        annotations.append(ann)
    return next_img_id, next_ann_id

def concat_sets(sets: List, out_path: str, out_size=None):
    """Concats COCO-like datasets. If `out_size` is provided selects a random contiguos split of images from each
    provided dataset and concats them into one set of size `out_size`.

    Args:
        sets (list(tuple(str, str)) | list(str)): list of annotation files or (annotation file, image path) tuples of
            datasets that should be concatted
        out_path (str): path for generated annotation file
        out_size (int, optional): number of images in final dataset. All datasets are combined fully if None.
    """
    images = []
    annotations = []
    next_img_id = 0
    next_ann_id = 0
    # if out_size is not evenly divisble by len(sets), more images will be taken from the first sets. To avoid bias,
    # the sets are shuffled
    if out_size is not None:
        random.shuffle(sets)
        img_per_set = []
        for s in reversed(range(len(sets))):
            img_per_set.append(math.ceil(out_size/(s+1)))
            out_size = out_size - img_per_set[-1]
    for i, s in enumerate(sets):
        if out_size is not None and img_per_set[i] == 0:
            continue

        if isinstance(s, tuple):
            json_file = s[0]
            image_path = s[1]
        else:
            json_file = s
            image_path = None
        c = coco.COCO(json_file)
        if out_size is None:
            next_img_id, next_ann_id = merge(images, annotations, next_img_id, next_ann_id, c, image_path)
        else:
            next_img_id, next_ann_id = merge(images, annotations, next_img_id, next_ann_id, c, image_path, img_per_set[i])
        # print([i['id'] for i in images])
        # print([a['image_id'] for a in annotations])
    print('total images:', len(images))
    print('total annotations:', len(annotations))
    categories = c.loadCats(c.getCatIds())
    result = {'categories': categories, 'images': images, 'annotations': annotations}
    # with open(out_path, 'w') as f:
    #     json.dump(result, f)

# def random_split(s, size, out_path):
#     raise NotImplementedError
#     if isinstance(s, tuple):
#         json_file = s[0]
#         image_path = s[1]
#     else:
#         json_file = s
#         image_path = None
#     c = coco.COCO(json_file)
#     categories = c.loadCats(c.getCatIds())
#     images = c.loadImgs(c.getImgIds())
#     annotations = c.loadAnns(c.getAnnIds())
#     # TODO randomly select subset
#     result = {'categories': categories, 'images': images, 'annotations': annotations}
#     with open(out_path, 'w') as f:
#         json.dump(result, f)

def id_string(n: int) -> str:
    out = ''
    while n >= 0:
        out = string.ascii_lowercase[n % 26] + out
        n = int(n/26) - 1
    return out

def random_splits(sets: List, out_path: str, out_size: int, num: int):
    for n in range(num):
        concat_sets(sets, out_path + '_' + str(out_size) + id_string(n) + '.json', out_size)

# paths = [
        # ('MW_18Mar/Test/MW-18Mar-1/annotations.json', 'MW_18Mar/Test/MW-18Mar-1/'),
        # ('MW_18Mar/Test/MW-18Mar-4/annotations.json', 'MW_18Mar/Test/MW-18Mar-4/'),
        # ('MW_18Mar/Test/MW-18Mar-5/annotations.json', 'MW_18Mar/Test/MW-18Mar-5/'),
        # ('MW_18Mar/Test/MW-18Mar-6/annotations.json', 'MW_18Mar/Test/MW-18Mar-6/'),
        # ('MW_18Mar/Test/MW-18Mar-9/annotations.json', 'MW_18Mar/Test/MW-18Mar-9/'),
        # ('MW_18Mar/Test/MW-18Mar-11/annotations.json', 'MW_18Mar/Test/MW-18Mar-11/'),
        # ('MW_18Mar/Test/MW-18Mar-15/annotations.json', 'MW_18Mar/Test/MW-18Mar-15/'),
        # ('MW_18Mar/Test/MW-18Mar-16/annotations.json', 'MW_18Mar/Test/MW-18Mar-16/'),
        # ('MW_18Mar/Test/MW-18Mar-20/annotations.json', 'MW_18Mar/Test/MW-18Mar-20/'),
        # ('MW_18Mar/Test/MW-18Mar-28/annotations.json', 'MW_18Mar/Test/MW-18Mar-28/'),
        # ('MW_18Mar/Test/MW-18Mar-29/annotations.json', 'MW_18Mar/Test/MW-18Mar-29/'),
        # ('MW_18Mar/Test/MW-18Mar-30/annotations.json', 'MW_18Mar/Test/MW-18Mar-30/'),
        # ('PIROPO/Room_A/omni_1A/omni1A_test2/annotations.json', 'PIROPO/Room_A/omni_1A/omni1A_test2/'),
        # ('PIROPO/Room_A/omni_1A/omni1A_test3/annotations.json', 'PIROPO/Room_A/omni_1A/omni1A_test3/'),
        # ('PIROPO/Room_A/omni_2A/omni2A_test2/annotations.json', 'PIROPO/Room_A/omni_2A/omni2A_test2/'),
        # ('PIROPO/Room_A/omni_2A/omni2A_test3/annotations.json', 'PIROPO/Room_A/omni_2A/omni2A_test3/'),
        # ('PIROPO/Room_A/omni_3A/omni3A_test2/annotations.json', 'PIROPO/Room_A/omni_3A/omni3A_test2/'),
        # ('PIROPO/Room_A/omni_3A/omni3A_test3/annotations.json', 'PIROPO/Room_A/omni_3A/omni3A_test3/'),
        # ('PIROPO/Room_B/omni_1B/omni1B_test2/annotations.json', 'PIROPO/Room_B/omni_1B/omni1B_test2/'),
        # ('PIROPO/Room_B/omni_1B/omni1B_test3/annotations.json', 'PIROPO/Room_B/omni_1B/omni1B_test3/'),
# ]

# paths = [
#         ('/data/PIROPO/omni_1A/omni1A_test2/annotations.json', '/data/PIROPO/omni_1A/omni1A_test2/'),
#         ('/data/PIROPO/omni_1B/omni1B_test2/annotations.json', '/data/PIROPO/omni_1B/omni1B_test2/'),
#         ('/data/PIROPO/omni_2A/omni2A_test2/annotations.json', '/data/PIROPO/omni_2A/omni2A_test2/'),
#         ('/data/PIROPO/omni_3A/omni3A_test2/annotations.json', '/data/PIROPO/omni_3A/omni3A_test2/'),
# ]
# concat_sets(paths, '/data/PIROPO/omni_test2.json')

# paths = [
#         ('/data/PIROPO/omni_1A/omni1A_test3/annotations.json', '/data/PIROPO/omni_1A/omni1A_test3/'),
#         ('/data/PIROPO/omni_1B/omni1B_test3/annotations.json', '/data/PIROPO/omni_1B/omni1B_test3/'),
#         ('/data/PIROPO/omni_2A/omni2A_test3/annotations.json', '/data/PIROPO/omni_2A/omni2A_test3/'),
#         ('/data/PIROPO/omni_3A/omni3A_test3/annotations.json', '/data/PIROPO/omni_3A/omni3A_test3/'),
# ]
# concat_sets(paths, '/data/PIROPO/omni_test3.json')

paths = [
        ('/data/PIROPO/omni_1A/omni1A_training/annotations.json', '/data/PIROPO/omni_1A/omni1A_training/'),
        ('/data/PIROPO/omni_1B/omni1B_training/annotations.json', '/data/PIROPO/omni_1B/omni1B_training/'),
        ('/data/PIROPO/omni_2A/omni2A_training/annotations.json', '/data/PIROPO/omni_2A/omni2A_training/'),
        ('/data/PIROPO/omni_3A/omni3A_training/annotations.json', '/data/PIROPO/omni_3A/omni3A_training/'),
]
# concat_sets(paths, '/data/PIROPO/omni_training.json')
# random_splits(paths, '/data/PIROPO/omni_training', 1, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 2, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 5, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 10, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 20, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 50, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 100, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 200, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 500, 10)
# random_splits(paths, '/data/PIROPO/omni_training', 1000, 10)
random_splits(paths, '/data/PIROPO/omni_training', 2000, 10)

# paths = [
#         ('/data/PIROPO/omni_1A/omni1A_test2/annotations.json', '/data/PIROPO/omni_1A/omni1A_test2/'),
#         ('/data/PIROPO/omni_1B/omni1B_test2/annotations.json', '/data/PIROPO/omni_1B/omni1B_test2/'),
#         ('/data/PIROPO/omni_2A/omni2A_test2/annotations.json', '/data/PIROPO/omni_2A/omni2A_test2/'),
#         ('/data/PIROPO/omni_3A/omni3A_test2/annotations.json', '/data/PIROPO/omni_3A/omni3A_test2/'),
#         ('/data/PIROPO/omni_1A/omni1A_test3/annotations.json', '/data/PIROPO/omni_1A/omni1A_test3/'),
#         ('/data/PIROPO/omni_1B/omni1B_test3/annotations.json', '/data/PIROPO/omni_1B/omni1B_test3/'),
#         ('/data/PIROPO/omni_2A/omni2A_test3/annotations.json', '/data/PIROPO/omni_2A/omni2A_test3/'),
#         ('/data/PIROPO/omni_3A/omni3A_test3/annotations.json', '/data/PIROPO/omni_3A/omni3A_test3/'),
# ]
# concat_sets(paths, '/data/PIROPO/omni_tests.json')

# paths = [
#         ('/data/PIROPO/omni_1A/omni1A_test2/annotations.json', '/data/PIROPO/omni_1A/omni1A_test2/'),
#         ('/data/PIROPO/omni_1B/omni1B_test2/annotations.json', '/data/PIROPO/omni_1B/omni1B_test2/'),
#         ('/data/PIROPO/omni_2A/omni2A_test2/annotations.json', '/data/PIROPO/omni_2A/omni2A_test2/'),
#         ('/data/PIROPO/omni_3A/omni3A_test2/annotations.json', '/data/PIROPO/omni_3A/omni3A_test2/'),
#         ('/data/PIROPO/omni_1A/omni1A_test3/annotations.json', '/data/PIROPO/omni_1A/omni1A_test3/'),
#         ('/data/PIROPO/omni_1B/omni1B_test3/annotations.json', '/data/PIROPO/omni_1B/omni1B_test3/'),
#         ('/data/PIROPO/omni_2A/omni2A_test3/annotations.json', '/data/PIROPO/omni_2A/omni2A_test3/'),
#         ('/data/PIROPO/omni_3A/omni3A_test3/annotations.json', '/data/PIROPO/omni_3A/omni3A_test3/'),
#         ('/data/PIROPO/omni_1A/omni1A_training/annotations.json', '/data/PIROPO/omni_1A/omni1A_training/'),
#         ('/data/PIROPO/omni_1B/omni1B_training/annotations.json', '/data/PIROPO/omni_1B/omni1B_training/'),
#         ('/data/PIROPO/omni_2A/omni2A_training/annotations.json', '/data/PIROPO/omni_2A/omni2A_training/'),
#         ('/data/PIROPO/omni_3A/omni3A_training/annotations.json', '/data/PIROPO/omni_3A/omni3A_training/'),
# ]
# concat_sets(paths, '/data/PIROPO/omni_all.json')