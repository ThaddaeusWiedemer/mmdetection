# concats multiple COCO like datasets to a single dataset

from pathlib import Path
import json
import pycocotools.coco as coco
import random
from typing import List
import math
import string
import numpy as np

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

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

def concat_sets(sets: List, out_path: str, out_size: int=None, cat: str=None):
    """Concats COCO-like datasets. If `out_size` is provided selects a random contiguos split of images from each
    provided dataset and concats them into one set of size `out_size`.

    Args:
        sets (list(tuple(str, str)) | list(str)): list of annotation files or (annotation file, image path) tuples of
            datasets that should be concatted
        out_path (str): path for generated annotation file
        out_size (int, optional): number of images in final dataset. All datasets are combined fully if None.
    """
    cocos = []
    img_paths = []
    images = []
    annotations = []
    next_img_id = 0
    next_ann_id = 0
    max_img_per_set = []
    cat_ids = []
    
    # shuffle if making random split to avoid bias
    if out_size is not None:
        random.shuffle(sets)
    
    # prepare set
    for s in sets:
        # get COCO set and image path for each set
        if isinstance(s, tuple):
            json_file = s[0]
            image_path = s[1]
        else:
            json_file = s
            image_path = None
        c = coco.COCO(json_file)
        cocos.append(c)
        img_paths.append[image_path]
        
        # get category id and max number of images for that category per set
        cat_id = []
        for id, cat_info in c.cats.items():
            if cat_info['name'] == cat:
                cat_id = id
        cat_ids.append(cat_id)
        max_img = len(c.getImgIds(catIds=cat_id))
        print(f'{s} contains {max_img} images with "{cat}" category')
        max_img_per_set.append(max_img)
    
    # use flood-fill to determine number of images from each set that is as evenly distributed as possible, even if sets
    # are not of same size
    if out_size is not None:
        if out_size > np.sum(max_img_per_set):
            print(f'There are only {np.sum(max_img_per_set)} images in total, \
                cannot make a split with {out_size} images')
            return
        img_per_set = np.array(max_img_per_set)
        while np.sum(img_per_set) > out_size:
            argmax =img_per_set.argmax()
            idx = argmax[0] if isinstance(argmax, np.ndarray) else argmax
            img_per_set[idx] -= 1

    # TODO merge sets
    # for i, s in enumerate(sets):
    #     if out_size is not None and img_per_set[i] == 0:
    #         continue

    #     if out_size is None:
    #         next_img_id, next_ann_id = merge(images, annotations, next_img_id, next_ann_id, c, image_path)
    #     else:
    #         next_img_id, next_ann_id = merge(images, annotations, next_img_id, next_ann_id, c, image_path, img_per_set[i])
        # print([i['id'] for i in images])
        # print([a['image_id'] for a in annotations])
    # print('total images:', len(images))
    # print('total annotations:', len(annotations))
    # categories = c.loadCats(c.getCatIds())
    # result = {'categories': categories, 'images': images, 'annotations': annotations}
    # with open(out_path, 'w') as f:
    #     json.dump(result, f)

def id_string(n: int) -> str:
    """Return a unique alphabetical id for an index `n`.
    
    E.g. 0 -> a, 1 -> b, 26 -> aa, 27 -> ab
    """
    out = ''
    while n >= 0:
        out = string.ascii_lowercase[n % 26] + out
        n = int(n/26) - 1
    return out

def random_splits(sets: List, out_path: str, out_size: int, num: int, cat: str=None):
    """Make `num` random splits of the datasets in `sets` containing `out_size` images each."""
    for n in range(num):
        concat_sets(sets, out_path + '_' + str(out_size) + id_string(n) + '.json', out_size, cat=cat)

# -----------------------------------
# PIROPO

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

# paths = [
#         ('/data/PIROPO/omni_1A/omni1A_training/annotations.json', '/data/PIROPO/omni_1A/omni1A_training/'),
#         ('/data/PIROPO/omni_1B/omni1B_training/annotations.json', '/data/PIROPO/omni_1B/omni1B_training/'),
#         ('/data/PIROPO/omni_2A/omni2A_training/annotations.json', '/data/PIROPO/omni_2A/omni2A_training/'),
#         ('/data/PIROPO/omni_3A/omni3A_training/annotations.json', '/data/PIROPO/omni_3A/omni3A_training/'),
# ]
# concat_sets(paths, '/data/PIROPO/omni_training.json')
# random_splits(paths, '/data/PIROPO/train', 1, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 2, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 5, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 10, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 20, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 50, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 100, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 200, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 500, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 1000, 10, 'person')
# random_splits(paths, '/data/PIROPO/train', 2000, 10, 'person')

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

# -----------------------------------
# MIRROR WORLDS

# paths = [
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-1/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-1/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-4/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-4/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-5/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-5/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-6/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-6/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-9/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-9/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-11/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-11/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-15/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-15/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-16/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-16/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-20/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-20/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-28/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-28/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-29/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-29/img1/'),
#         ('/data/MW-18Mar/MWAll/Test/MW-18Mar-30/img1/annotations.json', '/data/MW-18Mar/MWAll/Test/MW-18Mar-30/img1/'),
# ]
# concat_sets(paths, '/data/MW-18Mar/test.json')

paths = [
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-2/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-2/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-3/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-3/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-7/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-7/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-8/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-8/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-10/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-10/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-12/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-12/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-13/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-13/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-14/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-14/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-17/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-17/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-18/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-18/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-19/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-19/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-21/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-21/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-22/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-22/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-23/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-23/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-24/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-24/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-25/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-25/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-26/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-26/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-27/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-27/img1/'),
        ('/data/MW-18Mar/MWAll/Train/MW-18Mar-31/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-31/img1/'),
]
# concat_sets(paths, '/data/MW-18Mar/train.json')
# random_splits(paths, '/data/MW-18Mar/train', 1, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 2, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 5, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 10, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 20, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 50, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 100, 10, 'person')
# random_splits(paths, '/data/MW-18Mar/train', 200, 10, 'person')

random_splits(paths, '/data/MW-18Mar/train', 700, 1, 'person')

# -----------------------------------
# PIROPO + MIRROR WORLDS

# paths = [
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-2/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-2/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-3/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-3/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-7/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-7/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-8/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-8/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-10/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-10/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-12/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-12/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-13/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-13/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-14/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-14/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-17/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-17/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-18/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-18/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-19/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-19/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-21/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-21/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-22/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-22/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-23/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-23/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-24/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-24/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-25/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-25/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-26/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-26/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-27/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-27/img1/'),
#         ('/data/MW-18Mar/MWAll/Train/MW-18Mar-31/img1/annotations.json', '/data/MW-18Mar/MWAll/Train/MW-18Mar-31/img1/'),
#         ('/data/PIROPO/omni_1A/omni1A_training/annotations.json', '/data/PIROPO/omni_1A/omni1A_training/'),
#         ('/data/PIROPO/omni_1B/omni1B_training/annotations.json', '/data/PIROPO/omni_1B/omni1B_training/'),
#         ('/data/PIROPO/omni_2A/omni2A_training/annotations.json', '/data/PIROPO/omni_2A/omni2A_training/'),
#         ('/data/PIROPO/omni_3A/omni3A_training/annotations.json', '/data/PIROPO/omni_3A/omni3A_training/'),
# ]
# concat_sets(paths, '/data/piropo_mw_train.json')

# -----------------------------------
# BOMNI

# paths = [
#         ('/data/Bomni-DB/scenario1/top-0/annotations.json', '/data/Bomni-DB/scenario1/top-0/'),
# ]
# concat_sets(paths, '/data/Bomni-DB/test.json')

# paths = [
#         ('/data/Bomni-DB/scenario1/top-1/annotations.json', '/data/Bomni-DB/scenario1/top-1/'),
#         ('/data/Bomni-DB/scenario1/top-2/annotations.json', '/data/Bomni-DB/scenario1/top-2/'),
#         ('/data/Bomni-DB/scenario1/top-3/annotations.json', '/data/Bomni-DB/scenario1/top-3/'),
# ]
# concat_sets(paths, '/data/Bomni-DB/train.json')
# random_splits(paths, '/data/Bomni-DB/train', 1, 10)
# random_splits(paths, '/data/Bomni-DB/train', 2, 10)
# random_splits(paths, '/data/Bomni-DB/train', 5, 10)
# random_splits(paths, '/data/Bomni-DB/train', 10, 10)
# random_splits(paths, '/data/Bomni-DB/train', 20, 10)
# random_splits(paths, '/data/Bomni-DB/train', 50, 10)
# random_splits(paths, '/data/Bomni-DB/train', 100, 10)