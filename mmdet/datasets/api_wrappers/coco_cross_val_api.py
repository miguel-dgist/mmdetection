
# This file adds cross-validation for coco annotations

import os
import json
import time
import warnings
from collections import defaultdict

import pycocotools
from pycocotools.coco import COCO as _COCO


class COCOCrossVal(_COCO):
    """This class allows cross-validation using the official pycocotools package.
    """

    def __init__(self, local_dir, annotation_files=None):
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning)
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_files == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = {"images":[],"annotations":[],"categories":[]}
            annotation_fpaths = [os.path.join(local_dir,fpath) for fpath in annotation_files.split(",")[1:]]
            for ann_file in annotation_fpaths:
                with open(ann_file, "r") as f:
                    subset = json.load(f)
                    assert type(subset)==dict, 'annotation file format {} not supported'.format(type(subset))
                    dataset["images"] += subset["images"]
                    dataset["annotations"] += subset["annotations"]
                    dataset["categories"] = subset["categories"]
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)
