# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCOCrossVal
from .coco import CocoDataset


@DATASETS.register_module()
class CocoCrossValDataset(CocoDataset):
    """Dataset for COCO Cross Validation."""

    COCOAPI = COCOCrossVal

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        # with get_local_path(
        #         "./", backend_args=self.backend_args) as local_dir:
        #     self.coco = self.COCOAPI(local_dir,self.ann_file)
        self.coco = self.COCOAPI(self.data_root,self.ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
