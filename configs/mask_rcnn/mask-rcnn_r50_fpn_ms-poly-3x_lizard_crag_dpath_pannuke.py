_base_ = [
    './mask-rcnn_r50_fpn_ms-poly-3x_lizard_crag_dpath_glas.py'
]


train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='./,coco_annotations/crag.json,coco_annotations/dpath.json,coco_annotations/pannuke.json'
        )
    )
)
