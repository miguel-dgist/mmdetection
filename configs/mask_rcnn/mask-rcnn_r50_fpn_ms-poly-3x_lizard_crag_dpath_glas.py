_base_ = [
    '../common/ms-poly_3x_coco-instance.py',
    '../_base_/models/mask-rcnn_r50_fpn.py'
]


# dataset settings
data_root = 'data/lizard/'
metainfo = {
    'classes': (
        'neutrophil',
        'epithelial',
        'lymphocyte',
        'plasma',
        'eosinophil',
        'connective'
    ),
    'palette': [
        (220, 150, 80),
        (20, 220, 20),
        (220, 20, 50),
        (80, 220, 220),
        (40, 20, 200),
        (220, 220, 100),
    ]
}

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        dataset=dict(
            type='CocoCrossValDataset',
            data_root=data_root,
            ann_file='./,coco_annotations/crag.json,coco_annotations/dpath.json,coco_annotations/glas.json',
            data_prefix=dict(img='images/'),
            metainfo=metainfo)))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/consep.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo))
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/consep.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo))

val_evaluator = dict(
    ann_file=data_root + 'coco_annotations/consep.json')
test_evaluator = dict(
    ann_file=data_root + 'coco_annotations/consep.json')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)


# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=6,
        ),
        mask_head=dict(
            num_classes=6,
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms=dict(type='nms', iou_threshold=0.8)
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms=dict(type='nms', iou_threshold=0.8)
        ),
        rcnn=dict(
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=500
        )
    )
)


load_from = "experiments/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/checkpoint/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
