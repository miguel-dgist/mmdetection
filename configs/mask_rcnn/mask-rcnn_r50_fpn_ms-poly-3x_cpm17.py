_base_ = [
    '../common/ms-poly_3x_coco-instance.py',
    '../_base_/models/mask-rcnn_r50_fpn.py'
]


# dataset settings
data_root = 'data/cpm17/'
metainfo = {
    'classes': ('cell', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            ann_file='annotations/train.json',
            data_prefix=dict(img='train/'),
            metainfo=metainfo)))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo))
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo))

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
        )
    )
)


load_from = "experiments/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/checkpoint/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
