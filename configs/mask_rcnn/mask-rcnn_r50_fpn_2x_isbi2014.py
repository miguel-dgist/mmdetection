_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


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


# dataset settings
data_root = 'data/isbi2014/'
metainfo = {
    'classes': ('cell', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo))
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo))
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo))

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json')

load_from = "experiments/mask_rcnn_r50_fpn_2x_coco/checkpoint/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
