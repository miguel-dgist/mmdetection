_base_ = './rtmdet-ins_l_8xb32-300e_coco.py'

model = dict(
    backbone=dict(
        deepen_factor=0.67, 
        widen_factor=0.75),
    neck=dict(
        in_channels=[192, 384, 768], 
        out_channels=192, 
        num_csp_blocks=2),
    bbox_head=dict(
        num_classes=6,
        in_channels=192, 
        feat_channels=192),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=500,
        mask_thr_binary=0.5)
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=8,
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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
    batch_size=6,
    num_workers=6,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(
        type='CocoCrossValDataset',
        data_root=data_root,
        ann_file='./,coco_annotations/crag.json,coco_annotations/dpath.json,coco_annotations/glas.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=6,
    num_workers=6,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/consep.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=3,
    num_workers=3,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/consep.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        pipeline=test_pipeline))

val_evaluator = dict(
    metric=['bbox', 'segm'],
    ann_file=data_root + 'coco_annotations/consep.json',
    proposal_nums=(100, 300, 500),
    classwise=True)
test_evaluator = dict(
    ann_file=data_root + 'coco_annotations/consep.json',
    proposal_nums=(100, 300, 500),
    classwise=True)


tta_model = dict(
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.7), max_per_img=500))


max_epochs = 50
stage2_num_epochs = 10
base_lr = 0.004
interval = 5

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])


# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]



load_from = "experiments/rtmdet-ins_m_8xb32-300e_coco/checkpoint/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
