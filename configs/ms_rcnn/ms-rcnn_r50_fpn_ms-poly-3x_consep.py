_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_ms-poly-3x_consep.py'
model = dict(
    type='MaskScoringRCNN',
    roi_head=dict(
        type='MaskScoringRoIHead',
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1)),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5)))
