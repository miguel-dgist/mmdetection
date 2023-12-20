_base_ = [
    '../mask_rcnn/mask-rcnn_r50_fpn_ms-poly-3x_lizard_crag_dpath_glas.py'
]


# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=6,
            cls_predictor_cfg=dict(
                type='NormedLinear', tempearture=20
            ),
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=6,
                loss_weight=1.0
            )
        ),
        mask_head=dict(
            num_classes=6,
        )
    )
)
