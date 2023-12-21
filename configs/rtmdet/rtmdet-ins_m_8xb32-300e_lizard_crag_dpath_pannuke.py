_base_ = './rtmdet-ins_m_8xb32-300e_lizard_crag_dpath_glas.py'


train_dataloader = dict(
    dataset=dict(
        ann_file='./,coco_annotations/crag.json,coco_annotations/dpath.json,coco_annotations/pannuke.json',
    )
)
