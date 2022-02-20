_base_ = './pspnet_r50-d8_512x1024_40k_cityscapes.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    val=dict(
        type='DarkZurichDataset',
        data_root='data/dark_zurich/val',
        img_dir='rgb_anon/val/night/GOPR0356',
        ann_dir='gt/val/night/GOPR0356',
        pipeline=test_pipeline),
    test=dict(
        type='DarkZurichDataset',
        data_root='data/dark_zurich/val',
        img_dir='rgb_anon/val/night/GOPR0356',
        ann_dir='gt/val/night/GOPR0356',
        pipeline=test_pipeline))
