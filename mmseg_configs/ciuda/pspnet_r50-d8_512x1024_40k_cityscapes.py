_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/rcs_cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
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
data = dict(samples_per_gpu=2,
            workers_per_gpu=2,
            test=dict(type='DarkZurichDataset',
                      data_root='data/dark_zurich/',
                      img_dir='test/rgb_anon/test/night',
                      ann_dir='',
                      pipeline=test_pipeline))
