_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
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
                      img_dir='val/rgb_anon/val/night',
                      ann_dir='val/gt/val/night',
                      pipeline=test_pipeline))

model = dict(
    type='EncoderDecoder',
    backbone=dict(type='RepVGG',
                  num_blocks=[2, 4, 14, 1],
                  width_multiplier=[0.75, 0.75, 0.75, 2.5],
                  deploy=False,
                  use_se=True,
                  invariant='W'),
    decode_head=dict(type='RefineNet',
                     in_channels=[1280, 192, 96, 48],
                     channels=256,
                     num_classes=19,
                     in_index=[0, 1, 2, 3],
                     input_transform='multiple_select'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
