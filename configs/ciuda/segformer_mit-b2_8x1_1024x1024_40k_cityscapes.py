_base_ = './segformer_mit-b0_8x1_1024x1024_40k_cityscapes.py'

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

model = dict(backbone=dict(init_cfg=dict(type='Pretrained',
                                         checkpoint='pretrain/mit_b2.pth'),
                           embed_dims=64,
                           num_layers=[3, 4, 6, 3]),
             decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=2,
            val=dict(type='DarkZurichDataset',
                     data_root='data/dark_zurich/val',
                     img_dir='rgb_anon/val/night/GOPR0356',
                     ann_dir='gt/val/night/GOPR0356',
                     pipeline=test_pipeline),
            test=dict(type='DarkZurichDataset',
                      data_root='data/dark_zurich/val',
                      img_dir='rgb_anon/val/night/GOPR0356',
                      ann_dir='gt/val/night/GOPR0356',
                      pipeline=test_pipeline))
checkpoint = 'checkpoints/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'  # noqa
