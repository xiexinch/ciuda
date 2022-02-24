_base_ = [
    '../_base_/datasets/rcs_cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='PositionEncodingResNet',
                  img_size=(512, 1024),
                  resnet_cfg=dict(
                      type='ResNetV1c',
                      in_channels=3,
                      depth=101,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      dilations=(1, 1, 2, 4),
                      strides=(1, 2, 1, 1),
                      norm_cfg=norm_cfg,
                      norm_eval=False,
                      style='pytorch',
                      contract_dilation=True,
                      init_cfg=dict(type='Pretrained',
                                    checkpoint='open-mmlab://resnet101_v1c'))),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=1024,
                        in_index=2,
                        channels=256,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=19,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# checkpoint = 'checkpoints/cifcn_r50_iter_40000.pth'

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
    # test=dict(
    #     type='DarkZurichDataset',
    #     data_root='data/dark_zurich/val',
    #     img_dir='rgb_anon/val/night/GOPR0356',
    #     ann_dir='gt/val/night/GOPR0356',
    #     pipeline=test_pipeline)
)
