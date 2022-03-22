# dataset settings
dataset_type = 'CityZurichDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'is_source']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
data = dict(samples_per_gpu=6,
            workers_per_gpu=8,
            train=dict(type=dataset_type,
                       dataroot=data_root,
                       source_img_dir='leftImg8bit/train',
                       source_ann_dir='gtFine/train',
                       target_img_dir='dark_zurich_night',
                       target_ann_dir='dark_zurich_night_gt',
                       source_img_suffix='_leftImg8bit.png',
                       target_img_suffix='_rgb_anon.png',
                       source_seg_map_suffix='_gtFine_labelTrainIds.png',
                       target_seg_map_suffix='_gt_labelTrainIds.png',
                       test_mode=False,
                       split=None,
                       pipeline=train_pipeline),
            val=dict(type='DarkZurichDataset',
                     data_root='data/dark_zurich/',
                     img_dir='val/rgb_anon/val/night/GOPR0356',
                     ann_dir='val/gt/val/night/GOPR0356',
                     pipeline=test_pipeline),
            test=dict(type='DarkZurichDataset',
                      data_root='data/dark_zurich/',
                      img_dir='val/rgb_anon/val/night/GOPR0356',
                      ann_dir='val/gt/val/night/GOPR0356',
                      pipeline=test_pipeline))
