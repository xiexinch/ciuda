dataset_type = 'CityscapesDataset'
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
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

collect_pipeline = [
    dict(type='Collect',
         keys=['img', 'gt_semantic_seg', 'img_day', 'img_night'],
         meta_keys=['img', 'gt_semantic_seg', 'img_day', 'img_night'])
]

data = dict(samples_per_gpu=2,
            workers_per_gpu=2,
            train=dict(type='MixDataset2',
                       city=dict(type=dataset_type,
                                 data_root=data_root,
                                 img_dir='leftImg8bit/train',
                                 ann_dir='gtFine/train',
                                 pipeline=train_pipeline),
                       zurich=dict(
                           type='ZurichPairDataset',
                           input_size_target=960,
                           input_size=512,
                           root='data/dark_zurich/train/rgb_anon',
                           list_path='dataset/lists/zurich_dn_pair_train.csv'),
                       pipeline=collect_pipeline))  # noqa
