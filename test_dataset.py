from dataset import ZurichPairDataset  # noqa
from torch.utils.data import DataLoader
from mmseg.datasets import build_dataset, build_dataloader

target_crop_size = (540, 960)
cityscapes_type = 'CityscapesDataset'
cityscapes_data_root = 'data/cityscapes/'
crop_size = (512, 1024)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=target_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=target_crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

source_config = dict(type='RepeatDataset',
                     times=500,
                     dataset=dict(type=cityscapes_type,
                                  data_root=cityscapes_data_root,
                                  img_dir='leftImg8bit/train',
                                  ann_dir='gtFine/train',
                                  pipeline=train_pipeline))

target_config = dict(
    type='ZurichPairDataset',
    data_root='data/dark_zurich/train/rgb_anon',
    pair_list_path='configs/_base_/datasets/zurich_dn_pair_train.csv',
    pipeline=target_train_pipeline,
    repeat_times=500)

if __name__ == '__main__':
    source_dataset = build_dataset(source_config)
    target_dataset = build_dataset(target_config)

    source_dataloader = build_dataloader(source_dataset,
                                         samples_per_gpu=2,
                                         workers_per_gpu=2,
                                         num_gpus=1,
                                         dist=False,
                                         shuffle=True,
                                         seed=2022)
    target_dataloader = DataLoader(target_dataset,
                                   batch_size=2,
                                   num_workers=2,
                                   persistent_workers=True,
                                   shuffle=True,
                                   pin_memory=True)

    # test source loader
    source_iter = iter(source_dataloader)
    for i_iter in range(1600):
        batch = next(source_iter)
        print(batch['img_metas'].data)

    # test target loader
    target_iter = iter(target_dataloader)
    for i_iter in range(10000):
        batch = next(target_iter)
        print(batch['img_day_metas'], batch['img_night_metas'])
