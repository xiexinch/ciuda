_base_ = [
    '../_base_/datasets/mmgen_unpaired_imgs_256x256.py',
    '../_base_/default_mmgen_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CycleSeg',
    generator=dict(type='ResnetGenerator',
                   in_channels=3,
                   out_channels=3,
                   base_channels=64,
                   norm_cfg=dict(type='IN'),
                   use_dropout=False,
                   num_blocks=9,
                   padding_mode='reflect',
                   init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(type='PatchDiscriminator',
                       in_channels=3,
                       base_channels=64,
                       num_conv=3,
                       norm_cfg=dict(type='IN'),
                       init_cfg=dict(type='normal', gain=0.02)),
    segmentor_d=dict(
        # segformer b2
        type='EncoderDecoder',
        pretrained=None,
        backbone=dict(type='MixVisionTransformer',
                      in_channels=3,
                      embed_dims=64,
                      num_stages=4,
                      num_layers=[3, 4, 6, 3],
                      num_heads=[1, 2, 5, 8],
                      patch_sizes=[7, 3, 3, 3],
                      sr_ratios=[8, 4, 2, 1],
                      out_indices=(0, 1, 2, 3),
                      mlp_ratio=4,
                      qkv_bias=True,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      drop_path_rate=0.1,
                      init_cfg=dict(type='Pretrained',
                                    checkpoint='pretrain/mit_b2.pth')),
        decode_head=dict(type='SegformerHead',
                         in_channels=[64, 128, 320, 512],
                         in_index=[0, 1, 2, 3],
                         channels=256,
                         dropout_ratio=0.1,
                         num_classes=19,
                         norm_cfg=norm_cfg,
                         align_corners=False,
                         loss_decode=dict(type='CrossEntropyLoss',
                                          use_sigmoid=False,
                                          loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')),
    segmentor_n=dict(
        type='EncoderDecoder',
        backbone=dict(type='ResNetV1c',
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
                                    checkpoint='open-mmlab://resnet101_v1c')),
        decode_head=dict(type='PSPHead',
                         in_channels=2048,
                         in_index=3,
                         channels=512,
                         pool_scales=(1, 2, 3, 6),
                         dropout_ratio=0.1,
                         num_classes=19,
                         norm_cfg=norm_cfg,
                         align_corners=False,
                         loss_decode=dict(type='CrossEntropyLoss',
                                          use_sigmoid=False,
                                          loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')),
    ce_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    gan_loss=dict(type='GANLoss',
                  gan_type='lsgan',
                  real_label_val=1.0,
                  fake_label_val=0.0,
                  loss_weight=1.0),
    cycle_loss=dict(type='L1Loss', loss_weight=10.0, reduction='mean'),
    id_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'),
    pretrained='checkpoints/iter_250000.pth',
    pretrained_seg_d=
    'checkpoints/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
    # pretrained_seg_n='checkpoints/pspnet_r101_rcs_iter_80000.pth')
    pretrained_seg_n=
    'checkpoints/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
)
train_cfg = dict(direction='a2b', buffer_size=10)
test_cfg = dict(direction='a2b', show_input=True)

domain_a = 'day'  # set by user
domain_b = 'night'  # set by user
dataroot = './data/city2darkzurich'
train_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key=f'img_{domain_a}',
         flag='color'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key=f'img_{domain_b}',
         flag='color'),
    dict(type='Resize',
         keys=[f'img_{domain_a}', f'img_{domain_b}'],
         scale=(512, 512),
         interpolation='bicubic'),
    dict(type='Crop',
         keys=[f'img_{domain_a}', f'img_{domain_b}'],
         crop_size=(256, 512),
         random_crop=True),
    dict(type='Flip', keys=[f'img_{domain_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{domain_b}'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(type='Normalize',
         keys=[f'img_{domain_a}', f'img_{domain_b}'],
         to_rgb=False,
         mean=[0.5, 0.5, 0.5],
         std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(type='Collect',
         keys=[f'img_{domain_a}', f'img_{domain_b}'],
         meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]
dataroot = './data/city2darkzurich'
data = dict(samples_per_gpu=2,
            workers_per_gpu=4,
            train=dict(dataroot=dataroot,
                       domain_a=domain_a,
                       domain_b=domain_b,
                       pipeline=train_pipeline),
            val=dict(dataroot=dataroot, domain_a=domain_a, domain_b=domain_b),
            test=dict(dataroot=dataroot, domain_a=domain_a, domain_b=domain_b))

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
#                     std=[58.395, 57.12, 57.375],
#                     to_rgb=True)
# target_crop_size = (540, 960)
# target_train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     # dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
#     # dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=target_crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=target_crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img']),
# ]

# data = dict(samples_per_gpu=1,
#             workers_per_gpu=8,
#             train=dict(
#                 type='ZurichPairDataset',
#                 data_root='data/dark_zurich/train/rgb_anon',
#                 pair_list_path='configs/_base_/datasets/zurich_dn_pair_train.csv',
#                 pipeline=target_train_pipeline,
#                 repeat_times=500),
#             val=None,
#             test=None)

optimizer = dict(generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
                 discriminators=dict(type='Adam',
                                     lr=0.0002,
                                     betas=(0.5, 0.999)),
                 segmentor_n=dict(type='SGD',
                                  lr=0.01,
                                  momentum=0.9,
                                  weight_decay=0.0005))
lr_config = dict(policy='Linear',
                 by_epoch=False,
                 target_lr=0,
                 start=10000,
                 interval=1250)
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(type='MMGenVisualizationHook',
         output_dir='training_samples',
         res_name_list=['real_day', 'fake_night', 'real_night', 'fake_day'],
         interval=1000)
]

runner = None
use_ddp_wrapper = True
total_iters = 20000
workflow = [('train', 1)]
exp_name = 'cycleseg_city2darkzurich_inpre_resize_1920x1080'
work_dir = f'./work_dirs/experiments/{exp_name}'
