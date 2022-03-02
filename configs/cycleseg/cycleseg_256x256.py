_base_ = [
    '../_base_/datasets/unpaired_imgs_769x769.py',
    '../_base_/default_mmgen_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(type='CycleSeg',
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
                backbone=dict(
                    type='MixVisionTransformer',
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
                    init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b2.pth')),
                decode_head=dict(
                    type='SegformerHead',
                    in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    channels=256,
                    dropout_ratio=0.1,
                    num_classes=19,
                    norm_cfg=norm_cfg,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                # model training and testing settings
                train_cfg=dict(),
                test_cfg=dict(mode='whole')),
             segmentor_n=dict(
                type='EncoderDecoder',
                backbone=dict(
                    type='ResNetV1c',
                    depth=101,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    dilations=(1, 1, 2, 4),
                    strides=(1, 2, 1, 1),
                    norm_cfg=norm_cfg,
                    norm_eval=False,
                    style='pytorch',
                    contract_dilation=True,
                    init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')),
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
                auxiliary_head=dict(
                    type='FCNHead',
                    in_channels=1024,
                    in_index=2,
                    channels=256,
                    num_convs=1,
                    concat_input=False,
                    dropout_ratio=0.1,
                    num_classes=19,
                    norm_cfg=norm_cfg,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
                # model training and testing settings
                train_cfg=dict(),
                test_cfg=dict(mode='whole')),
             
             gan_loss=dict(type='GANLoss',
                           gan_type='lsgan',
                           real_label_val=1.0,
                           fake_label_val=0.0,
                           loss_weight=1.0),
             cycle_loss=dict(type='L1Loss', loss_weight=10.0,
                             reduction='mean'),
             id_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'),
             pretrained_seg_d='checkpoints/',
             pretrained_seg_n='checkpoints/')

dataroot = './data/city2darkziruch'
data = dict(train=dict(dataroot=dataroot),
            val=dict(dataroot=dataroot),
            test=dict(dataroot=dataroot))

optimizer = dict(generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
                 discriminators=dict(type='Adam',
                                     lr=0.0002,
                                     betas=(0.5, 0.999)),
                 segmentor_n=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
lr_config = None
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(type='MMGenVisualizationHook',
         output_dir='training_samples',
         res_name_list=['real_day', 'fake_night', 'real_night', 'fake_day'],
         interval=1000)
]

runner = None
use_ddp_wrapper = True
total_iters = 250000
workflow = [('train', 1)]
exp_name = 'cycleseg_city2darkzurich'
work_dir = f'./work_dirs/experiments/{exp_name}'