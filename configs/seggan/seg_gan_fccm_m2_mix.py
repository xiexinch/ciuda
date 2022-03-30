_base_ = [
    '../_base_/datasets/mix_dataset.py', '../_base_/default_mmgen_runtime.py'
]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='SegGAN2',
    segmentor=dict(
        type='EncoderDecoder',
        pretrained=None,
        backbone=dict(type='MobileNetV2',
                      widen_factor=1.,
                      strides=(1, 2, 2, 2, 1, 2, 1),
                      dilations=(1, 1, 1, 1, 1, 1, 1),
                      out_indices=(1, 2, 4, 6),
                      init_cfg=dict(type='Pretrained',
                                    checkpoint='mmcls://mobilenet_v2')),
        decode_head=dict(type='FCCMHead_EXT',
                         channels=1024,
                         with_fuse_attn=True),
        auxiliary_head=[
            dict(type='FCNHead',
                 in_channels=24,
                 channels=24,
                 num_convs=2,
                 num_classes=19,
                 in_index=0,
                 norm_cfg=norm_cfg,
                 concat_input=False,
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=1.0)),
            dict(type='FCNHead',
                 in_channels=32,
                 channels=64,
                 num_convs=2,
                 num_classes=19,
                 in_index=1,
                 norm_cfg=norm_cfg,
                 concat_input=False,
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=1.0)),
            dict(type='FCNHead',
                 in_channels=96,
                 channels=256,
                 num_convs=2,
                 num_classes=19,
                 in_index=2,
                 norm_cfg=norm_cfg,
                 concat_input=False,
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=1.0)),
            dict(type='FCNHead',
                 in_channels=320,
                 channels=512,
                 num_convs=2,
                 num_classes=19,
                 in_index=3,
                 norm_cfg=norm_cfg,
                 concat_input=False,
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=1.0)),
        ],
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')),
    discriminator=dict(type='FCDiscriminator', in_channels=19),
    gan_loss=dict(type='GANLoss',
                  gan_type='vanilla',
                  real_label_val=1.0,
                  fake_label_val=0.0,
                  loss_weight=1.0),
    ce_loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        # class_weight=[
        #     0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0,
        #     1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        # ]
        class_weight=[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]),
    static_loss=dict(type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=[
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         0., 0., 0., 0., 0., 0., 0., 0.
                     ]))
train_cfg = dict(direction='a2b', buffer_size=10)
test_cfg = dict(direction='a2b', show_input=True)

optimizer = dict(discriminators=dict(type='Adam',
                                     lr=0.0002,
                                     betas=(0.5, 0.999)),
                 segmentors=dict(type='SGD',
                                 lr=0.01,
                                 momentum=0.9,
                                 weight_decay=0.0005))
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(
    policy='poly',
    # warmup='linear',
    # #  warmup_iters=1500,
    # warmup_ratio=1e-6,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)
# custom_hooks = [
#     dict(type='MMGenVisualizationHook',
#          output_dir='training_samples',
#          res_name_list=['seg_pred'],
#          interval=2000)
# ]

# use dynamic runner
# runner = dict(
#     type='IterBasedRunner',
#     is_dynamic_ddp=True,
#     pass_training_status=True)
runner = None

use_ddp_wrapper = True
total_iters = 40000
workflow = [('train', 1)]
exp_name = 'seggan_fccm_202203310245'
work_dir = f'./work_dirs/experiments/{exp_name}'
# evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
checkpoint = 'checkpoints/fccmext_mobilenetv2_74.12.pth'  # noqa

data = dict(samples_per_gpu=4)
