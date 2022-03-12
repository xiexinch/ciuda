_base_ = [
    '../_base_/datasets/unpaired_imgs_label_1024x512.py',
    '../_base_/default_mmgen_runtime.py'
]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SegGAN',
    segmentor=dict(
        type='EncoderDecoder',
        pretrained=None,
        backbone=dict(type='ResNetV1c',
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
        auxiliary_head=None,
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')),
    discriminator=dict(type='PatchDiscriminator',
                       in_channels=2048,
                       base_channels=64,
                       num_conv=3,
                       norm_cfg=dict(type='IN'),
                       init_cfg=dict(type='normal', gain=0.02)),
    gan_loss=dict(type='BatchGANLoss',
                  gan_type='vanilla',
                  real_label_val=1.0,
                  fake_label_val=0.0,
                  loss_weight=1.0),
    ce_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

checkpoint_config = dict(interval=8000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(type='MMGenVisualizationHook',
         output_dir='training_samples',
         res_name_list=['seg_pred'],
         interval=2000)
]

# use dynamic runner
# runner = dict(
#     type='IterBasedRunner',
#     is_dynamic_ddp=True,
#     pass_training_status=True)
runner = None

use_ddp_wrapper = True
total_iters = 80000
workflow = [('train', 1)]
exp_name = 'seggan_pspnet_202203121510'
work_dir = f'./work_dirs/experiments/{exp_name}'
# evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
