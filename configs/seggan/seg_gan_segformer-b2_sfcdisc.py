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
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='SegGAN',
    segmentor=dict(
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
    discriminator=dict(type='SimpleFCDiscriminator', in_channels=512),
    gan_loss=dict(type='BatchGANLoss',
                  gan_type='vanilla',
                  real_label_val=1.0,
                  fake_label_val=0.0,
                  loss_weight=1.0),
    ce_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                     0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                 ]),
    static_loss=dict(type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.4,
                     class_weight=[
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         0., 0., 0., 0., 0., 0., 0., 0.
                     ]))
train_cfg = dict(direction='a2b', buffer_size=10)
test_cfg = dict(direction='a2b', show_input=True)

optimizer = dict(discriminators=dict(type='Adam',
                                     lr=0.0002,
                                     betas=(0.5, 0.999)),
                 segmentors=dict(type='AdamW',
                                 lr=0.00006,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.01,
                                 paramwise_cfg=dict(
                                     custom_keys={
                                         'pos_block': dict(decay_mult=0.),
                                         'norm': dict(decay_mult=0.),
                                         'head': dict(lr_mult=10.)
                                     })))
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=0.0,
                 by_epoch=False)

checkpoint_config = dict(interval=4000, save_optimizer=True, by_epoch=False)
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
total_iters = 40000
workflow = [('train', 1)]
exp_name = 'seggan_202203171018'
work_dir = f'./work_dirs/experiments/{exp_name}'
# evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
checkpoint = 'checkpoints/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'  # noqa
