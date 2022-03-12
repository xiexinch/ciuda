norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(type='SegGAN',
             segmentor=dict(
                type='EncoderDecoder',
                pretrained='open-mmlab://resnet50_v1c',
                backbone=dict(
                    type='ResNetV1c',
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    dilations=(1, 1, 2, 4),
                    strides=(1, 2, 1, 1),
                    norm_cfg=norm_cfg,
                    norm_eval=False,
                    style='pytorch',
                    contract_dilation=True),
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
                auxiliary_head=None,
                # model training and testing settings
                train_cfg=dict(),
                test_cfg=dict(mode='whole')),
             discriminator=dict(type='PatchDiscriminator',
                                in_channels=3,
                                base_channels=64,
                                num_conv=3,
                                norm_cfg=dict(type='IN'),
                                init_cfg=dict(type='normal', gain=0.02)),
             gan_loss=dict(type='GANLoss',
                           gan_type='lsgan',
                           real_label_val=1.0,
                           fake_label_val=0.0,
                           loss_weight=1.0),
             ce_loss=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
train_cfg = dict(direction='a2b', buffer_size=10)
test_cfg = dict(direction='a2b', show_input=True)