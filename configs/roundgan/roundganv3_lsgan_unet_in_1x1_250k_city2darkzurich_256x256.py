_base_ = [
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_mmgen_runtime.py'
]

model = dict(type='RoundGANV3',
             generator=dict(type='BilateralGenerator',
                            unet_generator_cfg=dict(
                                type='UnetGenerator',
                                in_channels=3,
                                out_channels=3,
                                num_down=8,
                                base_channels=64,
                                norm_cfg=dict(type='BN'),
                                use_dropout=True,
                                init_cfg=dict(type='normal', gain=0.02)),
                            rrdb_generator_cfg=dict(
                                num_in_ch=3,
                                num_out_ch=3,
                                num_feat=64,
                                num_block=23,
                                num_grow_ch=32,
                                scale=4),
                            rrdb_pretrained='checkpoints/RealESRGAN_x4plus.pth'
             ),
             discriminator=dict(type='UNetDiscriminatorSN',
                                num_in_ch=3,
                                num_feat=64,
                                skip_connection=True),
             gan_loss=dict(type='GANLoss',
                           gan_type='lsgan',
                           real_label_val=1.0,
                           fake_label_val=0.0,
                           loss_weight=1.0),
             cycle_loss=dict(type='L1Loss', loss_weight=10.0,
                             reduction='mean'),
             id_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'),
             perceptual_loss=dict(
                type='PerceptualLoss',
                layer_weights={'34': 1.0},
                vgg_type='vgg19',
                perceptual_weight=1.0,
                style_weight=0,
                norm_img=False)
            )

train_cfg = dict(direction='a2b', buffer_size=10)
test_cfg = dict(test_direction='a2b', show_input=False)
dataroot = './data/city2darkzurich'
data = dict(train=dict(dataroot=dataroot),
            val=dict(dataroot=dataroot),
            test=dict(dataroot=dataroot))

optimizer = dict(generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
                 discriminators=dict(type='Adam',
                                     lr=0.0002,
                                     betas=(0.5, 0.999)))
lr_config = None
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(type='MMGenVisualizationHook',
         output_dir='training_samples',
         res_name_list=['fake_b','fake_c'],
         interval=5000)
]

runner = None
use_ddp_wrapper = True
total_iters = 250000
workflow = [('train', 1)]
exp_name = 'roundganv3_city2darkzurich'
work_dir = f'./work_dirs/experiments/{exp_name}'
# testA: 309, testB:238
metrics = dict(FID=dict(type='FID',
                        num_images=238,
                        image_shape=(3, 2048, 1024)),
               IS=dict(type='IS', num_images=238, image_shape=(3, 2048, 1024)))
