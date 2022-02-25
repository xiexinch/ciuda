_base_ = [
    '../_base_/methods/roundganv2_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_769x769.py',
    '../_base_/default_mmgen_runtime.py'
]
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
         res_name_list=['real_a', 'fake_a', 'rec_a_','fake_b', 'fake_c_',  ],
         interval=5000)
]

runner = None
use_ddp_wrapper = True
total_iters = 250000
workflow = [('train', 1)]
exp_name = 'roundgan_city2darkzurich'
work_dir = f'./work_dirs/experiments/{exp_name}'
# testA: 309, testB:238
metrics = dict(FID=dict(type='FID',
                        num_images=238,
                        image_shape=(3, 2048, 1024)),
               IS=dict(type='IS', num_images=238, image_shape=(3, 2048, 1024)))
