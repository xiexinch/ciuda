_base_ = [
    '../_base_/methods/roundganv1_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_1024x512.py',
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
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=125000, interval=1250)
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(type='MMGenVisualizationHook',
         output_dir='training_samples',
         res_name_list=['real_a', 'fake_b', 'real_b', 'fake_c', 'real_c', 'fake_a'],
         interval=1000)
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
