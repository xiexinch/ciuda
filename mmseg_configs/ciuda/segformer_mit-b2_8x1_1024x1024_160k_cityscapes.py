_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b2.pth'),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(
    samples_per_gpu=4,
    val=dict(
        type='DarkZurichDataset',
        data_root='data/dark_zurich/val',
        img_dir='rgb_anon/val/night/GOPR0356',
        ann_dir='gt/val/night/GOPR0356',
        pipeline=test_pipeline),
    test=dict(
        type='DarkZurichDataset',
        data_root='data/dark_zurich/val',
        img_dir='rgb_anon/val/night/GOPR0356',
        ann_dir='gt/val/night/GOPR0356',
        pipeline=test_pipeline))
