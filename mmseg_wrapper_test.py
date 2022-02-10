import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from mmcv.parallel import MMDataParallel
from mmcv.utils import Config
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.ops import resize
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader

from model.backbone import RepVGG  # noqa
from model.seg_head import RefineNet  # noqa
from model.discriminator import FCDiscriminator
from utils import PolyLrUpdater
# from evaluation import single_gpu_test

config = dict(model=dict(
    type='EncoderDecoder',
    backbone=dict(type='RepVGG',
                  num_blocks=[2, 4, 14, 1],
                  width_multiplier=[0.75, 0.75, 0.75, 2.5],
                  deploy=False,
                  use_se=True,
                  invariant='W'),
    decode_head=dict(type='RefineNet',
                     in_channels=[1280, 192, 96, 48],
                     channels=256,
                     num_classes=19,
                     in_index=[0, 1, 2, 3],
                     input_transform='multiple_select'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_config = dict(type='DarkZurichDataset',
                   data_root='data/dark_zurich/',
                   img_dir='val/rgb_anon/val/night',
                   ann_dir='val/gt/val/night',
                   pipeline=test_pipeline)

test = dict(type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/val',
            ann_dir='gtFine/val',
            pipeline=test_pipeline)

if __name__ == '__main__':

    config = Config(config)
    model = build_segmentor(config.model).cuda()
    discriminator = FCDiscriminator(in_channels=19).cuda()
    test_dataset = build_dataset(test)
    test_dataloader = build_dataloader(test_dataset,
                                       samples_per_gpu=1,
                                       workers_per_gpu=2,
                                       num_gpus=1,
                                       dist=False,
                                       shuffle=True)
    model.CLASSES = test_dataset.CLASSES
    model.PALETTE = test_dataset.PALETTE
    load_checkpoint(model.backbone, 'repvgg_40000.pth')
    load_checkpoint(model.decode_head, 'refinenet_40000.pth')
    results = single_gpu_test(MMDataParallel(model, device_ids=[0]),
                              test_dataloader,
                              pre_eval=True,
                              format_only=False)
    metric = test_dataset.evaluate(results, metric='mIoU')
    print(metric)
    model.train()
    adv_optimizer = torch.optim.Adam(discriminator.parameters(),
                                     lr=1e-4,
                                     betas=(0.9, 0.999))
    seg_lr_updater = PolyLrUpdater(base_lr=0.01,
                                   max_iters=1000,
                                   min_lr=1e-4,
                                   power=0.9)
    adv_lr_updater = PolyLrUpdater(base_lr=1e-4, max_iters=1000, power=0.9)

    img = Image.open('GOPR0351_frame_000001_rgb_anon.png')
    img = torch.from_numpy(np.array(img).astype(np.float32)).permute(
        2, 0, 1).unsqueeze(0).cuda()
    img = resize(img, scale_factor=0.5)

    seg_logits = model.encode_decode(img, img_metas=dict())
    print(seg_logits.shape)
