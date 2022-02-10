import sys
import os.path as osp
import time
import torch

from torch.utils.data import DataLoader
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.utils import Config
from mmseg.apis import single_gpu_test
from mmseg.ops import resize
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader

from model.backbone import RepVGG  # noqa
from model.seg_head import RefineNet  # noqa
from model.discriminator import FCDiscriminator
from model import StaticLoss
from dataset import ZurichPairDataset  # noqa
from utils import PolyLrUpdater

target_crop_size = (540, 960)
crop_size = (512, 1024)
# target_crop_size = (64, 64)
# crop_size = (64, 64)

cityscapes_type = 'CityscapesDataset'
cityscapes_data_root = 'data/cityscapes/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=target_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=target_crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

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

source_config = dict(type='RepeatDataset',
                     times=500,
                     dataset=dict(type=cityscapes_type,
                                  data_root=cityscapes_data_root,
                                  img_dir='leftImg8bit/train',
                                  ann_dir='gtFine/train',
                                  pipeline=train_pipeline))

target_config = dict(
    type='ZurichPairDataset',
    data_root='data/dark_zurich/train/rgb_anon',
    pair_list_path='configs/_base_/datasets/zurich_dn_pair_train.csv',
    pipeline=target_train_pipeline,
    repeat_times=500)

test_config = dict(type='DarkZurichDataset',
                   data_root='data/dark_zurich/',
                   img_dir='val/rgb_anon/val/night',
                   ann_dir='val/gt/val/night',
                   pipeline=test_pipeline)

model_config = dict(model=dict(
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


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def set_require_grad(model: torch.nn.Module, state: bool):
    for m in model.parameters():
        m.requires_grad = state


def main(max_iters: int, work_dirs='work_dirs'):
    config = Config(model_config)
    # init model
    # repvgg_a0 = RepVGG(num_blocks=[2, 4, 14, 1],
    #                    width_multiplier=[0.75, 0.75, 0.75, 2.5],
    #                    deploy=False,
    #                    use_se=True,
    #                    invariant='W')
    # repvgg_a0 = repvgg_a0.cuda()
    # refinenet = RefineNet(num_classes=19).cuda()
    model = build_segmentor(config.model).cuda()
    discriminator = FCDiscriminator(in_channels=19).cuda()

    # init dataloader
    source_dataset = build_dataset(source_config)
    target_dataset = build_dataset(target_config)
    test_dataset = build_dataset(test_config)

    source_dataloader = build_dataloader(source_dataset,
                                         samples_per_gpu=2,
                                         workers_per_gpu=2,
                                         num_gpus=1,
                                         dist=False,
                                         shuffle=True,
                                         seed=2022)
    target_dataloader = DataLoader(target_dataset,
                                   batch_size=2,
                                   num_workers=2,
                                   persistent_workers=True,
                                   shuffle=True,
                                   pin_memory=True)

    test_dataloader = build_dataloader(test_dataset,
                                       samples_per_gpu=1,
                                       workers_per_gpu=2,
                                       dist=False,
                                       shuffle=True)

    # optimizer and lr updater
    seg_optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.01,
                                    momentum=0.9,
                                    weight_decay=0.0005)

    adv_optimizer = torch.optim.Adam(discriminator.parameters(),
                                     lr=1e-4,
                                     betas=(0.9, 0.999))
    seg_lr_updater = PolyLrUpdater(base_lr=0.01,
                                   max_iters=max_iters,
                                   min_lr=1e-4,
                                   power=0.9)
    adv_lr_updater = PolyLrUpdater(base_lr=1e-4,
                                   max_iters=max_iters,
                                   power=0.9)
    weights = torch.log(
        torch.FloatTensor([
            0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272,
            0.01227341, 0.00207795, 0.0055127, 0.15928651, 0.01157818,
            0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456,
            0.00235192, 0.00232904, 0.00098658, 0.00413907
        ])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0

    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    static_loss = StaticLoss(num_classes=11, weights=weights[:11])
    bce_loss = torch.nn.BCEWithLogitsLoss()

    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)

    # set train iters
    seg_iters = 5
    # main loop
    for i in range(max_iters + 1):
        start_time = time.time()
        source_batch = next(source_iter)
        target_batch = next(target_iter)

        source_imgs, source_labels = source_batch['img'].data[0].cuda(
        ), source_batch['gt_semantic_seg'].data[0].cuda()
        target_img_day, target_img_night = target_batch['img_day'].cuda(
        ), target_batch['img_night'].cuda()

        # train segmentation network
        set_require_grad(discriminator, False)
        set_require_grad(model, True)
        seg_optimizer.zero_grad()

        source_predicts = model.encode_decode(source_imgs, img_metas=dict())

        source_predicts = resize(source_predicts,
                                 size=crop_size,
                                 mode='bilinear',
                                 align_corners=True)

        # source_predicts = F.softmax(source_predicts, dim=1)
        loss_source = ce_loss(source_predicts, source_labels.squeeze(1))
        # save cuda memory
        # loss_source.backward()

        target_predicts_day = model.encode_decode(target_img_day,
                                                  img_metas=dict())  # noqa

        target_predicts_night = model.encode_decode(target_img_night,
                                                    img_metas=dict())  # noqa

        pseudo_prob = torch.zeros_like(target_predicts_day)
        threshold = torch.ones_like(target_predicts_day[:, :11, :, :]) * 0.2
        threshold[target_predicts_day[:, :11, :, :] > 0.4] = 0.8
        pseudo_prob[:, :11, :, :] = threshold\
            * target_predicts_day[:, :11, :, :].detach()\
            + (1 - threshold) * target_predicts_night[:, :11, :, :].detach()
        pseudo_prob[:, 11:, :, :] = target_predicts_night[:,
                                                          11:, :, :].detach()
        weights_prob = weights.expand(pseudo_prob.size()[0],
                                      pseudo_prob.size()[3],
                                      pseudo_prob.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        pseudo_prob = pseudo_prob * weights_prob
        pseudo_gt = torch.argmax(pseudo_prob.detach(), dim=1)
        pseudo_gt[pseudo_gt >= 11] = 255

        loss_static = static_loss(target_predicts_night[:, :11, :, :],
                                  pseudo_gt)
        # save cuda memory
        # loss_static.backward()
        loss = loss_source + loss_static
        loss.backward()
        loss_seg_value = loss_source.item()
        loss_static_value = loss_static.item()
        seg_optimizer.step()
        seg_lr_updater.set_lr(seg_optimizer, i)

        if i % seg_iters != 0:
            iter_time = time.time() - start_time
            eta = iter_time * (max_iters - i)
            mins, s = divmod(eta, 60)
            hours, minute = divmod(mins, 60)
            days, hour = divmod(hours, 24)
            ETA = f'{int(days)}天{int(hour)}小时{int(minute)}分{int(s)}秒'
            print(
                'Iter-[{0:5d}|{1:6d}] loss_adv_source:         loss_adv_target_day:         loss_adv_target_night:         loss_source_value: {2:.5f} loss_static_value: {3:.5f} lr_seg: {4:.5f} lr_adv: {5:.5f} ETA: {6} '  # noqa
                .format(i, max_iters, loss_seg_value, loss_static_value,
                        seg_optimizer.param_groups[0]['lr'],
                        adv_optimizer.param_groups[0]['lr'], ETA))
            continue

        # train discriminator network
        set_require_grad(discriminator, True)
        set_require_grad(model, False)

        adv_optimizer.zero_grad()

        source_predicts = source_predicts.detach()
        target_predicts_day = target_predicts_day.detach()
        target_predicts_night = target_predicts_night.detach()

        # adv_source_predict = F.sigmoid(discriminator(source_predicts))
        # adv_target_predict_day = F.sigmoid(
        #     discriminator(F.softmax(target_predicts_day, dim=1)))
        # adv_target_predict_night = F.sigmoid(
        #     discriminator(F.softmax(target_predicts_night, dim=1)))

        adv_source_predict = discriminator(source_predicts)
        adv_target_predict_day = discriminator(target_predicts_day)
        adv_target_predict_night = discriminator(target_predicts_night)

        target_label = torch.FloatTensor(
            adv_target_predict_day.data.size()).fill_(1).cuda()
        source_label = torch.FloatTensor(
            adv_source_predict.data.size()).fill_(0).cuda()

        loss_adv_source = bce_loss(adv_source_predict, source_label)
        loss_adv_target_day = bce_loss(adv_target_predict_day, target_label)
        loss_adv_target_night = bce_loss(adv_target_predict_night,
                                         target_label)
        # loss_adv_source = least_square_loss(adv_source_predict, source_label)
        # loss_adv_target_day = least_square_loss(adv_target_predict_day,
        #                                         target_label)
        # loss_adv_target_night = least_square_loss(adv_target_predict_night,
        #                                           target_label)

        loss_adv = loss_adv_source + loss_adv_target_day + loss_adv_target_night  # noqa
        loss_adv_source_value = loss_adv_source.item()
        loss_adv_target_day_value = loss_adv_target_day.item()
        loss_adv_target_night_value = loss_adv_target_night.item()
        loss_adv.backward()
        adv_optimizer.step()
        adv_lr_updater.set_lr(adv_optimizer, i)

        iter_time = time.time() - start_time
        eta = iter_time * (max_iters - i)
        mins, s = divmod(eta, 60)
        hours, minute = divmod(mins, 60)
        days, hour = divmod(hours, 24)
        ETA = f'{int(days)}天{int(hour)}小时{int(minute)}分{int(s)}秒'
        print(
            'Iter-[{0:5d}|{1:6d}] loss_adv_source: {2:.5f} loss_adv_target_day: {3:.5f} loss_adv_target_night: {4:.5f} loss_source_value: {5:.5f} loss_static_value: {6:.5f} lr_seg: {7:.5f} lr_adv: {8:.5f} ETA: {9} '  # noqa
            .format(i, max_iters, loss_adv_source_value,
                    loss_adv_target_day_value, loss_adv_target_night_value,
                    loss_seg_value, loss_static_value,
                    seg_optimizer.param_groups[0]['lr'],
                    adv_optimizer.param_groups[0]['lr'], ETA))

        if i % 4000 == 0 and i != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),
                       osp.join(work_dirs, f'segmentor_{i}.pth'))

            torch.save(discriminator.state_dict(),
                       osp.join(work_dirs, f'discriminator_{i}.pth'))
            results = single_gpu_test(MMDataParallel(model, device_ids=[0]),
                                      test_dataloader,
                                      pre_eval=True,
                                      format_only=False)
            metric = test_dataset.evaluate(results, metric='mIoU')
            print(metric)
            model.train()


if __name__ == '__main__':
    # log_file
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    work_dirs = osp.join('work_dirs', f'ciuda_{timestamp}')
    mmcv.mkdir_or_exist(work_dirs)
    log_file = osp.join(work_dirs, f'{timestamp}.log')
    sys.stdout = Logger(log_file)
    main(max_iters=80000, work_dirs=work_dirs)
