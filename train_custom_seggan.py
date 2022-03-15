import argparse
import sys
import os
import os.path as osp
import time
import torch

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.utils import Config
from mmcv.runner import load_checkpoint, get_dist_info, init_dist
from mmseg.apis import single_gpu_test
from mmseg.ops import resize
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from model.backbone import *  # noqa
from model.seg_head import RefineNet  # noqa
from model.discriminator import FCDiscriminator
from model import StaticLoss, least_square_loss
from dataset import ZurichPairDataset  # noqa
from utils import PolyLrUpdater

target_crop_size = (540, 960)
crop_size = (512, 512)
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

# model_config = dict(model=dict(
#     type='EncoderDecoder',
#     backbone=dict(type='RepVGG',
#                   num_blocks=[2, 4, 14, 1],
#                   width_multiplier=[0.75, 0.75, 0.75, 2.5],
#                   deploy=False,
#                   use_se=True,
#                   invariant='W'),
#     decode_head=dict(type='RefineNet',
#                      in_channels=[1280, 192, 96, 48],
#                      channels=256,
#                      num_classes=19,
#                      in_index=[0, 1, 2, 3],
#                      input_transform='multiple_select'),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole')))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load-from',
                        help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus',
                            type=int,
                            help='number of gpus to use '
                            '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids',
                            type=int,
                            nargs='+',
                            help='ids of gpus to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


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


def main(max_iters: int, work_dirs='work_dirs', distributed=False):
    # config = Config(model_config)
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # init distributed env first
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        print(cfg.gpu_ids)

    model = build_segmentor(cfg.model)
    discriminator_1 = FCDiscriminator(in_channels=512)
    # discriminator_2 = FCDiscriminator(in_channels=19)

    if cfg.checkpoint:
        load_checkpoint(model, cfg.checkpoint)

    source_dataset = build_dataset(source_config)
    target_dataset = build_dataset(target_config)
    test_dataset = build_dataset(test_config)

    if distributed:

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False).module
        discriminator_1 = MMDistributedDataParallel(
            discriminator_1.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False).module

        # discriminator_2 = MMDistributedDataParallel(
        #     discriminator_2.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        #     find_unused_parameters=False).module

        # model = MMDataParallel(model.cuda(cfg.gpu_ids[0]),
        #                        device_ids=[torch.cuda.current_device()]).module
        # discriminator_1 = MMDataParallel(
        #     discriminator_1.cuda(cfg.gpu_ids[0]),
        #     device_ids=[torch.cuda.current_device()]).module
        # discriminator_2 = MMDataParallel(
        #     discriminator_2.cuda(cfg.gpu_ids[0]),
        #     device_ids=[torch.cuda.current_device()]).module

    else:
        model = model.cuda()
        discriminator_1 = discriminator_1.cuda()
        # discriminator_2 = discriminator_2.cuda()

    # init dataloader
    source_dataloader = build_dataloader(
        source_dataset,
        samples_per_gpu=2,
        workers_per_gpu=4,
        num_gpus=len(cfg.gpu_ids) if distributed else 1,
        dist=distributed,
        shuffle=True,
        seed=2022)
    mmcv.utils.print_log('source dataloader finished')

    target_dataloader = build_dataloader(
        target_dataset,
        samples_per_gpu=2,
        workers_per_gpu=4,
        num_gpus=len(cfg.gpu_ids) if distributed else 1,
        dist=distributed,
        shuffle=True,
        pin_memory=False,
        seed=2022)
    mmcv.utils.print_log('target dataloader finished')

    test_dataloader = build_dataloader(test_dataset,
                                       samples_per_gpu=1,
                                       workers_per_gpu=4,
                                       dist=False,
                                       shuffle=True)

    # optimizer and lr updater
    # seg_optimizer = torch.optim.SGD(model.parameters(),
    #                                 lr=0.01,
    #                                 momentum=0.9,
    #                                 weight_decay=0.0005)
    # optimizer and lr updater
    seg_optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.01,
                                     betas=(0.9, 0.999))

    adv_optimizer_1 = torch.optim.Adam(discriminator_1.parameters(),
                                       lr=1e-4,
                                       betas=(0.9, 0.999))
    # adv_optimizer_2 = torch.optim.Adam(discriminator_2.parameters(),
    #                                    lr=1e-4,
    #                                    betas=(0.9, 0.999))
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

    weights_static = torch.FloatTensor([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]).cuda()
    # weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0

    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    # static_loss = StaticLoss(num_classes=11, weights=weights[:11])
    static_loss = torch.nn.CrossEntropyLoss(ignore_index=255,
                                            weight=weights_static)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)

    # set train iters
    seg_iters = 1
    # main loop
    mmcv.utils.print_log('start training')
    for i in range(max_iters + 1):
        start_time = time.time()
        source_batch = next(source_iter)
        target_batch = next(target_iter)

        source_imgs, source_labels = source_batch['img'].data[0].cuda(
        ), source_batch['gt_semantic_seg'].data[0].cuda()

        target_batch = target_batch.data[0]

        target_img_day = torch.cat([
            target_batch[0]['img_day'].unsqueeze(0),
            target_batch[1]['img_day'].unsqueeze(0)
        ],
                                   dim=0).cuda()

        target_img_night = torch.cat([
            target_batch[0]['img_night'].unsqueeze(0),
            target_batch[1]['img_night'].unsqueeze(0)
        ],
                                     dim=0).cuda()

        ########################################################################################### # noqa

        # Train Segmentor
        set_require_grad(discriminator_1, False)
        # set_require_grad(discriminator_2, False)
        set_require_grad(model, True)
        seg_optimizer.zero_grad()

        ############################################################################################ # noqa
        # Train with source

        # source_predicts = model.encode_decode(source_imgs, img_metas=dict())
        source_feat = model.extract_feat(source_imgs)
        pred = model._decode_head_forward_test(source_feat, dict())
        source_predicts = resize(input=pred,
                                 size=source_imgs.shape[2:],
                                 mode='bilinear',
                                 align_corners=False)
        loss_source = ce_loss(source_predicts, source_labels.squeeze(1))
        loss_source.backward()

        ############################################################################################ # noqa

        # Train with target

        target_feat = model.extract_feat(target_img_day)
        pred = model._decode_head_forward_test(target_feat, dict())
        target_predicts_day = resize(input=pred,
                                     size=target_img_day.shape[2:],
                                     mode='bilinear',
                                     align_corners=False)

        # target_predicts_day = model.encode_decode(target_img_day,
        #                                           img_metas=dict())  # noqa

        target_feat_night = model.extract_feat(target_img_night)
        pred = model._decode_head_forward_test(target_feat_night, dict())
        target_predicts_night = resize(input=pred,
                                       size=target_img_night.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)

        d1_out = discriminator_1(target_feat_night)
        d1_label = torch.FloatTensor(d1_out.data.size()).fill_(1).cuda()
        # loss_adv_target_day = least_square_loss(d1_out, d1_label) * 0.01
        loss_adv_target_night = bce_loss(d1_out, d1_label)
        # loss_adv_target_day.backward(retain_graph=True)

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
        # pseudo_gt[pseudo_gt >= 11] = 255

        # loss_static = static_loss(target_predicts_night[:, :11, :, :],
        #                           pseudo_gt)
        loss_static = static_loss(target_predicts_night, pseudo_gt)

        loss = loss_static + 0.01 * loss_adv_target_night  # noqa
        loss.backward()

        loss_adv_value = loss_adv_target_night.item()

        loss_seg_value = loss_source.item()
        loss_static_value = loss_static.item()
        seg_optimizer.step()
        seg_lr_updater.set_lr(seg_optimizer, i)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if i % seg_iters != 0:
            iter_time = time.time() - start_time
            eta = iter_time * (max_iters - i)
            mins, s = divmod(eta, 60)
            hours, minute = divmod(mins, 60)
            days, hour = divmod(hours, 24)
            ETA = f'{int(days)}天{int(hour)}小时{int(minute)}分{int(s)}秒'
            if rank == 0:
                print(
                    'Iter-[{0:5d}|{1:6d}] loss_adv_source:         loss_adv_target_day:         loss_adv_target_night:         loss_source_value: {2:.5f} loss_static_value: {3:.5f} lr_seg: {4:.5f} lr_adv_1: {5:.5f}  ETA: {6} '  # noqa
                    .format(i, max_iters, loss_seg_value, loss_static_value,
                            seg_optimizer.param_groups[0]['lr'],
                            adv_optimizer_1.param_groups[0]['lr'], ETA))
            continue

        ########################################################################################### # noqa

        # Train Discriminator network
        set_require_grad(discriminator_1, True)
        # set_require_grad(discriminator_2, True)
        set_require_grad(model, False)
        adv_optimizer_1.zero_grad()
        # adv_optimizer_2.zero_grad()

        ########################################################################################### # noqa
        # Train with source

        source_feat = source_feat.detach().contiguous()
        d1_out = discriminator_1(source_feat)
        d1_label = torch.FloatTensor(d1_out.data.size()).fill_(0).cuda()
        # loss_d1 = least_square_loss(d1_out, d1_label) * 0.5
        loss_d1 = bce_loss(d1_out, d1_label) * 0.5
        loss_d1.backward()

        # source_predicts = source_predicts.detach().contiguous()
        # d2_out = discriminator_2(source_predicts)
        # d2_label = torch.FloatTensor(d2_out.data.size()).fill_(0).cuda()
        # loss_d2 = least_square_loss(d2_out, d2_label) * 0.5
        # loss_d2 = bce_loss(d2_out, d2_label) * 0.5
        # loss_d2.backward()

        loss_adv_value += loss_d1.item()

        ########################################################################################### # noqa
        # Train with target

        target_feat_night = target_feat_night.detach().contiguous()

        d1_out = discriminator_1(target_feat_night)
        d1_label = torch.FloatTensor(d1_out.data.size()).fill_(1).cuda()
        # loss_d1 = least_square_loss(d1_out, d1_label)
        loss_d1 = bce_loss(d1_out, d1_label) * 0.5
        loss_d1.backward()
        adv_optimizer_1.step()
        adv_lr_updater.set_lr(adv_optimizer_1, i)

        # d2_out = discriminator_2(target_predicts_night)
        # d2_label = torch.FloatTensor(d2_out.data.size()).fill_(1).cuda()
        # loss_d2 = least_square_loss(d2_out, d2_label)
        # loss_d2 = bce_loss(d2_out, d2_label)
        # loss_d2.backward()
        # adv_optimizer_2.step()
        # adv_lr_updater.set_lr(adv_optimizer_2, i)

        ########################################################################################### # noqa

        iter_time = time.time() - start_time
        eta = iter_time * (max_iters - i)
        mins, s = divmod(eta, 60)
        hours, minute = divmod(mins, 60)
        days, hour = divmod(hours, 24)
        ETA = f'{int(days)}天{int(hour)}小时{int(minute)}分{int(s)}秒'

        if rank == 0:
            print(
                'Iter-[{0:5d}|{1:6d}] loss_adv_source: {2:.5f} loss_adv_target_day: {3:.5f} loss_adv_target_night: {4:.5f} loss_source_value: {5:.5f} loss_static_value: {6:.5f} lr_seg: {7:.5f} lr_adv_1: {8:.5f} lr_adv_2: {9:.5f} ETA: {10} '  # noqa
                .format(
                    i,
                    max_iters,
                    loss_adv_value,
                    loss_d1.item(),
                    # loss_d2.item(), loss_seg_value, loss_static_value,
                    seg_optimizer.param_groups[0]['lr'],
                    adv_optimizer_1.param_groups[0]['lr'],
                    # adv_optimizer_2.param_groups[0]['lr'],
                    ETA))

        if i % 4000 == 0 and i != 0:
            model.eval()
            results = single_gpu_test(MMDataParallel(model, device_ids=[0]),
                                      test_dataloader,
                                      pre_eval=False,
                                      format_only=False)
            metric = test_dataset.evaluate(results, metric='mIoU')
            print(metric)
            model.train()
            print('taking snapshot ...')
            torch.save(model.state_dict(),
                       osp.join(work_dirs, f'segmentor_{i}.pth'))

            torch.save(discriminator_1.state_dict(),
                       osp.join(work_dirs, f'discriminator_1_{i}.pth'))
            # torch.save(discriminator_2.state_dict(),
            #            osp.join(work_dirs, f'discriminator_2_{i}.pth'))


if __name__ == '__main__':
    # log_file
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    work_dirs = osp.join('work_dirs', f'ciuda_{timestamp}')
    mmcv.mkdir_or_exist(work_dirs)
    log_file = osp.join(work_dirs, f'{timestamp}.log')
    sys.stdout = Logger(log_file)
    main(max_iters=40000, work_dirs=work_dirs, distributed=True)
