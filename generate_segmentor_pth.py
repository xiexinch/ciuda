import argparse
import os

import mmcv
import torch
from torchvision import utils
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmgen.apis import set_random_seed

from mmgen.datasets import build_dataloader, build_dataset, Compose  # noqa
from mmgen.models import build_model
from mmgen.utils import get_root_logger

from model.gans import RoundGAN  # noqa
from dataset import RoundImageDataset  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a GAN model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the model and load checkpoint
    model = build_model(cfg.model,
                        train_cfg=cfg.train_cfg,
                        test_cfg=cfg.test_cfg)

    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    segmentor = model.segmentors['a']
    torch.save(segmentor.state_dict(), 's_segformer_iter_80000.pth')


if __name__ == '__main__':
    main()
