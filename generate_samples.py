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

from mmseg.ops import resize

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a GAN model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10,
                        help='batch size of dataloader')
    parser.add_argument(
        '--samples-path',
        type=str,
        default=None,
        help='path to store images. If not given, remove it after evaluation\
             finished')
    parser.add_argument('--sample-model',
                        type=str,
                        default='ema',
                        help='use which mode (ema/orig) in sampling')
    parser.add_argument('--eval',
                        nargs='*',
                        type=str,
                        default=None,
                        help='select the metrics you want to access')
    parser.add_argument('--online',
                        action='store_true',
                        help='whether to use online mode for evaluation')
    parser.add_argument('--num-samples',
                        type=int,
                        default=-1,
                        help='whether to use online mode for evaluation')
    parser.add_argument('--save-path',
                        type=str,
                        default='./work_dirs/demos/translation_sample.png',
                        help='path to save translation sample')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--sample-cfg',
                        nargs='+',
                        action=DictAction,
                        help='Other customized kwargs for sampling function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    dirname = os.path.dirname(args.checkpoint)
    ckpt = os.path.basename(args.checkpoint)

    if 'http' in args.checkpoint:
        log_path = None
    else:
        log_name = ckpt.split('.')[0] + '_eval_log' + '.txt'
        log_path = os.path.join(dirname, log_name)

    logger = get_root_logger(log_file=log_path,
                             log_level=cfg.log_level,
                             file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            mmcv.print_log(f'set random seed to {args.seed}', 'mmgen')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    model = build_model(cfg.model,
                        train_cfg=cfg.train_cfg,
                        test_cfg=cfg.test_cfg)

    mmcv.print_log(f'Sampling model: {args.sample_model}', 'mmgen')

    model.eval()

    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(test_dataset,
                                       samples_per_gpu=1,
                                       workers_per_gpu=2,
                                       num_gpus=1,
                                       dist=False)

    # img = mmcv.imread('aachen_000000_000019_leftImg8bit.png')

    # test_pipeline = Compose(cfg.test_pipeline)
    # img_path = 'aachen_000000_000019_leftImg8bit.png'
    # device = next(model.parameters()).device
    # data = dict(img_a_path=img_path, img_b_path=img_path, img_c_path=img_path) # noqa
    # data = test_pipeline(data)
    # data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    progress_bar = mmcv.ProgressBar(len(test_dataset))
    for idx, data in enumerate(test_dataloader):
        # print(data['meta'])
        name = str(data['meta'].data[0][0]['img_night_path']).replace(
            './data/city2darkzurich/testB/', '')
        # print(name)
        
        dirname = os.path.dirname(name)
        # print(dirname)
        
        name = name.replace(dirname, '')
        # dirname = dirname.replace('\\', '/')
        
        with torch.no_grad():
            results = model(img=data['img_night'], test_mode=True, target_domain='day')
        # save_dir = os.path.dirname(args.save_path) + '/' + dirname
        save_dir = os.path.dirname(args.save_path)
        # print(save_dir + '/' + name)
        # raise '123'
        mmcv.mkdir_or_exist(save_dir)
        img = (results['target'][:, [2, 1, 0]] + 1.) / 2.
        # img = resize(img, scale_factor=2, align_corners=False, mode='bicubic')
        # print(save_dir + '/' + dirname+'/' + dirname+name[1:])
        # raise '123'
        utils.save_image(img, save_dir + '/' + dirname+'/' + dirname+name[1:])
        progress_bar.update()



if __name__ == '__main__':
    main()
