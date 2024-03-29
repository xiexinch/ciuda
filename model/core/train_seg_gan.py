# Copyright (c) OpenMMLab. All rights reserved.
import os
from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, IterBasedRunner, OptimizerHook, build_runner
from mmcv.runner import set_random_seed as set_random_seed_mmcv
from mmcv.utils import build_from_cfg

from mmgen.core.ddp_wrapper import DistributedDataParallelWrapper
from mmgen.core.optimizer import build_optimizers
from mmgen.core.runners.apex_amp_utils import apex_amp_initialize
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.utils import get_root_logger

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataset as build_mmseg_dataset, build_dataloader as build_mmseg_dataloader


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the similar
    function defined in MMCV.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    set_random_seed_mmcv(seed,
                         deterministic=deterministic,
                         use_rank_shift=use_rank_shift)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            persistent_workers=cfg.data.get('persistent_workers', True),
            seed=cfg.seed) for ds in dataset
    ]

    # dirty code for use apex amp
    # apex.amp request that models should be in cuda device before
    # initialization.
    if cfg.get('apex_amp', None):
        assert distributed, (
            'Currently, apex.amp is only supported with DDP training.')
        model = model.cuda()

    # build optimizer
    if cfg.optimizer:
        optimizer = build_optimizers(model, cfg.optimizer)
    # In GANs, we allow building optimizer in GAN model.
    else:
        optimizer = None

    _use_apex_amp = False
    if cfg.get('apex_amp', None):
        model, optimizer = apex_amp_initialize(model, optimizer,
                                               **cfg.apex_amp)
        _use_apex_amp = True

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        use_ddp_wrapper = cfg.get('use_ddp_wrapper', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_ddp_wrapper:
            mmcv.print_log('Use DDP Wrapper.', 'mmgen')
            model = DistributedDataParallelWrapper(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]),
                               device_ids=cfg.gpu_ids)

    # allow users to define the runner
    if cfg.get('runner', None):
        runner = build_runner(
            cfg.runner,
            dict(model=model,
                 optimizer=optimizer,
                 work_dir=cfg.work_dir,
                 logger=logger,
                 use_apex_amp=_use_apex_amp,
                 meta=meta))
    else:
        runner = IterBasedRunner(model,
                                 optimizer=optimizer,
                                 work_dir=cfg.work_dir,
                                 logger=logger,
                                 meta=meta)
        # set if use dynamic ddp in training
        # is_dynamic_ddp=cfg.get('is_dynamic_ddp', False))
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)

    # In GANs, we can directly optimize parameter in `train_step` function.
    if cfg.get('optimizer_cfg', None) is None:
        optimizer_config = None
    elif fp16_cfg is not None:
        raise NotImplementedError('Fp16 has not been supported.')
        # optimizer_config = Fp16OptimizerHook(
        #     **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # default to use OptimizerHook
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # update `out_dir` in  ckpt hook
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config['out_dir'] = os.path.join(
            cfg.work_dir, cfg.checkpoint_config.get('out_dir', 'ckpt'))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # # register eval hooks
    # if validate:
    #     val_dataset = build_mmseg_dataset(cfg.data.val, dict(test_mode=True))
    #     val_dataloader = build_mmseg_dataloader(
    #         val_dataset,
    #         samples_per_gpu=1,
    #         workers_per_gpu=cfg.data.workers_per_gpu,
    #         dist=distributed,
    #         shuffle=False)
    #     eval_cfg = cfg.get('evaluation', {})
    #     eval_cfg['by_epoch'] = False
    #     eval_hook = DistEvalHook if distributed else EvalHook
    #     # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
    #     # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
    #     runner.register_hook(eval_hook(val_dataloader, **eval_cfg),
    #                          priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)
