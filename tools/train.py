# Copyright (c) OpenMMLab. All rights reserved.
"""Train AsyncBEV with the CMT/MMDetection3D 1.0.0rc5 runner.

Adapted from the CMT training entry point; see tools/README.md for provenance.
"""

import argparse
import copy
import os
import sys
import time
import warnings
from os import path as osp
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._common import import_plugins


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train an AsyncBEV detector')
    parser.add_argument('config', help='training config file')
    parser.add_argument('--work-dir', help='directory for logs and checkpoints')
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume-from', help='resume optimizer and epoch state '
                        'from this checkpoint (overrides configured resume)')
    resume.add_argument('--auto-resume', action='store_true',
                        help='resume the latest checkpoint in work_dir')
    resume.add_argument('--no-resume', action='store_true',
                        help='clear configured resume_from and auto_resume; '
                        'load_from still initializes model weights')
    parser.add_argument('--no-validate', action='store_true',
                        help='disable evaluation during training')
    gpus = parser.add_mutually_exclusive_group()
    gpus.add_argument('--gpu-id', type=int, default=0,
                      help='logical CUDA device for non-distributed training')
    gpus.add_argument('--gpus', type=int,
                      help='deprecated; non-distributed training uses one GPU')
    gpus.add_argument('--gpu-ids', type=int, nargs='+',
                      help='deprecated; only the first device is used')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--diff-seed', action='store_true',
                        help='add distributed rank to the random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='enable deterministic cuDNN behavior')
    overrides = parser.add_mutually_exclusive_group()
    overrides.add_argument('--cfg-options', nargs='+', metavar='KEY=VALUE',
                           help='override config with key=value pairs; quote '
                           'list/tuple values containing shell characters')
    overrides.add_argument('--options', nargs='+', metavar='KEY=VALUE',
                           help='deprecated alias for --cfg-options')
    parser.add_argument('--launcher', default='none',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help='distributed job launcher')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank',
                        type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true',
                        help='scale optimizer lr by total world size / 8 '
                        '(the upstream CMT eight-GPU reference)')
    args = parser.parse_args(argv)
    if args.gpu_id < 0 or (args.gpu_ids and min(args.gpu_ids) < 0):
        parser.error('GPU IDs must be nonnegative logical device indices')
    if args.gpus is not None and args.gpus < 1:
        parser.error('--gpus must be positive')
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if args.options is not None:
        warnings.warn('--options is deprecated; use --cfg-options')
        args.cfg_options = args.options
    if args.cfg_options is not None:
        # Defer MMCV until overrides are needed, so --help needs only Python.
        from mmcv import DictAction
        action = DictAction(option_strings=[], dest='cfg_options')
        action(parser, args, args.cfg_options)
    return args


def configure_checkpoint(cfg, args):
    """Apply explicit resume choices without removing weight initialization."""
    if args.no_resume:
        cfg.resume_from = None
        cfg.auto_resume = False
    elif args.resume_from is not None:
        cfg.resume_from = args.resume_from
        cfg.auto_resume = False
    elif args.auto_resume:
        cfg.resume_from = None
        cfg.auto_resume = True


def scale_learning_rate(cfg, world_size):
    """Apply CMT's opt-in linear rule after distributed world size is known."""
    cfg.optimizer['lr'] = cfg.optimizer['lr'] * world_size / 8


def main(argv=None):
    args = parse_args(argv)

    import mmcv
    import torch
    from mmcv import Config
    from mmcv.runner import get_dist_info, init_dist
    from mmdet import __version__ as mmdet_version
    from mmdet.apis import set_random_seed
    from mmdet3d import __version__ as mmdet3d_version
    from mmdet3d.apis import init_random_seed, train_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import collect_env, get_root_logger
    from mmseg import __version__ as mmseg_version

    try:
        from mmdet.utils import setup_multi_processes
    except ImportError:
        from mmdet3d.utils import setup_multi_processes

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    configure_checkpoint(cfg, args)
    setup_multi_processes(cfg)
    import_plugins(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir') is None:
        cfg.work_dir = osp.join('work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.gpus is not None:
        warnings.warn('--gpus is deprecated; use --gpu-id. '
                      'Non-distributed training uses one GPU.')
        cfg.gpu_ids = [0]
    elif args.gpu_ids is not None:
        warnings.warn('--gpu-ids is deprecated; use --gpu-id. '
                      'Only the first logical GPU is used.')
        cfg.gpu_ids = args.gpu_ids[:1]
    else:
        cfg.gpu_ids = [args.gpu_id]

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))
    else:
        # Plugin constructors may allocate on the current CUDA device before
        # the runner wraps the model using cfg.gpu_ids.
        torch.cuda.set_device(cfg.gpu_ids[0])
    rank, world_size = get_dist_info()
    if distributed:
        cfg.gpu_ids = list(range(world_size))
    if args.autoscale_lr:
        scale_learning_rate(cfg, world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger_name = 'mmseg' if cfg.model.type == 'EncoderDecoder3D' else 'mmdet'
    logger = get_root_logger(log_file=log_file,
                             log_level=cfg.get('log_level', 'INFO'),
                             name=logger_name)

    seed = init_random_seed(args.seed)
    if args.diff_seed:
        seed += rank
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    if rank == 0:
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    env_info = '\n'.join('{}: {}'.format(k, v)
                         for k, v in collect_env().items())
    logger.info('Environment info:\n%s', env_info)
    logger.info('Distributed training: %s; world size: %s',
                distributed, world_size)
    logger.info('Random seed: %s; deterministic: %s', seed, args.deterministic)
    logger.info('Checkpoint policy: resume_from=%s, auto_resume=%s, load_from=%s',
                cfg.get('resume_from'), cfg.get('auto_resume', False),
                cfg.get('load_from'))
    logger.info('Config:\n%s', cfg.pretty_text)
    meta = dict(env_info=env_info, config=cfg.pretty_text, seed=seed,
                exp_name=osp.basename(args.config))

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    logger.info('Model:\n%s', model)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.get('checkpoint_config') is not None:
        checkpoint_meta = dict(cfg.checkpoint_config.get('meta', {}))
        checkpoint_meta.update(
            mmdet_version=mmdet_version, mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version, config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=getattr(datasets[0], 'PALETTE', None))
        cfg.checkpoint_config.meta = checkpoint_meta
    model.CLASSES = datasets[0].CLASSES
    train_model(model, datasets, cfg, distributed=distributed,
                validate=not args.no_validate, timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    main()
