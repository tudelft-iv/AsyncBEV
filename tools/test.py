# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate AsyncBEV with the MMCV 1.x / MMDetection3D 1.0.0rc5 stack.

Adapted from CMT's test entry point and the AsyncBEV development
``tools/cmt/test_cmt{,_dynamic}.py`` scripts. The two evaluation paths share
inference; ``--eval-dynamic`` selects the repository's all/static/dynamic
evaluator without changing its metric definitions.
"""

import argparse
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools._common import import_plugins


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='test configuration file')
    parser.add_argument('checkpoint', help='checkpoint file or MMCV-supported URL')
    parser.add_argument('--out', help='save raw predictions to a .pkl/.pickle file')
    parser.add_argument(
        '--work-dir', help='persistent evaluation directory; default: '
        'outputs/eval/<config stem>/<checkpoint stem>')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='logical GPU index for non-distributed inference')
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                        help='deprecated alias; only the first GPU is used')
    parser.add_argument('--format-only', action='store_true',
                        help='write nuScenes submission JSON without evaluation')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='dataset metrics, e.g. --eval bbox')
    parser.add_argument(
        '--eval-dynamic', action='store_true',
        help='run CustomNuScenesDataset all/static/dynamic evaluation '
        '(implies --eval bbox)')
    parser.add_argument('--show', action='store_true', help='show predictions')
    parser.add_argument('--show-dir', help='save visualizations (single GPU only)')
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir', help='distributed CPU result-collection directory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    # Resolve DictAction after parsing so --help works without the CUDA stack.
    parser.add_argument('--cfg-options', nargs='+', metavar='KEY=VALUE',
                        help='MMCV config overrides, including dotted keys')
    parser.add_argument('--eval-options', nargs='+', metavar='KEY=VALUE',
                        help='dataset evaluator/formatter keyword arguments')
    parser.add_argument('--options', nargs='+', metavar='KEY=VALUE',
                        help='deprecated alias for --eval-options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args(argv)
    if args.options is not None and args.eval_options is not None:
        parser.error('--options and --eval-options cannot be combined')
    if args.options is not None:
        warnings.warn('--options is deprecated; use --eval-options')
        args.eval_options = args.options
    for name in ('cfg_options', 'eval_options'):
        values = getattr(args, name)
        if values is not None:
            from mmcv import DictAction
            action = DictAction(option_strings=[], dest=name)
            action(parser, args, values)
    validate_args(args, parser)
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def validate_args(args, parser):
    if args.eval_dynamic:
        if args.eval is not None and args.eval != ['bbox']:
            parser.error('--eval-dynamic supports only --eval bbox')
        args.eval = ['bbox']
    if not any((args.out, args.eval, args.format_only, args.show, args.show_dir)):
        parser.error('specify --eval, --eval-dynamic, --out, --format-only, '
                     '--show, or --show-dir')
    if args.eval and args.format_only:
        parser.error('--eval/--eval-dynamic cannot be combined with --format-only')
    if args.out and not args.out.endswith(('.pkl', '.pickle')):
        parser.error('--out must end in .pkl or .pickle')
    if args.launcher != 'none' and (args.show or args.show_dir):
        parser.error('--show and --show-dir require non-distributed inference')
    if args.show and not args.show_dir:
        parser.error('--show requires --show-dir for the MMDetection3D visualizer')
    if args.gpu_ids is not None:
        warnings.warn('--gpu-ids is deprecated; using its first entry')
        args.gpu_id = args.gpu_ids[0]
    if args.gpu_id < 0:
        parser.error('--gpu-id must be a non-negative logical GPU index')


def evaluation_kwargs(cfg, args, work_dir):
    """Remove runner-only options and retain the dataset's actual arguments."""
    kwargs = dict(cfg.get('evaluation', {})) if args.eval else {}
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'by_epoch', 'broadcast_bn_buffer', 'dynamic_intervals'):
        kwargs.pop(key, None)
    kwargs.update(args.eval_options or {})
    if args.eval:
        # The CLI selects the evaluation path/metrics, even when a config has
        # an evaluation hook with different metrics.
        kwargs['metric'] = args.eval
    # A temporary prefix would be deleted by NuScenesDataset.evaluate(),
    # losing the result JSON and metrics summaries after a successful run.
    if not kwargs.get('jsonfile_prefix'):
        kwargs['jsonfile_prefix'] = str(Path(work_dir) / 'results')
    return kwargs


def evaluate_outputs(dataset, outputs, args, kwargs):
    evaluator = dataset.evaluate_dynamic if args.eval_dynamic else dataset.evaluate
    return evaluator(outputs, **kwargs)


def main(argv=None):
    args = parse_args(argv)

    import mmcv
    import torch
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                             wrap_fp16_model)
    from mmdet.apis import multi_gpu_test, set_random_seed
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    try:
        from mmdet.utils import compat_cfg, setup_multi_processes
    except ImportError:
        from mmdet3d.utils import compat_cfg, setup_multi_processes

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugins(cfg, args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.gpu_ids = [args.gpu_id]
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))
    else:
        torch.cuda.set_device(args.gpu_id)
    rank, world_size = get_dist_info()

    work_dir = Path(args.work_dir) if args.work_dir else (
        Path('outputs/eval') / Path(args.config).stem / Path(args.checkpoint).stem)
    cfg.work_dir = str(work_dir)
    if rank == 0:
        work_dir.mkdir(parents=True, exist_ok=True)
        if args.show_dir:
            Path(args.show_dir).mkdir(parents=True, exist_ok=True)
        print('Evaluation output directory: {}'.format(work_dir.resolve()))

    loader_cfg = dict(samples_per_gpu=1,
                      workers_per_gpu=cfg.data.get('workers_per_gpu', 2))
    loader_cfg.update(cfg.data.get('test_dataloader', {}))
    # Evaluation requires ordered results and the process group's actual mode.
    loader_cfg.update(dist=distributed, shuffle=False)
    test_configs = cfg.data.test if isinstance(cfg.data.test, list) else [cfg.data.test]
    for test_cfg in test_configs:
        test_cfg.test_mode = True
        if loader_cfg['samples_per_gpu'] > 1:
            test_cfg.pipeline = replace_ImageToTensor(test_cfg.pipeline)

    cfg.seed = args.seed
    set_random_seed(args.seed, deterministic=args.deterministic)
    dataset = build_dataset(cfg.data.test)
    if args.eval_dynamic and not callable(getattr(dataset, 'evaluate_dynamic', None)):
        raise TypeError('--eval-dynamic requires a dataset exposing evaluate_dynamic; '
                        'use a single CustomNuScenesDataset test split')
    kwargs = evaluation_kwargs(cfg, args, work_dir)
    if rank == 0:
        cfg.dump(str(work_dir / 'resolved_config.py'))
        mmcv.dump(dict(
            args=vars(args), config_path=str(Path(args.config).resolve()),
            checkpoint=args.checkpoint, seed=args.seed,
            launcher=args.launcher, distributed=distributed,
            world_size=world_size, logical_gpu=torch.cuda.current_device(),
            evaluator=('evaluate_dynamic' if args.eval_dynamic else
                       'evaluate' if args.eval else
                       'format_results' if args.format_only else None),
            evaluation_kwargs=kwargs if args.eval or args.format_only else {},
        ), str(work_dir / 'run_info.json'))
    data_loader = build_dataloader(dataset, **loader_cfg)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    metadata = checkpoint.get('meta') or {}
    model.CLASSES = metadata.get('CLASSES', dataset.CLASSES)
    if 'PALETTE' in metadata:
        model.PALETTE = metadata['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    else:
        model = MMDataParallel(model.cuda(args.gpu_id), device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

    if rank == 0:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            mmcv.dump(outputs, args.out)
            print('Raw predictions: {}'.format(args.out))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
            print('Formatted results: {}'.format(kwargs['jsonfile_prefix']))
        if args.eval:
            metrics = evaluate_outputs(dataset, outputs, args, kwargs)
            mmcv.dump(metrics, str(work_dir / 'metrics.json'))
            print(metrics)
            print('Evaluation results: {}'.format(kwargs['jsonfile_prefix']))


if __name__ == '__main__':
    main()
