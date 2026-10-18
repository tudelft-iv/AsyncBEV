"""CPU-only regression checks for training orchestration; no ML imports."""

import argparse
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import _common, train


class ConfigDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class PluginTests(unittest.TestCase):
    def test_plugin_paths_keep_last_component(self):
        relative = 'projects/CMT/mmdet3d_plugin'
        for value in (relative, relative + '/',
                      str(_common.REPO_ROOT / relative)):
            with self.subTest(value=value), patch.object(
                    _common.importlib, 'import_module') as import_module:
                _common.import_plugins(dict(plugin=True, plugin_dir=value),
                                       'unused.py')
                import_module.assert_called_once_with(
                    'projects.CMT.mmdet3d_plugin')

    def test_custom_imports_work_when_plugin_disabled(self):
        custom_import = Mock()
        with patch.dict(sys.modules, {
                'mmcv': module('mmcv'),
                'mmcv.utils': module('mmcv.utils',
                                     import_modules_from_strings=custom_import)
        }), patch.object(_common.importlib, 'import_module') as import_module:
            _common.import_plugins(
                dict(plugin=False, custom_imports=dict(imports=['example'])),
                'unused.py')
        custom_import.assert_called_once_with(imports=['example'])
        import_module.assert_not_called()


class TrainingTests(unittest.TestCase):
    def test_resume_choices_override_inherited_state(self):
        for choice, expected in (
                ({'no_resume': True}, (None, False)),
                ({'resume_from': 'new.pth'}, ('new.pth', False)),
                ({'auto_resume': True}, (None, True)),
                ({}, ('old.pth', True))):
            with self.subTest(choice=choice):
                args = dict(no_resume=False, resume_from=None,
                            auto_resume=False)
                args.update(choice)
                cfg = ConfigDict(resume_from='old.pth', auto_resume=True,
                                 load_from='initial.pth')
                train.configure_checkpoint(cfg, SimpleNamespace(**args))
                self.assertEqual((cfg.resume_from, cfg.auto_resume), expected)
                self.assertEqual(cfg.load_from, 'initial.pth')

    def test_resume_flags_conflict_and_rank_aliases_parse(self):
        with patch.dict(sys.modules, {'mmcv': module(
                'mmcv', DictAction=argparse.Action)}):
            for flag in ('--local_rank', '--local-rank'):
                args = train.parse_args(['config.py', flag, '3', '--no-resume'])
                self.assertEqual(args.local_rank, 3)
                self.assertTrue(args.no_resume)
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                train.parse_args(['config.py', '--resume-from', 'x.pth',
                                  '--no-resume'])

    def run_mock_training(self, distributed, rank, world_size, gpu_id=0):
        cfg = ConfigDict(
            model=ConfigDict(type='CmtDetector'),
            data=ConfigDict(train=ConfigDict(pipeline=[])), workflow=[('train', 1)],
            optimizer=dict(lr=0.008), dist_params=dict(backend='nccl'),
            checkpoint_config=ConfigDict(meta=dict(experiment='preserved')),
            resume_from='old.pth', load_from='initial.pth',
            work_dir='unused-output', pretty_text='resolved configuration',
            dump=Mock())
        args = SimpleNamespace(
            config='config.py', cfg_options=None, work_dir=None,
            no_resume=True, resume_from=None, auto_resume=False,
            gpu_id=gpu_id, gpu_ids=None, gpus=None, seed=17, diff_seed=True,
            deterministic=False, launcher='pytorch' if distributed else 'none',
            autoscale_lr=True, no_validate=False)
        init_dist = Mock()
        def get_dist_info():
            if distributed:
                init_dist.assert_called_once_with('pytorch', backend='nccl')
            return rank, world_size
        model = SimpleNamespace(init_weights=Mock())
        set_device = Mock()
        def build_model(*args, **kwargs):
            if distributed:
                set_device.assert_not_called()
            else:
                set_device.assert_called_once_with(gpu_id)
            return model
        dataset = SimpleNamespace(CLASSES=('car',))
        train_model, set_seed = Mock(), Mock()
        modules = {
            'mmcv': module('mmcv', Config=SimpleNamespace(fromfile=lambda _: cfg),
                           mkdir_or_exist=Mock()),
            'mmcv.runner': module('mmcv.runner', init_dist=init_dist,
                                  get_dist_info=get_dist_info),
            'torch': module('torch', cuda=SimpleNamespace(set_device=set_device),
                            backends=SimpleNamespace(
                                cudnn=SimpleNamespace(benchmark=False))),
            'mmdet': module('mmdet', __version__='2.24.0'),
            'mmdet.apis': module('mmdet.apis', set_random_seed=set_seed),
            'mmdet.utils': module('mmdet.utils', setup_multi_processes=Mock()),
            'mmdet3d': module('mmdet3d', __version__='1.0.0rc5'),
            'mmdet3d.apis': module('mmdet3d.apis',
                                   init_random_seed=lambda seed: seed,
                                   train_model=train_model),
            'mmdet3d.datasets': module('mmdet3d.datasets',
                                       build_dataset=lambda _: dataset),
            'mmdet3d.models': module('mmdet3d.models',
                                     build_model=build_model),
            'mmdet3d.utils': module('mmdet3d.utils', collect_env=lambda: {},
                                    get_root_logger=lambda **kw: Mock()),
            'mmseg': module('mmseg', __version__='0.29.1'),
        }
        with patch.dict(sys.modules, modules), patch.object(
                train, 'parse_args', return_value=args), patch.object(
                train, 'import_plugins'):
            train.main()
        self.assertAlmostEqual(cfg.optimizer['lr'], 0.008 * world_size / 8)
        self.assertEqual(cfg.seed, 17 + rank)
        set_seed.assert_called_once_with(17 + rank, deterministic=False)
        self.assertEqual(cfg.dump.call_count, 1 if rank == 0 else 0)
        self.assertEqual(cfg.checkpoint_config.meta['experiment'], 'preserved')
        self.assertIsNone(cfg.resume_from)
        self.assertEqual(cfg.load_from, 'initial.pth')
        self.assertEqual(train_model.call_count, 1)
        self.assertEqual(train_model.call_args[1]['distributed'], distributed)
        self.assertTrue(train_model.call_args[1]['validate'])

    def test_single_process_diff_seed_needs_no_process_group(self):
        self.run_mock_training(distributed=False, rank=0, world_size=1)

    def test_lr_uses_initialized_world_size_and_worker_does_not_dump(self):
        self.run_mock_training(distributed=True, rank=2, world_size=4)

    def test_nonzero_gpu_is_current_before_model_construction(self):
        self.run_mock_training(distributed=False, rank=0, world_size=1, gpu_id=3)


if __name__ == '__main__':
    unittest.main()
