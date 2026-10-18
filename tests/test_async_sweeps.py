"""Exercise the real async sweep loader without importing CUDA or MMDetection."""

import ast
import runpy
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'projects/CMT/mmdet3d_plugin/datasets/pipelines/loading_async.py'


class ArrayPoints:
    """Small array-backed stand-in for the BasePoints operations used here."""

    def __init__(self, values):
        self.tensor = np.array(values, dtype=np.float64, copy=True)

    def __getitem__(self, item):
        return type(self)(self.tensor[item])

    def new_point(self, values):
        return type(self)(values)

    @classmethod
    def cat(cls, items):
        return cls(np.concatenate([item.tensor for item in items], axis=0))


class SweepParent:
    """Only IO and point filtering are stubbed; loader logic stays untouched."""

    def __init__(self, sweeps_num, load_dim, use_dim, file_client_args,
                 pad_empty_sweeps, remove_close, test_mode):
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.point_files = {}

    def _load_points(self, path):
        return self.point_files[path]

    @staticmethod
    def _remove_close(points):
        values = points.tensor if isinstance(points, ArrayPoints) else points
        keep = ~((np.abs(values[:, 0]) < 1.0) &
                 (np.abs(values[:, 1]) < 1.0))
        return points[keep]


def load_actual_class():
    tree = ast.parse(SOURCE.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and
                node.name == 'LoadAsyncPointsFromMultiSweeps')
    node.decorator_list = []  # Registry import is unnecessary for this test.
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = {'np': np, 'LoadPointsFromMultiSweeps': SweepParent}
    exec(compile(ast.fix_missing_locations(isolated), str(SOURCE), 'exec'),
         namespace)
    return namespace[node.name]


class AsyncSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader_class = load_actual_class()

    def make_results(self, with_sweep=True):
        # Both frame timestamps are seconds; individual sweeps use microseconds.
        sweep = dict(
            data_path='older.bin', timestamp=9750000,
            sensor2lidar_rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            sensor2lidar_translation=np.array([1, 2, 3]))
        return dict(
            points=ArrayPoints([[2, 3, 4, 11, 999]]),
            timestamp=10.25,
            async_prev_sweeps=[sweep] if with_sweep else [],
            async_metas=dict(
                timestamp=10.0,
                sensor2lidar_rotation=np.array(
                    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                sensor2lidar_translation=np.array([10, 20, 30])))

    def run_loader(self, **kwargs):
        loader = self.loader_class(**kwargs)
        loader.point_files['older.bin'] = np.array(
            [[4, 5, 6, 12, 999]], dtype=np.float32)
        results = self.make_results()
        self.assertIs(loader(results), results)
        return results['points'].tensor

    def test_retained_baseline_train_and_test_configs_construct_loader(self):
        config = runpy.run_path(str(
            ROOT / 'projects/CMT/configs/fusion/cmt_sensor_staleness_lidar.py'))
        for pipeline in ('train_pipeline', 'test_pipeline'):
            kwargs = next(item.copy() for item in config[pipeline]
                          if item['type'] == 'LoadAsyncPointsFromMultiSweeps')
            kwargs.pop('type')
            with self.subTest(pipeline=pipeline):
                loader = self.loader_class(**kwargs)
                self.assertTrue(loader.sensor_staleness_feats)
                self.assertFalse(loader.early_motion_compensation)
                results = loader(self.make_results(with_sweep=False))
                # The actual configs request ten padded sweeps and all five dims.
                expected = np.tile([2, 3, 4, 11, 0.25], (11, 1))
                np.testing.assert_allclose(results['points'].tensor, expected)

    def test_default_keeps_async_coordinates_and_relative_sweep_age(self):
        values = self.run_loader()
        # Default use_dim drops intensity, retaining time as the last column.
        np.testing.assert_allclose(values, [[2, 3, 4, 0], [-4, 6, 9, 0.25]])

    def test_staleness_changes_time_without_transforming_xyz(self):
        values = self.run_loader(sensor_staleness_feats=True,
                                 use_dim=[0, 1, 2, 3, 4])
        np.testing.assert_allclose(
            values, [[2, 3, 4, 11, 0.25], [-4, 6, 9, 12, 0.5]])

    def test_emc_transforms_xyz_and_offsets_time(self):
        values = self.run_loader(early_motion_compensation=True)
        np.testing.assert_allclose(
            values, [[13, 18, 34, 0.25], [16, 24, 39, 0.5]])

    def test_both_flags_apply_temporal_offset_once(self):
        values = self.run_loader(early_motion_compensation=True,
                                 sensor_staleness_feats=True)
        np.testing.assert_allclose(
            values, [[13, 18, 34, 0.25], [16, 24, 39, 0.5]])

    def test_empty_sweeps_preserve_current_points_for_all_flag_combinations(self):
        for emc in (False, True):
            for staleness in (False, True):
                with self.subTest(emc=emc, staleness=staleness):
                    loader = self.loader_class(
                        early_motion_compensation=emc,
                        sensor_staleness_feats=staleness, use_dim=[4, 0, 3])
                    results = loader(self.make_results(with_sweep=False))
                    age = 0.25 if emc or staleness else 0.0
                    x = 13 if emc else 2
                    np.testing.assert_allclose(
                        results['points'].tensor, [[age, x, 11]])

    def test_padding_and_close_filter_do_not_multiply_temporal_offset(self):
        for remove_close in (False, True):
            with self.subTest(remove_close=remove_close):
                loader = self.loader_class(
                    sweeps_num=2, pad_empty_sweeps=True,
                    remove_close=remove_close, early_motion_compensation=True,
                    sensor_staleness_feats=True, use_dim=[0, 1, 2, 3, 4])
                results = self.make_results(with_sweep=False)
                results['points'] = ArrayPoints(
                    [[2, 3, 4, 11, 999], [0.25, 0.5, 1, 12, 888]])
                values = loader(results)['points'].tensor
                far = [13, 18, 34, 11, 0.25]
                close = [10.5, 19.75, 31, 12, 0.25]
                # The original points remain; close filtering affects copies.
                expected = [far, close] + (
                    [far, far] if remove_close else [far, close, far, close])
                np.testing.assert_allclose(values, expected)

    def test_existing_positional_test_mode_argument_is_preserved(self):
        loader = self.loader_class(
            1, 5, [0, 1, 2, 4], dict(backend='disk'), False, False, False, True)
        self.assertTrue(loader.test_mode)
        self.assertFalse(loader.sensor_staleness_feats)


if __name__ == '__main__':
    unittest.main()
