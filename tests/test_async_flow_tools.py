"""Synthetic metadata and filesystem checks for the portable flow tools.

Geometry is stubbed here: these tests establish safe map routing, field handling,
and write behavior, not agreement with real nuScenes annotations or a GPU run.
"""

import contextlib
import copy
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.data_converter import async_flow


SAMPLE = 'a' * 32
PREVIOUS = 'b' * 32
OTHER = 'c' * 32
SYNC_LIDAR = '1' * 32
PREV_LIDAR = '2' * 32
SYNC_CAMERA = '3' * 32
PREV_CAMERA = '4' * 32
OTHER_CAMERA = '5' * 32
FRONT_CAMERA = '6' * 32


class TinyNuScenes:
    """Only metadata tables used by map validation; never opens sensor files."""

    def __init__(self):
        self.dataroot = '/portable/nuScenes'
        current = dict(token=SAMPLE, scene_token='scene-one', timestamp=1000000,
                       data={'LIDAR_TOP': SYNC_LIDAR,
                             'CAM_BACK_LEFT': SYNC_CAMERA,
                             'CAM_FRONT': FRONT_CAMERA}, prev=PREVIOUS, next='')
        previous = dict(token=PREVIOUS, scene_token='scene-one', timestamp=500000,
                        data={'LIDAR_TOP': PREV_LIDAR,
                              'CAM_BACK_LEFT': PREV_CAMERA}, prev='', next=SAMPLE)
        other = dict(token=OTHER, scene_token='scene-two', timestamp=400000,
                     data={'CAM_BACK_LEFT': OTHER_CAMERA}, prev='', next='')
        self.sample = [current]
        records = [
            (SYNC_LIDAR, SAMPLE, 'LIDAR_TOP', 'samples/LIDAR_TOP/shared.bin', 1000000),
            (PREV_LIDAR, PREVIOUS, 'LIDAR_TOP', 'samples/LIDAR_TOP/previous.bin', 500000),
            (SYNC_CAMERA, SAMPLE, 'CAM_BACK_LEFT',
             'samples/CAM_BACK_LEFT/shared.jpg', 1010000),
            (PREV_CAMERA, PREVIOUS, 'CAM_BACK_LEFT',
             'samples/CAM_BACK_LEFT/previous.jpg', 510000),
            (OTHER_CAMERA, OTHER, 'CAM_BACK_LEFT',
             'samples/CAM_BACK_LEFT/other.jpg', 410000),
            (FRONT_CAMERA, SAMPLE, 'CAM_FRONT', 'samples/CAM_FRONT/shared.jpg', 1005000),
        ]
        self.tables = {'sample': {s['token']: s for s in (current, previous, other)},
                       'sample_data': {}}
        for token, parent, channel, filename, timestamp in records:
            self.tables['sample_data'][token] = dict(
                token=token, sample_token=parent, channel=channel,
                filename=filename, timestamp=timestamp, is_key_frame=True,
                ego_pose_token='pose-' + token,
                calibrated_sensor_token='calibration-' + channel,
                prev='', next='')

    def get(self, table, token):
        return self.tables[table][token]

    def mapping(self, modality='camera', prefix='/historical/root'):
        sync = SYNC_CAMERA if modality == 'camera' else SYNC_LIDAR
        target = PREV_CAMERA if modality == 'camera' else PREV_LIDAR
        source = self.get('sample_data', sync)
        dest = self.get('sample_data', target)
        key = str(Path(prefix) / source['filename'])
        if modality == 'camera':
            row = dict(data_token=target, sample_token=PREVIOUS,
                       data_path=str(Path(prefix) / dest['filename']),
                       timestamp=dest['timestamp'])
        else:
            row = dict(sample_token=target,
                       lidar_path=str(Path(prefix) / dest['filename']),
                       timestamp=dest['timestamp'])
        return {key: [row]}


class FlowMapTests(unittest.TestCase):
    def setUp(self):
        self.nusc = TinyNuScenes()

    def test_old_root_prefix_resolves_exact_relative_path_for_both_modalities(self):
        for modality, target in (('camera', PREV_CAMERA), ('lidar', PREV_LIDAR)):
            with self.subTest(modality=modality):
                tasks = async_flow.collect_tasks(
                    self.nusc, self.nusc.mapping(modality), modality)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].sample_token, SAMPLE)
                self.assertEqual(tasks[0].sync_data_token, SYNC_LIDAR)
                self.assertEqual(tasks[0].async_data_token, target)
                self.assertEqual(tasks[0].sync_cam_data_token,
                                 SYNC_CAMERA if modality == 'camera' else None)

    def test_camera_selects_back_left_and_rejects_duplicate_outputs(self):
        mapping = self.nusc.mapping()
        rows = next(iter(mapping.values()))
        mapping['/historical/root/samples/CAM_FRONT/shared.jpg'] = []
        tasks = async_flow.collect_tasks(self.nusc, mapping, 'camera')
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].async_data_token, PREV_CAMERA)
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):
            async_flow.collect_tasks(self.nusc, mapping, 'camera')

    def test_same_basename_in_wrong_relative_directory_is_not_accepted(self):
        mapping = self.nusc.mapping()
        rows = next(iter(mapping.values()))
        wrong = {'/old/sweeps/CAM_BACK_LEFT/shared.jpg': rows}
        with self.assertRaises(ValueError):
            async_flow.collect_tasks(self.nusc, wrong, 'camera')

    def test_target_path_cannot_disagree_with_its_token(self):
        for modality, field in (('camera', 'data_path'), ('lidar', 'lidar_path')):
            with self.subTest(modality=modality):
                mapping = self.nusc.mapping(modality)
                next(iter(mapping.values()))[0][field] = '/old/samples/LIDAR_TOP/wrong.bin'
                with self.assertRaises(ValueError):
                    async_flow.collect_tasks(self.nusc, mapping, modality)

    def test_cross_scene_future_and_wrong_channel_targets_are_rejected(self):
        for change in ('scene', 'future', 'channel'):
            with self.subTest(change=change):
                nusc = TinyNuScenes()
                target = nusc.get('sample_data', PREV_CAMERA)
                if change == 'scene':
                    nusc.get('sample', PREVIOUS)['scene_token'] = 'scene-two'
                elif change == 'future':
                    target['timestamp'] = 2000000
                else:
                    target['channel'] = 'CAM_FRONT'
                with self.assertRaises(ValueError):
                    async_flow.collect_tasks(nusc, nusc.mapping(), 'camera')

    def test_map_timestamp_and_parent_sample_must_match_metadata(self):
        for field, value in (('timestamp', 999), ('sample_token', SAMPLE)):
            with self.subTest(field=field):
                mapping = self.nusc.mapping()
                next(iter(mapping.values()))[0][field] = value
                with self.assertRaises(ValueError):
                    async_flow.collect_tasks(self.nusc, mapping, 'camera')

    def test_sample_filter_and_limit_select_only_requested_work(self):
        tasks = async_flow.collect_tasks(self.nusc, self.nusc.mapping(), 'camera',
                                        sample_tokens=[SAMPLE], limit=1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].sample_token, SAMPLE)


class FlowFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.output = self.base / 'output'
        self.nusc = TinyNuScenes()
        self.task = async_flow.FlowTask(SAMPLE, SYNC_LIDAR, PREV_CAMERA, SYNC_CAMERA)
        self.points = np.array([[1., 2., 3.], [4., 5., 6.]])
        self.grid = np.array([[1., 2., 3.], [4., 5., 6.], [9., 9., 9.]])
        self.point_mock = mock.Mock(return_value=self.points.copy())
        self.flow_mock = mock.Mock(side_effect=self.fake_flow)
        self.geometry = SimpleNamespace(points_for_sample=self.point_mock,
                                        calculate_nusc_sceneflow=self.flow_mock,
                                        build_reference_grid=mock.Mock(return_value=self.grid))
        patcher = mock.patch.dict(sys.modules, {'projects.CMT.flow_geometry': self.geometry})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def fake_flow(source, target, points, nusc):
        # Position-dependent values distinguish reordered rows with identical shape.
        return points * 0.25, np.full_like(points, 0.125)

    def label(self, complete=False, modality='camera'):
        result = dict(gt_flow_sync2async=self.points * .25,
                      extra_metadata={'source': 'synthetic', 'revision': 3})
        result['pose_flow_s2t' if modality == 'camera' else 'pose_flow_sync2async'] = (
            np.full_like(self.points, .125))
        if modality == 'camera':
            result.update(sync_lidar_data_token=SYNC_LIDAR,
                          sync_cam_data_token=SYNC_CAMERA,
                          async_cam_data_token=PREV_CAMERA)
        else:
            result.update(sync_data_token=SYNC_LIDAR, async_data_token=PREV_LIDAR)
        if complete:
            result['pts_in_boxes'] = self.points.copy()
        return result

    def path(self, root=None, task=None):
        task = task or self.task
        return (root or self.source) / task.sample_token / (task.async_data_token + '.pkl')

    def save(self, label=None):
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.label() if label is None else label, protocol=2))
        return path, path.read_bytes()

    def run_task(self, **kwargs):
        values = dict(task=self.task, nusc=self.nusc, grid=self.grid,
                      mode='repair', output_dir=self.output, input_dir=self.source)
        values.update(kwargs)
        return async_flow.process_task(**values)

    def test_generated_camera_and_lidar_labels_have_coordinates_pose_and_tokens(self):
        for modality, target in (('camera', PREV_CAMERA), ('lidar', PREV_LIDAR)):
            with self.subTest(modality=modality):
                task = async_flow.FlowTask(SAMPLE, SYNC_LIDAR, target,
                                          SYNC_CAMERA if modality == 'camera' else None)
                self.run_task(task=task, mode='generate', input_dir=None)
                self.assertEqual(self.path(self.output, task).read_bytes()[:2], b'\x80\x04')
                label = async_flow.load_pickle(self.path(self.output, task))
                np.testing.assert_array_equal(label['pts_in_boxes'], self.points)
                np.testing.assert_array_equal(label['gt_flow_sync2async'], self.points * .25)
                pose_key = 'pose_flow_s2t' if modality == 'camera' else 'pose_flow_sync2async'
                np.testing.assert_array_equal(label[pose_key], np.full_like(self.points, .125))
                self.assertEqual(label['sync_lidar_data_token' if modality == 'camera'
                                       else 'sync_data_token'], SYNC_LIDAR)
                self.assertEqual(label['async_cam_data_token' if modality == 'camera'
                                       else 'async_data_token'], target)

    def test_numpy2_serialization_is_rejected_but_read_only_checks_work(self):
        source, before = self.save()
        with mock.patch.object(async_flow.np, '__version__', '2.0.0'):
            for mode in ('generate', 'repair'):
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(RuntimeError, 'NumPy 1.x'):
                        self.run_task(mode=mode)
                    self.run_task(mode=mode, dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(source.read_bytes(), before)

    def test_generate_existing_file_is_never_overwritten(self):
        path = self.path(self.output)
        path.parent.mkdir(parents=True)
        path.write_bytes(b'keep this file, even though it is not a pickle')
        before = path.read_bytes()
        with self.assertRaises((FileExistsError, ValueError)):
            self.run_task(mode='generate', input_dir=None)
        self.assertEqual(path.read_bytes(), before)

    def test_competing_writer_between_check_and_publish_is_not_clobbered(self):
        real_link = os.link

        def publish_competing_file(temporary, destination):
            Path(destination).write_bytes(b'written concurrently')
            real_link(temporary, destination)

        with mock.patch.object(async_flow.os, 'link', side_effect=publish_competing_file):
            with self.assertRaises(FileExistsError):
                self.run_task(mode='generate', input_dir=None)
        self.assertEqual(self.path(self.output).read_bytes(), b'written concurrently')
        self.assertEqual(list(self.output.rglob('.async-flow-*.tmp')), [])

    def test_validated_resume_keeps_destination_bytes(self):
        self.run_task(mode='generate', input_dir=None)
        path = self.path(self.output)
        before = path.read_bytes()
        result = self.run_task(mode='generate', input_dir=None, existing='validate')
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(path.read_bytes(), before)

    def test_repair_dry_run_does_not_create_outputs_or_change_input_bytes(self):
        path, before = self.save()
        self.run_task(dry_run=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_repair_separate_output_preserves_flows_tokens_and_extra_metadata(self):
        original = self.label()
        path, before = self.save(original)
        self.run_task()
        self.assertEqual(path.read_bytes(), before)
        label = async_flow.load_pickle(self.path(self.output))
        self.assertEqual(set(label), set(original) | {'pts_in_boxes'})
        for key, value in original.items():
            with self.subTest(field=key):
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(label[key], value)
                else:
                    self.assertEqual(label[key], value)
        np.testing.assert_array_equal(label['pts_in_boxes'], self.points)

    def test_complete_input_is_copied_byte_identically(self):
        path, before = self.save(self.label(complete=True))
        self.run_task()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.path(self.output).read_bytes(), before)

    def test_repair_resume_rejects_other_metadata_even_when_flow_matches(self):
        self.save()
        self.run_task()
        path = self.path(self.output)
        payload = async_flow.load_pickle(path)
        payload['extra_metadata']['revision'] = 99
        path.write_bytes(pickle.dumps(payload))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            self.run_task(existing='validate')
        self.assertEqual(path.read_bytes(), before)

    def test_repair_refuses_overlapping_input_output_roots(self):
        path, before = self.save()
        for output in (self.source, self.source / 'copies', self.base):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    self.run_task(output_dir=output)
                self.assertEqual(path.read_bytes(), before)

    def test_repair_refuses_source_symlink(self):
        target = self.base / 'real.pkl'
        target.write_bytes(pickle.dumps(self.label()))
        path = self.path()
        path.parent.mkdir(parents=True)
        path.symlink_to(target)
        before = target.read_bytes()
        with self.assertRaises(ValueError):
            self.run_task()
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_missing_source_does_not_fabricate_a_flow_label(self):
        with self.assertRaises(FileNotFoundError):
            self.run_task()
        self.assertFalse(self.output.exists())
        self.flow_mock.assert_not_called()

    def test_bad_shape_nonfinite_values_and_tokens_are_rejected_without_writes(self):
        for bad in ('shape', 'nan', 'sync_token', 'async_token', 'camera_token'):
            with self.subTest(bad=bad):
                label = self.label()
                if bad == 'shape':
                    label['gt_flow_sync2async'] = np.zeros((2, 2))
                elif bad == 'nan':
                    label['pose_flow_s2t'][0, 0] = np.nan
                else:
                    key = {'sync_token': 'sync_lidar_data_token', 'async_token': 'async_cam_data_token',
                           'camera_token': 'sync_cam_data_token'}[bad]
                    label[key] = 'wrong-token'
                path, before = self.save(label)
                with self.assertRaises(ValueError):
                    self.run_task()
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(self.output.exists())

    def test_same_shape_but_wrong_grid_row_correspondence_is_rejected(self):
        label = self.label()
        label['gt_flow_sync2async'] = label['gt_flow_sync2async'][::-1].copy()
        path, before = self.save(label)
        with self.assertRaises(ValueError):
            self.run_task()
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_complete_label_with_wrong_coordinates_is_rejected(self):
        label = self.label(complete=True)
        label['pts_in_boxes'] = label['pts_in_boxes'][::-1].copy()
        self.save(label)
        with self.assertRaises(ValueError):
            self.run_task()
        self.assertFalse(self.output.exists())

    def test_conflicting_pose_aliases_are_rejected(self):
        label = self.label()
        label['pose_flow_sync2async'] = label['pose_flow_s2t'] + 1
        self.save(label)
        with self.assertRaises(ValueError):
            self.run_task()
        self.assertFalse(self.output.exists())

    def cli_args(self):
        data = self.base / 'data'
        data.mkdir(exist_ok=True)
        mapping = self.base / 'map.pkl'
        mapping.write_bytes(pickle.dumps(self.nusc.mapping()))
        return ['--data-root', str(data), '--map-file', str(mapping),
                '--input-dir', str(self.source), '--sample-token', SAMPLE,
                '--limit', '1']

    def run_cli(self, args):
        from tools.data_converter import _flow_cli
        modules = {'nuscenes': SimpleNamespace(),
                   'nuscenes.nuscenes': SimpleNamespace(NuScenes=mock.Mock(return_value=self.nusc))}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ), \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = _flow_cli.main('repair', args)
        return code, json.loads(stdout.getvalue()), stderr.getvalue()

    def test_cli_default_dry_run_emits_report_without_changing_labels(self):
        path, before = self.save()
        report_path = self.base / 'dry-run.json'
        code, report, stderr = self.run_cli(self.cli_args() + ['--report', str(report_path)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report['counts'], {'would_repair': 1})
        self.assertEqual(report['selected'], 1)
        self.assertEqual(report['completed'], 1)
        self.assertEqual(json.loads(report_path.read_text()), report)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_cli_apply_only_writes_to_separate_output(self):
        path, before = self.save()
        code, report, stderr = self.run_cli(
            self.cli_args() + ['--apply', '--output-dir', str(self.output)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report['counts'], {'repaired': 1})
        self.assertEqual(path.read_bytes(), before)
        self.assertIn('pts_in_boxes', async_flow.load_pickle(self.path(self.output)))

    def test_cli_reports_validation_failure_with_nonzero_exit(self):
        label = self.label()
        label['gt_flow_sync2async'] += 1
        path, before = self.save(label)
        code, report, stderr = self.run_cli(
            self.cli_args() + ['--apply', '--output-dir', str(self.output)])
        self.assertEqual(code, 1)
        self.assertEqual(report['counts'], {'error': 1})
        self.assertFalse(report['ok'])
        self.assertEqual(len(report['errors']), 1)
        self.assertIn('geometry', stderr)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(self.output.exists())


class FlowCliTests(unittest.TestCase):
    def test_help_runs_from_unrelated_directory_without_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in ('generate_async_flow.py', 'repair_pts_in_boxes.py'):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, '-S', str(ROOT / 'tools/data_converter' / script), '--help'],
                        cwd=directory, capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--data-root', result.stdout)
                    self.assertIn('--map-file', result.stdout)

    def test_invalid_flags_fail_with_argparse_usage(self):
        from tools.data_converter import _flow_cli
        base = ['--data-root', 'nusc', '--map-file', 'map.pkl', '--input-dir', 'labels']
        for extra in (['--apply'], ['--workers', '0'], ['--resolution', '-1'],
                      ['--apply', '--dry-run'], ['--limit', '0'],
                      ['--point-cloud-range', '0', '0', '0', '0', '1', '1']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    _flow_cli.parse_args('repair', base + extra)
                self.assertEqual(error.exception.code, 2)
                self.assertIn('usage:', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
