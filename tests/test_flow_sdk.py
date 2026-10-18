"""Opt-in real nuScenes SDK smoke tests using only temporary synthetic tables.

Run with ASYNCBEV_RUN_SDK_TESTS=1 in an environment with nuscenes-devkit and
pyquaternion installed. No sensor images/point clouds or existing dataset are
read. This verifies SDK/CLI integration, not a historical CMT environment or
real-label reproduction. The default lightweight suite skips these tests.
"""

import base64
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENABLED = os.environ.get('ASYNCBEV_RUN_SDK_TESTS') == '1'


def write_fixture(root):
    """Write two linked samples with moving ego/box and intermediate sweeps."""
    tables = {name: [] for name in (
        'category', 'attribute', 'visibility', 'instance', 'sensor',
        'calibrated_sensor', 'ego_pose', 'log', 'scene', 'sample',
        'sample_data', 'sample_annotation', 'map')}
    tables['category'] = [dict(token='category', name='vehicle.car', description='fixture')]
    tables['instance'] = [dict(token='instance', category_token='category',
                               nbr_annotations=2, first_annotation_token='ann0',
                               last_annotation_token='ann1')]
    tables['log'] = [dict(token='log', logfile='fixture', vehicle='fixture',
                         date_captured='2000-01-01', location='fixture')]
    tables['scene'] = [dict(token='scene', name='scene-fixture', description='fixture',
                           log_token='log', nbr_samples=2,
                           first_sample_token='sample0', last_sample_token='sample1')]
    tables['map'] = [dict(token='map', filename='maps/fixture.png', log_tokens=['log'])]
    maps = root / 'maps'
    maps.mkdir(parents=True)
    # The SDK checks that a map image exists during construction; no map query
    # or rendering is performed. This image is unrelated to sensor observations.
    (maps / 'fixture.png').write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4q8AAAAASUVORK5CYII='))
    for index, translation in enumerate((0.0, 1.0)):
        sample = 'sample{}'.format(index)
        tables['sample'].append(dict(
            token=sample, timestamp=(index + 1) * 1000000,
            prev='sample0' if index else '', next='' if index else 'sample1',
            scene_token='scene'))
        tables['sample_annotation'].append(dict(
            token='ann{}'.format(index), sample_token=sample,
            instance_token='instance', visibility_token='', attribute_tokens=[],
            translation=[translation - 0.5, 0, 0], size=[10, 10, 10],
            rotation=[1, 0, 0, 0], prev='ann0' if index else '',
            next='' if index else 'ann1', num_lidar_pts=1, num_radar_pts=0))
    for name, x in (('pose0', 0.0), ('posehalf', 0.5), ('pose1', 1.0)):
        tables['ego_pose'].append(dict(token=name, translation=[x, 0, 0],
                                       rotation=[1, 0, 0, 0]))
    mapping = {}
    for modality, channel, short in (('lidar', 'LIDAR_TOP', 'lidar'),
                                      ('camera', 'CAM_BACK_LEFT', 'camera')):
        tables['sensor'].append(dict(token='sensor_' + short, channel=channel,
                                     modality=modality))
        tables['calibrated_sensor'].append(dict(
            token='calibration_' + short, sensor_token='sensor_' + short,
            translation=[20, 0, 0] if modality == 'camera' else [0, 0, 0],
            rotation=[1, 0, 0, 0],
            camera_intrinsic=[[1, 0, 0], [0, 1, 0], [0, 0, 1]] if modality == 'camera' else []))
        rows = []
        for index, tag in enumerate(('0', 'half', '1')):
            keyframe = tag != 'half'
            timestamp = (1000000, 1500000, 2000000)[index]
            if modality == 'camera' and keyframe:
                timestamp += 10000
            extension = 'jpg' if modality == 'camera' else 'pcd.bin'
            filename = '{}/{}/{}.{}'.format('samples' if keyframe else 'sweeps',
                                            channel, short + tag, extension)
            row = dict(token=short + tag, sample_token='sample0' if tag == '0' else 'sample1',
                       ego_pose_token='pose' + tag, calibrated_sensor_token='calibration_' + short,
                       timestamp=timestamp, fileformat='jpg' if modality == 'camera' else 'pcd',
                       is_key_frame=keyframe, filename=filename, width=1, height=1,
                       prev=short + ('0' if tag == 'half' else 'half') if tag != '0' else '',
                       next=short + ('half' if tag == '0' else '1') if tag != '1' else '')
            tables['sample_data'].append(row)
            rows.append(row)
        frames = []
        for row in (rows[1], rows[0]):
            frame = dict(timestamp=row['timestamp'])
            if modality == 'camera':
                frame.update(data_token=row['token'], sample_token=row['sample_token'],
                             data_path='/legacy/root/' + row['filename'])
            else:
                frame.update(sample_token=row['token'], lidar_path='/legacy/root/' + row['filename'])
            frames.append(frame)
        mapping[modality] = {'/legacy/root/' + rows[2]['filename']: frames}
    version = root / 'v1.0-mini'
    version.mkdir()
    for name, records in tables.items():
        (version / (name + '.json')).write_text(json.dumps(records))
    for modality, entries in mapping.items():
        (root / (modality + '.pkl')).write_bytes(pickle.dumps(entries))


@unittest.skipUnless(ENABLED, 'set ASYNCBEV_RUN_SDK_TESTS=1 for real-SDK integration')
class RealNuScenesFlowTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from nuscenes.nuscenes import NuScenes
        self.np = np
        self.temporary = tempfile.TemporaryDirectory(prefix='asyncbev-sdk-test-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.data = self.work / 'metadata'
        write_fixture(self.data)
        self.external_cwd = self.work / 'unrelated-cwd'
        self.external_cwd.mkdir()
        self.nusc = NuScenes(version='v1.0-mini', dataroot=str(self.data), verbose=False)
        self.assertFalse((self.data / 'samples').exists())
        self.assertFalse((self.data / 'sweeps').exists())

    def run_cli(self, mode, modality='camera', output=None, workers=1,
                extra=(), expected_code=0):
        script = 'generate_async_flow.py' if mode == 'generate' else 'repair_pts_in_boxes.py'
        command = [sys.executable, str(ROOT / 'tools/data_converter' / script),
                   '--data-root', str(self.data), '--map-file', str(self.data / (modality + '.pkl')),
                   '--version', 'v1.0-mini', '--modality', modality, '--workers', str(workers),
                   '--bev-size', '2', '2', '--resolution', '1',
                   '--point-cloud-range', '-1', '-1', '-1', '1', '1', '1']
        if output is not None:
            command.extend(['--output-dir', str(output)])
        command.extend(str(value) for value in extra)
        result = subprocess.run(command, cwd=self.external_cwd, env=os.environ.copy(),
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, expected_code, result.stdout + '\n' + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['ok'], expected_code == 0)
        return report

    def payload(self, root, name):
        return pickle.loads((root / 'sample1' / (name + '.pkl')).read_bytes())

    def test_real_sdk_generation_and_multiprocessing_agree_without_sensor_files(self):
        from projects.CMT.flow_geometry import build_reference_grid, points_for_sample
        grid = build_reference_grid((2, 2), 1, (-1, -1, -1, 1, 1, 1))
        expected = self.np.array([[x, y, z] for z in (-0.5, 0.5)
                                  for y in (-0.5, 0.5) for x in (-0.5, 0.5)])
        self.np.testing.assert_array_equal(grid, expected)
        self.np.testing.assert_array_equal(points_for_sample(self.nusc, 'lidar1', grid), expected)
        for modality in ('camera', 'lidar'):
            first = self.work / (modality + '-one')
            parallel = self.work / (modality + '-two')
            for output, workers in ((first, 1), (parallel, 2)):
                report = self.run_cli('generate', modality, output, workers)
                self.assertEqual(report['counts'], {'generated': 2})
            for suffix, shift in (('0', 1.0), ('half', 0.5)):
                name = modality + suffix
                left, right = self.payload(first, name), self.payload(parallel, name)
                self.assertEqual(set(left), set(right))
                self.assertEqual(len(left), 6 if modality == 'camera' else 5)
                for key in left:
                    if isinstance(left[key], self.np.ndarray):
                        self.np.testing.assert_array_equal(left[key], right[key])
                    else:
                        self.assertEqual(left[key], right[key])
                self.np.testing.assert_array_equal(left['pts_in_boxes'], expected)
                scene = self.np.tile([-shift, 0, 0], (len(expected), 1))
                pose = self.np.tile([shift, 0, 0], (len(expected), 1))
                self.np.testing.assert_allclose(left['gt_flow_sync2async'], scene, atol=1e-12)
                pose_key = 'pose_flow_s2t' if modality == 'camera' else 'pose_flow_sync2async'
                self.np.testing.assert_allclose(left[pose_key], pose, atol=1e-12)

    def test_real_sdk_repair_is_read_only_by_default_and_rejects_wrong_flow(self):
        generated = self.work / 'generated'
        self.run_cli('generate', output=generated)
        source = self.work / 'legacy-copy'
        shutil.copytree(generated, source)
        missing = source / 'sample1/camera0.pkl'
        original = pickle.loads(missing.read_bytes())
        original.pop('pts_in_boxes')
        original['extra_metadata'] = {'source': 'synthetic fixture', 'revision': 1}
        missing.write_bytes(pickle.dumps(original))
        before = {path.relative_to(source): path.read_bytes() for path in source.rglob('*.pkl')}
        checked = self.run_cli('repair', extra=['--input-dir', source])
        self.assertEqual(checked['counts'], {'would_repair': 1, 'already_complete': 1})
        repaired = self.work / 'repaired'
        applied = self.run_cli('repair', output=repaired,
                               extra=['--input-dir', source, '--apply'])
        self.assertEqual(applied['counts'], {'repaired': 1, 'already_complete': 1})
        for relative, contents in before.items():
            self.assertEqual((source / relative).read_bytes(), contents)
        self.assertEqual((repaired / 'sample1/camerahalf.pkl').read_bytes(),
                         before[Path('sample1/camerahalf.pkl')])
        fixed = self.payload(repaired, 'camera0')
        self.assertEqual(set(fixed), set(original) | {'pts_in_boxes'})
        for key, value in original.items():
            if isinstance(value, self.np.ndarray):
                self.np.testing.assert_array_equal(fixed[key], value)
            else:
                self.assertEqual(fixed[key], value)
        broken = self.work / 'wrong-flow-copy'
        shutil.copytree(source, broken)
        invalid = dict(original)
        invalid['gt_flow_sync2async'] = original['gt_flow_sync2async'] + 3
        (broken / 'sample1/camera0.pkl').write_bytes(pickle.dumps(invalid))
        rejected_output = self.work / 'rejected'
        rejected = self.run_cli('repair', output=rejected_output,
                                extra=['--input-dir', broken], expected_code=1)
        self.assertEqual(rejected['counts']['error'], 1)
        self.assertIn('Scene flow does not match', rejected['errors'][0]['error'])
        self.assertFalse(rejected_output.exists())


if __name__ == '__main__':
    unittest.main()
