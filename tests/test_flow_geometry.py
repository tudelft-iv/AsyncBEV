"""Historical grid/transform regression checks without the model or SDK stack.

The small geometry fixtures below provide orthonormal transforms and box masks.
They exercise real helper bodies but do not substitute for nuScenes integration
testing.
"""

import ast
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'projects/CMT/flow_geometry.py'


class QuaternionFixture:
    def __init__(self, values=None, matrix=None):
        if matrix is not None:
            self.rotation_matrix = np.array(matrix, dtype=float)
        else:
            w, x, y, z = values
            self.rotation_matrix = np.array([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


class BoxFixture:
    def __init__(self, center, size, orientation, token=None):
        self.center = np.array(center, dtype=float)
        self.wlh = np.array(size, dtype=float)
        self.orientation = orientation
        self.token = token


def transform_fixture(translation, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def points_in_box_fixture(box, points):
    local = box.orientation.rotation_matrix.T @ (points - box.center[:, None])
    return (np.abs(local) <= box.wlh[[1, 0, 2], None] / 2).all(axis=0)


def load_geometry():
    # Isolate module functions to avoid nuScenes/pyquaternion installation in
    # this lightweight suite; their interfaces are supplied by the fixtures.
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(np=np, Quaternion=QuaternionFixture, Box=BoxFixture,
                     transform_matrix=transform_fixture,
                     points_in_box=points_in_box_fixture)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'),
         namespace)
    return namespace


class NuScenesFixture:
    def __init__(self):
        identity = [1, 0, 0, 0]
        self.records = {
            ('sample_data', 'sync'): dict(token='sync', ego_pose_token='source',
                                         calibrated_sensor_token='lidar'),
            ('sample_data', 'async'): dict(token='async', ego_pose_token='target',
                                          calibrated_sensor_token='camera'),
            ('ego_pose', 'source'): dict(translation=[0, 0, 0], rotation=identity),
            ('ego_pose', 'target'): dict(translation=[3, 0, 0], rotation=identity),
            ('calibrated_sensor', 'lidar'): dict(translation=[1, 0, 0],
                                                rotation=identity, camera_intrinsic=[]),
            ('calibrated_sensor', 'camera'): dict(translation=[100, 200, 300],
                                                 rotation=[0, 0, 0, 1],
                                                 camera_intrinsic=[[1, 0, 0]]),
            ('sample_annotation', 'box_source'): dict(instance_token='same'),
            ('sample_annotation', 'box_target'): dict(instance_token='same'),
        }
        self.boxes = {
            'sync': [BoxFixture([2, 0, 0], [2, 2, 2], QuaternionFixture(identity),
                                token='box_source')],
            'async': [BoxFixture([4, 0, 0], [2, 2, 2], QuaternionFixture(identity),
                                 token='box_target')],
        }

    def get(self, table, token):
        return self.records[(table, token)]

    def get_boxes(self, token):
        return self.boxes[token]


class FlowGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = load_geometry()

    def test_dependency_boundary_and_old_import_api(self):
        tree = ast.parse(SOURCE.read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {'numpy', 'pyquaternion',
                                  'nuscenes.utils.geometry_utils',
                                  'nuscenes.utils.data_classes'})
        for filename in ('projects/__init__.py', 'projects/CMT/__init__.py'):
            package = ast.parse((ROOT / filename).read_text())
            self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom))
                                 for node in ast.walk(package)))
        old_path = ROOT / 'projects/CMT/mmdet3d_plugin/datasets/pipelines/flow_utils.py'
        reexport = next(node for node in ast.parse(old_path.read_text()).body
                        if isinstance(node, ast.ImportFrom) and
                        node.module == 'projects.CMT.flow_geometry')
        self.assertEqual({alias.name for alias in reexport.names},
                         {'boxes2sensor', 'calculate_nusc_sceneflow'})

    def test_grid_historical_endpoints_truncation_and_order(self):
        # H=int(3/.8)=3, W=int(4/.8)=5, Z=int(2/.8)=2. These
        # endpoints intentionally differ from a voxel-center arange lattice.
        grid = self.geometry['build_reference_grid'](
            bev_size=(3, 4), resolution=.8,
            point_cloud_range=(-4, -6, -1, 8, 6, 1))
        expected = np.array([[x, y, z]
                             for z in (-.5, .5)
                             for y in (-4., 0., 4.)
                             for x in (-2.5, -.25, 2., 4.25, 6.5)])
        np.testing.assert_array_equal(grid, expected)
        self.assertEqual(grid.dtype, np.float64)

    def test_default_grid_parity_with_historical_numpy_recipe(self):
        h, w, z, resolution = 200, 200, 8, .5
        H, W, Z = int(h / resolution), int(w / resolution), int(z / resolution)
        xs = np.broadcast_to(np.linspace(.5, w-.5, W).reshape(1, 1, W), (Z,H,W)) / w
        ys = np.broadcast_to(np.linspace(.5, h-.5, H).reshape(1, H, 1), (Z,H,W)) / h
        zs = np.broadcast_to(np.linspace(.5, z-.5, Z).reshape(Z, 1, 1), (Z,H,W)) / z
        original = np.stack((xs, ys, zs), axis=-1)
        expected = original.copy()
        expected[..., 0:1] = original[..., 0:1] * 108 - 54
        expected[..., 1:2] = original[..., 1:2] * 108 - 54
        expected[..., 2:3] = original[..., 2:3] * 8 - 5
        np.testing.assert_array_equal(self.geometry['build_reference_grid'](),
                                      expected.reshape(H * W * Z, -1))

    def test_grid_rejects_invalid_dimensions(self):
        invalid = [dict(bev_size=(0, 2)), dict(bev_size=(1,)),
                   dict(bev_size=(float('inf'), 2)), dict(resolution=0),
                   dict(resolution=float('nan')), dict(resolution=999),
                   dict(point_cloud_range=(0, 0, 0, 1, 1, 0)),
                   dict(point_cloud_range=(0, 0, 0, 1, 1, float('inf'))),
                   dict(point_cloud_range=(-1e308, 0, 0, 1e308, 1, 1))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.geometry['build_reference_grid'](**kwargs)

    def test_camera_target_uses_source_lidar_extrinsics(self):
        nusc = NuScenesFixture()
        points = np.array([[1, 0, 0], [20, 0, 0]], dtype=float)
        flow, pose = self.geometry['calculate_nusc_sceneflow'](
            nusc.get('sample_data', 'sync'), nusc.get('sample_data', 'async'),
            points, nusc)
        np.testing.assert_allclose(flow, [[2, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(pose, [[-3, 0, 0], [-3, 0, 0]])

    def test_lidar_target_uses_its_own_extrinsics(self):
        nusc = NuScenesFixture()
        target_calibration = nusc.get('calibrated_sensor', 'camera')
        target_calibration.update(translation=[2, 0, 0], rotation=[1, 0, 0, 0],
                                  camera_intrinsic=[])
        flow, pose = self.geometry['calculate_nusc_sceneflow'](
            nusc.get('sample_data', 'sync'), nusc.get('sample_data', 'async'),
            np.array([[1, 0, 0]], dtype=float), nusc)
        np.testing.assert_allclose(flow, [[2, 0, 0]])
        np.testing.assert_allclose(pose, [[-4, 0, 0]])

    def test_disappearing_object_has_zero_scene_flow(self):
        nusc = NuScenesFixture()
        nusc.boxes['async'] = []
        flow, pose = self.geometry['calculate_nusc_sceneflow'](
            nusc.get('sample_data', 'sync'), nusc.get('sample_data', 'async'),
            np.array([[1, 0, 0]], dtype=float), nusc)
        np.testing.assert_array_equal(flow, [[0, 0, 0]])
        np.testing.assert_allclose(pose, [[-3, 0, 0]])

    def test_box_frame_rotation_and_translation(self):
        half = np.sqrt(.5)
        box = BoxFixture([10, 3, 0], [2, 4, 6], QuaternionFixture([1, 0, 0, 0]))
        converted = self.geometry['boxes2sensor'](
            [box], dict(translation=[10, 0, 0], rotation=[half, 0, 0, half]),
            dict(translation=[1, 0, 0], rotation=[1, 0, 0, 0]))
        np.testing.assert_allclose(converted[0].center, [2, 0, 0], atol=1e-14)
        np.testing.assert_allclose(converted[0].orientation.rotation_matrix,
                                   [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-14)
        np.testing.assert_array_equal(converted[0].wlh, box.wlh)
        np.testing.assert_array_equal(box.center, [10, 3, 0])

    def test_point_selection_preserves_box_order_and_overlap(self):
        nusc = NuScenesFixture()
        nusc.get('calibrated_sensor', 'lidar')['translation'] = [0, 0, 0]
        identity = QuaternionFixture([1, 0, 0, 0])
        nusc.boxes['sync'] = [BoxFixture([2, 0, 0], [4, 4, 4], identity),
                               BoxFixture([0, 0, 0], [2, 2, 2], identity)]
        grid = np.array([[3, 0, 0], [0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=float)
        points = self.geometry['points_for_sample'](nusc, 'sync', grid)
        np.testing.assert_array_equal(points, grid[[0, 1, 2, 1, 2]])

    def test_no_boxes_dummy_differs_from_no_selected_points(self):
        nusc = NuScenesFixture()
        grid = np.array([[99, 99, 99]], dtype=float)
        selected = self.geometry['points_for_sample'](nusc, 'sync', grid)
        self.assertEqual(selected.shape, (0, 3))
        self.assertEqual(selected.dtype, np.float64)
        nusc.boxes['sync'] = []
        dummy = self.geometry['points_for_sample'](nusc, 'sync', grid)
        np.testing.assert_array_equal(dummy, np.zeros((1, 3), dtype=np.float32))
        self.assertEqual(dummy.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
