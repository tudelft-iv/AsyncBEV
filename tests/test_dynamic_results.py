"""Check result aggregation without importing CUDA-dependent plugin modules."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'projects/CMT/mmdet3d_plugin/datasets/'
          'custom_nuscenes_dataset.py')


def load_evaluate_method():
    tree = ast.parse(SOURCE.read_text())
    dataset = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef)
                   and node.name == 'CustomNuScenesDataset')
    method = next(node for node in dataset.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'evaluate_dynamic')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE),
                 'exec'), namespace)
    return namespace['evaluate_dynamic']


class FakeDataset:
    evaluate_dynamic = load_evaluate_method()

    def format_results(self, results, jsonfile_prefix):
        return {name: name + '.json' for name in results}, None

    def _evaluate_dynamic_single(self, result_path, logger=None,
                                 metric='bbox', result_name='pts_bbox'):
        return {mode: {result_name + '_NuScenes/NDS': 0.5}
                for mode in ('all', 'dynamic', 'static')}


class DynamicResultsTests(unittest.TestCase):
    def test_default_head_preserves_historical_structure(self):
        result = FakeDataset().evaluate_dynamic(['pts_bbox'])
        self.assertEqual(result, {
            mode: {'pts_bbox_NuScenes/NDS': 0.5}
            for mode in ('all', 'dynamic', 'static')})

    def test_multiple_heads_are_kept_in_each_subset(self):
        result = FakeDataset().evaluate_dynamic(
            ['pts_bbox', 'img_bbox'], result_names=['pts_bbox', 'img_bbox'])
        for subset in ('all', 'dynamic', 'static'):
            self.assertEqual(result[subset], {
                'pts_bbox_NuScenes/NDS': 0.5,
                'img_bbox_NuScenes/NDS': 0.5})


if __name__ == '__main__':
    unittest.main()
