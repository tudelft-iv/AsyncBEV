"""Regression checks for analyzing partial or resumed MMCV training logs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/analysis_tools/analyze_logs.py'
SPEC = importlib.util.spec_from_file_location('analyze_logs', SCRIPT)
logs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logs)


class AnalyzeLogsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'run.log.json'
        records = [
            {'env_info': 'metadata'},
            {'epoch': 7, 'mode': 'train', 'time': 100, 'loss': 5},
            {'epoch': 7, 'mode': 'train', 'time': 2, 'loss': 4},
            {'epoch': 7, 'mode': 'val', 'time': 999, 'mAP': .3},
            {'epoch': 8, 'mode': 'train', 'time': 100, 'loss': 3},
            {'epoch': 8, 'mode': 'train', 'time': 4, 'loss': 2},
            {'epoch': 8, 'mode': 'train', 'time': 6},
            {'epoch': 8, 'mode': 'val', 'mAP': .4},
            {'epoch': 9, 'mode': 'train', 'time': 100, 'loss': 1},
        ]
        self.path.write_text('\n'.join(json.dumps(row) for row in records) + '\n\n')
        self.epochs = logs.load_json_logs([str(self.path)])[0]

    def test_metrics_stay_aligned_when_keys_and_modes_differ(self):
        self.assertEqual(
            logs.metric_points(self.epochs, 'loss', 'train', 1),
            ([1, 2, 3, 4, 6], [5, 4, 3, 2, 1]))
        self.assertEqual(
            logs.metric_points(self.epochs, 'mAP', 'eval', 1),
            ([7, 8], [.3, .4]))
        self.assertEqual(
            logs.metric_points(self.epochs, 'mAP', 'eval', 2),
            ([8], [.4]))

    def test_ragged_resumed_epochs_use_training_only_weighted_time(self):
        output = io.StringIO()
        args = SimpleNamespace(json_logs=[str(self.path)], include_outliers=False)
        with contextlib.redirect_stdout(output):
            logs.cal_train_time([self.epochs], args)
        text = output.getvalue()
        self.assertIn('slowest epoch 8, average time is 5.0000', text)
        self.assertIn('fastest epoch 7, average time is 2.0000', text)
        self.assertIn('average iter time: 4.0000 s/iter (3 logged timing samples)', text)

    def test_insufficient_timing_is_an_explicit_error(self):
        args = SimpleNamespace(json_logs=[str(self.path)], include_outliers=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'try --include-outliers'):
                logs.cal_train_time([{9: self.epochs[9]}], args)

    def test_malformed_json_identifies_line(self):
        self.path.write_text('{}\n{invalid json}\n')
        with self.assertRaisesRegex(ValueError, ':2: invalid JSON'):
            logs.load_json_logs([str(self.path)])


if __name__ == '__main__':
    unittest.main()
