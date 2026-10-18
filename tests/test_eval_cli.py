"""CPU-only regression checks for evaluation CLI routing and output retention.

These tests deliberately do not import PyTorch, MMCV, or the model plugin;
they do not establish inference or metric correctness.
"""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import test as eval_cli


class EvaluationCliTest(unittest.TestCase):
    def parse(self, *extra):
        with mock.patch.dict('os.environ', {}, clear=True):
            return eval_cli.parse_args(['config.py', 'model.pth', *extra])

    def test_dynamic_flag_defaults_to_bbox(self):
        args = self.parse('--eval-dynamic')
        self.assertTrue(args.eval_dynamic)
        self.assertEqual(args.eval, ['bbox'])

    def test_invalid_combinations_are_rejected(self):
        for extra in ((), ('--eval-dynamic', '--format-only'),
                      ('--eval-dynamic', '--eval', 'segm'),
                      ('--out', 'predictions.json'),
                      ('--show-dir', 'vis', '--launcher', 'pytorch')):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse(*extra)
                self.assertEqual(error.exception.code, 2)

    def test_both_launcher_local_rank_spellings(self):
        for spelling in ('--local_rank', '--local-rank'):
            with self.subTest(spelling=spelling):
                self.assertEqual(self.parse('--eval', 'bbox', spelling, '2').local_rank, 2)

    def test_help_does_not_require_framework_imports(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                eval_cli.parse_args(['--help'])
        self.assertEqual(error.exception.code, 0)
        self.assertIn('--eval-dynamic', output.getvalue())

    def test_persistent_prefix_and_hook_options(self):
        args = self.parse('--eval', 'bbox')
        config = {'evaluation': {
            'interval': 2, 'by_epoch': True, 'save_best': 'NDS',
            'broadcast_bn_buffer': False, 'metric': ['old'],
            'result_names': ['pts_bbox'],
        }}
        kwargs = eval_cli.evaluation_kwargs(config, args, 'outputs/eval/test')
        self.assertEqual(kwargs, {
            'metric': ['bbox'], 'result_names': ['pts_bbox'],
            'jsonfile_prefix': 'outputs/eval/test/results',
        })
        self.assertEqual(config['evaluation']['interval'], 2)

    def test_explicit_eval_prefix_wins(self):
        args = self.parse('--eval-dynamic')
        args.eval_options = {'jsonfile_prefix': 'custom/results'}
        kwargs = eval_cli.evaluation_kwargs({}, args, 'outputs/eval/test')
        self.assertEqual(kwargs['jsonfile_prefix'], 'custom/results')

    def test_formatting_does_not_receive_evaluation_hook_options(self):
        args = self.parse('--format-only')
        kwargs = eval_cli.evaluation_kwargs(
            {'evaluation': {'metric': ['bbox'], 'interval': 1}}, args, 'submission')
        self.assertEqual(kwargs, {'jsonfile_prefix': 'submission/results'})

    def test_only_selected_evaluator_is_called(self):
        dataset = mock.Mock()
        predictions = [{'pts_bbox': 'synthetic prediction'}]
        for dynamic in (False, True):
            dataset.reset_mock()
            args = self.parse('--eval-dynamic') if dynamic else self.parse('--eval', 'bbox')
            kwargs = {'metric': ['bbox'], 'jsonfile_prefix': 'persistent/results'}
            result = eval_cli.evaluate_outputs(dataset, predictions, args, kwargs)
            selected = dataset.evaluate_dynamic if dynamic else dataset.evaluate
            other = dataset.evaluate if dynamic else dataset.evaluate_dynamic
            selected.assert_called_once_with(predictions, **kwargs)
            other.assert_not_called()
            self.assertIs(result, selected.return_value)


if __name__ == '__main__':
    unittest.main()
