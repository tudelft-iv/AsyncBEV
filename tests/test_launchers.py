"""Exercise shell argument/env forwarding without launching PyTorch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='asyncbev launcher ')
        self.addCleanup(self.tmp.cleanup)
        self.python = Path(self.tmp.name) / 'fake python'
        self.python.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, sys\n'
            'print(json.dumps({"args": sys.argv[1:], '
            '"visible": os.environ.get("CUDA_VISIBLE_DEVICES"), '
            '"pythonpath": os.environ.get("PYTHONPATH")}))\n'
            'sys.exit(int(os.environ.get("FAKE_EXIT", "0")))\n')
        self.python.chmod(0o755)
        self.env = dict(os.environ, PYTHON=str(self.python),
                        CUDA_VISIBLE_DEVICES='2', PYTHONPATH='existing path',
                        NNODES='1', NODE_RANK='0', PORT='29777',
                        MASTER_ADDR='127.0.0.1', FAKE_EXIT='0')

    def run_launcher(self, name, *args):
        return subprocess.run(
            ['bash', str(ROOT / 'tools' / name), *args],
            cwd=self.tmp.name, env=self.env, capture_output=True, text=True)

    def test_train_quotes_and_preserves_resources(self):
        result = self.run_launcher('dist_train.sh', 'config with space.py',
                                   '1', '--work-dir', 'output with space',
                                   '--no-resume')
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload['visible'], '2')
        self.assertIn('--master_port=29777', payload['args'])
        self.assertEqual(payload['args'][-6:], [
            'config with space.py', '--launcher', 'pytorch', '--work-dir',
            'output with space', '--no-resume'])
        self.assertTrue(payload['pythonpath'].endswith(':existing path'))

    def test_dynamic_eval_and_multinode_forwarding(self):
        self.env.update(NNODES='2', NODE_RANK='1', MASTER_ADDR='192.0.2.1')
        result = self.run_launcher('dist_test.sh', 'config.py',
                                   'checkpoint with space.pth', '2',
                                   '--eval-dynamic', '--eval-options',
                                   'jsonfile_prefix=output with space')
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn('--nnodes=2', payload['args'])
        self.assertIn('--node_rank=1', payload['args'])
        self.assertIn('--master_addr=192.0.2.1', payload['args'])
        self.assertEqual(payload['args'][-7:], [
            'config.py', 'checkpoint with space.pth', '--launcher',
            'pytorch', '--eval-dynamic', '--eval-options',
            'jsonfile_prefix=output with space'])

    def test_no_implicit_gpu_selection(self):
        self.env.pop('CUDA_VISIBLE_DEVICES')
        result = self.run_launcher('dist_train.sh', 'config.py', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(json.loads(result.stdout)['visible'])

    def test_child_exit_status(self):
        self.env['FAKE_EXIT'] = '7'
        result = self.run_launcher('dist_test.sh', 'c.py', 'c.pth', '1',
                                   '--eval', 'bbox')
        self.assertEqual(result.returncode, 7)

    def test_invalid_gpu_count_and_rank(self):
        for count in ('0', '-1', 'abc'):
            with self.subTest(count=count):
                self.assertEqual(self.run_launcher(
                    'dist_train.sh', 'config.py', count).returncode, 2)
        self.env['NODE_RANK'] = '1'
        self.assertEqual(self.run_launcher(
            'dist_test.sh', 'c.py', 'c.pth', '1').returncode, 2)

    def test_missing_arguments(self):
        for name in ('dist_train.sh', 'dist_test.sh'):
            with self.subTest(name=name):
                self.assertEqual(self.run_launcher(name).returncode, 2)


if __name__ == '__main__':
    unittest.main()
