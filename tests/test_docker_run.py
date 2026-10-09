"""The host launcher only makes the requested new run writable."""
import contextlib
import io
from pathlib import Path
import runpy
import shlex
import sys
import unittest
from unittest.mock import patch


class LauncherTests(unittest.TestCase):
    def test_transferred_storage_is_read_only_at_both_paths(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'scripts/docker_run.py'
        storage = Path('/home/datasets_v2/chenmao')
        arguments = [str(script), '--print-command', '--storage-root', str(storage), '--cpu', 'test']
        output = io.StringIO()
        with patch.object(sys, 'argv', arguments), patch('pathlib.Path.is_dir', return_value=True), \
             contextlib.redirect_stdout(output):
            runpy.run_path(str(script), run_name='__main__')
        command = shlex.split(output.getvalue())
        self.assertIn(f'type=bind,source={storage},target={storage},readonly', command)
        self.assertIn(f'type=bind,source={storage},target=/run/user/1016,readonly', command)
        self.assertNotIn('type=bind,source=/run/user/1016,target=/run/user/1016,readonly', command)

    def test_distributed_training_uses_localhost_in_offline_container(self):
        root = Path(__file__).resolve().parents[1]
        script = root/'scripts/docker_run.py'
        arguments = [str(script), '--print-command', '--gpu', '0,1,2,3', '--processes', '4', 'train',
                     '--config', str(root/'configs/sim2real.json'), '--output-dir', 'runs/distributed-dry-run']
        output = io.StringIO()
        with patch.object(sys, 'argv', arguments), patch('os.getcwd', return_value=str(root)), \
             contextlib.redirect_stdout(output):
            runpy.run_path(str(script), run_name='__main__')
        command = shlex.split(output.getvalue())
        self.assertIn('torchrun', command)
        self.assertIn('--master-addr=127.0.0.1', command)
        self.assertIn('--nproc-per-node=4', command)
        self.assertEqual(command[command.index('--network')+1], 'none')

    def test_snapshot_controls_imports_and_test_directory(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'scripts/docker_run.py'
        snapshot = root / 'runs' / 'immutable-source'
        arguments = [str(script), '--print-command', '--cpu', '--source-snapshot', str(snapshot), 'test']
        output = io.StringIO()
        with patch.object(sys, 'argv', arguments), patch('pathlib.Path.is_dir', return_value=True), \
             contextlib.redirect_stdout(output):
            runpy.run_path(str(script), run_name='__main__')
        command = shlex.split(output.getvalue())
        self.assertIn(f'PYTHONPATH={snapshot}/src:/workspace/vendor/Pi3', command)
        self.assertEqual(command[command.index('-s')+1], str(snapshot/'tests'))

    def test_relative_output_is_forwarded_as_the_absolute_writable_mount(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'scripts/docker_run.py'
        arguments = [str(script), '--print-command', '--cpu', 'evaluate',
                     '--checkpoint', '/run/user/1016/example.pt', '--partition', 'validation',
                     '--output-dir', 'runs/launcher-dry-run']
        output = io.StringIO()
        with patch.object(sys, 'argv', arguments), patch('os.getcwd', return_value=str(root)), \
             contextlib.redirect_stdout(output):
            runpy.run_path(str(script), run_name='__main__')
        command = shlex.split(output.getvalue())
        path = str(root / 'runs' / 'launcher-dry-run')
        self.assertIn(f'type=bind,source={path},target={path}', command)
        self.assertIn(f'type=bind,source={root},target=/workspace,readonly', command)
        self.assertEqual(command[command.index('--output-dir') + 1], path)
        self.assertNotIn('--gpus', command)
        self.assertFalse(Path(path).exists())
