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
