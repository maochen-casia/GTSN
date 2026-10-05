"""Evaluation selects and records the requested route controller."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tsn.cli import evaluate


class EvaluationTests(unittest.TestCase):
    def arguments(self, output, *extra):
        return ['evaluate', '--checkpoint', 'model.pt', '--partition', 'validation',
                '--output-dir', str(output), '--device', 'cpu', *extra]

    def test_defaults_and_overrides_load_only_clearance(self):
        for extra, margin, penalty in (((), .04, .08),
                (('--clearance-margin', '.03', '--clearance-penalty', '.1'), .03, .1)):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'evaluation'
                policy = SimpleNamespace(execute=15, schedule='fixed15_clearance_deterministic',
                                         margin=margin, penalty=penalty)
                maps = object()
                checkpoint = dict(config={'train': {'seed': 31}, 'benchmark': {'root': 'data'},
                                          'eval': {'execute_horizon': 15}},
                                  splits={'validation': ['episode_001']})
                with patch('sys.argv', self.arguments(output, *extra)), \
                     patch.object(evaluate, 'output_path', return_value=output), \
                     patch.object(evaluate, 'require_device', return_value=torch.device('cpu')), \
                     patch.object(evaluate, 'load_clearance_policy', return_value=(policy, maps, checkpoint)) as load, \
                     patch.object(evaluate, 'episode_catalog', return_value={'episode_001': 'direct'}), \
                     patch.object(evaluate, 'validate_splits'), \
                     patch.object(evaluate, 'evaluate_rollouts') as rollouts, \
                     patch.object(evaluate, 'write_json') as write:
                    evaluate.main()
                load.assert_called_once_with(Path('model.pt'), torch.device('cpu'), margin=margin, penalty=penalty)
                self.assertIs(rollouts.call_args.args[4], policy)
                config = write.call_args_list[0].args[1]
                self.assertEqual(config['clearance'], dict(mode='deterministic', margin=margin, penalty=penalty))
                self.assertEqual(config['variant'], policy.schedule)
                self.assertTrue(config['full_partition'])

    def test_compact_controller_never_loads_clearance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'evaluation'
            policy = SimpleNamespace(execute=15, schedule='fixed15_compact')
            maps = object()
            checkpoint = dict(config={'train': {'seed': 31}, 'benchmark': {'root': 'data'},
                                      'eval': {'execute_horizon': 15}},
                              splits={'test': ['episode_001']})
            arguments = ['evaluate', '--checkpoint', 'model.pt', '--partition', 'test',
                         '--output-dir', str(output), '--device', 'cpu', '--controller', 'compact']
            with patch('sys.argv', arguments), \
                 patch.object(evaluate, 'output_path', return_value=output), \
                 patch.object(evaluate, 'require_device', return_value=torch.device('cpu')), \
                 patch.object(evaluate, 'load_compact_policy', return_value=(policy, maps, checkpoint)) as load, \
                 patch.object(evaluate, 'load_clearance_policy', side_effect=AssertionError('Clearance was loaded')) as clearance, \
                 patch.object(evaluate, 'episode_catalog', return_value={'episode_001': 'direct'}), \
                 patch.object(evaluate, 'validate_splits'), \
                 patch.object(evaluate, 'evaluate_rollouts') as rollouts, \
                 patch.object(evaluate, 'write_json') as write:
                evaluate.main()
            load.assert_called_once_with(Path('model.pt'), torch.device('cpu'))
            clearance.assert_not_called()
            self.assertIs(rollouts.call_args.args[4], policy)
            config = write.call_args_list[0].args[1]
            self.assertEqual(config['controller'], 'compact')
            self.assertEqual(config['clearance'], {'mode': 'disabled'})

    def test_removed_policy_switches_and_nonfinite_scores_fail_before_loading(self):
        extras = (('--clearance-mode', 'full'), ('--clearance-uncertainty', '.5'),
                  ('--body-clearance',), ('--clearance-trigger', '.5'),
                  ('--geometry-update', 'visibility'), ('--geometry-calibration', 'unused.json'),
                  ('--clearance-margin', 'nan'), ('--clearance-margin', '0'),
                  ('--clearance-penalty', 'inf'), ('--clearance-penalty', '-1'))
        for extra in extras:
            with self.subTest(extra=extra), patch('sys.argv', self.arguments(Path('unused'), *extra)), \
                 patch.object(evaluate, 'load_clearance_policy') as load, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()
