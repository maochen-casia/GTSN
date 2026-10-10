"""Reject stale initialization paths before starting a full joint run."""
import copy
import unittest

from tsn.common.config import PROJECT_ROOT, read_json
from tsn.training.runner import validate_fresh_joint


class FreshJointTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(PROJECT_ROOT/'configs/attention_joint_fresh.json')

    def test_fresh_joint_accepts_only_official_trainable_initialization(self):
        validate_fresh_joint(self.config)
        self.assertFalse(self.config['model']['perception']['freeze_encoder'])
        self.assertEqual(self.config['train']['epochs'], 30)

    def test_experiment_checkpoint_or_module_warmup_cannot_be_loaded(self):
        for key in ('initialize_from', 'module_initialization', 'geometry_only'):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config['train'][key] = True if key == 'geometry_only' else '/old/weights.pt'
                with self.assertRaisesRegex(ValueError, 'cannot load'):
                    validate_fresh_joint(config)

    def test_frozen_encoder_or_incomplete_policy_is_rejected(self):
        for group, key, value in (('perception', 'freeze_encoder', True),
                                  ('learned_geometry', 'c1', False),
                                  ('learned_geometry', 'c2', False),
                                  ('contributions', 'c3', False)):
            with self.subTest(group=group, key=key):
                config = copy.deepcopy(self.config)
                config['model'][group][key] = value
                with self.assertRaises(ValueError):
                    validate_fresh_joint(config)
