"""Behavior and trainability of the attention-based geometry replacements."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from tsn.common.checkpoint import save_checkpoint
from tsn.models.c1_memory import PersistentGeometry
from tsn.models.c2_embodiment import AttentionEmbodimentGeometry
from tsn.models.policy import NavigationPolicy, load_policy
from tsn.training.runner import initialize_geometry_update, training_loss
from test_main_model import TinyPerception, fixture


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_memory_uses_only_network_ranking_and_returns_hard_bank(self):
        memory = PersistentGeometry(capacity=1, replacement=True, hidden_dim=16)
        # A far point wins even though the old TCP-goal proximity rule favors
        # the first point. The tiny score difference cannot override old rules.
        points = torch.tensor([[.4, 0., .2], [.8, .5, .5]])
        with patch.object(memory.selector, 'forward', side_effect=lambda p, *a: p[:, 1]*.0001):
            memory.update(points, torch.ones(2, dtype=torch.bool), torch.full((2,), .03),
                          0, torch.tensor([.2, 0., .2]), torch.tensor([.6, 0., .2]), torch.zeros(16))
        torch.testing.assert_close(memory.query()[0], points[1:])
        self.assertEqual(len(memory.query()[0]), 1)
        torch.testing.assert_close(memory.query(True)[2], torch.ones(1))
        self.assertEqual(len(memory.selector.latent_blocks), 2)

    def test_learned_update_replaces_a_matched_anchor(self):
        memory = PersistentGeometry(capacity=2, replacement=True, hidden_dim=16)
        first, better = torch.tensor([[.401, .1, .2]]), torch.tensor([[.402, .1, .2]])
        with patch.object(memory.selector, 'forward', side_effect=lambda p, *a: p[:, 0]):
            for step, points in enumerate((first, better)):
                memory.update(points, torch.ones(1, dtype=torch.bool), torch.full((1,), .03),
                              step, torch.zeros(3), torch.ones(3), torch.zeros(16))
        torch.testing.assert_close(memory.points, better)

    def test_recent_query_uses_neural_hard_ranking_and_unit_weights(self):
        memory = PersistentGeometry(replacement=True, hidden_dim=16, query_source='recent', query_capacity=1)
        points = torch.tensor([[.4, 0., .2], [.7, .3, .4]])
        with patch.object(memory.selector, 'forward', side_effect=lambda p, *a: p[:, 1]):
            memory.update(points, torch.ones(2, dtype=torch.bool), torch.full((2,), .03),
                          0, torch.zeros(3), torch.ones(3), torch.zeros(16))
            p, _, weights = memory.query(True)
            torch.testing.assert_close(p, points[1:]); torch.testing.assert_close(weights, torch.ones(1))

    def test_calibrated_readout_uses_only_neural_responses_and_receives_gradients(self):
        body = AttentionEmbodimentGeometry(hidden_dim=16, depth=2, calibrated=True)
        candidates = torch.tensor([.5, .2, .3]).expand(14, 30, 3)
        rotations, tcp = torch.eye(3).expand(30, 3, 3), torch.eye(4)
        context = dict(state=torch.zeros(16), goal=torch.ones(3), arm_points=torch.zeros(7, 3))
        args = (candidates, rotations, tcp, torch.tensor([[.55, .2, .25]]), torch.zeros(1), torch.zeros(2))
        with patch.object(body, 'neural_response', side_effect=lambda d: torch.full_like(d, .37)):
            torch.testing.assert_close(body.contact_risk(*args, context=context), torch.full((14,), .37))
        risk = body.contact_risk(*args, context=context)
        (risk.sum()+body.calibration_loss()).backward()
        for parameter in (body.priority_head[-1].weight, body.risk_head[-1].weight,
                          body.blocks[0].cross_attention.in_proj_weight, body.blocks[-1].ffn[0].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(body.calibration_loss()))

    def test_calibrated_distance_features_follow_configured_margin(self):
        body = AttentionEmbodimentGeometry(hidden_dim=16, calibrated=True)
        candidates = torch.tensor([.5, .2, .3]).expand(14, 30, 3)
        rotations, tcp = torch.eye(3).expand(30, 3, 3), torch.eye(4)
        context = dict(state=torch.zeros(16), goal=torch.ones(3), arm_points=torch.zeros(7, 3))
        args = (candidates, rotations, tcp, torch.tensor([[.54, .2, .3]]), torch.zeros(1), torch.zeros(2), None, context)
        narrow = body.descriptors(*args, margin=.04)['body'][..., :6, 6]
        wide = body.descriptors(*args, margin=.08)['body'][..., :6, 6]
        torch.testing.assert_close(wide, narrow/2)

    def test_memory_attention_blocks_learn_and_are_permutation_equivariant(self):
        memory = PersistentGeometry(replacement=True, hidden_dim=16, depth=3)
        p = torch.rand(20, 3)
        u, support, scatter, age = torch.rand(20)*.1, torch.ones(20), torch.zeros(20), torch.arange(20).float()
        tcp, goal, state = torch.zeros(3), torch.ones(3), torch.randn(16)
        score = memory.selector(p, u, support, scatter, age, tcp, goal, state)
        order = torch.randperm(20)
        permuted = memory.selector(p[order], u[order], support[order], scatter[order], age[order], tcp, goal, state)
        torch.testing.assert_close(permuted, score[order], rtol=1e-4, atol=1e-5)
        score.square().mean().backward()
        for block, attention, ffn in zip(memory.selector.latent_blocks, memory.selector.point_attention,
                                         memory.selector.point_ffns):
            for parameter in (block.ffn[0].weight, block.cross_attention.in_proj_weight,
                              attention.in_proj_weight, ffn[1].weight):
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_body_returns_direct_neural_risk_and_empty_zero(self):
        body = AttentionEmbodimentGeometry(hidden_dim=16, depth=3)
        candidates = torch.tensor([.5, .2, .3]).expand(14, 30, 3)
        rotations, tcp = torch.eye(3).expand(30, 3, 3), torch.eye(4)
        points, padding = torch.tensor([[.55, .2, .25]]), torch.zeros(1)
        context = dict(state=torch.zeros(16), goal=torch.ones(3), arm_points=torch.zeros(7, 3))
        args = (candidates, rotations, tcp, points, padding, torch.zeros(2))
        with patch.object(body, 'predict_descriptors', return_value=torch.full((14,), .37)):
            torch.testing.assert_close(body.contact_risk(*args, context=context), torch.full((14,), .37))
        risk = body.contact_risk(*args, context=context)
        risk.sum().backward()
        for block in body.blocks:
            self.assertGreater(float(block.ffn[0].weight.grad.abs().sum()), 0)
            self.assertGreater(float(block.cross_attention.in_proj_weight.grad.abs().sum()), 0)
        torch.testing.assert_close(body.contact_risk(candidates, rotations, tcp, points[:0], padding[:0],
                                                     torch.zeros(2), context=context), torch.zeros(14))

    def test_only_encoder_is_frozen_and_old_heads_receive_gradients(self):
        config, parent, maps, qpos, goal, state, pose, K, rgb = fixture()
        updated = copy.deepcopy(config)
        updated['model']['learned_geometry'] = dict(mode='replacement', c1=True, c2=True, width=16, depth=2)
        updated['model']['perception']['freeze_encoder'] = True
        updated['train']['loss_weights'].update(memory=.05, embodiment=.2)
        perception = TinyPerception(); perception.encoder = nn.Linear(2, 2)
        parent.perception.encoder = nn.Linear(2, 2)
        model = NavigationPolicy(updated['model'], perception=perception)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parent.pt'
            save_checkpoint(path, dict(format_version=1, architecture='gtsn_main', config=config,
                                       model=parent.state_dict(), splits={}))
            initialize_geometry_update(model, path, updated, {})
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, not name.startswith('perception.encoder.'))
        tcp = model.kinematics(qpos[:, :7])
        batch = dict(qpos=qpos, goal_pose=goal, K=K, T_B_C=pose, target=torch.zeros(1, 30, 7),
            future_ee=tcp[:, None, :3, 3].expand(-1, 30, -1), valid_future=torch.ones(1, 30, dtype=torch.bool),
            depth=torch.ones(1, 24, 32), frame_index=torch.tensor([45]),
            history_rgb=rgb[:, None].expand(-1, 4, -1, -1, -1), history_qpos=qpos[:, None].expand(-1, 4, -1),
            history_goal_pose=goal[:, None].expand(-1, 4, -1), history_K=K[:, None].expand(-1, 4, -1, -1),
            history_T_B_C=pose[:, None].expand(-1, 4, -1, -1), history_ages=torch.tensor([[45., 30., 15., 0.]]),
            history_mask=torch.ones(1, 4, dtype=torch.bool))
        loss, components = training_loss(model, maps, batch, updated['train']['loss_weights'])
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        for parameter in (model.perception.features.weight, model.perception.joint.weight,
                          model.route.output[-1].weight, model.clearance.error_head[-1].weight,
                          model.clearance.trust_head[-1].weight, model.memory.selector.head[-1].weight,
                          model.embodiment.risk_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        model.eval(); model.reset_episode()
        with torch.inference_mode():expected = model(rgb, state, K, pose)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'new.pt'
            save_checkpoint(path, dict(format_version=1, architecture='gtsn_main', config=updated,
                                       model=model.state_dict(), splits={}))
            def factory(*args, **kwargs):
                fresh = TinyPerception()
                fresh.encoder = nn.Linear(2, 2)
                return fresh
            with patch('tsn.models.policy.RGBPerception', factory):loaded, _, _ = load_policy(path)
            with torch.inference_mode():torch.testing.assert_close(loaded(rgb, state, K, pose), expected)

    def test_replacement_rejects_residual_strength(self):
        config, *_ = fixture()
        config['model']['learned_geometry'] = dict(mode='replacement', c1=True, c1_strength=.1)
        with self.assertRaisesRegex(ValueError, 'do not accept residual'):
            NavigationPolicy(config['model'], perception=TinyPerception())


if __name__ == '__main__':unittest.main()
