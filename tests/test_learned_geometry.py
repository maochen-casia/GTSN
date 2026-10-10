"""Learned geometry parity, real gradients, causal state and moving robot probes."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tsn.common.checkpoint import save_checkpoint
from tsn.common.config import write_json
from tsn.models.c1_memory import PersistentGeometry
from tsn.models.c2_embodiment import EmbodimentGeometry, LearnedEmbodimentGeometry
from tsn.models.kinematics import PandaKinematics
from tsn.models.policy import NavigationPolicy, load_policy
from tsn.training.runner import initialize_geometry_update, train, training_loss
from tsn.training.geometry_update import train_adapter
from test_main_model import TinyPerception, fixture


class LearnedGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_memory_initialization_and_zero_strength_preserve_actual_anchors(self):
        original = PersistentGeometry(capacity=6)
        learned = PersistentGeometry(capacity=6, learned=True)
        for step in range(6):
            points = torch.rand(80, 3)*torch.tensor([.7, .6, .4])+torch.tensor([.2, -.3, .1])
            radius = torch.rand(80)*.1
            for memory in (original, learned):
                memory.update(points, torch.ones(80, dtype=torch.bool), radius, step*15,
                              torch.zeros(3), torch.ones(3), torch.zeros(16))
            torch.testing.assert_close(original.points, learned.points, rtol=0, atol=0)
            for expected, actual in zip(original.query(), learned.query()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(learned.query(True)[2], torch.ones_like(learned.query(True)[2]))
        self.assertFalse(any(name in learned.state_dict() for name in ('points', 'radius', 'last', 'context')))
        learned.reset()
        self.assertIsNone(learned.context)

    def test_learned_scores_choose_voxel_representatives_and_capacity_evictions(self):
        memory = PersistentGeometry(capacity=1, learned=True)
        points = torch.tensor([[.401, .201, .201], [.402, .202, .202], [.7, .3, .3]])
        with patch.object(memory.selector, 'forward', side_effect=lambda p, *args: (p[:, 0]-.5)*10):
            memory.update(points, torch.ones(3, dtype=torch.bool), torch.full((3,), .05), 0,
                          torch.zeros(3), torch.ones(3), torch.zeros(16))
        torch.testing.assert_close(memory.points, points[2:])
        memory = PersistentGeometry(capacity=8, learned=True)
        with patch.object(memory.selector, 'forward', side_effect=lambda p, *args: p[:, 0]*10):
            memory.update(points[:2], torch.ones(2, dtype=torch.bool), torch.full((2,), .05), 0,
                          torch.zeros(3), torch.ones(3), torch.zeros(16))
        torch.testing.assert_close(memory.points, points[1:2])

    def test_robot_probes_follow_opening_mount_and_arm_motion(self):
        body = LearnedEmbodimentGeometry()
        closed, ids = body.robot_points(torch.zeros(2))
        opened, _ = body.robot_points(torch.full((2,), .02))
        self.assertFalse(torch.equal(closed[ids == 2], opened[ids == 2]))
        mount = torch.eye(4); mount[:3, 3] = torch.tensor([.1, .2, .3])
        moved, _ = body.robot_points(torch.zeros(2), mount)
        self.assertFalse(torch.equal(closed[ids == 5], moved[ids == 5]))
        kinematics = PandaKinematics()
        q = torch.tensor([0., .4, 0., -1.96, 0., 2.35, .78])
        self.assertEqual(kinematics.arm_points(q).shape, (7, 3))
        changed = q.clone(); changed[0] += .5
        self.assertFalse(torch.equal(kinematics.arm_points(q), kinematics.arm_points(changed)))

    def test_neural_body_initialization_is_exact_and_empty_cloud_is_finite(self):
        baseline, learned = EmbodimentGeometry(), LearnedEmbodimentGeometry()
        candidates = torch.tensor([.5, .2, .3]).expand(14, 30, 3).clone()
        candidates[:, :, 0] += torch.linspace(-.05, .05, 14)[:, None]
        rotations, tcp = torch.eye(3).expand(30, 3, 3), torch.eye(4)
        points = torch.tensor([[.55, .2, .25], [.45, .25, .2]])
        args = (candidates, rotations, tcp, points, torch.zeros(2), torch.full((2,), .02))
        context = dict(state=torch.zeros(16), goal=torch.ones(3), arm_points=torch.zeros(7, 3))
        torch.testing.assert_close(baseline.contact_risk(*args), learned.contact_risk(*args, context=context), rtol=0, atol=0)
        for p, u in ((torch.empty(0, 3), torch.empty(0)),
                     (torch.full((1, 3), torch.nan), torch.full((1,), torch.nan))):
            result = learned.contact_risk(candidates, rotations, tcp, p, u, torch.zeros(2), context=context)
            torch.testing.assert_close(result, torch.zeros(14))

    def test_each_adapter_gets_nonzero_training_gradients(self):
        config, _, maps, qpos, goal, state, pose, K, rgb = fixture()
        config['model']['learned_geometry'] = dict(c1=True, c2=True)
        config['train']['loss_weights']['memory'] = .05
        model = NavigationPolicy(config['model'], perception=TinyPerception())
        tcp = model.kinematics(qpos[:, :7])
        batch = dict(qpos=qpos, goal_pose=goal, K=K, T_B_C=pose,
            target=torch.zeros(1, 30, 7), future_ee=tcp[:, None, :3, 3].expand(-1, 30, -1),
            valid_future=torch.ones(1, 30, dtype=torch.bool), depth=torch.ones(1, 24, 32),
            frame_index=torch.tensor([45]), history_rgb=rgb[:, None].expand(-1, 4, -1, -1, -1),
            history_qpos=qpos[:, None].expand(-1, 4, -1), history_goal_pose=goal[:, None].expand(-1, 4, -1),
            history_K=K[:, None].expand(-1, 4, -1, -1), history_T_B_C=pose[:, None].expand(-1, 4, -1, -1),
            history_ages=torch.tensor([[45., 30., 15., 0.]]), history_mask=torch.ones(1, 4, dtype=torch.bool))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_loss(model, maps, batch, config['train']['loss_weights'])
            loss.backward()
            for component in (model.memory.selector.head[-1], model.embodiment.risk_head[-1]):
                self.assertGreater(float(component.weight.grad.abs().sum()), 0)
            if step:
                for component in (model.memory.selector.summarize, model.embodiment.scene_attention):
                    self.assertGreater(float(component.in_proj_weight.grad.abs().sum()), 0)
            optimizer.step()
        self.assertIsNone(model.memory.points)

    def test_legacy_parent_loading_freezes_every_existing_tensor(self):
        config, parent, _, _, _, state, pose, K, rgb = fixture()
        updated = copy.deepcopy(config)
        updated['model']['learned_geometry'] = dict(c1=True, c2=True)
        updated['model']['perception']['freeze_encoder'] = True
        updated['train']['geometry_only'] = True
        model = NavigationPolicy(updated['model'], perception=TinyPerception())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parent.pt'
            save_checkpoint(path, dict(format_version=1, architecture='gtsn_main', config=config, model=parent.state_dict(), splits={}))
            initialize_geometry_update(model, path, updated, {})
            for key, expected in parent.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
            self.assertFalse(any(p.requires_grad for p in model.perception.parameters()))
            self.assertFalse(any(p.requires_grad for p in model.route.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.memory.selector.parameters()))
            parent.eval(); model.eval()
            with torch.inference_mode():
                for step in (0, 15, 30, 45, 60):
                    parent.observe_step(step); model.observe_step(step)
                    torch.testing.assert_close(parent(rgb, state, K, pose), model(rgb, state, K, pose), rtol=0, atol=0)
            with self.assertRaises(ValueError):initialize_geometry_update(model, path, updated, {'train': ['wrong']})

    def test_geometry_only_training_cannot_select_by_unchanged_route_rmse(self):
        with self.assertRaisesRegex(ValueError, 'unchanged route RMSE'):
            train({'train': {'geometry_only': True}}, Path('/unused'))

    def test_adapter_trainer_rejects_held_out_cache_and_different_parent(self):
        config, model, _, _, _, _, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'parent.pt'
            save_checkpoint(checkpoint, dict(format_version=1, architecture='gtsn_main', config=config,
                model=model.state_dict(), splits={'train': ['episode_train']}))
            cache = root/'cache'; cache.mkdir()
            for partition in ('validation', 'train'):
                other = root/'other.pt'; save_checkpoint(other, dict(format_version=1))
                manifest = dict(partition=partition, train_episodes=['episode_train'],
                    parent_checkpoint=str(checkpoint if partition == 'validation' else other))
                write_json(cache/'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'training partition'):
                    train_adapter(checkpoint, cache, root/partition, 'c1', device='cpu')

    def test_learned_policy_checkpoint_round_trip_and_episode_reset(self):
        config, _, _, _, _, state, pose, K, rgb = fixture()
        config['model']['learned_geometry'] = dict(c1=True, c2=True)
        model = NavigationPolicy(config['model'], perception=TinyPerception()).eval()
        with torch.inference_mode():expected = model(rgb, state, K, pose)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'updated.pt'
            save_checkpoint(path, dict(format_version=1, architecture='gtsn_main', config=config, model=model.state_dict(), splits={}))
            with patch('tsn.models.policy.RGBPerception', TinyPerception):loaded, _, _ = load_policy(path)
            with torch.inference_mode():
                torch.testing.assert_close(expected, loaded(rgb, state, K, pose), rtol=0, atol=0)
                loaded.observe_step(15); loaded(rgb, state, K, pose)
                loaded.reset_episode()
                self.assertIsNone(loaded.memory.context)
                torch.testing.assert_close(expected, loaded(rgb, state, K, pose), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
