"""Main policy, fresh training gradients and standalone checkpoint round trips."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from tsn.common.checkpoint import save_checkpoint
from tsn.common.config import PROJECT_ROOT, read_json
from tsn.features.maps import GeometryMaps
from tsn.features.state import policy_state
from tsn.models.policy import NavigationPolicy, load_policy
from tsn.training.runner import training_loss


class TinyPerception(nn.Module):
    """Differentiable RGB/state fixture; geometry/control modules remain real."""
    def __init__(self, config=None, initialize_encoder=True):
        super().__init__()
        self.encoder = nn.Identity()
        self.features = nn.Linear(17, 768)
        self.joint = nn.Linear(768, 210)
        self.map_values = nn.Parameter(torch.tensor([0., .45, .15, 0., 0., 0.]))

    def forward(self, rgb, state, K, pose):
        appearance = rgb.float().mean((1, 2, 3), keepdim=False)[:, None]/255
        features = self.features(torch.cat((state, appearance), -1))
        tokens = features[:, None].expand(-1, 16, -1)
        values = torch.cat((2*self.map_values[:3].tanh(), self.map_values[3:].sigmoid()))
        dense = values[None, :, None, None].expand(len(state), 6, 80, 80)
        geometry = F.adaptive_avg_pool2d(dense, (4, 4)).flatten(2).transpose(1, 2)
        return .1*self.joint(features).reshape(-1, 30, 7).tanh(), tokens, geometry, dense


def fixture():
    config = read_json(PROJECT_ROOT/'configs/main.json')
    model = NavigationPolicy(config['model'], perception=TinyPerception())
    maps = GeometryMaps(config['model']['maps'])
    qpos = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78, .02, .02]])
    goal = torch.tensor([[.72, .25, .33, 1., 0., 0., 0.]])
    state = policy_state(qpos, goal, maps.settings)
    pose = torch.eye(4)[None]
    K = torch.tensor([[[200., 0., 16.], [0., 200., 12.], [0., 0., 1.]]])
    rgb = torch.full((1, 24, 32, 3), 128, dtype=torch.uint8)
    return config, model, maps, qpos, goal, state, pose, K, rgb


class MainModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7); torch.set_num_threads(1)

    def test_complete_control_is_finite_and_episode_reset_is_exact(self):
        _, model, _, _, _, state, pose, K, rgb = fixture()
        model.eval()
        with torch.inference_mode():
            first = model(rgb, state, K, pose)
            for step in (15, 30, 45, 60, 75):
                model.observe_step(step)
                self.assertTrue(torch.isfinite(model(rgb, state, K, pose)).all())
            self.assertEqual(first.shape, (1, 30, 7))
            self.assertEqual(len(model.history), 4)
            model.reset_episode()
            self.assertIsNone(model.memory.points)
            torch.testing.assert_close(first, model(rgb, state, K, pose), rtol=0, atol=0)

    def test_fresh_training_updates_every_learned_main_component(self):
        config, model, maps, qpos, goal, _, pose, K, rgb = fixture()
        tcp = model.kinematics(qpos[:, :7])
        batch = dict(qpos=qpos, goal_pose=goal, K=K, T_B_C=pose,
            target=torch.zeros(1, 30, 7), future_ee=tcp[:, None, :3, 3].expand(-1, 30, -1),
            valid_future=torch.ones(1, 30, dtype=torch.bool), depth=torch.ones(1, 24, 32),
            frame_index=torch.tensor([45]), history_rgb=rgb[:, None].expand(-1, 4, -1, -1, -1),
            history_qpos=qpos[:, None].expand(-1, 4, -1), history_goal_pose=goal[:, None].expand(-1, 4, -1),
            history_K=K[:, None].expand(-1, 4, -1, -1), history_T_B_C=pose[:, None].expand(-1, 4, -1, -1),
            history_ages=torch.tensor([[45., 30., 15., 0.]]), history_mask=torch.ones(1, 4, dtype=torch.bool))
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss, components = training_loss(model, maps, batch, config['train']['loss_weights'])
        self.assertEqual(set(components), set(config['train']['loss_weights']))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in (('maps', model.perception.map_values), ('joints', model.perception.joint.weight),
                               ('route', model.route.output[-1].weight), ('uncertainty', model.clearance.error_head[-1].weight),
                               ('trust', model.clearance.trust_head[-1].weight)):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        optimizer.step()
        self.assertIsNone(model.memory.points, 'Batched training must not contaminate streaming episode state')

    def test_checkpoint_is_standalone_and_contains_no_episode_memory(self):
        config, model, _, _, _, state, pose, K, rgb = fixture()
        model.eval()
        with torch.inference_mode():expected = model(rgb, state, K, pose)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            config = copy.deepcopy(config)
            config['model']['perception']['pretrained_weights'] = '/missing/pretrained/file'
            save_checkpoint(path, dict(format_version=1, architecture='gtsn_main', config=config,
                                       model=model.state_dict(), splits={}))
            with patch('tsn.models.policy.RGBPerception', TinyPerception):
                loaded, _, saved = load_policy(path)
            self.assertIsNone(loaded.memory.points)
            self.assertFalse(any('memory' in key for key in saved['model']))
            with torch.inference_mode():actual = loaded(rgb, state, K, pose)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
