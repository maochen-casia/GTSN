"""Physical surface membership, adaptive selection and full joint gradients."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tsn.common.config import PROJECT_ROOT, read_json
from tsn.common.checkpoint import save_checkpoint
from tsn.models.c3_clearance import uncertainty_features
from tsn.models.robot_surface import RobotSurfaceCloud
from tsn.models.surface_embodiment import SurfaceEmbodimentGeometry
from tsn.models.policy import NavigationPolicy, load_policy, metric_points
from tsn.models.c1_memory import PersistentGeometry
from tsn.training.runner import encode_observations, training_loss, validate_fresh_joint
from test_main_model import TinyPerception, fixture


class SurfaceNodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13); torch.set_num_threads(1)

    def inputs(self):
        _, model, _, qpos, goal, state, _, _, _ = fixture()
        tcp = model.kinematics(qpos[:, :7])[0]
        candidates = tcp[:3, 3].expand(14, 30, 3).clone()
        candidates[:, :, 0] += torch.linspace(0, .04, 30)
        rotations = tcp[:3, :3].expand(30, 3, 3)
        context = dict(state=state[0], goal=goal[0, :3], qpos=qpos[0],
                       joint_seeds=qpos[0, :7].expand(30, 7))
        points = tcp[:3, 3]+torch.tensor([[.09, .01, .03], [.12, -.05, .02], [.15, .04, -.04]])
        return candidates, rotations, tcp, points, torch.zeros(len(points)), qpos[0, 7:], context

    def test_surface_pool_covers_arm_hand_fingers_and_moves_with_fk(self):
        surface = RobotSurfaceCloud(pool_size=512)
        _, policy, _, qpos, *_ = fixture()
        frames = surface.frames(qpos)
        torch.testing.assert_close(frames[:, surface.tcp_index], policy.kinematics(qpos[:, :7]))
        points, normals = surface(qpos)
        self.assertEqual(points.shape, (1, 512, 3))
        self.assertEqual(len(surface.surface_link_names), 11)
        self.assertEqual(len(torch.unique(surface.labels)), 11)
        torch.testing.assert_close(normals.norm(dim=-1), torch.ones(1, 512), atol=1e-5, rtol=1e-5)
        # Local samples lie on some original collision primitive of their link.
        for label, name in enumerate(surface.surface_link_names):
            selected = surface.local_points[surface.labels == label]
            link = surface.names.index(name)
            fields = [hull(selected).abs() for hull, index in zip(surface.hulls, surface.primitive_links) if index == link]
            self.assertLess(float(torch.stack(fields).amin(0).amax()), .001)
        changed = qpos.clone(); changed[:, 1] += .3; changed[:, 7] += .01
        moved, _ = surface(changed)
        self.assertGreater(float((points-moved).norm(dim=-1).amax()), .01)
        again = RobotSurfaceCloud(pool_size=512)
        torch.testing.assert_close(surface.local_points, again.local_points, atol=0, rtol=0)

    def test_64_nodes_are_unique_actual_surface_samples_selected_only_by_network(self):
        body = SurfaceEmbodimentGeometry(hidden_dim=16, depth=2, pool_size=512)
        candidates, rotation, tcp, points, padding, fingers, context = self.inputs()
        def descending(embedding):
            score = torch.arange(512, device=embedding.device, dtype=embedding.dtype)
            return score.expand(*embedding.shape[:-1])[..., None]
        with patch.object(body.selection_head, 'forward', side_effect=descending):
            values = body.descriptors(candidates, rotation, tcp, points, padding, fingers, None, context)
        expected = torch.arange(511, 447, -1).expand(14, 5, 64)
        torch.testing.assert_close(values['indices'], expected)
        cloud, _ = body.surface(values['qpos'])
        torch.testing.assert_close(values['positions'], cloud.gather(-2, expected[..., None].expand(-1, -1, -1, 3)))
        self.assertEqual(values['body'].shape, (14, 5, 64, 14))

    def test_surface_selection_and_every_attention_block_receive_gradients(self):
        body = SurfaceEmbodimentGeometry(hidden_dim=16, depth=2, pool_size=512)
        candidates, rotation, tcp, points, padding, fingers, context = self.inputs()
        risk = body.contact_risk(candidates, rotation, tcp, points, padding, fingers, context=context)
        aux = body.pop_training_aux()
        self.assertTrue(torch.isfinite(aux['teacher']).all())
        loss = (risk-aux['teacher']).square().mean()+.1*aux['selection_loss']+body.calibration_loss()
        loss.backward()
        for parameter in (body.selection_head[-1].weight, body.surface_embedding[0].weight,
                          body.link_embedding.weight, body.risk_head[-1].weight, body.priority_head[-1].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for block in body.blocks:
            self.assertGreater(float(block.cross_attention.in_proj_weight.grad.abs().sum()), 0)
            self.assertGreater(float(block.ffn[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(body._training_aux)

    def test_scene_grid_really_contains_1024_points_and_matching_error_features(self):
        dense = torch.arange(6*80*80).float().reshape(1, 6, 80, 80)/10000
        points = metric_points(dense, (32, 32))
        self.assertEqual(points.shape, (1, 1024, 3))
        self.assertEqual(len(torch.unique(points[0], dim=0)), 1024)
        features = uncertainty_features(torch.randn(1, 16, 64), points, torch.eye(4)[None], torch.eye(4)[None], (32, 32))
        self.assertEqual(features.shape, (1, 1024, 74))
        self.assertTrue(torch.isfinite(features).all())

    def test_learned_scene_query_can_deliver_1024_actual_points(self):
        memory = PersistentGeometry(capacity=4096, replacement=True, hidden_dim=16,
                                    query_source='recent', query_capacity=1024)
        points = torch.rand(2048, 3)*torch.tensor([.6, .8, .4])+torch.tensor([.3, -.4, .15])
        with patch.object(memory.selector, 'forward', side_effect=lambda p, *args: p[:, 0]):
            memory.update(points, torch.ones(2048, dtype=torch.bool), torch.full((2048,), .03),
                          0, torch.zeros(3), torch.ones(3), torch.zeros(16))
            queried, _, weights = memory.query(True)
        expected = points[torch.argsort(points[:, 0], descending=True, stable=True)[:1024]]
        torch.testing.assert_close(queried, expected)
        torch.testing.assert_close(weights, torch.ones(1024))

    def test_arm_contact_labels_and_self_filter_use_full_arm_surface(self):
        body = SurfaceEmbodimentGeometry(hidden_dim=16, depth=2, pool_size=512)
        candidates, rotation, tcp, _, _, fingers, context = self.inputs()
        candidates = tcp[:3, 3].expand(14, 30, 3)
        cloud, _ = body.surface(context['qpos'])
        point = cloud[body.surface.labels == 2].mean(0)[None]
        self.assertTrue(body.self_mask(point, tcp, fingers, context=context).item())
        body.contact_risk(candidates, rotation, tcp, point, torch.zeros(1), fingers, context=context)
        aux = body.pop_training_aux()
        self.assertGreater(float(aux['teacher'].amin()), .99)
        empty = body.contact_risk(candidates, rotation, tcp, point[:0], torch.zeros(0), fingers, context=context)
        torch.testing.assert_close(empty, torch.zeros(14))

    def test_fresh_surface_contract_rejects_missing_node_loss_or_scene_bottleneck(self):
        config = read_json(PROJECT_ROOT/'configs/attention_surface_fresh.json')
        validate_fresh_joint(config)
        missing = copy.deepcopy(config); missing['train']['loss_weights']['nodes'] = 0
        with self.assertRaisesRegex(ValueError, 'positive selection'):
            validate_fresh_joint(missing)
        bottleneck = copy.deepcopy(config); bottleneck['model']['learned_geometry']['scene_attention_capacity'] = 128
        with self.assertRaisesRegex(ValueError, 'complete queried scene'):
            validate_fresh_joint(bottleneck)

    def test_new_model_joint_loss_and_standalone_round_trip(self):
        config, _, maps, qpos, goal, state, pose, K, rgb = fixture()
        surface = read_json(PROJECT_ROOT/'configs/attention_surface_fresh.json')
        config = copy.deepcopy(config)
        config['model']['learned_geometry'] = {**surface['model']['learned_geometry'], 'width': 16, 'depth': 2,
                                              'surface_pool_size': 512}
        config['model']['perception']['point_grid_hw'] = [32, 32]
        config['model']['perception']['freeze_encoder'] = False
        config['train']['loss_weights'] = surface['train']['loss_weights']
        model = NavigationPolicy(config['model'], perception=TinyPerception())
        tcp = model.kinematics(qpos[:, :7])
        batch = dict(qpos=qpos, goal_pose=goal, K=K, T_B_C=pose, target=torch.zeros(1, 30, 7),
            future_ee=tcp[:, None, :3, 3].expand(-1, 30, -1), valid_future=torch.ones(1, 30, dtype=torch.bool),
            depth=torch.ones(1, 24, 32), frame_index=torch.tensor([45]),
            history_rgb=rgb[:, None].expand(-1, 4, -1, -1, -1), history_qpos=qpos[:, None].expand(-1, 4, -1),
            history_goal_pose=goal[:, None].expand(-1, 4, -1), history_K=K[:, None].expand(-1, 4, -1, -1),
            history_T_B_C=pose[:, None].expand(-1, 4, -1, -1), history_ages=torch.tensor([[45., 30., 15., 0.]]),
            history_mask=torch.ones(1, 4, dtype=torch.bool))
        features = encode_observations(model, maps, batch)
        self.assertEqual(features['points'].shape, (1, 4, 1024, 3))
        loss, components = training_loss(model, maps, batch, config['train']['loss_weights'])
        self.assertEqual(set(components), set(config['train']['loss_weights']))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertGreater(float(model.embodiment.selection_head[-1].weight.grad.abs().sum()), 0)
        model.eval(); model.reset_episode()
        with torch.inference_mode(): expected = model(rgb, state, K, pose)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'new.pt'
            save_checkpoint(checkpoint, dict(format_version=1, architecture='gtsn_main', config=config,
                                            model=model.state_dict(), splits={}))
            with patch('tsn.models.policy.RGBPerception', TinyPerception): loaded, _, _ = load_policy(checkpoint)
            with torch.inference_mode(): actual = loaded(rgb, state, K, pose)
        torch.testing.assert_close(expected, actual)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
