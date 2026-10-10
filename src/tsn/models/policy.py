"""Main RGB navigation model: compose C1 memory, C2 embodiment and C3 clearance."""
from collections import deque
import math

import torch
from torch import nn
from torch.nn import functional as F

from tsn.common.checkpoint import load_checkpoint
from tsn.features.maps import GeometryMaps
from tsn.models.c1_memory import PersistentGeometry, workspace_mask
from tsn.models.c2_embodiment import EmbodimentGeometry, LearnedEmbodimentGeometry, AttentionEmbodimentGeometry, TCPGeometry
from tsn.models.c3_clearance import UncertaintyClearance
from tsn.models.kinematics import PandaKinematics
from tsn.models.perception import RGBPerception
from tsn.models.route import RouteHead, cartesian_proposal, goal_position


def metric_points(dense):
    """Decode predicted normalized XYZ into a 20x20 base-frame point grid."""
    points = F.interpolate(dense[:, :3].float(), (20, 20), mode='nearest-exact').flatten(2).transpose(1, 2)
    return points*points.new_tensor([.55, .55, .5])+points.new_tensor([.65, 0, .22])


class NavigationPolicy(nn.Module):
    chunk_size, execute = 30, 15
    schedule = 'fixed15_main'

    def __init__(self, config, initialize_encoder=True, perception=None):
        super().__init__()
        if (list(config['maps']['point_center_m']) != [.65, 0, .22] or
                list(config['maps']['point_scale_m']) != [.55, .55, .5] or
                config['joint_head']['chunk_size'] != self.chunk_size):
            raise ValueError('The main model requires the shared XYZ normalization and 30-step horizon')
        self.perception = perception if perception is not None else RGBPerception(config, initialize_encoder)
        self.route = RouteHead()
        self.robot = config.get('robot', 'panda')
        contributions = config.get('contributions', {})
        self.use_history = contributions.get('c1', True)
        self.use_embodiment = contributions.get('c2', True)
        self.use_clearance = contributions.get('c3', True)
        self.kinematics = PandaKinematics(self.robot)
        learned = config.get('learned_geometry', {})
        self.learned_c1 = learned.get('c1', False) and self.use_history
        self.learned_c2 = learned.get('c2', False) and self.use_embodiment
        replacement = learned.get('mode', 'adapter') == 'replacement'
        if learned.get('mode', 'adapter') not in ('adapter', 'replacement'):
            raise ValueError('Unknown learned geometry mode')
        if replacement and any(key in learned for key in ('c1_strength', 'c2_strength')):
            raise ValueError('Replacement modules do not accept residual strengths')
        self.embodiment = (AttentionEmbodimentGeometry(self.robot, learned.get('width', 64), learned.get('depth', 2),
                                                      learned.get('calibrated_risk', False))
                          if self.learned_c2 and replacement else
                          LearnedEmbodimentGeometry(self.robot, learned.get('c2_strength', 1.)) if self.learned_c2
                          else EmbodimentGeometry(self.robot) if self.use_embodiment else TCPGeometry())
        self.clearance = UncertaintyClearance(**config['clearance'])
        if not self.use_clearance:
            self.clearance.requires_grad_(False)
        self.memory = PersistentGeometry(**config['memory'], learned=self.learned_c1,
                                         strength=learned.get('c1_strength', 1.),
                                         replacement=replacement and self.learned_c1,
                                         hidden_dim=learned.get('width', 64) if replacement else 32,
                                         depth=learned.get('depth', 2), query_source=learned.get('query_source', 'bank'),
                                         query_capacity=learned.get('query_capacity'))
        self.freeze_encoder = config['perception']['freeze_encoder']
        if self.freeze_encoder:
            self.perception.encoder.requires_grad_(False)
        self.reset_episode()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.perception.encoder.eval()
        return self

    def reset_episode(self):
        self.memory.reset()
        self.history = deque(maxlen=4 if self.use_history else 1)
        self.step = 0

    def observe_step(self, step):
        self.step = int(step)

    def execution_horizon(self, default):
        return self.execute

    def forward(self, rgb, state, K, pose):
        """One live episode -> (1,30,7) joint offsets from measured q, radians."""
        if len(state) != 1:
            raise ValueError('Streaming control expects one episode')
        joint, tokens, geometry, dense = self.perception(rgb, state, K, pose)
        q = state[:, :7].float()*math.pi
        with torch.autocast(device_type=q.device.type, enabled=False):
            tcp, goal = self.kinematics(q), goal_position(state)
            mount = torch.linalg.inv(tcp[0])@pose[0].float()
            points = metric_points(dense)
            fingers = state[0, 7:9].float()*.04
            radius = (self.clearance.predict_error(self.route.visual(tokens.float()), points, pose.float(), tcp)
                      if self.use_clearance else points.new_zeros(points.shape[:2]))
            valid = workspace_mask(points[0]) & ~self.embodiment.self_mask(points[0], tcp[0], fingers, mount)
            if not self.use_history:
                self.memory.reset()
            self.memory.update(points[0], valid, radius[0], self.step, tcp[0, :3, 3], goal[0], state[0])
            self.history.append((tokens.detach(), geometry.detach(), pose.detach(), self.step))
            observed = [torch.stack([frame[i] for frame in self.history], 1) for i in range(3)]
            ages = q.new_tensor([[self.step-frame[3] for frame in self.history]])
            offsets, _ = self.route(*observed, ages, torch.ones_like(ages, dtype=torch.bool), state, tcp)
            positions, rotations, seeds = cartesian_proposal(offsets, joint, q, tcp, goal, self.kinematics)
            surfaces, errors, point_weights = self.memory.query(return_weights=True)
            context = dict(state=state[0], goal=goal[0], arm_points=self.kinematics.arm_points(q[0]))
            refined = (self.clearance.refine(positions[0], rotations[0], tcp[0], goal[0],
                                            surfaces, errors, fingers, self.embodiment, mount, point_weights, context)
                       if self.use_clearance else positions[0])
            targets = self.kinematics.inverse(seeds[0], refined, rotations[0])
        return (targets-q[0])[None]


def load_policy(path, device='cpu'):
    saved = load_checkpoint(path)
    if saved.get('architecture') != 'gtsn_main':
        raise ValueError('Expected a new main-model checkpoint; historical adapters use archived source')
    policy = NavigationPolicy(saved['config']['model'], initialize_encoder=False)
    policy.load_state_dict(saved['model'], strict=True)
    maps = GeometryMaps(saved['config']['model']['maps']).to(device)
    return policy.to(device).eval(), maps, saved
