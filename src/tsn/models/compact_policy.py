"""Single-memory route head and fixed-horizon compact controller."""
import math

import torch
from torch import nn

from tsn.common.checkpoint import load_checkpoint
from tsn.features.maps import GeometryMaps
from tsn.models.kinematics import PandaKinematics
from tsn.models.pi3_policy import Pi3MapPolicy


def goal_xyz(state):
    return state[..., 9:12].float() * state.new_tensor([.55, .55, .5]) + state.new_tensor([.65, 0, .22])


class CompactRouteHead(nn.Module):
    """Attend to four observed frames with learned geometry reliability."""
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.SiLU())
        self.distribution = nn.Linear(64, 6)
        self.geometry = nn.Sequential(nn.Linear(6, 32), nn.SiLU())
        self.frame = nn.Sequential(nn.Linear(16*96+16+1, 256), nn.SiLU(), nn.LayerNorm(256))
        self.query = nn.Sequential(nn.Linear(16+16+3, 256), nn.SiLU(), nn.LayerNorm(256))
        self.key = nn.Linear(256, 256)
        self.output = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 18))
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        visual = self.visual(tokens.float())
        distribution = self.distribution(visual).float()
        mean = geometry[..., :3].float() + .25 * distribution[..., :3].tanh()
        logvar = -5 + 4 * distribution[..., 3:].tanh()
        reliability = (-.5 * logvar.detach().mean(-1)).softmax(-1) * 16
        visual = visual * reliability[..., None]
        geom = self.geometry(torch.cat((mean, geometry[..., 3:].float()), -1))
        frames = self.frame(torch.cat((torch.cat((visual, geom), -1).flatten(2),
                                      poses.flatten(2).float(), ages[..., None].float() / 60), -1))
        query = self.query(torch.cat((state.float(), tcp.flatten(1).float(),
                                     (goal_xyz(state) - tcp[:, :3, 3]) / .3), -1))
        scores = (self.key(frames) * query[:, None]).sum(-1) / 16
        context = (scores.masked_fill(~mask, -torch.inf).softmax(-1)[..., None] * frames).sum(1)
        out = .3 * self.output(torch.cat((query, context), -1)).reshape(-1, 6, 3).tanh()
        return out.float(), {'mean': mean[:, -1], 'logvar': logvar[:, -1]}


class CompactPolicy(nn.Module):
    """Predict 30 joint targets; execute 15 before observing the next RGB frame."""
    uses_predicted_maps = True
    chunk_size = 30
    history_length = 4
    execute = 15
    servo_radius = .08
    schedule = 'fixed15_single_memory'

    def __init__(self, backbone, head, kinematics):
        super().__init__()
        self.backbone, self.head, self.kinematics = backbone, head, kinematics
        self.reset_episode()

    def reset_episode(self):
        self.history, self.step = [], 0

    def observe_step(self, step):
        self.step = step

    def execution_horizon(self, default):
        return self.execute

    def perceive(self, rgb, state, K, pose):
        return self.backbone(rgb, state, K, pose, return_features=True)

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        return waypoints

    def forward(self, rgb, state, K, pose):
        base, tokens, geom = self.perceive(rgb, state, K, pose)
        self.history.append((tokens.detach(), geom.detach(), pose.detach(), self.step))
        self.history = self.history[-self.history_length:]
        q = state[:, :7].float() * math.pi
        with torch.autocast(device_type=q.device.type, enabled=False):
            tcp = self.kinematics(q)
        goal = goal_xyz(state)
        near = (goal - tcp[:, :3, 3]).norm(dim=-1) < self.servo_radius
        observations = [torch.stack([frame[i] for frame in self.history], 1) for i in range(3)]
        ages = q.new_tensor([[self.step - frame[3] for frame in self.history]])
        delta, _ = self.head(*observations, ages, torch.ones_like(ages, dtype=torch.bool), state, tcp)
        # Interpolate route knots at t=0,5,...,30 into a target for every control step.
        knots = torch.cat((torch.zeros_like(delta[:, :1]), delta), 1)
        times = torch.arange(1, 31, device=q.device) / 5
        lo = times.long().clamp(max=5)
        alpha = (times - lo)[None, :, None]
        displacement = knots[:, lo] * (1-alpha) + knots[:, lo+1] * alpha
        waypoints = tcp[:, None, :3, 3] + displacement
        direct = tcp[:, None, :3, 3] + (goal-tcp[:, :3, 3])[:, None] * torch.linspace(
            1/30, 1, 30, device=q.device)[None, :, None]
        waypoints = torch.where(near[:, None, None], direct, waypoints)
        with torch.autocast(device_type=q.device.type, enabled=False):
            initial = q[:, None].expand(-1, 30, -1)
            rotation = tcp[:, None, :3, :3].expand(-1, 30, -1, -1)
            candidate = q[:, None] + base.float()
            candidate = candidate.maximum(self.kinematics.limits[:, 0]).minimum(self.kinematics.limits[:, 1])
            predicted_rotation = self.kinematics(candidate)[..., :3, :3]
            initial = torch.where(near[:, None, None], initial, candidate)
            rotation = torch.where(near[:, None, None, None], rotation, predicted_rotation)
            waypoints = self.refine_waypoints(waypoints, rotation, tcp, near, pose)
            targets = self.kinematics.inverse(initial.reshape(-1, 7), waypoints.reshape(-1, 3),
                                              rotation.reshape(-1, 3, 3))
        chunk = targets.reshape(-1, 30, 7) - q[:, None]
        # Preserve the exported controller's floating-point round trip.
        return (chunk + q[:, None]) - q[:, None]


def compact_state(checkpoint, *, allow_training_source=False):
    """Read compact weights, or extract only the memory branch for initialization."""
    if checkpoint.get('architecture') == 'compact_consensus' and checkpoint.get('variant') == 'single_memory':
        state = checkpoint['head']
        if 'weights' in state:
            if state['weights'].numel() != 1 or state['weights'].item() != 1:
                raise ValueError('Expected a single unit-weight memory head')
            if any(not key.startswith('heads.0.') for key in state if key != 'weights'):
                raise ValueError('Unexpected extra route heads')
            state = {key.removeprefix('heads.0.'): value for key, value in state.items() if key != 'weights'}
        return checkpoint['backbone'], state
    if allow_training_source and any(key.startswith('heads.memory.') for key in checkpoint.get('model', {})):
        state = checkpoint['model']
        return ({key.removeprefix('backbone.'): value for key, value in state.items() if key.startswith('backbone.')},
                {key.removeprefix('heads.memory.'): value for key, value in state.items() if key.startswith('heads.memory.')})
    raise ValueError('Expected a single_memory compact checkpoint')


def load_compact_policy(path, device='cpu', *, allow_training_source=False):
    """Load the saved single-memory export without constructing discarded models."""
    checkpoint = load_checkpoint(path)
    if checkpoint.get('architecture') == 'route_evidence':
        from tsn.models.evidence_policy import EvidenceRouteHead
        policy, maps, source = load_compact_policy(checkpoint['base_checkpoint'], device)
        head = EvidenceRouteHead(policy.head, **checkpoint['head_config'])
        head.load_state_dict(checkpoint['head'], strict=True)
        policy.head = head.to(device).eval()
        policy.schedule = 'fixed15_route_evidence_' + head.variant
        return policy, maps, {**source, **checkpoint}
    backbone_state, head_state = compact_state(checkpoint, allow_training_source=allow_training_source)
    config = checkpoint['config']['model']
    if config['name'] != 'pi3_map_policy' or config['action_map_source'] != 'predicted' or config['chunk_size'] != 30:
        raise ValueError('Compact control requires the 30-step learned Pi3 backbone')
    channels = ['point_x', 'point_y', 'point_z', 'goal_projected', 'goal_visible', 'future_action']
    if config['map_channels'] != channels:
        raise ValueError('Unexpected Pi3 geometry channel order')
    backbone = Pi3MapPolicy(config, initialize_backbone=False)
    backbone.load_state_dict(backbone_state, strict=True)
    head = CompactRouteHead()
    head.load_state_dict(head_state, strict=True)
    policy = CompactPolicy(backbone, head, PandaKinematics()).to(device).eval()
    return policy, GeometryMaps(config['maps']).to(device), checkpoint
