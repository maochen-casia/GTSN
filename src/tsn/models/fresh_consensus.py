"""Jointly train Pi3 and route heads without detached or cached visual features."""
import torch
from torch import nn

from tsn.features.state import policy_state
from tsn.models.cartesian_policy import CartesianHead, PandaKinematics, goal_xyz
from tsn.models.factory import make_maps, make_policy

MODES = ('state', 'visual', 'uncertainty', 'memory')


class StateRouteHead(nn.Module):
    """The original state route branch, with its unused visual parameters omitted."""
    def __init__(self):
        super().__init__()
        original = CartesianHead('state')
        self.query, self.output = original.query, original.output

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        query = self.query(torch.cat((state.float(), tcp.flatten(1).float(),
                                     (goal_xyz(state)-tcp[:, :3, 3])/.3), -1))
        out = .3*self.output(torch.cat((query, torch.zeros_like(query)), -1)).reshape(-1, 6, 3).tanh()
        return out.float(), {'risk': out.new_zeros(len(out))}


class ZeroDistribution(nn.Module):
    """The deterministic visual branch does not learn a geometry distribution."""
    def forward(self, visual):
        return visual.new_zeros(*visual.shape[:-1], 6)


class FreshConsensus(nn.Module):
    def __init__(self, config, initialize_backbone=True):
        super().__init__()
        self.backbone = make_policy(config, initialize_backbone=initialize_backbone)
        self.heads = nn.ModuleDict({mode: StateRouteHead() if mode == 'state' else CartesianHead(mode)
                                    for mode in MODES})
        self.heads['visual'].distribution = ZeroDistribution()
        self.kinematics = PandaKinematics()
        self.maps = make_maps(config)

    def forward(self, batch):
        # Pack real observations only. Recovery states have no fabricated history.
        mask = batch['history_mask']
        b, h = mask.shape
        selected = mask.flatten().nonzero().flatten()
        states = policy_state(batch['history_qpos'].flatten(0, 1)[selected],
                              batch['goal_pose'][:, None].expand(-1, h, -1).flatten(0, 1)[selected],
                              self.maps.settings)
        with torch.autocast(device_type=mask.device.type, dtype=torch.bfloat16,
                            enabled=mask.device.type == 'cuda'):
            base, tokens, geom, dense = self.backbone(
                batch['history_rgb'].flatten(0, 1)[selected], states,
                batch['K'][:, None].expand(-1, h, -1, -1).flatten(0, 1)[selected],
                batch['history_pose'].flatten(0, 1)[selected], return_features=True, return_maps=True)
            # index_copy is differentiable: every real historical observation is trained.
            def unpack(value):
                return value.new_zeros(b*h, *value.shape[1:]).index_copy(0, selected, value).reshape(b, h, *value.shape[1:])
            token_history, geom_history = unpack(tokens), unpack(geom)
            current = mask.sum(1).cumsum(0)-1
            state = states[current]
            with torch.autocast(device_type=mask.device.type, enabled=False):
                tcp = self.kinematics(batch['qpos'][:, :7])
            predictions, auxiliaries = {}, {}
            for mode, head in self.heads.items():
                predictions[mode], auxiliaries[mode] = head(
                    token_history, geom_history, batch['history_pose'], batch['history_ages'],
                    mask, state, tcp)
            return {'base': base[current], 'maps': dense[current], 'waypoints': predictions,
                    'aux': auxiliaries, 'tcp': tcp}
