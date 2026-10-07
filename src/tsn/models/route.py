"""Four-frame route prediction and 30-step Cartesian/joint conversion."""
import torch
from torch import nn


def goal_position(state):
    return state[..., 9:12].float()*state.new_tensor([.55, .55, .5])+state.new_tensor([.65, 0, .22])


class RouteHead(nn.Module):
    """Attend to 4x4 spatial cells and predict six TCP offsets, in metres."""
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.SiLU())
        self.distribution = nn.Linear(64, 6)
        self.geometry = nn.Sequential(nn.Linear(6, 32), nn.SiLU())
        self.frame = nn.Sequential(nn.Linear(16*96+16+1, 256), nn.SiLU(), nn.LayerNorm(256))
        self.query = nn.Sequential(nn.Linear(35, 256), nn.SiLU(), nn.LayerNorm(256))
        self.key = nn.Linear(256, 256)
        self.output = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 18))
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        """Inputs (B,T,16,768/6); returns (B,6,3) offsets and geometry moments."""
        if not mask.any(-1).all():
            raise ValueError('Each route sample needs a current observation')
        visual = self.visual(tokens.float())
        distribution = self.distribution(visual)
        mean = geometry[..., :3].float()+.25*distribution[..., :3].tanh()
        logvar = -5+4*distribution[..., 3:].tanh()
        reliability = (-.5*logvar.detach().mean(-1)).softmax(-1)*16
        geom = self.geometry(torch.cat((mean, geometry[..., 3:].float()), -1))
        frames = self.frame(torch.cat((torch.cat((visual*reliability[..., None], geom), -1).flatten(2),
                                      poses.flatten(2).float(), ages[..., None].float()/60), -1))
        query = self.query(torch.cat((state.float(), tcp.flatten(1),
                                     (goal_position(state)-tcp[:, :3, 3])/.3), -1))
        scores = (self.key(frames)*query[:, None]).sum(-1)/16
        context = (scores.masked_fill(~mask, -torch.inf).softmax(-1)[..., None]*frames).sum(1)
        offsets = .3*self.output(torch.cat((query, context), -1)).reshape(-1, 6, 3).tanh()
        return offsets.float(), {'mean': mean[:, -1], 'logvar': logvar[:, -1]}


def cartesian_proposal(offsets, joint_proposal, q, tcp, goal, kinematics):
    """Interpolate knots at 5/10/.../30; preserve proposed wrist pose outside servo."""
    knots = torch.cat((torch.zeros_like(offsets[:, :1]), offsets), 1)
    times = torch.arange(1, 31, device=q.device)/5
    lo = times.long().clamp(max=5)
    alpha = (times-lo)[None, :, None]
    positions = tcp[:, None, :3, 3]+knots[:, lo]*(1-alpha)+knots[:, lo+1]*alpha
    near = (goal-tcp[:, :3, 3]).norm(dim=-1) < .08
    direct = tcp[:, None, :3, 3]+(goal-tcp[:, :3, 3])[:, None]*torch.linspace(1/30, 1, 30, device=q.device)[None, :, None]
    positions = torch.where(near[:, None, None], direct, positions)
    seeds = (q[:, None]+joint_proposal.float()).maximum(kinematics.limits[:, 0]).minimum(kinematics.limits[:, 1])
    rotations = kinematics(seeds)[..., :3, :3]
    seeds = torch.where(near[:, None, None], q[:, None], seeds)
    rotations = torch.where(near[:, None, None, None], tcp[:, None, :3, :3], rotations)
    return positions, rotations, seeds
