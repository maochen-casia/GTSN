"""Task-conditioned uncertain geometry tokens and causal keyframe memory."""
import math

import torch
from torch import nn

MODES = ('state_correction', 'visual', 'uncertainty', 'memory')


class GTSNHead(nn.Module):
    def __init__(self, mode='memory', width=128):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width), nn.SiLU())
        self.geometry = nn.Sequential(nn.Linear(6 + 16 + 2, width), nn.SiLU())
        self.distribution = nn.Linear(width, 6)
        nn.init.zeros_(self.distribution.weight)
        nn.init.zeros_(self.distribution.bias)
        self.query = nn.Sequential(nn.Linear(16, width), nn.LayerNorm(width), nn.SiLU())
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Sequential(nn.Linear(width * 2, 256), nn.LayerNorm(256),
                                    nn.SiLU(), nn.Linear(256, 210))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, geometry, poses, ages, mask, state, baseline):
        """Histories are oldest to newest; final entry is always current.

        tokens B,T,16,768; geometry B,T,16,6; poses B,T,4,4;
        ages B,T in control steps; mask B,T (True means observed).
        No mutable memory is used in training; the rollout adapter owns memory.
        """
        if not mask[:, -1].all():
            raise ValueError('Every sample requires a current observation')
        if (ages[mask] < 0).any():
            raise ValueError('Future observations are forbidden')
        if self.mode != 'memory':
            tokens, geometry, poses = tokens[:, -1:], geometry[:, -1:], poses[:, -1:]
            ages, mask = ages[:, -1:], mask[:, -1:]
        visual = self.visual(tokens.float())
        # Keep probability arithmetic and exported calibration metrics in FP32,
        # including under CPU/CUDA bfloat16 autocast.
        distribution = self.distribution(visual).float()
        mean = geometry[..., :3].float() + .25 * distribution[..., :3].tanh()
        logvar = -5 + 4 * distribution[..., 3:].tanh()
        uncertain = self.mode in ('uncertainty', 'memory')
        geom = torch.cat((mean if uncertain else geometry[..., :3].float(),
                          geometry[..., 3:].float()), -1)
        novelty = torch.ones_like(ages[:, :, None]).expand(-1, -1, 16).clone().float()
        if tokens.shape[1] > 1:
            # Novelty relative to earlier frames only, never to self/future tokens.
            for t in range(1, tokens.shape[1]):
                distance = torch.cdist(mean[:, t].detach(), mean[:, :t].detach().flatten(1, 2))
                distance = distance.masked_fill(~mask[:, :t, None].expand(-1, -1, 16).flatten(1)[:, None], 1e3)
                novelty[:, t] = (distance.amin(-1) / .1).clamp(0, 1)
        pose = poses.flatten(2)[:, :, None].expand(-1, -1, 16, -1).float()
        age = (ages.float() / 60)[:, :, None].expand_as(novelty)
        features = visual + self.geometry(torch.cat((geom, pose, age[..., None], novelty[..., None]), -1))
        query = self.query(state.float())
        logits = (self.key(features) * query[:, None, None]).sum(-1) / math.sqrt(query.shape[-1])
        logits = logits - .1 * age
        if uncertain:
            # A proper scoring rule trains variance; action gradients cannot game confidence.
            logits = logits - .5 * logvar.detach().mean(-1)
        logits = logits.masked_fill(~mask[:, :, None], -torch.inf)
        attention = logits.flatten(1).softmax(-1).reshape_as(logits)
        context = (attention[..., None] * self.value(features)).sum((1, 2))
        if self.mode == 'state_correction':
            context = torch.zeros_like(context)
        correction = .35 * self.output(torch.cat((query, context), -1)).reshape(-1, 30, 7).tanh()
        # RMS marginal standard deviation in normalized point coordinates.
        risk = (attention.detach() * logvar.detach().exp().mean(-1)).sum((1, 2)).sqrt()
        return baseline.float() + correction, {'mean': mean[:, -1], 'logvar': logvar[:, -1],
                                               'risk': risk, 'attention': attention}


def geometry_nll(aux, teacher, valid):
    error = (aux['mean'].float() - teacher.float()).square()
    nll = .5 * (error * (-aux['logvar'].float()).exp() + aux['logvar'].float())
    weight = valid[..., None].float()
    return (nll * weight).sum() / (3 * weight.sum().clamp_min(1))


class GTSNPolicy(nn.Module):
    """Episode-local online memory, storing only actual past RGB observations."""
    uses_predicted_maps = True
    chunk_size = 30

    def __init__(self, backbone, head, schedule='fixed15', threshold=float('inf')):
        super().__init__()
        if schedule not in ('fixed15', 'fixed5', 'adaptive'):
            raise ValueError(schedule)
        self.backbone, self.head = backbone, head
        self.schedule, self.threshold = schedule, threshold
        self.reset_episode()

    def reset_episode(self):
        self.history = []
        self.control_step = 0
        self.last_risk = 0.

    def observe_step(self, step):
        self.control_step = step

    def execution_horizon(self, default):
        return 5 if self.schedule == 'fixed5' or (self.schedule == 'adaptive' and self.last_risk > self.threshold) else default

    def forward(self, rgb, state, calibration, camera_transform):
        if len(rgb) != 1:
            raise ValueError('Online memory requires one episode per policy instance')
        base, tokens, geometry = self.backbone(rgb, state, calibration, camera_transform, return_features=True)
        self.history.append((tokens.detach(), geometry.detach(), camera_transform.detach(), self.control_step))
        self.history = self.history[-4:]
        history = self.history if self.head.mode == 'memory' else self.history[-1:]
        ages = torch.tensor([[self.control_step - x[3] for x in history]], device=rgb.device)
        prediction, aux = self.head(torch.stack([x[0] for x in history], 1),
                                    torch.stack([x[1] for x in history], 1),
                                    torch.stack([x[2] for x in history], 1), ages,
                                    torch.ones_like(ages, dtype=torch.bool), state, base)
        self.last_risk = float(aux['risk'][0])
        return prediction
