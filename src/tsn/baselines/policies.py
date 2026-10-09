"""Four baseline algorithms with a shared observation/action contract."""
import math
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from tsn.baselines.carp import ActionTokenizer, NextScalePolicy
from tsn.baselines.networks import ConditionalUNet, ObservationEncoder
from tsn.baselines.observations import observation_state, point_cloud, rgb_image
from tsn.features.camera import resize_observation

METHODS = ('diffusion_policy', 'dp3', 'flowpolicy', 'carp')


def consistency_loss(network, target, condition, noise, t, delta=.01, alpha=1e-5, segments=2, boundary=1.):
    """FlowPolicy's endpoint consistency plus velocity consistency, not plain FM.

    Matches official flowpolicy.py compute_loss (including gradients through both
    neighboring velocities). Two training segments and one-step deployment match
    the released implementation's defaults.
    """
    r = (t+delta).clamp(max=1.)
    end = torch.ceil(t*segments)/segments
    tt, rr, ee = (x[:, None, None] for x in (t, r, end))
    xt, xr = tt*target+(1-tt)*noise, rr*target+(1-rr)*noise
    vt, vr = network(xt, t*99, condition), network(xr, r*99, condition)
    ft = xt+(ee-tt)*vt
    fr = torch.where(rr < boundary, xr+(ee-rr)*vr, ee*target+(1-ee)*noise)
    endpoint = (ft-fr).square().flatten(1).mean(1)
    mask = (t < boundary) & ((end-t) > 1.01*delta)
    velocity = (vt-vr).square().flatten(1).mean(1)*mask
    return (endpoint+alpha*velocity).mean()


class BaselinePolicy(nn.Module):
    chunk_size, execute, robot = 30, 15, 'panda'
    use_embodiment, use_clearance, use_history = False, False, True

    def __init__(self, config):
        super().__init__()
        self.method = config['baseline']['method']
        if self.method not in METHODS:
            raise ValueError(f'Unknown baseline: {self.method}')
        self.requires_depth = self.method in ('dp3', 'flowpolicy')
        self.observation_modality = 'depth_point_cloud' if self.requires_depth else 'rgb'
        self.config = config
        self.encoder = ObservationEncoder('points' if self.requires_depth else 'rgb')
        if self.method == 'carp':
            self.action_model = NextScalePolicy(self.encoder.output_dim, ActionTokenizer())
        else:
            self.action_model = ConditionalUNet(self.encoder.output_dim)
        self.register_buffer('action_center', torch.zeros(7))
        self.register_buffer('action_scale', torch.ones(7))
        steps = torch.arange(101, dtype=torch.float32)
        alpha_bar = torch.cos((steps/100+.008)/1.008*math.pi/2).square()
        beta = (1-alpha_bar[1:]/alpha_bar[:-1]).clamp(.0001, .999)
        self.register_buffer('alpha_bar', (1-beta).cumprod(0))
        self.history = deque(maxlen=2)
        self.generator = None

    @property
    def tokenizer(self):
        return self.action_model.tokenizer

    def normalize(self, actions):
        return (actions-self.action_center)/self.action_scale

    def loss(self, batch):
        condition = self.encoder(batch)
        target = self.normalize(batch['target'])
        if self.method == 'carp':
            return self.action_model.loss(target, condition)
        noise = torch.randn_like(target)
        if self.method == 'flowpolicy':
            t = torch.rand(len(target), device=target.device)*.99+.01
            return consistency_loss(self.action_model, target, condition, noise, t)
        step = torch.randint(0, 100, (len(target),), device=target.device)
        alpha = self.alpha_bar[step, None, None]
        noisy = alpha.sqrt()*target+(1-alpha).sqrt()*noise
        output = self.action_model(noisy, step, condition)
        desired = target if self.method == 'dp3' else noise
        return F.mse_loss(output, desired)

    def sample(self, batch, generator=None):
        condition = self.encoder(batch)
        actions = torch.randn((len(condition), 30, 7), device=condition.device, generator=generator)
        if self.method == 'carp':
            actions = self.action_model.sample(condition, generator)
        elif self.method == 'flowpolicy':
            t = torch.full((len(condition),), .99, device=condition.device)  # eps=.01, scaled by 99
            actions = actions+self.action_model(actions, t, condition)
        else:
            schedule = torch.linspace(99, 0, self.config['baseline']['inference_steps'], device=condition.device).round().long()
            for i, step in enumerate(schedule):
                alpha = self.alpha_bar[step]
                prediction = self.action_model(actions, step.expand(len(actions)), condition)
                if self.method == 'dp3':
                    clean = prediction.clamp(-1, 1)
                    noise = (actions-alpha.sqrt()*clean)/(1-alpha).sqrt()
                else:
                    noise = prediction
                    clean = ((actions-(1-alpha).sqrt()*noise)/alpha.sqrt()).clamp(-1, 1)
                    noise = (actions-alpha.sqrt()*clean)/(1-alpha).sqrt()
                previous = self.alpha_bar[schedule[i+1]] if i+1 < len(schedule) else actions.new_tensor(1.)
                actions = previous.sqrt()*clean+(1-previous).sqrt()*noise
        return actions.float()*self.action_scale+self.action_center

    def reset_episode(self):
        self.history.clear()
        device = self.action_center.device
        self.generator = torch.Generator(device=device).manual_seed(self.config['train']['seed'])

    def predict_observation(self, rgb, depth, qpos, goal, K, pose):
        device = self.action_center.device
        convert = lambda value: torch.as_tensor(value, device=device)[None]
        frame = resize_observation({'rgb': convert(rgb)[0], 'depth': convert(depth)[0], 'K': convert(K)[0]},
                                   self.config['train']['observation_hw'])
        calibration = frame['K'][None]
        state = observation_state(convert(qpos), convert(goal), calibration, convert(pose),
                                  frame['depth'].shape, self.config['model']['maps'])
        visual = point_cloud(frame['depth'][None], calibration, convert(pose)) if self.requires_depth else rgb_image(frame['rgb'][None])
        self.history.append((visual[0], state[0]))
        context = list(self.history)
        if len(context) == 1:
            context = context*2
        batch = {'points' if self.requires_depth else 'rgb': torch.stack([x[0] for x in context])[None],
                 'state': torch.stack([x[1] for x in context])[None]}
        return self.sample(batch, self.generator)
