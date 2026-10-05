"""Fixed-size, base-frame scene memory with parallel causal training.

The writer never reads its own previous state. Nonlinear encoders and readouts
surround an affine associative scan; streaming and sequence modes share weights.
Geometry slots are spatial addresses, not tracked object/keypoint identities.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from tsn.models.compact_policy import CompactRouteHead, goal_xyz


def affine_scan(a, b, initial=None):
    """Inclusive affine prefix scan on time axis 1, in FP32.

    ``a`` may broadcast over the feature dimension of ``b``. Identity updates
    implement padding; zero multipliers implement episode resets. Doubling
    has logarithmic temporal depth, with no loop over individual frames.
    """
    a, b = a.float(), b.float()
    stride = 1
    while stride < b.shape[1]:
        old_a, old_b = a, b
        a = torch.cat((old_a[:, :stride], old_a[:, stride:] * old_a[:, :-stride]), 1)
        b = torch.cat((old_b[:, :stride], old_b[:, stride:] +
                       old_a[:, stride:] * old_b[:, :-stride]), 1)
        stride *= 2
    if initial is not None:
        b = b + a * initial.float()[:, None]
    return b


def metric_points(dense):
    """Sample actual predicted pixels, avoiding spatial XYZ averaging."""
    points = F.interpolate(dense[:, :3].float(), (20, 20), mode='nearest-exact')
    points = points.flatten(2).transpose(1, 2)
    return points * points.new_tensor([.55, .55, .50]) + points.new_tensor([.65, 0, .22])


class PersistentSceneHead(CompactRouteHead):
    """Eight coarse slots and optional 256 spatial surface-patch slots.

    Existing current-view encoders initialize from CompactRouteHead. A zero
    initialized memory residual starts at its current-frame-only prediction.
    No old raw frames are retained. The first four coarse slots and all point
    slots persist without decay; remaining slots learn input-dependent decay.
    """
    def __init__(self, use_points=True, scene_slots=8, width=128, point_width=32):
        super().__init__()
        self.use_points = use_points
        self.scene_slots, self.width, self.point_width = scene_slots, width, point_width
        self.cell = nn.Sequential(nn.Linear(96 + 3, width), nn.SiLU(), nn.LayerNorm(width))
        self.slot_query = nn.Parameter(torch.randn(scene_slots, width) / math.sqrt(width))
        self.decay = nn.Linear(256, scene_slots)
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, -4.)
        self.scene_read = nn.Sequential(nn.Linear(width + 1, 256), nn.SiLU(), nn.LayerNorm(256))
        self.memory_residual = nn.Linear(256, 256, bias=False)
        nn.init.zeros_(self.memory_residual.weight)
        if use_points:
            axes = [torch.linspace(.10 + .95/16, 1.05 - .95/16, 8),
                    torch.linspace(-.60 + 1.20/16, .60 - 1.20/16, 8),
                    torch.linspace(.04 + .61/8, .65 - .61/8, 4)]
            anchors = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
            self.register_buffer('anchors', anchors)
            # Pixel-to-coarse-cell correspondence is deterministic and view-local.
            ys, xs = torch.meshgrid(torch.arange(20), torch.arange(20), indexing='ij')
            self.register_buffer('pixel_cell', ((ys//5)*4 + xs//5).flatten())
            self.point_encode = nn.Sequential(nn.Linear(64 + 3, point_width), nn.SiLU())
            self.point_distribution = nn.Linear(point_width, 4)
            nn.init.zeros_(self.point_distribution.weight)
            nn.init.zeros_(self.point_distribution.bias)
            self.point_read = nn.Sequential(nn.Linear(point_width + 7, 256), nn.SiLU(), nn.LayerNorm(256))

    def initialize(self, compact):
        """Copy all compatible pretrained route parameters."""
        self.load_state_dict(compact.state_dict(), strict=False)

    def options(self):
        return dict(use_points=self.use_points, scene_slots=self.scene_slots,
                    width=self.width, point_width=self.point_width)

    def encode(self, tokens, geometry, poses):
        visual = self.visual(tokens.float())
        distribution = self.distribution(visual).float()
        mean = geometry[..., :3].float() + .25 * distribution[..., :3].tanh()
        logvar = -5 + 4 * distribution[..., 3:].tanh()
        reliability = (-.5 * logvar.detach().mean(-1)).softmax(-1) * 16
        visual = visual * reliability[..., None]
        geom = self.geometry(torch.cat((mean, geometry[..., 3:].float()), -1))
        cells = torch.cat((visual, geom), -1)
        frame = self.frame(torch.cat((cells.flatten(2), poses.flatten(2).float(),
                                     torch.zeros_like(poses[..., :1, 0])), -1))
        return visual, cells, frame, mean, logvar

    def sequence(self, tokens, geometry, poses, times, valid, state, tcp,
                 points=None, initial=None, resets=None):
        """Return predictions at EVERY causal timestep and the final memory.

        Inputs have (B,T,...) leading axes. Initial state is a tuple of FP32
        coarse/point accumulators. Padding makes identity updates; resets clear
        earlier memory BEFORE incorporating that timestep's observation.
        """
        if valid.dtype != torch.bool or not valid[:, 0].all():
            raise ValueError('Sequences must start with a valid observation')
        visual, cells, frame, mean, logvar = self.encode(tokens, geometry, poses)
        features = self.cell(torch.cat((cells, mean), -1))
        writes = torch.sigmoid(torch.einsum('btip,kp->btki', features, self.slot_query) / math.sqrt(self.width))
        writes = writes * valid[..., None, None]
        mass = writes.sum(-1, keepdim=True).float()
        coarse_write = torch.cat((torch.einsum('btki,btip->btkp', writes, features).float(), mass), -1)
        elapsed = torch.cat((torch.zeros_like(times[:, :1]), times[:, 1:] - times[:, :-1]), 1).clamp_min(0)
        if initial is not None:
            elapsed[:, 0] = times[:, 0].clamp_min(0)
        rate = F.softplus(self.decay(frame).float())
        static = torch.arange(self.scene_slots, device=times.device) < self.scene_slots//2
        rate = rate.masked_fill(static, 0)
        a = torch.exp(-rate * elapsed[..., None] / 15)[..., None]
        a = torch.where(valid[..., None, None], a, torch.ones_like(a))
        if resets is not None:
            a = a.masked_fill(resets[..., None, None] & valid[..., None, None], 0)
        coarse = affine_scan(a, coarse_write, None if initial is None else initial[0])
        coarse_mass = coarse[..., -1:]
        scene = self.scene_read(torch.cat((coarse[..., :-1] / coarse_mass.clamp_min(1e-6),
                                          coarse_mass.log1p()), -1))
        memories, memory_masks = [scene], [coarse_mass[..., 0] > 0]
        auxiliary = dict(mean=mean, logvar=logvar)
        final = [coarse[:, -1]]
        if self.use_points:
            if points is None:
                raise ValueError('Detailed geometry requires unpooled metric points')
            point_features = self.point_encode(torch.cat((visual[..., self.pixel_cell, :],
                                      (points.float() - points.new_tensor([.65, 0, .22])) / .55), -1))
            correction = self.point_distribution(point_features).float()
            corrected = points.float() + .05 * correction[..., :3].tanh()
            confidence = correction[..., 3].sigmoid()
            # Ignore out-of-workspace predictions and the immediate hand/camera.
            usable = torch.isfinite(corrected).all(-1)
            usable = usable & (corrected[..., 0] > .10) & (corrected[..., 0] < 1.05)
            usable = usable & (corrected[..., 1].abs() < .60)
            usable = usable & (corrected[..., 2] > .04) & (corrected[..., 2] < .65)
            usable = usable & ((corrected - tcp[..., None, :3, 3]).norm(dim=-1) > .07)
            usable = usable & ((corrected - poses[..., None, :3, 3]).norm(dim=-1) > .04)
            usable = usable & valid[..., None]
            # Addressing uses frozen raw coordinates, never evolving centroids.
            distance = (points.float()[..., None, :] - self.anchors).square().sum(-1)
            address = distance.argmin(-1)
            weight = confidence * usable.float()
            values = torch.cat((point_features.float(), corrected, corrected.square(),
                                torch.ones_like(weight[..., None])), -1)
            values = torch.where(usable[..., None], values, torch.zeros_like(values))
            point_write = values.new_zeros(*values.shape[:2], len(self.anchors), values.shape[-1])
            point_write = point_write.scatter_add(2, address[..., None].expand_as(values), values * weight[..., None])
            point_a = torch.ones_like(point_write[..., :1])
            if resets is not None:
                point_a = point_a.masked_fill(resets[..., None, None] & valid[..., None, None], 0)
            accumulated = affine_scan(point_a, point_write, None if initial is None else initial[1])
            support = accumulated[..., -1:]
            average = accumulated[..., :-1] / support.clamp_min(1e-6)
            location = average[..., self.point_width:self.point_width+3]
            variance = (average[..., -3:] - location.square()).clamp_min(0)
            detail = self.point_read(torch.cat((average[..., :self.point_width],
                          (location - location.new_tensor([.65, 0, .22]))/.55,
                          variance.clamp_min(1e-8).sqrt()/.55, support.log1p()), -1))
            memories.append(detail)
            memory_masks.append(support[..., 0] > 0)
            auxiliary.update(point_mean=corrected, point_confidence=confidence, point_logits=correction[..., 3],
                             memory_points=location, memory_valid=support[..., 0] > 0)
            final.append(accumulated[:, -1])
        memory = torch.cat(memories, 2)
        memory_valid = torch.cat(memory_masks, 2)
        # Padded timesteps still have at least one valid coarse slot.
        query = self.query(torch.cat((state.float(), tcp.flatten(2).float(),
                             (goal_xyz(state) - tcp[..., :3, 3])/.3), -1))
        scores = (self.key(memory) * query[..., None, :]).sum(-1) / 16
        attention = scores.masked_fill(~memory_valid, -torch.inf).softmax(-1)
        context = (attention[..., None] * memory).sum(2)
        context = frame + self.memory_residual(context)
        prediction = .3 * self.output(torch.cat((query, context), -1)).reshape(*state.shape[:2], 6, 3).tanh()
        return prediction.float(), auxiliary, tuple(final)

    def stream(self, tokens, geometry, pose, elapsed, state, tcp, points=None, memory=None):
        """Single online update, with elapsed CONTROL STEPS since last update."""
        prediction, aux, memory = self.sequence(tokens[:, None], geometry[:, None], pose[:, None],
            elapsed[:, None], torch.ones(len(state), 1, dtype=torch.bool, device=state.device),
            state[:, None], tcp[:, None], None if points is None else points[:, None], memory)
        return prediction[:, 0], {k: v[:, 0] for k, v in aux.items()}, memory


def compact_sequence(head, tokens, geometry, poses, times, valid, state, tcp, current_only=False):
    """Vectorized original four-frame/current-frame baselines at every timestep."""
    batch, length = tokens.shape[:2]
    count = 1 if current_only else 4
    t = torch.arange(length, device=tokens.device)
    offsets = torch.arange(count-1, -1, -1, device=tokens.device)
    cursor = t[None].minimum(valid.sum(1)[:, None]-1)
    indices = (cursor[..., None] - offsets).clamp_min(0)
    b = torch.arange(batch, device=tokens.device)[:, None, None]
    history_valid = valid[b, indices] & (cursor[..., None] >= offsets)
    gathered = [x[b, indices].reshape(batch*length, count, *x.shape[2:]) for x in (tokens, geometry, poses)]
    ages = (times[..., None] - times[b, indices]).reshape(batch*length, count)
    prediction, auxiliary = head(*gathered, ages, history_valid.reshape(batch*length, count),
                                state.reshape(-1, 16), tcp.reshape(-1, 4, 4))
    return prediction.reshape(batch, length, 6, 3), {
        k: v.reshape(batch, length, *v.shape[1:]) for k, v in auxiliary.items()}
