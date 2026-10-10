"""C1: bounded surface anchors and recent observations, in base-frame metres."""
from collections import deque
import math

import torch
from torch import nn
from tsn.models.geometry_attention import GeometryAttentionBlock


class PointRetention(nn.Module):
    """Task-conditioned point reliability with bounded, linear-size attention.

    Eight latent queries summarize the cloud; points attend to those summaries
    and the measured robot/goal state. A zero output head reproduces unit point
    weights and the original retention order before adapter training.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        if hidden_dim < 4 or hidden_dim % 4:
            raise ValueError('Point attention width must be a positive multiple of four')
        self.point = nn.Sequential(nn.Linear(11, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.context = nn.Linear(16, hidden_dim)
        self.slots = nn.Parameter(torch.randn(1, 8, hidden_dim)*.02)
        self.summarize = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.attend = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, points, radius, support, scatter, age, tcp, goal, state):
        if not len(points):
            return points.new_empty(0)
        # Quantized occupancy describes redundancy without an N x N attention
        # matrix. Geometry is measured in metres, before feature normalization.
        _, inverse, counts = torch.unique((points/.04).floor().long(), dim=0,
                                         return_inverse=True, return_counts=True)
        values = torch.cat(((points-tcp)/.5, (goal-points)/.5,
            radius[:, None]/.1, support[:, None].log1p()/3,
            scatter.clamp_min(0).sqrt()[:, None]/.02,
            age[:, None].float()/400, counts[inverse, None].float().log1p()/3), -1).clamp(-10, 10)
        tokens = self.point(values.float())[None]
        context = self.context(state.float().reshape(1, 1, 16))
        summaries, _ = self.summarize(self.slots+context, tokens, tokens, need_weights=False)
        keys = torch.cat((summaries, context), 1)
        attended, _ = self.attend(tokens, keys, keys, need_weights=False)
        return self.head(tokens+attended)[0, :, 0]


def workspace_mask(points):
    return (torch.isfinite(points).all(-1) & (points[..., 0] > .10) &
            (points[..., 0] < 1.05) & (points[..., 1].abs() < .60) &
            (points[..., 2] > .04) & (points[..., 2] < .65))


class AttentionPointRetention(nn.Module):
    """Learn the entire point priority with stacked latent/point attention and FFNs.

    Cross attention uses 16 latent slots, avoiding a quadratic cloud attention
    matrix. Scores decide voxel representatives, eviction and the queried cloud.
    No geometric priority or residual weighting is added to the network output.
    """
    def __init__(self, hidden_dim=64, depth=2):
        super().__init__()
        if hidden_dim < 4 or hidden_dim % 4 or depth < 1:
            raise ValueError('Invalid replacement memory width/depth')
        self.point = nn.Sequential(nn.Linear(11, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.context = nn.Linear(16, hidden_dim)
        self.slots = nn.Parameter(torch.randn(1, 16, hidden_dim)*.02)
        self.latent_blocks = nn.ModuleList([GeometryAttentionBlock(hidden_dim, cross=True) for _ in range(depth)])
        # Point-to-latent attention is linear in the cloud size. Each point also
        # has a residual FFN; full point self-attention is deliberately avoided.
        self.point_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.point_attention = nn.ModuleList([nn.MultiheadAttention(hidden_dim, 4, batch_first=True) for _ in range(depth)])
        self.point_ffns = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim)) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, points, radius, support, scatter, age, tcp, goal, state):
        if not len(points):
            return points.new_empty(0)
        _, inverse, counts = torch.unique((points/.04).floor().long(), dim=0,
                                         return_inverse=True, return_counts=True)
        values = torch.cat(((points-tcp)/.5, (goal-points)/.5, radius[:, None]/.1,
            support[:, None].log1p()/3, scatter.clamp_min(0).sqrt()[:, None]/.02,
            age[:, None].float()/400, counts[inverse, None].float().log1p()/3), -1).clamp(-10, 10)
        tokens = self.point(values.float())[None]
        context = self.context(state.float().reshape(1, 1, 16))
        slots = self.slots+context
        for block, norm, attention, ffn in zip(self.latent_blocks, self.point_norms,
                                               self.point_attention, self.point_ffns):
            slots = block(slots, tokens)
            keys = torch.cat((slots, context), 1)
            tokens = tokens+attention(norm(tokens), keys, keys, need_weights=False)[0]
            tokens = tokens+ffn(tokens)
        return self.head(tokens)[0, :, 0]


class PersistentGeometry(nn.Module):
    """Keep actual predictions, their creation-time uncertainty and view agreement.

    The legacy path fixes anchors and uses agreement/proximity for eviction.
    Replacement mode learns anchor updates and the entire eviction priority.
    Streaming state belongs to one episode and is not part of model weights.
    """
    def __init__(self, capacity=1600, merge_radius=.025, voxel_size=.02,
                 learned=False, strength=1., hidden_dim=32, replacement=False, depth=2,
                 query_source='bank', query_capacity=None):
        super().__init__()
        if capacity < 1 or not all(math.isfinite(v) and v > 0 for v in (merge_radius, voxel_size)):
            raise ValueError('Invalid persistent geometry settings')
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Memory strength must lie in [0,1]')
        self.capacity, self.merge_radius, self.voxel_size = capacity, merge_radius, voxel_size
        self.replacement = replacement
        if query_source not in ('bank', 'recent') or (query_capacity is not None and query_capacity < 1):
            raise ValueError('Invalid learned query source/capacity')
        self.query_source, self.query_capacity = query_source, query_capacity or capacity
        self.selector = (AttentionPointRetention(hidden_dim, depth) if replacement
                         else PointRetention(hidden_dim) if learned else None)
        self.strength = strength
        self.reset()

    def reset(self):
        self.recent = deque(maxlen=4)
        self.points = self.radius = self.support = self.scatter = self.last = None
        self.step = None
        self.context = None

    def scores(self, points, radius, support=None, scatter=None, age=None):
        if self.selector is None or self.context is None:
            return points.new_zeros(len(points))
        tcp, goal, state = self.context
        return self.selector(points, radius,
            points.new_ones(len(points)) if support is None else support,
            points.new_zeros(len(points)) if scatter is None else scatter,
            points.new_zeros(len(points)) if age is None else age, tcp, goal, state)

    @torch.no_grad()
    def update(self, points, valid, radius, step, tcp, goal, state=None):
        """Insert one self-filtered observation; points (N,3), radius (N,), m."""
        if self.step is not None and step <= self.step:
            raise ValueError('Observation steps must increase; reset between episodes')
        self.step = int(step)
        if self.selector is not None:
            if state is None or state.numel() != 16 or not torch.isfinite(state).all():
                raise ValueError('Learned memory requires a finite measured state of length 16')
            self.context = (tcp.detach(), goal.detach(), state.detach())
        points, radius = points.detach().float(), radius.detach().float()
        valid = valid.bool() & workspace_mask(points) & torch.isfinite(radius)
        valid &= (points-tcp).norm(dim=-1) > .07
        p, u = points[valid], radius[valid]
        self.recent.append((self.step, p, u))
        if len(p):
            _, inverse = torch.unique((p/self.voxel_size).floor().long(), dim=0, return_inverse=True)
            first = torch.full((int(inverse.max())+1,), len(p), device=p.device, dtype=torch.long)
            candidates = torch.arange(len(p), device=p.device)
            if self.selector is not None and (self.replacement or self.strength):
                candidates = torch.argsort(self.scores(p, u), descending=True, stable=True)
            # Rank within each voxel using learned scores; ties preserve the
            # historical first actual observation, including at initialization.
            ranks = torch.empty_like(candidates)
            ranks[candidates] = torch.arange(len(p), device=p.device)
            first.scatter_reduce_(0, inverse, ranks, reduce='amin')
            first = candidates[first]
            p, u = p[first], u[first]
        if self.points is None:
            self.points, self.radius = p.clone(), u.clone()
            self.support, self.scatter = p.new_ones(len(p)), p.new_zeros(len(p))
            self.last = torch.full((len(p),), self.step, device=p.device, dtype=torch.long)
        elif len(p):
            if len(self.points):
                distance, index = torch.cdist(p, self.points).min(-1)
                matched = distance <= self.merge_radius
                residual = self.points.new_full((len(self.points),), torch.inf)
                residual.scatter_reduce_(0, index[matched], distance[matched].square(), reduce='amin')
                seen = torch.isfinite(residual)
                count = self.support[seen].clamp_max(8)
                self.scatter[seen] = (self.scatter[seen]*count+residual[seen])/(count+1)
                self.support[seen] += 1
                self.last[seen] = self.step
                if self.replacement and matched.any():
                    incoming = self.scores(p, u)
                    retained = self.scores(self.points, self.radius, self.support, self.scatter,
                                           self.step-self.last)
                    order = torch.argsort(incoming, descending=True, stable=True)
                    ranks = torch.empty_like(order); ranks[order] = torch.arange(len(p), device=p.device)
                    best = torch.full((len(self.points),), len(p), device=p.device, dtype=torch.long)
                    best.scatter_reduce_(0, index[matched], ranks[matched], reduce='amin')
                    anchors = torch.where(best < len(p))[0]
                    observations = order[best[anchors]]
                    improve = incoming[observations] > retained[anchors]
                    anchors, observations = anchors[improve], observations[improve]
                    self.points[anchors], self.radius[anchors] = p[observations], u[observations]
                p, u = p[~matched], u[~matched]
            if len(p):
                self.points = torch.cat((self.points, p))
                self.radius = torch.cat((self.radius, u))
                self.support = torch.cat((self.support, p.new_ones(len(p))))
                self.scatter = torch.cat((self.scatter, p.new_zeros(len(p))))
                self.last = torch.cat((self.last, torch.full((len(p),), self.step, device=p.device, dtype=torch.long)))
        if len(self.points) > self.capacity:
            if self.replacement:
                priority = self.scores(self.points, self.radius, self.support, self.scatter, self.step-self.last)
            else:
                direction = goal-tcp
                t = ((self.points-tcp)*direction).sum(-1)/direction.square().sum().clamp_min(1e-8)
                distance = (self.points-tcp-t.clamp(0, 1)[:, None]*direction).norm(dim=-1)
                agreement = ((self.support-1)/2).clamp(0, 1)*(-self.scatter/(2*.02**2)).exp()
                priority = agreement+.25*(-distance/.2).exp()
                if self.selector is not None and self.strength:
                    priority = priority+self.strength*self.scores(self.points, self.radius, self.support,
                                                                self.scatter, self.step-self.last).tanh()
            keep = torch.argsort(priority, descending=True, stable=True)[:self.capacity]
            for name in ('points', 'radius', 'support', 'scatter', 'last'):
                setattr(self, name, getattr(self, name)[keep])

    def query(self, return_weights=False):
        """Fresh clouds plus anchors last seen before the fresh window; no expiry."""
        if not self.recent:
            raise RuntimeError('Observe geometry before querying memory')
        if self.replacement and self.query_source == 'bank':
            # The bank contains actual observations. Hard retention changes the
            # geometry presented to C3; learned scores never rescale its risk.
            points, radius = self.points, self.radius+self.scatter.sqrt()
            return (points, radius, points.new_ones(len(points))) if return_weights else (points, radius)
        old = self.last < self.recent[0][0]
        points = torch.cat([p for _, p, _ in self.recent]+[self.points[old]])
        radius = torch.cat([u for _, _, u in self.recent]+[self.radius[old]+self.scatter[old].sqrt()])
        if not return_weights and not self.replacement:
            return points, radius
        support = torch.cat([p.new_ones(len(p)) for _, p, _ in self.recent]+[self.support[old]])
        scatter = torch.cat([p.new_zeros(len(p)) for _, p, _ in self.recent]+[self.scatter[old]])
        age = torch.cat([p.new_full((len(p),), self.step-step) for step, p, _ in self.recent]+
                        [(self.step-self.last[old]).float()])
        if self.replacement:
            keep = torch.argsort(self.scores(points, radius, support, scatter, age),
                                 descending=True, stable=True)[:self.query_capacity]
            points, radius = points[keep], radius[keep]
            return (points, radius, points.new_ones(len(points))) if return_weights else (points, radius)
        weights = 1+self.strength*(2*self.scores(points, radius, support, scatter, age).sigmoid()-1)
        return points, radius, weights
