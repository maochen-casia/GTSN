"""C1: bounded surface anchors and recent observations, in base-frame metres."""
from collections import deque
import math

import torch


def workspace_mask(points):
    return (torch.isfinite(points).all(-1) & (points[..., 0] > .10) &
            (points[..., 0] < 1.05) & (points[..., 1].abs() < .60) &
            (points[..., 2] > .04) & (points[..., 2] < .65))


class PersistentGeometry:
    """Keep actual predictions, their creation-time uncertainty and view agreement.

    Nearby observations update support/scatter without moving an anchor. Capacity
    eviction uses agreement and proximity to the TCP-goal segment, not age.
    Streaming state belongs to one episode and is not part of model weights.
    """
    def __init__(self, capacity=1600, merge_radius=.025, voxel_size=.02):
        if capacity < 1 or not all(math.isfinite(v) and v > 0 for v in (merge_radius, voxel_size)):
            raise ValueError('Invalid persistent geometry settings')
        self.capacity, self.merge_radius, self.voxel_size = capacity, merge_radius, voxel_size
        self.reset()

    def reset(self):
        self.recent = deque(maxlen=4)
        self.points = self.radius = self.support = self.scatter = self.last = None
        self.step = None

    @torch.no_grad()
    def update(self, points, valid, radius, step, tcp, goal):
        """Insert one self-filtered observation; points (N,3), radius (N,), m."""
        if self.step is not None and step <= self.step:
            raise ValueError('Observation steps must increase; reset between episodes')
        self.step = int(step)
        points, radius = points.detach().float(), radius.detach().float()
        valid = valid.bool() & workspace_mask(points) & torch.isfinite(radius)
        valid &= (points-tcp).norm(dim=-1) > .07
        p, u = points[valid], radius[valid]
        self.recent.append((self.step, p, u))
        if len(p):
            _, inverse = torch.unique((p/self.voxel_size).floor().long(), dim=0, return_inverse=True)
            first = torch.full((int(inverse.max())+1,), len(p), device=p.device, dtype=torch.long)
            first.scatter_reduce_(0, inverse, torch.arange(len(p), device=p.device), reduce='amin')
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
                p, u = p[~matched], u[~matched]
            if len(p):
                self.points = torch.cat((self.points, p))
                self.radius = torch.cat((self.radius, u))
                self.support = torch.cat((self.support, p.new_ones(len(p))))
                self.scatter = torch.cat((self.scatter, p.new_zeros(len(p))))
                self.last = torch.cat((self.last, torch.full((len(p),), self.step, device=p.device, dtype=torch.long)))
        if len(self.points) > self.capacity:
            direction = goal-tcp
            t = ((self.points-tcp)*direction).sum(-1)/direction.square().sum().clamp_min(1e-8)
            distance = (self.points-tcp-t.clamp(0, 1)[:, None]*direction).norm(dim=-1)
            agreement = ((self.support-1)/2).clamp(0, 1)*(-self.scatter/(2*.02**2)).exp()
            keep = torch.argsort(agreement+.25*(-distance/.2).exp(), descending=True, stable=True)[:self.capacity]
            for name in ('points', 'radius', 'support', 'scatter', 'last'):
                setattr(self, name, getattr(self, name)[keep])

    def query(self):
        """Fresh clouds plus anchors last seen before the fresh window; no expiry."""
        if not self.recent:
            raise RuntimeError('Observe geometry before querying memory')
        old = self.last < self.recent[0][0]
        points = torch.cat([p for _, p, _ in self.recent]+[self.points[old]])
        radius = torch.cat([u for _, _, u in self.recent]+[self.radius[old]+self.scatter[old].sqrt()])
        return points, radius
