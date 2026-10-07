"""Bounded persistent geometry from past RGB observations.

Fresh evidence uses the established recent-cloud scorer. A spatial map adds
older surface anchors without interpreting unobserved space as free. Optional
support-weighted reads are a matched control. All coordinates and distances
are in the robot base frame, in metres.
"""
import math

import torch

from tsn.models.geometric_energy import (
    GeometricEnergyPolicy, load_geometric_energy, surface_risk, trust_features,
)


class ConsensusGeometry:
    """A bounded surfel map with independent-view support and disagreement.

    Positions remain anchored to an actual prediction: averaging surfaces near
    discontinuities can invent geometry. Each incoming frame gives at most one
    vote per stored cell, regardless of its pixel count. Capacity eviction uses
    support, disagreement and task proximity, rather than observation age.
    """
    def __init__(self, capacity=1600, merge_radius=.025):
        if capacity < 1 or not math.isfinite(merge_radius) or merge_radius <= 0:
            raise ValueError('Invalid map capacity or merge radius')
        self.capacity, self.merge_radius = capacity, merge_radius
        self.points = self.support = self.scatter = self.first = self.last = None
        self.step = None

    @torch.no_grad()
    def update(self, points, valid, step, tcp, goal):
        step = int(step)
        if self.step is not None and step <= self.step:
            raise ValueError('Map writes must have strictly increasing timestamps')
        self.step = step
        with torch.autocast(device_type=points.device.type, enabled=False):
            p = points.float()
            valid = valid.bool() & torch.isfinite(p).all(-1)
            # Prevent moving hand observations from becoming persistent walls.
            valid &= (p-tcp.float()).norm(dim=-1) > .07
            valid &= ((p[:, 0] > .1) & (p[:, 0] < 1.05) &
                      (p[:, 1].abs() < .6) & (p[:, 2] > .04) & (p[:, 2] < .65))
            p = p[valid]
            # Keep one actual observation per 2 cm cell, deterministically.
            if len(p):
                cells = (p/.02).floor().long()
                _, inverse = torch.unique(cells, dim=0, sorted=True, return_inverse=True)
                order = torch.arange(len(p), device=p.device)
                first = torch.full((int(inverse.max())+1,), len(p), device=p.device, dtype=torch.long)
                first.scatter_reduce_(0, inverse, order, reduce='amin', include_self=True)
                p = p[first]
            if self.points is None:
                self.points = p.clone()
                self.support = p.new_ones(len(p))
                self.scatter = p.new_zeros(len(p))
                self.first = torch.full((len(p),), step, device=p.device, dtype=torch.long)
                self.last = self.first.clone()
            elif len(p):
                if len(self.points):
                    distances = torch.cdist(p, self.points)
                    distance, index = distances.min(-1)
                    matched = distance <= self.merge_radius
                    # A cell gets one support increment per view; use the
                    # closest prediction for its disagreement observation.
                    residual = self.points.new_full((len(self.points),), torch.inf)
                    residual.scatter_reduce_(0, index[matched], distance[matched].square(),
                                             reduce='amin', include_self=True)
                    seen = torch.isfinite(residual)
                    count = self.support[seen].clamp_max(8)
                    self.scatter[seen] = (self.scatter[seen]*count+residual[seen])/(count+1)
                    self.support[seen] += 1
                    self.last[seen] = step
                    p = p[~matched]
                if len(p):
                    self.points = torch.cat((self.points, p))
                    self.support = torch.cat((self.support, p.new_ones(len(p))))
                    self.scatter = torch.cat((self.scatter, p.new_zeros(len(p))))
                    stamps = torch.full((len(p),), step, device=p.device, dtype=torch.long)
                    self.first = torch.cat((self.first, stamps))
                    self.last = torch.cat((self.last, stamps))
            if len(self.points) > self.capacity:
                # Distance to the current TCP-goal segment is a read/write
                # relevance cue, not a scene oracle or expert route.
                direction = goal.float()-tcp.float()
                t = ((self.points-tcp)*direction).sum(-1)/direction.square().sum().clamp_min(1e-8)
                distance = (self.points-(tcp+t.clamp(0, 1)[:, None]*direction)).norm(dim=-1)
                score = self.confidence() + .25*(-distance/.2).exp()
                keep = torch.argsort(score, descending=True, stable=True)[:self.capacity]
                for name in ('points', 'support', 'scatter', 'first', 'last'):
                    setattr(self, name, getattr(self, name)[keep])

    def confidence(self):
        """Repeated views strengthen a surfel; positional disagreement weakens it."""
        return ((self.support-1)/2).clamp(0, 1) * (-self.scatter/(2*.02**2)).exp()

    def state_bytes(self):
        return sum(x.numel()*x.element_size() for x in
                   (self.points, self.support, self.scatter, self.first, self.last) if x is not None)


def weighted_surface_risk(candidates, rotation, tcp, points, weights, margin=.04):
    """Same swept-hand queries, with support weights applied before the max."""
    if not len(points):
        return candidates.new_zeros(candidates.shape[:2])
    valid = torch.isfinite(points).all(-1) & ((points-tcp[0, :3, 3]).norm(dim=-1) > .07)
    safe = torch.where(valid[:, None], points.float(), torch.zeros_like(points).float())
    hand = torch.stack([candidates-rotation[:, None, :, :, 2]*length
                        for length in (0., .05, .10)], -2)[:, :, 2:15:3]
    distance = (hand[..., None, :]-safe).square().sum(-1)
    evidence = (-.5*distance/margin**2).exp()*weights.clamp(0, 1)
    return evidence.masked_fill(~valid, 0).amax(-1).mean((-1, -2))


class AdaptiveGeometryPolicy(GeometricEnergyPolicy):
    """Recent RGB geometry plus anchored spatial memory and optional support weights."""
    MODES = ('persistent', 'recent', 'current', 'unconfirmed', 'persistent_visual',
             'persistent_geometry', 'recent_geometry', 'unconfirmed_geometry')

    def __init__(self, backbone, head, kinematics, trust, mode='unconfirmed'):
        if mode not in self.MODES:
            raise ValueError('Unknown memory mode')
        self.mode = mode
        super().__init__(backbone, head, kinematics, trust,
                         'current_frame' if mode == 'current' else 'full')
        self.route_history_length = (16 if mode == 'persistent_visual' else
                                     1 if mode in ('current', 'persistent_geometry', 'recent_geometry', 'unconfirmed_geometry') else 4)
        self.schedule = 'fixed15_adaptive_geometry_'+mode

    def reset_episode(self):
        super().reset_episode()
        self.geometry_memory = ConsensusGeometry()

    def memory_config(self):
        if self.mode == 'current':
            read = 'Current cloud only; no spatial map or earlier visual features'
        elif self.mode in ('recent', 'recent_geometry'):
            read = 'Fresh four clouds only; no persistent map'
        elif self.mode in ('unconfirmed', 'unconfirmed_geometry'):
            read = 'Fresh four clouds plus old-only unit-weight anchors'
        else:
            read = 'Fresh four clouds plus old-only confidence-weighted surfels'
        return dict(capacity=1600, merge_radius_m=.025, write_voxel_m=.02,
                    support='At most one vote per independently observed frame',
                    read=read,
                    route_feature_frames=self.route_history_length,
                    expiry='No age cutoff; task relevance and support bound capacity')

    def perceive(self, rgb, state, K, pose):
        result = super().perceive(rgb, state, K, pose)
        if self.mode not in ('current', 'recent', 'recent_geometry'):
            with torch.autocast(device_type=state.device.type, enabled=False):
                tcp = self.kinematics(state[:, :7].float()*math.pi)
                points, valid = self.clouds[-1]
                self.geometry_memory.update(points[0], valid[0], self.step,
                                            tcp[0, :3, 3], self.current_goal[0])
        return result

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        candidates, offsets, fade = self.route_candidates(waypoints, tcp)
        points = torch.cat([p for p, _ in self.clouds], 1)
        valid = torch.cat([v for _, v in self.clouds], 1)
        fresh_risk = surface_risk(candidates, rotation, tcp, points, valid)
        risk = fresh_risk
        memory = self.geometry_memory
        old_count = 0
        old_age = 0
        old_weight = 0.
        old_risk = torch.zeros_like(fresh_risk)
        if memory.points is not None:
            # Repeated observations within the fresh window are already in the
            # raw clouds. Read map cells last observed BEFORE that window.
            oldest_fresh_step = max(0, self.step-45)
            old = memory.last < oldest_fresh_step
            weights = memory.confidence()
            if self.mode in ('unconfirmed', 'unconfirmed_geometry'):
                weights = torch.ones_like(weights)
            old &= weights > 0
            old_count = int(old.sum())
            if old_count:
                old_risk = weighted_surface_risk(candidates, rotation, tcp,
                                                memory.points[old], weights[old])
                # Max must happen per hand query before averaging; concatenate
                # fresh (weight 1) and old supported evidence for exact scoring.
                risk = weighted_surface_risk(candidates, rotation, tcp,
                    torch.cat((points[0][valid[0]], memory.points[old])),
                    torch.cat((points.new_ones(int(valid.sum())), weights[old])))
                old_age = int((self.step-memory.last[old]).max())
                old_weight = float(weights[old].mean())
        penalty = self.trust(trust_features(waypoints, tcp, self.current_goal, risk))
        norm = (offsets.norm(dim=-1)/.05).square()[None]
        chosen = (risk+penalty[:, None]*norm).argmin(-1)
        fresh_penalty = self.trust(trust_features(waypoints, tcp, self.current_goal, fresh_risk))
        fresh_choice = (fresh_risk+fresh_penalty[:, None]*norm).argmin(-1)
        self.last_risk = float(risk[0, 0])
        self.diagnostics.append(dict(step=self.step, risk=self.last_risk, choice=int(chosen[0]),
            correction_m=float(offsets[chosen[0]].norm()*fade[0]), points=int(valid.sum()),
            trust_penalty=float(penalty[0]), selected_risk=float(risk[0, chosen[0]]),
            map_points=0 if memory.points is None else len(memory.points),
            old_only_points=old_count, oldest_read_age_steps=old_age, old_mean_weight=old_weight,
            old_risk=float(old_risk[0, 0]), fresh_choice=int(fresh_choice[0]),
            old_changed_choice=bool(chosen[0] != fresh_choice[0]),
            persistent_geometry_bytes=memory.state_bytes(), route_feature_frames=len(self.history)))
        return candidates[torch.arange(len(waypoints), device=waypoints.device), chosen]


def load_adaptive_geometry(path, device='cpu', mode='unconfirmed'):
    original, maps, checkpoint = load_geometric_energy(path, device)
    policy = AdaptiveGeometryPolicy(original.backbone, original.head, original.kinematics,
                                    original.trust, mode).to(device).eval()
    return policy, maps, checkpoint
