"""Learned selection and attention over actual moving robot surface nodes."""
import torch
from torch import nn
from torch.nn import functional as F

from tsn.models.c2_embodiment import EmbodimentGeometry
from tsn.models.geometry_attention import GeometryAttentionBlock
from tsn.models.kinematics import PandaKinematics
from tsn.models.robot_surface import RobotSurfaceCloud


class SurfaceEmbodimentGeometry(EmbodimentGeometry):
    """Select 64 unique surface samples by neural scores, then predict risk.

    The dense candidate pool covers the whole moving arm and tool. Only learned
    scores choose representation nodes. Surface membership and FK are physical
    geometry constraints, rather than joint-origin tokens or fixed priorities.
    """
    surface_nodes = True
    calibrated = True
    replacement = True

    def __init__(self, robot='panda', hidden_dim=64, depth=4, nodes=64,
                 pool_size=2048, scene_capacity=1024):
        super().__init__(robot)
        if hidden_dim < 4 or hidden_dim % 4 or depth < 1 or not 1 <= nodes <= pool_size or scene_capacity < 1:
            raise ValueError('Invalid adaptive surface-node architecture')
        self.node_count, self.scene_capacity = nodes, scene_capacity
        self.surface = RobotSurfaceCloud(robot, pool_size)
        self.kinematics = PandaKinematics(robot)
        self.surface_embedding = nn.Sequential(nn.Linear(13, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.link_embedding = nn.Embedding(len(self.surface.surface_link_names), hidden_dim)
        self.state_encoder = nn.Linear(16, hidden_dim)
        self.selection_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                            nn.GELU(), nn.Linear(hidden_dim, 1))
        self.body_embedding = nn.Sequential(nn.Linear(14, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.scene_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.blocks = nn.ModuleList([GeometryAttentionBlock(hidden_dim, cross=True) for _ in range(depth)])
        self.priority_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.risk_head = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        self._training_aux = None

    def self_mask(self, points, tcp, fingers, camera_extrinsic=None, context=None):
        if context is None or 'qpos' not in context:
            raise ValueError('Full robot self filtering needs measured joint state')
        return self.surface.signed_distances(points.float(), context['qpos'], camera_extrinsic) <= .002

    @torch.no_grad()
    def candidate_geometry(self, candidates, rotation, fingers, camera_extrinsic, context):
        count, times = candidates.shape[0], len(candidates[0, 2:15:3])
        seeds = context['joint_seeds'][2:15:3].float().expand(count, times, 7)
        target_rotation = rotation[2:15:3].float().expand(count, times, 3, 3)
        q = self.kinematics.inverse(seeds, candidates[:, 2:15:3], target_rotation)
        qpos = torch.cat((q, fingers.float().expand(count, times, 2)), -1)
        points, normals = self.surface(qpos, camera_extrinsic)
        return qpos, points, normals

    def descriptors(self, candidates, rotation, tcp, points, padding, fingers,
                    camera_extrinsic, context, margin=.04):
        if context is None or not {'state', 'goal', 'qpos', 'joint_seeds'} <= context.keys():
            raise ValueError('Adaptive surface nodes need state, goal, qpos and joint proposal seeds')
        valid = torch.isfinite(points).all(-1) & torch.isfinite(padding)
        valid &= (points-tcp[:3, 3]).norm(dim=-1) > .07
        if not valid.any():
            return None
        points, padding = points[valid].float(), padding[valid].float().clamp_min(0)
        with torch.no_grad():
            qpos, surface, normals = self.candidate_geometry(candidates, rotation, fingers, camera_extrinsic, context)
            count, times, pool = surface.shape[:3]
            distances = torch.cdist(surface, points.expand(count, times, -1, 3))
            nearest, indices = (distances-padding).clamp_min(0).min(-1)
            uncertainty = padding[indices]
            progress = torch.linspace(0, 1, times, device=points.device).reshape(1, times, 1, 1).expand(count, times, pool, 1)
            features = torch.cat(((surface-tcp[:3, 3])/.5, (context['goal']-surface)/.5, normals,
                nearest[..., None]/margin, uncertainty[..., None]/margin, progress,
                fingers.mean().expand(count, times, pool, 1)/.04), -1).clamp(-10, 10)
            # Cross-attention sees the expanded queried scene, without reducing
            # it back to the older 128-point input budget.
            if len(points) > self.scene_capacity:
                raise ValueError('Queried scene exceeds the configured attention capacity')
            relative = (points-tcp[:3, 3])/.5
            scene = torch.cat((relative, padding[:, None]/.1), -1).clamp(-10, 10)
        embedding = (self.surface_embedding(features)+self.link_embedding(self.surface.labels)[None, None]
                     +self.state_encoder(context['state'].float())[None, None, None])
        scores = self.selection_head(embedding)[..., 0]
        keep = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :self.node_count]
        selected = features.gather(-2, keep[..., None].expand(-1, -1, -1, 13))
        selected_scores = scores.gather(-1, keep).sigmoid()
        body = torch.cat((selected, selected_scores[..., None]), -1)
        values = dict(body=body, scene=scene, state=context['state'].float(), scores=scores,
                      indices=keep, positions=surface.gather(-2, keep[..., None].expand(-1, -1, -1, 3)),
                      qpos=qpos, points=points, padding=padding, nearest=nearest, margin=margin)
        return values

    def predict_descriptors(self, values):
        body, scene = values['body'], values['scene']
        count, times, nodes = body.shape[:3]
        tokens = self.body_embedding(body).reshape(count*times, nodes, -1)
        tokens = tokens+self.state_encoder(values['state'])[None, None]
        scene = self.scene_encoder(scene)[None].expand(count*times, -1, -1)
        for block in self.blocks:
            tokens = block(tokens, scene)
        priority = self.priority_head(tokens)[..., 0].reshape(count, times*nodes).softmax(-1)
        response = self.neural_response(body[..., 9]).reshape(count, times*nodes)
        return (priority*response).sum(-1)

    def neural_response(self, distance):
        return self.risk_head(torch.stack((distance, distance.square()/10), -1))[..., 0].sigmoid()

    def calibration_loss(self):
        distance = torch.linspace(0, 10, 256, device=self.box_centers.device)
        return F.mse_loss(self.neural_response(distance), torch.exp(-.5*distance.square()))

    def selection_loss(self, values):
        # Label relevance during training; inference uses neural scores alone.
        labels = .25+.75*torch.exp(-.5*(values['nearest'].detach()/.08).square())
        relevance = F.binary_cross_entropy_with_logits(values['scores'], labels)
        probabilities = values['scores'].softmax(-1)
        mass = torch.zeros(*probabilities.shape[:-1], len(self.surface.surface_link_names), device=probabilities.device)
        mass.scatter_add_(-1, self.surface.labels.expand_as(probabilities), probabilities)
        # Encourage coverage without prescribing a per-link node allocation.
        coverage = -(mass.clamp_min(1e-6).log()).mean()-torch.log(mass.new_tensor(mass.shape[-1]))
        return relevance+.05*coverage

    @torch.no_grad()
    def teacher_risk(self, values, camera_extrinsic):
        distances = self.surface.signed_distances(values['points'], values['qpos'], camera_extrinsic).clamp_min(0)
        effective = (distances-values['padding']).clamp_min(0)
        return torch.exp(-.5*(effective/values['margin']).square()).amax(-1).mean(-1)

    def contact_risk(self, candidates, rotation, tcp, points, padding, fingers, margin=.04,
                     camera_extrinsic=None, point_weights=None, context=None):
        self._training_aux = None
        values = self.descriptors(candidates, rotation, tcp, points, padding, fingers,
                                  camera_extrinsic, context, margin)
        if values is None:
            risk = candidates.new_zeros(len(candidates))
            if self.training and torch.is_grad_enabled():
                self._training_aux = dict(risk=risk, teacher=risk,
                    selection_loss=sum(p.sum()*0 for p in self.selection_head.parameters()))
            return risk
        risk = self.predict_descriptors(values)
        if self.training and torch.is_grad_enabled():
            self._training_aux = dict(risk=risk, teacher=self.teacher_risk(values, camera_extrinsic),
                                     selection_loss=self.selection_loss(values))
        return risk

    def pop_training_aux(self):
        value, self._training_aux = self._training_aux, None
        if value is None:
            raise RuntimeError('Evaluate contact risk before consuming surface supervision')
        return value
