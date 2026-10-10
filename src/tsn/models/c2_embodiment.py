"""C2: measured Franka hand and rigid-tool geometry in TCP coordinates."""

import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from tsn.simulation.robot import robot_spec, robot_tree, S_FROM_CV


def origin_transform(tag):
    matrix = np.eye(4)
    if tag is not None:
        matrix[:3, 3] = np.fromstring(tag.get('xyz', '0 0 0'), sep=' ')
        matrix[:3, :3] = Rotation.from_euler('xyz', np.fromstring(tag.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return matrix


def box_signed_distance(points, centers, rotations, half_sizes):
    """Exact oriented-box signed distance, with one final dimension per box."""
    local = torch.einsum('...ki,kij->...kj', points[..., None, :]-centers, rotations)
    gap = local.abs()-half_sizes
    return gap.clamp_min(0).norm(dim=-1)+gap.amax(-1).clamp_max(0)


class EmbodimentGeometry(nn.Module):
    """Six regions: TCP, palm, left/right fingers, rigid wrist and wrist camera.

    Meshes use convex support planes: exact inside, a lower distance bound
    outside. Boxes are exact. This excludes the articulated arm, and clearance
    scores do not certify collision-free IK motion.
    """
    def __init__(self, robot='panda'):
        super().__init__()
        import trimesh
        spec = robot_spec(robot)
        root = robot_tree(robot)
        self.calibrated_camera = True
        transforms = {spec['tcp']: np.eye(4)}
        motions = {spec['tcp']: np.zeros((2, 3))}
        changed = True
        while changed:
            changed = False
            for joint in root.findall('joint'):
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                transform = origin_transform(joint.find('origin'))
                if joint.get('type') == 'fixed':
                    if parent in transforms and child not in transforms:
                        transforms[child] = transforms[parent]@transform
                        motions[child] = motions[parent].copy(); changed = True
                    elif child in transforms and parent not in transforms:
                        transforms[parent] = transforms[child]@np.linalg.inv(transform)
                        motions[parent] = motions[child].copy(); changed = True
                elif joint.get('type') == 'prismatic' and parent in transforms and child not in transforms:
                    index = dict(zip(spec['fingers'], (0, 1))).get(child)
                    if index is not None:
                        transforms[child] = transforms[parent]@transform
                        motions[child] = motions[parent].copy()
                        motions[child][index] = transforms[child][:3, :3]@np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                        changed = True
        links = {link.get('name'): link for link in root.findall('link')}
        centers, rotations, halves, axes, groups = [], [], [], [], []
        self._mesh_probes = {}
        for name in (spec['hand'], *spec['fingers'], spec['wrist'], 'camera_link'):
            for collision in links[name].findall('collision'):
                transform = transforms[name]@origin_transform(collision.find('origin'))
                primitive = collision.find('geometry')
                mesh, box = primitive.find('mesh'), primitive.find('box')
                if mesh is not None:
                    vertices = np.asarray(trimesh.load(mesh.get('filename'), process=False).vertices)
                    vertices = vertices*np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                    vertices = vertices@transform[:3, :3].T+transform[:3, 3]
                    self._mesh_probes['palm' if name == spec['hand'] else 'wrist'] = vertices[
                        np.linspace(0, len(vertices)-1, 8, dtype=int)].copy()
                    planes = ConvexHull(vertices).equations.copy()
                    planes /= np.linalg.norm(planes[:, :3], axis=-1, keepdims=True)
                    self.register_buffer('palm_planes' if name == spec['hand'] else 'wrist_planes', torch.tensor(planes, dtype=torch.float32))
                elif box is not None:
                    centers.append(transform[:3, 3]); rotations.append(transform[:3, :3])
                    halves.append(np.fromstring(box.get('size'), sep=' ')/2)
                    axes.append(motions[name]); groups.append(name)
                else:
                    raise ValueError('Unsupported collision primitive: '+name)
        for name, values in (('box_centers', centers), ('box_rotations', rotations), ('box_halves', halves), ('finger_axes', axes)):
            self.register_buffer(name, torch.tensor(np.stack(values), dtype=torch.float32))
        for label, link in (('left', spec['fingers'][0]), ('right', spec['fingers'][1]), ('camera', 'camera_link')):
            self.register_buffer(label+'_mask', torch.tensor([name == link for name in groups]))

    def signed_distances(self, points, fingers, camera_extrinsic=None):
        """Points (...,N,3), measured finger positions (2,) -> distances (...,N,6)."""
        if fingers.shape != (2,):
            raise ValueError('Expected two measured finger positions, in metres')
        centers = self.box_centers+torch.einsum('f,kfi->ki', fingers.float(), self.finger_axes)
        rotations = self.box_rotations
        if self.calibrated_camera and camera_extrinsic is not None:
            centers, rotations = centers.clone(), rotations.clone()
            cv_from_s = points.new_tensor(S_FROM_CV.T)
            rotation = camera_extrinsic[:3, :3].float()@cv_from_s
            centers[self.camera_mask] = rotation@points.new_tensor([-.0125, -.02, 0.])+camera_extrinsic[:3, 3].float()
            rotations[self.camera_mask] = rotation
        boxes = box_signed_distance(points.float(), centers, rotations, self.box_halves)
        palm = (points@self.palm_planes[:, :3].T+self.palm_planes[:, 3]).amax(-1)
        wrist = (points@self.wrist_planes[:, :3].T+self.wrist_planes[:, 3]).amax(-1)
        return torch.stack((points.norm(dim=-1), palm, boxes[..., self.left_mask].amin(-1),
                            boxes[..., self.right_mask].amin(-1), wrist, boxes[..., self.camera_mask].amin(-1)), -1)

    def self_mask(self, points, tcp, fingers, camera_extrinsic=None):
        local = (points.float()-tcp[:3, 3])@tcp[:3, :3]
        return self.signed_distances(local, fingers, camera_extrinsic).amin(-1) <= .002

    def contact_risk(self, candidates, rotation, tcp, points, padding, fingers, margin=.04, camera_extrinsic=None,
                     point_weights=None, context=None):
        """Per-region scene max, then mean over six regions and five future times."""
        if not len(points):
            return candidates.new_zeros(len(candidates))
        valid = torch.isfinite(points).all(-1) & torch.isfinite(padding)
        valid &= (points-tcp[:3, 3]).norm(dim=-1) > .07
        safe = torch.where(valid[:, None], points.float(), torch.zeros_like(points).float())
        padding = torch.where(valid, padding, torch.zeros_like(padding)).clamp_min(0)
        relative = safe[None, None]-candidates[:, 2:15:3, None]
        local = torch.einsum('ctni,tij->ctnj', relative, rotation[2:15:3])
        distance = self.signed_distances(local, fingers, camera_extrinsic).clamp_min(0)
        effective = (distance-padding[:, None]).clamp_min(0)
        response = (-.5*(effective/margin).square()).exp().masked_fill(~valid[:, None], 0)
        if point_weights is not None:
            weights = torch.where(valid & torch.isfinite(point_weights), point_weights,
                                  torch.zeros_like(point_weights)).clamp(0, 2)
            response = response*weights[:, None]
        return response.amax(-2).mean((-1, -2))


class LearnedEmbodimentGeometry(EmbodimentGeometry):
    """Moving robot probes attend to scene points and learn regional relevance.

    Measured FK, finger opening and camera calibration update the probe cloud.
    Seven arm probes provide posture context; hand/tool metric distance and self
    filtering remain grounded in the URDF. The learned output rescales each
    region/time/candidate contact response, bounded to [0,2].
    """
    def __init__(self, robot='panda', strength=1., hidden_dim=32):
        super().__init__(robot)
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Embodiment strength must lie in [0,1]')
        if hidden_dim < 4 or hidden_dim % 4:
            raise ValueError('Embodiment attention width must be a positive multiple of four')
        self.strength = strength
        self.register_buffer('palm_probes', torch.tensor(self._mesh_probes['palm'], dtype=torch.float32))
        self.register_buffer('wrist_probes', torch.tensor(self._mesh_probes['wrist'], dtype=torch.float32))
        self.point_embedding = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.scene_embedding = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.region_embedding = nn.Embedding(7, hidden_dim)
        self.context_embedding = nn.Linear(16, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.scene_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.risk_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].bias)

    def robot_points(self, fingers, camera_extrinsic=None, arm_local=None):
        """A metric TCP-frame cloud and region IDs, refreshed for every observation."""
        corners = self.box_centers.new_tensor(list(itertools.product((-1., 1.), repeat=3)))
        centers = self.box_centers+torch.einsum('f,kfi->ki', fingers.float(), self.finger_axes)
        rotations = self.box_rotations
        if camera_extrinsic is not None:
            centers, rotations = centers.clone(), rotations.clone()
            rotation = camera_extrinsic[:3, :3].float()@centers.new_tensor(S_FROM_CV.T)
            centers[self.camera_mask] = rotation@centers.new_tensor([-.0125, -.02, 0.])+camera_extrinsic[:3, 3].float()
            rotations[self.camera_mask] = rotation
        boxes = torch.einsum('kni,kji->knj', corners[None]*self.box_halves[:, None], rotations)+centers[:, None]
        regions = [centers.new_zeros(1, 3), self.palm_probes,
                   boxes[self.left_mask].flatten(0, 1), boxes[self.right_mask].flatten(0, 1),
                   self.wrist_probes, boxes[self.camera_mask].flatten(0, 1)]
        if arm_local is not None:
            regions.append(arm_local)
        ids = torch.cat([torch.full((len(p),), i, device=centers.device, dtype=torch.long)
                         for i, p in enumerate(regions)])
        return torch.cat(regions), ids

    def regional_weights(self, local_scene, valid, padding, candidates, rotation, tcp, fingers,
                         camera_extrinsic, context):
        count, times, _, _ = local_scene.shape
        indices = torch.where(valid)[0]
        if len(indices) > 256:
            indices = indices[torch.linspace(0, len(indices)-1, 256, device=indices.device).long()]
        local_scene, padding = local_scene[:, :, indices], padding[indices]
        state = context['state'].float()
        arm = context.get('arm_points')
        arm_local = (arm-tcp[:3, 3])@tcp[:3, :3] if arm is not None else None
        probes, ids = self.robot_points(fingers, camera_extrinsic, arm_local)
        goal_relative = torch.einsum('cti,tij->ctj', context['goal'][None, None]-candidates[:, 2:15:3],
                                     rotation[2:15:3])
        robot = probes[None, None].expand(count, times, -1, -1)
        features = torch.cat((robot/.3, (goal_relative[:, :, None]-robot)/.5), -1).clamp(-10, 10)
        tokens = self.point_embedding(features).flatten(0, 1)+self.region_embedding(ids)[None]
        tokens = tokens+self.context_embedding(state)[None, None]
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        tokens = tokens+attended
        scene = torch.cat((local_scene/.5, padding[None, None, :, None].expand(count, times, -1, -1)/.1), -1)
        scene = self.scene_embedding(scene.clamp(-10, 10)).flatten(0, 1)
        attended, _ = self.scene_attention(tokens, scene, scene, need_weights=False)
        logits = self.risk_head(tokens+attended)[..., 0]
        logits = torch.stack([logits[:, ids == region].mean(-1) for region in range(6)], -1)
        return (1+self.strength*(2*logits.sigmoid()-1)).reshape(count, times, 6)

    def contact_risk(self, candidates, rotation, tcp, points, padding, fingers, margin=.04, camera_extrinsic=None,
                     point_weights=None, context=None):
        if not len(points):
            return candidates.new_zeros(len(candidates))
        valid = torch.isfinite(points).all(-1) & torch.isfinite(padding)
        valid &= (points-tcp[:3, 3]).norm(dim=-1) > .07
        if not valid.any():
            return candidates.new_zeros(len(candidates))
        safe = torch.where(valid[:, None], points.float(), torch.zeros_like(points).float())
        padding = torch.where(valid, padding, torch.zeros_like(padding)).clamp_min(0)
        relative = safe[None, None]-candidates[:, 2:15:3, None]
        local = torch.einsum('ctni,tij->ctnj', relative, rotation[2:15:3])
        distance = self.signed_distances(local, fingers, camera_extrinsic).clamp_min(0)
        response = (-.5*((distance-padding[:, None]).clamp_min(0)/margin).square()).exp()
        response = response.masked_fill(~valid[:, None], 0)
        if point_weights is not None:
            weights = torch.where(valid & torch.isfinite(point_weights), point_weights,
                                  torch.zeros_like(point_weights)).clamp(0, 2)
            response = response*weights[:, None]
        regional = response.amax(-2)
        if context is None:
            raise ValueError('Learned embodiment requires measured state and goal context')
        weights = self.regional_weights(local, valid, padding, candidates, rotation, tcp, fingers,
                                        camera_extrinsic, context)
        return (regional*weights).mean((-1, -2))


class AttentionEmbodimentGeometry(LearnedEmbodimentGeometry):
    """Direct neural candidate risk from moving body probes and scene clearance.

    URDF distances supply metric features and self filtering. Stacked residual
    self/cross attention and FFNs replace the fixed region/time risk aggregation.
    A learned readout predicts risk directly; no analytic score is blended in.
    """
    def __init__(self, robot='panda', hidden_dim=64, depth=2, calibrated=False):
        from tsn.models.geometry_attention import GeometryAttentionBlock
        # Reuse physical geometry and probe construction, not adapter parameters.
        super().__init__(robot, hidden_dim=hidden_dim)
        for name in ('point_embedding', 'scene_embedding', 'region_embedding', 'context_embedding',
                     'self_attention', 'scene_attention', 'risk_head'):
            delattr(self, name)
        if depth < 1:
            raise ValueError('Replacement embodiment needs at least one attention block')
        self.replacement = True
        self.calibrated = calibrated
        self.body_embedding = nn.Sequential(nn.Linear(10, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.scene_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.region_encoder = nn.Embedding(7, hidden_dim)
        self.state_encoder = nn.Linear(16, hidden_dim)
        self.blocks = nn.ModuleList([GeometryAttentionBlock(hidden_dim, cross=True) for _ in range(depth)])
        self.readout = nn.Parameter(torch.randn(1, 1, hidden_dim)*.02)
        self.pool = GeometryAttentionBlock(hidden_dim, cross=True)
        self.risk_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                       nn.GELU(), nn.Linear(hidden_dim, 1))
        if calibrated:
            del self.readout, self.pool
            self.risk_head = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
            self.priority_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            nn.init.normal_(self.priority_head[-1].weight, std=.001)
            nn.init.zeros_(self.priority_head[-1].bias)

    def descriptors(self, candidates, rotation, tcp, points, padding, fingers,
                    camera_extrinsic, context, margin=.04):
        """Construct measured features, also cacheable for training-only warmup."""
        if context is None:
            raise ValueError('Replacement embodiment requires measured state and goal')
        valid = torch.isfinite(points).all(-1) & torch.isfinite(padding)
        valid &= (points-tcp[:3, 3]).norm(dim=-1) > .07
        if not valid.any():
            return None
        points, padding = points[valid].float(), padding[valid].float().clamp_min(0)
        local = torch.einsum('ctni,tij->ctnj', points[None, None]-candidates[:, 2:15:3, None], rotation[2:15:3])
        distance = self.signed_distances(local, fingers, camera_extrinsic).clamp_min(0)
        # Distances have physical units, with uncertainty carried as a separate
        # feature. They are inputs to a network, never a deployment risk formula.
        if self.calibrated:
            nearest, indices = (distance-padding[:, None]).clamp_min(0).min(-2)
        else:
            nearest, indices = distance.min(-2)
        uncertainty = padding[indices]
        arm = context.get('arm_points')
        arm_local = (arm-tcp[:3, 3])@tcp[:3, :3] if arm is not None else None
        probes, ids = self.robot_points(fingers, camera_extrinsic, arm_local)
        centers = torch.stack([probes[ids == region].mean(0) for region in range(6)])
        count, times = local.shape[:2]
        robot = centers[None, None].expand(count, times, -1, -1)
        relative_goal = torch.einsum('cti,tij->ctj', context['goal'][None, None]-candidates[:, 2:15:3], rotation[2:15:3])
        progress = torch.linspace(0, 1, times, device=local.device)[None, :, None, None].expand(count, times, 6, 1)
        opening = fingers.mean().expand(count, times, 6, 1)/.04
        scale = margin if self.calibrated else .04
        features = torch.cat((robot/.3, (relative_goal[:, :, None]-robot)/.5,
                              nearest[..., None]/scale, uncertainty[..., None]/scale, progress, opening), -1)
        # Arm posture is encoded through additional tokens, updated by measured
        # FK; their features do not assert unmodeled arm collision volumes.
        if arm_local is not None:
            arm_robot = arm_local[None, None].expand(count, times, -1, -1)
            zeros = local.new_zeros(count, times, len(arm_local), 4)
            arm_features = torch.cat((arm_robot/.3, (relative_goal[:, :, None]-arm_robot)/.5, zeros), -1)
            features = torch.cat((features, arm_features), -2)
            region_ids = torch.cat((torch.arange(6, device=local.device), torch.full((len(arm_local),), 6, device=local.device)))
        else:
            region_ids = torch.arange(6, device=local.device)
        # Preserve nearby surfaces in the cross-attention cloud. Selecting the
        # 128 closest physical surfaces is an input-size bound, not a risk score.
        proximity = distance.amin(-1)
        chosen = proximity.topk(min(128, len(points)), largest=False).indices
        scene = torch.cat((local.gather(-2, chosen[..., None].expand(-1, -1, -1, 3))/.5,
            padding[chosen][..., None]/.1), -1)
        return dict(body=features.clamp(-10, 10), scene=scene.clamp(-10, 10),
                    ids=region_ids, state=context['state'].float())

    def predict_descriptors(self, values):
        body, scene = values['body'], values['scene']
        # Accept either one observation or a batch of cached observations.
        leading, times, regions = body.shape[:-3], body.shape[-3], body.shape[-2]
        tokens = self.body_embedding(body).reshape(-1, times, regions, self.state_encoder.out_features)
        state = self.state_encoder(values['state']).reshape(-1, 1, 1, self.state_encoder.out_features)
        tokens = tokens+self.region_encoder(values['ids'])[None, None]+state
        scene = self.scene_encoder(scene).reshape(-1, scene.shape[-2], self.state_encoder.out_features)
        tokens = tokens.flatten(0, 1)
        for block in self.blocks:
            tokens = block(tokens, scene)
        if self.calibrated:
            # Both the local response and attention readout are learned. No
            # analytic risk appears here. Arm tokens influence attention keys,
            # while only the six modeled tool regions contribute risk values.
            tokens = tokens.reshape(-1, times, regions, self.state_encoder.out_features)
            distance = body[..., :6, 6].reshape(-1, times, 6)
            response = self.neural_response(distance)
            priority = self.priority_head(tokens[:, :, :6])[..., 0].flatten(1).softmax(-1)
            return (priority*response.flatten(1)).sum(-1).reshape(leading)
        tokens = tokens.reshape(-1, times*regions, self.state_encoder.out_features)
        summary = self.pool(self.readout.expand(len(tokens), -1, -1), tokens)
        return self.risk_head(summary)[:, 0, 0].sigmoid().reshape(leading)

    def neural_response(self, distance):
        return self.risk_head(torch.stack((distance, distance.square()/10), -1))[..., 0].sigmoid()

    def calibration_loss(self):
        # Physical-kernel supervision is training-only. Sampling distances
        # covers unobserved clearances without reading held-out scenes.
        distance = torch.linspace(0, 10, 256, device=self.box_centers.device)
        return torch.nn.functional.mse_loss(self.neural_response(distance), torch.exp(-.5*distance.square()))

    def contact_risk(self, candidates, rotation, tcp, points, padding, fingers, margin=.04,
                     camera_extrinsic=None, point_weights=None, context=None):
        values = self.descriptors(candidates, rotation, tcp, points, padding, fingers, camera_extrinsic, context, margin)
        return (candidates.new_zeros(len(candidates)) if values is None else self.predict_descriptors(values))


class TCPGeometry(EmbodimentGeometry):
    """C2 ablation: score only a zero-radius TCP, with no body self filtering."""
    def __init__(self):
        nn.Module.__init__(self)

    def signed_distances(self, points, fingers, camera_extrinsic=None):
        return points.float().norm(dim=-1, keepdim=True)

    def self_mask(self, points, tcp, fingers, camera_extrinsic=None):
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
