"""C2: measured Franka hand and rigid-tool geometry in TCP coordinates."""

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
        for name in (spec['hand'], *spec['fingers'], spec['wrist'], 'camera_link'):
            for collision in links[name].findall('collision'):
                transform = transforms[name]@origin_transform(collision.find('origin'))
                primitive = collision.find('geometry')
                mesh, box = primitive.find('mesh'), primitive.find('box')
                if mesh is not None:
                    vertices = np.asarray(trimesh.load(mesh.get('filename'), process=False).vertices)
                    vertices = vertices*np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                    vertices = vertices@transform[:3, :3].T+transform[:3, 3]
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

    def contact_risk(self, candidates, rotation, tcp, points, padding, fingers, margin=.04, camera_extrinsic=None):
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
        return response.amax(-2).mean((-1, -2))


class TCPGeometry(EmbodimentGeometry):
    """C2 ablation: score only a zero-radius TCP, with no body self filtering."""
    def __init__(self):
        nn.Module.__init__(self)

    def signed_distances(self, points, fingers, camera_extrinsic=None):
        return points.float().norm(dim=-1, keepdim=True)

    def self_mask(self, points, tcp, fingers, camera_extrinsic=None):
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
