"""URDF collision surfaces moved by measured or candidate robot configurations."""
import numpy as np
import torch
from torch import nn
from scipy.spatial import ConvexHull

from tsn.models.c2_embodiment import origin_transform
from tsn.simulation.robot import robot_spec, robot_tree, S_FROM_CV


class CollisionHull(nn.Module):
    def __init__(self, vertices):
        super().__init__()
        planes = ConvexHull(vertices).equations
        planes /= np.linalg.norm(planes[:, :3], axis=-1, keepdims=True)
        self.register_buffer('planes', torch.tensor(planes, dtype=torch.float32))

    def forward(self, points):
        return (points@self.planes[:, :3].T+self.planes[:, 3]).amax(-1)


class RobotSurfaceCloud(nn.Module):
    """A deterministic surface pool; the network decides which nodes to use.

    Samples lie on collision triangles of every moving arm link, the hand,
    fingers and camera. Physical FK moves these samples; joint origins never
    appear as representation nodes. Meshes are geometry, not pretrained weights.
    """
    def __init__(self, robot='panda', pool_size=2048):
        super().__init__()
        import trimesh
        spec, tree = robot_spec(robot), robot_tree(robot)
        names = [spec['base']]
        pending = list(tree.findall('joint'))
        origins, axes, parents, indices, kinds = [], [], [], [], []
        while pending:
            ready = [joint for joint in pending if joint.find('parent').get('link') in names]
            if not ready:
                raise ValueError('Robot URDF is not a rooted kinematic tree')
            for joint in ready:
                parents.append(names.index(joint.find('parent').get('link')))
                names.append(joint.find('child').get('link'))
                origins.append(origin_transform(joint.find('origin')))
                axis = joint.find('axis')
                axes.append(np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([0., 0., 1.]))
                kind = joint.get('type')
                if kind not in ('fixed', 'revolute', 'prismatic'):
                    raise ValueError('Unsupported surface joint: '+kind)
                kinds.append(kind)
                indices.append(spec['joints'].index(joint.get('name')) if kind != 'fixed' else -1)
                pending.remove(joint)
        self.names, self.parents, self.indices, self.kinds = names, parents, indices, kinds
        self.tcp_index, self.camera_index = names.index(spec['tcp']), names.index('camera_link')
        self.register_buffer('origins', torch.tensor(np.stack(origins), dtype=torch.float32))
        self.register_buffer('axes', torch.tensor(np.stack(axes), dtype=torch.float32))
        links = {link.get('name'): link for link in tree.findall('link')}
        moving = [f'{robot}_link{i}' for i in range(1, 8)]+[spec['hand'], *spec['fingers'], 'camera_link']
        meshes, primitive_links, hulls = [], [], []
        self.surface_link_names = []
        for name in moving:
            surfaces = []
            for collision in links[name].findall('collision'):
                geometry = collision.find('geometry')
                mesh, box = geometry.find('mesh'), geometry.find('box')
                if mesh is not None:
                    shape = trimesh.load(mesh.get('filename'), force='mesh', process=False)
                    shape.vertices = np.asarray(shape.vertices)*np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                elif box is not None:
                    shape = trimesh.creation.box(extents=np.fromstring(box.get('size'), sep=' '))
                else:
                    raise ValueError('Unsupported surface collision primitive on '+name)
                shape.apply_transform(origin_transform(collision.find('origin')))
                hulls.append(CollisionHull(np.asarray(shape.vertices)))
                primitive_links.append(names.index(name))
                surfaces.append(shape)
            if not surfaces:
                raise ValueError('Missing collision surface for '+name)
            meshes.append(trimesh.util.concatenate(surfaces))
            self.surface_link_names.append(name)
        if pool_size < 16*len(meshes):
            raise ValueError('Surface pool must cover every collision link with at least 16 samples')
        self.hulls = nn.ModuleList(hulls)
        self.primitive_links = primitive_links
        area = np.array([shape.area for shape in meshes])
        counts = np.full(len(meshes), 16, dtype=int)
        quota = (pool_size-counts.sum())*area/area.sum()
        counts += np.floor(quota).astype(int)
        counts[np.argsort(-(quota-np.floor(quota)), kind='stable')[:pool_size-counts.sum()]] += 1
        rng = np.random.default_rng(0)
        points, normals, link_indices, labels = [], [], [], []
        for label, (name, shape, count) in enumerate(zip(moving, meshes, counts)):
            triangles = np.asarray(shape.triangles)
            triangle_area = np.asarray(shape.area_faces)
            faces = rng.choice(len(triangles), count, p=triangle_area/triangle_area.sum())
            random = rng.random((count, 2)); root = np.sqrt(random[:, :1])
            weights = np.concatenate((1-root, root*(1-random[:, 1:]), root*random[:, 1:]), -1)
            points.append((triangles[faces]*weights[..., None]).sum(1))
            normals.append(np.asarray(shape.face_normals)[faces])
            link_indices.extend([names.index(name)]*count); labels.extend([label]*count)
        self.register_buffer('local_points', torch.tensor(np.concatenate(points), dtype=torch.float32))
        self.register_buffer('local_normals', torch.tensor(np.concatenate(normals), dtype=torch.float32))
        self.register_buffer('sample_links', torch.tensor(link_indices, dtype=torch.long))
        self.register_buffer('labels', torch.tensor(labels, dtype=torch.long))

    def frames(self, qpos, camera_extrinsic=None):
        if qpos.shape[-1] != 9:
            raise ValueError('Surface FK needs seven arm angles and two finger positions')
        qpos = qpos.float()
        identity = torch.eye(4, device=qpos.device).expand(*qpos.shape[:-1], 4, 4)
        frames = [identity]
        for parent, origin, axis, index, kind in zip(self.parents, self.origins, self.axes, self.indices, self.kinds):
            motion = identity.clone()
            if kind == 'revolute':
                if not torch.equal(axis, axis.new_tensor([0., 0., 1.])):
                    raise ValueError('Unsupported non-z revolute surface joint')
                c, s = qpos[..., index].cos(), qpos[..., index].sin()
                motion[..., 0, 0], motion[..., 1, 1] = c, c
                motion[..., 0, 1], motion[..., 1, 0] = -s, s
            elif kind == 'prismatic':
                motion[..., :3, 3] = qpos[..., index, None]*axis
            frames.append(frames[parent]@origin@motion)
        if camera_extrinsic is not None:
            cv_from_link = identity.clone()
            cv_from_link[..., :3, :3] = qpos.new_tensor(S_FROM_CV.T)
            frames[self.camera_index] = frames[self.tcp_index]@camera_extrinsic.float()@cv_from_link
        return torch.stack(frames, -3)

    def forward(self, qpos, camera_extrinsic=None):
        frames = self.frames(qpos, camera_extrinsic)[..., self.sample_links, :, :]
        points = (frames[..., :3, :3]@self.local_points[..., None])[..., 0]+frames[..., :3, 3]
        normals = (frames[..., :3, :3]@self.local_normals[..., None])[..., 0]
        return points, normals

    def signed_distances(self, points, qpos, camera_extrinsic=None):
        """Convex collision fields for training labels and robot self filtering."""
        frames = self.frames(qpos, camera_extrinsic)
        distances = []
        for hull, link in zip(self.hulls, self.primitive_links):
            frame = frames[..., link, :, :]
            local = (points-frame[..., None, :3, 3])@frame[..., :3, :3]
            distances.append(hull(local))
        return torch.stack(distances, -1).amin(-1)
