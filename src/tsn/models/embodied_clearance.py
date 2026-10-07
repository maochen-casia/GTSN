"""URDF-conditioned hand/tool volume fields over persistent uncertain surfaces."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from tsn.models.geometric_energy import trust_features
from tsn.models.adaptive_geometry import ConsensusGeometry
from tsn.models.uncertain_clearance import (
    UncertainClearancePolicy, load_uncertain_clearance, padding_factor,
)


def origin_matrix(tag):
    matrix = np.eye(4)
    if tag is not None:
        matrix[:3, 3] = np.fromstring(tag.get('xyz', '0 0 0'), sep=' ')
        matrix[:3, :3] = Rotation.from_euler('xyz', np.fromstring(tag.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return matrix


def box_signed_distance(points, centers, rotations, half):
    """Exact oriented-box signed distance; negative inside, metres outside."""
    relative = points[..., None, :]-centers
    local = torch.einsum('...ki,kij->...kj', relative, rotations)
    gap = local.abs()-half
    return gap.clamp_min(0).norm(dim=-1)+gap.amax(-1).clamp_max(0)


class PandaToolGeometry(nn.Module):
    """Rigid palm/wrist/camera plus measured prismatic fingers, in TCP axes.

    Collision meshes become convex support-plane fields. These are exact inside
    the convex hull and conservative lower bounds on Euclidean distance outside
    it; corner distance is not exact. Finger/camera boxes use exact distances.
    No scene, simulator contacts or expert trajectory enters this representation.
    """
    def __init__(self):
        super().__init__()
        from mani_skill import PACKAGE_ASSET_DIR
        import trimesh
        path = Path(PACKAGE_ASSET_DIR)/'robots/panda/panda_v3.urdf'
        root = ET.parse(path).getroot()
        joints = root.findall('joint')
        # Traverse the rigid tool from its TCP; actuated arm joints are barriers.
        transforms = {'panda_hand_tcp':np.eye(4)}
        axes = {'panda_hand_tcp':np.zeros((2,3))}
        changed = True
        while changed:
            changed = False
            for joint in joints:
                kind = joint.get('type')
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                transform = origin_matrix(joint.find('origin'))
                if kind == 'fixed':
                    if parent in transforms and child not in transforms:
                        transforms[child] = transforms[parent]@transform
                        axes[child] = axes[parent].copy(); changed = True
                    elif child in transforms and parent not in transforms:
                        transforms[parent] = transforms[child]@np.linalg.inv(transform)
                        axes[parent] = axes[child].copy(); changed = True
                elif kind == 'prismatic' and parent in transforms and child not in transforms:
                    index = {'panda_leftfinger':0,'panda_rightfinger':1}.get(child)
                    if index is not None:
                        transforms[child] = transforms[parent]@transform
                        axes[child] = axes[parent].copy()
                        axes[child][index] = transforms[child][:3,:3]@np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                        changed = True
        links = {v.get('name'):v for v in root.findall('link')}
        records, hashes = [], {str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
        centers, rotations, halves, motion, groups = [], [], [], [], []
        for name in ('panda_hand','panda_leftfinger','panda_rightfinger','panda_link7','camera_link'):
            if name not in transforms:
                raise ValueError('Missing rigid tool transform: '+name)
            for collision in links[name].findall('collision'):
                transform = transforms[name]@origin_matrix(collision.find('origin'))
                geometry = collision.find('geometry')
                mesh, box = geometry.find('mesh'), geometry.find('box')
                if mesh is not None:
                    mesh_path = (path.parent/mesh.get('filename')).resolve()
                    loaded = trimesh.load(mesh_path,process=False)
                    scale = np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                    vertices = np.asarray(loaded.vertices)*scale
                    vertices = vertices@transform[:3,:3].T+transform[:3,3]
                    equations = ConvexHull(vertices).equations.copy()
                    equations /= np.linalg.norm(equations[:,:3],axis=-1,keepdims=True)
                    buffer = 'palm_planes' if name == 'panda_hand' else 'wrist_planes'
                    self.register_buffer(buffer,torch.tensor(equations,dtype=torch.float32))
                    hashes[str(mesh_path)] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
                    records.append(dict(link=name,representation='Convex support planes',planes=len(equations),
                                        tcp_bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()]))
                elif box is not None:
                    half = np.fromstring(box.get('size'),sep=' ')/2
                    centers.append(transform[:3,3]); rotations.append(transform[:3,:3]); halves.append(half)
                    motion.append(axes[name]); groups.append(name)
                    records.append(dict(link=name,representation='Oriented collision box',center_tcp_m=centers[-1].tolist(),
                                        half_size_m=half.tolist(),finger_axes_tcp=axes[name].tolist()))
                else:
                    raise ValueError('Unsupported tool collision primitive')
        self.register_buffer('box_centers',torch.tensor(np.stack(centers),dtype=torch.float32))
        self.register_buffer('box_rotations',torch.tensor(np.stack(rotations),dtype=torch.float32))
        self.register_buffer('box_half',torch.tensor(np.stack(halves),dtype=torch.float32))
        self.register_buffer('box_motion',torch.tensor(np.stack(motion),dtype=torch.float32))
        self.register_buffer('finger_mask',torch.tensor([n!='camera_link' for n in groups]))
        self.metadata = dict(assets_sha256=hashes,bodies=records,
            fingers='Measured state[7:9] * .04 m; no commanded change',
            meshes='Convex support-plane distance, exact inside hull; lower bound outside',
            tcp='Retained as a virtual reference point in the union',
            scope='Palm + fingers; tool additionally includes rigid link7 and wrist camera; excludes articulated arm')

    def distance(self, points, fingers, tool=False,signed=False):
        """Distance to TCP union with the selected physical volume, no inflation."""
        if fingers.shape != (2,):
            raise ValueError('Streaming geometry requires two measured finger positions')
        palm = (points@self.palm_planes[:,:3].T+self.palm_planes[:,3]).amax(-1)
        centers = self.box_centers+torch.einsum('f,kfi->ki',fingers.float(),self.box_motion)
        mask = torch.ones_like(self.finger_mask) if tool else self.finger_mask
        boxes = box_signed_distance(points,centers[mask],self.box_rotations[mask],self.box_half[mask]).amin(-1)
        distance = torch.minimum(torch.minimum(palm,boxes),points.norm(dim=-1))
        if tool:
            wrist = (points@self.wrist_planes[:,:3].T+self.wrist_planes[:,3]).amax(-1)
            distance = torch.minimum(distance,wrist)
        return distance if signed else distance.clamp_min(0)

    def part_distance(self,points,fingers,tool=False,signed=False):
        """Named TCP/palm/finger/wrist/camera clearance features, not one union."""
        if fingers.shape!=(2,):raise ValueError('Two measured finger positions required')
        palm = (points@self.palm_planes[:,:3].T+self.palm_planes[:,3]).amax(-1)
        centers = self.box_centers+torch.einsum('f,kfi->ki',fingers.float(),self.box_motion)
        boxes = box_signed_distance(points,centers,self.box_rotations,self.box_half)
        values = [points.norm(dim=-1),palm,boxes[...,:4].amin(-1),boxes[...,4:8].amin(-1)]
        if tool:
            wrist = (points@self.wrist_planes[:,:3].T+self.wrist_planes[:,3]).amax(-1)
            values += [wrist,boxes[...,8]]
        result = torch.stack(values,-1)
        return result if signed else result.clamp_min(0)


class BodyFilteredGeometry(ConsensusGeometry):
    """Use known current robot volume to reject self surfaces at observation time."""
    def __init__(self,geometry,tool):
        super().__init__()
        self.geometry,self.tool = geometry,tool
        self.tcp = self.fingers = self.last_self_mask = None

    def set_pose(self,tcp,fingers):
        self.tcp,self.fingers = tcp,fingers

    def update(self,points,valid,step,tcp,goal):
        if self.tcp is None:
            raise ValueError('Measured robot pose required before self filtering')
        local = (points.float()-self.tcp[:3,3])@self.tcp[:3,:3]
        self.last_self_mask = self.geometry.distance(local,self.fingers,tool=self.tool)<=.002
        super().update(points,valid&~self.last_self_mask,step,tcp,goal)


def wrist_pose_candidates(candidates,rotation,offsets,fade,wide=False):
    """Smooth TCP-axis roll alternatives; candidate zero preserves the route."""
    values = ([0.,-math.pi/6,math.pi/6,-math.pi/3,math.pi/3,-math.pi/2,math.pi/2] if wide else
              [0.,-math.pi/12,math.pi/12,-math.pi/6,math.pi/6])
    angles = candidates.new_tensor(values)
    ramp = torch.linspace(1/30,1,30,device=candidates.device).sin()*1.1883951
    theta = fade[:,None,None]*angles[None,:,None]*ramp[None,None]
    c,s = theta.cos(),theta.sin()
    roll = torch.eye(3,device=candidates.device).expand(*theta.shape,3,3).clone()
    roll[...,0,0],roll[...,1,1] = c,c
    roll[...,0,1],roll[...,1,0] = -s,s
    rotations = rotation[:,None]@roll
    count = candidates.shape[1]
    position = candidates[:,:,None].expand(-1,-1,len(angles),-1,-1).reshape(len(candidates),-1,30,3)
    rotations = rotations[:,None].expand(-1,count,-1,-1,-1,-1).reshape(len(candidates),-1,30,3,3)
    offsets = offsets[:,None].expand(-1,len(angles),-1).reshape(-1,3)
    angles = angles[None].expand(count,-1).reshape(-1)
    return position,rotations,offsets,angles


def embodied_surface_risk(candidates, rotation, tcp, points, padding, geometry=None,
                          fingers=None, mode='tcp', body_weight=1., margin=.04,field='gaussian',representation='union'):
    """TCP-only or blended TCP/volume proximity at five identical future times."""
    if not len(points):
        zero = candidates.new_zeros(candidates.shape[:2])
        return zero,zero,zero
    valid = torch.isfinite(points).all(-1)&torch.isfinite(padding)
    valid &= (points-tcp[0,:3,3]).norm(dim=-1)>.07
    safe = torch.where(valid[:,None],points.float(),torch.zeros_like(points).float())
    sampled = candidates[:,:,2:15:3]
    relative = safe[None,None,None]-sampled[...,None,:]
    tcp_distance = relative.norm(dim=-1)
    def proximity(distance):
        if field=='signed':
            value = F.softplus((padding.clamp_min(0)-distance)/margin)/math.log(2)
        elif field=='gaussian':
            effective = (distance-padding.clamp_min(0)).clamp_min(0)
            value = (-.5*effective.square()/points.new_tensor(margin).square()).exp()
        else:
            raise ValueError('Unknown embodiment clearance field')
        return value.masked_fill(~valid,0).amax(-1).mean(-1)
    tcp_risk = proximity(tcp_distance)
    if mode == 'tcp' or body_weight == 0:
        return tcp_risk,tcp_risk,tcp_risk
    if mode not in ('hand','tool') or geometry is None or fingers is None:
        raise ValueError('Volume scoring requires valid embodiment geometry')
    if rotation.ndim == 5:
        local = torch.einsum('bctni,bctij->bctnj',relative,rotation[:,:,2:15:3])
    else:
        local = torch.einsum('bctni,btij->bctnj',relative,rotation[:,2:15:3])
    if representation=='parts':
        distances = geometry.part_distance(local,fingers,tool=mode=='tool',signed=field=='signed')
        if field=='signed':
            value = F.softplus((padding.clamp_min(0)[:,None]-distances)/margin)/math.log(2)
        else:
            effective = (distances-padding.clamp_min(0)[:,None]).clamp_min(0)
            value = (-.5*effective.square()/points.new_tensor(margin).square()).exp()
        body_risk = value.masked_fill(~valid[:,None],0).amax(-2).mean((-1,-2))
        risk = (1-body_weight)*tcp_risk+body_weight*body_risk
    elif representation=='union':
        body_risk = proximity(geometry.distance(local,fingers,tool=mode=='tool',signed=field=='signed'))
        risk = tcp_risk+body_weight*(body_risk-tcp_risk).clamp_min(0)
    else:
        raise ValueError('Unknown body contact representation')
    return risk,tcp_risk,body_risk


class EmbodiedClearancePolicy(UncertainClearancePolicy):
    """Keep C1/C3 fixed and replace only the hand geometry scoring interface."""
    BODY_MODES = ('tcp','axial','hand','tool')

    def __init__(self,backbone,head,kinematics,trust,uncertainty,mode='tool',body_weight=.7,
                 max_padding=.03,uniform_factor=.5,mask_self=False,pose_mode='fixed',field='gaussian',representation='union'):
        if mode not in self.BODY_MODES:
            raise ValueError('Unknown embodiment mode')
        if not math.isfinite(body_weight) or not 0<=body_weight<=1:
            raise ValueError('Body contribution must lie in [0,1]')
        if pose_mode not in ('fixed','roll','wide_roll'):
            raise ValueError('Unknown posture refinement mode')
        if field not in ('gaussian','signed'):
            raise ValueError('Unknown embodiment clearance field')
        if representation not in ('union','parts'):
            raise ValueError('Unknown body contact representation')
        super().__init__(backbone,head,kinematics,trust,uncertainty,'adaptive',max_padding,uniform_factor)
        self.body_geometry = PandaToolGeometry()
        self.body_mode,self.body_weight = mode,body_weight
        self.mask_self,self.pose_mode,self.body_field = bool(mask_self),pose_mode,field
        self.body_representation = representation
        if self.mask_self and mode in ('hand','tool'):
            self.geometry_memory = BodyFilteredGeometry(self.body_geometry,mode=='tool')
        self.schedule = f'fixed15_embodied_{mode}_{body_weight:.2f}_{mask_self}_{pose_mode}_{field}'

    def reset_episode(self):
        super().reset_episode()
        self.finger_positions = None
        if getattr(self,'mask_self',False) and self.body_mode in ('hand','tool'):
            self.geometry_memory = BodyFilteredGeometry(self.body_geometry,self.body_mode=='tool')

    def perceive(self,rgb,state,K,pose):
        self.finger_positions = (state[0,7:9].float()*.04).detach().clone()
        if self.mask_self and self.body_mode in ('hand','tool'):
            with torch.autocast(device_type=state.device.type,enabled=False):
                tcp = self.kinematics(state[:,:7].float()*math.pi)[0]
                self.geometry_memory.set_pose(tcp,self.finger_positions)
        result = super().perceive(rgb,state,K,pose)
        if self.mask_self and self.body_mode in ('hand','tool'):
            points,valid = self.clouds[-1]
            self.clouds[-1] = points,valid&~self.geometry_memory.last_self_mask[None]
        return result

    def embodiment_config(self):
        return dict(mode=self.body_mode,body_weight=self.body_weight,geometry=self.body_geometry.metadata,
                    representation=self.body_representation,parts=['TCP','palm','left finger','right finger','wrist','camera'],
                    aggregation='Mean of per-part, per-time maximum surface response' if self.body_representation=='parts' else 'Maximum over all body regions before temporal mean',
                    field=self.body_field,signed_cost='softplus((padding-signed_distance)/.04)/log(2)',
                    self_surface_mask=self.mask_self and self.body_mode in ('hand','tool'),self_mask_distance_m=.002,
                    posture=self.pose_mode,roll_angles_deg=([0,-30,30,-60,60,-90,90] if self.pose_mode=='wide_roll' else
                        [0,-15,15,-30,30] if self.pose_mode=='roll' else [0]),roll_cost=.015,
                    risk='(1-weight)*TCP risk + weight*regional risk' if self.body_representation=='parts' else
                         'TCP risk + weight * max(volume risk - TCP risk, 0)',
                    time_steps=[3,6,9,12,15],margin_m=.04,
                    shared='Same C1 map, C3 uncertainty, route/trust, candidate lattice, IK and execution')

    def refine_waypoints(self,waypoints,rotation,tcp,near,pose):
        if self.body_mode == 'axial':
            result = super().refine_waypoints(waypoints,rotation,tcp,near,pose)
            self.diagnostics[-1].update(body_mode='axial',body_weight=1.)
            return result
        candidates,offsets,fade = self.route_candidates(waypoints,tcp)
        query_rotation = rotation
        angles = offsets.new_zeros(len(offsets))
        # For TCP geometry all roll alternatives have identical geometric risk
        # and a nonnegative posture cost. Prune these dominated alternatives.
        if self.pose_mode!='fixed' and self.body_mode in ('hand','tool') and self.body_weight>0:
            candidates,query_rotation,offsets,angles = wrist_pose_candidates(candidates,rotation,offsets,fade,self.pose_mode=='wide_roll')
        points = torch.cat([p for p,_ in self.clouds],1)[0]
        valid = torch.cat([v for _,v in self.clouds],1)[0]&torch.isfinite(points).all(-1)
        radius = torch.cat(self.uncertainty_clouds,1)[0]
        memory = self.geometry_memory
        old = memory.last<max(0,self.step-45)
        points = torch.cat((points[valid],memory.points[old]))
        radius = torch.cat((radius[valid],self.map_uncertainty[old]+memory.scatter[old].sqrt()))
        padding = self.max_padding*padding_factor(radius)
        risk,tcp_risk,body_risk = embodied_surface_risk(candidates,query_rotation,tcp,points,padding,
            self.body_geometry,self.finger_positions,self.body_mode,self.body_weight,field=self.body_field,
            representation=self.body_representation)
        penalty = self.trust(trust_features(waypoints,tcp,self.current_goal,risk))
        cost = risk+penalty[:,None]*(offsets.norm(dim=-1)/.05).square()[None]
        max_roll = math.pi/2 if self.pose_mode=='wide_roll' else math.pi/6
        cost = cost+.015*(angles/max_roll).square()[None]
        chosen = cost.argmin(-1)
        if query_rotation.ndim==5:
            rotation.copy_(query_rotation[torch.arange(len(waypoints),device=waypoints.device),chosen])
        self.last_risk = float(risk[0,0])
        self.diagnostics.append(dict(step=self.step,risk=self.last_risk,choice=int(chosen[0]),
            correction_m=float(offsets[chosen[0]].norm()*fade[0]),points=len(points),
            trust_penalty=float(penalty[0]),selected_risk=float(risk[0,chosen[0]]),
            refinement='adaptive',mean_padding_m=float(padding.mean()) if len(padding) else 0.,
            max_padding_m=float(padding.max()) if len(padding) else 0.,
            old_only_points=int(old.sum()),route_feature_frames=len(self.history),
            body_mode=self.body_mode,body_weight=self.body_weight,
            body_field=self.body_field,
            body_representation=self.body_representation,
            pose_mode=self.pose_mode,roll_angle_deg=float(angles[chosen[0]]*fade[0]*180/math.pi),
            self_masked_points=int(self.geometry_memory.last_self_mask.sum()) if isinstance(self.geometry_memory,BodyFilteredGeometry) else 0,
            tcp_risk=float(tcp_risk[0,0]),body_risk=float(body_risk[0,0]),
            finger_positions_m=self.finger_positions.tolist()))
        return candidates[torch.arange(len(waypoints),device=waypoints.device),chosen]


def load_embodied_clearance(path,device='cpu',mode='tool',body_weight=.7,max_padding=.03,
                           mask_self=False,pose_mode='fixed',field='gaussian',representation='union'):
    original,maps,checkpoint = load_uncertain_clearance(path,device,'adaptive',max_padding)
    policy = EmbodiedClearancePolicy(original.backbone,original.head,original.kinematics,
        original.trust,original.uncertainty,mode,body_weight,max_padding,original.uniform_factor,mask_self,pose_mode,field,representation).to(device).eval()
    return policy,maps,checkpoint
