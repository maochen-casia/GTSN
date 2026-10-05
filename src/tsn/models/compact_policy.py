"""Single-memory route head and fixed-horizon compact controller.

Shape notation: B is batch size, T is observed history length (at most four
in streaming control), and the 16 spatial cells form a pooled 4 x 4 grid.
All metric positions and poses use the robot base frame. A camera pose maps
OpenCV camera coordinates into that frame; TCP denotes the hand tool centre.
The 16 state entries are seven arm angles / pi, two finger positions / .04 m,
three normalized goal coordinates, and a unit goal quaternion in wxyz order.
Point/goal XYZ uses centre (.65, 0, .22) m and scale (.55, .55, .50) m.
"""
import math

import torch
from torch import nn

from tsn.common.checkpoint import load_checkpoint
from tsn.features.maps import GeometryMaps
from tsn.models.kinematics import PandaKinematics
from tsn.models.pi3_policy import Pi3MapPolicy


def goal_xyz(state):
    """Decode the normalized goal position into robot-base metres.

    Args:
        state (torch.Tensor): Floating policy state (..., 16), with normalized
            goal XYZ in entries 9:12. Leading dimensions are preserved.

    Returns:
        torch.Tensor: Floating goal XYZ (..., 3) in metres (float32 for the
        usual float32 state). This decodes the stored state, including any
        clipping applied during state construction.
    """
    return state[..., 9:12].float() * state.new_tensor([.55, .55, .5]) + state.new_tensor([.65, 0, .22])


class CompactRouteHead(nn.Module):
    """Attend to four observed frames with learned geometry reliability."""
    def __init__(self):
        """Create the spatial encoders, temporal attention, and six-knot head.

        Args:
            None.

        Returns:
            None. Initializes trainable parameters; each frame combines 16
            cells of 64 visual + 32 geometry features, a pose, and an age.
        """
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.SiLU())
        self.distribution = nn.Linear(64, 6)
        self.geometry = nn.Sequential(nn.Linear(6, 32), nn.SiLU())
        self.frame = nn.Sequential(nn.Linear(16*96+16+1, 256), nn.SiLU(), nn.LayerNorm(256))
        self.query = nn.Sequential(nn.Linear(16+16+3, 256), nn.SiLU(), nn.LayerNorm(256))
        self.key = nn.Linear(256, 256)
        self.output = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 18))
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        """Predict six TCP offsets by attending to valid historical frames.

        Args:
            tokens (torch.Tensor): Floating Pi3 features (B, T, 16, 768).
            geometry (torch.Tensor): Floating pooled maps (B, T, 16, 6), in
                normalized base XYZ, projected goal, visible goal, future-action
                score order. The final three channels are predicted scores.
            poses (torch.Tensor): Floating camera-to-base transforms
                (B, T, 4, 4), with translations in metres.
            ages (torch.Tensor): Numeric frame ages (B, T) in control steps;
                zero denotes the current frame. Internally divided by 60.
            mask (torch.Tensor): Boolean frame validity (B, T). Each sample
                must contain at least one True entry for finite attention.
            state (torch.Tensor): Floating normalized policy state (B, 16);
                see the module docstring for entry meanings.
            tcp (torch.Tensor): Floating current TCP-to-base poses (B, 4, 4).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Float32 offsets
            (B, 6, 3) in base-frame metres, bounded per coordinate by .3 m,
            at steps 5, 10, ..., 30 relative to the SAME current TCP. Auxiliary
            'mean' and 'logvar' are float32 (B, 16, 3) for the last frame:
            corrected normalized point XYZ and log diagonal variance in
            normalized coordinate units squared. The last frame is assumed
            current; auxiliary outputs are not selected using the mask.
        """
        visual = self.visual(tokens.float())
        distribution = self.distribution(visual).float()
        mean = geometry[..., :3].float() + .25 * distribution[..., :3].tanh()
        logvar = -5 + 4 * distribution[..., 3:].tanh()
        # Per-frame cell weights sum to 16, preserving average feature scale.
        # Detaching variance prevents action loss from tuning these weights.
        reliability = (-.5 * logvar.detach().mean(-1)).softmax(-1) * 16
        visual = visual * reliability[..., None]
        geom = self.geometry(torch.cat((mean, geometry[..., 3:].float()), -1))
        frames = self.frame(torch.cat((torch.cat((visual, geom), -1).flatten(2),
                                      poses.flatten(2).float(), ages[..., None].float() / 60), -1))
        query = self.query(torch.cat((state.float(), tcp.flatten(1).float(),
                                     (goal_xyz(state) - tcp[:, :3, 3]) / .3), -1))
        scores = (self.key(frames) * query[:, None]).sum(-1) / 16
        # sqrt(256)=16 scales dot-product attention over the time dimension.
        context = (scores.masked_fill(~mask, -torch.inf).softmax(-1)[..., None] * frames).sum(1)
        out = .3 * self.output(torch.cat((query, context), -1)).reshape(-1, 6, 3).tanh()
        return out.float(), {'mean': mean[:, -1], 'logvar': logvar[:, -1]}


class CompactPolicy(nn.Module):
    """Predict 30 joint targets; execute 15 before observing the next RGB frame."""
    uses_predicted_maps = True
    chunk_size = 30
    history_length = 4
    execute = 15
    servo_radius = .08
    schedule = 'fixed15_single_memory'

    def __init__(self, backbone, head, kinematics):
        """Assemble perception, route prediction, and robot kinematics.

        Args:
            backbone (Pi3MapPolicy): RGB perception and 30-step joint proposal.
            head (CompactRouteHead | torch.nn.Module): Compatible memory head
                returning six Cartesian offsets and auxiliary predictions.
            kinematics (PandaKinematics): Forward and inverse Panda kinematics.

        Returns:
            None. Registers the modules and initializes empty episode memory.
        """
        super().__init__()
        self.backbone, self.head, self.kinematics = backbone, head, kinematics
        self.reset_episode()

    def reset_episode(self):
        """Clear observation history and reset the control-step counter.

        Args:
            None.

        Returns:
            None. Call before a new episode to prevent cross-episode memory.
        """
        self.history, self.step = [], 0

    def observe_step(self, step):
        """Set the timestamp used when recording and ageing observations.

        Args:
            step (int): Current episode control-step index, normally increasing.

        Returns:
            None. Updates ``self.step``; does not itself record an observation.
        """
        self.step = step

    def execution_horizon(self, default):
        """Return how many predicted targets to execute before observing again.

        Args:
            default (int): Caller-provided fallback horizon; unused here.

        Returns:
            int: Fixed execution horizon, 15 control steps.
        """
        return self.execute

    def perceive(self, rgb, state, K, pose):
        """Extract the action proposal and spatial features for one observation.

        Args:
            rgb (torch.Tensor): Uint8 sensor images (B, H, W, 3), values 0..255.
            state (torch.Tensor): Floating normalized policy state (B, 16).
            K (torch.Tensor): Floating intrinsics (B, 3, 3) in sensor pixels.
            pose (torch.Tensor): Floating camera-to-base transforms (B, 4, 4).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Floating joint
            offsets (B, 30, 7) in radians from current joints, Pi3 tokens
            (B, 16, 768), and pooled six-channel maps (B, 16, 6). See
            CompactRouteHead.forward for channel meanings. CUDA perception
            uses autocast, so these tensors need not all be float32.
        """
        return self.backbone(rgb, state, K, pose, return_features=True)

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        """Provide the Cartesian refinement hook used by geometry extensions.

        Args:
            waypoints (torch.Tensor): Floating base XYZ targets (B, 30, 3), m.
            rotation (torch.Tensor): Floating target TCP-to-base rotations
                (B, 30, 3, 3), one per horizon step.
            tcp (torch.Tensor): Floating current TCP-to-base poses (B, 4, 4).
            near (torch.Tensor): Boolean terminal-servo mask (B,).
            pose (torch.Tensor): Floating camera-to-base poses (B, 4, 4).

        Returns:
            torch.Tensor: The original ``waypoints`` object (B, 30, 3).
            The base implementation leaves all targets unchanged.
        """
        return waypoints

    def forward(self, rgb, state, K, pose):
        """Record an observation and produce an executable joint-target chunk.

        Args:
            rgb (torch.Tensor): Uint8 sensor images (1, H, W, 3), values 0..255.
            state (torch.Tensor): Floating normalized policy state (1, 16).
            K (torch.Tensor): Floating intrinsics (1, 3, 3) in sensor pixels.
            pose (torch.Tensor): Floating camera-to-base transform (1, 4, 4),
                with translation in metres. Inputs and modules share a device.

        Returns:
            torch.Tensor: Float32 joint offsets (1, 30, 7), radians relative
            to the SAME measured arm configuration, not successive increments.
            Add the current seven joint angles to obtain absolute targets.

        Notes:
            Streaming memory is for one episode (B=1); the head itself supports
            batches. Appends detached features/pose and keeps four observations.
            Six route knots are interpolated into 30 positions. Within .08 m
            of the goal, a straight-line terminal servo replaces the route and
            holds current wrist orientation. Otherwise the backbone supplies
            orientation and IK seeds. Subclasses may refine positions before IK.
        """
        base, tokens, geom = self.perceive(rgb, state, K, pose)
        self.history.append((tokens.detach(), geom.detach(), pose.detach(), self.step))
        self.history = self.history[-self.history_length:]
        q = state[:, :7].float() * math.pi
        with torch.autocast(device_type=q.device.type, enabled=False):
            tcp = self.kinematics(q)
        goal = goal_xyz(state)
        near = (goal - tcp[:, :3, 3]).norm(dim=-1) < self.servo_radius
        observations = [torch.stack([frame[i] for frame in self.history], 1) for i in range(3)]
        ages = q.new_tensor([[self.step - frame[3] for frame in self.history]])
        delta, _ = self.head(*observations, ages, torch.ones_like(ages, dtype=torch.bool), state, tcp)
        # Interpolate route knots at t=0,5,...,30 into a target for every control step.
        knots = torch.cat((torch.zeros_like(delta[:, :1]), delta), 1)
        times = torch.arange(1, 31, device=q.device) / 5
        lo = times.long().clamp(max=5)
        alpha = (times - lo)[None, :, None]
        displacement = knots[:, lo] * (1-alpha) + knots[:, lo+1] * alpha
        waypoints = tcp[:, None, :3, 3] + displacement
        direct = tcp[:, None, :3, 3] + (goal-tcp[:, :3, 3])[:, None] * torch.linspace(
            1/30, 1, 30, device=q.device)[None, :, None]
        waypoints = torch.where(near[:, None, None], direct, waypoints)
        with torch.autocast(device_type=q.device.type, enabled=False):
            initial = q[:, None].expand(-1, 30, -1)
            rotation = tcp[:, None, :3, :3].expand(-1, 30, -1, -1)
            candidate = q[:, None] + base.float()
            candidate = candidate.maximum(self.kinematics.limits[:, 0]).minimum(self.kinematics.limits[:, 1])
            predicted_rotation = self.kinematics(candidate)[..., :3, :3]
            initial = torch.where(near[:, None, None], initial, candidate)
            rotation = torch.where(near[:, None, None, None], rotation, predicted_rotation)
            waypoints = self.refine_waypoints(waypoints, rotation, tcp, near, pose)
            targets = self.kinematics.inverse(initial.reshape(-1, 7), waypoints.reshape(-1, 3),
                                              rotation.reshape(-1, 3, 3))
        chunk = targets.reshape(-1, 30, 7) - q[:, None]
        # Preserve the exported controller's floating-point round trip.
        return (chunk + q[:, None]) - q[:, None]


def compact_state(checkpoint, *, allow_training_source=False):
    """Extract backbone and single-memory head weights from a checkpoint.

    Args:
        checkpoint (dict[str, object]): Loaded checkpoint containing compact
            'backbone'/'head' state dicts, or a training 'model' state dict.
        allow_training_source (bool): Allow extracting 'backbone.' and
            'heads.memory.' prefixes from an earlier training checkpoint.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: Backbone and
        route-head state dicts, with wrapper prefixes removed where needed.
        Tensor shapes are the corresponding module parameter/buffer shapes.

    Raises:
        ValueError: The checkpoint is not a supported single-memory export or
            contains extra heads/non-unit ensemble weights.
    """
    if checkpoint.get('architecture') == 'compact_consensus' and checkpoint.get('variant') == 'single_memory':
        state = checkpoint['head']
        if 'weights' in state:
            if state['weights'].numel() != 1 or state['weights'].item() != 1:
                raise ValueError('Expected a single unit-weight memory head')
            if any(not key.startswith('heads.0.') for key in state if key != 'weights'):
                raise ValueError('Unexpected extra route heads')
            state = {key.removeprefix('heads.0.'): value for key, value in state.items() if key != 'weights'}
        return checkpoint['backbone'], state
    if allow_training_source and any(key.startswith('heads.memory.') for key in checkpoint.get('model', {})):
        state = checkpoint['model']
        return ({key.removeprefix('backbone.'): value for key, value in state.items() if key.startswith('backbone.')},
                {key.removeprefix('heads.memory.'): value for key, value in state.items() if key.startswith('heads.memory.')})
    raise ValueError('Expected a single_memory compact checkpoint')


def load_compact_policy(path, device='cpu', *, allow_training_source=False):
    """Load a compact controller, map-supervision module, and saved metadata.

    Args:
        path (str | pathlib.Path): Path to a serialized project checkpoint.
        device (str | torch.device): Device for the policy and map module.
        allow_training_source (bool): Permit memory-branch extraction for
            initialization from an earlier training checkpoint.

    Returns:
        tuple[CompactPolicy, GeometryMaps, dict[str, object]]: Loaded policy
        in evaluation mode, depth-based supervision map generator, and raw
        checkpoint metadata. Evidence exports recursively load their base
        checkpoint and replace its head with EvidenceRouteHead; metadata is
        merged with the evidence checkpoint taking precedence.

    Raises:
        ValueError: Architecture, horizon, map source, or channel order is
            incompatible. State dict loading is strict for all loaded modules.
    """
    checkpoint = load_checkpoint(path)
    if checkpoint.get('architecture') == 'route_evidence':
        from tsn.models.evidence_policy import EvidenceRouteHead
        policy, maps, source = load_compact_policy(checkpoint['base_checkpoint'], device)
        head = EvidenceRouteHead(policy.head, **checkpoint['head_config'])
        head.load_state_dict(checkpoint['head'], strict=True)
        policy.head = head.to(device).eval()
        policy.schedule = 'fixed15_route_evidence_' + head.variant
        return policy, maps, {**source, **checkpoint}
    backbone_state, head_state = compact_state(checkpoint, allow_training_source=allow_training_source)
    config = checkpoint['config']['model']
    if config['name'] != 'pi3_map_policy' or config['action_map_source'] != 'predicted' or config['chunk_size'] != 30:
        raise ValueError('Compact control requires the 30-step learned Pi3 backbone')
    channels = ['point_x', 'point_y', 'point_z', 'goal_projected', 'goal_visible', 'future_action']
    if config['map_channels'] != channels:
        raise ValueError('Unexpected Pi3 geometry channel order')
    backbone = Pi3MapPolicy(config, initialize_backbone=False)
    backbone.load_state_dict(backbone_state, strict=True)
    head = CompactRouteHead()
    head.load_state_dict(head_state, strict=True)
    policy = CompactPolicy(backbone, head, PandaKinematics()).to(device).eval()
    return policy, GeometryMaps(config['maps']).to(device), checkpoint
