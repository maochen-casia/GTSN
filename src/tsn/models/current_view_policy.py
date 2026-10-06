"""Retained correctively trained current-view route head; no spatial history."""
import torch
from torch import nn
from torch.nn import functional as F

from tsn.models.compact_policy import CompactRouteHead, goal_xyz


def metric_points(dense):
    """Sample actual predicted pixels, avoiding spatial XYZ averaging."""
    points = F.interpolate(dense[:, :3].float(), (20, 20), mode='nearest-exact')
    points = points.flatten(2).transpose(1, 2)
    return points * points.new_tensor([.55, .55, .50]) + points.new_tensor([.65, 0, .22])


def nearest_surface(samples, points, valid):
    """Nearest observed surface distances/vectors; empty maps return zeros.

    Leading dimensions are shared; the penultimate axes index queries/points.
    Geometry is supplied by the caller, never fetched from simulator state.
    """
    shape = samples.shape
    queries = samples.float().reshape(-1, shape[-2], 3)
    surfaces = points.float().reshape(-1, points.shape[-2], 3)
    mask = valid.reshape(-1, valid.shape[-1])
    distance = torch.cdist(queries, surfaces).masked_fill(~mask[:, None], 10.)
    nearest, index = distance.min(-1)
    vectors = surfaces.gather(1, index[..., None].expand(-1, -1, 3))-queries
    present = mask.any(-1)
    nearest = torch.where(present[:, None], nearest, torch.full_like(nearest, 10.))
    vectors = torch.where(present[:, None, None], vectors, torch.zeros_like(vectors))
    return nearest.reshape(*shape[:-1]), vectors.reshape(shape)


class CurrentViewHead(CompactRouteHead):
    """Current RGB geometry plus the original compact four-frame attention.

    Spatial slots are rebuilt for each observation. The only carried state is
    four compressed frame features with ages and validity. Parameter names are
    retained to load the archived current-view checkpoint without modifying it.
    """
    use_points = True
    hand_read = True

    def __init__(self, grid_shape=(12, 12, 6), point_width=32, corrective=True):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        if len(grid_shape) != 3 or any(n < 1 for n in grid_shape):
            raise ValueError('Expected three positive grid dimensions')
        self.point_width, self.corrective = point_width, corrective
        axes = [torch.linspace(lo+span/(2*n), lo+span-span/(2*n), n)
                for lo, span, n in zip((.10, -.60, .04), (.95, 1.20, .61), grid_shape)]
        self.register_buffer('anchors', torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3))
        ys, xs = torch.meshgrid(torch.arange(20), torch.arange(20), indexing='ij')
        self.register_buffer('pixel_cell', ((ys//5)*4+xs//5).flatten())
        self.point_encode = nn.Sequential(nn.Linear(64+3, point_width), nn.SiLU())
        self.point_distribution = nn.Linear(point_width, 4)
        self.local_value = nn.Sequential(nn.Linear(point_width+10, 64), nn.SiLU(), nn.LayerNorm(64))
        self.local_query = nn.Linear(256, 64)
        self.local_output = nn.Sequential(nn.Linear(295, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))
        if corrective:
            self.history_output = nn.Sequential(nn.Linear(590, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))
            nn.init.zeros_(self.history_output[-1].weight)
            nn.init.zeros_(self.history_output[-1].bias)

    def options(self):
        return dict(grid_shape=self.grid_shape, point_width=self.point_width, corrective=self.corrective)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get('architecture') == 'current_view':
            head = cls(**checkpoint['head_options'])
            head.load_state_dict(checkpoint['head'], strict=True)
            return head
        options = checkpoint['memory_options']
        required = ('use_points', 'spatial_read', 'balanced_writes', 'temporal_base', 'hand_read', 'always_reset_persistent')
        if not all(options.get(key) for key in required) or options.get('history_branch', 'none') not in ('none', 'current'):
            raise ValueError('Only archived current-view checkpoints are supported')
        state = checkpoint['head']
        if state['memory_residual.weight'].count_nonzero():
            raise ValueError('Archived head has a nonzero spatial-memory residual')
        head = cls(options['grid_shape'], options['point_width'], options.get('history_branch') == 'current')
        ignored = ('cell.', 'slot_query', 'decay.', 'scene_read.', 'memory_residual.', 'point_read.')
        active = {k: v for k, v in state.items() if not k.startswith(ignored)}
        head.load_state_dict(active, strict=True)
        return head

    def spatial_inputs(self, average, support, query, prediction, tcp, state):
        """Read one explicitly chosen surface store at each proposed route knot."""
        location = average[..., self.point_width:self.point_width+3]
        variance = (average[..., -3:] - location.square()).clamp_min(0)
        origin, goal = tcp[..., :3, 3], goal_xyz(state)
        local = self.local_value(torch.cat((average[..., :self.point_width],
            (location-origin[..., None, :])/.3, (location-goal[..., None, :])/.3,
            variance.clamp_min(1e-8).sqrt()/.1, support.log1p()), -1))
        q = self.local_query(query)
        semantic = (local*q[..., None, :]).sum(-1)/8
        proposals = origin[..., None, :] + prediction.detach()
        distances = (proposals[..., None, :] - location[..., None, :, :]).square().sum(-1)
        radii = distances.new_tensor([.06, .12, .24])
        logits = semantic[..., None, None, :] - distances[..., None, :, :]/(2*radii[:, None, None].square())
        mask = support[..., 0] > 0
        logits = logits.masked_fill(~mask[..., None, None, :], -1e4)
        weights = logits.softmax(-1)*mask[..., None, None, :]
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum('btskn,btnd->btksd', weights.to(local.dtype), local).flatten(-2)
        inputs = torch.cat((pooled, q[..., None, :].expand(*q.shape[:2], 6, 64),
                            prediction.detach()/.3, (goal[..., None, :]-proposals)/.3), -1)
        if self.hand_read:
            hand = proposals[..., None, :] - tcp[..., None, None, :3, 2]*proposals.new_tensor([0., .05, .10])[None, :, None]
            offsets = proposals.new_tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0],
                                            [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])*.06
            probes = (hand[..., None, :]+offsets).flatten(2, 4)
            distances, vectors = nearest_surface(probes, location, mask)
            distances = distances.reshape(*state.shape[:2], 6, 3, 7)
            vectors = vectors.reshape(*state.shape[:2], 6, 3, 7, 3)
            occupancy = torch.exp(-.5*(distances/.04).square()).flatten(-2)
            center = vectors[..., 0, :].clamp(-.2, .2).flatten(-2)/.2
            nearest = distances[..., 0].clamp(max=.2)/.2
            inputs = torch.cat((inputs, occupancy, center, nearest), -1)
        return inputs

    def temporal_context(self, cells, poses, times, valid, query, initial=None, resets=None):
        """Exact compact attention with four compressed frame preactivations.

        Each carried row holds 256 features, age, and validity. No raw Pi3
        history is needed. Keeping preactivations allows the original age input
        to be applied before SiLU/LayerNorm, preserving the pretrained head.
        """
        pre = self.frame[0](torch.cat((cells.flatten(2), poses.flatten(2).float(),
                                      torch.zeros_like(times[..., None])), -1)).float()
        batch, length = times.shape
        if initial is None:
            initial = pre.new_zeros(batch, 4, 258)
        all_pre = torch.cat((initial[..., :256], pre), 1)
        all_times = torch.cat((-initial[..., 256], times.float()), 1)
        all_valid = torch.cat((initial[..., 257].bool(), valid), 1)
        cursor = torch.arange(length, device=times.device)[None].minimum(valid.sum(1)[:, None]-1)
        indices = cursor[..., None] + torch.arange(1, 5, device=times.device)
        b = torch.arange(batch, device=times.device)[:, None, None]
        mask = all_valid[b, indices]
        if resets is not None:
            reset_at = torch.where(resets & valid, torch.arange(length, device=times.device)[None]+4, 0)
            boundary = reset_at.cummax(1).values
            mask = mask & (indices >= boundary[..., None])
        ages = times.gather(1, cursor)[..., None]-all_times[b, indices]
        encoded = all_pre[b, indices] + ages[..., None]/60*self.frame[0].weight[:, -1].float()
        frames = self.frame[2](self.frame[1](encoded))
        scores = (self.key(frames)*query[..., None, :]).sum(-1)/16
        context = (scores.masked_fill(~mask, -torch.inf).softmax(-1)[..., None]*frames).sum(2)
        carry = torch.cat((all_pre[b, indices][:, -1], ages[:, -1, :, None], mask[:, -1, :, None].float()), -1)
        return context, carry

    def point_address(self, points):
        """Regular-grid indexing, equivalent to nearest anchors inside bounds."""
        unit = (points.float().nan_to_num() - points.new_tensor([.10, -.60, .04])) / points.new_tensor([.95, 1.20, .61])
        index = (unit * points.new_tensor(self.grid_shape)).floor().long()
        index = torch.maximum(index, torch.zeros_like(index))
        index = torch.minimum(index, index.new_tensor(self.grid_shape)-1)
        return (index[..., 0]*self.grid_shape[1] + index[..., 1])*self.grid_shape[2] + index[..., 2]

    def encode(self, tokens, geometry, poses):
        visual = self.visual(tokens.float())
        distribution = self.distribution(visual).float()
        mean = geometry[..., :3].float() + .25 * distribution[..., :3].tanh()
        logvar = -5 + 4 * distribution[..., 3:].tanh()
        reliability = (-.5 * logvar.detach().mean(-1)).softmax(-1) * 16
        visual = visual * reliability[..., None]
        geom = self.geometry(torch.cat((mean, geometry[..., 3:].float()), -1))
        cells = torch.cat((visual, geom), -1)
        return visual, cells, mean, logvar

    def sequence(self, tokens, geometry, poses, times, valid, state, tcp,
                 points=None, initial=None, resets=None):
        if valid.dtype != torch.bool or not valid[:, 0].all():
            raise ValueError('Sequences must start with a valid observation')
        if points is None:
            raise ValueError('Current-view geometry requires RGB-predicted points')
        visual, cells, mean, logvar = self.encode(tokens, geometry, poses)
        finite = torch.isfinite(points).all(-1)
        safe_points = torch.where(finite[..., None], points.float(), torch.zeros_like(points).float())
        point_features = self.point_encode(torch.cat((visual[..., self.pixel_cell, :],
                                  (safe_points - points.new_tensor([.65, 0, .22])) / .55), -1))
        correction = self.point_distribution(point_features).float()
        corrected = safe_points + .05 * correction[..., :3].tanh()
        confidence = correction[..., 3].sigmoid()
        # Ignore out-of-workspace predictions and the immediate hand/camera.
        usable = finite & torch.isfinite(corrected).all(-1)
        usable = usable & (corrected[..., 0] > .10) & (corrected[..., 0] < 1.05)
        usable = usable & (corrected[..., 1].abs() < .60)
        usable = usable & (corrected[..., 2] > .04) & (corrected[..., 2] < .65)
        usable = usable & ((corrected - tcp[..., None, :3, 3]).norm(dim=-1) > .07)
        usable = usable & ((corrected - poses[..., None, :3, 3]).norm(dim=-1) > .04)
        usable = usable & valid[..., None]
        # Addressing uses frozen raw coordinates, never evolving centroids.
        address = self.point_address(safe_points)
        weight = confidence * usable.float()
        values = torch.cat((point_features.float(), corrected, corrected.square(),
                            torch.ones_like(weight[..., None])), -1)
        values = torch.where(usable[..., None], values, torch.zeros_like(values))
        point_write = values.new_zeros(*values.shape[:2], len(self.anchors), values.shape[-1])
        point_write = point_write.scatter_add(2, address[..., None].expand_as(values), values * weight[..., None])
        # One confidence-weighted vote per observed voxel per frame;
        # pixel density cannot masquerade as independent evidence.
        count = weight.new_zeros(*weight.shape[:2], len(self.anchors), 1)
        count = count.scatter_add(2, address[..., None], usable.float()[..., None])
        point_write = point_write / count.clamp_min(1)
        support = point_write[..., -1:]
        average = point_write[..., :-1]/support.clamp_min(1e-6)
        query = self.query(torch.cat((state.float(), tcp.flatten(2).float(),
                          (goal_xyz(state)-tcp[..., :3, 3])/.3), -1))
        context, carry = self.temporal_context(cells, poses, times, valid, query,
                                               None if initial is None else initial[0], resets)
        prediction = .3*self.output(torch.cat((query, context), -1)).reshape(*state.shape[:2], 6, 3).tanh()
        inputs = self.spatial_inputs(average, support, query, prediction, tcp, state)
        current = prediction + .08*self.local_output(inputs).tanh()
        present = support[..., 0].gt(0).any(-1)
        residual = (.04*self.history_output(torch.cat((inputs, inputs), -1)).tanh()*present[..., None, None]
                    if self.corrective else torch.zeros_like(current))
        return (current+residual).float(), dict(current_prediction=current, correction=residual,
                    mean=mean, logvar=logvar), (carry,)

    def stream(self, tokens, geometry, pose, elapsed, state, tcp, points=None, memory=None):
        """Single online update, with elapsed CONTROL STEPS since last update."""
        prediction, aux, memory = self.sequence(tokens[:, None], geometry[:, None], pose[:, None],
            elapsed[:, None], torch.ones(len(state), 1, dtype=torch.bool, device=state.device),
            state[:, None], tcp[:, None], None if points is None else points[:, None], memory)
        return prediction[:, 0], {k: v[:, 0] for k, v in aux.items()}, memory
