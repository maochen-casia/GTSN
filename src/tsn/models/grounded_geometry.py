"""Visibility-consistent surface memory and metric-alignment interventions.

All rays are inferred from RGB-predicted surfaces and the measured camera pose.
They are geometric evidence, not measured free-space or safety certificates.
"""
import math

import torch
from torch.nn import functional as F

from tsn.models.clearance_policy import ClearancePolicy


def contradicted_points(history, current, current_valid, camera_pose, tolerance,
                        cone_degrees=3.):
    """Invalidate an old point only when a nearby current ray extends beyond it.

    Behind the current surface means occluded, and absence of an aligned ray
    means unobserved. Neither condition removes historical evidence.

    Args:
        history (torch.Tensor): Floating old base-frame points (B, P, 3), m.
        current (torch.Tensor): Floating new base-frame points (B, Q, 3), m;
            Q must be positive for the nearest-ray reduction.
        current_valid (torch.Tensor): Boolean validity of new points (B, Q).
        camera_pose (torch.Tensor): Floating current camera-to-base transforms
            (B, 4, 4). Camera +z is forward; translations are metres.
        tolerance (float): Required excess range of the new surface, metres.
        cone_degrees (float): Maximum angular separation of matched rays, deg.

    Returns:
        torch.Tensor: Boolean mask (B, P), True where a historical point is
        contradicted. Uses the most angularly aligned usable current ray and
        strict angle/range comparisons. Does not modify either point cloud or
        apply historical validity; the caller combines this with its old mask.
    """
    origin = camera_pose[:, None, :3, 3]
    old_ray, new_ray = history-origin, current-origin
    old_range, new_range = old_ray.norm(dim=-1), new_ray.norm(dim=-1)
    old_direction = old_ray / old_range[..., None].clamp_min(1e-6)
    new_direction = new_ray / new_range[..., None].clamp_min(1e-6)
    front = (new_ray*camera_pose[:, None, :3, 2]).sum(-1) > 0
    usable = current_valid & front & torch.isfinite(current).all(-1) & (new_range > .01)
    cosine = old_direction @ new_direction.transpose(-1, -2)
    cosine = cosine.masked_fill(~usable[:, None], -2.)
    agreement, nearest = cosine.max(-1)
    observed_range = new_range.gather(1, nearest)
    return ((agreement > math.cos(math.radians(cone_degrees))) &
            (old_range > .01) & (observed_range > old_range+tolerance))


class GroundedGeometryPolicy(ClearancePolicy):
    """Update the refiner's surface memory while retaining proposal features."""
    def __init__(self, *args, geometry_update='visibility', visibility_tolerance=.04,
                 translation_m=None, **kwargs):
        """Configure visibility pruning or a metric alignment intervention.

        Args:
            *args (tuple): ClearancePolicy positional arguments: backbone,
                head, kinematics, and optional clearance settings.
            geometry_update (str): 'visibility' prunes contradicted old points;
                'shift_pos'/'shift_neg' shift each new cloud by +/- .10 m in
                base x; 'calibrated' adds translation_m; 'pooled_mean' adds
                the frozen route head's predicted point-mean correction.
            visibility_tolerance (float): Positive range disagreement, metres.
            translation_m (list[float] | tuple[float, ...] | None): Three base
                XYZ offsets in metres, required only for 'calibrated'.
            **kwargs (dict[str, object]): Named ClearancePolicy arguments,
                including mode (str), margin (m), uncertainty, and penalty.

        Returns:
            None. Initializes inherited memory and records the update settings.

        Raises:
            ValueError: Update mode, tolerance, or calibration length is invalid.
        """
        if geometry_update not in ('visibility', 'shift_pos', 'shift_neg', 'calibrated', 'pooled_mean'):
            raise ValueError(geometry_update)
        if visibility_tolerance <= 0:
            raise ValueError('Visibility tolerance must be positive')
        self.geometry_update = geometry_update
        self.visibility_tolerance = visibility_tolerance
        if geometry_update == 'calibrated' and (translation_m is None or len(translation_m) != 3):
            raise ValueError('Calibrated geometry requires three metric translation coordinates')
        self.translation_m = translation_m
        super().__init__(*args, **kwargs)
        self.schedule += '_geometry_' + geometry_update

    def reset_episode(self):
        """Clear inherited memory and visibility-removal counters.

        Args:
            None.

        Returns:
            None. Sets removed_history_points and historical_valid_before to 0.
        """
        super().reset_episode()
        self.removed_history_points = self.historical_valid_before = 0

    def perceive(self, rgb, state, K, pose):
        """Append predicted geometry and apply the configured memory update.

        Args:
            rgb (torch.Tensor): Uint8 sensor RGB (1, H, W, 3), values 0..255.
            state (torch.Tensor): Floating normalized policy state (1, 16),
                following CompactPolicy's entry layout.
            K (torch.Tensor): Floating sensor intrinsics (1, 3, 3), pixels.
            pose (torch.Tensor): Floating current camera-to-base pose (1, 4, 4).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Unmodified floating
            backbone outputs: joint offsets (1, 30, 7), radians; Pi3 tokens
            (1, 16, 768); pooled maps (1, 16, 6), normalized XYZ / projected
            goal / visible goal / future-action scores.

        Notes:
            Modifies only stored refiner clouds (points (1, 400, 3) in metres,
            sigma (1, 400) in metres, validity (1, 400)). Calibration and pooled
            mean updates recompute workspace validity; shift interventions keep
            the original validity. Visibility updates only old validity masks,
            counting valid points before removal and points removed this call.
        """
        output = super().perceive(rgb, state, K, pose)
        self.removed_history_points = self.historical_valid_before = 0
        if self.geometry_update in ('calibrated', 'pooled_mean'):
            points, sigma, _ = self.clouds[-1]
            if self.geometry_update == 'calibrated':
                correction = points.new_tensor(self.translation_m)
            else:
                raw = self.head.distribution(self.head.visual(output[1].float())).float()
                correction = .25*raw[..., :3].tanh()
                correction = F.interpolate(correction.transpose(1, 2).reshape(1, 3, 4, 4),
                                           (20, 20), mode='nearest-exact').flatten(2).transpose(1, 2)
                correction = correction*points.new_tensor([.55, .55, .5])
            points = points+correction
            valid = ((points[..., 0] > .10) & (points[..., 0] < 1.05) &
                     (points[..., 1].abs() < .60) & (points[..., 2] > .04) & (points[..., 2] < .65))
            self.clouds[-1] = (points.detach(), sigma, valid.detach())
        elif self.geometry_update.startswith('shift_'):
            # Intervention on the refiner only. Keep the proposal's features,
            # geometry, calibration, and state exactly as originally predicted.
            points, sigma, valid = self.clouds[-1]
            shift = .10 if self.geometry_update == 'shift_pos' else -.10
            self.clouds[-1] = (points+points.new_tensor([shift, 0., 0.]), sigma, valid)
        elif len(self.clouds) > 1:
            current, _, current_valid = self.clouds[-1]
            updated = []
            for points, sigma, valid in self.clouds[:-1]:
                contradicted = contradicted_points(points, current, current_valid, pose,
                                                   self.visibility_tolerance)
                self.historical_valid_before += int(valid.sum())
                self.removed_history_points += int((valid & contradicted).sum())
                updated.append((points, sigma, valid & ~contradicted))
            self.clouds = updated + [self.clouds[-1]]
        return output

    def refine_waypoints(self, *args):
        """Refine the route and attach geometry-update counts to diagnostics.

        Args:
            *args (tuple[torch.Tensor, ...]): In order: floating base XYZ
                waypoints (1, 30, 3), m; floating TCP-to-base rotations
                (1, 30, 3, 3); floating current TCP pose (1, 4, 4); boolean
                near-goal mask (1,); floating camera-to-base pose (1, 4, 4).

        Returns:
            torch.Tensor: Selected floating base-frame waypoints (1, 30, 3),
            metres, from ClearancePolicy.refine_waypoints. Adds update mode
            and visibility-removal counters to the newest diagnostic record.
        """
        result = super().refine_waypoints(*args)
        self.diagnostics[-1].update(geometry_update=self.geometry_update,
                                   removed_history_points=self.removed_history_points,
                                   historical_valid_before=self.historical_valid_before)
        return result
