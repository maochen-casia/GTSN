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
    def __init__(self, *args, geometry_update='visibility', visibility_tolerance=.04,
                 translation_m=None, **kwargs):
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
        super().reset_episode()
        self.removed_history_points = self.historical_valid_before = 0

    def perceive(self, rgb, state, K, pose):
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
        result = super().refine_waypoints(*args)
        self.diagnostics[-1].update(geometry_update=self.geometry_update,
                                   removed_history_points=self.removed_history_points,
                                   historical_valid_before=self.historical_valid_before)
        return result
