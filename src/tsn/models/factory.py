"""Construct the same policy and feature generator in training and evaluation."""

import inspect
from typing import Any

from tsn.features.maps import GeometryMaps
from tsn.models.geometry_policy import GeometryPolicy


def make_policy(config: dict[str, Any], initialize_backbone: bool = True) -> GeometryPolicy:
    """Build the baseline from config or checkpoint; reject unsupported map semantics."""
    channels = ["point_x", "point_y", "point_z", "goal_projected", "goal_visible", "future_action"]
    if config['name'] == 'pi3_map_policy':
        if config['action_map_source'] != 'predicted' or config['map_channels'] != channels:
            raise ValueError('Pi3 policy requires learned maps in the standard channel order')
        from tsn.models.pi3_policy import Pi3MapPolicy
        return Pi3MapPolicy(config, initialize_backbone=initialize_backbone)
    if (config["name"] != "geometry_policy" or config["action_map_source"] != "expert_future"
            or config["map_channels"] != channels):
        raise ValueError("Unsupported model name, map order, or action map source")
    parameters = inspect.signature(GeometryPolicy).parameters
    return GeometryPolicy(**{key: config[key] for key in parameters})


def make_maps(config: dict[str, Any]) -> GeometryMaps:
    return GeometryMaps(config["maps"])
