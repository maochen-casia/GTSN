"""Shared feature preparation and deterministic open-loop prediction evaluation."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from tsn.evaluation.metrics import PredictionMetrics
from tsn.features.maps import GeometryMaps
from tsn.features.state import policy_state
from tsn.models.geometry_policy import GeometryPolicy


def device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def predict_batch(model: GeometryPolicy, maps: GeometryMaps, batch: dict[str, Any],
                  return_maps: bool = False) -> torch.Tensor:
    """Prepare online maps and state in float32; run the network in the caller's precision."""
    if getattr(model, 'uses_predicted_maps', False):
        device_type = batch['qpos'].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            state = policy_state(batch['qpos'], batch['goal_pose'], maps.settings)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == 'cuda'):
            if return_maps:
                return model.forward_with_maps(batch['rgb'], state, batch['K'], batch['T_B_C'])
            return model(batch['rgb'], state, batch['K'], batch['T_B_C'])
    with torch.autocast(device_type=batch["depth"].device.type, enabled=False):
        geometry = maps(batch["depth"], batch["K"], batch["T_B_C"], batch["goal_pose"][:, :3],
                        batch["future_ee"], batch["valid_future"])
        state = policy_state(batch["qpos"], batch["goal_pose"], maps.settings)
    return model(geometry, state)


@torch.inference_mode()
def evaluate_predictions(model: GeometryPolicy, maps: GeometryMaps, loader: DataLoader,
                         device: torch.device) -> dict[str, Any]:
    """Evaluate all frames in the selected held-out split without optimizer updates."""
    model.eval()
    metrics = PredictionMetrics(model.chunk_size)
    learned_maps = getattr(model, 'uses_predicted_maps', False)
    map_squared = torch.zeros(6, device=device, dtype=torch.float64)
    map_pixels = 0
    for raw in loader:
        batch = device_batch(raw, device)
        if learned_maps:
            prediction, predicted_maps = predict_batch(model, maps, batch, return_maps=True)
            # Teacher labels are used only AFTER inference, for held-out head metrics.
            teacher = maps(batch['depth'], batch['K'], batch['T_B_C'], batch['goal_pose'][:, :3],
                           batch['future_ee'], batch['valid_future'])
            map_squared += (predicted_maps.float() - teacher).square().sum((0, 2, 3)).double()
            map_pixels += len(prediction) * teacher.shape[-2] * teacher.shape[-1]
        else:
            prediction = predict_batch(model, maps, batch)
        metrics.update(prediction, batch["target"], batch["valid_future"], batch["route"])
    result = {**metrics.result(), "privileged_action_map": not learned_maps, "frames": len(loader.dataset)}
    if learned_maps:
        channels = ['point_x', 'point_y', 'point_z', 'goal_projected', 'goal_visible', 'future_action']
        result['predicted_map_rmse'] = dict(zip(channels, (map_squared / map_pixels).sqrt().cpu().tolist()))
    return result
