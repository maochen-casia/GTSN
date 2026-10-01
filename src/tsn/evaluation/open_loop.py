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


def predict_batch(model: GeometryPolicy, maps: GeometryMaps, batch: dict[str, Any]) -> torch.Tensor:
    """Prepare online maps and state in float32; run the network in the caller's precision."""
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
    for raw in loader:
        batch = device_batch(raw, device)
        prediction = predict_batch(model, maps, batch)
        metrics.update(prediction, batch["target"], batch["valid_future"], batch["route"])
    return {**metrics.result(), "privileged_action_map": True, "frames": len(loader.dataset)}
