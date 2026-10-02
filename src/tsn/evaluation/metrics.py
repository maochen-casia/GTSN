"""Masked joint prediction statistics and episode-weighted rollout summaries."""

from __future__ import annotations

import math
from typing import Any

import torch

from tsn.data.splits import ROUTES


class PredictionMetrics:
    """Accumulate errors for supervised future states, including terminal holds."""

    def __init__(self, horizon: int) -> None:
        self.total = torch.zeros(4, dtype=torch.float64)
        self.horizon = torch.zeros((horizon, 2), dtype=torch.float64)
        self.routes = torch.zeros((len(ROUTES), 2), dtype=torch.float64)
        self.endpoint = torch.zeros(2, dtype=torch.float64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               route: torch.Tensor) -> None:
        squared = (prediction.float() - target).square().sum(-1)
        absolute = (prediction.float() - target).abs().sum(-1)
        count = valid * target.shape[-1]
        self.total += torch.stack(((squared * valid).sum(), (absolute * valid).sum(),
                                  (target.square().sum(-1) * valid).sum(), count.sum())).double().cpu()
        self.horizon += torch.stack(((squared * valid).sum(0), count.sum(0)), -1).double().cpu()
        for index in range(len(ROUTES)):
            mask = valid & (route == index)[:, None]
            self.routes[index] += torch.stack(((squared * mask).sum(),
                                               mask.sum() * target.shape[-1])).double().cpu()
        self.endpoint += torch.stack(((squared[:, -1] * valid[:, -1]).sum(),
                                       count[:, -1].sum())).double().cpu()

    def result(self) -> dict[str, Any]:
        if self.total[3] <= 0:
            raise ValueError("Cannot report metrics for an empty evaluation")

        def rmse(pair: torch.Tensor) -> float | None:
            return math.sqrt(float(pair[0] / pair[1])) if pair[1] > 0 else None

        return {
            "rmse_rad": rmse(self.total[[0, 3]]),
            "mae_rad": float(self.total[1] / self.total[3]),
            "zero_action_rmse_rad": rmse(self.total[[2, 3]]),
            "endpoint_rmse_rad": rmse(self.endpoint),
            "rmse_by_horizon_rad": [rmse(pair) for pair in self.horizon],
            "rmse_by_route_rad": {route: rmse(self.routes[i]) for i, route in enumerate(ROUTES)},
            "valid_joint_elements": int(self.total[3]),
        }


def summarize_rollouts(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute success/collision rates per episode, both overall and by route."""
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "episodes": count,
            "success_rate": sum(item["success"] for item in items) / count if count else None,
            "collision_rate": sum(item["collision"] is not None for item in items) / count if count else None,
            "xyz_orientation_success_rate": sum(item.get("xyz_orientation_success", item["success"]) for item in items) / count if count else None,
            "mean_final_position_error_m": sum(item["final_position_error_m"] for item in items) / count if count else None,
            "mean_control_steps": sum(item["control_steps"] for item in items) / count if count else None,
        }
    return {"overall": summarize(results), "by_route": {
        route: summarize([item for item in results if item["route"] == route]) for route in ROUTES
    }, "privileged_action_map": True}
