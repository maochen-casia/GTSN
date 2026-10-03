"""Masked joint prediction statistics and episode-weighted rollout summaries."""

from __future__ import annotations

from typing import Any


from tsn.data.splits import ROUTES


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
    }, "privileged_action_map": any(item.get('privileged_action_map', True) for item in results)}
