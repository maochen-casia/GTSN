"""Predeclared recovery checkpoint validation; test episodes never enter selection."""

from __future__ import annotations

import random
from pathlib import Path

from tsn.common.config import write_json
from tsn.data.splits import ROUTES
from tsn.evaluation.closed_loop import evaluate_rollouts


def validation_episodes(split, catalog, per_route, seed):
    rng = random.Random(seed)
    ids = []
    for route in ROUTES:
        candidates = sorted(episode for episode in split["validation"] if catalog[episode] == route)
        count = int(per_route[route])
        if count <= 0 or count > len(candidates):
            raise ValueError(f"Invalid validation episode count for {route}")
        ids.extend(rng.sample(candidates, count))
    return sorted(ids)


def selection_key(closed_loop, recovery, expert):
    """Success first, then fewer collisions, recovery prediction and expert prediction.

    Strict ties keep the parent. Expert RMSE alone cannot replace a safer model.
    """
    overall = closed_loop["overall"]
    return (overall["success_rate"], -overall["collision_rate"],
            -recovery["rmse_rad"], -expert["rmse_rad"])


def evaluate_candidate(model, maps, device, root, output: Path, ids, catalog,
                       options, expert, recovery):
    output.mkdir(parents=True, exist_ok=False)
    closed = evaluate_rollouts(ids, catalog, root, output, model, maps, device,
                              {**options, "render_videos": False})
    record = {"episodes": ids, "source_partition": "validation", "expert": expert,
              "recovery": recovery, "closed_loop": {k: v for k, v in closed.items() if k != "results"},
              "selection_key": list(selection_key(closed, recovery, expert))}
    write_json(output / "selection.json", record)
    return record
