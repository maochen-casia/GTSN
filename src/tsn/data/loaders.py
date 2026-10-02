"""DataLoaders with reproducible shuffling and no train/validation frame overlap."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from tsn.common.seed import seed_worker
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.splits import ROUTES


def make_loader(dataset: FrameDataset, options: dict[str, Any], training: bool,
                seed: int) -> DataLoader:
    """Shuffle only training frames; preserve every validation/test frame, including tail batches."""
    workers = int(options["num_workers"])
    if workers < 0 or int(options["batch_size"]) <= 0:
        raise ValueError("Invalid DataLoader batch size or worker count")
    return DataLoader(
        dataset, batch_size=int(options["batch_size"]), shuffle=training,
        num_workers=workers, pin_memory=options["device"].startswith("cuda"),
        persistent_workers=workers > 0, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )


def make_recovery_loader(expert: FrameDataset, recoveries: list[Any], options: dict[str, Any],
                         seed: int, recovery_fractions: list[float],
                         route_fractions: dict[str, float]) -> DataLoader:
    """Mix expert and multiple recovery sources at fixed source and route ratios."""
    if not recoveries or len(recoveries) != len(recovery_fractions):
        raise ValueError("Each recovery dataset must have a sampling fraction")
    if any(not 0.0 < fraction < 1.0 for fraction in recovery_fractions):
        raise ValueError("Each recovery sampling fraction must be between zero and one")
    if sum(recovery_fractions) >= 1.0:
        raise ValueError("Recovery sampling fractions must sum to less than one")
    route_probabilities = np.asarray([route_fractions[name] for name in ROUTES], dtype=np.float64)
    if np.any(route_probabilities <= 0) or not np.isclose(route_probabilities.sum(), 1.0):
        raise ValueError("route_fractions must be positive and sum to one")

    route_counts = np.zeros((1 + len(recoveries), len(ROUTES)), dtype=np.int64)
    expert_routes: list[int] = []
    for episode, count in zip(expert.ids, expert.counts):
        route = ROUTES.index(expert.catalog[episode])
        frame_count = len(range(0, count - 1, expert.stride))
        expert_routes.extend([route] * frame_count)
        route_counts[0, route] += frame_count
    source_routes = [expert_routes]
    for source_index, recovery in enumerate(recoveries, start=1):
        recovery_routes = list(recovery.route_indices)
        source_routes.append(recovery_routes)
        for route in recovery_routes:
            route_counts[source_index, route] += 1
    if any(not routes for routes in source_routes) or np.any(route_counts == 0):
        raise ValueError(
            "Every expert/recovery source needs samples in every route class: "
            f"{route_counts.tolist()}"
        )

    source_fractions = [1.0 - sum(recovery_fractions), *recovery_fractions]
    weights = np.empty(len(expert) + sum(len(value) for value in recoveries), dtype=np.float64)
    offset = 0
    for source, routes in enumerate(source_routes):
        counts = route_counts[source]
        weights[offset:offset + len(routes)] = [
            source_fractions[source] * route_probabilities[route] / counts[route]
            for route in routes
        ]
        offset += len(routes)

    dataset = ConcatDataset((expert, *recoveries))
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(expert), replacement=True, generator=generator,
    )
    workers = int(options["num_workers"])
    return DataLoader(
        dataset, batch_size=int(options["batch_size"]), sampler=sampler,
        num_workers=workers, pin_memory=options["device"].startswith("cuda"),
        persistent_workers=workers > 0, worker_init_fn=seed_worker,
        generator=generator, drop_last=False,
    )
