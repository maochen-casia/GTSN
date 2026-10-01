"""Fixed episode-level 800/100/100 stratification for the published manifest."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tsn.common.config import read_json

ROUTES = ("direct", "over", "side")
PARTITIONS = ("train", "validation", "test")
EXPECTED_COUNTS = {"direct": 200, "over": 400, "side": 400}
EXPECTED_PARTITIONS = {
    "train": {"direct": 160, "over": 320, "side": 320},
    "validation": {"direct": 20, "over": 40, "side": 40},
    "test": {"direct": 20, "over": 40, "side": 40},
}


def episode_catalog(root: Path) -> dict[str, str]:
    """Return episode ID -> route from manifest, checking all expected input files."""
    manifest = read_json(root / "manifest.json")
    result: dict[str, str] = {}
    for entry in manifest["episodes"]:
        episode, route = entry["episode"], entry["route"]
        if not re.fullmatch(r"episode_\d{3}", episode) or episode in result or route not in ROUTES:
            raise ValueError(f"Invalid or duplicate manifest entry: {entry}")
        folder = root / episode
        for filename in ("episode.h5", "scene.json"):
            if not (folder / filename).is_file():
                raise FileNotFoundError(folder / filename)
        if read_json(folder / "scene.json")["route_type"] != route:
            raise ValueError(f"Scene and manifest disagree on route for {episode}")
        result[episode] = route
    if len(result) != manifest["total_episodes"] or dict(Counter(result.values())) != EXPECTED_COUNTS:
        raise ValueError("tsn-1k requires 1000 episodes with direct/over/side counts 200/400/400")
    return result


def catalog_digest(catalog: dict[str, str]) -> str:
    """Identify episode membership and route labels independently of manifest ordering."""
    text = "\n".join(f"{key}:{catalog[key]}" for key in sorted(catalog))
    return hashlib.sha256(text.encode()).hexdigest()


def make_splits(config: dict[str, Any]) -> dict[str, Any]:
    """Shuffle sorted IDs within each route with a local seeded RNG; never split frames."""
    if config["route_counts"] != EXPECTED_COUNTS or config["partitions"] != EXPECTED_PARTITIONS:
        raise ValueError("The tsn-1k split sizes and 2:4:4 route ratio are fixed")
    catalog = episode_catalog(Path(config["root"]))
    rng = random.Random(int(config["seed"]))
    record: dict[str, Any] = {
        "seed": int(config["seed"]), "catalog_sha256": catalog_digest(catalog),
        "route_counts": EXPECTED_PARTITIONS,
        **{name: [] for name in PARTITIONS},
    }
    for route in ROUTES:
        ids = sorted(key for key, value in catalog.items() if value == route)
        rng.shuffle(ids)
        offset = 0
        for partition in PARTITIONS:
            size = EXPECTED_PARTITIONS[partition][route]
            record[partition].extend(ids[offset:offset + size])
            offset += size
    for partition in PARTITIONS:
        record[partition].sort()
    validate_splits(record, catalog)
    return record


def validate_splits(record: dict[str, Any], catalog: dict[str, str]) -> None:
    """Reject leakage, omissions, repeated IDs, changed labels, and incorrect route ratios."""
    if record["catalog_sha256"] != catalog_digest(catalog):
        raise ValueError("Benchmark manifest membership or routes differ from checkpoint")
    seen: set[str] = set()
    for partition in PARTITIONS:
        ids = record[partition]
        if len(ids) != len(set(ids)) or seen.intersection(ids) or set(ids) - catalog.keys():
            raise ValueError(f"Invalid or overlapping {partition} split")
        if dict(Counter(catalog[key] for key in ids)) != EXPECTED_PARTITIONS[partition]:
            raise ValueError(f"Wrong route proportions in {partition}")
        seen.update(ids)
    if seen != set(catalog):
        raise ValueError("Splits do not cover the whole benchmark")

