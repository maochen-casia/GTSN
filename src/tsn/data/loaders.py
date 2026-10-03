"""DataLoaders with reproducible shuffling and no train/validation frame overlap."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from tsn.common.seed import seed_worker
from tsn.data.hdf5_dataset import FrameDataset


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
