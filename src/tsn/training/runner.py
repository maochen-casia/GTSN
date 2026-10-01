"""Train on 800 episodes, select best checkpoint on 100 validation episodes."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from tsn.common.checkpoint import save_checkpoint
from tsn.common.config import experiment_path, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.splits import episode_catalog, make_splits
from tsn.evaluation.metrics import PredictionMetrics
from tsn.evaluation.open_loop import device_batch, evaluate_predictions, predict_batch
from tsn.models.factory import make_maps, make_policy
from tsn.training.losses import imitation_loss


def train(benchmark: dict[str, Any], model_config: dict[str, Any], options: dict[str, Any],
          output: Path) -> Path:
    """Create a new run directory and return best.pt after the configured epoch schedule.

    Configs, split IDs, environment versions, logs and both best/latest checkpoints
    remain in the external experiment directory. Test episodes are never loaded.
    """
    output = experiment_path(output)
    if int(options["epochs"]) <= 0 or float(options["learning_rate"]) <= 0:
        raise ValueError("Training requires positive epochs and learning rate")
    if float(options["gradient_clip_norm"]) <= 0 or float(options["weight_decay"]) < 0:
        raise ValueError("Invalid gradient clip or weight decay")
    device = require_device(options["device"])
    seed_everything(int(options["seed"]))
    split = make_splits(benchmark)
    catalog = episode_catalog(Path(benchmark["root"]))
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("tsn.training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = logging.StreamHandler()
    file_log = logging.FileHandler(output / "training.log", encoding="utf-8")
    for handler in (stream, file_log):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    configuration = {"benchmark": benchmark, "model": model_config, "train": options}
    write_json(output / "config.json", configuration)
    write_json(output / "splits.json", split)
    write_json(output / "environment.json", {
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "numpy", "h5py", "sapien", "mani-skill", "trimesh", "pyrender", "PyOpenGL")},
        "device": str(device), "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    })
    datasets: dict[str, FrameDataset] = {}
    try:
        for partition in ("train", "validation"):
            datasets[partition] = FrameDataset(
                Path(benchmark["root"]), split[partition], catalog, int(model_config["chunk_size"]),
                int(options["frame_stride"]), int(options["max_open_files"]),
            )
        train_loader = make_loader(datasets["train"], options, True, int(options["seed"]))
        validation_loader = make_loader(datasets["validation"], options, False, int(options["seed"]) + 1)
        model = make_policy(model_config).to(device)
        maps = make_maps(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["learning_rate"]),
                                      weight_decay=float(options["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(options["epochs"]))
        amp = bool(options["mixed_precision"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best_rmse = math.inf
        best_epoch = 0
        logger.info("Run %s: %d training frames, %d validation frames; expert future action maps",
                    output.name, len(datasets["train"]), len(datasets["validation"]))
        for epoch in range(1, int(options["epochs"]) + 1):
            started = time.perf_counter()
            model.train()
            metrics = PredictionMetrics(model.chunk_size)
            total_loss, samples = 0.0, 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for step, raw in enumerate(train_loader, start=1):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    prediction = predict_batch(model, maps, batch)
                    loss = imitation_loss(
                        prediction, batch["target"], batch["valid_future"], float(options["huber_beta_rad"]),
                        float(options["horizon_decay_steps"]), float(options["horizon_weight_floor"]),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(options["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                size = len(batch["target"])
                total_loss += float(loss.detach()) * size
                samples += size
                metrics.update(prediction.detach(), batch["target"], batch["valid_future"], batch["route"])
                if step % 100 == 0:
                    logger.info("Epoch %d batch %d/%d loss %.6f", epoch, step, len(train_loader), total_loss / samples)
            validation = evaluate_predictions(model, maps, validation_loader, device)
            improved = validation["rmse_rad"] < best_rmse
            if improved:
                best_rmse, best_epoch = validation["rmse_rad"], epoch
            scheduler.step()
            record = {"epoch": epoch, "learning_rate": learning_rate, "train_loss": total_loss / samples,
                      "train": metrics.result(), "validation": validation,
                      "elapsed_seconds": time.perf_counter() - started, "best_epoch": best_epoch}
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            payload = {
                "format_version": 1, "epoch": epoch, "config": configuration, "splits": split,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "validation": validation, "best_rmse_rad": best_rmse, "best_epoch": best_epoch,
            }
            save_checkpoint(output / "latest.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            write_json(output / "summary.json", {
                "completed_epochs": epoch, "best_epoch": best_epoch, "best_validation_rmse_rad": best_rmse,
                "best_checkpoint": str(output / "best.pt"), "privileged_action_map": True,
            })
            logger.info("Epoch %d/%d train loss %.6f validation RMSE %.6f rad (best epoch %d)",
                        epoch, options["epochs"], total_loss / samples, validation["rmse_rad"], best_epoch)
        return output / "best.pt"
    finally:
        for dataset in datasets.values():
            dataset.close()
        for handler in (stream, file_log):
            logger.removeHandler(handler)
            handler.close()

