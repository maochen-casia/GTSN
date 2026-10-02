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

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import experiment_path, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader, make_recovery_loader
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import episode_catalog, make_splits
from tsn.evaluation.metrics import PredictionMetrics
from tsn.evaluation.open_loop import device_batch, evaluate_predictions, predict_batch
from tsn.models.factory import make_maps, make_policy
from tsn.training.losses import imitation_loss, predicted_map_loss
from tsn.features.state import policy_state


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
    learned_maps = model_config['name'] == 'pi3_map_policy'
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
            "torch", "numpy", "h5py", "sapien", "mani-skill", "trimesh", "pyrender", "PyOpenGL",
            *(['huggingface-hub', 'safetensors'] if learned_maps else []))},
        "device": str(device), "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    })
    datasets: dict[str, FrameDataset] = {}
    recovery_datasets: list[RecoveryDataset] = []
    try:
        for partition in ("train", "validation"):
            datasets[partition] = FrameDataset(
                Path(benchmark["root"]), split[partition], catalog, int(model_config["chunk_size"]),
                int(options["frame_stride"]), int(options["max_open_files"]),
                include_rgb=learned_maps,
            )
        recovery_sources = options.get("recovery_sources")
        if recovery_sources is None:
            recovery_dir = options.get("recovery_data_dir")
            recovery_sources = ([{
                "path": recovery_dir,
                "sampling_fraction": options.get("recovery_sampling_fraction"),
            }] if recovery_dir else [])
        recovery_fractions: list[float] = []
        resolved_sources: list[dict[str, Any]] = []
        seen_recovery_dirs: set[Path] = set()
        for index, source in enumerate(recovery_sources):
            if not isinstance(source, dict) or not source.get("path"):
                raise ValueError(f"Recovery source {index} must define a data path")
            if source.get("sampling_fraction") is None:
                raise ValueError(f"Recovery source {index} must define sampling_fraction")
            recovery_root = Path(source["path"]).resolve()
            if recovery_root in seen_recovery_dirs:
                raise ValueError(f"Recovery source is listed more than once: {recovery_root}")
            seen_recovery_dirs.add(recovery_root)
            recovery = RecoveryDataset(recovery_root, int(model_config["chunk_size"]), include_rgb=learned_maps)
            recovery_datasets.append(recovery)
            manifest_path = recovery_root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing recovery manifest: {manifest_path}")
            recovery_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if recovery_manifest.get("source_partition") != "train":
                raise ValueError("Recovery data must be generated from the training partition")
            recovery_ids = set(recovery.route_by_episode)
            manifest_ids = set(recovery_manifest.get("episode_ids", []))
            if recovery_manifest.get("source_dataset") != str(Path(benchmark["root"])):
                raise ValueError("Recovery data was generated from a different benchmark root")
            if not manifest_ids or manifest_ids - set(split["train"]):
                raise ValueError("Recovery data must come from episodes in the current training split")
            if recovery_ids != manifest_ids:
                raise ValueError("Recovery archives must match the episode IDs in their manifest")
            recovery_fractions.append(float(source["sampling_fraction"]))
            resolved_sources.append({
                **source,
                "path": str(recovery_root),
                "samples": len(recovery),
                "episodes": len(recovery_ids),
            })
        if recovery_datasets:
            train_loader = make_recovery_loader(
                datasets["train"], recovery_datasets, options, int(options["seed"]),
                recovery_fractions,
                options.get("route_sampling_fractions", {
                    route: int(count) / sum(benchmark["route_counts"].values())
                    for route, count in benchmark["route_counts"].items()}),
            )
        else:
            train_loader = make_loader(datasets["train"], options, True, int(options["seed"]))
        validation_loader = make_loader(datasets["validation"], options, False, int(options["seed"]) + 1)
        model = make_policy(model_config).to(device)
        if learned_maps:
            if options.get('initialize_from_checkpoint'):
                raise ValueError('The fresh Pi3 experiment must not initialize from a policy checkpoint')
            if not all(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError('Pi3 experiment requires full fine-tuning')
            write_json(output / 'initialization.json', {
                **model.initialization, 'all_parameters_trainable': True,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'backbone_parameters': sum(p.numel() for p in model.encoder.parameters()),
                'policy_checkpoint': None,
            })
        initialize_from = options.get("initialize_from_checkpoint")
        if initialize_from:
            initialization = load_checkpoint(Path(initialize_from))
            original_model = initialization["config"]["model"]
            architectural_config = {k: v for k, v in model_config.items() if k != "maps"}
            if {k: v for k, v in original_model.items() if k != "maps"} != architectural_config:
                raise ValueError("Initialization checkpoint uses a different model configuration")
            if original_model.get("maps") != model_config.get("maps"):
                logger.info("Initialization map settings changed explicitly: %s -> %s",
                            original_model.get("maps"), model_config.get("maps"))
            if initialization["splits"] != split:
                raise ValueError("Initialization checkpoint uses a different data split")
            model.load_state_dict(initialization["model"], strict=True)
        else:
            initialize_from = None
        maps = make_maps(model_config).to(device)
        training_model = model
        if learned_maps and len(options.get('gpu_ids', [])) > 1:
            if device.type != 'cuda' or options['gpu_ids'][0] != (device.index or 0):
                raise ValueError('DataParallel primary GPU must match the policy device')
            training_model = torch.nn.DataParallel(model, device_ids=options['gpu_ids'])
        parameters = model.parameters()
        if learned_maps:
            encoder_ids = {id(p) for p in model.encoder.parameters()}
            parameters = [
                {'params': list(model.encoder.parameters()), 'lr': float(options['backbone_learning_rate']),
                 'name': 'pi3_image_encoder'},
                {'params': [p for p in model.parameters() if id(p) not in encoder_ids],
                 'lr': float(options['learning_rate']), 'name': 'decoder_and_heads'},
            ]
        optimizer = torch.optim.AdamW(parameters, lr=float(options["learning_rate"]),
                                      weight_decay=float(options["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(options["epochs"]))
        amp = bool(options["mixed_precision"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp and not learned_maps)
        best_rmse = math.inf
        best_epoch = 0
        selection = options.get("checkpoint_selection", {"method": "expert_rmse"})
        selection_method = selection["method"]
        if selection_method not in {"expert_rmse", "parent_or_final_recovery"}:
            raise ValueError("Unsupported checkpoint selection method")
        candidate = None
        recovery_validation_loader = None
        if selection_method == "parent_or_final_recovery":
            from tsn.training.selection import validation_episodes, evaluate_candidate
            if not initialize_from:
                raise ValueError("Recovery selection requires an initialization checkpoint")
            validation_recovery = RecoveryDataset(Path(selection["recovery_data_dir"]), model.chunk_size)
            recovery_datasets.append(validation_recovery)
            if set(validation_recovery.route_by_episode) - set(split["validation"]):
                raise ValueError("Checkpoint selection recovery states must belong to validation")
            recovery_validation_loader = make_loader(validation_recovery, options, False, int(options["seed"]) + 2)
            selection_ids = validation_episodes(split, catalog, selection["episodes_by_route"], int(options["seed"]) + 3)
            selection_options = read_json(selection["eval_config"])
            selection_options["device"] = str(device)
            write_json(output / "selection_protocol.json", {
                **selection, "episodes": selection_ids, "source_partition": "validation",
                "candidates": ["parent", "final_epoch"],
                "ranking": ["success_rate", "negative_collision_rate", "negative_recovery_rmse", "negative_expert_rmse"],
                "ties": "retain parent", "eval": selection_options,
            })
        if initialize_from:
            parent_validation = evaluate_predictions(model, maps, validation_loader, device)
            best_rmse = parent_validation["rmse_rad"]
            parent_payload = {
                **initialization, "epoch": 0, "config": configuration, "splits": split,
                "model": model.state_dict(), "validation": parent_validation,
                "best_rmse_rad": best_rmse, "best_epoch": 0,
                "initialize_from_checkpoint": str(initialize_from),
                "recovery_sources": resolved_sources, "selection_method": selection_method,
            }
            if recovery_validation_loader is not None:
                parent_recovery = evaluate_predictions(model, maps, recovery_validation_loader, device)
                candidate = evaluate_candidate(model, maps, device, Path(benchmark["root"]),
                    output / "selection" / "parent", selection_ids, catalog, selection_options,
                    parent_validation, parent_recovery)
                parent_payload["recovery_selection"] = candidate
            save_checkpoint(output / "parent.pt", parent_payload)
            save_checkpoint(output / "best.pt", parent_payload)
            write_json(output / "parent_validation.json", parent_validation)
            logger.info("Parent candidate validation RMSE %.6f rad", best_rmse)
        logger.info("Run %s: %d training frames, %d validation frames; %s maps",
                    output.name, len(datasets["train"]), len(datasets["validation"]),
                    'Pi3 predicted' if learned_maps else 'expert future action')
        if recovery_datasets:
            logger.info("Recovery training policy initialization checkpoint: %s", initialize_from)
            for source in resolved_sources:
                logger.info("  %s: %d samples, %.1f%% of sampled frames",
                            source.get("name", Path(source["path"]).name), source["samples"],
                            100 * float(source["sampling_fraction"]))
        for epoch in range(1, int(options["epochs"]) + 1):
            started = time.perf_counter()
            model.train()
            metrics = PredictionMetrics(model.chunk_size)
            total_loss, samples = 0.0, 0
            map_loss_sum = point_loss_sum = heatmap_loss_sum = 0.0
            learning_rate = optimizer.param_groups[0]["lr"]
            learning_rates = {group.get('name', 'policy'): group['lr'] for group in optimizer.param_groups}
            for step, raw in enumerate(train_loader, start=1):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16 if learned_maps else torch.float16, enabled=amp):
                    if learned_maps:
                        with torch.autocast(device_type=device.type, enabled=False):
                            state = policy_state(batch['qpos'], batch['goal_pose'], maps.settings)
                            teacher = maps(batch['depth'], batch['K'], batch['T_B_C'],
                                           batch['goal_pose'][:, :3], batch['future_ee'], batch['valid_future'])
                        prediction, predicted_maps = training_model(batch['rgb'], state,
                            batch['K'], batch['T_B_C'], return_maps=True)
                    else:
                        prediction = predict_batch(model, maps, batch)
                    loss = imitation_loss(
                        prediction, batch["target"], batch["valid_future"], float(options["huber_beta_rad"]),
                        float(options["horizon_decay_steps"]), float(options["horizon_weight_floor"]),
                    )
                    if learned_maps:
                        map_loss, components = predicted_map_loss(predicted_maps, teacher)
                        loss = loss + float(options['map_loss_weight']) * map_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(options["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                size = len(batch["target"])
                if learned_maps:
                    map_loss_sum += float(map_loss.detach()) * size
                    point_loss_sum += float(components['point']) * size
                    heatmap_loss_sum += float(components['heatmap']) * size
                total_loss += float(loss.detach()) * size
                samples += size
                metrics.update(prediction.detach(), batch["target"], batch["valid_future"], batch["route"])
                if step % 100 == 0:
                    logger.info("Epoch %d batch %d/%d loss %.6f", epoch, step, len(train_loader), total_loss / samples)
            validation = evaluate_predictions(model, maps, validation_loader, device)
            improved = validation["rmse_rad"] < best_rmse if selection_method == "expert_rmse" else False
            recovery_selection = None
            if selection_method == "parent_or_final_recovery" and epoch == int(options["epochs"]):
                recovery_metrics = evaluate_predictions(model, maps, recovery_validation_loader, device)
                recovery_selection = evaluate_candidate(model, maps, device, Path(benchmark["root"]),
                    output / "selection" / "final", selection_ids, catalog, selection_options,
                    validation, recovery_metrics)
                improved = tuple(recovery_selection["selection_key"]) > tuple(candidate["selection_key"])
                write_json(output / "checkpoint_selection.json", {
                    "parent": candidate, "final": recovery_selection,
                    "selected": "final_epoch" if improved else "parent",
                    "selected_epoch": epoch if improved else 0,
                })
            if improved:
                best_rmse, best_epoch = validation["rmse_rad"], epoch
            scheduler.step()
            record = {"epoch": epoch, "learning_rate": learning_rate, "train_loss": total_loss / samples,
                      "train": metrics.result(), "validation": validation,
                      "elapsed_seconds": time.perf_counter() - started, "best_epoch": best_epoch}
            if learned_maps:
                record['map_loss'] = {'total': map_loss_sum / samples, 'point': point_loss_sum / samples,
                                      'heatmap': heatmap_loss_sum / samples}
                record['learning_rates'] = learning_rates
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            payload = {
                "format_version": 1, "epoch": epoch, "config": configuration, "splits": split,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "validation": validation, "best_rmse_rad": best_rmse, "best_epoch": best_epoch,
                "initialize_from_checkpoint": str(initialize_from) if initialize_from else None,
                "recovery_sources": resolved_sources,
                "selection_method": selection_method, "recovery_selection": recovery_selection,
            }
            save_checkpoint(output / "latest.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            write_json(output / "summary.json", {
                "completed_epochs": epoch, "best_epoch": best_epoch, "best_validation_rmse_rad": best_rmse,
                "best_checkpoint": str(output / "best.pt"), "privileged_action_map": not learned_maps,
                "initialize_from_checkpoint": str(initialize_from) if initialize_from else None,
                "recovery_train_samples": sum(source["samples"] for source in resolved_sources),
                "recovery_sources": resolved_sources,
                "recovery_sampling_fraction": sum(recovery_fractions),
                "expert_sampling_fraction": 1.0 - sum(recovery_fractions),
                "selection_method": selection_method,
                "route_sampling_fractions": options.get("route_sampling_fractions"),
            })
            logger.info("Epoch %d/%d train loss %.6f validation RMSE %.6f rad (best epoch %d)",
                        epoch, options["epochs"], total_loss / samples, validation["rmse_rad"], best_epoch)
        return output / "best.pt"
    finally:
        for dataset in datasets.values():
            dataset.close()
        for recovery in recovery_datasets:
            recovery.close()
        for handler in (stream, file_log):
            logger.removeHandler(handler)
            handler.close()
