# GTSN compact model

This checkout supports the **single-memory** model selected in
`/run/user/1016/experiments/gtsn_simplification_20261003`.
Its saved `simplified.pt` loads directly, without the former consensus models.

The model uses Pi3 RGB features, learned geometry reliability, four observations
of memory, and six Cartesian route knots. It predicts 30 joint targets and
executes 15 per observation, preserving the saved controller's wrist orientation
and terminal goal servo. Depth and expert targets are training supervision only.

## Docker

Run all model code and tests in Docker. The host launcher uses only Python's
standard library; no host package installation is needed.

```bash
docker build -t gtsn-compact:latest .
python3 scripts/docker_run.py test
```

On this machine, the existing project runtime also supports an offline build:

```bash
docker build --network none \
  --build-arg BASE_IMAGE=gtsn-pi3:20261002 \
  --build-arg INSTALL_RUNTIME=0 -t gtsn-compact:latest .
```

This creates a new image without changing the existing image. The launcher mounts
source and `/run/user/1016` read-only, and grants writes only to the new output
directory. Output must be under this checkout's `runs/` or
`/run/user/1016/experiments/`. Existing output directories are rejected.
Use `--print-command` before the command to inspect mounts and arguments.
Use `--cpu` for CPU execution, or `--gpu ID` to select an available GPU.

## Training

Train the single memory head with the reference's frozen epoch-15 Pi3 backbone:

```bash
python3 scripts/docker_run.py --gpu 4 train \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/baseline.pt \
  --output-dir /home/chenmao/GTSN/runs/compact-training
```

A compact `simplified.pt` or new `best.pt` can instead initialize further head
training. The baseline import extracts only the Pi3 weights and memory head; it
does not instantiate or train other branches. This workflow reproduces the
compact fine-tuning stage, and requires an existing trained Pi3 checkpoint and
its recorded recovery dataset. It does not retrain Pi3 from scratch or generate
recovery data.

`configs/train/compact.json` sets five epochs, AdamW at `3e-5`, a 256-sample
head batch, and seed `20261002`. Training caches only train/validation features,
uses the original 65:35 expert/recovery and 20:40:40 route sampling, and selects
weights by validation waypoint RMSE, including the initialization as epoch zero.
Recovery observations have no fabricated history. Cached batches are loaded on
demand rather than keeping the full cache in GPU memory.

The output includes the standalone `best.pt`, selected `head.pt`, configuration,
fixed episode splits, cache, per-epoch metrics, and completion summary.

## Closed-loop validation and testing

Validate the existing compact export:

```bash
python3 scripts/docker_run.py --gpu 4 evaluate \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --partition validation \
  --output-dir /home/chenmao/GTSN/runs/compact-validation
```

Then evaluate the fixed model on the test split with `--partition test` and a
new output directory. For a newly trained model, use its `best.pt`. Evaluation
uses the saved checkpoint's simulator settings and split; test episodes are
never used by training or checkpoint selection. `--episode episode_NNN` may be
repeated to run a subset belonging to the requested partition. Optional flags
include `--render-videos`, `--dataset-root`, and `--eval-config`.

Each evaluation writes episode trajectories and metrics, a cumulative
`closed_loop.json`, and `complete.json` only after all requested episodes finish.

## Source layout

- `src/tsn/models/`: compact controller, memory head, Pi3 components, and Panda kinematics.
- `src/tsn/data/`, `features/`, `training/`: training inputs, supervision, and frozen-feature training.
- `src/tsn/simulation/`, `evaluation/`: simulator and closed-loop metrics.
- `src/tsn/cli/`: training and evaluation entry points.
- `scripts/docker_run.py`: the Docker launcher.
- `tests/`: compact model, training, data, and rollout regression checks.

`instructions/` and `documents/` retain the original historical material, whose
older commands may refer to removed experiments. The pre-existing `tsn_old/`
nested checkout is not used by this package or included in its Docker build.
