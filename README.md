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

`instructions/` contains project requirements; `documents/research_story.md`
describes the current research and experiments. Other historical documents may
refer to removed experiments. The pre-existing `tsn_old/`
nested checkout is not used by this package or included in its Docker build.

## Research extension: preserve the route, correct the clearance

The current research story and its claim boundaries are in
[`documents/research_story.md`](documents/research_story.md). The extension retains
the compact checkpoint and queries its RGB-predicted surfaces with the proposed
hand motion. It stores four observations, uses a fixed surface-proximity scale,
and selects a bounded route correction with a penalty for departure from the
proposal. It adds no trainable
weights. Depth and simulator object poses remain unavailable to the policy.

The revised method is `deterministic`, margin `0.04` m, and correction penalty
`0.08`. It obtains 85% validation and 83% test success, versus 82% and 78% for
the compact baseline. This is an exploratory choice informed by earlier test
results; the original validation-selected uncertainty variant remains reported
at 86% validation and 79% test. Evaluate the revised method in a fresh directory:

```bash
python3 scripts/docker_run.py --gpu 0 evaluate \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --partition test --clearance-mode deterministic \
  --clearance-margin .04 --clearance-uncertainty 0 --clearance-penalty .08 \
  --output-dir /run/user/1016/experiments/route-preserving-replay
```

`--clearance-mode current` and `tcp` remove historical surface points and the two
additional hand samples. Set `--clearance-uncertainty 0` for matched controls of
the revised fixed-margin method. Set `--clearance-penalty 0` in deterministic
mode to remove the correction penalty. These switches affect only the geometric refiner; the frozen proposal
retains its original memory and learned reliability. Omit the switch to run the
unchanged compact baseline. This geometric cost is a heuristic, not a collision
probability or a safety guarantee.

Reproduce validation selection, four ablations, fixed test evaluation, paired
statistics, and PNG/PDF figures with available project GPUs:

```bash
python3 scripts/run_clearance_study.py \
  --root /home/chenmao/GTSN/runs/clearance-reproduction --gpus 0,1,2,3,4
```

The script creates a source snapshot and logs, runs all numerical work in Docker,
calibrates the constant-margin control on training observations, and freezes the
selected setting before test evaluation. Existing artifacts are never overwritten.
The original iteration is under `runs/clearance_20261003/`.

Reproduce the revised study's three controls on both fixed splits, reusing the
archived baseline/method runs explicitly:

```bash
python3 scripts/run_route_preserving_study.py \
  --root /run/user/1016/experiments/route-preserving-reproduction \
  --gpus 0,1,2,3,4
```

The completed source snapshots, protocols, trajectories, and reports are stored
under `/run/user/1016/experiments/gtsn_research_20261003/` and
`/run/user/1016/experiments/gtsn_route_preserving_20261003/`.

An unsuccessful learned residual-retrieval experiment is retained separately in
`runs/evidence_20261003/`. All five fifteen-epoch fits selected the unchanged
initialization. Its implementation is `EvidenceRouteHead`; it is not the selected
clearance method and should not be described as a successful trained upgrade.
