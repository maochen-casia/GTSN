# GTSN deterministic hand clearance

This checkout retains one navigation policy: `ClearancePolicy`, the deterministic
hand-clearance controller. It uses the existing single-memory Pi3 checkpoint to
propose a route, then scores fourteen bounded corrections against four remembered
RGB-predicted surface clouds. Clearance uses a fixed **0.04 m** proximity scale,
a **0.08** correction penalty, and three samples along the hand axis. It predicts
30 joint targets and executes 15 before observing again. Corrections fade near
the goal; the learned proposal, wrist orientation, and terminal servo are preserved.

The clearance stage stores points and validity masks and computes no uncertainty
inflation. The learned route head retains its trained geometry distribution and
reliability gate, which are needed to reproduce the saved proposal. Depth and
expert trajectories provide training supervision only. The geometric score is
a proximity heuristic, not a collision probability.

## Docker

Run model code and tests in Docker. The host launcher uses only Python's standard
library; install dependencies only in the project container.

```bash
docker build -t gtsn-compact:latest .
python3 scripts/docker_run.py --cpu test
```

An existing project runtime supports an offline build under a new tag:

```bash
docker build --network none \
  --build-arg BASE_IMAGE=gtsn-pi3:20261002 \
  --build-arg INSTALL_RUNTIME=0 -t gtsn-clearance:latest .
python3 scripts/docker_run.py --image gtsn-clearance:latest --cpu test
```

The launcher mounts source and `/run/user/1016` read-only and grants writes only
to a new output directory under this checkout's `runs/` or
`/run/user/1016/experiments/`. Existing output directories are rejected.
Use `--print-command` to inspect the command, `--cpu` for CPU execution, or
`--gpu ID` to select an available GPU. No existing experiment is overwritten.

## Closed-loop validation and testing

The saved `simplified.pt` loads directly into deterministic hand clearance:

```bash
python3 scripts/docker_run.py --gpu 4 evaluate \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --partition validation \
  --output-dir /home/chenmao/GTSN/runs/clearance-validation
```

Use `--partition test` and a fresh output directory for the fixed test split.
For a newly trained head, use its `best.pt`. `--episode episode_NNN` may be
repeated to evaluate a subset of the requested partition. Optional flags include
`--render-videos`, `--dataset-root`, and `--eval-config`.

Every evaluation uses deterministic hand clearance. `--clearance-margin` and
`--clearance-penalty` override its two scoring parameters; defaults reproduce
the selected method. The former mode, uncertainty, body, trigger, and geometry
intervention switches have been removed.

Each run records the effective clearance settings in `config.json`, per-episode
trajectories, metrics and correction diagnostics, cumulative `closed_loop.json`,
and `complete.json` after every requested episode finishes.

## Training

Train the learned memory route head with a frozen Pi3 backbone:

```bash
python3 scripts/docker_run.py --gpu 4 train \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/baseline.pt \
  --output-dir /home/chenmao/GTSN/runs/clearance-training
```

A compact `simplified.pt` or new `best.pt` can initialize further head training.
The baseline import extracts only the backbone and memory head. This workflow
requires an existing trained Pi3 checkpoint and its recorded recovery dataset.
The clearance refiner has no trainable weights; training optimizes the route
proposal and exports weights that the clearance loader can use directly.

`configs/train/compact.json` sets five epochs, AdamW at `3e-5`, batch size 256,
and seed `20261002`. Training caches train/validation features, preserves the
65:35 expert/recovery and 20:40:40 route sampling, and selects weights by
validation waypoint RMSE, including initialization as epoch zero. Recovery
observations have no fabricated history. Test episodes are excluded from
training and checkpoint selection.

## Python API and source

```python
from tsn.models.clearance_policy import load_clearance_policy

policy, supervision_maps, checkpoint = load_clearance_policy(
    '/run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt',
    device='cuda',
)
```

- `src/tsn/models/clearance_policy.py`: surface memory, candidate corrections,
  hand proximity, deterministic selection, and the policy loader.
- `src/tsn/models/compact_policy.py`: checkpoint-compatible learned route head,
  shared abstract route controller, and component loading.
- `src/tsn/models/pi3_policy.py`, `map_action_head.py`, `kinematics.py`: perception,
  learned joint proposal, and Panda kinematics required by the controller.
- `src/tsn/data/`, `features/`, `training/`: training inputs and supervision.
- `src/tsn/simulation/`, `evaluation/`: simulation and closed-loop metrics.
- `src/tsn/cli/`: training, evaluation, paired reporting, and visualization of the
  retained policy's predicted geometry. Reporting can compare archived results.
- `scripts/docker_run.py`: Docker launcher; `tests/`: regression checks.

`instructions/` and `documents/` preserve the original requirements and research
history. Historical documents may describe removed policies or commands;
archived experiments retain their own source snapshots. The pre-existing nested
`tsn_old/` checkout is reference material and is excluded from the package and
Docker build.
