# GTSN deterministic hand clearance

This checkout provides learned compact route control and an optional deterministic
hand-clearance controller. Clearance uses the existing single-memory Pi3 checkpoint to
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

Evaluation defaults to deterministic hand clearance. `--clearance-margin` and
`--clearance-penalty` override its two scoring parameters; defaults reproduce
the selected method. The former mode, uncertainty, body, trigger, and geometry
intervention switches have been removed.

Use `--controller compact` to execute `CompactPolicy` directly, without any
clearance corrections or clearance surface history. This retains the learned
route, wrist proposal, inverse kinematics, terminal servo and 15-step execution
cadence. Persistent geometry heads still use their own learned point memory.
The output records `controller: compact` and `clearance.mode: disabled`.

Each run records the controller and effective clearance settings in `config.json`,
per-episode trajectories and metrics, cumulative `closed_loop.json`, and
`complete.json` after every requested episode finishes. Clearance runs also save
correction diagnostics.

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

## Persistent scene memory experiments

`PersistentSceneHead` replaces raw frame history with eight scene slots and,
optionally, 256 geometry slots addressed by a fixed 8 x 8 x 4 base-frame grid.
The geometry slots store surface-patch features, weighted XYZ, second moments,
and observation support. They are spatial representatives, not tracked keypoint
identities. Four scene slots have no decay; four learn observation-dependent
decay. Geometry persists until episode reset. A current-view branch and a zero
initialized memory residual preserve pretrained current-view predictions at
initialization. The frozen Pi3 backbone is shared with the existing controller.

Training uses a causal affine prefix scan, computing predictions and losses at
every timestep in parallel. The PyTorch doubling implementation has O(T log T)
work and logarithmic temporal depth; it does not require another package or a
compiled scan kernel. Online control updates the same fixed-size accumulators
once per observation, without retaining old route features. When clearance is
enabled, it uses the same four raw surface clouds for every head. Compact-only
evaluation omits that separate clearance memory.

Run each stage inside the project Docker runtime, with fresh experiment paths:

```bash
docker build --network none \
  --build-arg BASE_IMAGE=gtsn-clearance:20261005 \
  --build-arg INSTALL_RUNTIME=0 -t gtsn-persistent:20261005-v1 .

python3 scripts/docker_run.py --image gtsn-persistent:20261005-v1 --gpu 4 \
  persistent_study --stage cache \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --cache /run/user/1016/experiments/gtsn_simplification_20261003/cache \
  --output-dir /run/user/1016/experiments/persistent-cache-new --batch-size 64

python3 scripts/docker_run.py --image gtsn-persistent:20261005-v1 --gpu 4 \
  persistent_study --stage train \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --cache /run/user/1016/experiments/persistent-cache-new \
  --output-dir /run/user/1016/experiments/persistent-training-new \
  --batch-size 32 --epochs 30 --draws 1024 \
  --seed 20261005 --seed 20261006 --seed 20261007

python3 scripts/docker_run.py --image gtsn-persistent:20261005-v1 --gpu 4 \
  persistent_study --stage analyze \
  --cache /run/user/1016/experiments/persistent-cache-new \
  --study /run/user/1016/experiments/persistent-training-new \
  --output-dir /run/user/1016/experiments/persistent-analysis-new

python3 scripts/run_persistent_validation.py \
  --image gtsn-persistent:20261005-v1 \
  --analysis /run/user/1016/experiments/persistent-analysis-new \
  --output-dir /run/user/1016/experiments/persistent-validation-new \
  --gpus 0,1,2,3 --full-partition
```

Caching reuses the existing pooled features and obtains 20 x 20 predicted and
depth-supervised point samples with one frozen-backbone pass. Expert sequences
use full episode prefixes at approximately the deployed 15-step cadence,
including real timestamps. Recovery samples remain isolated. Training compares
`current`, `four`, `scene`, and `scene_points`; use repeated `--variant` flags to
select a subset. Sequence probabilities and mean sequence losses preserve the
65:35 source and 20:40:40 route mixture in expectation. Checkpoint selection uses
validation waypoint RMSE and admits epoch zero. No test observations enter
training, cache preparation, or checkpoint selection.

Analysis measures reset-every-frame ablations and recall of first-view surfaces
absent from the four latest views. The paired simulator launcher selects the
validation-best seed per variant and keeps clearance parameters fixed. Add
`--partition test --full-partition` to evaluate those locked checkpoints on the
held-out test split. `persistent_study --stage report --analysis PATH --rollouts
PATH --study PATH --cache PATH --output-dir PATH` generates paired episode
bootstrap intervals, JSON results, and a source snapshot; run it with the Docker
launcher and `--cpu`. An interrupted training study can reuse completed models
with explicit `--resume`; incomplete models restart from initialization, and
the recorded protocol must match exactly.

To compare the locked checkpoints without clearance, rerun the test partition:

```bash
docker build --network none \
  --build-arg BASE_IMAGE=gtsn-clearance:20261005 \
  --build-arg INSTALL_RUNTIME=0 -t gtsn-persistent:20261005-compact-only .

python3 scripts/run_persistent_validation.py \
  --image gtsn-persistent:20261005-compact-only \
  --analysis /run/user/1016/experiments/gtsn_persistent_analysis_20261005 \
  --output-dir /run/user/1016/experiments/persistent-compact-test-new \
  --partition test --full-partition --controller compact --gpus 0,1,2,3
```

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
  shared route controller, direct `CompactPolicy`, and component loading.
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
