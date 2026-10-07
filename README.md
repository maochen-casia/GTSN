# GTSN: geometry-grounded table scene navigation

This checkout contains one main model and its training, closed-loop validation
and test paths. The policy composes three geometry modules:

| Module | File | Responsibility |
|---|---|---|
| C1 | [c1_memory.py](src/tsn/models/c1_memory.py) | Bounded persistent surface anchors, recent observations and retained uncertainty |
| C2 | [c2_embodiment.py](src/tsn/models/c2_embodiment.py) | Measured hand and rigid-tool geometry, self-surface filtering and regional contact features |
| C3 | [c3_clearance.py](src/tsn/models/c3_clearance.py) | Point-error prediction, uncertainty padding and route-preserving clearance refinement |

[policy.py](src/tsn/models/policy.py) composes these modules with RGB perception,
a four-frame route head and Panda kinematics. The hand, both fingers, rigid
wrist and camera housing are modelled; the articulated arm is excluded. The
policy predicts a 30-step chunk and executes 15 steps before observing again.
Per-episode memory resets explicitly and is never serialized as model weights.

## Main experiment

[configs/main.json](configs/main.json) is the sole experiment configuration.
It specifies the fixed 800/100/100 tsn-1k split and 2:4:4 route ratio, existing
expert plus independent perturbation data, model settings, optimizer, and
simulation settings. Primary success is collision-free XYZ reaching within
1 cm, with full XYZ-plus-orientation success reported separately.

Training initializes the geometry decoder, joint proposal, route head,
uncertainty head and clearance trust head afresh. It does **not** take a previous
navigation checkpoint. The default uses published Pi3 image-encoder weights
and freezes that encoder. For entirely random initialization, set
`model.perception.pretrained_weights` to `null` and `freeze_encoder` to `false`.
Only measured state, requested goal, camera calibration and RGB predictions
enter deployment; depth and expert futures provide training targets.

Train in the existing project Docker image, using a new output directory:

```bash
python3 scripts/docker_run.py --gpu 0 train \
  --config /home/chenmao/GTSN/configs/main.json \
  --output-dir /run/user/1016/experiments/gtsn-main-new
```

The trainer optimizes the navigation heads and decoder jointly. Earlier RGB
features are detached during training; their timestamps and episode boundaries
remain causal. Each perturbation observation starts with fresh history.
Validation waypoint RMSE selects `best.pt`; test data is never used for training
or checkpoint selection. The exported checkpoint contains the complete model
state, configuration and splits, with no parent-checkpoint chain.

Evaluate the trained main model on each full partition:

```bash
python3 scripts/docker_run.py --gpu 0 evaluate \
  --checkpoint /run/user/1016/experiments/gtsn-main-new/best.pt \
  --partition validation --no-render-videos \
  --output-dir /run/user/1016/experiments/gtsn-main-validation-new

python3 scripts/docker_run.py --gpu 0 evaluate \
  --checkpoint /run/user/1016/experiments/gtsn-main-new/best.pt \
  --partition test --no-render-videos \
  --output-dir /run/user/1016/experiments/gtsn-main-test-new
```

Optional repeated `--episode` arguments select members of the requested
partition for a short run; the result records whether the partition is complete.
Evaluation saves overall/per-route metrics, episode trajectories and optional
videos. Policy diagnostics and separate analysis/ablation entry points have
been removed.

## Source layout and verification

```text
configs/main.json
scripts/docker_run.py
src/tsn/
  models/       C1, C2, C3, composed policy, perception, route and kinematics
  data/         Benchmark schema, fixed splits and causal observation histories
  training/     Main training loop and supervised objectives
  evaluation/   Closed-loop rollouts and success/collision metrics
  simulation/   Benchmark geometry, robot execution and rendering
  features/     Training geometry targets and measured-state normalization
  common/       Configuration, seeds and standalone checkpoints
  cli/          train and evaluate
tests/          Main-model, data, geometry and execution regression checks
```

Run regression tests in Docker:

```bash
python3 scripts/docker_run.py --cpu test
```

The launcher mounts source, data and existing experiments read-only. Only the
requested new output directory is writable. It runs without networking and
requires no host package installation. The default image is
`gtsn-persistent:20261005-compact-only`; its historical tag does not select a
policy. [Dockerfile](Dockerfile) builds a new project image when needed.

## Historical research record

[documents/research_progress.md](documents/research_progress.md) and the method
documents describe the completed C1/C2/C3 studies. Their results, weights and
frozen source remain in the existing experiment archives. Historical documents
are preserved unchanged and may reference removed development scripts.

This refactor uses a new standalone checkpoint structure and fresh training
path. Historical success rates belong to the archived implementation; the
rewritten main model needs its own training and full validation/test evaluation.
Use an archive's frozen `source/` for historical checkpoint replay, rather than
loading an adapter into this main model.
