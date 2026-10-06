# GTSN current-view corrective control

The retained model uses current RGB-predicted geometry, four compressed frame
features, and a correctively trained route residual. Spatial slots are rebuilt
at every observation; there is no persistent spatial memory. The frozen Pi3
backbone, wrist proposal, inverse kinematics, terminal servo, and 15-step
execution cadence are preserved. Evaluation uses `--controller compact`, which
disables the separate clearance controller.

On the fixed tsn-1k splits, the retained checkpoint achieved **85/100 validation**
and **88/100 test** successes. Original compact achieved 82/100 and 78/100.
The persistent corrective alternative achieved 89/100 and 86/100. These results
support retaining corrective training and current-view control; they do not
establish an additional benefit from persistent spatial memory.

Cleanup verification passed 51 regression tests and reproduced 88/100 test
successes. Two individual episode outcomes swapped (one gain, one loss), so the
GPU replay is not trajectory-identical. CPU predictions matched the archived
implementation exactly across 1,004 cached observations; GPU differences were
within the measured mixed-precision variation of the unchanged reference.
The original checkpoint and evaluation trajectories remain unchanged.

## Retained experiment

The experiment stays at
`/run/user/1016/experiments/gtsn_spatial_v7_corrective_20261005`.

- `training/spatial_history_corrective_current/spatial_history_corrective_current/seed_20261005/best.pt`:
  original checkpoint, preserved byte for byte.
- `followup/validation/spatial_history_corrective_current/` and
  `followup/test/spatial_history_corrective_current/`: original evaluations and trajectories.
- `collection/shard_0/` through `shard_3/`: consolidated corrective histories and metadata.
- `initialization/parent_head.pt` and `initialization/initial_residual.pt`:
  the frozen parent head and original residual initialization, without duplicate backbones.
- `reference/compact_validation/` and `reference/compact_test/`: baseline evidence.
- `source/` and `source_sha256.json`: immutable source from the original experiment.
- `cleanup/`: approved removal manifest and code regression checks.
- `retained_model.json`: current retained-artifact index. Earlier reports remain
  historical records and can refer to removed alternatives.

The shared feature cache remains at
`/run/user/1016/experiments/gtsn_persistent_cache_20261005`. The existing
`gtsn_simplification_20261003/simplified.pt` supplies the unchanged backbone for
retraining. The benchmark, pre-existing experiments, and Docker images are retained.

## Docker and evaluation

Run model code, simulation, and tests in Docker. The host scripts use only the
Python standard library. The existing runtime is
`gtsn-persistent:20261005-compact-only`; its historical name does not select a model.

```bash
RUN=/run/user/1016/experiments/gtsn_spatial_v7_corrective_20261005
CHECKPOINT="$RUN/training/spatial_history_corrective_current/spatial_history_corrective_current/seed_20261005/best.pt"

python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --cpu test

python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 evaluate \
  --controller compact --checkpoint "$CHECKPOINT" --partition test --no-render-videos \
  --output-dir /run/user/1016/experiments/current-view-test-new
```

Use `--partition validation` for validation. Each output directory must be new.
The launcher mounts source and data read-only, allowing writes only to that
output directory. `--source-snapshot PATH` executes an archived compatible source.
No host environment installation is needed.

## Corrective training

The saved data contains 800 training episodes and 8,255 observations: 8,072 safe
proposals, 163 verified corrections, and 20 observations with masked supervision.
Each safe proposal becomes its own target. Unsafe proposals receive the first
correction verified for all 30 simulator steps, with offsets up to 3.5 cm.
Teacher trials restore simulator state, velocities, and motor targets. Only
training collection uses that search; deployment performs no teacher search.

Only the added residual trains, with a 4 cm per-coordinate bound. Corrected
observations receive weight 8. Sampling uses only corrective histories and keeps
the direct/over/side ratio at 2:4:4. The schedule is 20 epochs, 4,096 sequence draws
per epoch, batch size 32, and seed 20261005. The final epoch is selected in advance.
The parent parameters remain frozen and are checked before export. GPU reduction
and mixed-precision rounding can prevent bitwise reproduction across runs; the
original checkpoint and source snapshot remain the exact reproducibility record.

To train and validate a new copy using the retained data:

```bash
RUN=/run/user/1016/experiments/gtsn_spatial_v7_corrective_20261005
python3 scripts/run_corrective_study.py \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --initial-head "$RUN/initialization/parent_head.pt" \
  --initial-residual "$RUN/initialization/initial_residual.pt" \
  --cache /run/user/1016/experiments/gtsn_persistent_cache_20261005 \
  --recovery-cache "$RUN/collection/shard_0" \
  --recovery-cache "$RUN/collection/shard_1" \
  --recovery-cache "$RUN/collection/shard_2" \
  --recovery-cache "$RUN/collection/shard_3" \
  --baseline-validation "$RUN/reference/compact_validation" \
  --baseline-test "$RUN/reference/compact_test" \
  --output-dir /run/user/1016/experiments/current-view-training-new --gpu 0
```

The workflow snapshots source, trains one model, hashes the fixed checkpoint,
and evaluates validation. Add `--test` explicitly to evaluate the fixed test split.
The test result never selects or changes the checkpoint. `corrective --stage train`
through `scripts/docker_run.py` provides the training stage alone.

To recollect training data with the frozen parent, use the same base checkpoint
and small parent head:

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 corrective \
  --stage collect \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --initial-head "$RUN/initialization/parent_head.pt" \
  --collection-episodes 800 --seed 20261005 \
  --output-dir /run/user/1016/experiments/current-view-collection-new
```

For parallel collection, use `--shards 4 --shard 0` through `--shard 3`, each with
its own GPU and output directory. Pass all four caches to training.

## Source

- `src/tsn/models/current_view_policy.py`: current geometry, four-frame carry,
  bounded corrective residual, and strict loading of the retained legacy checkpoint.
- `src/tsn/models/compact_policy.py`: shared compact controller and model loading.
- `src/tsn/training/corrective.py`: corrective sampling, residual training, export,
  and frozen-parent checks.
- `src/tsn/training/sequence_recovery.py`: physics-verified training labels.
- `src/tsn/cli/corrective.py`: collection and training entry points.
- `scripts/run_corrective_study.py`: single-model training and evaluation workflow.
- `scripts/verify_current_view.py`: comparison against immutable original source.

The existing generic compact trainer (`train`) and optional hand-clearance
controller remain available. Evaluation defaults to clearance for compatibility;
always pass `--controller compact` for the retained corrective model's protocol.
For a compact-head comparison on the same four corrective shards, run:

```bash
python3 scripts/run_compact_corrective_study.py \
  --output-dir /run/user/1016/experiments/compact-corrective-new
```

This freezes the Pi3 backbone and fine-tunes the original compact head at `3e-5`
for 20 epochs with the same corrective sampling and weighted action objective.
It locks epoch 20 before evaluating both full splits on GPUs 0 and 1, and saves
paired reports against the retained current-view model and original compact.
The evaluated original compact checkpoint was removed during cleanup; the new
run initializes from `gtsn_simplification_20261003/simplified.pt`.
`report` provides paired comparisons and figures from completed evaluations.
Historical research variants and their drivers have been removed from active code.
`instructions/`, `documents/`, and the pre-existing nested `tsn_old/` checkout remain
reference material; historical commands there may describe removed experiments.
