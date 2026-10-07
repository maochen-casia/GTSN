# Implemented project structure

```text
GTSN/
├── configs/
│   ├── benchmark/tsn-1k.json       # Dataset location and fixed 800/100/100 split settings
│   ├── model/geometry_policy.json  # Architecture and input/output dimensions
│   ├── train/baseline.json         # Optimizer, schedule, and training options
│   ├── train/onpolicy_recovery10.json # Recovery fine-tuning options
│   └── eval/closed_loop.json       # Test episodes, control, and rendering options
├── src/tsn/
│   ├── data/
│   │   ├── splits.py               # Reproducible direct/over/side stratification
│   │   ├── hdf5_dataset.py         # Episode and frame sample access
│   │   ├── loaders.py              # Expert/recovery training and evaluation DataLoaders
│   │   ├── recovery_dataset.py     # Lazy access to recovery observations
│   │   └── recovery_generation.py  # Safe joint-perturbation recovery samples
│   ├── features/
│   │   ├── maps.py                 # Online GPU point, goal, and action maps
│   │   └── state.py                # Shared robot/goal state preparation
│   ├── models/
│   │   ├── geometry_policy.py      # Geometry-map action policy
│   │   ├── pi3_policy.py           # Pi3 small decoder and learned map heads
│   │   └── factory.py              # One model constructor for train and evaluation
│   ├── training/
│   │   ├── runner.py               # Epoch loop, validation, and checkpoint selection
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── open_loop.py            # Prediction metrics
│   │   ├── map_diagnostics.py      # Map accuracy, teacher interventions, paired episode bootstrap
│   │   ├── closed_loop.py          # Rollouts and optional on-policy sample capture
│   │   └── metrics.py              # Success, collision, and summary metrics
│   ├── simulation/
│   │   └── episode.py              # Scene setup, stepping, contacts, and rendering
│   ├── common/
│   │   ├── config.py
│   │   ├── checkpoint.py
│   │   └── seed.py
│   └── cli/
│       ├── train.py                # python -m tsn.cli.train
│       ├── evaluate.py             # python -m tsn.cli.evaluate
│       └── collect_recovery.py     # python -m tsn.cli.collect_recovery
├── scripts/
│   └── docker_run.py              # Docker launcher for training, collection, and evaluation
├── documents/
│   └── file_structure.md
├── Dockerfile
├── Dockerfile.pi3                 # Project image for Pi3 experiments
├── pyproject.toml
└── tsn_old/                        # Reference only; never imported by the new package
```

Keep the benchmark at `/run/user/1016/tsn-1k`. Generate and save the fixed split with each run; preserve the 2:4:4 route ratio in the 800/100/100 train, validation, and test partitions. Store checkpoints, logs, split records, and evaluation outputs in an external run directory mounted into Docker, not in the source tree.

The package is independently implemented and has no imports from `tsn_old`.
The Docker build context excludes that reference directory. Python replaces the
two proposed shell launchers; see `scripts/docker_run.py` and `documents/baseline.md`.

The revised baseline uses `configs/{model,train,eval}/pi3_small.json` and
`scripts/run_pi3_docker.py`. Pi3 source is pinned in `vendor/Pi3`; published
weights remain outside the repository under `/run/user/1016/pi3`. Map labels
supervise training and held-out metrics, while the action policy consumes only
predicted maps. See `documents/pi3_small_perturbation10.md` for its protocol.

`scripts/validate_pi3_maps.py` evaluates map quality and all teacher replacement
combinations on validation episodes. `cache_pi3_training_maps.py`,
`adapt_pi3_action_head.py`, and `run_pi3_head_diagnostics.py` compare action heads
with frozen maps using train-only updates. `summarize_pi3_map_validation.py`
audits and consolidates results, and `plot_pi3_map_validation.py` exports figures.
See `documents/pi3_map_validation.md` for the completed diagnosis.

`data/hdf5_dataset.py` validates the current schema and joint ordering, reads frame
samples through a process-local bounded HDF5 cache, and repeats the known terminal
state as valid holding targets. `features/maps.py` generates all maps
online on the policy device using each sample's camera calibration. Training and
evaluation share the policy factory and robot/goal normalization.

Checkpoints contain the complete model configuration and fixed episode split.
Evaluation validates this split against the current manifest and admits only its
test episodes. Validation RMSE selects `best.pt` and includes the initialization
checkpoint as a candidate. Recovery fine-tuning can instead select between the
parent and final epoch using fixed validation rollouts and recovery predictions.
Test data never enters training or checkpoint selection.

`collect-recovery` runs a deterministic, route-stratified subset of the checkpoint's
training episodes and saves observations from its closed-loop replans with expert
future-action labels. Its manifest records the selected training IDs. Fine-tuning
accepts a recovery subset only when every archive matches that manifest and every
episode belongs to the training partition.

`simulation/benchmark_scene.py` supplies the complete tsn-1k depth scene and fixed
benchmark physics settings. `training/selection.py` isolates validation candidates.
`fix-experiment` executes acceptance, data generation, matched control/mixed runs,
selection, and full test evaluation; see `documents/recovery_fix_experiment.md`.
Its configurations are `configs/{model,train,eval}/recovery_fixed.json`.

The frozen epoch-15 simplification study is documented in
`documents/model_simplification.md`. `scripts/run_simplification_docker.py`
snapshots and launches the matched ablations; `verify_simplification_docker.py`
verifies the exported policy. `models/simplified_consensus.py` loads the standalone
compact checkpoint, and `scripts/evaluate_simplified.py` evaluates its saved test
split in Docker.
# Geometry-grounded energy study (2026-10-07)

- `src/tsn/models/geometric_energy.py`: explicit swept-surface energy, bounded
  learned route trust, component controls and a strict thin-checkpoint loader.
- `src/tsn/cli/geometric_energy.py`: fixed expert/perturbation-cache training.
- `scripts/run_geometric_energy_study.py`: source/checkpoint freezing and the
  validation/test/report panels in project Docker.
- `src/tsn/cli/geometric_energy_report.py`: completed-result consolidation,
  criteria receipts and exportable architecture/ablation figures.
- `tests/test_geometric_energy.py`: geometry, empty evidence, hand extent, trust
  bounds, episode resets and exact fixed-trust decision parity.
- `documents/research_progress.md`, `research_story.md`,
  `geometric_energy_method.md`: brief progress, full story and method specification.
- `runs/geometric_energy_20261007`: link to the complete experiment archive.

## C1 persistent-geometry follow-up

- `src/tsn/models/adaptive_geometry.py`: bounded anchored map, observation
  agreement, optional support weights, persistent reads and matched memory modes.
- `src/tsn/cli/evaluate.py`: `adaptive_geometry` controller and strict current
  frame / geometry-only / recent / persistent controls.
- `scripts/run_c1_study.py`: immutable source, validation nomination, frozen test
  panels, compatible control reuse and Docker reports.
- `src/tsn/cli/c1_report.py`: paired outcomes, actual map-use audits, target
  criteria and PNG/PDF figures.
- `src/tsn/cli/c1_verify.py`: active/frozen complete-controller command parity.
- `tests/test_adaptive_geometry.py`: view votes, retention, disagreement,
  capacity, causality, query parity, reset and strict current-RGB invariance.
- `documents/c1_persistent_geometry.md`: method, complete results and limits.
- `runs/c1_consensus_20261007`, `c1_geometry_only_20261007` and
  `c1_map_attribution_20261007`: completed primary and attribution archives.

## C3 uncertainty-clearance follow-up

- `src/tsn/models/uncertain_clearance.py`: point-error prediction, persistent
  anchor uncertainty, point-specific inflation and no/fixed/uniform controls.
- `src/tsn/cli/uncertainty.py`: fixed-epoch quantile training using existing
  expert/perturbation point targets, calibration records and input hashes.
- `scripts/run_c3_study.py`: frozen validation selection, matching test controls
  and reuse of the verified C1 fixed-clearance reference.
- `src/tsn/cli/c3_report.py`: paired effects, calibration, trajectory and action
  audits, target receipts, and exportable method/result figures.
- `tests/test_uncertain_clearance.py`: quantile gradients, cell alignment,
  monotone inflation, finite/empty risk, reset and complete control parity.
- `documents/c3_uncertainty_clearance.md`: architecture, training, controls and
  reproduction; experiment storage is under the designated external root.

## C2 embodied-geometry follow-up

- `src/tsn/models/embodied_clearance.py`: known Panda collision geometry,
  measured finger opening, observation-time self filtering, regional/signed
  contact representations and exact TCP/axial controls.
- `scripts/run_c2_study.py`: immutable validation panels, enforced nomination,
  frozen full-test comparisons and compatible reference reuse in Docker.
- `src/tsn/cli/c2_report.py`: primary/secondary paired effects, full trajectory
  audits with reuse deduplication, target receipts and PNG/PDF research figures.
- `src/tsn/cli/c2_verify.py`: active/archived complete-command parity through
  map/history, uncertainty, embodiment scoring and IK.
- `tests/test_embodied_clearance.py`: metric collision primitives, finger
  opening, rotation covariance, self filtering, regional contact aggregation,
  empty evidence, posture candidates and complete-controller control parity.
- `documents/c2_embodied_geometry.md`: method, staged evidence and reproduction.
