# Implemented project structure

```text
GTSN/
├── configs/
│   ├── benchmark/tsn-1k.json       # Dataset location and fixed 800/100/100 split settings
│   ├── model/geometry_policy.json  # Architecture and input/output dimensions
│   ├── train/baseline.json         # Optimizer, schedule, and training options
│   └── eval/closed_loop.json       # Test episodes, control, and rendering options
├── src/tsn/
│   ├── data/
│   │   ├── splits.py               # Reproducible direct/over/side stratification
│   │   ├── hdf5_dataset.py         # Episode and frame sample access
│   │   └── loaders.py              # Train/validation/test DataLoaders
│   ├── features/
│   │   ├── maps.py                 # Online GPU point, goal, and action maps
│   │   └── state.py                # Shared robot/goal state preparation
│   ├── models/
│   │   ├── geometry_policy.py      # Policy network
│   │   └── factory.py              # One model constructor for train and evaluation
│   ├── training/
│   │   ├── runner.py               # Epoch loop, validation, and checkpoint selection
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── open_loop.py            # Prediction metrics
│   │   ├── closed_loop.py          # Test rollout orchestration
│   │   └── metrics.py              # Success, collision, and summary metrics
│   ├── simulation/
│   │   └── episode.py              # Scene setup, stepping, contacts, and rendering
│   ├── common/
│   │   ├── config.py
│   │   ├── checkpoint.py
│   │   └── seed.py
│   └── cli/
│       ├── train.py                # python -m tsn.cli.train
│       └── evaluate.py             # python -m tsn.cli.evaluate
├── scripts/
│   └── docker_run.py              # Readable Docker launcher for both commands
├── documents/
│   └── file_structure.md
├── Dockerfile
├── pyproject.toml
└── tsn_old/                        # Reference only; never imported by the new package
```

Keep the benchmark at `/run/user/1016/tsn-1k`. Generate and save the fixed split with each run; preserve the 2:4:4 route ratio in the 800/100/100 train, validation, and test partitions. Store checkpoints, logs, split records, and evaluation outputs in an external run directory mounted into Docker, not in the source tree.

The package is independently implemented and has no imports from `tsn_old`.
The Docker build context excludes that reference directory. Python replaces the
two proposed shell launchers; see `scripts/docker_run.py` and `documents/baseline.md`.

`data/hdf5_dataset.py` validates the current schema and joint ordering, reads frame
samples through a process-local bounded HDF5 cache, and supplies a mask for future
states beyond the demonstration endpoint. `features/maps.py` generates all maps
online on the policy device using each sample's camera calibration. Training and
evaluation share the policy factory and robot/goal normalization.

Checkpoints contain the complete model configuration and fixed episode split.
Evaluation validates this split against the current manifest and admits only its
test episodes. Validation RMSE selects `best.pt`; test data never enters training
or checkpoint selection.
