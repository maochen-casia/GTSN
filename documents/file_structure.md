# Proposed project structure

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
│   ├── train_docker.sh
│   └── evaluate_docker.sh
├── documents/
│   └── file_structure.md
├── Dockerfile
├── pyproject.toml
└── tsn_old/                        # Reference only; never imported by the new package
```

Keep the benchmark at `/run/user/1016/tsn-1k`. Generate and save the fixed split with each run; preserve the 2:4:4 route ratio in the 800/100/100 train, validation, and test partitions. Store checkpoints, logs, split records, and evaluation outputs in an external run directory mounted into Docker, not in the source tree.
