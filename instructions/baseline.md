# Table Scene Navigation

## Benchmark: tsn-1k

The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4).

## Baseline

There's one baseline model to work as a model prototype. The previous version is in `tsn_old`. Please read it carefully. This previous version is only for reference, as it is used to conduct experiment on another benchmark. The new baseline code has been implemented in this project.

## Project Structure
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

The experiment log, result, checkpoint, should be saved in `/run/user/1016/experiments`.

## Task
The first stage of training and closed-loop testing has been finished, in `/run/user/1016/experiments/geometry_baseline_gpu_20261001/`. Now please try to introduce recovery data, as in `tsn_old`, and then see if performance can be improved. Only train model for 10 epochs. 

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data storage). You cannot edit or save anything outside these two directories.