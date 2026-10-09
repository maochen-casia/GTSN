# Table Scene Navigation

## Benchmark: tsn-1k-var

The benchmark has been saved in `/home/datasets_v2/chenmao/tsn-1k-var/`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. During training, use the existing perturbation-only recovery data as well.

The experiment log, result, checkpoint, should be saved in `/home/datasets_v2/chenmao/experiments/`.

## Task
Now I want to run experiments for baselines and compare with my model. You should:
1) Search the internet to find potential baselines that may be suitable for this benchmark. Maybe some baselines need to be adapted. I would prefer those published since 2025, preferrably at some top conferences and journals, like CVPR, ECCV, ICCV, NIPS, ICML, etc.
2) Implement these baselines, run the training and closed-loop evaluation.
3) 3-4 baselines are needed.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on `/home/chenmao/GTSN/` (for codes and scripts) and `/home/datasets_v2/chenmao/` (for data storage and experiment). You cannot edit or save anything outside these two directories.