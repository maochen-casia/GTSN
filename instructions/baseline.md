# Table Scene Navigation

## Benchmark: tsn-1k

The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4).

## Baseline

There's one baseline model to work as a model prototype. The previous version is in `tsn_old`. Please read it carefully. This previous version is only for reference, as it is used to conduct experiment on another benchmark. You cannot directly use any codes or scripts from it. Instead, you should re-implement the model part while allowing it to run in current benchmark.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean. For training and testing, you may build a docker image based on tablescenenav:demo, if it can be used, or build a new minimal one. Don’t change or edit any existing dockers. You can only work on /home/chenmao/TableSceneNav/ (for codes and scripts) and /run/user/1016/ (for data storage). You cannot edit or save anything outside this directory.