# Project Clean Up

## Model Version
I want to retain only the simplified compact model (single-memory model) in this project. The previous corresponding experiment is in `/run/user/1016/experiments/gtsn_simplification_20261003`.

## Task and Goal
You should clean up current project:
1) Remove previous redundant codes (`src/`) and scripts (`scripts/`) that are not directly used for the compact model version.
2) Rewrite necessary codes and scripts, as some previous one may be redundant. Keep the code and script concise, but also highly readable. Keep only the necessary one for main experiment (training, closed-loop validation and testing). You can remove those used for analysis or diagnostic experiments.
3) Keep `instructions/` and `documents/` unchanged.
4) Just clean up this project, `~/GTSN/`. Don't change anything outside this project, including the experiment result and data directory.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data, pretrained weight, and experiment storage). You cannot edit or save anything outside these two directories.