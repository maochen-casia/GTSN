# Simplify Model

## Baseline
A good baseline has been established, in `/run/user/1016/experiments/gtsn_consensus_20261003_fresh/`, which achieves a success rate of 78 (epoch 15 checkpoint). You can use this baseline, like using its trained backbone pi3, and frozen this backbone in your experiments to speed up.

## Goal and task
You are supposed to simplify the model code, simplifying or removing those complicated but maybe redundant modules that contribute slightly to overall performance. So you should first analyze the model architecture to see what may be simplified or removed. And then run experiment to see if it lead to significant performance drop. If not (like success rate drop less than 5), retain this simplification. For each experiment, maybe just training 10 epochs or fewer, since you can use the tuned pi3 backbone from previous experiment and frozen it.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data, pretrained weight, and experiment storage). You cannot edit or save anything outside these two directories.