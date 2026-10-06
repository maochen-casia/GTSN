# Table Scene Navigation

## Benchmark: tsn-1k

The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4). Training uses the corrective caches, 20 epochs.

The experiment log, result, checkpoint, should be saved in `/run/user/1016/experiments`.

## Persistent Spatial Representation
I want to introduce persistent spatial representation into current compact policy, and compare their performance. Specifically, the scene is represented by $K$ geometric gaussian (with center position, variance, etc). Given a new RGB, first extract new $N$ gaussians from this observation, then concat with previous $K$ gaussians ($K$ should be larger than $N$), and conduct self-attention among these $N+K$ gaussians and update them (the gaussian parameters). This step works as a merging step, to merge history and new gaussians. Finally, predict a score for each gaussian, and keep only the top-$K$. This score should be incorporated into the following process instead of only for selection, so that it can be updated by back-propogation.

These gaussians are used for clearance-based refine, like in clearance policy. Specifically, several candidate routes will be ranked according to their clearance to the spatial representation (gaussians) to refine a more robust and safe route. Different from current clearance policy, this stage should be designed as a learnable module and should be trained.

During training, maybe for each frame, you can sample several history frames, extract $N$ gaussians from them, and conduct self-attention among all these gaussians and select top-$K$ for clearance refinement. During evaluation, maintain $K$ gaussians, and extract $N$ from current observations. It is important to keep the training and inference behavior consistent, so that no gap will be introduced. 

## Task
You should follow the above design and implement it. However, you are also free to revise or improve the above design when necessary. After implementation, you should also run experiments to validate the effectiveness.

## Goal
You should continue working on analyzing, proposing new methods, implementing and validating through experiments, until significant improvement is attained (success rate improvement >= 5 compared to compact policy). However, if there are long training or testing process, you don't have to wait for them alive. You can complete your current work and tell me when they are expected to finish, and I will wake you up when they finish. (The compact policy evaluation is currently in progress, and it is expected to finish in 20 minutes.)

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data storage). You cannot edit or save anything outside these two directories.