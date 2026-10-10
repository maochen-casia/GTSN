# Table Scene Navigation

## Benchmark: tsn-1k

The benchmark has been saved in `/home/datasets_v2/chenmao/tsn-1k-var`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing.

The experiment log, result, checkpoint, should be saved in `/home/datasets_v2/chenmao/experiments`.

## Problem
Current C1 (persistent geometry) and C2 (embodiment representation) doesn't use learning-based modules. I want to introduce some learning-based modules into it.

## Potential Direction
1) For C1, maybe you can maintain several points (like in current implementation). However, the update, addition, or removal of the points are learning-based, considering current state, goal, and point redundancy. For example, apply several attention blocks to the points (self-attention or cross-attention), and a head predicts score to decide what points should be retained.
2) For C2, maybe use several points to represent the embodiment. Update these points during navigation, representing the motion of the robot arm.
3) The scene points from C1 and the robot points from C2 can therefore support C3's clearance refinement.

## Target
You are highly free and flexible to update the model. The above potential directions are only for your reference. The main target is to introduce learning-based modules into C1 and C2. New updates can be accepted if they doesn't lead to performance drop. It is suggested to update C1 and C2 independently to examine the effect. And you can use frozen Pi3 backbone from previous checkpoint.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on `/home/chenmao/GTSN/` (for codes and scripts) and `/home/datasets_v2/chenmao/` (for experiment and data storage). You cannot edit or save anything outside these two directories. Remember to remove your unsuccessful trials that you will no longer use to free the space.