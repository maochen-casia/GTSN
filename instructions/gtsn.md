# Geometry-grounded Table Scene Navigation

## Background
Only using one camera for table scene navigation is flexible and light weight. However, there is also limitation. A single camera can only provide weak geometry, which is essential for table scene navigation. So, we want to introduce explicit geometry-grounded representations for robust and accurate table scene navigation.

## Baseline
A good baseline has been established, in `/run/user/1016/experiments/gtsn_simplification_20261003`.

## Challenges 
The following challenges are only for your reference. You can revise or refine them freely. But I would like to keep a 3-challenge style, and each should be reasonable and intuitive and not too technically detailed.
1) Uncertainty: what is visible may be uncertainty, like weak texture surface, moving objects, occlusions.
2) Invisibility: the FoV of single camera is limited. so some obstacles may be invisible.
3) Action-independent: the geometry is primarily descriptive, independent from the task and goal.

## Contribution
The following contributions are only for your reference. You can revise or refine them freely. But they should be keep consistent to the challenges, and corresponding technical implementation.
1) Explicit uncertainty modelling, and this uncertainty can help more robust following navigation.
2) Persistent and active observation: don't only use one frame observation, but accumulate several observations from different time steps, and maintain a persistent scene representation, while also allowing the robot to actively choose future observation that provide most information when necessary.
3) Action-aware geometry: incorporate task and goal information into geometry representation.


## Benchmark
The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4).

## Your task and goal
You should:
1) Figure out a coherent research story, including challenges and contributions. You are highly free to design this part, just make it coherent and reasonable. Write the story into `documents/research_story.md`.
2) Implement the updated model according to research story, and run the experiments to validate the effectiveness. 
3) Continue working on 2 and 3, until achieving both a novel and coherent research story and performance improvment.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data, pretrained weight, and experiment storage). You cannot edit or save anything outside these two directories.