# Geometry-grounded Table Scene Navigation

## Background
Only using one camera for table scene navigation is flexible and light weight. However, there is also limitation. A single camera can only provide weak geometry, which is essential for table scene navigation. So, we want to introduce explicit geometry-grounded representations for robust and accurate table scene navigation.

## Previous Attempt
I have implemented a geometry-grounded map policy previously. First, it utilizes privileged maps, including point map, goal map, and affordance map, to predict action chunk. The experiment is saved in `/run/user/1016/experiments/perturbation_scratch10_20261002/`, reaching a high success rate. Then, I remove the privileged maps, and use a fine-tuned backbone (pi3) to predict these maps. The experiment is saved in `/run/user/1016/experiments/pi3_small_perturbation10_20261002/`, with a significant performance drop, maybe due to the map prediction bottleneck, as diagnosed in `/run/user/1016/experiments/pi3_map_validation_20261002/. ` These are only previous attemt, which can be considered as baselines. In this geometry-grounded table scene navigation (gtsn) research, I want to build a model that is consistent with the following research challenges and contributions, and the model may be significantly different from previous attempts.

## Challenges 
1) Uncertainty: what is visible may be uncertainty, like weak texture surface, moving objects, occlusions.
2) Invisibility: the FoV of single camera is limited. so some obstacles may be invisible.
3) Action-independent: the geometry is primarily descriptive, independent from the task and goal.

## Contribution
1) Explicit uncertainty modelling, and this uncertainty can help more robust following navigation.
2) Persistent and active observation: don't only use one frame observation, but accumulate several observations from different time steps, and maintain a persistent scene representation, while also allowing the robot to actively choose future observation that provide most information when necessary.
3) Action-aware geometry: incorporate task and goal information into geometry representation.

## Technical Design
Note that the following technical design is only some rough ideas. Feel free to refine or revise them, if you find them theoretically or pratically bad, or you have any better ideas.
1) uncertainty: In addition to predict geometry map (depth map or point map), also predict uncertainty. 
2) persistent and active observation: Maintain something like scene tokens (represent different subareas of the scene), keypoints (more accurate and fine-grained geometry information), and novelty (what has been observed and what has not) across different time steps and frames.
3) action-aware geometry: introduce some task, goal, action-relevant representation, like goal map, affordance map (where the future action can be).
For the second point, maybe the representation can be supervised like: choose a history viewpoint, use the representation to recover its observation geometry (maybe like point map). choose a future viewpoint, also try to recover its geometry. For active observation, maybe something like sampling several candidate viewpoints, and rank them or use contrastive learning. it is important that the robot's main target is not to get a complete observation of the scene, but only those relevant to the task and navigation, especially those lie in the route.

## Benchmark
The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4).

## Your task and goal
You should:
1) Design the methodology and complete a document in `documents/gtsn_methodology.md` first. You may include both what you want to implement now, and potential future design if current design does not work well. You need to revise or udpate it if you change the methodology later.
2) Implement the model, and then conduct experiments including training and closed-loop testing. I would suggest you start from the previous baseline, instead of implementing everything from scratch. You don't need to implement every module all at once. It would be more beneficial to implement them step by step to see the performance change, and decide if a module design is beneficial or need further revision.
3) You are highly free for model design, including backbone selection, training strategy, and others. Just make sure that they can be summarized into something consistent with the overall story, including challenges and contributions.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data, pretrained weight, and experiment storage). You cannot edit or save anything outside these two directories.