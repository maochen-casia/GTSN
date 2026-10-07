# Table Scene Navigation

## Benchmark: tsn-1k

The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4).

The experiment log, result, checkpoint, should be saved in `/run/user/1016/experiments`.

## Research Topic

The main idea is 3D geometry-grounded table scene navigation. I have some general ideas for your reference that may be helpful:

1. persistent geometry, incorporate history geometry information and maintain persistent representation that can help following action prediction.
2. actionable geometry, geometry is usually descriptive, and maybe can be incorporated with task-relevant information or embodied information.

These ideas are just for your reference. You don't have to strictly follow them. You are free to propose your own ideas.

## Current resources

I have implemented three baseline policies: pi3 policy, compact policy, and clearance policy. They may achieve competitive performance, but do not strongly align with the main idea of geometry-grounding topic. You can refer to them, see what may be reusable and helpful. But you don't have to strictly follow their implementation. Feel free to propose your new design.

## Task
You should think about the research gaps and contributions centered around the topic of 3D geometry-grounded navigation, finishing a complete and coherent research story with 3 challenges and corresponding contributions. And you should also implement your ideas, run the experiment to validate the effectiveness. You are highly free in idea thinking, design, and implementation. The criteria are:

1. Each contribution should be verified and supported by performance improvement. So it should be supported by ablation result.
2. The overall success rate of both validation and testing set should >= 70. No need to surpass the existing baselines.
3. A coherent research story should be established centered around geometry-grounded navigation.
4. Try to work on model architecture first, using expert route + perturbation-only recovery data for training. Don't start with data changing. But you are allowed to do so when necessary.
5. Summarize your successful progress (idea + result) in `documents/research_progress.md` briefly.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for experiment and data storage). You cannot edit or save anything outside these two directories. Remember to remove your unsuccessful trials that you will no longer use to free the space.