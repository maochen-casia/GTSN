## Updated benchmark
A new benchmark has been generated in `/run/user/1016/tsn-1k-var`, where different episodes have different camera intrinsics and mounted positions.

## Task
You should update the current model and also run experiment on this updated benchmark.
1) Generate perturbation-only recovery data for training.
2) Update the model code wherever necessary to accomodate the changing camera settings. For example, predicting geometry in camera frame and then transforming it into base frame may be more robust.
3) Run the experiment on the updated benchmark. Similarly, 800 training, 100 validation and 100 testing episodes. The pi3 backbone is also fully tuned.
4) in addition to main experiment, also run three ablation study on three contribution models. w/o c1: only use current frame without any history frames. w/o c2: only modelling TCP instead of the embodiment. w/o c3: without clearance refinement.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for experiment and data storage). You cannot edit or save anything outside these two directories. Remember to remove your unsuccessful trials that you will no longer use to free the space.