# Table Scene Navigation

## Benchmark: tsn-1k

The benchmark has been saved in `/run/user/1016/tsn-1k`, with 1,000 episodes, including 200 direct type, 400 over type, and 400 side type. 800 are used for training, 100 for validation and 100 for testing. Ensure that the ratio of different route types remain the same (2:4:4).

The experiment log, result, checkpoint, should be saved in `/run/user/1016/experiments`.

## Task
The codes have been implemented. Now please revise the model, so that it does not rely on privileged maps. Instead, it uses Pi3 as backbone with prediction heads to predict those maps, both during training and testing. Use a small version pi3 at first and full fine-tune it during training. Then run the experiment using perturbation-only recovery data, training for 10 epochs, and then closed-loop testing. Don't use existing checkpoint to initialize model. During testing, the model predicts action chunk of size 30, and then execute 15. Pi3's pretrained weights should be saved in `/run/user/1016/pi3/`.

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data storage). You cannot edit or save anything outside these two directories.