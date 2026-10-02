## Task

The problem diagnosis is in `documents/recovery_diagnosis.md`. You should first read it, and then fix the problem 1, 3, 4, 5, and rerun the mixed recovery data experiment. You don't need to fix problem 2 currently. 

## Importante Notes

You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data storage and experiment results). You cannot edit or save anything outside these two directories.