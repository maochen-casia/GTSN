# Simplify Model

## Baselines
There are several different baselines implemented. After experiments, I want to retain only the deterministic hand clearance version, that is `clearance_policy.py`.

## Goal and task
You should clean and remove other unwanted policy codes (including `body_clearance.py`), retain only the deterministic hand clearance version. And you may also need to rewrite or clean the relevant codes to the retained policy, removing extra functions (like uncertainty), and make the retained code structure more clear. 

## Importante Notes
You cannot directly install environment or packages in user environment. Instead, prepare the needed environment in Docker, and use Docker to run process. You cannot edit or change other existing docker images irrelevant to this project or belong to other users. You can directly edit codes or scripts in user environment, and read files. If any actions require installing new packages, do it in docker. Keep the user environment clean.  You can only work on /home/chenmao/GTSN/ (for codes and scripts) and /run/user/1016/ (for data, pretrained weight, and experiment storage). You cannot edit or save anything outside these two directories.