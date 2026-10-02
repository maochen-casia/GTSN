# Perturbation recovery baseline trained from scratch

This experiment implements `instructions/baseline.md`: ten epochs from random
initialization, followed by closed-loop evaluation of the reserved test set.
The run directory is
`/run/user/1016/experiments/perturbation_scratch10_20261002`.

## Protocol

The fixed episode partitions contain 800 training, 100 validation, and 100 test
episodes, with direct/over/side proportions 2:4:4 in every partition. Sampling
also uses probabilities 0.2/0.4/0.4 within each source. Training uses 65% expert
demonstration frames and 35% safe joint perturbation recovery observations.
"Perturbation-only recovery" means perturbations are the sole recovery source;
expert demonstrations remain in the training mixture. No on-policy recovery
observations or existing model weights are used.

The corrected existing perturbation pool in
`recovery_fixed_20261001/perturbation_data` supplies 2,997 samples from 797
training episodes. Recovery sample counts are 639 direct, 1,106 over, and 1,252
side. Three episodes had no safe perturbations. All archive targets are finite,
terminal holds are supervised, and minimum clearance exceeds 2 mm. Archive and
manifest hashes are recorded in `data_audit.json`.

The model uses `configs/model/recovery_fixed.json`, including 30-step action
chunks and corrected action-map sigma/decay settings of 0.02 m / 20 steps.
AdamW starts at LR 0.0003 with cosine decay over ten epochs; batch size is 64,
frame stride is 2, and training seed is 20261001. The remaining optimization
settings match the scratch baseline. `best.pt` is selected by expert prediction
RMSE over all 100 validation episodes. `latest.pt` retains epoch 10. Test
episodes never enter checkpoint selection.

Evaluation uses `configs/eval/recovery_fixed.json`: predict 30 actions, execute
15, then replan; 20 Hz control, 100 Hz physics, up to 400 control steps, and
stop on collision. Success requires TCP XYZ within 1 cm without collision;
orientation-constrained success within 0.15 rad is also reported. All 100 test
episodes receive open-loop prediction metrics and closed-loop rollouts.

The future-action map uses privileged expert TCP positions at inference. These
results therefore describe an expert-conditioned geometry baseline. Simulation
uses the corrected reconstructed benchmark geometry and conservative collision
envelopes; it does not establish identical original rendering/contact behavior.

## Execution and artifacts

The experiment runs in the existing `gtsn-baseline:tsn-1k-gpu` Docker image on
GPU 0. Source and benchmark mounts are read-only, container networking is
disabled, and outputs go to the external experiment directory. No host packages
are installed, and unrelated images and containers are unchanged.

Inside that container, the reproducible workflow is:

```bash
python /workspace/scripts/run_scratch_perturbation.py \
  --output-dir /run/user/1016/experiments/perturbation_scratch10_20261002
```

The driver audits the recovery pool, runs the seven existing regression tests,
trains, evaluates the selected checkpoint, and checks all 100 trajectories.
`protocol.json` records configuration and source hashes. `status.json` records
progress; `results.json` and `audit.json` are written after successful evaluation.
Logs live in `logs/`; model checkpoints and epoch metrics live in `train/`;
test metrics and episode trajectories live in `test/`.

## Results

The experiment completed on 2026-10-02. All ten training epochs finished, and
epoch 10 achieved the lowest validation RMSE, 0.043361 rad. Both `train/best.pt`
and `train/latest.pt` consequently contain the final epoch. Training took
820.8 seconds; open-loop plus closed-loop testing took 560.5 seconds.

| Test route | Episodes | XYZ successes | Collisions | Timeouts | XYZ successes also within orientation tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 20 | 20 | 0 | 0 | 13 |
| Over | 40 | 31 | 7 | 2 | 23 |
| Side | 40 | 23 | 8 | 9 | 16 |
| Overall | 100 | 74 | 15 | 11 | 52 |

Overall closed-loop success is **74%**, with **15%** collisions and **11%**
timeouts. The median final position error is 0.009678 m; the mean is 0.040841 m.
The mean rollout length is 170.82 control steps. The orientation column measures
orientation at the stopping point of the XYZ rollout, rather than running a
separate orientation-dependent rollout.

Test open-loop joint RMSE is **0.044078 rad**, MAE is 0.025270 rad, and the
30th-horizon RMSE is 0.072134 rad, over all 14,606 test frames. The zero-action
comparison has RMSE 0.157534 rad. Route RMSEs are 0.040863 rad for direct,
0.048919 rad for over, and 0.041780 rad for side.

All seven regression tests passed. The final audit confirms ten completed
epochs, no initialization checkpoint, finite model weights, fixed disjoint
splits, all 100 test episodes, finite saved trajectories, and 30/15 prediction
and execution horizons. The experiment container exited with code 0.
`status.json` reports `complete`; `results.json` contains the consolidated
measurements and `audit.json` records the successful final checks.

The selected checkpoint is
`/run/user/1016/experiments/perturbation_scratch10_20261002/train/best.pt`.
Every test episode has `metrics.json` and `trajectory.npz` under `test/episodes/`.
Videos were disabled; live depth rendering remained active for policy input.
These are results from one training seed. Comparisons against earlier
fine-tuning runs are not controlled ablations because initialization, learning
rate, source/route sampling, and checkpoint selection differ.
