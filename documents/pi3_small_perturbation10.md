# Pi3 small decoder perturbation recovery experiment

This completed experiment implements the revised `instructions/baseline.md`. Pi3 predicts
six geometry maps from wrist RGB and measured robot and camera state, and the
action policy consumes those predictions during both training and closed-loop
testing. The experiment artifacts are saved under
`/run/user/1016/experiments/pi3_small_perturbation10_20261002`. All ten epochs and
100 test rollouts completed. Closed-loop success was 6%, collisions 58%, and
timeouts 36%, so this run's navigation performance is poor.

## Model and initialization

The official [Pi3 implementation](https://github.com/yyfz/Pi3) is vendored at
commit `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`. Its small decoder has 24 blocks,
384 features, six attention heads, and five register tokens. The ViT-L/14 image
encoder is retained to load the published Pi3 encoder weights exactly. A new
1024-to-384 projection connects it to the small decoder; this supplies a
projection missing from the upstream small configuration. A device-aware patch
position cache permits training across GPUs 0–3.

The published checkpoint uses a large decoder. Consequently, the image encoder
is pretrained, while the small decoder, point-map head, goal-map head,
future-action-map head, and action policy start from random initialization.
All parameters receive gradients and optimizer updates. No existing TSN policy
checkpoint initializes this run. This is a small-decoder Pi3 adaptation, not a
separately pretrained Pi3-small checkpoint.

The model has 351,351,914 trainable parameters, including 304,371,712 in the
image encoder. Initialization loaded all 343 encoder tensors exactly. The final
audit confirms that every encoder tensor changed in the selected checkpoint.

The full published weights are saved at `/run/user/1016/pi3/model.safetensors`.
Their SHA256 is
`33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`, matching
the [official checkpoint metadata](https://huggingface.co/yyfz233/Pi3/blob/main/model.safetensors).
`provenance.json` records the download mirror, authoritative metadata URL, file
size, and verified hash. The HTTP ETag is an Xet hash, distinct from file SHA256.

## Observations and supervision

The model receives current uint8 wrist RGB, nine joint positions, the requested
goal pose, camera intrinsics, and the measured camera-to-base transform. RGB is
resized to 84 by 112 pixels; the learned maps have 80 by 80 pixels. Predicted
channels retain the original point XYZ, projected goal, visible goal, and future
action order. The action policy predicts 30 joint residuals from the current
joint positions, and the controller executes 15 before observing and replanning.

Depth and expert future TCP positions generate map labels for the auxiliary
training loss and held-out map metrics. They never enter the Pi3 policy. Action
loss gradients flow through the predicted maps into the backbone. Closed-loop
testing loads only the initial expert joint and TCP state for simulator setup;
it never reads future expert states or computes an expert progress index.

The existing safe perturbation pool is augmented with RGB rendered at each
stored measured joint state. Camera transforms and reconstructed depth are
checked against the saved observations. Original labels and observations are
preserved byte for byte in array values, and source archives are unchanged.
The augmented pool lives in
`/run/user/1016/experiments/pi3_rgb_perturbations_20261002`.

## Training and evaluation protocol

The fixed episode partitions have 800 training, 100 validation, and 100 test
episodes. Every partition retains direct/over/side proportions 2:4:4. Training
uses 65% expert demonstration frames and 35% perturbation recovery frames, with
0.2/0.4/0.4 route sampling within each source. Perturbations are the sole recovery
source. Neither on-policy recovery nor validation/test recovery enters training.

Training runs ten epochs with seed 20261001 and frame stride two. AdamW uses
learning rates 0.00001 for the pretrained encoder and 0.0003 for the new decoder
and heads, cosine decay, weight decay 0.0001, and gradient clipping at 1.0.
The total batch size is 128 across four GPUs. Native PyTorch attention uses
bfloat16 precision. The imitation loss follows the original horizon-weighted
Smooth-L1 settings. An auxiliary loss, weighted by 0.25, combines point-map
Smooth-L1 regression with foreground-weighted heatmap squared error.

`best.pt` is selected by expert action RMSE over the 100 validation episodes;
`latest.pt` retains epoch ten. Test episodes do not participate in selection.
All test frames receive action and map prediction metrics. All 100 test episodes
receive closed-loop rollouts with 30/15 prediction/execution horizons, 20 Hz
control, 100 Hz physics, and a 400-step limit. Collision stops the rollout.
Success uses the existing XYZ criterion within 1 cm; orientation within 0.15 rad
is also reported at the stopping point.

The simulator reconstructs the benchmark geometry and conservative collision
envelopes. Its RGB omits the original generator's textures, so rendering is a
potential appearance mismatch between expert images and simulated recovery/test
images. This single-seed experiment does not establish a controlled performance
comparison against the earlier privileged-map CNN run.

## Docker execution and artifacts

All dependencies and experiment processes run in the new `gtsn-pi3:20261002`
Docker image, built from `Dockerfile.pi3`. The host Python environment is
unchanged. Training and evaluation disable container networking. Source,
benchmark, and pretrained weights mount read-only; logs, checkpoints, data
audits, metrics, and trajectories are written to the requested experiment root.

With those mounts inside the container, the driver is:

```bash
python /workspace/scripts/run_pi3_perturbation.py \
  --output-dir /run/user/1016/experiments/pi3_small_perturbation10_20261002
```

The host launcher creates these mounts and records the Docker image identifier:

```bash
python3 scripts/run_pi3_docker.py
```

The driver checks data membership and label preservation, runs regression tests,
trains, evaluates, and audits all trajectories. `protocol.json` records source
hashes and configurations; `status.json` records execution state. Training
metrics and checkpoints live in `train/`, and test metrics and per-episode
trajectories live in `test/`. `results.json` consolidates measurements, and
`audit.json` records completed epochs, horizons, split membership, finite
trajectories, absence of privileged maps, and changes to every encoder tensor.

## Results

The run completed on 2026-10-02. All 11 regression tests passed, the experiment
container exited with code 0, and `status.json` reports `complete`. The final
audit passed for all ten training epochs and all 100 finite test trajectories,
with no expert progress indices or privileged map inputs. The recorded source
hashes match the files used for the run. Training took 4,064.0 seconds (67.7
minutes), and test prediction evaluation plus rollouts took 840.8 seconds (14.0
minutes).

Epoch 7 achieved the lowest validation joint RMSE, **0.064843 rad**, and is stored
in `train/best.pt`. `train/latest.pt` retains epoch 10, whose validation RMSE was
0.064904 rad. Checkpoint selection used validation only.

| Test route | Episodes | XYZ successes | Collisions | Timeouts | Successes also within orientation tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 20 | 2 | 2 | 16 | 0 |
| Over | 40 | 4 | 21 | 15 | 2 |
| Side | 40 | 0 | 35 | 5 | 0 |
| Overall | 100 | 6 | 58 | 36 | 2 |

Overall closed-loop success was **6%**, collision rate **58%**, and timeout rate
**36%**. The median final position error was 0.115551 m and the mean was
0.153683 m. Mean rollout length was 234.71 control steps. Only 2% of episodes
also met the orientation tolerance when the XYZ rollout stopped. Side routes
had no successes.

Test prediction RMSE was **0.065209 rad**, MAE 0.040855 rad, and horizon-30 RMSE
0.101076 rad over all 14,606 frames. Horizon-15 RMSE was 0.060132 rad. The
zero-action comparison had RMSE 0.157534 rad. Route RMSEs were 0.078355 rad for
direct, 0.058568 rad for over, and 0.065755 rad for side.

| Predicted map channel | Test pixel RMSE |
| --- | ---: |
| Point X | 0.124807 |
| Point Y | 0.140444 |
| Point Z | 0.094281 |
| Projected goal | 0.094532 |
| Visible goal | 0.096775 |
| Future action | 0.067525 |

Point errors use the configured normalized coordinates; heatmap errors use
scores in [0, 1]. These pixel averages include background. The experiment
establishes completion of the requested training and testing protocol, with poor
closed-loop performance. It does not isolate whether rendering appearance,
learned map quality, or errors from executing predicted chunks caused failures.

The selected checkpoint is
`/run/user/1016/experiments/pi3_small_perturbation10_20261002/train/best.pt`.
`results.json` contains consolidated metrics; `audit.json` records the checks.
Every test episode has `metrics.json` and `trajectory.npz` under `test/episodes/`.
`figures/learning_curves.{png,pdf}` and `figures/route_outcomes.{png,pdf}` provide
exportable plots of training, horizon errors, and route outcomes.

The follow-up [map validation](pi3_map_validation.md) evaluates all 100 validation
episodes with teacher-map replacements and matched action-head adaptation.
It finds that map prediction limits open-loop accuracy, while a state-only
adapted head also outperforms the adapted head retaining predicted maps.
This follow-up diagnoses open-loop behavior without changing the checkpoint or
adding closed-loop tests.
