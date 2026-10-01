# tsn-1k geometry baseline

This package reimplements the geometry policy described in `tsn_old` for the
current 1,000-episode benchmark. It uses one frame of depth and robot/goal state,
generates geometry maps online, and regresses a deterministic 30-step chunk of
seven arm joint residuals. It does not import or execute the reference code.

## Data and fixed partitions

The benchmark is read from `/run/user/1016/tsn-1k`. Its manifest lists the episode
directory and route label. Each directory supplies `episode.h5` and `scene.json`.
The implementation uses the verified current HDF5 fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `depth_m` | `(T,240,320)` | Wrist-camera Z-depth, metres |
| `qpos` | `(T,9)` | Seven Panda arm angles plus two finger positions |
| `ee_pose` | `(T,7)` | Executed TCP XYZ and quaternion wxyz in base frame |
| `T_base_camera_cv` | `(T,4,4)` | OpenCV camera coordinates to robot base |
| `intrinsics` | `(3,3)` | Per-episode pinhole calibration |
| `T_ee_camera_cv` | `(4,4)` | Camera to TCP transform |
| `goal_pose_xyz_wxyz` | `(7,)` | Goal XYZ and orientation |
| `joint_names` | `(9,)` | Explicit joint ordering |
| `time_seconds` | `(T,)` | 20 Hz demonstration timestamps |

Episode IDs are shuffled within routes with seed `20261001`. Frames from one
episode always remain together. Partition membership and the catalog hash are
stored with every run and checkpoint, then reused during evaluation.

| Partition | Direct | Over | Side | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 160 | 320 | 320 | 800 |
| Validation | 20 | 40 | 40 | 100 |
| Test | 20 | 40 | 40 | 100 |

Training uses every second frame; validation uses the same stride for checkpoint
selection. Final test prediction metrics use every frame. All demonstrated
future states contribute; terminal padding is masked out of both the loss and
future action map.

## Policy and map semantics

The six input channels at 80 by 80 pixels are normalized point XYZ, projected
goal, visible goal, and future action. Backprojection uses per-sample intrinsics
with pixel-center corrections when resizing. Projected goals ignore occlusion
and invalid depth; visible goals and actions use measured depth to test visibility.
Metric Gaussian distance to camera rays defines goal/action scores. Future
action scores decay exponentially with horizon and reduce by maximum, so nearer
future TCP positions have more weight. Trajectory blocks bound GPU map-generation
memory; no maps are saved as a preprocessed dataset.

A residual CNN with widths 32/64/128/192 preserves spatial information using
4 by 4 pooling and a 384-dimensional map projection. A separate MLP encodes
16 state values: arm angles divided by pi, finger positions divided by 0.04 m,
normalized goal XYZ, and a canonical unit goal quaternion. The fused head
produces `(B,30,7)` residuals bounded by 0.75 radians. Each horizon is relative
to the joint state at the beginning of the chunk. Neither RGB nor route labels
enter the policy.

The future-action channel uses future **expert** TCP positions in both training
and evaluation, matching the geometry prototype's input semantics. Closed-loop
evaluation chooses expert progress by monotonic nearest arm joint state.
This is a privileged geometry baseline: it requires an expert demonstration at
inference and its results must be interpreted with that input available.
Output configs and metrics explicitly record `privileged_action_map: true`.

## Optimization and evaluation

Training uses AdamW, a cosine learning-rate schedule, gradient clipping, and
optional CUDA mixed precision. Smooth-L1 loss weights near horizons more heavily
and excludes padded targets. `best.pt` is selected only by validation RMSE;
`latest.pt` preserves the final epoch's model and optimizer state. Checkpoints
also include scheduler/scaler state, configuration, and split membership.
The current CLI starts fresh runs; it does not provide a resume command.

Open-loop metrics report masked RMSE, MAE, a zero-action comparison, final-horizon
RMSE, and RMSE per route/horizon. Closed-loop evaluation reconstructs the scene,
uses fresh wrist depth, predicts a chunk, executes its first five targets, and
replans from the measured state. Fingers retain their initial positions. Joint
targets are clipped to URDF limits, and collisions are checked at every 240 Hz
physics substep. A collision prevents an episode from counting as successful.
Success requires both position and orientation tolerances. Evaluation defaults
to all 100 fixed test episodes.

`scene.json` exports object kinds, centers, half-sizes, yaw, and colors, but does
not export the generator's full geometry, table dimensions or controller
settings. The implemented simulator uses static boxes and cylinders defined by
those fields, configurable table/drive settings, and the ManiSkill Panda URDF.
It recreates primitive geometry rather than original decorative object meshes
or surface textures. Consequently these rollouts are reconstructed-scene results;
exact agreement with the generator's rendering and contact behavior is not
established. The evaluation config makes the missing scene/control assumptions
explicit. A startup guard checks TCP position/orientation against the recorded
initial state and rejects incompatible robot assets or TCP settings.

## Docker environment and commands

The existing `tablescenenav:demo` image was inspected without modification. Its
PyTorch is CPU-only, so the new Dockerfile starts from
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` and declares the remaining packages
in `pyproject.toml`: NumPy, h5py, SAPIEN 3.0.3, ManiSkill 3.0.1, trimesh,
PyOpenGL, freetype-py, pyglet, imageio/FFmpeg, Pillow, NetworkX, SciPy, and six.
Pyrender 0.1.45 is installed separately with `--no-deps` because its obsolete
PyOpenGL pin conflicts with the modern OSMesa wrapper; all its actual runtime
dependencies are declared explicitly. `pip check` will therefore report that
upstream metadata mismatch. System libraries provide OSMesa, OpenGL, freetype,
OpenMP, and FFmpeg. Panda URDF/meshes are included in the ManiSkill package.

References for the selected interfaces: [PyTorch CUDA 12.4 installation](https://docs.pytorch.org/get-started/previous-versions/),
[SAPIEN CPU-only scenes](https://sapien-sim.github.io/docs/user_guide/getting_started/physics.html),
and [pyrender offscreen rendering](https://pyrender.readthedocs.io/en/stable/examples/offscreen.html).

The following are commands for a later run; no image build, smoke test, training,
or evaluation was executed as part of this implementation.

```bash
docker build -t gtsn-baseline:tsn-1k .
python3 scripts/docker_run.py --print-command train
python3 scripts/docker_run.py train --run-dir /run/user/1016/experiments/baseline
python3 scripts/docker_run.py evaluate --checkpoint /run/user/1016/experiments/baseline/best.pt
```

For prediction metrics only, append `--mode open-loop` to the evaluation command.
Use `--no-render-videos` to omit MP4s while still rendering the live depth needed
by the policy. `--episode episode_NNN` may be repeated to select a subset of the
checkpoint's test split; the saved config distinguishes this from the full test
benchmark. The launcher accepts `--cpu` before the subcommand when CUDA is not
available, or an alternative built image through `--image`.

The launcher mounts source and benchmark read-only, experiments read/write,
uses the caller's UID/GID, disables container networking, and allocates shared
memory for DataLoader workers. It never installs packages on the host or builds
images. Runtime temporary files live in a disposable container tmpfs.
The Python module entry points inside Docker are `python -m tsn.cli.train` and
`python -m tsn.cli.evaluate`; both accept config paths via `--help`.

## Artifacts and verification scope

All experiment outputs are restricted to `/run/user/1016/experiments`.
New run/evaluation directories are required to avoid overwriting earlier results.

```text
RUN/
  config.json, environment.json, splits.json
  training.log, epochs.jsonl, summary.json
  best.pt, latest.pt
  evaluation_TIMESTAMP/
    config.json, splits.json, open_loop.json, closed_loop.json
    episodes/episode_NNN/
      metrics.json, trajectory.npz, wrist.mp4
```

The implementation was checked through static Python syntax parsing, JSON
configuration parsing, package-local import inspection, and read-only benchmark
schema inspection. Runtime behavior, dependency installation, simulator fidelity,
GPU operation and learning performance remain unverified, as requested.
