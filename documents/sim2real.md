# Simulation-to-real deployment analysis

Date: 2026-10-08. Scope: the current `src/tsn` main model, `configs/main.json`,
the tsn-1k data, and the fresh frozen/tuned checkpoint results. This is an
analysis and proposed implementation plan; no model, controller, or training
configuration was changed, and no real-robot performance has been measured.

## 1. Conclusion and meaning of direct deployment

**The policy has an interface that can be supplied from real sensors, but the
current checkpoint is not yet demonstrated to transfer to a different camera
or real robot.** Correct calibration and image resizing are necessary, but
cannot by themselves ensure transfer from this training distribution.

There are two useful deployment goals:

- **Existing-checkpoint transfer:** keep learned weights fixed; supply calibrated
  RGB, synchronized measured state, and a base-frame goal through a hardware
  adapter. This is a valid experiment, with unmeasured success and collision rates.
- **Simulation-only training for real deployment:** improve simulation coverage
  and the model, then deploy without real demonstration fine-tuning. This remains
  direct sim-to-real transfer and is the recommended development target. Camera
  calibration and robot commissioning are still required.

Assume initially the same Panda arm, gripper, TCP definition, and static tabletop
reaching task. A different arm, grasping/contact task, moving scene, or changed
tool is a larger transfer problem. The actual camera model, mounting transform,
robot interface, and computing hardware have not been specified; numerical
perturbation ranges below are proposed experiments, not measured hardware limits.

The best immediate path is to build the calibrated sensor/controller adapter,
measure the fixed checkpoint under controlled camera changes, and then train on
camera-varied simulation. Explicit camera-ray geometry is the most useful
architectural improvement if learned calibration conditioning fails.

## 2. What the current system actually does

| Stage | Current behavior | Deployment consequence |
|---|---|---|
| Observations | RGB `uint8 (B,H,W,3)`, 16-value state, `K (B,3,3)`, and camera-to-base `T_B_C (B,4,4)` | These inputs can be measured or calibrated without simulator geometry. Streaming policy requires `B=1`. |
| State | Seven arm joints divided by pi; two finger positions divided by 0.04 m; normalized base-frame goal XYZ and canonical `wxyz` quaternion | Joint ordering, units, TCP, and goal frame must match exactly. No joint velocity is included. |
| Perception | Resize every image to **84×112 (H×W)**; ImageNet normalization; Pi3/DINOv2 image encoder and a newly trained decoder | Source resolution can vary, but internal spatial information remains small. Published Pi3 geometry/camera heads are not loaded. |
| Camera conditioning | Normalize row 0 of K by source width and row 1 by source height; concatenate K, state, and pose into a 41-value MLP input | Calibration is a learned conditioning signal, not an explicit projection constraint. |
| Geometry | Predict normalized base-frame XYZ, projected/visible goal scores, and action scores on an 80×80 grid; sample XYZ to 20×20 | Runtime metric geometry comes from RGB predictions, not depth backprojection. |
| Route | Four observation histories, each pooled into 4×4 cells; predict six Cartesian knots, interpolated to 30 steps | Time, view distribution, and coarse spatial pooling matter. Pi3 decoding itself is called with one frame. |
| C1/C2/C3 | Persist predicted surface points; filter modeled hand/tool surfaces; choose among 14 route corrections using learned error padding | This is a heuristic refinement over observed/predicted surfaces, not complete collision checking. |
| Actions | IK converts the refined route to joint targets; return 30 joint offsets relative to measured joints at observation time | The hardware adapter must preserve the anchor and trajectory timing. |
| Execution | Execute 15 targets at 20 Hz before the next observation; simulation uses 100 Hz physics | A 1.5 s proposal supplies a nominal 0.75 s execution interval. Inference does not advance simulation time. |

Sources: [perception](../src/tsn/models/perception.py),
[state representation](../src/tsn/features/state.py),
[policy](../src/tsn/models/policy.py), [route](../src/tsn/models/route.py),
[closed-loop evaluation](../src/tsn/evaluation/closed_loop.py).

Depth and expert future TCP positions generate training targets in
[GeometryMaps](../src/tsn/features/maps.py) and
[training_loss](../src/tsn/training/runner.py). They do not enter live policy
inference. The evaluator reads an initial expert state and a goal to initialize
the simulated episode; a real adapter instead obtains its initial state from
robot measurements and its goal from the task interface. Scene JSON and simulator
contact information support simulation/evaluation, not policy perception.

### Evidence from data and existing results

A read-only scan of **all 1,000** `/run/user/1016/tsn-1k/episode_*/episode.h5`
files found:

- All RGB arrays have per-frame shape `(240,320,3)` and depth `(240,320)`.
- All intrinsics are identical: `fx=fy=110.85301971435547`, `cx=160`, `cy=120`.
  Thus `fx/W≈0.346416`, `fy/H≈0.461888`, and both normalized principal points
  are 0.5. The nominal edge-extent field of view is approximately **110.57°
  horizontal / 94.54° vertical**—a wide view that a narrower real camera cannot
  reproduce by resizing.
- `T_ee_camera_cv` is effectively fixed: the largest absolute matrix-element
  difference from episode 000 is `2.3921e-6`. In episode 000 its translation is
  approximately `[0.046498, -0.020001, -0.067399]` m and its rotation approximately
  `[[0,-1,0],[1,0,0],[0,0,1]]`. Use the full calibrated matrix, not this rounded
  description, in any adapter.
- The **mount** is fixed, but `T_base_camera_cv` changes as the arm moves. For
  example, episode 000's camera translation spans approximately
  `[0.02454, 0.43761, 0.02144]` m over its trajectory.

The current loaders/trainer contain no camera randomization or RGB augmentation.
They consume stored expert and perturbation observations; robot-state recovery
perturbations are not evidence of camera-domain robustness.

The [tuned run summary](../runs/main_full_finetune_20261007/summary.json) records
83/100 validation and 87/100 test XYZ successes, with 17%/13% collisions.
XYZ-plus-orientation success is only **31%/29%**, and is 0% on side routes in both
partitions. The frozen-encoder run reaches 73%/69% XYZ success; see
[fresh training results](main_fresh_training.md). These are fixed-camera
simulation results. Neither the tuned model's better simulation success nor
the frozen encoder's pretrained features establish better real-world transfer.
The main config still sets `freeze_encoder=true`; the tuned checkpoint contains
its own configuration with this set to false.

## 3. Intrinsics, extrinsics, and image resolution

### 3.1 Intrinsics: an input field does not guarantee generalization

In `RGBPerception.forward`, the image encoder and decoder process RGB **before**
K or pose is introduced. A shared MLP embedding of state/pose/K is then added to
every decoded patch, followed by linear dense heads and output nonlinearities.
For the point head, before `tanh`, this has the form
`W_point h_patch + W_point c(state,pose,K) + b`: calibration contributes the same
within-patch output pattern at every patch. It cannot directly change attention
or enforce the pixel's camera ray. Later route processing is richer, but the
point field itself has no analytic camera transformation.

Because normalized K is constant throughout expert data, training does not
require the network to learn how focal length or principal point changes affect
metric geometry. Similarly, pose and joint state are correlated through one
fixed camera mount. Supplying the correct new K and pose is semantically right,
but success at unseen calibrations remains an empirical question.

For an initial fixed-checkpoint deployment:

- Calibrate the actual camera mode and rectify distortion into a pinhole image.
  Pass the resulting image's K. The current policy has no distortion-coefficient
  input, rectification stage, or invalid-image mask.
- Match normalized field of view and principal point to the training camera
  when physically possible. A crop/remap can select rays that the camera sees;
  it cannot manufacture missing peripheral coverage.
- Keep focus/zoom and capture mode stable, or update and validate their
  calibration. Test changes on recorded RGB/state before moving the robot.

OpenCV documents pinhole calibration, distortion, and camera-to-gripper
hand-eye calibration; these are suitable tools for creating the adapter's
calibration, not evidence of policy robustness.
[OpenCV calibration API](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

### 3.2 Extrinsics: distinguish optical pose and physical tool geometry

Define `T_A_B` to transform coordinates from frame B into frame A. Let E be the
model's `panda_hand_tcp`, C the rectified OpenCV optical frame, and B the Panda
base. At image exposure time `t_img`, the adapter should compute

```text
T_B_C(t_img) = T_B_E(q(t_img)) @ T_E_C
```

If hand-eye calibration is relative to the flange F instead, use
`T_E_C = inverse(T_F_E) @ T_F_C`. If image rectification rotates the optical
coordinate frame, include that rotation in the supplied pose. Supply
camera-to-base, not its inverse. The current renderer converts OpenCV axes to
OpenGL internally with `diag(1,-1,-1,1)`; this graphics conversion does not belong
in real OpenCV input data.

Synchronize RGB exposure and joint measurements, interpolate robot state to
exposure time, and validate calibrated FK/TCP against independent measurements
over multiple arm configurations. Real controller end-effector frames can differ
from the model TCP. A base-frame goal must also be obtained without simulator
ground truth, for example through a surveyed target or calibrated perception.
Goal-registration error is part of the reaching error budget.

There is a second dependency: [C2](../src/tsn/models/c2_embodiment.py) loads the
camera housing, wrist, hand, and finger collision shapes from ManiSkill's Panda
v3 URDF. Updating `T_E_C` does **not** move this camera housing. A new mount can
therefore corrupt self-point filtering and body-clearance scoring even with
perfect optical calibration. Model its actual housing, bracket, tool and relevant
cable envelope in TCP coordinates, and use consistent geometry in simulation.

Kinematics and embodiment tensors are registered model buffers and are saved
inside checkpoints. `load_policy()` constructs them from the URDF and then
strictly restores the saved state dict. **Editing the installed URDF alone may
be overwritten by checkpoint loading** (or cause a shape mismatch). Deployment
geometry needs an explicit, validated override after loading, or a versioned
separation of learned weights from robot calibration/geometry. Changing the
collision model also changes the input distribution of the learned clearance
trust function, so re-evaluate it.

### 3.3 Resolution: mostly supported at the interface, with precise limits

The model's input contract accepts arbitrary positive source H and W for an
individual image and always resizes to 84×112. Only the configured **internal**
dimensions must be divisible by 14. A 640×480 source does not require changing
`input_hw`, the map size, or checkpoint tensors. Increased camera resolution
does not automatically increase the 48 encoder patches, 16 route cells, or 400
clearance points used by this configuration.

For pixel coordinates whose integer values denote pixel centers, a crop starting
at `(x0,y0)`, followed by `align_corners=False` resizing with scales `(sx,sy)` and
padding `(px,py)` on the left/top, gives the following derived transform:

```text
u' = sx * (u - x0 + 0.5) - 0.5 + px
v' = sy * (v - y0 + 0.5) - 0.5 + py

A = [[sx, 0, sx*(0.5-x0)-0.5+px],
     [0, sy, sy*(0.5-y0)-0.5+py],
     [0,  0, 1]]
K' = A @ K
```

This is a derivation for the repository's resampling convention, not a rule
that every sensor mode uses the same pixel-center mapping. A hardware crop,
binning mode, rotated image, or ISP resize must use its actual mapping/calibration.
Distortion rectification is nonlinear and first establishes its own output K.

The runtime contract should be: **K describes exactly the RGB tensor supplied to
the model.** Pass native RGB with native/rectified K, or preprocessed RGB with its
transformed K. Do not pre-scale K to 84×112 while still supplying native RGB;
`RGBPerception` will normalize against the native tensor dimensions again.

The training map generator already accounts for the half-pixel offset when it
resizes depth to 80×80. A CPU check using the real `GeometryMaps` implementation,
a constant-Z plane, and the benchmark K produced **zero maximum map difference**
between 240×320 and a consistently transformed 480×640 input. This verifies this
analytic resize case, not learned-policy resolution invariance or behavior at
depth discontinuities.

The perception conditioning uses `cx/W, cy/H`, so it is only approximately
invariant under that pixel-center resize convention. In the same doubling test,
the normalized principal-point changes were approximately `0.00078125` and
`0.00104167`. This small discrepancy is distinct from a changed field of view.
For a newly trained model, use consistent `(cx+0.5)/W, (cy+0.5)/H` encoding or
explicit rays; changing this silently for an existing checkpoint alters its
trained input distribution.

Other practical resolution gaps:

- Different aspect ratios are stretched to 4:3. This is a valid affine pixel
  transform with matched K, but appearance and ray coverage differ from training.
  Evaluate it; do not assume switching to letterboxing preserves behavior.
- Black rectification borders or letterbox pixels have no validity mask and can
  create spurious predicted geometry. Prefer a calibrated valid crop where
  feasible; train a validity-aware path for substantial missing image regions.
- Bilinear RGB downsampling has no explicit antialias option enabled. Test
  blur/aliasing, compression, thin objects, and exposure differences. Introducing
  a new filter is a preprocessing change that needs evaluation.
- The datasets stack four histories and use default DataLoader collation. Mixed
  image sizes within a history/batch will fail even though single-image inference
  accepts them. Add calibrated preprocessing before stacking, or group batches
  by size. Keep RGB/depth/validity and K transformations consistent; avoid
  averaging foreground/background depths across object boundaries.

### 3.4 Why image warping alone does not solve a new mount

For an ideal pinhole camera at the same optical center, a ray remap can change
intrinsics and rotation over overlapping coverage. With a translated wrist
camera, reprojection depends on scene depth and changes occlusions. A single
depth-independent homography cannot reproduce arbitrary new views. Use genuine
re-rendered camera poses for training/evaluation; depth-based offline reprojection
can supplement them but must mark disocclusion holes. Never change K or pose
metadata without changing the image unless the experiment specifically measures
calibration error.

## 4. Other transfer gaps and proposed remedies

| Gap and source evidence | Why it matters | Proposed remedy |
|---|---|---|
| Synthetic appearance: fixed room/material construction with scene colors/light; no trainer augmentation | Real reflections, shadows, sensor response, blur and clutter can corrupt both routes and metric XYZ | Randomize illumination, material/texture, background and camera response; apply temporally plausible color/noise/blur/exposure augmentation to expert and recovery data. Compare frozen and tuned encoders under held-out domains. |
| Metric XYZ is inferred from RGB; invalid depth targets mask XYZ loss, but no runtime point-validity head exists | Unseen objects and missing/invalid image regions may produce finite, plausible-looking false surfaces | Add validity supervision and ray-based depth geometry; measure metric error and missed obstacles, not only visual quality. Keep unknown space distinct from observed free space. |
| Fixed workspace in `workspace_mask`: `0.10<x<1.05`, `abs(y)<0.60`, `0.04<z<0.65` m | Tabletop at z≈0 and obstacles outside this box are excluded from C1/C3; different table/base geometry is not covered | Register the workspace to the robot base, provide explicit table/fixture collision geometry, and make admissible workspace and body constraints configurable. Widening the mask alone does not retrain metric perception. |
| C1 keeps old anchors without expiry, fixes their positions/creation uncertainty, and uses 2 cm voxels / 2.5 cm matching | Calibration bias can accumulate as duplicate or misplaced surfaces; moved obstacles leave stale anchors; small obstacles can be lost | Reset on calibration changes; add time-aware uncertainty and observation-consistency/free-space invalidation, with explicit unknown-space handling. Evaluate moving scenes separately. |
| Both C1 and C2 discard points within 7 cm of current TCP; C2 excludes the articulated arm | Near-contact obstacles and arm collisions can be missed even with good RGB geometry | Replace the spherical exclusion with measured robot-surface filtering where possible; check full robot/tool and swept trajectories against geometry. |
| C3 scores only steps 3, 6, 9, 12, 15 and averages six body-region responses | A collision between samples or at one region may not dominate cost; there is no reject/stop candidate if all routes are bad | Add hard feasibility rejection and denser/swept full-body checks on the final joint path, plus a controlled hold/replan outcome. |
| Point-error head trained with 0.9 quantile loss; radius in 5–200 mm, padding capped at 30 mm | Distribution shift, calibration bias, missing objects and correlated errors are not certified by nominal training coverage | Evaluate empirical conditional coverage and severe underestimation; include calibration/tracking/latency terms and trigger abstention when budget is exceeded. Do not equate 30 mm padding with a safety bound. |
| Clearance refinement happens before IK; IK clamps joint position but does not verify residual, rate limits, or swept collisions | Independent IK solutions may jump, fail to realize a correction, or violate actuator limits | Check final FK residual and consecutive joint differences; enforce continuity and position/velocity/acceleration/jerk limits; re-check the executed interpolated path. |
| Static benchmark objects, expanded box collision envelopes, disabled link gravity, fixed drives | Visual surfaces differ from collision envelopes; real friction, gravity compensation, payload and tracking errors change trajectories | Identify tracking behavior and add controller/payload/delay variation. Record both nominal surfaces and collision-envelope definitions so metrics remain interpretable. |
| Near-goal servo switches to a straight path within 8 cm and preserves current orientation; C3 correction fades to zero within 2.5 cm | Obstacle avoidance weakens near the target, and precise orientation is not assured | Require a verified free terminal approach or collision-aware terminal controller; add an explicit orientation objective if the real task needs one. |
| Training reconstructs C1 from at most four history observations; deployment retains older anchors | Long real episodes and repeated failures can produce memory distributions not exercised by training | Train/evaluate longer causal sequences, including recovery, occlusion and reset cases. |

Evidence: [C1](../src/tsn/models/c1_memory.py),
[C2](../src/tsn/models/c2_embodiment.py),
[C3](../src/tsn/models/c3_clearance.py),
[kinematics](../src/tsn/models/kinematics.py),
[route/terminal behavior](../src/tsn/models/route.py),
[losses](../src/tsn/training/losses.py),
[benchmark construction](../src/tsn/simulation/benchmark_scene.py),
[simulation execution](../src/tsn/simulation/episode.py).

Some values are hard-coded despite nearby config fields. Point normalization is
fixed in the policy/route; workspace bounds live in C1; and
`EpisodeSimulation` merges `BENCHMARK_SETTINGS` over evaluation options. Merely
editing a table/drive setting in `configs/main.json` may therefore leave the
actual simulation unchanged. Introduce explicit domain overrides and log the
resolved values when building a transfer benchmark.

### Timing and action semantics need a real execution layer

The evaluator forms every target as
`q_target[i] = q_observation[:7] + policy_offset[i]`. It does not add each offset
to the previous command or the latest measured q. Keep this anchor semantics,
and use actual measured joints for subsequent observations. Gripper positions
remain fixed in the simulator; this policy does not generate grasp actions.
If a real gripper API reports total opening width, convert it into the two
per-finger joint positions expected by `policy_state`.

Summing `mean_inference_ms * replans` across each tuned run's 100 episode metric
files gives replan-weighted mean inference times of **115.77 ms validation /
119.88 ms test**. Median *episode mean* times are 95.59/99.75 ms. These are archived
inference measurements, not target-hardware latency or per-call tail percentiles;
the timer excludes rendering/acquisition. The simulated robot is paused during
inference, so these evaluations do not test observation staleness while moving.

Use a separate real-time execution process that consumes validated trajectories,
interpolates 20 Hz knots, monitors tracking and stale observations, and holds or
stops on invalid results. Keep GPU inference outside that process. Franka's FCI
uses a 1 kHz real-time interface and constrains motion derivatives, so 20 Hz
policy targets require an execution bridge with smooth commands.
[Franka FCI overview](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html).

Measure acquisition-to-command latency, including transfer and queue delay. An
initial controlled hold while observing/inferencing most closely matches the
simulator's pause semantics, but its transitions must be tested. If execution
continues during inference, stale predictions need explicit time alignment and
matching delayed-observation training. Do not silently re-anchor old offsets.
Shorter execution horizons may improve responsiveness, but `execution_horizon()`
currently returns a hard-coded 15 even if the evaluation option is changed.
Also adapt history sampling/ages: training uses 15-step observation spacing,
and the route encodes age in control steps divided by 60. More frequent views
change both temporal context and compute requirements.

A useful engineering error-budget approximation is

```text
position error ≲ translation-calibration error
               + range * rotation-calibration error (radians)
               + perception error + tracking error
               + relative scene/body speed * observation-to-action delay
```

This is a first-order planning estimate, not a probabilistic guarantee; include
angular body motion when estimating relative speed. At 0.5 m range, 1° rotation
error contributes approximately 8.7 mm. At 0.1 m/s, 100 ms delay corresponds to
10 mm displacement. Either is already comparable to the benchmark's 10 mm
reaching tolerance. Persistent memory does not remove a shared systematic bias.

## 5. Simulation-only improvements, in priority order

### A. Calibrated adapter and an unchanged-checkpoint baseline

Implement timestamped RGB/joint capture, pinhole rectification with output K,
calibrated camera-to-TCP transformation, goal registration, and explicit robot
geometry loading. Preserve the checkpoint's input normalization and 84×112
processing initially. Add episode reset, monotonic observation steps, finite and
SE(3)/intrinsic validation, stale-image detection, IK residual checks, and the
trajectory execution layer. Specify fail/hold behavior for missing observations
or empty geometry: current C2 assigns empty scenes zero risk.

This supports a meaningful zero-shot checkpoint test without real fine-tuning.
Measure calibration error on held-out poses, recorded-frame geometry, and open
space trajectory tracking before obstacle trials. The existing 13% simulated
test collision rate means clearance refinement alone cannot serve as the
hardware's collision protection.

### B. Camera-varied rendering and simulation training

Render the same training scene/robot trajectories from varied K, resolutions,
and mounts. Use a fixed physical mount and optics within each episode, with
per-frame variations only for physically plausible noise, exposure, delay or
actual mount motion. Update `T_B_C` from FK and the sampled mount for every frame.
Regenerate RGB, depth, validity and geometry targets consistently. Extend recovery
observations too; otherwise their fixed camera creates a shortcut.

Keep two experiments distinct:

- **Camera diversity:** render with changed true calibration and supply that
  correct calibration. This measures generalization across cameras.
- **Calibration uncertainty:** render with true calibration but supply controlled
  perturbed estimates. Retain true geometry for supervision/evaluation. This
  measures robustness to imperfect calibration; it does not teach the camera
  geometry of a physically changed image.

Existing trajectory labels may be reused for optical changes if the task remains
observable. If moving the physical camera/bracket changes collision geometry,
revalidate or regenerate expert/recovery trajectories. Reject impossible or
unobservable configurations instead of supplying conflicting imitation targets.

Use the real hardware's plausible range to set training bounds, while holding
out camera configurations and their combinations. Preserve the existing scene
split; different renders of one scene/trajectory must stay in the same split.

### C. Explicit rays and camera-frame metric depth

If camera changes expose systematic metric errors, replace direct base-XYZ
regression with a validity-aware positive camera-Z depth head and an analytic
transform at each output pixel:

```text
r(u,v) = inverse(K_grid) @ [u,v,1]
p_C(u,v) = z_hat(u,v) * r(u,v)
p_B(u,v) = R_B_C @ p_C(u,v) + t_B_C
```

Use K transformed to that grid's actual sampling convention. This equation uses
Z-depth and rays with third coordinate 1; a Euclidean range predictor instead
needs normalized rays. Supply ray embeddings to the decoder so predicted depth
can depend on lens geometry before the dense head. Apply the existing base-XYZ
normalization only after the analytic transformation, retaining C1/C2/C3's metric
interface. Goal-ray features can similarly use analytic projection, with learned
visibility where depth is uncertain.

This guarantees consistency of the **coordinate transformation**, not correct
monocular depth or scale on unseen objects. Camera/appearance diversity and
metric simulation supervision are still needed. This is a new model version
requiring simulation training; it is not a checkpoint-compatible preprocessing
fix. Higher internal image resolution and denser obstacle points should be
separate ablations with end-to-end latency measurements.

### D. Calibrated uncertainty and execution coverage

Train/evaluate error prediction across camera and appearance domains, report
point-error coverage, and include measured calibration/tracking uncertainty in
the execution budget. Add invalid-point masks, memory consistency checks,
terminal approach constraints, and full-body path feasibility. When available,
RGB-D can be an additional measured-sensor comparison or protective monitor,
but it changes the RGB-only deployment claim and should be reported separately.

## 6. Benchmark and validation plan

Use the fixed checkpoint first to locate failure causes, then compare proposed
simulation-trained variants under the same perturbations. The following levels
are starting experiments; select final ranges using the intended hardware.

| Axis | Proposed evaluation cases | What must remain consistent |
|---|---|---|
| Pure sampling resolution | W×H = 160×120, 320×240, 640×480, 1280×960 at the same physical view | Correct K/pixel-center mapping; keep model input 84×112 to isolate acquisition resolution |
| Focal length / principal point | Focal ratios 0.75, 1.0, 1.25, 1.5; principal offsets 0, ±5%, ±10% of image dimensions | Re-render rays with the changed K; report actual FOV and include the measured hardware K |
| Aspect ratio / crop | Native 4:3, 16:9 and square crops | Record preserved FOV, crop transform and invalid pixels; avoid conflating crop with resolution |
| Physical camera mount | Translation magnitudes 1, 3, 5 cm; rotations 5°, 10°, 20° about multiple axes; actual intended mount | Render true poses; update body geometry if hardware moves; validate mount/scene feasibility |
| Calibration error | Estimated translation errors 2, 5, 10 mm; rotation errors 0.5°, 1°, 2°; focal errors 1%, 3%, 5% | Keep true render/metric ground truth fixed; perturb only estimates supplied to policy |
| Sensor/appearance | Distortion and rectification, blur, noise, compression, illumination/material/background changes | Track validity; consistent temporal appearance except intended sensor variation |
| Timing/control | Delay 0, 50, 100, 200 ms; dropped frames; measured tracking/gain variations; execution 15/5/1 after code changes | Advance simulation during latency, log observation timestamps and executed timing |
| Scene/embodiment | New obstacle sizes/materials, thin objects, table heights, camera brackets, near-goal obstructions | Separate changes in task geometry from changes in rendering; evaluate arm and tool contacts |

Begin with one-factor sweeps, then combined shifts. Use paired scene/start/goal
conditions and multiple domain seeds; report scene-level uncertainty intervals
so repeated renders are not treated as independent scenes. Tune on validation
only, lock design/checkpoint choices, then run the complete test partition.
Keep the original fixed-camera test as a regression baseline.

Measure:

1. Collision-free XYZ success, XYZ-plus-orientation success, route breakdown,
   and collision body/location, with confidence intervals. Define real contact
   events from available measurements; simulator impulse threshold `1e-7 Ns`
   is not directly a real contact-sensor threshold.
2. Geometry error in metres (median/tails), depth scale bias, obstacle recall,
   valid-point fraction, self-filter mistakes, and temporal map consistency.
   Compare against independent reference geometry for real offline measurements;
   this reference need not be available to the deployed policy.
3. Empirical 90% point-error coverage by camera/domain and critical obstacle
   region, along with frequency of padding saturation and missing obstacles.
4. Final FK residual, joint derivative violations, actual path clearance,
   target tracking error, and stops/replans caused by the execution monitor.
5. End-to-end latency p50/p95/p99, stale/drop rates, actual observation frequency,
   memory/compute usage, and missed command deadlines on target hardware.

Existing tests check geometry modules and control contracts, but the main-model
test replaces real perception with `TinyPerception`. They do not demonstrate
the learned model's camera transfer. Before implementing deployment, add focused
contract checks for projection/resize/crop consistency, transform direction and
timestamp alignment, checkpoint geometry overrides, anchor-relative actions,
and failure behavior. Actual transfer requires learned-checkpoint rollouts.

For real trials, progress from recorded-frame checks and open-space tracking to
static, well-separated obstacles and then the intended task distribution.
Freeze weights before the final evaluation. Declare any use of real data for
model selection, uncertainty fitting, or fine-tuning; calibration alone is
distinct from real-task training. Choose acceptance thresholds from the actual
task's tolerance and contact requirements rather than inventing a transferable
success-rate target from simulation alone.

## 7. Implementation map and verification performed

| Work item | Main locations |
|---|---|
| Shared image/K transforms and variable-size batching | `src/tsn/data/navigation.py`, `loaders.py`, `models/perception.py`, `features/maps.py`; new reusable camera preprocessing utility |
| Camera-domain rendering, truthful calibration metadata and delayed execution | `src/tsn/simulation/episode.py`, `benchmark_scene.py`, `evaluation/closed_loop.py`; new data-generation/experiment entry point |
| Explicit rays, depth and validity predictions | `src/tsn/models/perception.py`, `policy.py`, `training/runner.py`, `training/losses.py` |
| Deployable robot geometry independent of saved buffers | `src/tsn/models/kinematics.py`, `c2_embodiment.py`, `policy.py`, checkpoint/config schema |
| Memory invalidation, uncertainty and feasibility rejection | `src/tsn/models/c1_memory.py`, `c2_embodiment.py`, `c3_clearance.py`, `route.py` |
| Real sensors, clock synchronization and robot execution | New hardware observation adapter and real-time trajectory executor; reuse policy/state interfaces |

This analysis was checked against current source, all 1,000 episode camera
metadata records, and archived full-partition result/timing JSON files. The
resize calculation ran the current `GeometryMaps` on CPU in the existing project
Docker image with read-only mounts and networking disabled. It checked a
constant-depth plane and did not load or evaluate learned checkpoints. No new
training, camera-shift rollout, hardware trial, or runtime benchmark was
performed. The proposed improvements therefore remain hypotheses to test, while
the interface limitations, fixed-camera data coverage, and recorded baseline
results above are directly supported by the inspected implementation/data.
