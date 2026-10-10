# Learned persistent scene geometry and embodiment

This document records the earlier soft-adapter study. The requested replacement
architecture and encoder-only freezing are described in
[attention_replacement.md](attention_replacement.md).

The update introduces independently switchable learning modules into C1 and C2.
The original configuration and historical checkpoints still load strictly. New
checkpoint configurations enable the adapters through `model.learned_geometry`.
Both adapters start with zero output heads, reproducing the original geometry
weights and complete control behavior before training.

## C1: learned point retention and relevance

`PointRetention` in `src/tsn/models/c1_memory.py` embeds each observed point using
its position relative to the current TCP and goal, reconstruction uncertainty,
view support, scatter, age and quantized point redundancy. Eight latent queries
summarize the scene, then each point attends to those summaries and the measured
16-dimensional robot/goal state. Attention cost grows linearly with cloud size.

The learned score chooses an actual observed representative within each voxel
and modifies capacity eviction priority. It also supplies a bounded point weight
to C3's contact response. Learned ranking uses actual observations; it does not
invent or displace surface coordinates. Reobservations update support/scatter;
the anchor and its creation uncertainty remain fixed. Four recent clouds and
old-only anchors remain causal, with a maximum of 1,600 persistent anchors.

`PersistentGeometry` is now an `nn.Module` so its selector is optimized and saved.
Points, recent clouds, timestamps and task context remain episode state, absent
from `state_dict`; `reset_episode()` clears all of them. Learned memory requires
the measured state at every update. A score's training supervision combines
40 mm reconstruction reliability from training depth with candidate navigation
regret. Discrete voxel selection and eviction are scored by the learned network;
their ranking operations themselves are not differentiable.

## C2: learned moving robot point representation

`LearnedEmbodimentGeometry` in `src/tsn/models/c2_embodiment.py` encodes metric
surface probes for the TCP, palm, fingers, wrist and camera. Measured finger
positions and the calibrated mount update the point cloud. Seven joint-origin
probes from URDF forward kinematics describe current arm posture. The arm probes
provide learned context; they do not add articulated arm collision volumes.

Robot probes use self-attention and cross-attention to predicted scene points,
conditioned on measured state and the goal relative to each proposed body pose.
The output learns region-, time- and candidate-specific contact relevance.
Scene attention uses up to 256 visible probes; the analytic contact response
still evaluates every valid scene point. Physical distance and self-surface
filtering retain the measured URDF geometry. No learned deformation replaces
the known body surface.

Both adapters use bounded weights `1 + strength * (2 * sigmoid(logit) - 1)`.
Strength lies in `[0,1]`, and zero strength restores the original risk weights.
C1's discrete representative selection is also bypassed at zero strength.
C3 receives the learned point weights and robot context through the same
`costs`/`refine` interface in training and deployment. The 30-step proposal,
15-step execution, IK, uncertainty padding and trust head are retained.

## Independent experiment

The experiment is stored at
`/home/datasets_v2/chenmao/experiments/learned_geometry_20261009`.
Its parent is the full variable-camera model in
`gtsn_cam_var_20261008/full/training/best.pt`. All existing parent tensors,
including the previously trained Pi3 encoder and decoder, route and C3 heads,
are frozen. Only the newly introduced C1 or C2 adapter is optimized in each run.

The cache samples 2,048 causal training examples with seed 20261009, the existing
65/35 expert/perturbation mixture and 20/40/40 route mixture. It contains only
the 800 training episodes and their independent perturbations. Earlier views
remain within their episode; perturbations start fresh history. Training depth
and expert futures provide targets, and neither is read by deployed policies.

Each adapter trains for four epochs with AdamW at 1e-4 and gradient clipping at
1. The adapter objective combines candidate navigation cross-entropy, C1 point
reliability (weight 0.1), and a C1 point-weight deviation prior (weight 0.01).
C2 uses navigation cross-entropy and optimizer weight decay; both adapters have
bounded residual strengths.
Full validation evaluates strengths 0.05, 0.2 and 0.5 independently. Highest
collision-free XYZ success selects strength; ties choose the smaller strength.
A nominated adapter must match or exceed the parent's validation success count.
The general main-policy trainer rejects geometry-only checkpoint selection by
unchanged route RMSE; use the cached adapter trainer for this frozen experiment.

When both independent candidates pass validation, their separately trained
weights are combined without retraining. Validation compares half and full
selected strengths using the same success/tie rule. The nomination and hashes
are saved before any updated-policy test rollout. Test acceptance additionally
requires matching or exceeding the parent's full test success count. This
checks observed performance on the existing partitions; it does not establish
population noninferiority across new scenes or training seeds.

Every rollout uses the original 10 mm collision-free XYZ criterion, with
orientation success reported separately. Simulator processes are refreshed
every five episodes. Source hashes, parent tensor equality, partition membership,
finite trajectories, empty expert reference indices and fixed 15-step execution
are audited. Selected checkpoints are standalone; redundant complete trial
checkpoints are removed, retaining compact adapter weights and selection evidence.

To reproduce the complete study in a new directory:

```bash
python3 scripts/docker_geometry_study.py \
  --root /home/datasets_v2/chenmao/experiments/learned_geometry_new \
  --gpus 0,1,3,6
```

The launcher uses the existing `gtsn-experiment:20261009-clean` image, disables
container networking, mounts code and old data read-only, and writes only the
new experiment root. No host packages or unrelated images are modified.

Run the complete regression suite in Docker:

```bash
python3 scripts/docker_run.py --cpu test
```

Results and selected checkpoint paths are recorded in the experiment's
`RESULTS.md`, `summary.json` and `test_freeze.json`.

## Completed results — 2026-10-09

| Model | Selected strength | Validation /100 | Test /100 |
|---|---|---:|---:|
| Frozen parent | Original geometry | 68 | 69 |
| Learned C1 only | C1 0.05 | 69 | 71 |
| Learned C2 only | C2 0.05 | 68 | 71 |
| Learned C1 and C2 | C1 0.025, C2 0.025 | 68 | 71 |

All three selected updates meet the requested **no observed primary success
drop** on both full partitions. The combined model adds 19,618 learned
parameters. Both independent adapters have four completed training epochs, and
every existing parent tensor remains unchanged. The study completes 800
validation candidate rollouts, 300 selected-model test rollouts and five exact
parent replay checks. The final repository regression suite passes 53 tests in
Docker. The parent and learned models have the same 10% test orientation success.

Training-cache diagnostics show nonconstant response weights: across 32 examples,
C1 weights range from 0.9556 to 1.0308 (standard deviation 0.0243), and C2 weights
from 0.9536 to 0.9755 (standard deviation 0.0021), at their independently selected
0.05 strengths. C2 primarily learns a modest risk attenuation, with smaller
context-dependent differences. There is no matched uniform-rescaling control,
so these results do not isolate a navigation benefit from attention conditioning.
The combined model gains two test episodes and loses none; its paired,
route-stratified 95% bootstrap interval is [0,+5] percentage points. Validation
gains one and loses one, with interval [-3,+3]. These scene-resampling intervals
do not measure variation across training seeds.

The first supervisor exceeded its 28 GB RAM limit during combined validation.
Recovery reused 35 complete trajectories and continued with two GPU workers,
unchanged frozen numerical source and fixed weights. The launcher now reserves
40 GB, releases checkpoint tensors before evaluation, and provides
`scripts/resume_geometry_study.py` for interrupted studies. Redundant trial
checkpoints were removed; selected standalone policies, compact adapters,
causal training cache, selection evidence and trajectories remain available.

The combined policy is
`/home/datasets_v2/chenmao/experiments/learned_geometry_20261009/combined/selected.pt`.
Evaluate it into a fresh output directory:

```bash
python3 scripts/docker_run.py --gpu 0 evaluate \
  --checkpoint /home/datasets_v2/chenmao/experiments/learned_geometry_20261009/combined/selected.pt \
  --partition test --no-render-videos \
  --output-dir /home/datasets_v2/chenmao/experiments/learned_geometry_replay_new
```

For an independent variant, replace `combined/selected.pt` with `c1/selected.pt`
or `c2/selected.pt`. The checkpoint enables its learned modules automatically.
