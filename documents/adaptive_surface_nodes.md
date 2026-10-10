# 1,024 scene points and 64 adaptive robot surface nodes — 2026-10-10

The new joint model uses up to **1,024 queried scene points** and **64 robot
surface nodes per sampled candidate pose**. Both C1 and C2 keep four attention
blocks with residual connections and FFNs, at width 64. The two fresh training
runs initialize only the Pi3 image encoder from official weights and train all
parameters, including the encoder. Previous experiment checkpoints and module
warmup weights remain excluded.

## Scene geometry

Perception now samples a 32×32 grid, producing 1,024 predicted XYZ points per
RGB observation. The error head and depth-supervised targets use the same grid.
This increases the spatial sampling of the predicted map; it does not increase
the RGB encoder's 84×112 input resolution.

C1 stores up to 4,096 persistent anchors and the latest four filtered clouds.
Its learned scores choose at most 1,024 actual observations from this pool.
The number can be smaller when fewer valid observations exist. C2 cross-attention
receives the complete queried scene, with a 1,024-point capacity. It does not
apply the older 128-point cross-attention reduction.

## Robot geometry and learned adaptation

[robot_surface.py](../src/tsn/models/robot_surface.py) builds a deterministic pool
of 2,048 points on URDF collision triangles. The pool covers all seven moving arm
links, the palm, both fingers and the calibrated camera housing. Samples are
distributed by surface area, with minimum dense-pool coverage of each link.
There is no fixed allocation of the 64 representation nodes among links.

Measured arm/finger state drives URDF forward kinematics. At each of the five
sampled poses along each clearance candidate, the module runs the same 12-step
bounded IK procedure used by execution, then moves the full collision surfaces
with FK. Arm surfaces follow the candidate joint configurations. Camera samples
use the episode's calibrated TCP-to-camera transform.

[surface_embodiment.py](../src/tsn/models/surface_embodiment.py) scores all surface
samples with a neural selector. Inputs include the sample position and normal,
goal, measured state, nearest scene clearance, uncertainty, trajectory progress
and finger opening. A learned link embedding provides body-part identity. Hard
ranking selects 64 unique pool indices per pose. Node locations stay on the
physical surface, and their selection adapts to the current scene and task.
Joint origins are not representation nodes.

The selected node features include the selector's probability, allowing risk
losses to train its scores in addition to direct selection supervision. Four
self/cross-attention blocks with FFNs encode the 64 nodes against scene points.
A learned readout pools neural distance responses over 64×5 node instances per
candidate. Deployed risk uses the network output without blending in an analytic
priority or risk score.

## Training and physical supervision

Selection supervision favors surface samples relevant to scene clearance. A
soft coverage regularizer discourages concentrating all selection probability
on one body part; it does not prescribe inference node counts per link. The
`nodes` loss has weight 0.02. Other loss weights and optimizer settings match
the previous fresh joint experiments.

Risk supervision now includes the complete arm and tool collision geometry.
Training-only convex collision fields provide a maximum contact response over
scene points, averaged across the five candidate poses. A synthetic distance
calibration loss learns the local neural response. Self filtering also uses the
full measured robot geometry. The convex fields and sparse surface samples
remain approximations; the representation does not certify collision-free
motion between sampled poses.

Hard ranking has no derivative through index selection. The selector learns
from its explicit relevance/coverage loss and the selected score features in
the risk network. Depth and expert future states remain training-only labels.
Deployment uses RGB, measured state, goal and calibration.

## Experiments

Both runs use 30 epochs, 8,192 training draws per epoch, global batch 32, local
batch eight, AdamW learning rate 1e-4, weight decay 1e-4 and gradient clipping
at 1. The 800/100/100 benchmark split and 65/35 expert/independent-perturbation
sampling are unchanged. The benchmark seed remains 20261001.

| Seed | Physical GPUs | New experiment directory | Prior 256-scene / 13-robot validation, test |
|---|---|---|---|
| 20261010 | 0–3 | `/home/datasets_v2/chenmao/experiments/attention_surface1024_robot64_seed20261010` | 63/100, 70/100 |
| 20261011 | 4–7 | `/home/datasets_v2/chenmao/experiments/attention_surface1024_robot64_seed20261011` | 60/100, 67/100 |

This comparison changes scene sampling, query/memory capacity, the robot
representation, full-arm self filtering and physical risk supervision. It tests
the requested larger surface-node architecture; it does not isolate node count
alone. The new results are pending and the model is not yet accepted against the
no observed performance drop target.

All 73 regression tests pass inside Docker, including actual surface membership,
FK motion, neural hard selection, full-arm contact/self-filter labels, the
1,024-point scene query, gradients in the selector and attention blocks, full
joint training and standalone checkpoint loading. Each detached pipeline repeats
these checks, runs a separate real distributed smoke test, then trains from fresh
initialization and evaluates both complete 100-episode partitions. Validation
waypoint RMSE selects the checkpoint before test evaluation. `status.json`,
`training/initialization.json`, gradient audits, model audits and `RESULTS.md`
record each stage.

```bash
python3 scripts/docker_fresh_joint.py --config configs/attention_surface_fresh.json \
  --root /home/datasets_v2/chenmao/experiments/attention_surface1024_robot64_seed20261010 \
  --gpus 0,1,2,3 --seed 20261010

python3 scripts/docker_fresh_joint.py --config configs/attention_surface_fresh.json \
  --root /home/datasets_v2/chenmao/experiments/attention_surface1024_robot64_seed20261011 \
  --gpus 4,5,6,7 --seed 20261011
```
