# Frozen-Pi3 model simplification — 2026-10-03

Experiment root: `/run/user/1016/experiments/gtsn_simplification_20261003`.
Reference: epoch 15 from `gtsn_consensus_20261003_fresh`, preserved in
`gtsn_consensus_20261003_fresh_interim_20261003_141728/train/best.pt`.
SHA-256: `5f5e5c2b920a98f4d464a673e76eea372a15cd93e6ac748a235cddf2d5b696af`.
The reference achieved 85/100 validation successes and 78/100 test successes.

## Architectural analysis

The reference trains Pi3, its dense map/action heads, and four Cartesian route
heads jointly. Its deployed controller averages only the state and memory route
predictions. The separate visual and uncertainty route heads are consequently
unnecessary at inference and are excluded from every compact checkpoint.

The memory route projects visual tokens, predicts geometry means and variances,
gates visual features by learned reliability, encodes each observation, and
attends over up to four observations. A separate state route predicts another
trajectory; their average passes through the existing Cartesian controller.
The controller's adaptive replanning and temporal blending are disabled in the
reference configuration, so the compact controller omits those mechanisms.

We tested removing the learned uncertainty distribution and gate, replacing
temporal attention with a masked average, removing history, removing the separate
state route, and jointly removing history and uncertainty. The retained Pi3
action prediction still supplies wrist orientation and inverse-kinematics
initialization; its map/action heads are not unused and were preserved. Robot
state and goal inputs also remain when the separate state route is removed.

## Protocol

- Use the immutable epoch-15 checkpoint, not the changing `fresh/train/best.pt`.
- Freeze the entire Pi3/map/action backbone and cache its outputs anew. Cache
  57,335 expert and 2,997 recovery training samples and 14,420 validation samples;
  do not cache test observations. Preserve the fixed 800/100/100 episode split.
- Initialize surviving route modules from epoch 15. Train all six variants for
  five epochs using identical sampled indices: 65% expert / 35% recovery and
  direct/over/side probabilities 0.2/0.4/0.4. Use AdamW, learning rate 3e-5,
  weight decay 1e-4, cosine decay, batch size 256, seed 20261002.
- Preserve separate waypoint Huber supervision per surviving branch and the
  geometry NLL for branches retaining a distribution. Select each checkpoint by
  validation waypoint RMSE, including initialization as epoch 0. The backbone is
  absent from the optimizer; training operates on immutable cached features.
- Evaluate every selected checkpoint on all 100 validation episodes. A candidate
  qualifies only if its success loss is strictly below five percentage points
  against both the original 85% validation baseline and the matched control.
- Choose the qualifying architecture with the fewest route-head parameters.
  Record this choice before test evaluation. Evaluate only that candidate and
  the matched control on all 100 test episodes; use test as a confirmation gate,
  not to rank the other candidates or choose their training epochs.
- Keep the original physics, RGB observations, goal servo (0.08 m), 15-step
  execution horizon, 400-step limit, collision termination, and 0.01 m XYZ goal
  tolerance. No expert future or depth labels enter the deployed policy.
- Run numerical work in the existing project Docker image, restricted to GPU 4
  (A800 80 GB), without modifying other images or the ongoing baseline run.

## Validation results

All rows use 100 validation episodes. Parameter counts cover active route heads,
not the frozen backbone. RMSE is validation waypoint error in millimeters.

| Variant | Removed/replaced | Selected epoch | Route parameters | RMSE (mm) | Success | Collision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original epoch 15 | Reference state + memory | 15 | 868,042 | 14.700 | 85% | 15% |
| Matched control | No active route module | 2 | 868,042 | 14.680 | 84% | 15% |
| No uncertainty | Distribution and reliability gate | 2 | 867,652 | 14.630 | 84% | 16% |
| Mean memory | Attention key/scores → masked average | 2 | 802,250 | 15.327 | 81% | 18% |
| Current only | History and temporal attention | 2 | 802,250 | 14.647 | 83% | 17% |
| Single memory | Separate state route and ensemble | 0 | 691,768 | 15.821 | 82% | 18% |
| Deterministic current | History, attention, distribution, gate | 2 | 801,860 | 14.639 | 84% | 16% |

All candidates passed the validation threshold. The predefined size-first rule
selected **single memory**: 176,274 fewer active route parameters (20.3%), with
a three-point loss against the original validation baseline and a two-point loss
against the matched control. Its five training epochs did not improve on its
initialization checkpoint, so epoch 0 was selected. The other candidates remain
documented alternatives; their test performance is not established by this run.

## Test confirmation and deployment

**Retain the single-memory model.** It achieved **78/100 test successes**, matching
the original epoch-15 baseline and trailing the equally fine-tuned control by
two percentage points. It passes the strict less-than-five-point criterion on
both validation and test. The experiment completed 600 validation and 200 test
rollouts, in addition to export replay checks.

| Model | Test success | Test collision | XYZ + orientation success |
| --- | ---: | ---: | ---: |
| Original epoch 15 | 78% | 22% | 28% |
| Matched control | 80% | 20% | 28% |
| Retained single memory | **78%** | **22%** | **28%** |

| Test route | Episodes | Original | Matched control | Retained |
| --- | ---: | ---: | ---: | ---: |
| Direct | 20 | 100% | 100% | 100% |
| Over | 40 | 87.5% | 87.5% | 92.5% |
| Side | 40 | 57.5% | 62.5% | 52.5% |

The identical overall success does not mean identical behavior: the retained
model improves the over-obstacle result and worsens the side-route result.
The acceptance decision uses the requested overall success metric.

The exported graph has one memory route head. It excludes the separate state,
visual, and uncertainty route heads, the consensus average/disagreement wrapper,
adaptive replanning, and temporal action blending. It retains geometry inputs,
the memory head's uncertainty distribution/gate, four-observation attention,
and the original backbone and kinematic controller. The independently passing
ablations were not all combined: their interactions were not tested.

The standalone export passed exact action-output comparisons across real
observations, accumulated history, and an episode reset (maximum difference
zero). Replaying `episode_009`, `episode_201`, and `episode_600` with the exported
controller reproduced the evaluated model's outcomes, control steps, qpos, TCP
transforms, and predicted joint targets **bit-for-bit**. The training audit also
confirmed matched sampling, frozen backbone, unchanged baseline hash, and
unchanged experiment source snapshot.

| Size / isolated timing | Original | Retained |
| --- | ---: | ---: |
| Active route parameters | 868,042 | 691,768 |
| Active policy parameters, including Pi3 | 352,219,956 | 352,043,682 |
| Training graph parameters, including unused comparison heads | 353,603,102 | 352,043,682 |
| Route-head median latency, batch 1, 100 repetitions | 1.991 ms | 1.328 ms |

This is a 20.3% route-head parameter reduction, a 0.050% reduction against the
original active policy, and a 0.441% reduction against its full training graph.
The isolated route timing includes removing the consensus wrapper; it excludes
Pi3 and inverse kinematics and is not an end-to-end latency measurement.
The standalone checkpoint is 1,408,525,654 bytes.

The paired test success difference is 0 points against epoch 15, with a 95%
paired bootstrap interval of **[-6, +6] points** (10,000 resamples; five gained
and five lost episodes). Against the matched control it is -2 points, interval
**[-8, +4]**. Thus the observed threshold passes, but this sample does not prove
statistical noninferiority within five points.

Verification also included the repository's GPU test suite (34 passed, one
two-GPU integration test skipped), the updated CPU suite (34 passed, two GPU
tests skipped), and all five final compact-model tests. An initial export audit
failed in its latency harness because a newly created consensus weight buffer
was on CPU; moving that benchmark module to CUDA fixed the harness. Its failed
snapshot/log are preserved with `.failed1` suffixes. No training or rollout
results were replaced by that retry.

## Reproduction and artifacts

Launch a fresh run from the repository root (the destination must not exist):

```bash
python3 scripts/run_simplification_docker.py \
  --root /run/user/1016/experiments/gtsn_simplification_reproduction --gpu 4
```

After it completes, verify its standalone deployment:

```bash
python3 scripts/verify_simplification_docker.py \
  --root /run/user/1016/experiments/gtsn_simplification_reproduction
```

These host launchers use only the Python standard library to snapshot files and
invoke Docker. Dependencies and numerical execution remain inside Docker.

Key artifacts in the experiment root:

- `protocol.json`, `launch.json`, `source/`: immutable experiment protocol,
  container command/image ID, and training/evaluation source snapshot.
- `baseline.pt`, `cache/{train,validation}/`: fixed reference and fresh features.
- `heads/*/{initial.pt,best.pt,epochs.json,summary.json}`: all variants, selected
  epochs, losses, and identical per-epoch sampling hashes.
- `rollouts/{validation,test}/*/`: per-episode metrics, joint/TCP trajectories,
  aggregate results, and completion records.
- `selection.json`, `results.json`, `audit.json`: validation choice, final
  confirmation, and source/checkpoint/sampling integrity checks.
- `simplified.pt`: standalone backbone plus selected route weights and config.
- `deployment_source/`, `deployment_provenance.json`, `export_audit.json`,
  `export_smoke/`: deployment snapshot, hashes, prediction equivalence checks,
  paired success comparisons, and replayed trajectories.

Load the exported model inside the project Docker environment:

```python
from tsn.models.simplified_consensus import load_compact_policy
policy, maps, checkpoint = load_compact_policy(
    '/run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt',
    device='cuda',
)
```

`scripts/evaluate_simplified.py` provides closed-loop evaluation of this format
with the saved test split. The legacy generic evaluation CLI expects the earlier
checkpoint schema; use this dedicated entry point for compact checkpoints.

## Interpretation limits

The five-point rule is the requested empirical acceptance criterion, not a
statistical proof of equivalence. There are only 100 episodes per partition and
one training seed. Paired episode bootstrap intervals are reported in the export
audit. This study supports simplification when reusing the trained frozen Pi3;
it does not establish that the removed auxiliary route losses are unnecessary
when training Pi3 from scratch. Pi3 dominates total parameters, so route-head
reductions should not be presented as equal percentage reductions of the whole
model. Concurrent rollout timings are not used to claim end-to-end speedup.
