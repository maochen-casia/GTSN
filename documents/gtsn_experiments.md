# GTSN design iterations and experiments

**Completed result:** the validation-selected hybrid policy achieves **72% test
success**, versus 7% for the first pilot and 54% for Pi3 with the same terminal
servo. Its gain over that matched controller baseline is **18 percentage points
(paired bootstrap 95% interval 8–29; exact McNemar p=0.0021)**. All 120 new
training epochs, 1,140 validation rollouts, and 600 test rollouts are complete.
The artifact audit and all 25 regression tests pass. The uncertainty ablation
scores 79% on test but was not selected by validation; the selected model remains
unchanged. The results establish a substantial action/control improvement,
while uncertainty and memory contributions remain unconfirmed.

## Revision 2: action representation and terminal convergence

The continuation is stored in
`/run/user/1016/experiments/gtsn_cartesian_20261003`. All numerical work uses
the existing `gtsn-pi3:20261002` Docker image, ID
`sha256:5716cc81a619a9d00e6d08edecedff42a5b2b57fe10c8ff22835920cb480e009`.
The first pilot's files and checkpoints are preserved.

A calibrated kinematic goal servo exposed a major terminal-convergence
failure. On the same stratified 20-episode validation screen, Pi3 increased
from 2 to 13 successes when the servo replaced its joint predictions within
8 cm of the goal. The actual success threshold remains 1 cm, and collisions
and the 400-step time limit are unchanged. This is a controller improvement,
not evidence for an uncertainty or memory contribution.

Four new Cartesian decoders were trained for 30 matched epochs. Each allocates
691,768 parameters and predicts six TCP waypoints at 5-step intervals. Training
uses the first pilot's frozen features, 65% expert/35% perturbation sampling,
20/40/40 route weights, batch 256, AdamW at 3e-4, cosine decay, weight decay
1e-4, gradient clipping at 1, and Huber loss with delta 0.02 m. Geometry-aware
variants add 0.001 times Gaussian NLL. Training order hashes match across all
four conditions. Checkpoints use validation waypoint RMSE, not test outcomes.
The earlier joint-residual heads retain their original ten-epoch training;
cross-representation comparisons therefore do not have identical training
budgets. Within each family the ablations are matched.

| Cartesian decoder | Selected epoch | Validation waypoint RMSE (m) | Initial screen success |
|---|---:|---:|---:|
| State waypoints | 27 | 0.016702 | 30% |
| Current-view visual | 20 | 0.017284 | 35% |
| Uncertainty | 20 | 0.016911 | 40% |
| Causal memory | 20 | 0.016979 | 45% |

These initial decoders retained the current wrist orientation while tracking
positions. The memory variant reached only 31% success on full validation.
Using Pi3's predicted future wrist orientations and joint configurations for IK
initialization increased its screen success from 45% to 70%. All four Cartesian
heads were consequently evaluated on full validation with this common orientation
controller. The state-waypoint control still uses RGB for orientation.

The original joint-residual heads were also reevaluated with the common goal
servo. Full validation success is 59% for Pi3, 64% for state correction, 71% for
visual correction, 64% for uncertainty, and 67% for memory.

| Full validation condition | Success | Collision | Timeout |
|---|---:|---:|---:|
| Pi3 + goal servo | 59% | 38% | 3% |
| Joint state correction + servo | 64% | 35% | 1% |
| Joint visual correction + servo | 71% | 28% | 1% |
| Joint uncertainty + servo | 64% | 35% | 1% |
| Joint memory + servo | 67% | 31% | 2% |
| Cartesian memory, fixed wrist orientation + servo | 31% | 67% | 2% |
| Cartesian state, Pi3 orientation + servo **(selected)** | **80%** | **20%** | **0%** |
| Cartesian visual, Pi3 orientation + servo | 75% | 23% | 2% |
| Cartesian uncertainty, Pi3 orientation + servo | 74% | 24% | 2% |
| Cartesian memory, Pi3 orientation + servo | 79% | 18% | 3% |

Each row uses all 100 fixed validation episodes. The selected composition is
`state_e15_s0.08_obaseline`, frozen in `selection.json` before any revision-2 test
run. The predeclared six-condition test comparison is complete. Selection
was not changed using those test results. In the selected model the new
waypoint branch is state-conditioned, while its orientation and IK initialization
remain RGB-dependent; calling the entire policy state-only would be incorrect.

### Final fixed-test comparison

| Policy | Success | Collision | Timeout | Direct / over / side success |
|---|---:|---:|---:|---|
| Historical Pi3, no servo | 6% | 58% | 36% | historical reference |
| Historical selected pilot, no servo | 7% | 50% | 43% | 5% / 15% / 0% |
| Pi3 + servo | 54% | 42% | 4% | 80% / 70% / 25% |
| Joint visual correction + servo | 62% | 34% | 4% | 90% / 70% / 40% |
| Cartesian state + Pi3 orientation + servo **(selected)** | **72%** | **28%** | **0%** | **100% / 70% / 60%** |
| Cartesian visual + Pi3 orientation + servo | 75% | 24% | 1% | 100% / 90% / 47.5% |
| Cartesian uncertainty + Pi3 orientation + servo | 79% | 20% | 1% | 100% / 92.5% / 55% |
| Cartesian memory + Pi3 orientation + servo | 76% | 24% | 0% | 100% / 90% / 50% |

Every row contains the same 100 test episodes with 20/40/40 route counts.
The selected model's Wilson 95% success interval is 62.5–79.9%. It requires
132.09 control steps on average and has no timeouts. Simultaneous XYZ and
orientation success is only 22%; the reported 72% uses the benchmark's original
XYZ criterion, not full-pose success. Collisions remain substantial, especially
on side routes, so the policy is not a validated safe controller.

| Selected model minus reference | Success gain | Paired bootstrap 95% interval | Discordant wins / losses | Exact McNemar p |
|---|---:|---|---|---:|
| Original Pi3 | +66 pp | +57 to +75 pp | 66 / 0 | 2.71e-20 |
| First selected pilot | +65 pp | +55 to +75 pp | 67 / 2 | 8.19e-18 |
| Pi3 with the same servo | +18 pp | +8 to +29 pp | 25 / 7 | 0.00210 |
| Joint visual correction with servo | +10 pp | 0 to +20 pp | 18 / 8 | 0.0755 |

Intervals use 20,000 paired episode bootstrap resamples; exact McNemar tests
condition on discordant episode outcomes. These results support a gain over
the earlier pilot and the controller-matched Pi3 baseline. They do not establish
a significant gain over the stronger joint visual head at a 0.05 threshold.

The uncertainty ablation's 79% test score is exploratory and does not replace
the frozen selection. Its +4 pp over Cartesian visual has interval -1 to +10 pp
and p=0.289. Memory is -3 pp relative to uncertainty, with interval -7 to 0 pp
and p=0.25. Thus neither uncertainty nor persistent memory has a statistically
supported incremental gain in this experiment. Further tuning on these test
outcomes would require a fresh confirmation set.

The final artifact audit covers **23 conditions and 1,740 rollout episodes**:
seven 20-episode screens, ten 100-episode validation conditions, and six
100-episode test conditions. It verifies disjoint fixed splits, matched training
sampling, finite trajectories, empty expert-progress inputs, unchanged success
and collision rules, and unchanged numerical source throughout every test run.
All 60 repeated screen/full-validation episodes match in outcome, control-step
count, and joint trajectory to 1e-5 tolerance. `container_runs.json` and `logs/`
preserve the project containers' commands, timestamps, and outputs.

Figures: `figures/closed_loop.{png,pdf}`, `figures/test_routes.{png,pdf}`, and
`figures/cartesian_training.{png,pdf}` under the revision-2 experiment root.

The new uncertainty decoders improve pooled-point RMSE to about 0.0475 in
normalized coordinates but retain only 87.2% marginal coverage for nominal 95%
intervals. Standardized squared error is about 2.02 and variance/error Spearman
correlation about 0.638. Thus they rank errors usefully while remaining
overconfident. These diagnostics alone do not establish navigation benefit.

Kinematics validation compared recorded TCP positions from ten validation
episodes, with maximum error 6.94e-7 m. Numerical Jacobian error was 6.05e-5;
a nearby IK target was reached within 1.51e-8 m. Decoder padding, visual-history
isolation, shared-servo behavior, predicted-orientation tracking, and IK limit
checks pass. The final full regression suite passes all **25 tests**, including
both GPU integration tests, with no skips.

### Reproduction and artifacts

The selected checkpoint composition, input contract, and weight checksums are
in `selected_policy.json`; selection is fixed in `selection.json`. Each head
has `best.pt`, `epochs.json`, and `results.json`. `decoder_diagnostics.json`
records waypoint errors and calibration. Rollout directories contain per-episode
metrics and complete state/target trajectories. Figures are emitted as PNG/PDF.
Source snapshots in `source/` and `provenance/` preserve the initial screen,
shared-servo comparison, orientation revision, and final test implementation.
Earlier validation completion hashes describe the files on disk at completion;
the corresponding snapshots preserve the implementation loaded by each run.
Final test records include source hashes at both start and completion.

Run from `/home/chenmao/GTSN`; the host launcher uses only Python's standard
library and executes numerical work inside the existing Docker image:

```bash
# Reproduce the selected policy into a fresh output directory.
python3 scripts/cartesian_docker.py --name gtsn-reproduce --gpu 0 \
  python scripts/run_cartesian.py rollout --mode state --orientation baseline \
  --partition test --output-root /run/user/1016/experiments/gtsn_reproduce

# Recompute reports and audit saved experiments without retraining.
python3 scripts/cartesian_docker.py --name gtsn-report \
  python scripts/report_cartesian.py
python3 scripts/cartesian_docker.py --name gtsn-audit \
  python scripts/audit_cartesian.py

# Full tests, including the two-GPU check.
python3 scripts/cartesian_docker.py --name gtsn-tests --gpu 0,1 \
  python -m unittest discover -s tests -v
```

For training reproduction use `run_cartesian.py train --mode state` (or visual,
uncertainty, memory), with `--root` set to a fresh experiment directory. It
reuses the frozen feature cache described above and defaults to the matched
30-epoch protocol. Retain that directory's checkpoints when evaluating it.

The renderer and frozen perception backbone are unchanged. The known rendering
appearance mismatch remains. These results isolate an action/control revision;
they do not validate end-to-end perception adaptation, an uncertainty-calibrated
safety guarantee, spatial occupancy fusion, or active viewpoint selection.
The strong state-waypoint control also cautions against claiming that new scene
memory is necessary for this benchmark. Test comparisons reuse the established
test split, so they are exploratory; publication claims need a fresh evaluation
set and additional training seeds.

## First pilot: frozen perception without terminal servo

This experiment implements the staged design in `gtsn_methodology.md`. Artifacts
are under `/run/user/1016/experiments/gtsn_pilot_20261002`. All four ten-epoch
training runs, 600 validation rollouts, and 100 test rollouts are complete.
The validation-selected state-correction control achieves **7% test success,
50% collisions, and 43% timeouts**. The proposed uncertainty, memory, and active
timing additions do not demonstrate a navigation improvement in this pilot.
The experiment and reporting containers exited successfully, and the final
audit passed. This is a completed negative/diagnostic study, not a successful
solution to the requested robust-navigation research problem.

## Protocol and implemented scope

The Pi3 epoch-7 checkpoint from the previous experiment remains frozen. Its
decoder tokens bypass the six-map bottleneck through a new 259,672-parameter
head. All variants allocate the same head; ablations disable some branches.
The initial correction is zero, so all variants start at the same baseline
prediction. The state-correction control retains that visual baseline and trains
only a state-dependent correction. It is not an entirely state-only policy.

The deterministic visual variant uses current tokens and predicted geometry.
The uncertainty variant adds pooled-point mean correction, diagonal variance
supervision, and confidence-weighted attention. The memory variant additionally
attends to up to four observed keyframes with base-frame positions, camera
poses, age, and novelty. Adaptive observation timing is evaluated separately
with the memory checkpoint, alongside fixed-fifteen and fixed-five controls.
No candidate-viewpoint search or information-gathering detour is implemented.

The fixed episode partitions retain direct/over/side counts 160/320/320 for
training and 20/40/40 each for validation and testing. Cached observations are
57,335 expert training frames at stride two, 2,997 RGB perturbation recovery
samples, 14,420 validation frames, and 14,606 test frames. Histories never cross
episode boundaries or use future observations. Recovery samples receive only
their actual current observation, without invented expert histories.

All four runs use seed 20261002, ten epochs, batch size 128, AdamW at 0.0003,
weight decay 0.0001, cosine decay, and gradient clipping at 1.0. Every epoch
samples 57,335 observations with 65% expert/35% recovery probabilities and
20%/40%/40% route probabilities within each source. Sample-order hashes support
the matched comparison. Validation action RMSE selects each checkpoint from
epochs zero through ten. Test data does not participate in training or selection.

Closed-loop validation evaluates each of four heads at fifteen-step execution,
plus memory with adaptive five/fifteen and fixed-five execution. All predict
30-step chunks. The final condition is chosen by validation success, then lower
collision rate, then lower validation action RMSE. It receives all 100 test
rollouts. The simulator uses 20 Hz control, 100 Hz physics, a 400-step limit,
collision termination, and a 1 cm XYZ success threshold. Orientation within
0.15 rad is reported separately. Depth and expert futures are training labels
only; the deployed policy receives RGB, joints, goal, and camera calibration.

## Training results

| Correction head | Selected epoch | Validation joint RMSE (rad) |
| --- | ---: | ---: |
| Frozen baseline, epoch zero | 0 | 0.064720 |
| State correction | 9 | 0.063610 |
| Current visual + deterministic geometry | 10 | 0.063807 |
| Current visual + uncertainty | 10 | 0.064015 |
| Uncertainty + memory | 9 | 0.064048 |

All learned corrections slightly reduce action RMSE, but the state-correction
control is best. The visual, uncertainty, and memory additions do not improve
this metric over that control. Open-loop action error alone does not establish
closed-loop navigation performance.

## Closed-loop validation and selection

Each row covers the same 100 validation episodes, including 20 direct, 40 over,
and 40 side routes. Timeouts are episodes with neither success nor collision.

| Head and execution schedule | Success | Collision | Timeout | Mean observations per episode |
| --- | ---: | ---: | ---: | ---: |
| State correction, fixed 15 | 10% | 50% | 40% | 17.21 |
| Current visual, fixed 15 | 9% | 47% | 44% | 18.09 |
| Current uncertainty, fixed 15 | 6% | 54% | 40% | 17.09 |
| Memory, fixed 15 | 7% | 47% | 46% | 17.69 |
| Memory, adaptive 5/15 | 7% | 51% | 42% | 26.63 |
| Memory, fixed 5 | 4% | 60% | 36% | 51.95 |

The predeclared rule selects **state correction with fixed-fifteen execution**.
`selection.json` was written before test evaluation. Thus the selected system
does not use the new geometry uncertainty or memory in its action correction.
The complete GTSN memory variant remains an evaluated research candidate, not
the validation winner.

Adaptive sensing uses 50.5% more observations than fixed-fifteen memory, with
the same success rate and four percentage points more collisions. Fixed-five
uses 2.94 times as many observations and succeeds less often. These outcomes do
not support the proposed active-timing heuristic. Shorter scheduling also changes
the elapsed-time extent of the four-keyframe history relative to training, so
the comparison does not isolate sensing frequency from history distribution.

Adaptive sensing requests five-step execution for 47.35% of its 2,663 chunks;
the fixed-fifteen and fixed-five memory conditions use 1,769 and 5,195
observations, respectively. The adaptive rule therefore does activate, rather
than merely reproducing the fixed schedule.

Paired, route-stratified episode bootstrapping uses 5,000 resamples. Success
changes and descriptive 95% intervals, in percentage points, are: visual versus
state correction -1 [-8, 6]; uncertainty versus visual -3 [-9, 3]; memory versus
uncertainty +1 [-3, 5]; adaptive versus fixed-fifteen memory 0 [-5, 5]; and
adaptive versus fixed-five memory +3 [-3, 10]. All these intervals include zero.
They preserve within-episode dependence, have no multiple-comparison correction,
and do not measure training-seed variation. `results.json` also contains paired
collision and open-loop RMSE comparisons.

Mean model inference latency ranges from 51.6 to 55.5 ms per observation across
conditions. This excludes rendering and is measured during concurrent simulator
experiments, not a dedicated hardware-latency benchmark. All scene simulation is
synchronous; these measurements do not establish a real-time moving-object
control guarantee.

## Held-out test of the selected condition

The selected checkpoint is `heads/state_correction/best.pt`, epoch nine, paired
with the unchanged Pi3 backbone checkpoint identified in `protocol.json`.

| Route | Episodes | Successes | Collisions | Timeouts | Successes also meeting orientation tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 20 | 1 | 1 | 18 | 0 |
| Over | 40 | 6 | 16 | 18 | 5 |
| Side | 40 | 0 | 33 | 7 | 0 |
| Overall | 100 | 7 | 50 | 43 | 5 |

The 7% success rate has a Wilson 95% interval of **3.4–13.7%**. Mean final XYZ
error is 0.13638 m, mean rollout length is 239.82 control steps, and mean
observation count is 16.42. Open-loop test joint RMSE is **0.064093 rad** over
14,606 frames. This control's constant variance output is untrained and is not
used for sensing or interpreted as a calibrated estimate.

The original frozen Pi3 policy achieved 6% success and 58% collisions on the
same test episodes. The paired success change is +1 percentage point, with a
95% bootstrap interval of [-5, 7]; it does not establish an improvement in task
completion. The collision change is -8 points, with interval [-15, -1],
conditional on these trained models and without a multiplicity adjustment.
Timeouts increase from 36% to 43%, so fewer collisions do not imply that the
navigation task is solved. Both models perform poorly in absolute terms.

The prior 74% privileged-map result uses measured geometry and expert-future
information unavailable to these policies. It remains an information-rich
reference, not an input-matched deployment comparison.

## Geometry uncertainty

The probabilistic heads supervise 4 by 4 pooled point coordinates, with three
axis-specific normalized coordinates per token. These errors must not be
compared directly with the previous pixel-level point-map distances. Pooling
can average different surfaces and conceal thin obstacles. The 230,720 validation
tokens are correlated within frames and episodes.

| Model | Pooled-coordinate RMSE | Marginal 95% coverage | Mean squared standardized error | Variance/error Spearman correlation |
| --- | ---: | ---: | ---: | ---: |
| Current uncertainty | 0.047935 | 87.44% | 2.011 | 0.627 |
| Memory | 0.048056 | 87.52% | 1.976 | 0.625 |

The uncorrected pooled-coordinate RMSE is 0.050017. Both heads modestly improve
mean geometry and rank errors usefully: the highest-uncertainty decile has
69.7 and 69.3 times the mean squared error of the lowest decile, respectively.
However, their confidence intervals under-cover. Their nominal 95% intervals
cover approximately 87.5% of coordinates, and the mean squared standardized
errors exceed the ideal value of one. This is evidence of useful error ranking
with underestimated uncertainty, not calibrated probability or demonstrated
collision-risk estimation. These diagnostics do not update model parameters.

`figures/uncertainty_calibration.{png,pdf}` plots equal-count variance bins
against observed squared error. `uncertainty_diagnostics.json` preserves all bin
values. Control heads' variance outputs are untrained constants and should not
be interpreted as uncertainty estimates.

## Matched-view appearance diagnostic

`figures/appearance_comparison.{png,pdf}` compares recorded and reconstructed
RGB at frame zero of the first validation episode in each route: episodes 006,
204, and 601. These examples were chosen by split order, not image quality.
Camera transforms agree to a maximum absolute matrix-entry difference below
1.2e-6. The images nevertheless differ in table textures, shading, and some
object appearance. Mean absolute RGB differences on the zero-to-one scale are
0.109, 0.167, and 0.180. Mean corresponding decoder-token cosine similarities
are 0.976, 0.962, and 0.970.

The memory head's singleton uncertainty scores change from 0.0183/0.0215/0.0260
on recorded RGB to 0.0185/0.0255/0.0271 on reconstructed RGB. All six scores are
below the train-calibrated adaptive threshold, 0.02869. These three examples
confirm an appearance mismatch but do not measure its causal contribution to
navigation errors or establish that uncertainty detects distribution shift.
`appearance_diagnostics.json` retains the measurements and selection rule.

## Execution and checks

The project image is `gtsn-pi3:20261002`; its immutable image identifier is recorded
in `protocol.json`. All numerical processes run in Docker on project GPUs 0–3,
with networking disabled. Code and benchmark mounts are read-only. The host
environment and unrelated images/containers are unchanged.

The initial caching process was restarted before head training to load an FP32
uncertainty-arithmetic fix consistently. Its completed training cache was reused:
perception computations were unchanged. `protocol_initial_cache.json` preserves
that stage's source hashes and revision explanation; `protocol.json` records
the final training/evaluation sources. The final regression run passed all
20 tests, including GPU Pi3 integration, inference isolation, causal-history
construction, episode reset, bounded streaming-memory equivalence, mixed
precision, and blocked direct action gradients through predicted variance.

Reproduce the complete experiment with a fresh name and output directory:

```bash
python3 scripts/run_gtsn_docker.py --name gtsn-reproduction \
  --root /run/user/1016/experiments/gtsn_reproduction
```

The launcher starts a detached Docker container. Inspect its logs and wait for
`status.json` to report completion. The command
`python3 scripts/run_gtsn_docker.py --report --root <run>` computes uncertainty
and appearance diagnostics if absent, then produces the final summary and
figures. `python3 scripts/run_gtsn_docker.py --test` runs the regression suite
in Docker.

The final audit checks matched sampling across all four ten-epoch runs, split
separation, causal histories, all 700 finite trajectories, empty expert-progress
indices, and matching training/evaluation source hashes. The reporting scripts
have separate hashes in `results.json`. The relevant artifacts are:

- `protocol.json`, `selection.json`, `status.json`, and `audit.json`: provenance,
  validation-only selection, completion status, and consistency checks.
- `heads/<mode>/best.pt`, `latest.pt`, `epochs.json`, and `results.json`: all four
  trained variants, selected epochs, learning curves, and calibration metrics.
- `rollouts/validation/<condition>/` and
  `rollouts/test/state_correction_fixed15/`: summaries and every episode's
  metrics and joint/camera/target trajectories.
- `results.json`: consolidated results, paired confidence intervals, realized
  sensing schedules, and comparison with the original Pi3 test run.
- `figures/`: PNG and PDF versions of training curves, validation outcomes,
  uncertainty calibration, and matched-view appearance examples.

The head checkpoints require the frozen backbone; they are not standalone full
Pi3 weights. The fully implemented memory candidate is retained at
`heads/memory/best.pt` even though validation selected the state-correction
control for final testing.

## Interpretation limits

This is one training seed with frozen, previously task-trained perception and
a bounded residual head. It does not establish performance for end-to-end
perception training. Expert RGB and reconstructed simulator RGB have a known
texture mismatch. The benchmark scenes are static; moving-object robustness,
novel sensor occlusion, and unseen-space reconstruction are not evaluated.
Active timing is a narrower capability than active viewpoint selection.
Existing baseline test results were already inspected before this research
iteration, so the new test is an exploratory benchmark estimate.
