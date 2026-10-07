# GTSN: from predicted surfaces to task-constrained motion

## C1 follow-up: adaptive persistent geometry

**The revised C1 model scores 89/100 validation and 81/100 test, versus 79/75
with only the current RGB frame.** The control removes all earlier geometry
and visual features, as requested. The +10/+6-point gains meet the ≥3-point
target on both splits; paired intervals are [+4,+17] and [+2,+11] points.
The existing expert/perturbation-only checkpoint, hand queries, route trust,
IK, terminal servo and 15-step execution remain fixed.

C1 now uses a bounded map of actual RGB-predicted surface anchors. It merges
nearby observations, tracks distinct-frame support and disagreement, retains
older evidence without fixed age expiry, and reads older-only surfaces along
candidate hand motion. Validation selected unit-weight reads; confidence gating
and extending the visual buffer were less successful. The selected controller
keeps four recent visual features. Geometry is queried up to 315/330 control
steps old and changes 81/74 correction choices on validation/test.

| C1 comparison | Validation /100 | Test /100 |
|---|---:|---:|
| **Selected persistent map + four recent visual features** | **89** | **81** |
| Current geometry + current visual frame | 79 | 75 |
| Four clouds + four visual features | 87 | 81 |
| Persistent map + current visual frame | 80 | Not advanced |

The extra map gives two validation gains and no losses over four clouds, but
identical test success indicators. With visual history removed, geometry-memory
variants gain only one validation point and fail the gate. Therefore the primary
result supports combined temporal context; it does not establish a three-point
gain from long-term geometry alone. **1,200 new C1 rollouts and 67 regression
tests** are complete, with source/checkpoint freezing, active/frozen controller
parity and trajectory/map-use audits. The historically reused test remains
exploratory. Details and reproduction are in
[c1_persistent_geometry.md](c1_persistent_geometry.md); figures are in
[the C1 archive](../runs/c1_consensus_20261007/final).

The C2/C3 ablations below belong to the initial four-cloud model and were not
rerun with the new map. Their individual effect estimates retain that scope.

## Initial geometric-energy study

**A geometry-grounded energy architecture achieves 87/100 validation and 81/100
test successes using existing expert routes and perturbation-only recovery data.**
Both exceed the requested 70% threshold. Three matched component removals lower
validation success by 6, 6 and 4 percentage points. Their test effects are
−2, +2 and −1 points: the component benefits do not transfer uniformly.

The contribution is an explicit, inspectable connection between **remembered
surfaces, proposed hand motion, and learned task progress**, with a controlled
study of each interface. The new trainable component is a small positive
route-trust network. Surface buffering, hand probes and the candidate lattice
reuse the existing clearance implementation. We do not claim three independently
novel algorithms or a new visual reconstruction backbone.

The full method improves the reused compact baseline from 82/78 to 87/81 on
validation/test. It exceeds the freshly replayed fixed-trust clearance baseline
on validation, 87 versus 85, but falls below it on test, 81 versus 83. The model
and its settings were fixed before this study's test runs and remain unchanged.
These are exploratory benchmark results on a historically inspected test split.

## Research question and gap

**How should a navigation policy turn imperfect geometry from a moving RGB
camera into useful action information?** Reconstruction supplies surfaces, but
navigation requires relationships: which surfaces remain relevant after the
camera turns, whether the moving hand approaches them, and how strongly an
uncertain local field should redirect a useful task route. A descriptive point
map does not resolve those decisions by itself.

Geometry reconstruction, semantic mapping and motion optimization provide
related building blocks. [Pi3](https://arxiv.org/abs/2507.13347) studies visual
geometry reconstruction; [ConceptGraphs](https://arxiv.org/abs/2309.16650) builds
semantic 3D representations for planning;
[CHOMP](https://www.cs.cmu.edu/~mzucker/icra09-chomp.pdf) optimizes trajectories
with obstacle costs. Our scoped question concerns their interface in this
benchmark: complete navigation outcomes under partial, RGB-predicted geometry
and a fixed expert/recovery training budget. This positioning is not an external
baseline comparison or an exhaustive novelty claim.

## Exactly three challenges and corresponding contributions

### C1 — Relevant surfaces disappear when the wrist camera moves

A current image can omit surfaces observed a few moments earlier. Appearance
history does not provide an explicit geometric relation to the current robot.

**Initial contribution: episodic metric surface persistence.** Retain four predicted
surface clouds in robot-base coordinates and query their union from the current
motion. This is bounded geometric history, not an unbounded fused map. The
no-history control retains the same four-frame visual proposal attention and
removes only older surface clouds.

Full versus no-history success is **87 versus 81 on validation**, with nine
gained and three lost episodes; paired 95% interval [−1,+13] points. Test is
**81 versus 83**, two gained/four lost, interval [−7,+3]. Persistence helps the
observed validation outcomes but is not a demonstrated test improvement. No
claim that longer memory is universally better follows from this comparison.

### C2 — TCP clearance does not describe the geometry of a moving hand

A TCP can pass above a surface while the hand behind it approaches the obstacle.
Describing scene geometry alone leaves the robot's physical extent implicit.

**Contribution: action-conditioned swept-hand queries.** Query each candidate
at executed-prefix steps 3/6/9/12/15 using the TCP and two samples 5/10 cm behind
it along the proposed wrist axis. Surface geometry thereby becomes a motion
cost. The TCP-only control removes the two additional body probes.

Full versus TCP-only success is **87 versus 81 on validation**, seven gained/one
lost, interval [+1,+11] points; test is **81 versus 79**, six gained/four lost,
interval [−4,+8]. This is the component with positive success estimates on both
partitions. Its exact McNemar p values are 0.070 and 0.754; the bootstrap interval
and exact test are different procedures, so these are not two independent claims
of statistical significance. Three axial probes do not cover the entire arm or
certify realized collision-free IK trajectories.

### C3 — Local geometric improvement can sacrifice task progress

Noisy surfaces can make a large detour look attractive. Pure clearance scoring
can repeatedly discard a useful route and delay completion.

**Contribution: positive, task-conditioned route preservation.** A trained
8→32→1 network supplies a bounded coefficient for the squared correction cost.
It sees goal-relative position, proposed progress and geometric risk summaries;
its 321 parameters are the only parameters updated. The no-trust control sets
this cost to zero. A separate fixed-0.08 control isolates the benefit of learning
from the benefit of having a preservation term at all.

Full versus no-trust success is **87 versus 83 on validation**, ten gained/six
lost, interval [−4,+12] points; test is **81 versus 82**, six gained/seven lost,
interval [−8,+6]. Preservation reduces validation mean nominal correction from
4.51 to 1.34 cm and removes two timeouts. On test it removes one timeout and
reduces mean control steps from 165.62 to 149.33, but success falls by one point.
These are measured progress/clearance tradeoffs, not a universal safety gain.

Learning versus fixed trust is **+2 validation points** and **−2 test points**,
with paired intervals [−3,+7] and [−7,+3]. Learning the coefficient improves the
observed validation result but does not establish a test upgrade over the
existing fixed-cost controller.

![Implemented geometry-to-action architecture](../runs/geometric_energy_20261007/final/architecture.png)

## Model and training

The frozen RGB model supplies metric surfaces and a learned six-knot route.
Fourteen bounded side/up corrections share that route and its wrist proposal.
For candidate c, average the largest Gaussian surface response over its fifteen
swept-hand queries, using a fixed 4 cm proximity scale. Select the minimum of
that geometric cost plus lambda(context) times the squared normalized correction
length. Lambda lies in [0.02,0.12], initially 0.08. Corrections ramp along the
chunk and fade near the goal. The shared IK, terminal servo and fixed 15-step
execution are preserved.

Train only the added network on 7,951 existing cadence-sampled expert
observations plus 2,997 independent perturbation-only recovery observations.
Use 65/35 source sampling and 2:4:4 route sampling, with no new corrective data.
Recoveries reset history; expert context contains only earlier cached
observations. Candidate errors against the existing expert route form soft
training targets. Fit cross entropy plus a trust regularizer for 20 fixed epochs,
32,768 draws per epoch, batch size 256, AdamW at 0.001, and seed 20261007.

The checkpoint is epoch 20, fixed before rollout selection. Candidate-level
validation RMSE changes from 18.751 to 18.647 mm; this small offline improvement
is not used as proof of navigation benefit. Training approximates future wrist
rotation with the current TCP rotation because the cache lacks future backbone
joint proposals. Deployment uses the original proposed wrist rotations. Cached
point decoding and deployment mixed precision can also differ. The explicit
model equations, input semantics and approximation limits are in
[geometric_energy_method.md](geometric_energy_method.md).

## Complete experiment results

Use the unchanged 800/100/100 episode split, with route counts 160/320/320 in
training and 20/40/40 in each evaluation partition. All five conditions share
one trained checkpoint and the same control settings. These are inference-time
component removals without retraining, not claims about the best independently
trained alternative architectures.

Primary success is reaching within 1 cm in XYZ without collision, within the
inherited 400-step limit. Full XYZ-plus-orientation success is reported
separately: **41% validation and 28% test** for the full model, versus 44/31% for
fixed trust. The 87/81% primary result must not be called full-pose success.

| Condition | Validation success | Test success | Test collision / timeout | Test direct / over / side |
|---|---:|---:|---:|---|
| Original compact, reused historical evidence | 82% | 78% | 22 / 0% | 100 / 92.5 / 52.5% |
| **Learned geometric energy** | **87%** | **81%** | **19 / 0%** | **100 / 90 / 62.5%** |
| Without geometric history | 81% | 83% | 17 / 0% | 100 / 92.5 / 65% |
| TCP-only queries | 81% | 79% | 20 / 1% | 100 / 90 / 57.5% |
| Without route preservation | 83% | 82% | 17 / 1% | 100 / 95 / 60% |
| Fixed trust, freshly replayed clearance | 85% | 83% | 17 / 0% | 100 / 90 / 67.5% |

Against original compact, full gains eight validation episodes and loses three:
+5 points, interval [−1,+11], exact McNemar p=0.227. On test it gains seven and
loses four: +3 points, interval [−3,+9], p=0.549. The small benchmark gains remain
uncertain; none of these estimates establishes population-level superiority.

Intervals use 20,000 paired resamples within route strata. McNemar tests are
exact, exploratory and uncorrected for multiple comparisons. Component effects
cannot be added together: deleting a module changes subsequent visited states.

![Complete ablation outcomes](../runs/geometric_energy_20261007/final/ablation_success.png)

![Component effects with paired uncertainty](../runs/geometric_energy_20261007/final/contribution_effects.png)

## What is established, and what remains limited

The completed implementation meets the requested overall success threshold on
both partitions and provides positive validation success ablations for each of
three geometry-grounded interfaces. The story is coherent because geometry
enters an explicit chain from retained scene evidence to proposed body motion
and task-preserving execution, rather than appearing only as an auxiliary
reconstruction target.

The test controls constrain the strength of that story. Hand extent has positive
success estimates on both partitions; surface persistence and route preservation
have validation gains but negative test estimates in this learned variant.
Learning the coefficient has not improved test success over fixed trust. Report
those findings alongside the favorable validation results. They are not three
robust independent improvements or a reason to discard the existing baseline.

Earlier project experiments already explored longer memory, fused/Gaussian
representations and altered execution cadence. Their failure to establish a
stable upgrade motivated this bounded architecture; it does not prove those
ideas cannot work elsewhere. The prior clearance narrative is preserved in
[research_story_clearance_20261003.md](research_story_clearance_20261003.md), and
all earlier experiment archives remain unchanged.

Evidence comes from one adapter seed, static simulated tabletop scenes, and a
historically reused test split. Repeated conditions on these 100 scenes do not
create 500 independent test scenes. Independent seeds and fresh scenes would be
needed for stronger generalization claims; they are not results of this study.
There is no full-arm collision guarantee, dynamic-scene evaluation or real-camera
validation.

## Reproduction and artifact record

The completed run is
`/run/user/1016/experiments/gtsn_geometric_energy_20261007`, also linked from
`runs/geometric_energy_20261007`. It contains the small adapter checkpoint,
training records, immutable rollout source, cache/parent hash receipts,
pre-test method decision and freeze record, all 1,000 trajectories and outcomes,
paired reports, finite/no-expert/fixed-execution audits, and PNG/PDF figures.
The frozen parent remains at
`/run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt`;
its weights are reused rather than duplicated in the adapter.

Replay the full learned model with a new output directory:

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 evaluate \
  --controller geometric_energy \
  --checkpoint /run/user/1016/experiments/gtsn_geometric_energy_20261007/training/energy.pt \
  --partition test --no-render-videos \
  --output-dir /run/user/1016/experiments/geometric-energy-replay-new
```

Add `--energy-ablation no_history`, `tcp_only`, `no_trust` or `fixed_trust` for a
control. Use `--partition validation` for validation. To reproduce training and
the complete panel, use a fresh experiment root:

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 geometric_energy \
  --checkpoint /run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt \
  --cache /run/user/1016/experiments/gtsn_persistent_cache_20261005 \
  --output-dir /run/user/1016/experiments/geometric-energy-new/training
python3 scripts/run_geometric_energy_study.py --root /run/user/1016/experiments/geometric-energy-new --stage validation
python3 scripts/run_geometric_energy_study.py --root /run/user/1016/experiments/geometric-energy-new --stage test
python3 scripts/run_geometric_energy_study.py --root /run/user/1016/experiments/geometric-energy-new --stage report
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --cpu geometric_energy_report \
  --root /run/user/1016/experiments/geometric-energy-new \
  --output-dir /run/user/1016/experiments/geometric-energy-new/final
```

All numerical work, tests and simulation run in the existing project Docker
image. No host package installation or unrelated image modification occurred.
**58 regression tests pass**, including geometry invariance, empty/nonfinite
surfaces, hand extent, positive trust bounds, episode resets, and exact
fixed-trust decision parity. Empty failed initialization directories were
removed; completed controls are retained because they are required evidence.
