# Geometry-grounded table scene navigation: research revision

**Completed outcome:** task-space route consensus raises validation success from
80% to **87%** and fixed-test success from 72% to **79%**, with test collisions
falling from 28% to 20%. The selected model's +7 percentage-point test gain has
a paired 95% bootstrap interval of **−2 to +16 points**, so the numerical
improvement is not statistically conclusive. A predeclared current-view-only
ablation scores 83%, but does not replace the validation-frozen selection.
All 600 validation and 400 test rollouts are complete and audited; all 29
regression tests pass. This revision composes existing learned heads without
additional training.

The research thesis is: **single-camera geometry is useful when its task-space
route estimate complements a strong state prior, and its value must be measured
at execution time. More temporal context and more frequent observation are not
automatically better.** This is an empirical systems result, not a claim of a
new geometry foundation model or calibrated safety guarantee.

## Question and starting point

Can a single wrist camera support more reliable table navigation when visual
geometry informs an executable route, rather than serving only as a descriptive
feature map? The required reference is the validation-selected Cartesian hybrid
in `/run/user/1016/experiments/gtsn_cartesian_20261003`: 80/100 validation and
72/100 test successes. Its inputs are RGB, joint state, goal pose, intrinsics,
and camera-to-base pose. Its new position decoder uses state; RGB still controls
orientation and the inverse-kinematics initialization. It is not a state-only
system. Success requires 1 cm position accuracy without collision within 400
control steps; full-pose success is a separate metric.

The earlier experiments support task-space control but do not establish an
incremental benefit from geometric uncertainty or memory. The 79% uncertainty
test result was not the validation winner and cannot be relabeled as a new
improvement. The present revision develops and selects on validation only.

## Three challenges and corresponding methodological responses

1. **Unreliable evidence.** A plausible reconstruction need not imply an accurate
   motion. Use complementary state-conditioned and visual-geometry-conditioned
   route estimates in a common metric space. Their consensus can reduce errors;
   their disagreement measures action ambiguity. Test whether this disagreement
   ranks actual waypoint errors and whether it helps schedule observations.
   Neither reconstruction variance nor disagreement is a collision probability.
2. **Incomplete views.** A wrist camera loses sight of previously observed
   surfaces as the robot moves. Retain a bounded causal history of actual visual
   geometry tokens with camera poses and observation ages. Evaluate this history
   against a matched current-view branch. Memory preserves evidence; it does not
   reconstruct never-observed obstacles or prove that unseen space is clear.
3. **Geometry without an execution contract.** A geometric description does not
   directly specify a reachable motion or terminal convergence. Predict
   goal-conditioned metric TCP waypoints, preserve the learned wrist orientation,
   and execute with bounded damped-least-squares inverse kinematics and a terminal
   servo. This component is inherited from the reference and held fixed when
   attributing any new gain to consensus or memory.

These responses map the challenges to implemented components and controlled
tests; they are not three separately established positive effects. Consensus
improves the measured score, the explicit map pathway has positive intervention
evidence, and memory's incremental effect changes sign between validation and
test. Active viewpoint selection and moving-object robustness are outside the
implemented scope.

## Implementation

`src/tsn/models/consensus_policy.py` composes frozen Cartesian heads after one
shared perception pass. Each head predicts six three-dimensional displacements
at steps 5, 10, ..., 30. A convex combination produces the route. The weighted
root-mean-square Euclidean dispersion over the first three knots measures
disagreement over the nominal execution interval. Optional disagreement-based
five-step execution is compared with fixed fifteen-step execution; optional
temporal action averaging uses only overlapping absolute joint targets from
the previous prediction and is reset at every episode.

The geometry branch retains the existing Gaussian mean/variance supervision and
detached reliability weighting. The memory branch attends to at most four actual
observations. This revision adds no depth, object coordinates, route labels,
expert progress, or future observations to deployment. It reuses the trained
heads; it is a new inference composition, not new backbone training.

For the selected two-head model, let `d_s(k)` and `d_m(k)` be the state and
memory displacements in metres. The executable displacement is
`d(k) = 0.5 d_s(k) + 0.5 d_m(k)`. The diagnostic disagreement is the square
root of the mean squared Euclidean deviation of those estimates from `d(k)`
over `k = 1, 2, 3`. Unlike averaging joint solutions, averaging these routes
uses a common physical coordinate system before orientation-constrained IK.
The resulting mean is not guaranteed collision-free, particularly if hypotheses
pass on different sides of an obstacle; closed-loop collisions remain the
primary failure metric.

## Simplification and research positioning

Keep the frozen RGB perception, metric waypoints, learned orientation, bounded
IK, and common goal servo. Replace an all-or-nothing choice between the strong
state prior and visual route estimates with an explicitly measured consensus.
Do not add an occupancy planner to pooled surface coordinates: these cells can
average multiple surfaces and do not represent free-space evidence. Do not
claim that the earlier under-covering variance is calibrated safety uncertainty.
Discard active timing and temporal averaging from the selected policy because
their closed-loop validation screens underperform fixed execution.

The components have precedents: [deep ensembles](https://arxiv.org/abs/1612.01474)
motivate predictive diversity, while [ACT](https://arxiv.org/abs/2304.13705)
uses action chunking and temporal ensembling. Our experts share a backbone and
training seed, so this is heterogeneous route consensus, not an independent
deep ensemble. The existing backbone is adapted from
[Pi3](https://arxiv.org/abs/2507.13347). The contribution is the empirical
case for combining a kinematic route prior with camera geometry at the executable
waypoint level, not invention of ensembling, memory, or inverse kinematics.

## Protocol and validation evidence

All numerical work runs in the existing project Docker image. New artifacts
live in `/run/user/1016/experiments/gtsn_consensus_20261003`; historical artifacts
remain intact. Episode partitions retain 160/320/320 training and 20/40/40 each
validation/test route counts. Screens use the first 4/8/8 validation episodes.
Full validation, not screens or test, selects the final composition. Test
conditions must be explicitly frozen in `selection.json` before execution.
Every rollout retains checkpoint checksums, source snapshots, complete states,
actions, observation schedules, contacts, and unaltered success rules.

Cached validation waypoint RMSE is 16.70 mm for the reference state head,
15.06 mm for equal state/memory consensus, and 15.03 mm for equal
state/current-uncertainty consensus. State/memory disagreement has Spearman
correlation 0.616 with first-fifteen-step waypoint error; high- and low-disagreement
quartiles have mean Euclidean errors 20.49 and 6.00 mm. These frame-level results
support error ranking but do not establish closed-loop gains or calibration.

Full validation is 87% success/13% collision/0% timeout for state/memory
consensus, versus 80%/20%/0% for the reference. Current-view deterministic and
uncertainty-weighted consensus both achieve 83%. The 87% result consists of
20/20 direct, 36/40 over, and 31/40 side successes. The paired validation
comparison has ten recovered episodes and three regressions (exact McNemar
p=0.092); selection on this validation set precludes a confirmatory interpretation.

The first-screen results are 17/20 for state/memory, 16/20 for
state/current-uncertainty, and 14/20 for the three-head combination. State/memory
with adaptive five/fifteen-step execution falls to 12/20, with two timeouts,
and increases observations from 10.7 to 23.2 per episode. Temporal averaging
scores 15/20. Neither addition belongs in the selected method. A useful error
ranking signal therefore does not automatically produce a useful sensing policy.

Validation residual correlation is approximately 0.60 between the state and
memory heads, but 0.984 between memory and current-view uncertainty heads.
This supports complementary state/visual errors as the consensus mechanism,
and explains why adding another highly correlated visual estimate is not
necessarily beneficial. All 29 regression tests pass, including the two GPU
integration tests and new temporal alignment, disagreement, and reset checks.

Removing history from the same memory checkpoint yields 83% success, 16%
collision, and 1% timeout. The complete model recovers four episodes with no
regressions relative to this intervention (exact McNemar p=0.125). This is a
small positive memory effect, not statistically decisive evidence by itself.
Zeroing the six explicit map inputs yields 72% success and 28% collision.
These six channels include predicted XYZ, two goal channels, and an action
affordance channel. The intervention therefore measures reliance on the
**explicit task-conditioned map pathway**, not XYZ alone. RGB tokens, camera
pose, learned weights, and the kinematic controller remain available. Because
the intervention shifts the input distribution, it does not replace a matched
retrained map-free model comparison.

The selected model is frozen in `selection.json`: state/memory consensus with
equal weights, fifteen-step execution, no temporal averaging, and the common
8 cm terminal-servo activation radius (success tolerance remains 1 cm). The
three predeclared test ablations are current-view-only memory, zero explicit
maps, and state/deterministic-visual consensus. The geometry intervention was
still finishing validation when selection was frozen; it was diagnostic only,
and could no longer exceed 87% even if all its remaining episodes succeeded.

## Final fixed-test results and defensible claims

Every full condition uses the same 100 episodes with 20/40/40 route counts.
The reference's results are the supplied historical artifacts; its numerical
model, renderer, contact rules, and goal criteria remain unchanged.

| Condition | Validation success | Test success | Test collision | Test timeout |
|---|---:|---:|---:|---:|
| Required reference | 80% | 72% | 28% | 0% |
| State + memory consensus **(selected)** | **87%** | **79%** | **20%** | **1%** |
| Same memory checkpoint, current view only | 83% | 83% | 17% | 0% |
| Same checkpoints, explicit maps zeroed | 72% | 67% | 33% | 0% |
| State + deterministic current-view visual | 83% | 76% | 23% | 1% |
| State + current-view uncertainty | 83% | not run | — | — |

The selected model succeeds on 20/20 direct, 35/40 over, and 24/40 side routes.
The reference succeeds on 20/20, 28/40, and 24/40 respectively. The aggregate
gain is concentrated in over routes; there is no aggregate side-route gain.
Selected full-pose success is only 25%, versus 22% for the reference. Mean
selected rollout length is 140.5 control steps, versus 132.09 for the reference,
so a speed improvement is not claimed. The selected method uses 9.77
observations per episode on average.

| Paired test comparison | Success difference | 95% paired bootstrap interval | Wins / losses | Exact McNemar p |
|---|---:|---:|---:|---:|
| Selected − reference | +7 pp | −2 to +16 pp | 14 / 7 | 0.1892 |
| Current-view-only ablation − reference | +11 pp | +3 to +19 pp | 15 / 4 | 0.0192 |
| Selected − zero explicit maps | +12 pp | +3 to +21 pp | 17 / 5 | 0.0169 |
| Selected − current-view-only ablation | −4 pp | −8 to −1 pp | 0 / 4 | 0.1250 |

Intervals use 20,000 paired episode bootstrap resamples stratified by route.
They are conditional on these fixed weights and scenes and do not measure
training-seed variation. Small discordant counts make percentile bootstrap
intervals fragile: the memory comparison's interval excludes zero while the
exact test is not significant. Secondary comparisons have no multiplicity
correction and remain exploratory.

The final contribution claims should be phrased as follows:

1. **Complementary route inference:** combine a state prior and a visual
   geometric estimate before kinematic execution to reduce waypoint error and
   improve observed benchmark success. The primary selected-model gain is
   numerical, not statistically established; the stronger 83% ablation supports
   the direction but was not selected for deployment.
2. **Controlled temporal evidence:** bounded causal memory is implemented and
   tested with a same-checkpoint history-removal intervention. Its +4 pp
   validation effect becomes −4 pp on test. Memory has mixed evidence and is
   not a proven contribution to robustness. Adaptive timing's negative result
   also prevents claims of useful active observation in this study.
3. **Executable task-conditioned geometry:** the explicit map pathway affects
   navigation under an unchanged controller: removing it costs 15 pp on
   validation and 12 pp on test. The maps include spatial and task channels;
   neither pure XYZ benefit nor superiority over a retrained map-free model is
   isolated. Kinematics and the servo are inherited controls, not new
   contributions credited with the incremental gain.

The requested measured improvement and a consistent research story are
delivered, while the evidence does **not** justify claiming that all three
proposed modules independently improve navigation. The selected score also
ties the earlier *unselected* uncertainty head's 79% test score: it improves
over the required 72% selected reference, not every historical ablation. The
fixed test split was examined by earlier research iterations. These results
are exploratory; independent scenes, additional training seeds, retrained
map-free controls, and stronger primary-test evidence are required for a broad
claim of robust general improvement.

## Artifacts and reproduction

The experiment root is `/run/user/1016/experiments/gtsn_consensus_20261003`.
`selection.json` freezes validation selection, test specifications, and source
hashes. `selected_heads.pt` contains both selected head state dictionaries and
composition metadata; `selected_policy.json` records its checksum and the
required frozen backbone checksum. It is a head package, not a standalone RGB
backbone checkpoint. `results.json` contains route-wise outcomes and 20,000
route-stratified paired episode bootstrap comparisons. `audit.json` checks
completed conditions. `container_runs.json` and `logs/` retain image identity,
commands, timestamps, exit codes, and console output.

Each `rollouts/<partition>/<condition>/` contains a protocol, a numerical-source
snapshot, episode metrics, complete joint/TCP/camera/target trajectories, and a
completion record. `figures/validation_paths.{png,pdf}` shows the first recovered
and regressed validation episodes by route and episode ID, with the selection
rule saved alongside the plots. These examples include a side-route regression;
the method does not improve every episode.

Run these commands from `/home/chenmao/GTSN`. The host launcher uses only the
standard library; numerical programs execute in the existing project image
`gtsn-pi3:20261002`, immutable ID
`sha256:5716cc81a619a9d00e6d08edecedff42a5b2b57fe10c8ff22835920cb480e009`.

```bash
# Reproduce selected-policy validation in a fresh directory.
python3 scripts/cartesian_docker.py --name gtsn-consensus-reproduce --gpu 0 \
  python scripts/run_consensus.py --name state_memory \
  --root /run/user/1016/experiments/gtsn_consensus_reproduce

# Recompute the completed study's statistics, audit, and figures.
python3 scripts/cartesian_docker.py --name gtsn-consensus-report --gpu 0 \
  python scripts/report_consensus.py

# Full regression suite, including both GPU integration tests.
python3 scripts/cartesian_docker.py --name gtsn-consensus-tests --gpu 0,1 \
  python -m unittest discover -s tests -v
```

For a fresh test reproduction, copy the frozen `selection.json` into the fresh
experiment root before adding `--partition test` to the first command. This
preserves the same selected model and ablation definitions. Use
`run_memory_ablation.py --name memory_current_only` and
`run_geometry_ablation.py --name memory_no_geometry` for the two same-checkpoint
interventions; their distinct source hashes are part of the audit. Existing
complete conditions are not overwritten. Incomplete conditions require a fresh
destination so partial evidence remains intact.
