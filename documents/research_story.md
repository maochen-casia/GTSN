# Geometry-grounded navigation: persistence, embodiment and uncertain clearance

The completed follow-ups achieve **91/100 validation and 89/100 test successes**
with persistent RGB geometry, regional body modelling and uncertainty-aware
clearance refinement. Both exceed 70%. The requested C1, C2 and C3 comparisons
exceed three observed percentage
points on each full split. The geometry interface connects retained surface
evidence, the physical extent of proposed hand motion, and uncertainty-dependent
local corrections.

| Contribution and its control | Model validation / test | Control validation / test | Difference, points |
|---|---:|---:|---:|
| C1 persistent map vs strict current frame, C1 study | 89 / 81 | 79 / 75 | **+10 / +6** |
| C2 regional hand + rigid tool vs TCP-only, C2 study | 91 / 89 | 87 / 85 | **+4 / +4** |
| C3 uncertainty refinement vs no refinement, C3 study | 89 / 82 | 82 / 78 | **+7 / +4** |

These are staged, matched ablations, not three removals from one final model.
The C1 result includes geometric and visual history together. C2 compares
regional body scoring and self filtering against TCP-only on the C1/C3 model.
C3's primary comparison establishes the benefit of refinement; it does not
isolate the benefit of adaptive uncertainty.
The individual differences cannot be added together.

## Research question

How can partial, imperfect RGB geometry become useful action information?
A reconstructed surface alone does not determine whether older evidence should
persist, whether a moving hand approaches it, or what margin to retain when its
position is uncertain. We study those interfaces under a fixed expert plus
independent perturbation training budget and unchanged low-level execution.

The implementation reuses the existing visual backbone, route proposal, bounded
candidate lattice and clearance machinery. Its contributions are explicit
geometry-to-action interfaces and controlled measurements in this benchmark.
They are not claims of independently novel reconstruction, mapping or planning
algorithms. Earlier context and related-work positioning remain in the
[initial energy study](research_story_initial_energy_20261007.md).

## C1 — The camera leaves useful geometry behind

A four-cloud buffer forgets a surface when it falls outside a short temporal
window, even if it remains relevant to the route. A feature history also lacks
an explicit metric relationship between that surface and current robot motion.

**Contribution: bounded, adaptive persistent surface anchors.** Keep actual
RGB-predicted points in robot-base coordinates, merge nearby observations,
track distinct-frame support and positional disagreement, and evict according
to agreement and TCP-to-goal relevance at capacity. Do not expire geometry by
age. Read older-only anchors together with four fresh clouds along candidate
hand motion. Reset the map at each episode; maximum capacity is 1,600 anchors.
Validation selected unit-weight reads over support-weighted alternatives.

The requested strict control retains only current geometry and current visual
features. C1 scores **89/81 versus 79/75**, with paired 95% intervals
[+4,+17]/[+2,+11] points. The controller queries evidence as old as 315/330
control steps and the old map changes 81/74 correction decisions.

The additional map gives **+2 validation points and a test tie** over four clouds
and four visual features. Geometry-memory variants restricted to one visual
frame gain only one validation point and fail the gate. The primary improvement
therefore supports combined temporal context; long-term geometry alone has not
established the requested three-point gain. Complete controls, map-use evidence
and reproduction are in [the C1 method](c1_persistent_geometry.md).

## C2 — TCP distance leaves body geometry implicit

A TCP can pass above an obstacle while the hand behind it approaches the surface.
Scene geometry becomes actionable only when related to the proposed body's
motion, rather than scored at a single reference point.

**Contribution: candidate-conditioned regional embodiment geometry.** Use known
robot collision meshes and oriented boxes for the palm, measured left/right
fingers, rigid wrist and camera housing. Add the TCP as a virtual reference.
Evaluate each region's strongest scene proximity at executed-prefix steps
3/6/9/12/15, then average across regions and time. This retains contact extent
that a single nearest-body maximum discards. Filter predicted self surfaces at
their observation pose before inserting them into persistent geometry.

The matched TCP-only model queries one reference point and skips all body-based
scoring and self filtering. Both retain C1 memory, four visual features, C3
uncertainty padding, proposal/trust weights, candidate positions and low-level
execution. C2 adds no training data or learned parameters. Full validation
selects six regions with weight 1; source and test controls are frozen beforehand.

The selected model scores **91/89 versus 87/85**, meeting the requested target
at **+4/+4 points**. Paired intervals are [0,+9]/[−1,+10]; the observed effect
does not establish a positive population gain. Side-route successes improve
from 30/28 to 34/32 out of 40, while direct and over totals tie. Test collision
rate falls from 15% to 11%. Compared with the old three-probe C3 model, gains
are +2/+7 points. That is a supplemental comparison.

Unfiltered body-union and roll-library panels fail the validation gate. A signed
hand nominee gives +5/+2 and misses the test target; its prespecified full-tool
secondary gives +3/+4. Both remain recorded. The regional prototype predates
those test outcomes and has its own validation nomination. The final primary
ablation measures regional scoring plus self filtering, not either in isolation.
Convex support-plane distances approximate outside corner distances; regional
averaging can dilute an isolated contact. The articulated arm is excluded, and
there is no full-motion collision guarantee. Complete controls, equations and
reproduction are in [the C2 method](c2_embodied_geometry.md).

## C3 — A fixed margin ignores uncertainty in predicted surfaces

RGB reconstruction can place a surface incorrectly, and a persistent anchor can
accumulate disagreement. A uniform geometric margin treats reliable and
uncertain predictions alike. Larger local padding could retain useful clearance
around the uncertain points without increasing every margin equally.

**Contribution: learned geometry-error signals and uncertainty-aware obstacle
inflation.** Train a 4,865-parameter 74→64→1 head on existing per-pixel metric
point-error targets. It consumes current visual cells, point position and local
roughness, measured camera pose and TCP distance. Retain its creation-time error
estimate with each persistent anchor and add observed positional disagreement
when querying old geometry. Convert predicted errors into bounded 0–30 mm
padding, then subtract that padding from hand-to-surface distance before scoring
proximity. The existing positive route-trust term limits unnecessary correction.

Select the maximum padding from 10/20/30 mm using full validation success,
breaking ties toward smaller padding. Freeze the chosen 30 mm checkpoint,
source and controls before test. It scores **89/82 versus 82/78 without any
clearance refinement**, meeting the requested gain at **+7/+4 points**. The
control keeps four visual features and the same route, map, IK and servo; it
returns the proposed waypoints unchanged.

| C3 condition | Validation /100 | Test /100 |
|---|---:|---:|
| **Selected point-specific padding, 0–30 mm** | **89** | **82** |
| No clearance refinement | 82 | 78 |
| C1 clearance without inflation | 89 | 81 |
| Uniform padding, 15.577 mm | 87 | 83 |

The primary paired intervals are [+1,+13]/[−2,+10] points. The selected candidate
has **0/+1 points** over non-inflated C1 and **+2/−1** over the matching uniform
control. Thus clearance refinement has positive observed effects, but an
additional success benefit from adaptive uncertainty is not established. Do
not replace the validation-selected candidate with the test-best control.

The learned signal ranks reconstruction quality: actual mean error rises from
21.5 to 170.8 mm across five prediction bins. Measured coverage is **77.7%**,
below the nominal 90% training target. This is an uncertainty signal rather than
a certified bound, collision probability or hard clearance constraint. The
primary result measures collision-free XYZ reaching; full XYZ-plus-orientation
success is separately 43/29%. Details, equations, controls and calibration are
in [the C3 method](c3_uncertainty_clearance.md).

![Current geometry-to-action architecture](../runs/c2_part_geometry_20261007/final/c2_architecture.png)

## Training, evaluation and reproducibility

The benchmark uses the original 800/100/100 split, with 160/320/320 route counts
in training and 20/40/40 in each evaluation partition. Train new components only
on the existing 7,951 cadence-sampled expert and 2,997 independent perturbation
observations. No new recovery collection or data change is used. C1 requires
no new trained parameters. The initial 321-parameter positive route-trust adapter
is frozen; C3 trains only its small uncertainty head for ten fixed epochs.
Existing recorded-depth targets supervise geometry error during training;
deployment consumes RGB predictions and measured robot/camera/goal state.
C2's known collision geometry adds no trained parameters.

All conditions preserve the original route backbone, wrist proposal, IK,
terminal servo, 15-step execution and 400-step limit. Primary success means
collision-free XYZ reaching within 1 cm. All simulation, numerical analysis and
tests run in the existing project Docker image, with source and data mounted
read-only and writes limited to the new experiment output. No host package
installation or unrelated container/image modification occurs.

Completed **4,600 new follow-up rollouts**: 1,200 C1, 900 C3 and 2,500 C2. The C3 fixed
control reuses 200 audited C1 results after exact command parity checks.
**89 regression tests pass**. C2 additionally audits 2,700 unique trajectories,
including 200 reused C3 axial results, and preserves every selection panel.
Completed artifacts include immutable training and
rollout source, input/checkpoint hashes, pre-test decisions, per-episode
trajectories, action/map diagnostics, paired statistics, calibration, audits,
and exportable PNG/PDF figures. Archives are linked from
[runs/c1_consensus_20261007](../runs/c1_consensus_20261007) and
[runs/c3_uncertainty_20261007](../runs/c3_uncertainty_20261007) and
[runs/c2_part_geometry_20261007](../runs/c2_part_geometry_20261007).

The test split was historically inspected and remains exploratory. There is one
uncertainty training seed, static simulated scenes, imperfect error calibration,
and no real-camera or dynamic-scene result. Paired intervals reflect scene-level
uncertainty; repeated conditions are not new independent scenes. The requested
observed success targets are achieved, while claims of a robust isolated
long-term-map gain or adaptive-margin superiority require further evidence.

[Brief progress](research_progress.md) · [Initial results and related work](research_story_initial_energy_20261007.md).
