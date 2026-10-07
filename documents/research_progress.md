# Geometry-grounded navigation progress — 2026-10-07

**C1, C2 and C3 meet their requested ≥3-point observed success gains on both full
splits.** The final regional-body model reaches **91% validation / 89%
test**, above the 70% threshold. All comparisons use the fixed 100-episode
splits with 20 direct, 40 over and 40 side scenes.

| Contribution and matched control | Model validation / test | Control validation / test | Gain, points |
|---|---:|---:|---:|
| C1 persistent map vs strict current RGB frame | 89 / 81 | 79 / 75 | **+10 / +6** |
| C2 regional hand + rigid tool vs TCP-only | 91 / 89 | 87 / 85 | **+4 / +4** |
| C3 uncertainty-aware refinement vs no refinement | 89 / 82 | 82 / 78 | **+7 / +4** |

**C1:** bounded actual surface anchors, adaptive merging/eviction and no age
expiry; old-only geometry is queried up to 315/330 control steps old. The strict
control disables all earlier geometry and visual features. Selected C1 keeps
four visual features; its extra map adds +2/0 over four-cloud memory. Geometry-only
variants gain +1 validation point. The whole primary gain therefore reflects
combined temporal context, rather than long-term geometry alone.

**C2:** model known collision geometry for palm, measured fingers, rigid wrist
and camera housing; filter self surfaces before map insertion. Score separate
regional contact features along candidate motion. Validation selects six-region
weight 1 before test. The matched TCP control skips all body scoring/filtering.
Paired intervals are [0,+9]/[−1,+10] points. Earlier union/posture failures and
the signed hand primary's +5/+2 miss are preserved as selection evidence.

**C3:** train a 4,865-parameter point-error head on existing expert/perturbation
targets, retain uncertainty with persistent anchors, and inflate uncertain
surfaces by up to 30 mm. Validation selects this maximum before test. Primary
paired 95% intervals are [+1,+13]/[−2,+10] points. **Adaptive uncertainty's extra
success benefit is not established:** fixed clearance scores 89/81, and matching
uniform padding 87/83. Measured quantile coverage is 77.7%, below its 90% target;
this is an uncertainty signal, not a certified bound.

These are staged comparisons, not three removals from the final model; gains
cannot be added. Completed **4,600 new follow-up rollouts**
(1,200 C1 + 900 C3 + 2,500 C2), frozen source and
checkpoint records, paired reports, trajectory/action audits and PNG/PDF
figures. **89 regression tests pass.** Existing expert plus independent
perturbation data, RGB backbone, IK, servo and 15-step execution are preserved.
All numerical work runs in the existing Docker image. Historically reused test
results remain exploratory.

[Research story](research_story.md) · [C1 evidence](c1_persistent_geometry.md) ·
[C2 method and results](c2_embodied_geometry.md) ·
[C3 method and results](c3_uncertainty_clearance.md) ·
[Final C2 figures and receipts](../runs/c2_part_geometry_20261007/final).
