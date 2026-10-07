# C1: adaptive persistent geometry

The C1 target is a gain of at least **3 percentage points** over a strict
current-frame-only control on each fixed 100-episode validation and test split.
The control has neither earlier surface clouds nor earlier visual features.
Success remains collision-free XYZ reaching within 1 cm; the original 70%
minimum, route proportions, physics and 15-step execution interval remain fixed.

## Representation and action interface

`ConsensusGeometry` stores at most 1,600 surface anchors in robot-base metres.
It receives only RGB-predicted geometry, the measured TCP, the current goal and
the observation timestamp. It rejects invalid/out-of-workspace points and points
within 7 cm of the TCP at insertion. It keeps one actual prediction per 2 cm
voxel and associates observations within 2.5 cm with an existing anchor. Anchor
positions remain actual predictions rather than averages across depth edges.

Every stored anchor has a first/last observation timestamp, distinct-frame
support count and running spatial disagreement. A frame contributes at most one
support vote per anchor, regardless of the number of matching pixels. Its
confidence is

\[
w_i=\operatorname{clip}((n_i-1)/2,0,1)
\exp[-s_i/(2\cdot 0.02^2)],
\]

where `n` counts distinct timestamps and `s` is squared positional disagreement.
Disagreement updates use support capped at eight. If capacity is exceeded,
confidence plus proximity to the measured TCP-to-goal segment determines
retention. There is no fixed age expiry and no free-space carving from missing
observations. This map persists within one episode and resets between episodes.

The freshest four raw clouds retain the established scorer. Older-only anchors
(last observed before the four-frame window) join these clouds. Two matched
read variants use either confidence weights or unit weights; support and task
proximity remain capacity-retention signals in both. Swept-hand risk takes a
maximum of weighted Gaussian proximity **per
query**, then averages the same five execution-prefix times and three hand-axis
probes. The candidate lattice, learned positive route-trust coefficient, wrist
proposal, goal fade, IK and terminal servo remain unchanged. The scorer is a
proximity heuristic, not a calibrated collision probability.

## Comparisons and causal attribution

The first frozen validation panel compares a strict current-frame control,
the original four-frame model, the supported persistent map, an unconfirmed-map
control, and a map with a sixteen-frame visual buffer. It tests whether map
support matters and whether extending visual history helps.

A second panel isolates geometry: `recent_geometry` and `persistent_geometry`
both give the route head **one current frame**. Only their geometric surface
memory differs. Their strict `current` control is reused from the compatible
first panel with checkpoint, source and result hashes recorded. Thus a benefit
of these variants cannot be attributed to earlier visual features. Comparisons
against four clouds separately test the additional value of older map evidence.
An additional `unconfirmed_geometry` control removes visual history from the
selected unit-read map while preserving its geometry settings.

All runs reuse the original expert-plus-independent-perturbation-only
geometric-energy checkpoint. There is no new training or observation collection.
Architectures and sources are frozen separately for each validation panel;
the final method and controls are nominated before test. This historically
reused test split remains exploratory. Component deletions are inference-time
interventions, not comparisons with independently retrained optimal controls.

Diagnostics record old-only point counts, maximum queried age, old-induced
changes in correction choice, map storage and visual-history length. Tests check
distinct-view support, disagreement, bounded capacity, retention beyond four
frames, timestamp order, empty evidence, reset, query parity, and the invariance
of the strict control to every earlier RGB frame. Paired statistics use 20,000
route-stratified episode bootstraps and exact McNemar tests. Point-estimate gains
and statistical uncertainty are reported separately.

Implementation: [adaptive_geometry.py](../src/tsn/models/adaptive_geometry.py),
[run_c1_study.py](../scripts/run_c1_study.py), and
[c1_report.py](../src/tsn/cli/c1_report.py). Numerical work and simulation use
`gtsn-persistent:20261005-compact-only`; source/data mounts are read-only and
outputs stay under `/run/user/1016/experiments`.

## Results

| Condition | Visual frames | Validation /100 | Test /100 |
|---|---:|---:|---:|
| **Anchored persistent map, unit reads (`unconfirmed`)** | 4 | **89** | **81** |
| Strict current-frame control (`current`) | 1 | 79 | 75 |
| Four clouds (`recent`) | 4 | 87 | 81 |
| Support-weighted map (`persistent`) | 4 | 87 | 81 |
| Map plus longer visual buffer (`persistent_visual`) | 16 | 84 | — |
| Four clouds, current visual frame (`recent_geometry`) | 1 | 80 | — |
| Support-weighted map, current visual frame (`persistent_geometry`) | 1 | 80 | — |
| Unit-read map, current visual frame (`unconfirmed_geometry`) | 1 | 80 | — |

The primary **+10 validation / +6 test points** meets the requested ≥3-point
target on both splits, and 89/81% exceeds the 70% minimum. Validation has eleven
paired gains and one loss, bootstrap interval [+4,+17] points, exact McNemar
p=0.00635. Test has six gains and no losses, interval [+2,+11], p=0.03125.
These are fixed-benchmark estimates on a historically reused test split.

The map's incremental benefit over the four-cloud model is **+2 validation
points (two gains, no losses; interval [0,+5], p=0.5) and a test tie with
identical success indicators**. It is not an established test upgrade over
four-frame memory. With visual history removed, each geometric-memory variant
scores 80 versus 79 validation successes: four gained/three lost, interval
[−4,+6], p=1. Those variants fail the validation gate and are not tested. Thus
the primary C1 comparison supports combined temporal context, while the full
three-point benefit of geometry alone remains unestablished. The original
geometric-only deletion, which kept visual history, is a different comparison
and its historical 81/83 result is not used as the new strict control.

The selected map reads older-only geometry in 99/100 episodes on each split,
at 669/645 replans, with maximum age 315/330 control steps. Older geometry
changes 81/74 correction choices in 53/43 episodes. Peak map sizes are 520/668
anchors and 18,720/24,048 bytes, excluding raw clouds, visual features and query
temporaries. Support-weighted reads change only two validation choices and tie
four-cloud success; the unit-read variant was selected by highest full
validation success before inspecting test outcomes. There is no claim that
the support weights themselves improve performance.

Selected route counts are 20/20 direct, 36/40 over and 33/40 side on validation;
20/20, 36/40 and 25/40 on test. All failures are collisions, with no timeouts.
XYZ-plus-orientation success is 41%/27%, distinct from the primary XYZ metric.

Completed **1,200 new rollouts** (800 validation / 400 test); reused strict
controls in the attribution panels are counted once. All trajectories pass
finite-value, no-expert-progress and fixed-execution audits. **67 regression
tests pass**. Active versus archived controllers produce identical commands and
diagnostics for all five first-panel modes over 27 synthetic observations each,
including complete perception/history/map/energy/IK processing. This parity
check exercises the controller, not Pi3 reconstruction or simulator replay.

Some initial frozen `config.json` files use a generic confidence-weighted read
description even for the unit-read condition. Their mode names and executed
source correctly specify unit reads; this descriptive metadata is corrected in
the active code and the final receipt. Frozen historical files are preserved.

Main results, memory audit, criteria and figures:
[final](../runs/c1_consensus_20261007/final).
All validation conditions and paired comparisons:
[validation report](../runs/c1_consensus_20261007/reports/validation).
Geometry-only evidence:
[four-cloud/support-weighted panel](../runs/c1_geometry_only_20261007/reports/validation),
[unit-read visual-history removal](../runs/c1_map_attribution_20261007/reports/validation).

## Reproduction

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 evaluate \
  --controller adaptive_geometry --memory-mode unconfirmed \
  --checkpoint /run/user/1016/experiments/gtsn_geometric_energy_20261007/training/energy.pt \
  --partition test --no-render-videos \
  --output-dir /run/user/1016/experiments/c1-replay-new
```

Use `--memory-mode current` for the strict control and `recent` for four-frame
memory. The adaptive controller defaults to the selected unit-read variant;
explicit mode arguments are preferred for research reproduction. All output
directories must be new. The original energy checkpoint is reused unchanged;
the new C1 representation is deterministic, with no additional learned weights.
Frozen runtime source is under the main archive's `source/`; use
`--source-snapshot /run/user/1016/experiments/gtsn_c1_consensus_20261007/source`
to replay exactly the evaluated implementation.

To reproduce the primary full panel in fresh storage:

```bash
python3 scripts/run_c1_study.py \
  --root /run/user/1016/experiments/c1-study-new --stage validation
python3 scripts/run_c1_study.py \
  --root /run/user/1016/experiments/c1-study-new --stage test \
  --candidate unconfirmed --conditions current,recent,unconfirmed,persistent
python3 scripts/run_c1_study.py \
  --root /run/user/1016/experiments/c1-study-new --stage report --candidate unconfirmed
```

The launcher freezes fresh source/checkpoint hashes before validation and the
selected test panel before test; it rejects a candidate below the validation
gate. The saved C1 archive also contains the independent fixed-control test
freeze used to overlap baseline evaluation with remaining validation work.
