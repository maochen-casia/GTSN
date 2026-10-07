# C3: uncertainty-aware clearance refinement

This follow-up retains the selected C1 persistent geometry and adds a small
geometry-error head. Points predicted to have greater reconstruction error
receive larger obstacle inflation when scoring candidate hand motion. The
primary comparison removes clearance refinement completely while retaining the
same four-frame visual proposal, C1 map, checkpoint, IK, terminal servo and
15-step execution. It differs from C1's strict current-frame-only control.

## Geometry uncertainty and persistent state

The frozen backbone supplies a 20×20 metric point grid and 4×4 visual tokens.
The frozen route head projects each visual token to 64 features. A 74→64→1
SiLU network, with **4,865 trainable parameters**, predicts a bounded scalar
positional-error quantile between 5 and 200 mm for each point. Its inputs are
the corresponding visual cell (64), normalized point position (3), local
row/column jumps and curvature (3), point position relative to the measured
camera (3), and distance from the measured TCP (1). No teacher geometry, future
expert motion or collision result enters the deployed head.

C1 still stores actual predicted surface anchors, with the same merging,
capacity and relevance eviction rules. Each anchor retains its creation-time
predicted error. When querying older-only map points, add the square root of
observed positional scatter to this error estimate. Reobserving an anchor does
not replace its uncertainty with that of an unrelated current pixel. Fresh
clouds retain their own point uncertainties. Reset all state between episodes.

Convert predicted error u into padding using

\[
b(u)=\operatorname{clip}((u-0.015)/0.085,0,1),\qquad
\delta_p=\delta_{\max}b(u_p).
\]

For a hand query q and surface point p, replace Euclidean distance with

\[
d_{\mathrm{eff}}(q,p)=\max(\|q-p\|-\delta_p,0).
\]

Use this distance in the existing 40 mm Gaussian proximity field. Higher
uncertainty therefore increases the risk assigned to a nearby surface and can
favor a wider correction. The 14 side/up candidates, TCP and two axial hand
probes, five executed-prefix query times, learned route-trust cost and near-goal
fade are unchanged. The selected correction can differ, but this sampled field
does not impose a hard minimum clearance on realized IK motion.

## Training and controls

Train only the uncertainty head using the existing cache: **7,951 expert plus
2,997 independent perturbation observations**, from the unchanged 800 training
episodes. Existing recorded-depth teacher points supervise the norm of metric
reconstruction error at matching pixels. Mask invalid targets, out-of-workspace
predictions and points within 7 cm of the measured TCP. This uses existing
supervision without collecting additional data. Depth is absent at deployment.

Fit the 0.9 quantile with per-observation mean pinball loss, 65/35 expert/recovery
sampling and 2:4:4 route sampling. Use 10 fixed epochs, 8,192 row draws per epoch,
64 rows per batch, AdamW at 0.001 with weight decay 0.0001, gradient norm limit
1, and seed 20261008. The checkpoint is the fixed final epoch; validation labels
only measure calibration. Neither optimizer nor calibration uses test data.

Evaluate maximum padding 10/20/30 mm on all 100 validation episodes. Select
highest success, breaking ties toward smaller padding. **30 mm is selected**
with 89 successes, versus 88 at 20 mm and 86 at 10 mm. Freeze the checkpoint,
source, candidate and controls before test. Each split contains 20 direct,
40 over and 40 side episodes; primary success is collision-free XYZ reaching
within 1 cm under the original 400-step limit.

Controls retain the same inference checkpoint:

- **No refinement:** return the route proposal unchanged; diagnostics verify
  zero correction and candidate index zero throughout.
- **Fixed clearance:** use C1's exact non-inflated refinement. Reuse its archived
  runs after complete command parity tests; count these as reused evidence.
- **Uniform inflation:** apply one constant padding factor to all points,
  derived from the mean training-set factor (0.519223). The selected-strength
  control therefore uses 15.577 mm everywhere, versus point-specific 0–30 mm.
  This controls the margin scale using the training distribution; deployment
  mean margins need not match exactly.

The original validation panel included uniform inflation with a 20 mm maximum
(10.384 mm constant). After selecting 30 mm, add the matching uniform30 control
before test. This supplemental control does not alter candidate selection.

## Calibration evidence and limits

On 53,883 valid validation points, the trained head predicts mean error radius
81.44 mm versus actual mean error 62.36 mm. Across five bins ordered by predicted
error, actual mean errors are **21.5, 24.3, 32.5, 62.8 and 170.8 mm**. The head
therefore ranks reconstruction quality usefully. Measured coverage is **77.66%**,
below the 90% training target. Treat this output as an uncertainty signal rather
than a calibrated 90% bound or collision probability. The 200 mm prediction cap,
domain shift and stale-map errors remain limitations.

## Complete navigation results

**The requested ≥3-point gain over no clearance refinement is met on both full
splits:** **+7 validation points and +4 test points**. Both selected success
rates exceed 70%.

| Condition | Validation /100 | Test /100 | Test direct /20, over /40, side /40 |
|---|---:|---:|---|
| **Selected uncertainty padding, 0–30 mm** | **89** | **82** | **20 / 36 / 26** |
| No clearance refinement | 82 | 78 | 20 / 37 / 21 |
| C1 fixed clearance, reused with parity | 89 | 81 | 20 / 36 / 25 |
| Uniform padding, 15.577 mm | 87 | 83 | 20 / 36 / 27 |
| Uncertainty padding, 0–20 mm | 88 | Not advanced | — |
| Uncertainty padding, 0–10 mm | 86 | Not advanced | — |
| Uniform padding, 10.384 mm | 87 | Not advanced | — |

The primary paired comparison gains nine validation episodes and loses two,
95% route-stratified bootstrap interval **[+1,+13] points**, exact McNemar
p=0.0654. Test gains seven and loses three, interval **[−2,+10]**, p=0.3438.
These are observed benchmark gains, with uncertainty that includes no test
population improvement. Use 20,000 paired resamples within route strata; the
bootstrap interval and exact test are distinct procedures. Comparisons are
exploratory and uncorrected for multiplicity.

**The extra success benefit of adaptive uncertainty is not established.**
Against non-inflated C1, success differences are 0/+1 points, intervals
[−5,+5]/[−3,+5]. Against matching uniform inflation, they are +2/−1 points,
intervals [−4,+8]/[−5,+3]. Keep the validation-selected candidate rather than
selecting the test-best control. The primary ablation supports clearance
refinement; it does not isolate a ≥3-point gain caused by learned uncertainty.

Mean deployed padding is 15.19/14.78 mm on validation/test, compared with
15.58 mm for the uniform control. Mean nominal corrections are 17.06/18.76 mm.
Collision rates fall from 18/22% without refinement to 10/18%; the selected
validation run has one timeout, and test has none. Mean steps rise from
146.93/143.52 to 156.74/153.35. XYZ-plus-orientation success is separately
43/29%; the primary 89/82% result measures XYZ reaching.

Completed **900 new rollouts** (600 validation, 300 test) plus 200 reused C1
fixed-clearance episode results. All 1,100 condition trajectories pass finite
value, no-expert-progress, fixed-execution and complete-split audits. The
no-refinement control has zero corrections throughout. **75 regression tests
pass**, and at completion active runtime modules matched the archived rollout
source byte for byte. The later C2 study adds a separate controller and CLI
options; C3's frozen source remains the exact replay record. No new data, host
package installation or unrelated image change occurred.
The test split was historically reused, so this remains exploratory evidence;
there are no independent-seed, dynamic-scene or real-camera results.

![C3 results and geometry-error ranking](../runs/c3_uncertainty_20261007/final/c3_results.png)

![Persistent geometry, embodied queries and uncertainty refinement](../runs/c3_uncertainty_20261007/final/c3_architecture.png)

## Reproduction and artifact record

The study is stored at
`/run/user/1016/experiments/gtsn_c3_uncertainty_20261007`. Training and rollout
source snapshots, SHA-256 manifests, input hashes, training records, test freeze,
per-episode trajectories and action diagnostics preserve the executed method.
The uncertainty checkpoint references the unchanged energy adapter and frozen
backbone rather than duplicating either.

Replay the selected method into a new output directory:

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 evaluate \
  --controller uncertain_clearance --refinement-mode adaptive --uncertainty-padding 0.03 \
  --checkpoint /run/user/1016/experiments/gtsn_c3_uncertainty_20261007/training/uncertainty.pt \
  --partition test --no-render-videos \
  --output-dir /run/user/1016/experiments/c3-replay-new
```

Set `--refinement-mode none`, `fixed` or `uniform` for controls. The uniform
factor is loaded from the training checkpoint. All numerical training,
simulation, analysis and regression tests run in the existing project Docker
image. Host launchers use only the standard library.

For a fresh study, train using the existing energy checkpoint and cache:

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 uncertainty \
  --energy-checkpoint /run/user/1016/experiments/gtsn_geometric_energy_20261007/training/energy.pt \
  --cache /run/user/1016/experiments/gtsn_persistent_cache_20261005 \
  --output-dir /run/user/1016/experiments/c3-study-new/training
python3 scripts/run_c3_study.py --root /run/user/1016/experiments/c3-study-new --stage validation
```

Inspect validation results and use its selected candidate for the test stage;
the driver enforces the selection rule and ≥3-point validation gate. For the
recorded selection:

```bash
python3 scripts/run_c3_study.py --root /run/user/1016/experiments/c3-study-new \
  --stage test --candidate adaptive30 --conditions none,fixed,adaptive30,uniform30
python3 scripts/run_c3_study.py --root /run/user/1016/experiments/c3-study-new \
  --stage report --candidate adaptive30
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --cpu c3_report \
  --root /run/user/1016/experiments/c3-study-new --candidate adaptive30 \
  --output-dir /run/user/1016/experiments/c3-study-new/final
```

The fixed-clearance reference defaults to the completed C1 archive. A fresh
equivalent reference can be supplied with `--reuse-fixed`. Incomplete or changed
reference/source/checkpoint artifacts are rejected; existing outputs are reused
only when marked complete, and test nomination cannot change between invocations.
