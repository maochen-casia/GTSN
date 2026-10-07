# C2: regional hand and rigid-tool geometry

The follow-up replaces a single TCP distance with candidate-conditioned contact
features for the **TCP, palm, left finger, right finger, rigid wrist and camera
housing**. It uses known robot collision geometry and measured finger opening,
without adding training data or learned parameters. The selected regional model
scores **91/100 validation and 89/100 test**, versus **87/100 and 85/100** for
matched TCP-only modelling. Its pre-test nominee, `tool_parts100`, meets the
requested threshold with **+4 percentage points on both full splits**.

## Representation and action interface

Three axial probes cannot express the palm's width, finger opening or camera
housing. The module reads the Panda collision URDF bundled in the existing
ManiSkill image. It traverses fixed transforms from the TCP, stops at actuated
arm joints, and includes the two measured prismatic finger joints. Palm and
rigid link7 collision meshes become convex support-plane fields. Each finger
uses its four oriented collision boxes; the camera uses its collision box.
Assets and their SHA-256 hashes are recorded in each rollout configuration.

The palm spans approximately 204 mm across its widest TCP-frame dimension.
Finger positions come from measured state, rather than a fixed opening. The
regional model keeps the route's wrist orientation proposal and commands no
additional wrist roll or finger motion. For each positional candidate, transform
scene points into its proposed TCP frame at executed-prefix steps 3/6/9/12/15.
This gives one point-to-body distance field per region, rather than reducing the
whole body to a single nearest-region distance.

Let d_k(p,t,a) be nonnegative distance from surface point p to region k at future
time t under candidate a. Retain C3's point-specific padding delta_p, with a
30 mm maximum, and its 40 mm Gaussian contact response:

\[
\phi_k(p,t,a)=\exp\left[-\frac{\max(d_k(p,t,a)-\delta_p,0)^2}
                                  {2(0.04)^2}\right],\qquad
R_{\mathrm{parts}}(a)=\frac1{5K}\sum_t\sum_k\max_p\phi_k(p,t,a).
\]

K is four for hand-only and six for the full rigid tool, including the TCP
virtual reference. The mean preserves contact extent across regions after the
maximum over scene points. A union maximum can saturate from a single body
contact and discard that extent. The regional mean remains a heuristic: an
isolated dangerous contact can be diluted, and it is not a collision constraint.

Validation compares blends

\[
R(a)=(1-\beta)R_{\mathrm{TCP}}(a)+\beta R_{\mathrm{parts}}(a).
\]

It selects **beta=1**, so the final cost uses the six regional features directly.
The unchanged frozen route-trust adapter adds a positive quadratic positional
correction cost. Choose from the same 14 side/up offsets, fade corrections near
the goal, apply the same damped IK and terminal servo, and execute 15 steps.
Distances describe proposed TCP/body poses; realized IK motion can differ.

At perception time, reject predicted points within 2 mm of the measured current
robot body before inserting them into C1's persistent map or retaining fresh
clouds. Otherwise a visible robot surface can persist as a fictitious obstacle.
The TCP-only control skips this body-based filtering entirely. Both still use
the original workspace crop and 7 cm TCP exclusion. The reported raw self-point
classification overlaps these pre-existing masks; it is not the net count removed.

Convex support-plane distance is exact inside each convex hull and a conservative
lower bound on Euclidean distance outside it; corner distance is approximate.
Oriented-box signed distance is exact. This models the hand and **rigid** wrist
attachment, excluding the articulated arm links. Sparse RGB points, imperfect
reconstruction and five temporal samples preclude a hard safety guarantee.

## Matched controls and staged selection

All C2 conditions share the selected C1 persistent map, four visual features,
C3 uncertainty checkpoint and 30 mm maximum padding, proposal/head/trust weights,
candidate positions, IK, servo, 400-step limit and 15-step execution cadence.
The deployed controller consumes RGB predictions and measured robot/camera/goal
state. It has no scene depth, simulator contacts or expert progress input.
Existing expert plus independent perturbation training data is unchanged.

**TCP-only** queries exactly one virtual point; it neither evaluates physical
body volumes nor applies body-based self filtering or posture refinement.
**Axial** reproduces the previously completed C3 model exactly, with three
virtual probes. Complete-command parity tests support reuse of that reference.
The primary C2 comparison includes both regional scoring and body-based self
filtering; it does not isolate either mechanism alone.

Four source-frozen validation panels were evaluated in order. Every score uses
all 100 validation episodes, with 20 direct, 40 over and 40 side scenes:

| Panel | Validation scores, successes /100 | Decision |
|---|---|---|
| Gaussian body union, no self filtering | TCP 87; axial 89; hand beta .35/.70: 86/84; tool .35/.70: 86/85 | No body candidate meets +3; no test |
| Self-filtered union and posture library | Fixed hand/tool .35: 87/87; hand roll .35: 85; wide tool roll .35/.70: 73/58 | No candidate meets +3; no test |
| Signed union contact cost | TCP 86; axial 89; hand .35/.70: 91/88; tool .35/.70: 89/90 | Hand .35 nominated before test |
| Separate regional contact features | TCP 87; axial 89; hand .35/.70: 88/88; tool .35/.70/1: 89/89/**91** | **Tool beta 1 nominated before test** |

Signed union uses softplus((padding - signed_distance)/0.04)/log(2), with the
**same response function for its TCP-only control**. It retains negative
penetration rather than saturating at zero distance. Its hand nominee scores
91/81 versus TCP 86/79: **+5/+2**, failing the test target. The prespecified
full-tool secondary scores 89/83: **+3/+4**, meeting the observed target as a
secondary result. Its label and the unsuccessful primary are preserved.
The regional prototype was prepared before those test outcomes; its separate
panel and nomination record do not replace the signed primary retrospectively.

For each successful validation panel, select highest hand/tool success, break
ties toward smaller beta, simpler posture, then hand-only. Freeze source,
checkpoint, nominee and test controls before test evaluation. The regional
nominee beats TCP by four validation points and is tested with TCP and axial
controls. Reused TCP validation and axial results are recorded as reused, rather
than counted as new rollouts.

## Complete navigation results

**The requested primary gain is achieved: +4 validation and +4 test points.**
Both selected success rates exceed 70%. Primary success means collision-free XYZ
reaching within 1 cm, rather than the stricter XYZ-plus-orientation metric.

| Condition | Validation /100 | Test /100 | Validation direct /20, over /40, side /40 | Test direct /20, over /40, side /40 |
|---|---:|---:|---|---|
| **Selected regional hand + rigid tool, beta 1** | **91** | **89** | **20 / 37 / 34** | **20 / 37 / 32** |
| Matched Gaussian TCP-only | 87 | 85 | 20 / 37 / 30 | 20 / 37 / 28 |
| Previous C3 axial model, reused with parity | 89 | 82 | 20 / 37 / 32 | 20 / 36 / 26 |

Against TCP, validation gains five scenes and loses one; test gains six and loses
two. Paired 95% route-stratified bootstrap intervals are **[0,+9] and [−1,+10]
points**, with exact McNemar p=0.2188/0.2891. Use 20,000 paired resamples within
route strata. These satisfy the requested observed threshold; the intervals do
not establish a positive population effect. Comparisons remain exploratory and
uncorrected for multiple architecture/strength comparisons.

Relative to the old axial model, gains are +2/+7 points, with intervals
[−3,+7]/[+3,+12]. These are supplemental comparisons, not the requested control.
Keep the validation nominee; do not pick a model using its test score.

Collision rates decline from **12/15% to 9/11%** on validation/test; one TCP
validation timeout becomes zero. Mean final position error falls from
32.46/34.47 mm to 26.10/27.56 mm. Mean steps decline from 159.25/159.02 to
156.06/155.80. Mean nominal corrections are 17.02/17.29 mm, versus TCP's
20.64/21.66 mm. Full XYZ-plus-orientation success is separately **44/31%**
versus TCP's 37/31%; no test orientation gain is claimed.

Completed **2,500 new C2 rollouts**: 2,000 validation and 500 test. The complete
study audits **2,700 unique trajectories**, including 200 reused C3 axial
results; 3,300 condition-episode instances include repeated links to shared
controls. All pass finite-value, no-expert-progress, fixed15 and full stratified
split checks. All frozen panel source hashes pass. Active regional runtime
modules match the rollout source byte for byte. The preceding unsuccessful
validation panels and signed primary miss are retained as selection evidence;
there are no unused new checkpoints or package/image installations.

![Matched C2 success rates](../runs/c2_part_geometry_20261007/final/c2_results.png)

![Known hand and rigid-tool collision geometry, measured finger positions 20 mm each](../runs/c2_part_geometry_20261007/final/c2_tool_geometry.png)

![Regional geometry-to-action interface](../runs/c2_part_geometry_20261007/final/c2_architecture.png)

## Reproduction and artifacts

The regional study is at
`/run/user/1016/experiments/gtsn_c2_part_geometry_20261007`; earlier selection
panels are retained as required evidence under `gtsn_c2_embodied_20261007`,
`gtsn_c2_body_pose_20261007` and `gtsn_c2_signed_geometry_20261007`.
Each has frozen source and hashes, protocol, logs and full per-episode traces.
The regional `test_freeze.json` names `tool_parts100` before any test result.
C2 adds no checkpoint; it reuses the unchanged C3 uncertainty adapter and parents.

Replay into a new output directory:

```bash
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --gpu 0 evaluate \
  --controller embodied_clearance --body-mode tool --body-weight 1 \
  --body-self-mask --body-pose fixed --body-field gaussian --body-representation parts \
  --uncertainty-padding 0.03 \
  --checkpoint /run/user/1016/experiments/gtsn_c3_uncertainty_20261007/training/uncertainty.pt \
  --partition test --no-render-videos \
  --output-dir /run/user/1016/experiments/c2-regional-replay-new
```

For matched TCP-only use `--body-mode tcp --body-weight 0 --no-body-self-mask`;
for the historical C3 control use `--body-mode axial`. All numerical work,
simulation, tests and figures run inside the unchanged project Docker image.
Host launchers use only the standard library, with no host package installation.

For a fresh equivalent panel:

```bash
python3 scripts/run_c2_study.py --root /run/user/1016/experiments/c2-study-new \
  --stage validation \
  --conditions tcp,axial,hand_parts35,hand_parts70,tool_parts35,tool_parts70,tool_parts100
python3 scripts/run_c2_study.py --root /run/user/1016/experiments/c2-study-new \
  --stage test --candidate tool_parts100 --conditions tcp,tool_parts100,axial
python3 scripts/run_c2_study.py --root /run/user/1016/experiments/c2-study-new \
  --stage report --candidate tool_parts100
python3 scripts/docker_run.py --image gtsn-persistent:20261005-compact-only --cpu c2_report \
  --root /run/user/1016/experiments/c2-study-new --candidate tool_parts100 \
  --output-dir /run/user/1016/experiments/c2-study-new/final
```

The driver enforces the actual validation nominee and gate, so a fresh run may
require a different nomination. Axial reuse defaults to the completed C3 archive.
Use `--source-snapshot <study>/source` for frozen rollout replay. Existing output
directories and changed frozen source/checkpoint/nomination are rejected.

**89 regression tests pass**, including metric body distance, measured opening,
rotation covariance, region/union consistency, self-point rejection before map
insertion, empty/nonfinite evidence, control parity, episode resets and complete
finite commands. Source parity additionally exercises 27 synthetic observations
per mode through map/history/uncertainty/body scoring and IK; this is not a Pi3
or simulation replay. Test results remain exploratory because this benchmark
has been inspected historically. There is no independent-seed, new-scene or
real-camera confirmation.
