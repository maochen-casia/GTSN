# GTSN: persistent surfaces, embodied queries, and route trust

This document specifies the initial four-cloud energy study. The completed C1
follow-up replaces the fixed history limit with an anchored persistent map and
adds a strict current-frame-only comparison; see
[c1_persistent_geometry.md](c1_persistent_geometry.md). It reuses the trained
adapter described here, reaching 89/81 versus 79/75 for that new control.
The subsequent [C3 study](c3_uncertainty_clearance.md) freezes this adapter and
adds uncertainty-dependent obstacle inflation, reaching 89/82 versus 82/78
without clearance refinement. Its new uncertainty head has 4,865 parameters;
the equations and original ablations below describe the initial model.

This implementation asks how an RGB-derived point map can become useful action
information under partial views and reconstruction error. Its three interfaces
are **retain surfaces → query proposed hand motion → balance clearance with
learned task progress**. It extends the existing clearance policy with a trained,
context-dependent route-preservation cost. Surface history, hand probes and the
candidate lattice reuse established project components; they are not claimed as
independently new mapping or planning algorithms.

## Three challenges and corresponding system contributions

1. **A changing camera hides previously observed surfaces.** Keep the last four
   RGB-predicted surface clouds in robot-base coordinates. This is short episodic
   geometric persistence, distinct from the proposal's four-frame visual
   attention. Removing surface history leaves that attention intact, allowing
   the geometric contribution to be tested separately.
2. **Point clearance does not describe a moving hand.** Evaluate each proposed
   motion at control steps 3, 6, 9, 12 and 15, with probes at the TCP and 5/10 cm
   behind it along the proposed wrist axis. The resulting action-conditioned
   field makes surface geometry actionable. A TCP-only control tests hand extent.
3. **Imperfect geometry can redirect a useful route excessively.** Combine
   proximity with a positive, learned route-preservation cost. Its coefficient
   depends on the goal, proposed progress and geometric context, and stays within
   [0.02, 0.12]. A zero-cost control tests preservation; a fixed-0.08 control
   separately tests whether learning improves the existing clearance baseline.

## Explicit geometric energy

The frozen RGB model predicts XYZ in the robot frame. Sample a 20×20 cloud using
nearest-pixel interpolation; exclude predictions outside x=(0.10,1.05),
y=(−0.60,0.60), z=(0.04,0.65) m. Remember four clouds, resetting at every episode.
Suppress surfaces within 7 cm of the measured TCP when querying the field.
Unknown or invalid evidence has zero proximity cost; it is not certified free
space. No depth, scene object geometry, expert route or collision oracle enters
the deployed energy.

For candidate c, executed-prefix time t and hand sample h, define query q(c,t,h).
Its proximity is the largest Gaussian response from any valid remembered point:

\[
R_c=\frac1{15}\sum_{t\in\{3,6,9,12,15\}}
\sum_{h\in\{0,.05,.10\}}
\max_{p\in M_t}\exp\left(-\frac{\|q(c,t,h)-p\|^2}{2(.04)^2}\right).
\]

The empty-set maximum is zero. This is a surface-proximity heuristic, not a
signed-distance field, calibrated collision probability or safety guarantee.

Fourteen candidate offsets include zero, signed lateral corrections, upward
corrections and lateral/up combinations. They share the frozen route and wrist
proposal. Corrections ramp along the 30-step route, vanish within 2.5 cm of the
goal, and reach full nominal strength at 8 cm. The largest nominal offset is
sqrt(0.05²+0.04²)=6.40 cm; the executed first 15 steps receive a smaller shift.

The decision is

\[
c^*=\arg\min_c\left[R_c+\lambda_\theta(x)
\left(\frac{\|\delta_c\|}{.05}\right)^2\right],\quad
\lambda_\theta(x)=.02+.10\sigma(f_\theta(x)).
\]

The 8→32→1 SiLU network has 321 parameters. Its eight inputs are goal-minus-TCP
XYZ divided by 0.3 m, the proposed step-15 displacement in metres, the zero
candidate's proximity and the minimum candidate proximity. Initialization gives
lambda=0.08. All perception and route parameters remain frozen. The shared IK,
terminal servo and fixed 15-step execution are unchanged.

## Training without data changes

Use `/run/user/1016/experiments/gtsn_persistent_cache_20261005`: 7,951 expert
observations and the original 2,997 independent perturbation-only recovery
observations. There is no new corrective collection or on-policy supervision.
Expert history uses the last four cached causal observations and their actual
timestamps. Recovery observations reset history. Source sampling is 65/35 and
route sampling is 20/40/40. Depth teacher arrays are not read by the adapter.

For each existing observation, form candidate positions and their mean squared
error against all six expert route knots. Convert those errors into a soft
candidate target using temperature (0.015 m)². Train the energy distribution
softmax(−cost/0.03) by cross entropy, with 0.1*((lambda−0.08)/0.04)² regularization.
Use AdamW (lr 0.001, weight decay 1e−4), gradient clipping at 1, batch size 256,
32,768 sampled observations per epoch, 20 epochs and seed 20261007. Epoch 20 is
fixed before rollouts; test outcomes never select a checkpoint.

Training approximates the future wrist axis with the observation's current TCP
rotation because the cache lacks the backbone's future joint proposals.
Deployment uses the unchanged backbone's proposed wrist rotations. Cached metric
points and deployed mixed-precision decoding also need not be bitwise identical.
These are explicit approximation limits of this small adapter, not additional
training data or access to deployment ground truth.

## Evaluation and attribution

Reuse the fixed 800/100/100 episode split. Its route counts are 160/320/320 in
training and 20/40/40 in each evaluation partition. Primary success is collision-
free XYZ reaching within 1 cm under the inherited 400-step limit. Report the
stricter XYZ-plus-orientation metric separately.

Evaluate full, no_history, tcp_only, no_trust and fixed_trust on all 100
validation and 100 test episodes. The same trained adapter checkpoint and
proposal are used throughout. These are inference-time component removals;
they establish dependence of this system, not optimality of independently
retrained alternatives. Report paired route-stratified bootstrap intervals and
exact McNemar tests; do not describe positive point estimates as significant
when intervals include zero. The historically reused test split is exploratory.

## Position relative to prior research

Visual reconstruction and downstream navigation are separate problems.
[Pi3](https://arxiv.org/abs/2507.13347) studies geometry reconstruction; this
project reuses its encoder within the existing trained metric-prediction model.
[ConceptGraphs](https://arxiv.org/abs/2309.16650) builds semantic 3D representations
for planning; the present interface focuses on local physical motion queries.
[CHOMP](https://www.cs.cmu.edu/~mzucker/icra09-chomp.pdf) combines obstacle costs
and trajectory optimization; the current system scores a small bounded candidate
lattice around an existing learned route. These comparisons motivate the design;
they are not experimental superiority claims or an exhaustive novelty search.

The scoped research gap is the connection between partial predicted geometry and
actual navigation benefit: retained evidence, body-conditioned queries and task
progress must be evaluated together through complete outcomes. A larger map or
lower reconstruction error alone does not establish that connection.

Implementation: `src/tsn/models/geometric_energy.py` and
`src/tsn/cli/geometric_energy.py`. The host workflow is
`scripts/run_geometric_energy_study.py`; numerical work and simulation run only
in the existing project Docker image.

Completed results and the strength of each claim are in
[research_story.md](research_story.md). The full model scores 87% validation and
81% test. All three removal ablations reduce validation success, while test
effects are mixed; learning does not improve test success over fixed trust.
