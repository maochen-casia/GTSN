# Geometry-grounded Table Scene Navigation

## Research hypothesis and scope

**Current validated design (revision 2):** use a hybrid Cartesian/joint policy.
A learned state-conditioned decoder predicts TCP route waypoints; the frozen
RGB Pi3 map policy supplies wrist orientations and joint-space initialization;
known Panda kinematics reconcile the two. A shared terminal goal servo replaces
learned commands inside 8 cm while retaining the original 1 cm success test.
This model was selected at 80% success on all 100 validation episodes and
achieved **72% success on the fixed 100 test episodes**, versus 54% for Pi3 with
the same servo (paired gain +18 percentage points, 95% interval +8 to +29,
exact McNemar p=0.0021). The earlier selected pilot achieved 7%. The
uncertainty and memory variants remain evaluated alternatives, not established
improvements: their corresponding validation rates are 74% and 79%, and test
rates 79% and 76%. The test ranking does not replace validation selection. The
following original pilot sections describe the development history; the
revision-2 section specifies the subsequent implementation and controls.

The baseline's six predicted maps are an information bottleneck: the validation
diagnosis finds a state-only adapted head competitive with predicted maps.
GTSN therefore retains visual features alongside explicit geometry. Geometry is
an uncertain, task-conditioned memory used to predict actions, rather than the
only path from the image to the action. The initial version of this document
was written before implementation. Measurements are recorded separately in
`gtsn_experiments.md`.

The first implementation reuses the trained Pi3-small perception checkpoint
and the existing simulator, splits, recovery pool, and 30-step action targets.
Perception is frozen for a controlled, affordable first experiment. Its 768-D
decoder features provide a direct visual path around the map bottleneck. All
new heads are trained on the training partition only. Existing baseline test
results have already been inspected; new comparisons are exploratory rather
than a previously untouched benchmark evaluation.

The completed validation pilot does not justify adopting this frozen-feature
design as the final research model. State correction wins at 10% success; the
visual, uncertainty, and memory candidates achieve 4–9% depending on the variant
and schedule.
The full passive memory model achieves 7%. Uncertainty ranks geometry errors
but under-covers, and adaptive timing adds observations without improving
success. The later revision section specifies the implemented controller and
action representation changes. Ray-constrained depth, spatial occupancy memory,
held-out-view reconstruction, and active viewpoint selection remain proposals.

## Observation and representation

Deployment observes wrist RGB, current joints, goal pose, camera intrinsics,
and the measured camera-to-base transform. Depth, scene objects, route labels,
expert futures, and expert progress indices never enter the policy. Simulator
scene geometry is used only for rendering and physical collision evaluation.

Pool the decoder feature grid to 4 by 4 tokens. Each token retains the pooled
predicted six-channel maps and hence point position in the robot base frame,
goal relevance, and predicted future-action relevance. This small representation
is an initial resolution choice; it does not resolve thin obstacles reliably.
The current robot/goal state creates an action query. Cross attention retrieves
visual and geometry tokens relevant to that query. A residual action head adds
a learned bounded correction to the frozen baseline's action chunk, preserving
its initial behavior while allowing direct feature-to-action learning.

## Explicit uncertainty

A visual-token head predicts a correction to pooled point XYZ and diagonal
log variance, bounded to avoid numerical degeneracy. Supervise the mean and
variance with Gaussian negative log likelihood against pooled teacher point
means. This estimates errors in token-average geometry, not individual-pixel
surface uncertainty. Variance is conditional predictive error, without a claim
that aleatoric and epistemic components are separately identified.

Detach the variance in the attention reliability penalty, preventing action
loss from artificially shrinking uncertainty. Geometry NLL still trains the
uncertainty head. Compare empirical squared error with predicted variance and
report coverage of marginal 1.96-sigma intervals on held-out episodes.
Teacher geometry is a training label only. Expert future-action maps are never
substituted for predicted input maps.

## Persistent, causal observation memory

Keep the last four observed keyframes, each with sixteen visual/geometry
tokens, the measured camera pose, and its observation time. Base-frame XYZ and
camera-pose embeddings let attention relate tokens across viewpoints. Age
embeddings and a recency penalty reduce reliance on stale observations.
The memory is cleared at every episode boundary. During training, histories
come only from earlier frames of the same episode, approximately fifteen
control steps apart. Recovery observations have no genuine observed histories
and therefore use singleton memory; fabricated expert histories are forbidden.

Novelty is the distance from each current predicted surface token to its nearest
historical surface token. It is a noisy indicator of newly seen geometry, not
proof that a region is free or an exact visibility/occupancy map. Task-conditioned
attention combines novelty, geometry confidence, and visual appearance. This
bounded keyframe memory is the implemented persistent scene representation;
long-term world-grid fusion and dynamic-object tracking remain future work.

## Active sensing, staged honestly

The implemented first stage actively chooses the next observation time. If
task-attended geometry uncertainty is high, execute five actions before
observing again; otherwise execute fifteen. The threshold is calibrated using
training observations. Report observation count, success, collision rate, and
latency, including a fixed-five-step control for the same model. This comparison
distinguishes adaptive scheduling from simply increasing observation frequency.

This does **not** implement active viewpoint selection or information-gathering
detours. A later stage would decode geometry at candidate reachable camera
poses, estimate route-weighted uncertainty reduction, and score it against
motion cost and collision risk. That stage should only be added after memory
and uncertainty demonstrate navigation value. Past/future geometry decoding,
contrastive viewpoint ranking, fine-grained keypoints, and explicit unseen-space
occupancy are likewise proposed extensions, not completed contributions.

## Experiment protocol

Retain the existing disjoint episode splits: 800/100/100 with 2:4:4 routes in
each partition. Reuse the Pi3 checkpoint trained on the 800 training episodes;
its selection previously used validation RMSE. Cache frozen features for all
partitions independently, with episode/frame metadata and source hashes.
Training uses expert stride two plus safe RGB perturbation recovery, sampled
65:35 with 20:40:40 route probabilities within each source. Validation uses all
frames. Test observations never participate in training or model selection.

Compare matched residual-head training conditions: state-only correction; current visual
features plus deterministic geometry; current features with uncertainty; and
uncertainty with causal memory. Use the same sampling seed, ten epochs, batch
size 128, AdamW, gradient clipping, and validation-RMSE checkpoint selection.
Evaluate all four on all 100 validation episodes in closed loop. Compare the
memory model at fixed fifteen, adaptive five/fifteen, and fixed five execution
steps. Choose the final condition using validation closed-loop success, then
collision rate, then RMSE; run all 100 test episodes once for that condition.
Report the full validation table, including unsuccessful additions.

The state-only correction control retains the common frozen RGB/map policy as
its base prediction; only its newly trained correction is state-only. It is not
a policy whose entire input is state. This isolates the value of visual features
in the new head while matching baseline initialization across all four variants.
The Gaussian NLL coefficient is 0.002; correction magnitude is bounded to 0.35
radians per joint/horizon. All heads use 128-dimensional tokens, a 256-dimensional
output hidden layer, learning rate 0.0003, weight decay 0.0001, and cosine decay.
Uncertainty labels exclude pooled cells with any invalid depth sample.

## Implemented equations

For frozen visual token $v_i$ and pooled predicted point $p_i$, a shared visual
projection $h_i$ drives six probabilistic outputs $(d_i,s_i)$:

$$\mu_i=p_i+0.25\tanh(d_i),\qquad
\log\sigma_i^2=-5+4\tanh(s_i).$$

Coordinates use the baseline's base-frame center $(0.65,0,0.22)$ m and axis
scales $(0.55,0.55,0.50)$ m. The token label $y_i$ averages the teacher's
normalized, clipped points in the corresponding 20 by 20 map cell. Geometry
loss averages valid coordinates of

$$\mathcal L_{geo}=\tfrac12[(y_i-\mu_i)^2/\sigma_i^2+\log\sigma_i^2].$$

The Gaussian constant is omitted. Geometry statistics are FP32 even when visual
projections use bfloat16. This objective calibrates pooled-coordinate errors;
it does not quantify all uncertainty in a complete obstacle surface.

Let $z_i$ combine visual features with corrected geometry, the three predicted
task-map channels, camera pose, normalized age, and novelty. The state query
$q$ reads memory using

$$a_i=\operatorname{softmax}_i\left(q^T Kz_i/\sqrt{128}
-0.1\,age_i/60-0.5\,\operatorname{mean}_c[\operatorname{stopgrad}(\log\sigma_{ic}^2)]\right).$$

Unavailable history entries receive negative-infinite logits. Deterministic
visual ablations omit the uncertainty term and point correction. The context
$c=\sum_i a_i Vz_i$ and state query predict a 30 by 7 correction:

$$\hat A=A_{frozen}+0.35\tanh(MLP([q,c])).$$

The last linear layer starts at zero, making all conditions equal to the frozen
baseline at epoch zero. The state-correction control sets $c=0$. The total loss
is the baseline horizon-weighted Huber action loss plus $0.002\mathcal L_{geo}$
for the uncertainty variants. Thus the uncertainty ablation adds both mean
refinement and reliability weighting; it does not isolate weighting alone.

Adaptive sensing uses $r=\sqrt{\sum_i a_i\operatorname{mean}_c\sigma_{ic}^2}$
and the train-only 65th-percentile threshold. It changes observation timing,
without an extra information-gain training objective. Fixed-five and adaptive
rollouts use shorter histories in elapsed time than the fixed-fifteen training
histories; explicit age features expose that shift, but do not eliminate it.

Save training curves, uncertainty calibration, per-episode rollout metrics,
trajectories, selection decisions, split counts, runtime, and source hashes.
Use paired episode comparisons for validation ablations and Wilson intervals
for test success. One training seed is a pilot and does not establish robustness
across seeds. Never describe an improvement or contribution as empirically
supported unless the corresponding measurements support it.

## Revision criteria

If direct features do not improve navigation, examine perception domain shift
between textured expert images and reconstructed simulator images before making
the memory more complex. If uncertainty is poorly calibrated, use train-only
variance scaling or ensembles. If memory hurts, test reduced history and better
spatial alignment. If adaptive observation loses against fixed-five execution,
retain fixed scheduling and report that active timing was not beneficial.
End-to-end perception fine-tuning, higher image resolution, and on-policy data
collection are follow-up experiments, not assumed successes of this first run.

All numerical execution and dependencies stay inside the existing project
Docker image. Source changes stay in `/home/chenmao/GTSN`; experiment outputs
stay in `/run/user/1016/experiments`. Other containers and images are untouched.

## Next design if the frozen-feature pilot is insufficient

The next experiment should change the perception/representation bottleneck
before increasing the size of the residual head. The following is a proposed
revision, not an implemented or evaluated model:

1. **Align observation domains.** Re-render training demonstrations at their
   recorded joint states with the evaluation renderer, or restore the original
   textures and lighting consistently. Compare both choices on validation,
   without changing physical geometry or success criteria. Fine-tune perception
   with appearance augmentation and a larger image resolution. First establish
   a competitive single-frame RGB policy under the same rendering conditions.
2. **Predict geometry along calibrated rays.** Replace unconstrained XYZ heads
   with positive camera-depth distributions and backproject their moments using
   measured intrinsics and poses. Compute projected-goal relevance analytically
   from the given goal and calibration, rather than asking a network to relearn
   that exact transformation. Infer visible-goal probability by integrating the
   uncertain surface-depth distribution; no measured deployment depth is used.
3. **Persist spatial evidence with explicit unknown cells.** Fuse ray evidence
   into a base-frame sparse occupancy memory, distinguishing unknown, observed
   free, and occupied space. Keep visual features, observation age, and effective
   observation count. Correlated frames must not count as independent precision
   gains. Contradictory recent evidence should increase uncertainty or replace
   stale content. Retain fine surface keypoints near predicted action corridors,
   where pooling different surfaces is particularly harmful.
4. **Supervise memory through held-out views.** From strictly past RGB history,
   decode geometry at a past held-out camera pose and a later training pose.
   Future depth is a supervision target only. Pose-query decoding at deployment
   uses candidate poses computed from known robot kinematics, without future
   images or expert motion. Reserve a stratified subset of training episodes for
   calibration; measure coverage separately on newly seen and previously seen
   surfaces and under the deployment appearance distribution.
5. **Select views by task value.** Generate short, reachable candidate action
   chunks and their camera poses. Weight predicted information gain by proximity
   to the policy's own proposed TCP/arm corridor and goal, subtract motion/time
   cost, and reject candidates exceeding learned collision-risk constraints.
   Candidate utility supervision can compare uncertainty/error before and after
   actual later observations in training trajectories. An information-gain proxy
   must be validated against realized error reduction; entropy alone can reward
   irrelevant views or miscalibrated predictions. Compare with fixed-frequency
   sensing, passive memory, and equally costly non-informative view changes.

This revision addresses the pilot's specific limitations: frozen perception,
coarse surface averaging, absent unknown-space representation, and observation
timing without viewpoint utility. Each stage requires its own controlled
closed-loop validation before it is adopted. Near-goal convergence should also
be diagnosed independently of obstacle avoidance; any kinematic servo added to
the controller must be applied equally to every ablation.

## Revision 2: Cartesian route prediction and kinematic tracking (2026-10-03)

The negative pilot motivates a controlled change to the action representation.
The next experiments predict six TCP waypoints at offsets 5, 10, ..., 30 control
steps, in metres relative to the current TCP. Known Panda forward kinematics
convert training joint targets into supervision. At deployment, differentiable
forward kinematics and damped least-squares inverse kinematics convert predicted
waypoints into joint targets. This uses only measured joints, robot calibration,
RGB, and the requested goal. Neither scene objects, depth, route labels nor
expert progress are supplied to the controller.

The query includes measured TCP position, goal displacement, and proprioception;
visual inputs remain the common Pi3 features for the first controlled experiment.
Compare a state-only waypoint decoder, a current-view decoder, uncertainty-aware
geometry, and causal four-view memory. Train the new decoder from scratch rather
than constraining it to a residual of the failed joint-action head. Supervise
coordinate uncertainty with the same proper Gaussian score as the first pilot.
A shared short-range goal servo is an explicit controller factor, evaluated on
the old joint-action policy as well. It does not establish collision-free space.
All approaches retain the benchmark's collision termination and 1 cm goal test.

Reuse the fixed training/validation/test split and cached training features;
preserve the first experiment directory. Choose checkpoints by validation
waypoint error, then controller settings and model variants by closed-loop
validation success. Use a stratified 20-episode validation screen, followed by
all 100 validation episodes for candidates, before final held-out testing.
Numerical results, exact hyperparameters and any further revision will be logged
under a new experiment root. A gain must be large in closed-loop success and
supported by paired episode-level confidence intervals; lower imitation error
alone is insufficient. Renderer changes are a separate experimental factor,
because changing appearance concurrently would confound attribution.

The completed 20-episode screen isolated a large terminal-controller effect:
Pi3 with the 8 cm servo succeeded on 13 episodes versus 2 without it. The
Cartesian variants succeeded on 6--9 episodes, so they are not yet competitive
with the strengthened baseline. Before adopting a new route representation,
apply exactly the same servo to the original pilot's state residual, visual,
uncertainty and memory heads. This additional matched comparison tests whether
the earlier negative module results were masked by convergence failures.
Full validation, rather than the small screen, determines the final selection.
For the mixed-representation comparison, rank by success, then fewer collisions,
then fewer newly trained parameters, then condition name. Joint-angle and
Cartesian imitation losses have different units and cannot break a tie fairly.

An additional Cartesian control tests an identified limitation of the first
tracking design: holding the current wrist orientation changes the arm's swept
volume even when TCP positions are accurate. This control uses the common
Pi3 joint chunk to predict each waypoint's orientation and initialize IK, while
the learned Cartesian decoder still supplies positions. It requires no new
supervision or privileged inputs. Screen it on the same 20 validation episodes;
promote it to full validation only if it beats the initial Cartesian variant.
It reached 14/20 successes (70%) versus 9/20 (45%) with fixed wrist orientation,
so all four Cartesian decoders are now compared on full validation with this
shared orientation controller. The state-waypoint control in this comparison
still obtains orientation from the RGB baseline; it is not a wholly state-only
navigation policy. These four candidates join the six existing full-validation
conditions before any new test rollout is used for selection.

The final test comparison is fixed before those four validation runs finish:
evaluate all four orientation-aware Cartesian heads, the strongest joint-action
variant on validation (current-view visual correction, 71%), and Pi3, all with
the same 8 cm servo. The validation-selected model remains the selected model
even if another ablation scores higher on test. Report paired episode bootstrap
intervals and exact McNemar tests against the baseline and relevant component
ablations; mark multiple component comparisons exploratory. The older 6% Pi3
and 7% first-pilot test results are historical paired references. This is an
exploratory reuse of the fixed benchmark test set, not a new untouched dataset.

### Terminal controller

Let `p(q)` be the calibrated Panda TCP position and `g` the supplied goal. Outside
an 8 cm ball, execute the learned 30-step joint chunk with 15-step replanning.
Inside that ball, construct `p_h = p(q) + (h/30)(g-p(q))`, for `h=1,...,30`.
Solve bounded damped least-squares IK, initialized at measured joints, keeping
the current wrist orientation. The Jacobian weights position by 1 and rotation
by 0.3, uses damping squared `1e-4`, limits each solver update to 0.12 rad, and
performs 12 iterations. Joint targets remain inside URDF limits with a 0.01 rad
margin. The same benchmark position drives execute them; collision checking
continues at every physics substep. Recompute the servo from measured state at
every replan. This controller does not receive scene geometry, depth, or an
expert endpoint configuration, and does not certify that a straight segment is
collision-free. Its purpose is stable terminal goal convergence.

For Cartesian route prediction, linearly interpolate the six predicted knots
through the current position. The initial variant tracks all knots with current
orientation. The revised variant computes orientations and IK initialization
from the common Pi3 joint chunk, except within the terminal ball where the same
shared servo applies. Primary success remains XYZ within 1 cm; orientation
success is reported separately. Holding current orientation near the goal does
not guarantee the requested terminal orientation.

## Relation to prior methods

[Pi3](https://arxiv.org/abs/2507.13347) supplies the visual-geometry backbone
family. This pilot uses the repository's task-fine-tuned small decoder and
per-frame inference; it does not claim to reproduce published Pi3 reconstruction
quality or to use its multiview inference unchanged.

[Kendall and Gal](https://arxiv.org/abs/1703.04977) distinguish observation and
model uncertainty and study input-dependent uncertainty for vision. The
heteroscedastic regression objective here follows that general modeling idea.
An ensemble or posterior approximation would be needed for a separate
epistemic estimate; a single variance head is insufficient evidence for it.

[Active Neural SLAM](https://arxiv.org/abs/2004.05155) combines learned mapping
with explicit planning and navigation policies for exploration. It motivates
separating scene memory from action selection. Our bounded tokens and adaptive
observation times are a smaller prototype; evaluating route-specific viewpoint
utility remains a distinct, unimplemented research step.
