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

## Fresh cleaned-model experiment — 2026-10-08

Training and full testing of the rewritten C1/C2/C3 model are complete. With
the Pi3 image encoder fully trainable, it reaches **83% validation / 87% test**;
with the encoder frozen, it reaches **73% / 69%**. Each split has 100 episodes
with 20/40/40 route counts. Both runs train all 30 epochs from random navigation
weights, loading only official Pi3 encoder weights. Validation waypoint RMSE
selects epochs 30 and 18 respectively; test results do not guide checkpoint
selection. Tuning gains +10/+18 observed points and clears the 70% target on
both splits. All 400 rollouts and source/checkpoint/trajectory audits completed.
The frozen encoder matches all 343 official tensors; all 343 were updated in
the tuned run. These fresh full-model results are separate from the historical
staged contribution ablations above.

The overnight host supervisor stopped before launching evaluation. The
checkpoints remained intact; detached Docker evaluation and audit containers
completed the handoff on October 8.

[Fresh training report](main_fresh_training.md) ·
[Tuned results and artifacts](../runs/main_full_finetune_20261007/RESULTS.md).

## Learned C1/C2 update — 2026-10-09

The variable-camera main model now supports learned point retention in C1 and
a moving robot-point attention encoder in C2. Independent adapter training
freezes the previously trained Pi3 and every existing navigation tensor. Each
adapter has about 9,800 parameters and trains for four epochs on 2,048 causal
training-only expert/perturbation examples. Full validation selects strengths
before any updated-policy test rollout.

| Model | Validation /100 | Test /100 | Strength |
|---|---:|---:|---|
| Frozen variable-camera parent | 68 | 69 | Original geometry |
| Learned C1 | 69 | 71 | 0.05 |
| Learned C2 | 68 | 71 | 0.05 |
| Learned C1 + C2 | 68 | 71 | 0.025 each |

All three updates meet the requested no observed primary success drop on both
full partitions. The combined model adds 19,618 learned parameters, preserves
10% test orientation success, and improves test collisions from 30 to 29 scenes.
Its two test gains and zero losses give a paired 95% bootstrap interval of
[0,+5] points; a positive population benefit is not established. There is no
matched uniform-risk-rescaling control for the conditioning mechanism.

Completed 1,105 new rollouts: 800 validation candidates, 300 selected-policy
tests and five exact parent replays. A container RAM interruption was recovered
with unchanged weights and frozen numerical source. Parent tensor equality,
complete partitions, finite trajectories and fixed execution passed audit;
53 repository regression tests pass in Docker. Selected standalone models and
compact adapters are stored under
`/home/datasets_v2/chenmao/experiments/learned_geometry_20261009`; redundant trial
checkpoints have been removed.

[Method, results and reproduction](learned_geometry.md).

## Attention replacements — 2026-10-09

The revision replaces C1's fixed priority with neural hard ranking and C2's fixed
risk aggregation with calibrated neural responses and stacked attention. Both
use residual connections and FFNs; the preferred joint model uses four blocks
at width 64. Only Pi3's image encoder is frozen. Every existing
decoder/map/joint/route/clearance head remains trainable and changed during raw
RGB fine tuning.

The first four-trial batch failed the test gate and was rejected. A second
parallel batch preserves fresh geometry with a 256-point learned query and
calibrates local neural responses using training-only physical targets.

| Model | Validation /100 | Test /100 |
|---|---:|---:|
| Original fixed-geometry parent | 68 | 69 |
| Head-fine-tuning control | 67 | 70 |
| Learned C1, two blocks | 68 | 70 |
| Learned C2, two blocks | 67 | Not nominated |
| Learned C1 + C2, four blocks | 68 | 70 |

C1 and the joint model meet the no observed primary-success drop against both
parent and matched control; the joint model is preferred for replacing both
modules. Its 742,371 new parameters bring the trainable total to 48,267,455.
The extra test success also occurs in the head-only control, so a separate
benefit from learned geometry is not established. The earlier soft-adapter test
score was 71/100. These reused development splits and one seed do not establish
population-level noninferiority.

A validation-only 512-point query comparison scored 67/100 for C1 and joint,
so 256 remains selected. Completed 1,600 full-partition rollouts across eight
training trials and two query variants, plus three final-source replay scenes.
All 343 encoder tensors remain bitwise unchanged, every existing head changed,
and replayed complete trajectories match within 1e-6. All 62 tests pass in
Docker. Failed standalone checkpoints were removed; selected models and audit
evidence remain under `attention_replacement_v2_20261009`.

[Method, accepted checkpoint and reproduction](attention_replacement.md).

## Fresh full joint training and seed repeats — 2026-10-09

The full joint attention replacement model trained from random navigation/module
weights and official Pi3 image-encoder weights only. All 352,639,167 parameters,
including the encoder, were trainable. The 30-epoch, 8,192-draw/global-batch-32 run
completed on eight GPUs. Epoch 24 was selected by validation waypoint RMSE
(21.08 mm), before either closed-loop partition was evaluated. Full validation
and test success were 67/100 and 65/100, below the earlier joint model's 68/100
and 70/100; the no observed performance drop target is not met. This comparison
also changes initialization and batch size, so it does not isolate backbone
training. Test successes were 20/20 direct, 30/40 over and 15/40 side, with 31
collisions. All 65 tests and all 200 rollout trace audits passed. All 343 encoder
parameter tensors started exactly at the official values and changed during
training; every model parameter tensor changed.

Two additional training seeds, 20261010 and 20261011, now run in parallel on
GPUs 0–3 and 4–7, respectively. Each receives four GPUs with global batch 32,
local batch eight, 30 epochs and otherwise identical model/data/training
settings. The benchmark seed remains 20261001. Model source matches the initial
fresh run. Each detached Docker pipeline includes full validation/test evaluation
and initialization/gradient/checkpoint/trajectory audits. Prior checkpoint
initialization and module warmup remain excluded.

[Protocol, run directories and reproduction](attention_joint_fresh.md).

## Expanded scene and adaptive robot surface nodes — 2026-10-10

The two previous four-GPU seed repeats completed. Seed 20261010 selected epoch
30 at 20.50 mm waypoint RMSE and scored 63/100 validation, 70/100 test. Seed
20261011 selected epoch 29 at 20.99 mm and scored 60/100 validation, 67/100 test.
Both passed the complete audits but miss the validation no-drop target.

The next model samples 1,024 scene points per RGB observation, queries at most
1,024 points from C1 and increases anchor capacity to 4,096. C2 learns 64 unique
surface-node selections from a 2,048-point URDF collision-mesh pool covering the
moving arm, hand, fingers and calibrated camera. Joint origins are not its node
representation. Candidate IK/FK moves the full arm geometry; full-arm collision
fields now provide training-only risk labels and measured-state self filtering.
The node selector learns relevance and coverage, and the four attention blocks
use the complete queried scene. All 73 tests pass in Docker.

Seeds 20261010 and 20261011 run from official Pi3-only initialization, with all
parameters trainable, on GPUs 0–3 and 4–7. Each uses 30 epochs, global batch 32
and the same benchmark/data/optimizer setup as its previous seed run. Full
validation/test and audits are automatic; results are pending. This changes
physical representation and supervision as well as point counts.

[Architecture, limits and reproduction](adaptive_surface_nodes.md).
