# Attention replacements for C1 and C2

This revision implements the requested replacement of fixed priorities. Enable
`model.learned_geometry.mode = "replacement"` with independent `c1` and `c2`
switches, a `width` divisible by four, and a positive `depth`. Replacement
configurations reject residual-strength settings. Earlier adapter checkpoints
remain reproducible with their original configuration and archived source.

## Point memory

`AttentionPointRetention` predicts the entire point priority. Each layer has
latent self-attention, cross-attention to scene points, residual connections and
an FFN. Points then cross-attend to the updated latents and pass through their
own residual FFN. Sixteen latent slots keep attention linear in cloud size.
Inputs include measured state, goal, relative point coordinates, uncertainty,
support, scatter, age and occupancy. The trials use width 64 and depth 2 or 4.

Scores alone choose voxel representatives and evictions. When an observation
matches an existing anchor, its learned score decides whether to replace the
anchor with that actual observation. The revised recipe ranks the four recent
clouds plus old anchors and supplies at most 256 points to C3, with unit point
weights. The persistent bank remains capped at 1,600 points. There is no agreement/proximity priority
added to the scores and no learned risk rescaling. Workspace checks, measured
self filtering and geometric correspondence remain physical preprocessing.

Discrete retention uses supervised learning. Labels combine training-depth
reconstruction reliability within 40 mm with proximity to the expert's future
path. This gives state/goal conditioning a task-dependent training target.
Hard selection is not differentiable; it does not claim end-to-end gradients
through the chosen indices. Episode geometry remains outside model weights and
is cleared by `reset_episode()`.

## Embodiment risk

`AttentionEmbodimentGeometry` builds six body-region tokens and seven measured
arm-posture tokens. Finger opening, camera calibration and forward kinematics
refresh the representation during navigation. Metric signed distances and
uncertainty are input features. Up to 128 nearby scene points per candidate and
sampled pose provide cross-attention context.

Stacked blocks apply pre-normalized self-attention, scene cross-attention,
residual connections and FFNs. In the revised recipe, a small neural network
predicts local responses from uncertainty-padded metric clearance. A learned
attention readout pools those neural responses across time/region tokens.
The original analytic Gaussian/max/mean risk is absent from this inference path;
attention never blends its output with an analytic risk. Known URDF geometry
still supplies distance features and self filtering. Arm tokens provide posture
context; they do not add unmodeled articulated collision volumes.

Training-only calibration targets come from the parent's physical clearance
score and synthetic scalar clearances, alongside expert candidate-ranking
supervision. Synthetic targets read no benchmark episodes. This is distillation,
not an inference blend. The experiment tests whether a direct neural scorer can
retain the parent's navigation performance after replacing the aggregation.

## Training and selection

The study is stored under
`/home/datasets_v2/chenmao/experiments/attention_replacement_v2_20261009`.
Four concurrent GPU trials compare a matched head-fine-tuning control, C1 alone,
C2 alone and a deeper joint replacement. All initialize from the same
variable-camera parent. Only `perception.encoder` is frozen: the 24 decoder
blocks, image projection, conditioning, map and joint heads, route head, and
both clearance heads remain trainable.

New modules first warm up for two epochs on the existing 2,048-example,
training-only causal geometry cache. This stage initializes only new tensors.
Every revised trial then fine tunes on raw RGB for two epochs, with 1,024 weighted
draws per epoch, batch size four, AdamW, existing-head learning rate 1e-6,
new-module learning rate 3e-5, weight decay 1e-4 and gradient clipping at 1.
The unchanged expert/perturbation mixture is 65/35 and route mixture 20/40/40.
Training depth and expert futures are targets only.

Validation waypoint RMSE selects an epoch within each trial. Complete 100-scene
closed-loop validation compares trials. Nomination is frozen before any new
test rollout, and replacements must match both the parent and the matched
control on the primary collision-free XYZ-within-10-mm success metric. The
benchmark was inspected in earlier studies, so these are development results,
not a fresh unseen benchmark or a multi-seed generalization claim.

The runner saves source hashes, immutable protocol and nomination files,
configs, gradient audits, epoch records, complete trajectories and checkpoint
audits. Audits compare every encoder tensor exactly and verify that every
existing head actually changed. Failed standalone checkpoints are removed;
their logs, configs, compact module weights and evaluation evidence remain.

Run a new study entirely inside the existing Docker image:

```bash
python3 scripts/docker_replacement_study.py \
  --root /home/datasets_v2/chenmao/experiments/attention_replacement_new \
  --gpus 0,2,5,7 --recipe calibrated
```

The four-GPU container has capped memory, read-only data/source mounts and no
network. Its numerical source snapshot stays fixed throughout the experiment.
Repository regression tests run inside Docker before training.

## First batch and revision

The first batch in `attention_replacement_20261009` used a bank-only C1 query,
a direct global C2 risk head, existing-head learning rate 1e-5 and three epochs.
It failed the no-drop gate and is retained as a failed development experiment:

| Model | Validation /100 | Test /100 |
|---|---:|---:|
| Parent | 68 | 69 |
| Head-fine-tuning control | 67 | 72 |
| C1, two blocks | 66 | Not nominated |
| C2, two blocks | 68 | 65 |
| Joint, four blocks | 70 | 67 |

No replacement from that batch was promoted. Failed standalone checkpoints
were removed; compact warmup weights, audited hashes, logs and trajectories
remain. On 128 training examples, the bank query averaged about 115 points
versus 193 for the legacy query, and risk calibration RMSE was about 0.06.
These diagnostics motivated preserving fresh coverage and calibrating neural
local responses. They do not prove the cause of the failures.

The revised experiment was designed after inspecting the first batch's tests.
Its nominations are still frozen before its own test rollouts, but the reused
test partition is a development benchmark, not an untouched holdout.

## Completed replacement results

| Model | Validation /100 | Test /100 | Decision |
|---|---:|---:|---|
| Original fixed-geometry parent | 68 | 69 | Reference |
| Existing-head fine-tuning control | 67 | 70 | Matched control |
| C1, two blocks, 256-point hard query | 68 | 70 | Accepted |
| C2, two calibrated blocks | 67 | Not nominated | Rejected |
| Joint C1/C2, four blocks, 256-point hard query | 68 | 70 | Preferred |

The joint replacement meets the requested no observed primary-success drop
against the original fixed-geometry policy and matches the head-fine-tuning
control's test success. It adds 742,371 geometry parameters and trains
48,267,455 parameters in total. All 343 Pi3 encoder tensors stayed bitwise
unchanged; 533 existing parent tensors changed, covering every existing
decoder/map/joint/route/clearance head. The accepted checkpoint selected epoch 2.

Both accepted replacements gain the same single test scene as the head-only
control, with no test losses against the parent. Their paired 95% test bootstrap
interval is [0,+3] percentage points. The joint validation comparison has three
gains and three losses, with interval [-5,+5]. An additional navigation benefit
from the new modules, beyond head fine tuning, is not established. The earlier
soft-adapter study scored 71/100 on test; this pure replacement scores 70/100.

The joint test routes score direct 19/20, over 35/40 and side 16/40. There are
30 collisions and 11 combined XYZ/orientation successes. Primary success uses
collision-free XYZ reaching within 10 mm, with orientation reported separately.

The separate hard-query comparison in `attention_query_budget_20261009` scored
67/100 validation for both C1-only and joint 512-point queries, versus 68/100
for their 256-point counterparts. Budget selection used validation only; neither
512-point candidate was tested or promoted. Across the two training batches and
the query comparison, eight training trials and two additional query candidates
completed 1,600 full-partition rollouts, followed by three release replay scenes.
Failed standalone checkpoints were removed while compact weights and evidence
remain available.

All 62 regression tests pass in Docker. The final source snapshot replayed one
direct, over and side scene; complete saved trajectories match the experiment
within 1e-6. The final clearance-margin handling is covered by a regression test
and preserves the experiment's 0.04-m setting.

Preferred standalone checkpoint:
`/home/datasets_v2/chenmao/experiments/attention_replacement_v2_20261009/joint_d4/training/best.pt`.
Its SHA-256 is
`4bf3e5bf591f5483fbd24a3067921082df0209a3b77d76ecdfb581bec69d14c9`.
The experiment stores `RESULTS.md`, `paired_results.json`, `audit.json`,
`release_audit.json`, source hashes, configs, training records and trajectories.
The accepted training recipe is also available in
[configs/attention_replacement.json](../configs/attention_replacement.json),
including the saved module-warmup initialization path.

Replay the preferred model with the verified source snapshot:

```bash
python3 scripts/docker_run.py --gpu 0 \
  --source-snapshot /home/datasets_v2/chenmao/experiments/attention_replacement_v2_20261009/release_source \
  evaluate \
  --checkpoint /home/datasets_v2/chenmao/experiments/attention_replacement_v2_20261009/joint_d4/training/best.pt \
  --partition test \
  --output-dir /home/datasets_v2/chenmao/experiments/attention_replacement_replay_new
```
