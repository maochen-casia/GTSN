# Fresh full joint attention model — 2026-10-09

This experiment trains the complete C1/C2/C3 replacement policy with the Pi3
image encoder trainable. The official Pi3 encoder weights are the only pretrained
initialization. The geometry decoder, map and joint heads, route head, clearance
heads, and both four-block attention modules initialize afresh. No previous
experiment checkpoint, module warmup weights or cached model predictions enter
training. All **352,639,167 parameters** are trainable, including the
304,371,712-parameter Pi3 image encoder. The distributed smoke audit confirmed
nonzero gradients in every audited group and updates to all 343 encoder tensors.

The official file is `yyfz233/Pi3/model.safetensors`, SHA-256
`33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`.
Only its image encoder is compatible with this project's smaller custom
decoder; the decoder is initialized randomly and trained with the other heads.

The architecture is described in [attention_replacement.md](attention_replacement.md).
C1 selects observations with hard neural ranking. C2 uses neural distance
responses and attention pooling over measured moving robot geometry. Both use
four attention blocks with residual connections and FFNs at width 64. All
contributions remain enabled. Previous fine-tuning results are reference metrics
only and do not supply initialization.

The [fresh configuration](../configs/attention_joint_fresh.json) uses 30 epochs,
8,192 draws per epoch, global batch size 32, AdamW learning rate 1e-4, weight decay
1e-4 and gradient clipping at 1. Eight distributed workers process four examples
each per step. Compared with the earlier full-training batch of eight, the larger
global batch reduces optimizer steps per epoch; this is a new training run,
not a matched isolated backbone-freezing comparison. The split remains
800 training / 100 validation / 100 test episodes. Sampling mixes 65% expert and
35% independent perturbation observations and balances direct/over/side routes
at 20/40/40. The existing recovery observations were generated independently of
any policy and contain training episodes only.

All parameter groups train jointly from live RGB inputs and their supervised
losses. Historical RGB features remain detached for causal memory training;
hard point selection is discrete and learns through the retention auxiliary
loss. Depth, expert future states and physical risk targets are training labels.
Deployment consumes RGB, measured state, goal and calibration.

The trainer's `fresh_joint` guard rejects checkpoint initialization, module
warmup, a frozen encoder and incomplete learned contributions. The Docker
launcher mounts only the frozen source, the benchmark, official Pi3 weights,
independent recovery observations and the new output directory. Earlier
experiment checkpoints are inaccessible to the container. Initial parameter
fingerprints and per-epoch gradient audits verify initialization and training.

A separate two-step distributed smoke run checks real encoder gradients and
parameter updates. Its checkpoint is deleted and is never used to initialize
the full run, which starts again from the official weights and the configured
random seed. Minimum validation waypoint RMSE selects the complete checkpoint
across all epochs before full closed-loop validation and test evaluation.

Run the pipeline inside the existing Docker image:

```bash
python3 scripts/docker_fresh_joint.py \
  --root /home/datasets_v2/chenmao/experiments/attention_joint_fresh_NEW \
  --gpus 0,1,2,3,4,5,6,7
```

The launch returns a detached container ID. The container runs repository tests,
distributed smoke training, full training, both 100-episode evaluations and
audits. Its immutable numerical source is stored under `source/`.

The initial experiment is
`/home/datasets_v2/chenmao/experiments/attention_joint_fresh_20261009`.
`status.json` records the current phase; `training/progress.json` and
`training/epochs.json` record training. The final checkpoint is
`training/best.pt`; `model_audit.json`, `summary.json` and `RESULTS.md` contain
the completed parameter and rollout results.

The initial run finished all 30 epochs and both 100-episode evaluations. Epoch
24 was selected with validation waypoint RMSE **21.08 mm**. Collision-free XYZ
success was **67/100 validation and 65/100 test**, versus 68/100 and 70/100 for
the earlier checkpoint-initialized, frozen-encoder joint model. This fresh run
does not meet the no observed performance drop target. It differs in
initialization, encoder training and batch size, so this comparison does not
isolate the effect of unfreezing the encoder.

All 65 tests passed. Initial encoder fingerprints exactly matched the official
file, all 343 encoder parameter tensors changed, all 1,116 model parameter
tensors changed, and no parameters were frozen. Test success by route was
20/20 direct, 30/40 over and 15/40 side. The test collision rate was 31%; XYZ
plus orientation success was 5%. The checkpoint remains available for comparison
and diagnosis; its SHA-256 is
`9c2fa70faad389d562f53e22ecd9263ac932ea0eb22b6272f4afb3fb6ade86f0`.

## Two additional seeds

Two fresh joint experiments use the same architecture, split and hyperparameters
as the initial run, with four GPUs per experiment. The global batch remains 32,
giving a local batch of eight per distributed worker. The model/data/training
source matches the completed run; only orchestration adds a training-seed
override and explicit seed metadata. The benchmark split seed remains 20261001.

| Training seed | Physical GPUs | Experiment directory |
|---|---|---|
| 20261010 | 0, 1, 2, 3 | `/home/datasets_v2/chenmao/experiments/attention_joint_fresh_seed20261010` |
| 20261011 | 4, 5, 6, 7 | `/home/datasets_v2/chenmao/experiments/attention_joint_fresh_seed20261011` |

Both pipelines run independently in detached, network-isolated Docker
containers. Each verifies official initialization, runs tests and a separate
distributed smoke check, trains all parameters for 30 epochs, selects by
validation waypoint RMSE, evaluates the complete validation/test partitions,
and writes audits plus `RESULTS.md`. No checkpoint from the first run or from
the smoke checks initializes either experiment.

```bash
python3 scripts/docker_fresh_joint.py \
  --root /home/datasets_v2/chenmao/experiments/attention_joint_fresh_seed20261010 \
  --gpus 0,1,2,3 --seed 20261010

python3 scripts/docker_fresh_joint.py \
  --root /home/datasets_v2/chenmao/experiments/attention_joint_fresh_seed20261011 \
  --gpus 4,5,6,7 --seed 20261011
```

Both additional runs completed all 30 epochs and both complete evaluations.

| Seed | Selected epoch | Validation waypoint RMSE | Validation success | Test success |
|---|---:|---:|---:|---:|
| 20261010 | 30 | 20.50 mm | 63/100 | 70/100 |
| 20261011 | 29 | 20.99 mm | 60/100 | 67/100 |

All initialization, gradient, checkpoint and rollout audits passed. The first
additional seed matches the earlier joint model's 70/100 test successes, but
both fall below its 68/100 validation successes. Neither satisfies the no
observed performance drop criterion across both splits. Test results have not
been used to nominate a seed.

The next experiment expands scene/robot node counts and replaces joint-origin
tokens with adaptive surface selection; see
[adaptive_surface_nodes.md](adaptive_surface_nodes.md).
