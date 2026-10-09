# Published-policy baselines for tsn-1k-var

The current suite contains four **TSN adaptations**, including two 2025 methods.
The earlier `documents/baseline.md` describes a historical privileged geometry
prototype and is not part of this comparison.

| Baseline | Publication | Deployment input | Algorithm retained |
|---|---|---|---|
| CARP | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_CARP_Visuomotor_Policy_Learning_via_Coarse-to-Fine_Autoregressive_Prediction_ICCV_2025_paper.pdf) | Wrist RGB and measured state/calibration | Independent action-dimension tokenizers; residual multiscale VQ; frozen tokenizer; next-scale autoregression with same-scale bidirectional attention |
| FlowPolicy | [AAAI 2025 paper](https://ojs.aaai.org/index.php/AAAI/article/view/33617) | Live depth-derived XYZ and measured state/calibration | PointNet conditioning; conditional U-Net; endpoint and velocity consistency; one-step Euler generation |
| DP3 | [RSS 2024 project/paper](https://3d-diffusion-policy.github.io/) | Live depth-derived XYZ and measured state/calibration | Compact PointNet; conditional action diffusion with clean-action prediction; DDIM sampling |
| Diffusion Policy | [RSS 2023 / IJRR 2024 project/paper](https://diffusion-policy.cs.columbia.edu/) | Wrist RGB and measured state/calibration | Conditional temporal U-Net; cosine noise schedule; noise-prediction training; DDIM sampling and receding-horizon execution |

CARP provides a recent discrete/autoregressive alternative; FlowPolicy provides
a recent continuous one-step alternative. DP and DP3 are established references
that separate image and 3D observation conditioning under diffusion. This is an
algorithm-family comparison on the available benchmark, not a reproduction of
the authors' original benchmarks or reported scores.

The research also considered [DP4 (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Spatial-Temporal_Aware_Visuomotor_Diffusion_Policy_Learning_ICCV_2025_paper.pdf).
Its dynamic Gaussian scene model would require a separate reconstruction and
world-model training pipeline. The selected suite instead covers four deployable
policies with one shared data and simulator interface.

## Pinned references and adaptations

Selected official algorithm files and their MIT licenses are stored in
`vendor/baseline_reference/`. `provenance.json` records immutable commits, source
URLs and SHA-256 hashes. The runtime implements the algorithms in ordinary
PyTorch; it does not import their simulator, Hydra or experiment dependencies.

- All methods receive a requested XYZ/orientation goal, seven arm joints, two
  fingers, measured camera intrinsics and measured camera-to-base transform.
  The goal and robot use the main policy's existing 16-value normalization;
  six normalized intrinsic values and twelve transform entries form a
  34-value per-observation state. Route labels are used only for training
  sampling and reporting.
- There are two causal observations, separated by the dataset's 15-frame
  stride or the live 15-control-step replan period. The initial observation
  is repeated. Each independent perturbation starts fresh. Every live episode
  clears history and resets its seeded sampling generator.
- Training and live frames use the same registered 192×256 resize with
  half-pixel intrinsics; RGB then resizes to 84×112. Depth backprojects at
  48×64 with measured calibration, crops visible XYZ to the workspace, and
  uniformly selects/repeats 512 points in raster order. This substitutes
  deterministic uniform sampling for upstream farthest-point sampling.
- A compact residual CNN and PointNet replace the original visual encoders.
  The temporal FiLM U-Net uses widths 64/128/256 with average downsampling and
  nearest upsampling, rather than the original larger learned resampling
  networks. Diffusion uses 100 cosine noise steps and 20 deterministic DDIM
  steps; RGB DP predicts noise and DP3 predicts clean normalized actions.
- The output is 30 seven-joint residuals relative to the current replan anchor.
  Min/max action normalization is fit once from **training data only** and
  stored with each checkpoint. The shared controller executes the first 15
  targets without GTSN's memory, embodiment filtering, clearance refinement,
  inverse-kinematics goal correction or expert progress lookup.
- FlowPolicy retains the released objective's two segments, epsilon 0.01,
  delta 0.01, boundary 1, velocity weight 1e-5 and gradients through both
  adjacent velocity predictions. Deployment uses its released one-step
  noise-plus-velocity rule. This is not ordinary flow-matching MSE.
- CARP uses grouped temporal convolutional encoders/decoders with separate
  joint codebooks, 8-dimensional latents, vocabulary 256 and scales 1/2/4/8.
  Internally terminal holds pad the 30-step chunk to 32 and decode back to 30.
  Residual quantization retains cosine code matching, commitment/codebook
  losses and scale-specific half-identity/half-convolution refinements.
  A four-layer, width-192 Transformer predicts next-scale codes. Its mask
  prevents a scale from attending to finer scales, while allowing attention
  within the current scale. Seeded categorical sampling generates all codes
  in a scale together. These compact modules replace the released 2D VQ
  architecture and larger adaptive-normalization Transformer.

## Matched experiment protocol

Use exactly the archived main model's `configs/full.json` base configuration,
including split seed 20261001, train seed 20261002, 800/100/100 episodes with
2:4:4 routes, existing perturbation-only recovery archives, and 65/35
expert/perturbation sampling. Recovery provenance and train-only membership are
checked by the existing dataset loader. No test inputs are cached.

Each policy trains fresh for 30 epochs × 8,192 sampled chunks, batch size 8,
AdamW at 1e-4, weight decay 1e-4 and gradient clipping at 1. CARP first fits its
tokenizer for an additional 30 epochs on training chunks, then freezes it.
The EMA follows power 0.75, capped at 0.999. No baseline uses external model
weights. GTSN's published Pi3 image-encoder initialization is an explicit
difference in pretraining; this experiment does not isolate pretraining effects.

Network capacity is also **not matched**. The archived GTSN model has 351,896,796
parameters; these compact adaptations have the following total counts:

| Model | Parameters |
|---|---:|
| GTSN | 351,896,796 |
| Diffusion Policy adaptation | 5,988,615 |
| DP3 adaptation | 5,281,607 |
| FlowPolicy adaptation | 5,281,607 |
| CARP adaptation, including tokenizer | 2,743,519 |

Success differences in this experiment cannot isolate the algorithm family,
geometric contributions, model capacity or pretraining. They describe the stated
TSN adaptations under a matched policy-training sample budget. They do not
establish superiority to the full published implementations or their best tuned
training recipes.

Minimum Cartesian waypoint RMSE over all validation samples selects each
EMA checkpoint. Six waypoints are obtained by Panda forward kinematics at
future steps 5/10/15/20/25/30. Sampling uses the same fixed validation seed
every epoch. Joint RMSE is also saved. Checkpoint SHA-256 receipts are written
before closed-loop test evaluation.

Every selected model runs all 100 validation and 100 test episodes, including
20 direct / 40 over / 40 side episodes per partition. The numerical simulator
is byte-identical to the archived main experiment. Evaluation uses 20 Hz
control, collision checks at 100 Hz, fixed 15-step execution, at most 400
control steps, and collision-free XYZ reaching within 10 mm. Orientation
success at 0.15 rad is reported separately.

The main result is taken from the completed audited archive
`/home/datasets_v2/chenmao/experiments/gtsn_cam_var_20261008` (69/100 test
successes). The suite pins the archive's result and checkpoint hashes; it does
not retrain the main model. DP3 and FlowPolicy receive depth at deployment,
so their results must be labeled as a different observation setting from RGB
GTSN, DP and CARP. One training seed and paired scene-bootstrap intervals do
not estimate training-seed variability.

## Completed results

All four policies finished 30 training epochs; CARP also finished its 30-epoch
tokenizer stage. All 800 baseline rollouts passed the final audit. Success below
means collision-free XYZ reaching within 10 mm; orientation is reported
separately in the experiment report.

| Model | Selected epoch | Val. waypoint RMSE (mm) | Val. success | Test success | Test collisions |
|---|---:|---:|---:|---:|---:|
| Archived GTSN | 16 | 19.51 | 68/100 | 69/100 | 30 |
| Diffusion Policy adaptation | 30 | 40.55 | 5/100 | 0/100 | 83 |
| DP3 adaptation, depth | 27 | 22.50 | 8/100 | 1/100 | 40 |
| FlowPolicy adaptation, depth | 30 | 27.01 | 7/100 | 9/100 | 64 |
| CARP adaptation | 29 | 39.77 | 2/100 | 1/100 | 82 |

Post hoc test success at additional XYZ tolerances is shown below. A success
requires a recorded end-effector position within the radius and no collision
anywhere in the original episode; these are the original 1 cm-stopping runs.

| Model | 3 cm success | 5 cm success | 10 cm success | Collision rate |
|---|---:|---:|---:|---:|
| Archived GTSN | 70% | 70% | 70% | 30% |
| Diffusion Policy adaptation | 7% | 13% | 16% | 83% |
| DP3 adaptation, depth | 25% | 44% | 57% | 40% |
| FlowPolicy adaptation, depth | 15% | 26% | 34% | 64% |
| CARP adaptation | 6% | 11% | 18% | 82% |

`scripts/report_success_thresholds.py` derives these metrics and verifies the
original 1 cm scores from all 1,000 main/baseline trajectories. The experiment's
`success_thresholds.json` includes validation/test route breakdowns. The report
also distinguishes estimates from stopping at the first looser-tolerance reach,
which can exclude later collisions. Neither definition requires retraining or
new simulator runs. `threshold_reporting/` saves the analysis source, three
passing metric-semantics tests, input hashes and per-episode first-hit steps.
The current Docker pipeline generates this additional report automatically.

These results describe compact adaptations with different capacity, pretraining
and observation inputs, as quantified above. They do not establish superiority
to the full published implementations. One training seed was used.

The [completed report](/home/datasets_v2/chenmao/experiments/baselines_20261009/RESULTS.md)
includes route breakdowns, paired scene-bootstrap intervals, validation-only
diagnostics and execution provenance. The
[comparison figure](/home/datasets_v2/chenmao/experiments/baselines_20261009/comparison.png)
also labels the comparison limits. All 41 regression tests and all four real GPU
update/live-camera smoke checks passed.

The first audit reader expected a different field name from the archived GTSN
summary. Its corrected schema reader passed the full audit; `audit_recovery/`
preserves both the failed audit and the corrected reader. No weights, numerical
source or episode metrics were changed by that reporting correction.

## Docker and artifacts

No packages are installed on the host. The suite reuses the existing
`gtsn-experiment:20261009-clean` image without changing any images. The new
container runs offline, with read-only source/data/reference archives, a
writable new experiment root, four distinct GPUs, 16 CPUs and 32 GiB host
memory. Each training worker caps its CUDA allocator at 10% of one A800's
memory. Other users' containers/processes are untouched.

Held-out evaluation uses fresh processes for batches of at most five episodes,
which bounds native simulator/renderer memory. The initial experiment's
long-lived evaluators hit the container's 32 GiB host-memory limit after all
training had finished. Its remaining episodes were recovered with the same
frozen policies, simulator, seeds and thresholds. Complete finite episode
records were retained; incomplete episodes were archived and rerun.
`evaluation_recovery/` records the failure and recovery protocol, and
`evaluation_batches/` keeps the isolated batch logs. Checkpoint hashes are
verified before and after recovery. The current launcher uses isolated batches
for every new experiment.

```bash
python3 scripts/docker_baselines.py --detach --gpus 0,2,4,6 \
  --root /home/datasets_v2/chenmao/experiments/baselines_NEW
```

The launched experiment root is
`/home/datasets_v2/chenmao/experiments/baselines_20261009`.
`status.json`, `pipeline.log`, `cache.log`, and per-method logs track progress.
The runner freezes `source/` and hashes before training. Each method saves
standalone `training/best.pt` and `latest.pt`, epoch records, initialization,
checkpoint receipt, and complete `validation/` and `test/` trajectories and
metrics. CARP additionally saves its tokenizer checkpoint and epoch records.
All four methods also passed real GPU forward/backward/AdamW updates and short
live-camera simulator rollouts; `smoke_complete.json` records those checks.

After all 800 rollouts, `report_baselines.py` verifies complete partitions,
validation-only selection, source/benchmark/reference hashes, simulator
identity and finite weights/trajectories. It writes `summary.json`, `RESULTS.md`,
paired main-minus-baseline confidence intervals, `comparison.png` and
`comparison.pdf`. The command exits unsuccessfully if any worker or audit fails;
an incomplete experiment must not be presented as a completed benchmark.
