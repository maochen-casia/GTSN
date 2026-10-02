# Corrected mixed recovery experiment

The experiment requested by `instructions/fix.md` addresses problems 1, 3, 4,
and 5 in `documents/recovery_diagnosis.md`. Problem 2's nearest-expert labels,
monotonic progress search, and absence of collision-aware recovery planning are
unchanged.

## Implementation

- Known terminal states remain valid targets for the full 30-step chunk, including
  the 15 executed steps. Endpoint corrections now contribute to both the loss and
  future-action map. Legacy recovery archives are accepted as terminal holds only
  when their masked suffix repeats their last known target.
- Scene reconstruction follows the actual tsn-1k generator, rather than the old
  benchmark generator: full room and furniture, object fittings and revolved vessel
  meshes, Panda v3 wrist-camera robot, SRDF collision exclusions, conservative box
  collision envelopes (22 mm inflation, additional mug-handle allowance), 100 Hz
  physics, 20 Hz control, and ManiSkill's disabled-link-gravity control setup.
  The OSMesa sensor reproduces the generator's millimetre depth quantization.
  RGB textures and lighting are not an acceptance target; the geometry policy uses
  depth. Policy controls remain predicted joint-position targets with zero drive
  velocity, rather than the demonstration generator's expert velocity feedforward.
- Fine-tuning retains the parent as an explicit candidate, rescored under the new
  feature and target settings. Expert RMSE selection also includes the parent.
  The recovery experiment compares only the parent and final epoch, following the
  successful recipe's predefined final-epoch choice. Selection uses fixed validation
  episodes and recovery states; test episodes never enter selection.
- A perturbation episode with no accepted samples is recorded as rejected, rather
  than aborting the complete data generation or weakening its clearance criterion.
  The manifest lists the episodes actually represented by archives, and separately
  records all requested episodes and rejections.

## Predefined protocol

Both the control and mixed run initialize from
`/run/user/1016/experiments/geometry_baseline_gpu_20261001_recovery10/best.pt`.

| Setting | Corrected protocol |
|---|---|
| Fine-tuning | 5 epochs, AdamW, LR 0.00008, cosine decay |
| Chunk / execution horizon | 30 / 15 |
| Action-map sigma / decay | 0.02 m / 20 steps |
| Route weights within each source | Equal thirds |
| Perturbation control | 65% expert, 35% perturbation |
| Mixed run | 42% expert, 35% perturbation, 23% on-policy |
| Split | Original fixed 800/100/100; direct/over/side = 2:4:4 |
| Collection | 80 training episodes; direct/over/side = 16/32/32 |
| Checkpoint validation | Fixed 15 validation episodes (3/6/6), plus their safe perturbed observations |
| Ranking | Success rate, then fewer collisions, then lower recovery RMSE, then lower expert RMSE |
| Exact ties | Keep the parent |
| Goal criterion | XYZ within 1 cm; also report orientation-constrained success within 0.15 rad |
| Final test | All 100 reserved episodes, identical settings for parent/control/mixed |

The source mixture deliberately preserves the existing 35% perturbation weight.
Adding on-policy samples replaces expert samples only. This differs from the old
55/22/23 recipe and removes its simultaneous perturbation-weight change. Architecture,
seed, loss weights, batch size, and frame stride remain as in the current baseline.
The earlier results used different rendering, collision geometry, execution
horizon, maps, and success criteria, so their rates are not a matched comparison.
The orientation column counts XYZ-successful episodes that also satisfy the
orientation tolerance when the XYZ rollout stops; it is not a separate rollout
with an orientation-dependent stopping rule.

The regenerated perturbation pool has 2,997 samples from 797 training episodes
(direct/over/side: 639/1,106/1,252). Episodes 301, 355, and 472 had no safe
perturbations and were excluded from the archive manifest. Minimum accepted
clearance is 2.002 mm. Fresh parent-policy collection on 80 training episodes
produced 815 samples (136/281/398). The independent validation recovery pool has
55 samples from the predefined 15 validation episodes. Archive coverage, finite
targets, all-valid terminal masks, sample counts, and split isolation passed audit.

## Verification and artifacts

Seven Docker regression tests pass for terminal gradients, near-terminal targets,
legacy-mask repair and rejection, exact source/route weights, safety-aware ranking,
validation episode isolation, and recording episodes with no safe perturbations.
The reconstruction acceptance check compares
three recorded states in each of the 80 collection episodes (240 frames): all
views have zero missing depth; every view has at least 99.9948% of pixels within
2 mm, and median error is zero after millimetre quantization.

Run the complete workflow in Docker:

```bash
python3 scripts/docker_run.py --image gtsn-baseline:tsn-1k-gpu fix-experiment \
  --output-dir /run/user/1016/experiments/recovery_fixed_20261001
```

Successful stages are retained on restart. Configurations and source hashes are
recorded in `protocol.json`; stage logs, recovery manifests, validation trajectories,
checkpoints, open-loop metrics, and all test trajectories remain under the run
directory. `status.json` records progress, and `results.json` is written only after
all final evaluations finish. No packages are installed in the host environment.

## Results

All three distinct models were evaluated on the same 100 test episodes under
the corrected protocol. The fine-tuned control and mixed models both complete
five epochs; their results below use epoch 5.

| Model | Successes | Collisions | Timeouts | XYZ successes also meeting orientation tolerance | Open-loop RMSE |
|---|---:|---:|---:|---:|---:|
| Parent with corrected inputs/settings | 38/100 | 45/100 | 17/100 | 20/100 | 0.060777 rad |
| Perturbation-only control, final epoch | 68/100 | 31/100 | 1/100 | 37/100 | 0.050871 rad |
| Mixed recovery, raw final epoch | 66/100 | 30/100 | 4/100 | 39/100 | 0.051803 rad |

| Route | Parent successes / collisions | Control successes / collisions | Mixed final successes / collisions |
|---|---:|---:|---:|
| Direct (20) | 13 / 2 | 20 / 0 | 17 / 3 |
| Over (40) | 15 / 19 | 31 / 9 | 29 / 10 |
| Side (40) | 10 / 24 | 17 / 22 | 20 / 17 |

Both fine-tuned models improve over the parent under these settings. Adding
on-policy data does not improve overall success over the matched control in this
run: it trades three fewer direct successes and two fewer over successes for
three additional side successes. Overall collisions decrease by one episode.
These are descriptive results from one training seed; individual fixes were not
ablated and recovery-label feasibility remains unverified as requested.
The paired comparison has 58 episodes successful for both models, 10 successful
only for the control, 8 successful only for mixed, and 24 unsuccessful for both.
Median final position error is 6.174 cm for the parent, 0.983 cm for the control,
and 0.974 cm for the raw mixed final model.

## Checkpoint selection and interpretation

| Validation candidate (15 episodes) | Successes | Collisions | Recovery-state RMSE | Expert RMSE |
|---|---:|---:|---:|---:|
| Parent | 7/15 | 6/15 | 0.077707 rad | 0.060152 rad |
| Control final | 10/15 | 3/15 | 0.070850 rad | 0.052077 rad |
| Mixed final | 6/15 | 6/15 | 0.073445 rad | 0.052844 rad |

The control selects its final epoch. The mixed run retains the parent in
`mixed/best.pt` (candidate epoch 0), while the actual fine-tuned model remains in
`mixed/latest.pt` (epoch 5). Selection was frozen before test evaluation. The raw
mixed final model was additionally tested for transparent reporting, and its
better test result does not change the selected checkpoint.

The predefined 15-episode validation subset did not rank the parent and mixed
final model in the same order as the full test set. This is a material limitation
of this small validation protocol. The fix prevents automatic replacement of the
parent by a worse validation candidate; it does not guarantee the best test
model. Do not interpret `mixed/best.pt` as the trained mixed final checkpoint.

The main artifacts are under
`/run/user/1016/experiments/recovery_fixed_20261001/`:

- `control/best.pt`: selected perturbation-only final model.
- `mixed/best.pt`: retained parent candidate with corrected input settings.
- `mixed/latest.pt`: actual mixed final model reported above.
- `control/checkpoint_selection.json` and `mixed/checkpoint_selection.json`: frozen validation comparisons.
- `test_parent/`, `test_control/`, `test_mixed/`, `test_mixed_final/`: full test metrics and trajectories.
- `results.json`: consolidated machine-readable results after completion.

The experiment is complete. The selected mixed checkpoint independently reproduces
the parent's 38 successes and 45 collisions on all 100 test episodes. Final audit
passes: seven regression tests, compilation of all 37 Python sources, Docker CPU
argument handling, all four 100-episode test evaluations and 400 trajectories,
frozen validation-only selection, and exact parent-weight retention versus changed
fine-tuned weights. `audit.json` and `regression_tests.log` retain this evidence;
`status.json` reports `complete`. The consolidated `results.json` contains parent,
control, selected mixed, and raw mixed final entries.
