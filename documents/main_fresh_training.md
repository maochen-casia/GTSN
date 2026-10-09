# Fresh complete-model training and testing — 2026-10-08

The cleaned C1/C2/C3 model reaches **83% validation / 87% test** when the
Pi3 image encoder is tuned. Its frozen-encoder counterpart reaches **73% / 69%**,
missing the 70% test target. Each result covers the complete fixed 100-episode
split: 20 direct, 40 over and 40 side scenes.

| Pi3 encoder | Selected epoch | Validation waypoint RMSE | Validation success | Test success | Validation / test collisions |
|---|---:|---:|---:|---:|---:|
| Frozen | 18 | 16.49 mm | 73% | 69% | 26% / 31% |
| Fully trainable | 30 | 16.56 mm | 83% | 87% | 17% / 13% |

Both runs train for 30 epochs from random navigation weights. Only the image
encoder from official Pi3 weights is loaded; the small geometry decoder, route,
joint proposal, uncertainty and trust heads start randomly. No previous
navigation checkpoint or feature cache initializes either run. The tuned run
has all 352,048,868 model parameters trainable; the frozen run has 47,677,156.
Only `model.perception.freeze_encoder` differs between the archived configurations.

Training uses the existing expert and independent perturbation observations:
8,192 sampled frames per epoch, batch size 8, learning rate 1e-4, a 65/35
expert/perturbation mixture, and 20/40/40 route sampling. The benchmark split is
800/100/100 episodes. Each run selects its complete checkpoint by minimum
validation waypoint RMSE across all 30 epochs, before test evaluation.

Closed-loop success requires reaching within 10 mm of the XYZ goal without
collision. Orientation success is recorded separately. Live wrist RGB and
measured robot state drive 30-step proposals with fixed 15-step execution,
episode-local persistent geometry, and a 400-control-step limit. Depth and
expert futures supervise training only.

| Route | Frozen validation / test | Tuned validation / test |
|---|---:|---:|
| Direct | 100% / 100% | 100% / 100% |
| Over | 85% / 87.5% | 92.5% / 92.5% |
| Side | 47.5% / 35% | 65% / 75% |

The observed tuning gains are **+10 / +18 percentage points**, concentrated in
side scenes. Similar waypoint RMSE does not imply similar closed-loop success.
These compare encoder training regimes in the rewritten complete model; they
do not re-establish the historical C1/C2/C3 ablation gains for this architecture.

All 400 rollouts and both result audits completed inside the existing Docker
image. Audits verify frozen source and vendor hashes, checkpoint selection,
complete disjoint partitions, finite checkpoints and trajectories, empty expert
reference indices, and fixed execution horizons. All 343 encoder tensors match
the official weights in the frozen checkpoint; all 343 differ in the tuned
checkpoint. The 18 main-model regression tests passed before training.

Training completed at 00:41 and 02:17 China time on October 8. The host
supervisors stopped before the automatic evaluation handoff. On October 8,
evaluation and audit supervision were moved into detached Docker containers;
full evaluations and audits then completed. The original checkpoints and
selection criterion were preserved.

- [Frozen results](../runs/main_fresh_20261007/RESULTS.md)
- [Tuned results](../runs/main_full_finetune_20261007/RESULTS.md)
- [Tuned checkpoint](../runs/main_full_finetune_20261007/training/best.pt)
- [Tuned archived configuration](../runs/main_full_finetune_20261007/source/configs/main.json)
- [Tuned training curves](../runs/main_full_finetune_20261007/training_curve.png)

The tuned checkpoint is a standalone state dict with SHA-256
`7adb9cd680cef5d8d4936dce270ef7dad032bb1a8949dde42bccd9cabd207449`.

