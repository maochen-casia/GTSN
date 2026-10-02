The regression is real, and I found several concrete differences that can explain it. **The strongest issue is terminal recovery handling: the new implementation discards supervision exactly where the old implementation teaches the policy to finish and hold the goal.** Recovery label quality and checkpoint selection also deserve attention.

I compared both codebases and inspected existing logs, metrics, and recovery archives in Docker. I did not run training or simulation, install packages, or change files.

Both current evaluations used identical test episodes, model configurations, and evaluation settings:

| Route | Perturbation-only successes | Mixed successes | Perturbation-only collisions | Mixed collisions |
|---|---:|---:|---:|---:|
| Direct | 5/20 | 6/20 | 3/20 | 6/20 |
| Over | 13/40 | 4/40 | 7/40 | 15/40 |
| Side | 3/40 | 5/40 | 14/40 | 13/40 |
| **Overall** | **21/100** | **15/100** | **24/100** | **34/100** |

The regression is concentrated in `over`. Overall, reaching within 2 cm actually improves slightly, from 57 to 58 episodes, while median final position error worsens from 2.34 cm to 5.32 cm. That suggests problems with finishing safely and remaining near the goal. Sources: perturbation-only results (/run/user/1016/experiments/geometry\_baseline\_gpu\_20261001\_recovery10/evaluation/closed\_loop.json), mixed results (/run/user/1016/experiments/geometry\_baseline\_gpu\_20261001\_recovery10\_mixed\_ft10\_retry1\_20261001T111248826869Z/evaluation\_20261001T113857918869Z/closed\_loop.json).

1. **Terminal recovery samples lose all supervision — confirmed implementation defect.**

   In padded\_future() (src/tsn/data/hdf5\_dataset.py:51), when progress reaches the final expert frame, the function repeats the final state but returns an entirely false mask. Consequently:
   - The loss (src/tsn/training/losses.py:13) ignores every target in that recovery sample.
   - The future-action map becomes zero.
   - The rollout (src/tsn/evaluation/closed\_loop.py:128) still executes five predictions.

   This affects **792/5,049 collected samples**: 169 direct, 499 over, and 124 side. Another 209 samples have only one to four supervised horizons, although execution still uses five.

   The targets being discarded are meaningful corrections. For terminal `over` samples, the median largest joint correction is approximately **0\.426 rad**.

   The old implementation (tsn\_old/experiments/evaluate\_closed\_loop.py:79) retains repeated final targets and trains on them. It therefore supplies explicit supervision for returning to and holding the terminal configuration.

   Archived trajectories support this concern: the mixed model has **six `over` collisions after reaching the final reference index**, versus zero for perturbation-only. This does not prove that masking caused every regression; nine mixed `over` collisions happen earlier.

   **Proposed solution:** distinguish terminal holding targets from unavailable future observations. Give terminal corrections explicit valid supervision and retain the terminal waypoint in the action map. Make near-terminal execution consistent with the supervised horizon. This issue already existed in the new baseline; adding recovery data exposes it because many collected states are terminal states.
2. **The collected states have expert-reference labels, but no verified recovery plan — substantial risk.**

   Collection (src/tsn/evaluation/closed\_loop.py:93) searches the entire remaining expert trajectory for the nearest joint configuration, then labels the observation with future expert joint positions minus current joints.

   There is no check that the motion from the deviated state to those targets is collision-free or dynamically feasible. Monotonic progress can also jump forward and cannot retreat afterward.

   The current pool is dominated by prolonged failures:
   - 48/80 collection episodes time out.
   - **3,840/5,049 samples—76%—come from timeout episodes.**
   - 138 samples with valid first targets request a joint correction greater than **0\.75 rad**, beyond the model’s output bound.

   Timeout states are useful when their corrective labels are valid. Here, repeated stalled states can receive unsafe or unattainable corrections and gain substantial training weight.

   This weakness is shared with `tsn_old`. Its earlier experiment log (tsn\_old/documents/experiment\_log.md:129) explicitly reports increased collisions after on-policy recovery and identifies nearest-expert pseudo-labels as a possible cause.

   **Proposed solution:** validate the path from each collected state to its recovery target, preferably using a local collision-aware planner. Limit progress jumps, bound executable corrections, and cap repeated samples at the same progress point. Balance samples across episodes and progress regions so long failures cannot dominate a route.
3. **Expert and recovery observations differ substantially at identical robot states — confirmed data mismatch.**

   I compared the first saved recovery observation against the original demonstration’s first frame across all 80 collection episodes. Joint positions match exactly; camera transforms differ by less than approximately 2×10⁻⁶, and intrinsics match.

   Nevertheless:

   | Route | Recorded depth valid pixels | Recovery renderer zero-depth pixels |
   |---|---:|---:|
   | Direct | 100% | 49\.4% |
   | Over | 100% | 18\.8% |
   | Side | 100% | 27\.1% |

   Where both images contain valid depth, their median difference is only about 0.5–0.6 mm. The large discrepancy is primarily missing rendered surfaces or background.

   The current simulator reconstruction (src/tsn/simulation/episode.py:145) uses configured table dimensions and primitive objects. The old pipeline reconstructs scenes through the same generator functions and full episode configuration used to create its demonstrations.

   This mismatch affects point maps and occlusion-dependent action maps. It existed during perturbation-only training too, so it cannot independently explain the additional regression, but it makes the old and new recovery pipelines materially different.

   **Proposed solution:** recover and reuse the benchmark generator’s complete scene, rendering, and controller configuration. The identical-state comparison provides a useful acceptance criterion for reconstruction fidelity.
4. **Checkpoint selection can replace the parent with a worse model — confirmed selection problem.**

   Training (src/tsn/training/runner.py:139) initializes `best_rmse` to infinity and does not include the initialization checkpoint as a candidate.

   The parent checkpoint’s validation RMSE is **0\.054708 rad**. The best mixed checkpoint is **0\.054813 rad**; every mixed epoch is worse by this selection metric. Nevertheless, epoch 6 becomes `best.pt`.

   Also, expert-frame RMSE measures demonstration prediction, which is poorly aligned with recovery safety. In the old latest-recovery experiment (tsn\_old/documents/experiment\_log.md:575), RMSE worsened while closed-loop success improved from 68% to 87%.

   **Proposed solution:** retain the parent as an explicit checkpoint candidate. Select recovery checkpoints using a predefined validation protocol that includes recovery states and closed-loop safety/success. Keep test episodes reserved for final reporting.
5. **The old successful recipe was not reproduced exactly.**

   | Setting | Old successful geometry run | Current mixed run |
   |---|---|---|
   | Terminal padding | Trained repeated final targets | Masked out |
   | Execution horizon | 15 | 5 |
   | Action-map Gaussian width | 0\.02 m | 0\.05 m |
   | Action-map decay | 20 steps | 10 steps |
   | Route sampling within each source | Equal thirds | 20% / 40% / 40% |
   | Mixed training | 5 epochs, LR 0.00008 | 10 epochs, LR 0.0001 |
   | Evaluated checkpoint | Final epoch | Minimum expert validation RMSE |
   | Goal success criterion | XYZ only | XYZ plus orientation |

   These differences change route information, execution behavior, and optimization. They are plausible contributors, rather than established bugs.

   Furthermore, the current transition reduces safe perturbation sampling from **35% to 22%**, while adding 23% on-policy samples and reducing expert sampling from 65% to 55%. It changes three source weights simultaneously, so the result does not isolate the effect of adding on-policy data.

   One comparison detail matters: the old reported **9%→68%** gain covers both recovery stages. Its intermediate **60%** result was measured on collection episodes from training, not the final held-out set. The later **68%→87%** improvement is a matched held-out comparison. Old experiment record (tsn\_old/documents/experiment\_log.md:480).

I would prioritize **terminal supervision and execution consistency**, then **recovery-label feasibility and duplicate control**, followed by **renderer fidelity and checkpoint selection**. I found no target subtraction/sign error: saved labels exactly match the implemented expert-reference formula, and both versions execute residuals relative to the same chunk anchor. The main concern is whether those labels provide valid, useful recovery supervision.