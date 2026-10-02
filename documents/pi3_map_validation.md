# Pi3 map prediction validation

Map prediction is a material limit on this model's open-loop accuracy, but it
does not account for all action error. On all 100 validation episodes, replacing
the predicted maps with ground truth lowers the original frozen action head's
RMSE by 12.0%. After matched action-head adaptation on training data, the
ground-truth condition improves RMSE by 25.6% relative to predicted maps.
The adapted state-only control also outperforms predicted maps by 6.3%, so the
current map representation and action head are not extracting a consistent
advantage over robot and goal state alone on expert validation observations.

These results diagnose the epoch-7 checkpoint from
`documents/pi3_small_perturbation10.md`. They measure action prediction on
recorded validation observations. They do not establish the cause of the
original experiment's 6% closed-loop success rate or predict a new success rate.
Artifacts are saved in
`/run/user/1016/experiments/pi3_map_validation_20261002`.

## Validation protocol

The frozen checkpoint is
`/run/user/1016/experiments/pi3_small_perturbation10_20261002/train/best.pt`.
Validation covers 14,420 frames at stride one across all 100 validation episodes:
20 direct, 40 over, and 40 side. The earlier checkpoint-selection metric used
stride two, explaining the small difference between its 0.064843 rad validation
RMSE and this evaluation's 0.064720 rad. Actions retain their 30-step prediction
horizon. The first-15 metric averages horizons 1 through 15, matching the part
of each chunk executed by the controller; it is not the horizon-15 endpoint.

Pi3 and all map heads remain frozen. The model first predicts maps from RGB,
robot and goal state, and measured camera calibration. Recorded depth and
expert future TCP positions then generate teacher maps with the same
`GeometryMaps` settings used during training. Teacher replacements are
deliberately privileged diagnostic inputs. In particular, a ground-truth
future-action map contains expert future information unavailable to a deployed
policy.

The first comparison holds the action head fixed and evaluates all eight
predicted/teacher combinations for the three groups: point XYZ, projected and
visible goal, and future action. Zeroing each group and all groups provides
additional sensitivity controls. Every condition uses the same observations,
state, targets, and valid terminal holds.

Confidence intervals use 5,000 paired bootstrap samples of episodes, stratified
to retain the 20/40/40 route counts. RMSE remains weighted by valid joint target
elements. Resampling episodes preserves correlation among frames within an
episode. Intervals for individual map replacements are descriptive diagnostic
comparisons, without a correction for multiple comparisons.

## Frozen action head results

| Maps replaced with ground truth | Joint RMSE rad | RMSE reduction | First 15 RMSE rad |
| --- | ---: | ---: | ---: |
| None | 0.064720 | 0.0% | 0.037644 |
| Point | 0.063594 | 1.7% | 0.037149 |
| Goal | 0.060239 | 6.9% | 0.034824 |
| Future action | 0.061162 | 5.5% | 0.035388 |
| Point and goal | 0.059541 | 8.0% | 0.034592 |
| Point and future action | 0.060202 | 7.0% | 0.035000 |
| Goal and future action | 0.057278 | 11.5% | 0.033056 |
| All three groups | 0.056944 | 12.0% | 0.033003 |

The full replacement's paired 95% interval for relative RMSE reduction is
8.7–15.1%. The individual intervals are 0.5–3.0% for point, 3.9–10.3% for goal,
and 4.4–6.7% for future action. Goal replacement gives the largest individual
improvement with the original action head. Full replacement improves every
route: direct RMSE falls from 0.068095 to 0.056470 rad, over from 0.059625 to
0.054171, and side from 0.066769 to 0.058520.

Zeroing point, goal, future action, or all maps raises RMSE to 0.110323,
0.091729, 0.076259, or 0.121827 rad, respectively. The trained head therefore
depends on all groups. These abrupt removals change its input distribution;
they do not establish that a policy trained without maps must perform worse.
The adapted state-only control below tests that distinction.

## Map accuracy

Point-channel normalized RMSEs are 0.122014, 0.145110, and 0.096546 for XYZ.
Multiplying channel errors by the configured coordinate scales gives a mean
Euclidean error of 5.77 cm and RMS Euclidean error of 11.49 cm across valid
depth pixels; 28.9% of pixels are within 2 cm. These distances are relative to
the encoded, clipped teacher coordinates. They are not a measurement of
unclipped physical reconstruction error for distant surfaces. All 92,288,000
validation map pixels have valid teacher depth.

Heatmap foreground means teacher score at least 0.1. The following scores
distinguish map structure from errors averaged over background:

| Channel | Pixel RMSE | Foreground RMSE | Zero-map foreground RMSE | Foreground IoU at 0.1 | Pixel average precision | Mean peak error px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Projected goal | 0.0939 | 0.1609 | 0.4701 | 0.757 | 0.925 | 6.00 |
| Visible goal | 0.0959 | 0.1594 | 0.4668 | 0.724 | 0.892 | 6.66 |
| Future action | 0.0680 | 0.1009 | 0.4892 | 0.791 | 0.956 | 1.62 |

Average precision uses 512 score bins, so it is an approximation. Peak error
includes only frames with an active teacher heatmap. The visible-goal map is
active in 12,979 of 14,420 frames; projected goal and future action are active
in every frame. Foreground occupies 26.2%, 23.4%, and 29.2% of pixels,
respectively. The future-action peak is within five pixels in 99.4% of active
frames; the corresponding goal rates are 63.1% and 57.8%.

The heatmaps clearly outperform empty maps and reproduce much of the target
structure. Point maps preserve coarse geometry but smooth object boundaries in
the fixed examples. Good future-action peak accuracy alone does not imply
accurate future joint targets: the map peak and aggregate overlap do not measure
the complete 30-step joint trajectory.

`figures/map_examples.{png,pdf}` shows the first validation episode of each
route at 25% and 75% of its trajectory. This selection is fixed by episode order,
not by prediction quality. Teacher and predicted heatmaps share the [0, 1]
color scale, and point XYZ uses the same coordinate-to-color transformation.

## Matched action head adaptation

A head trained only on predicted maps may handle accurate teacher maps poorly.
To account for that distribution change, copies of the original action head
receive ten epochs of adaptation in four conditions. Pi3, the map predictors,
and their cached outputs remain frozen. All conditions start from identical
action-head weights and use identical sampled training indices within each
seed. Two seeds, 20261002 and 20261003, repeat the comparison.

The cache contains 57,335 expert training frames at stride two and 2,997 safe
perturbation samples. Each epoch samples 57,335 observations with 65% expert
and 35% recovery probabilities, and 20%/40%/40% route probabilities within
each source. AdamW uses learning rate 0.0003, weight decay 0.0001, the original
horizon-weighted imitation loss, gradient clipping at 1.0, batch size 128, and
ten-epoch cosine decay. Only action-policy parameters receive optimizer updates.
No validation or test observation enters an update.

Each condition selects its lowest validation RMSE among epochs zero through
ten. This is an exploratory validation comparison, rather than a new unbiased
test-set estimate. The table pools squared errors across the two adaptation
seeds before computing RMSE. Its intervals resample validation episodes and
do not characterize uncertainty across independently trained Pi3 backbones.

| Adapted head inputs | Joint RMSE rad | Reduction versus adapted predicted | Paired 95% interval | First 15 RMSE rad |
| --- | ---: | ---: | ---: | ---: |
| Predicted maps | 0.064287 | 0.0% | — | 0.037523 |
| Ground-truth point and goal; predicted future action | 0.058217 | 9.4% | 7.1–12.3% | 0.033619 |
| Ground truth for all groups | 0.047800 | 25.6% | 21.8–30.3% | 0.027017 |
| All maps zero; robot and goal state only | 0.060225 | 6.3% | 0.7–12.3% | 0.034920 |

The two predicted-map RMSEs are 0.064236 and 0.064339 rad. The all-teacher
values are 0.047602 and 0.047998 rad, giving improvements of 25.9% and 25.4%.
The state-only values are 0.060181 and 0.060268 rad. Results agree across both
adaptation seeds. Full teacher maps reduce pooled squared error by 44.7%, while
point/goal replacement reduces it by 18.0%.

## Interpretation and next experiment

Better maps improve action predictions under both direct replacement and
matched adaptation. The larger improvement after adaptation shows that the
original action head did not fully exploit teacher-quality inputs. Correcting
point and goal maps while retaining predicted future action gives a smaller
9.4% gain; the additional all-teacher gain depends on privileged expert future
information. It should be treated as a diagnostic information ceiling, not an
expected result from supplying measured geometry alone.

The state-only control is also consequential. The original head depends on
maps, but adaptation with those maps removed achieves better open-loop accuracy
than adaptation retaining their predictions. This is consistent with noisy map
features or action-head fitting interfering with generalization. The experiment
does not distinguish those mechanisms. It does show that reducing pixel map
loss alone is insufficient evidence of better navigation.

The next controlled comparison should train the state-only and predicted-map
policies with matched full training budgets, then compare closed-loop validation
performance. For map improvements, prioritize goal localization and the use of
future-action features, and track foreground/trajectory-sensitive map metrics
alongside joint errors. Point geometry merits finer boundary accuracy, but its
direct replacement gave the smallest open-loop action gain in this checkpoint.
These are proposed next steps; this diagnostic performs no new rollouts and
makes no claim about resulting collision or success rates.

## Reproduction and artifacts

All computations run in `gtsn-pi3:20261002`, with network disabled and source and
benchmark mounted read-only. Outputs remain under `/run/user/1016/experiments`.
The original policy checkpoint is retained. Map caches use float16, state and
target caches float32, inference bfloat16, and metric accumulators float64.

Inside the image, with GPUs 0–3 and those mounts, use a fresh output directory:

```bash
python /workspace/scripts/validate_pi3_maps.py --output-dir "$DIAGNOSTIC_RUN"
python /workspace/scripts/cache_pi3_training_maps.py --output-dir "$DIAGNOSTIC_RUN"
python /workspace/scripts/run_pi3_head_diagnostics.py --root "$DIAGNOSTIC_RUN"
python /workspace/scripts/summarize_pi3_map_validation.py --root "$DIAGNOSTIC_RUN"
python /workspace/scripts/plot_pi3_map_validation.py --root "$DIAGNOSTIC_RUN"
```

`summary.json` consolidates map accuracy, original-head interventions, and pooled
adaptation results. `results.json` retains all original-head metrics;
`adapted_head_results.json` retains each seed and condition. Episode error arrays
support the paired bootstrap. Protocol files record source hashes, split IDs,
sampling checksums, and optimizer settings. The final `audit.json` passed for
validation membership, train/validation separation, identical initial heads and
sample orders, all eight ten-epoch adaptations, and matching source hashes.

The regression suite ran in Docker: twelve tests passed and two existing CUDA
integration tests were skipped in the CPU test container. All three new
diagnostic tests passed. The validation, caching, and adaptation containers
exited with code zero. Figures are exported as PNG and PDF:
`map_interventions`, `map_examples`, and `adapted_heads`.
