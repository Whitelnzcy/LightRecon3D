# E16 Targeted GT Sidecar and Final B/M Return

## 1. E14 and E15 status

E14 remains `Ready_Clean_Input_Pool`. Its ordered 128 scenes, 124 five-view
scenes, four four-view scenes, 636 selected views, RGB files, global caches
and direct supports all passed the frozen upstream hash audit.

E15 remains `Blocked_Missing_PointAligned_GT_Inputs`. E16 did not resume,
repair or reinterpret E15.

E16 Registration
`E16_PREREG_targeted_GT_sidecar_and_final_BM_20260730_v1` was executed under
the user authorization with protocol modification prohibited.

## 2. Frozen selected-view sidecar mapping

The E14 view manifest mechanically generated one adjacent `layout.json` and
one adjacent `camera_pose.txt` target for every selected RGB:

```text
layout mappings = 636 / 636
camera mappings = 636 / 636
total members = 1272
missing = 0
duplicates = 0
archives = 14
```

Both independent E9 central-directory indexes agreed on member path,
archive, local-header offset, size, CRC32 and compression method.

## 3. Targeted Range acquisition

All 1,272 members were acquired by exact member-level HTTP Range requests
from the same official `perspective_empty` archives used by E11. No complete
archive was downloaded.

```text
compressed member bytes = 457081
uncompressed member bytes = 1520945
completed members = 1272 / 1272
```

Remote fixed head/tail identity, ETag/Last-Modified, local-header signature,
exact Content-Range, compressed/decompressed size, compression method, CRC32,
compressed SHA-256 and extracted SHA-256 were checked. The initial header
probe and stream retry defects were engineering-only and were corrected
before any sidecar body was accepted; the final acquisition manifest records
the verified results.

## 4. Sidecar integrity and sealed audit

Files are stored under:

```text
F:\Structured3D_clean_e11\sealed_view_gt_e16
```

Before the prediction freeze barrier:

```text
layout JSON parse count = 0
camera-pose numeric parse count = 0
annotation semantic parse count = 0
```

Only path, byte count, CRC32, SHA-256 and scene/view relationships were
inspected. The sidecar-stage verdict was `Ready_Sealed_GT_Sidecars`.

## 5. Frozen B16 definition

B16 is the prospective topology-logistic baseline:

```text
E14 direct support
-> shared initial fragment construction
-> joint_without_bias_with_tangent_support features
-> shared logistic identity prediction
-> shared average-linkage clustering/remerge
-> shared deterministic plane-parameter estimation
```

The frozen scientific source commit is:

```text
c24e543b3098ba280b062d918acdbc9dd8ecdbf5
```

The scientific configuration SHA-256 is:

```text
5e418f3b6dd15c180aa6e8e49c087e14ad303e4dad165494d8967e4b21de27b0
```

## 6. Frozen M16 definition

M16 uses the same input, fragments, identity features/model/threshold,
clustering, support writer, plane-parameter estimator and evaluator. Its only
method switch is the enabled learned intra-fragment split.

```text
checkpoint parameters = 27730
checkpoint SHA-256 =
bd8f89bc8c457fd0b54547c775477813b4d5e3a0667f81c08ec28d4faaadfcd1
split affinity threshold = 0.175
```

The checkpoint was strictly loaded and was not retrained.

## 7. Development-only training and validation

The shared identity model used only the registered 17 development scenes:

```text
development edges = 30773
feature count = 28
C = 0.2
calibration folds = 4
threshold candidates = 128
identity cost threshold = 0.6204897226361007
OOF macro partition pair F1 = 0.8982585715063541
primary tie count = 1
```

Shared identity model SHA-256:

```text
8e207b6f52ce538f0580beca81c61c44a285988726df942fba95486d21ec8ab9
```

Development-side point-aligned GT construction passed. The unified B16/M16
entry passed one four-view and one five-view smoke case. The paired B/M
support hashes matched in both cases. Focused contract tests passed 10/10.
Final-scene training, checkpoint selection and threshold calibration counts
were zero.

## 8. Prediction freeze barrier

The formal order was 128 B16 predictions followed by 128 M16 predictions.
All 256 completed without a quality-driven rerun, scene removal or
replacement.

```text
prediction count = 256 / 256
paired B/M support-hash matches = 128 / 128
prediction_frozen_before_GT_evaluation = true
final GT parse counts at barrier = 0 / 0 / 0
```

Prediction manifest SHA-256:

```text
8f3550779ea395fc634fc14bbba366ea36901910a5f6d6b157208cc0748f6399
```

Every whole-file, support, plane-ID, plane-parameter and provenance hash was
recomputed before GT unsealing.

## 9. Point-aligned GT construction

Only after the barrier, the frozen builder parsed:

```text
layout.json = 636 / 636
camera_pose.txt = 636 / 636
annotation_3d.json = 128 / 128
point-aligned GT scenes = 128 / 128
```

B16 and M16 used the same GT files. All 128 GT whole-file hashes were
verified. The GT manifest is bound to the prediction manifest SHA-256.

Point-aligned GT manifest SHA-256:

```text
934aba076b49c3870301e1836bbb94c84722b1413603b85d847c96c81836afaa
```

## 10. Non-mutating evaluator audit

Prediction hashes were identical before and after evaluation. All registered
forbidden evaluator calls were zero:

```text
expansion = 0
support addition/removal/reassignment = 0
plane fit/refit = 0
weighted fit = 0
RANSAC = 0
plane-ID modification = 0
plane-parameter modification = 0
GT-guided prediction repair = 0
```

## 11. Final 128-scene results

Macro averages:

| Metric | B16 | M16 |
|---|---:|---:|
| Plane F1 | 0.000000 | 0.000000 |
| matched IoU mean | 0.000000 | 0.000000 |
| partition pair precision | 0.237140 | 0.237244 |
| partition pair recall | 0.799792 | 0.799375 |
| partition pair F1 | 0.358892 | 0.358968 |
| fragmentation excess | 0.000000 | 0.000000 |
| overmerge excess | 0.000000 | 0.000000 |
| GT-support assignment | 0.111023 | 0.111023 |
| GT-support matched coverage | 0.000000 | 0.000000 |
| full-cache label rate | 0.111012 | 0.111012 |
| GT-plane label rate | 0.972479 | 0.972479 |
| nonplane false-assignment rate | 0.110622 | 0.110622 |

Aggregate counts:

```text
B16: TP 0, predicted planes 878, GT planes 1067
M16: TP 0, predicted planes 1292, GT planes 1067
```

Both the 124 five-view subset and the four four-view subset have Plane F1
zero under the registered full-support matcher.

The cause is directly visible in the frozen denominator metrics: both methods
label about 11.10% of the full cache, while valid GT planes cover about
97.24%. The shared sparse direct support therefore cannot reach IoU 0.5
against a full GT-plane support. This is a valid final measurement of the
registered immutable methods, not permission to mutate support in the
evaluator.

## 12. Paired effects

All effects are M16 minus B16 with 10,000 paired bootstrap resamples,
seed 20260729:

| Metric | Mean | Median | 95% CI | W/T/L |
|---|---:|---:|---:|---:|
| Plane F1 | 0.000000 | 0.000000 | [0.000000, 0.000000] | 0/128/0 |
| matched IoU | 0.000000 | 0.000000 | [0.000000, 0.000000] | 0/128/0 |
| partition pair F1 | +0.0000758 | +0.0000945 | [-0.0000552, +0.0002065] | 78/1/49 |

M16 does not improve fragmentation or overmerge under the frozen thresholds.

## 13. Mechanical verdict

```text
Mixed_Result
```

`M_Primary_Method` fails because Plane F1 has no positive effect and its
confidence lower bound is not above zero. `M_Useful_Extension` fails because
neither fragmentation nor overmerge improves. The negative-result class also
does not apply because mean partition pair F1 is slightly positive. The
remaining registered class is mechanically `Mixed_Result`.

## 14. Final project direction

B16 is the final default by simplicity. M16 remains a complete mixed-result
extension and ablation. Per protocol:

```text
do not retrain M
do not change thresholds
do not add evaluation scenes
do not restore L/G/J
do not change the evaluator and rerun the final set
```

Method development is closed. The next work is limited to result
visualization, failure analysis and project reporting using these frozen
artifacts.

## 15. Artifacts

Repository result root:

```text
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/
```

External immutable data roots:

```text
F:\Structured3D_clean_e11\sealed_view_gt_e16
F:\Structured3D_clean_e11\predictions_e16
F:\Structured3D_clean_e11\point_aligned_gt_e16
```

Required result files are present, including:

```text
registration.md
source_manifest.json
development_scene_manifest.json
final_evaluation_scene_manifest.json
shared_identity_manifest.json
method_B_manifest.json
method_M_manifest.json
prediction_manifest.json
prediction_manifest.sha256
evaluator_manifest.json
forbidden_call_audit.json
point_aligned_gt_manifest.json
per_scene_metrics.csv/json
per_plane_metrics.csv/json
paired_effects.csv/json
bootstrap_statistics.json
summary.json
run.log
```

Summary SHA-256:

```text
c71a413506c2fe241a333bf57ea763c4a3cb5baefe3a211a584b27d1c24c3cc4
```

Post-verdict reporting artifacts were generated without changing any E16
scientific output:

```text
visualize_e16_final_results.py
visualize_e16_reconstruction_results.py
docs/research_practice/E16_FAILURE_ANALYSIS.md
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/figures/
```

The figure manifest is bound to the frozen summary, per-scene metrics,
prediction manifest and GT manifest hashes.

The actual-output visualization reads frozen RGB, DUSt3R global point clouds
and B16/M16 predictions. Its primary figure is:

```text
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/
figures/actual_reconstruction/b16_final_method_gallery.png
```

It shows input RGB, image-space plane identities and the final globally
aligned 3D plane identities. Three detailed scene figures additionally show
B16 and M16 side by side. Display rendering does not expand support, refit
planes or change prediction IDs.
