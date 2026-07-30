# E15 Final Prospective B/M Comparison Return

## 1. E14 upstream verification

E14 remains `Ready_Clean_Input_Pool`: 128 global caches and 128 direct
supports exist for 128 scenes. The raw pool contains 3,265 RGB files. The
sealed pool contains 128 `annotation_3d.json` files.

The evaluation-input audit found zero `layout.json` files and zero
`camera_pose.txt` files in the registered raw pool. No final annotation
semantics were parsed.

## 2. Frozen B15 definition

B15 was not implemented or frozen. Execution stopped at the evaluator-input
definition gate before scientific implementation.

## 3. Frozen M15 definition

The required checkpoint exists, its SHA-256 is
`bd8f89bc8c457fd0b54547c775477813b4d5e3a0667f81c08ec28d4faaadfcd1`,
and its 13/64/16 architecture with 27,730 parameters was verified. M15 was not
implemented or run because the experiment had already become non-evaluable.

## 4. Shared identity training and calibration

Not run. Training count and threshold-calibration count are zero.

## 5. Development-only validation

Only a development-scene `annotation_3d.json` structure was inspected to
establish whether that file type carries camera data. Its top-level structure
contains planes, lines, junctions, semantics and incidence data, but no camera
pose. No final-scene annotation semantics were inspected.

The repository's verified point-aligned GT builder requires, for every selected
view, both `layout.json` and `camera_pose.txt`. It raises before label
construction when either is absent.

## 6. Prediction freeze barrier

The barrier was not entered:

```text
B15 predictions = 0
M15 predictions = 0
prediction manifest = frozen empty blocked-state record
prediction_frozen_before_GT_evaluation = false
```

Stopping before 256 expensive predictions avoids producing a permanently
non-evaluable result.

## 7. Non-mutating evaluator audit

E14 global-cache registries contain RGB paths, point-map dimensions, view
indices and global-point offsets. They contain neither camera poses nor GT
pixel labels. `annotation_3d.json` supplies world-coordinate structure but
cannot by itself map a perspective RGB pixel or DUSt3R global point to a GT
plane.

Consequently, no evaluator satisfying the frozen E15 contract can be completed
from the registered inputs. Fetching the missing per-view files would resume
data acquisition; inventing another mapping would change the evaluator/GT
contract. Both actions are prohibited.

## 8. Final 128-scene results

No scientific metrics exist. The 128 final scenes were not consumed by B15,
M15, the GT matcher, or Plane metrics.

## 9. Paired statistical effects

Not computed because there are no valid paired final metrics.

## 10. Mechanical verdict

```text
verdict = Blocked
specific_blocker = Blocked_Missing_PointAligned_GT_Inputs
stop_stage = pre_implementation_pre_prediction
```

This is not a B/M result and cannot be interpreted as evidence for either
method.

## 11. Final project direction

Stop E15. It cannot resume under its frozen input and evaluator contract.
Acquiring `layout.json` and `camera_pose.txt`, or registering an independently
validated equivalent point-aligned GT source, would require a new registration;
E15 itself must remain Blocked.

## 12. Artifacts

```text
results/final_prospective_BM_comparison_e15_20260729_v1/registration.md
results/final_prospective_BM_comparison_e15_20260729_v1/source_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/development_scene_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/final_evaluation_scene_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/shared_identity_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/method_B_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/method_M_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/prediction_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/prediction_manifest.sha256
results/final_prospective_BM_comparison_e15_20260729_v1/evaluator_manifest.json
results/final_prospective_BM_comparison_e15_20260729_v1/evaluator_input_audit.json
results/final_prospective_BM_comparison_e15_20260729_v1/forbidden_call_audit.json
results/final_prospective_BM_comparison_e15_20260729_v1/per_scene_metrics.csv
results/final_prospective_BM_comparison_e15_20260729_v1/per_scene_metrics.json
results/final_prospective_BM_comparison_e15_20260729_v1/per_plane_metrics.csv
results/final_prospective_BM_comparison_e15_20260729_v1/per_plane_metrics.json
results/final_prospective_BM_comparison_e15_20260729_v1/paired_effects.csv
results/final_prospective_BM_comparison_e15_20260729_v1/paired_effects.json
results/final_prospective_BM_comparison_e15_20260729_v1/bootstrap_statistics.json
results/final_prospective_BM_comparison_e15_20260729_v1/summary.json
results/final_prospective_BM_comparison_e15_20260729_v1/run.log
docs/codex_tasks/e15_final_prospective_bm_comparison_20260729.md
docs/codex_returns/CODEX_RETURN_final_prospective_BM_comparison_e15_20260729.md
```
