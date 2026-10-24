# Matched direction propagation control

Base HEAD: 268c5e3. Fixed research line: learned boundary/direction-guided
feature propagation for plane ownership. No new reconstruction backbone or XYZ
modification. Task: docs/codex_tasks/direction_control_20261024.md.

## Protocol

TRAIN scenes 0–3, historical DEV scenes 199/190. Seed42,200updates,batch4,
identical initial weights and patch sequence. Same ownership loss plus .3
direction MSE, same original-support +2 bias, same radius .02 and four neighbors.
No propagation concatenates original features twice; fusion capacity unchanged.
Spatial uses distance only; learned uses shared predicted direction; oracle uses
GT direction broadcast to every candidate during BOTH training and evaluation.
Oracle is diagnostic only, not deployable, not a mathematical upper bound.
All normal DEV predictions saved before GT access. DEV repeatedly used for
development; no independent test, statistical or novelty claim.

## Results (1,361,282 valid points)

| Arm | Total errors | Boundary errors | Newly damaged |
|---|---:|---:|---:|
| Original | 308558 | 63996 | 0 |
| None | 270929 | 63883 | 4293 |
| Spatial | 299439 | 63847 | 2023 |
| Learned | 263904 | 62676 | 5392 |
| Oracle diagnostic | 301963 | 63729 | 1554 |

Learned versus none: 7025 fewer total errors,1207 fewer boundary errors,
1099 more originally correct points damaged. This one-seed pilot supports
further investigation, NOT robust superiority or mainline replacement.
Oracle does not improve over learned; this does not isolate why. Different
learned optimization trajectories and the suitability of nearest-boundary
targets remain confounds. Same seed is not a guarantee of bitwise GPU replay.

Both fixed-zoom figures inspected: scene199 retains ragged junctions; scene190
still contains substantial incorrect fragments. No claim of visually solved
boundaries. Fusion gradients after the zero-initialized score head's first step
are nonzero; this alone does not establish use of directional information.

## Artifacts and verification

Output: local_outputs/direction_control_20261024_v1.
Training and both evaluations completed. Twelve direction-related tests passed.
Independent audit recomputed all eight scene/arm metrics and checked hashes,
pixel/view/source ordering, candidate IDs and unchanged unassigned points.
Global XYZ projection diagnostic generated; no source coordinates changed.

Commands (E:/anaconda/envs/ai/python.exe):

```text
train_direction_control.py --output local_outputs/direction_control_20261024_v1 --steps 200
-m unittest discover -s tests -p '*direction*.py'
audit_direction_mechanism.py --run local_outputs/direction_control_20261024_v1
visualize_spatial_projection_audit.py --run local_outputs/direction_control_20261024_v1 --output local_outputs/direction_control_20261024_v1/projection_audit
```

Files changed: model optional diagnostic controls (old defaults/state keys
preserved), standalone runner, tests, audit/projection receipt-name compatibility,
task and handoff records. Existing outputs and unrelated worktree edits preserved.

## Next, within the same research line

Measure original versus propagated feature differences and ownership-logit
sensitivity, including boundary/interior strata. Then test ONE factor at a time
(neighborhood reach or original-support bias), with fixed split and budget.
These next tests have not run. Do not introduce another method family or scale
training based solely on this single-seed result.
