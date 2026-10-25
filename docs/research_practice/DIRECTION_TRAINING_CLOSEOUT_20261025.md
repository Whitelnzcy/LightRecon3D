# Interrupted training round: completed and independently audited

Implementation HEAD75e1922. The quota interruption stopped follow-up analysis,
not the two training processes: both completion.json files report complete.
This continuation did not rerun completed training or overwrite checkpoints.
Fixed TRAIN4 / repeatedly used historical DEV2, seed42; no independent-test
claim. No XYZ edits or new reconstruction backbone.

## Full-domain results (lower is better)

| Target / steps | Arm | Total errors | Boundary errors <=5px | Newly damaged correct points |
|---|---|---:|---:|---:|
| inward /800 |none|227704|55735|8768|
| inward /800 |learned|227458|56066|8940|
| inward /3200 |none|219621|54080|22799|
| inward /3200 |learned|219772|53476|17864|
| grid_inward /800 |none|239161|58596|4674|
| grid_inward /800 |learned|223889|56768|12494|

All rows have1361282 valid points, original308558 total/63996 boundary errors.
Longer learned training reduces boundary errors by2590 (4.62%) vs learned800,
but increases damage relative to that shorter run by8924. Relative to matched
none3200, learned3200 has604 fewer boundary errors and4935 fewer damaged points,
but151 more total errors. Per scene boundary deltas learned-minus-none are
+522 (scene199) and -1126 (scene190). Benefit is not uniform across scenes.

Grid-target learned800 beats its matched none arm by15272 total and1828 boundary
errors, but adds7820 damaged points. Relative to old inward learned800 it has
3569 fewer total errors,702 MORE boundary errors and3554 more damaged points.
Do not promote grid supervision as a proven boundary solution on these results.

## Direction quality

Same fixed first16 GT-free tiles/view; represented/editable occurrences, with
overlap. Long3200 evaluated against the same kNN inward target field as800:
TRAIN0/DEV199/DEV190 boundary median angles84.90/78.47/80.05 degrees, versus
88.42/80.60/89.25 at800. Still far from precise boundary direction learning.
Grid800 evaluated against an explicitly DIFFERENT grid-defined target field:
82.47/80.05/81.57 degrees. Do not compare these numbers directly with kNN-target
scores. Script records target_boundary and checkpoint hashes.

## Verification and artifacts

- Both runs: full metric recomputation, input hashes, explicit view/pixel/index
  checks, candidate/unassigned invariance and pre-GT prediction equality PASS.
- Both runs: error-change figures and global XYZ projection diagnostics produced.
- Inspected long3200 scene199 view0: corrected394 / damaged406 versus none3200;
  localized changes, not a globally cleaner-looking result. Do not cherry-pick it.
- `local_outputs/direction_inward_long_20261025_s42`
- `local_outputs/direction_grid_inward_20261025_s42`
- `local_outputs/boundary_direction_quality_long_20261025_v1`
- `local_outputs/boundary_direction_quality_grid_20261025_v1`
- `local_outputs/direction_round_summary_20261025_v1/paired_controls.png`
  contains all six completed paired controls in this round, inspected for layout.
- 27 direction tests pass; summarize_direction_round.py syntax check passes.
- Earlier common logged prefix32rows (steps1..776) matched old800 training;
  this is not a full intermediate-weight replay assertion.

Commands: audit_direction_mechanism.py --run RUN;
visualize_direction_error_changes.py --run RUN --baseline none --method learned;
visualize_spatial_projection_audit.py --run RUN --output RUN/projection_audit;
audit_boundary_direction_quality.py --represented-only --inward-checkpoint CKPT
--output OUT [--target-boundary grid]; summarize_direction_round.py --runs RUNS
--output OUT. All paths above preserve the source artifacts.

This interrupted round is closed. Seed43 long-budget replication remains a
planned next experiment, not executed. No primary configuration promotion.
