# Boundary-balanced supervision and target topology

Base implementation f8ad9ca. Fixed directional-propagation route; no XYZ updates.
TRAIN4 / repeatedly used historical DEV2; not independent test generalization.

## Full-domain results

Same inward targets, 800 steps, deterministic, grid1/bias2/agreementOFF.
Counts below are total errors / boundary errors / newly damaged correct points.

| Direction loss | Seed | No propagation | Learned propagation |
|---|---:|---:|---:|
| Uniform |42|227704 / 55735 / 8768|227458 / 56066 / 8940|
| Uniform |43|224366 / 55922 / 16699|224373 / 56758 / 17965|
| Boundary-balanced |42|239463 / 57971 / 7693|231373 / 56817 / 11010|
| Boundary-balanced |43|232208 / 59417 / 12365|227359 / 58058 / 23919|

Balanced propagation beats its matched no-propagation arm on boundary errors
in both seeds (-1154 / -1359), but loses to uniform learned propagation
(+751 / +1300), and causes more newly damaged points. Do not promote it.
Both full-domain independent audits passed source/index/view/xy checks,
candidate/unassigned invariance, pre-GT saved prediction equality and metrics.
Both runs have fixed-view error maps and global XYZ projection diagnostics.
Seed42 scene190 view0 inspected: localized improvements AND damage; broad
errors remain. Not a visually solved boundary problem.

## Direction quality

Represented/editable points, first16 GT-free tiles/view, overlapping occurrences.
This is a fixed sample, not full-domain independent-point statistics.

| Sample | Uniform median boundary angle | Balanced median boundary angle |
|---|---:|---:|
| TRAIN0 |88.42 deg|82.78 deg|
| DEV199 |80.60 deg|75.53 deg|
| DEV190 |89.25 deg|81.46 deg|

Balanced TRAIN0 direction norm decreases from .305 to .186, and within30deg
fraction from .101 to .089. Median-angle reduction alone is insufficient;
the MSE can decrease by shrinking predictions rather than learning precise
unit directions. This is a diagnostic interpretation, not proven sole cause.

## Target topology: full-domain GT-only diagnostic

`audit_direction_target_topology.py` evaluates registered directed 4-neighbor
edges in fixed global XYZ. No model predictions or training changes.
At eligible boundary points with both same- and cross-GT-plane neighbors,
the cross-plane neighbor is more aligned than the best same-plane neighbor in
only 0.87%-2.34% of TRAIN4 and 1.34%/1.42% of DEV2 cases. Targets usually prefer
an available same-plane neighbor. This does not prove the learned operator works.

Target inconsistency is still measurable: adjacent same-plane boundary targets
oppose each other on 10.55%-19.73% of TRAIN4 directed edges. TRAIN0 only43.16%
of 3D-kNN boundary points fall within5px of the rendered image boundary; other
TRAIN scenes66.53%,95.48%,99.90%. Hence global 3D-kNN boundary supervision is
not identical to the image-grid boundary of the downstream task. Cross-view
nearest-interior anchor fractions range6.19%-52.89%; cross-view anchors alone
are not an error. No target replacement based on DEV statistics is claimed.

## Artifacts and commands

- `local_outputs/direction_balanced_20261025_s42` and `_s43`: checkpoints,
  predictions saved before GT, completion, independent audit and figures.
- `local_outputs/boundary_direction_quality_balanced_20261025_v1`: fixed-sample
  direction diagnostic, checkpoint hashes and pre-GT direction predictions.
- `local_outputs/direction_target_topology_20261025_v1/report.json`: six scenes.
- `python -m unittest discover -s tests -p '*direction*.py'`:24 tests passed.
- `audit_direction_mechanism.py --run RUN`
- `visualize_direction_error_changes.py --run RUN --baseline none --method learned`
- `visualize_spatial_projection_audit.py --run RUN --output RUN/projection_audit`

Next: fixed uniform inward loss, same seed42/data/initialization/batches,
3200 updates, paired none/learned. Separate training insufficiency from operator
limitations. No automatic mainline promotion; compare against800-step controls.
