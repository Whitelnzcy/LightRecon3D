# Capacity replication in progress

Implementation be0b708. Seed42 width2 complete, independent full-domain audit
passed. Fixed3200 updates,TRAIN4/historicalDEV2, inward uniform loss.

| Width | Arm | Total errors | Boundary errors | Newly damaged |
|---|---|---:|---:|---:|
|1|none|219621|54080|22799|
|1|learned|219772|53476|17864|
|2|none|220057|53250|14809|
|2|learned|211331|52543|19475|

Width2 learned vswidth1 learned: total-8441,boundary-933,damage+1611.
Width2 learned vswidth2 none: total-8726,boundary-707,damage+4666.
Per-scene boundary propagation effect width2: scene199+83,scene190-790.
No stable or across-scene benefit established. No mainline promotion.

Additional sampled boundary direction audit completed for width2 seed42:
same represented/editable first16 GT-free tiles/view, overlapping occurrences,
same kNN inward target. Median angle width1->width2:
TRAIN0 84.90->83.46; DEV19978.47->84.58; DEV19080.05->81.92 degrees.
Thus better ownership metrics do NOT demonstrate more accurate boundary
directions; DEV angle worsens in both scenes. Fixed-sample, not full-domain
independent-point angle statistics.

Width2 seed42 error-change and global XYZ projection diagnostics generated.
Output local_outputs/direction_capacity_w2_20261025_s42;
direction audit local_outputs/boundary_direction_quality_w2_20261025_s42.

Second seed43 launched for BOTH widths, each none/learned3200steps, unchanged
loss,targets,optimizer,batches policy and strict determinism. Outputs:
local_outputs/direction_inward_long_20261025_s43
local_outputs/direction_capacity_w2_20261025_s43.
Results pending. Seed42 fullTRAIN4 fit evaluator extended to read checkpoint
width (old default1), syntax check passes; evaluation running for both widths.
Next finish/audit seed43, compare all8 arms, direction and training-fit metrics,
inspect actual figures. Do not change loss during this verification.
