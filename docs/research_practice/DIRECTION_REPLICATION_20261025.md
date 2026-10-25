# Inward target replication and boundary-quality diagnosis

Base HEAD113a94a. Fixed direction propagation route, no new backbone or XYZ edits.
Seed43 repeated zero/inward targets x none/learned propagation at800updates,
grid1,bias2,agreementOFF,TRAIN4,historicalDEV2,strict deterministic.

| Seed | Target | Propagation | Total errors | Boundary errors | Damage |
|---|---|---|---:|---:|---:|
| 42 | zero | none | 229797 | 55926 | 12620 |
| 42 | zero | learned | 237959 | 58456 | 19762 |
| 42 | inward | none | 227704 | 55735 | 8768 |
| 42 | inward | learned | 227458 | 56066 | 8940 |
| 43 | zero | none | 227819 | 56999 | 18583 |
| 43 | zero | learned | 226239 | 59132 | 14156 |
| 43 | inward | none | 224366 | 55922 | 16699 |
| 43 | inward | learned | 224373 | 56758 | 17965 |

Target completion reduces learned-arm boundary errors by2390/2374 across seeds.
But propagation adds331/836 boundary errors over matched inward-none. Seed43
damage INCREASES3809 versus zero-learned, so reduced damage does NOT replicate.
Do not promote propagation or claim independent generalization/statistical proof.
Both seed43 runs independently audited, including saved-before-GT predictions.
Four-way comparison and global XYZ projection diagnostic generated.

## Boundary-specific direction quality

Frozen seed42 zero/inward learned800 models, first16 GT-free tiles/view,
overlapping occurrences. Both scored against the completed inward field.
Prediction saved before GT scoring. Separate represented/editable-only check
excludes missing-candidate and noneditable points, matching supervision scope.

| Scene | Boundary median angle zero / inward | Within30deg inward |
|---|---|---:|
| TRAIN0 | 88.74 / 88.42 deg | 10.11% |
| DEV199 | 87.29 / 80.60 deg | 10.69% |
| DEV190 | 85.58 / 89.25 deg | 11.74% |

Direction remains weak at trainable boundary points; not merely missing-candidate
scoring. TRAIN0 sample has12051boundary vs308335interior occurrences (~3.76%
boundary), so uniform mean loss is dominated by interior pixels. This is a
hypothesis motivating loss balance, NOT proof that balancing solves it.

Outputs under local_outputs: direction_zero_learned_20261025_s43,
direction_inward_learned_20261025_s43, boundary_direction_quality_20261025_v1,
boundary_direction_quality_20261025_represented. Original all-valid diagnostic
retained; represented-only result preferred for learning-capacity interpretation.

## Active single-factor test

direction_balanced_20261025_s42: equal-mean boundary/interior direction MSE,
same total auxiliary coefficient.3, ownership loss and architecture unchanged.
Paired none/learned, inward targets,800steps,seed42,strict deterministic. Boundary
mask used ONLY in TRAIN loss, never model inference. 22direction tests passed,
including missing-stratum/ignored-target behavior. Run active at checkpoint.

Commands: train_direction_control.py with --seed43 --steps800 --deterministic
--control-arms none learned and --boundary-target zero/inward, separate outputs;
audit_direction_mechanism.py --run DIR;
compare_direction_target_runs.py --run INWARD --baseline ZERO;
audit_boundary_direction_quality.py --output DIR --represented-only;
balanced run adds --direction-balance boundary --boundary-target inward.
