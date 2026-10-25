# Unit-direction MSE control

## Completed results and decision

Both seeds completed3200steps, independent full-domain/input/pre-GT audits PASS.
Fixed1361282 valid DEV points; counts total/boundary/damage:

| Loss | Seed | None | Learned |
|---|---:|---:|---:|
|raw|42|220057/53250/14809|211331/52543/19475|
|unit|42|217371/55530/27337|216559/56222/19093|
|raw|43|217795/51729/19703|215710/51343/14298|
|unit|43|215534/51275/24416|224540/54943/18175|

Unit learned vsraw learned adds3679/3600 boundary errors and5228/8830 total
errors. Matched unit propagation adds692/3668 boundary errors versus none.
Do NOT promote unit loss. Default remains raw; all artifacts preserved.

Same-target sampled boundary direction medians raw->unit,TRAIN0/DEV199/DEV190:
seed42 83.46->82.19 /84.58->76.17 /81.92->84.70 degrees;
seed43 82.83->82.49 /73.42->74.85 /81.67->82.10 degrees.
Only one of four DEV scene/seed medians improves. Raw prediction norms increase,
but that does not imply accurate directions. First16 GT-free tiles/view,
represented/editable overlapping occurrences, not independent full-point scores.

Full TRAIN4 fit complete: unit42 none/learned total552335/541381,
boundary132361/128591; unit43 total533291/549048,boundary123947/129838.
Raw42 learned training total540025,boundary124893; normalized42 already worse
on TRAIN, so this is not solely a DEV-generalization failure.

Both runs have error-change/projection diagnostics. Inspected seed43 scene190
view0:1826 corrected,3511 damaged vs matched none, visible errors remain.
Summary local_outputs/direction_unit_comparison_20261025_v2/paired_controls.png
inspected; v1 retained, v2 fixes crowded axis labels. Summary script now records
width and normalization so distinct configurations do not get identical labels.
Direction diagnostics: local_outputs/direction_unit_quality_s42_v1 and_s43_v1,
raw seed43 comparison local_outputs/direction_raw_quality_s43_v1.
All this round's training/evaluations complete; no active job from this round.
Next research question remains how to learn useful boundary directions; neither
capacity nor output normalization alone established a robust boundary solution.
No new loss sweep or research-route switch launched as part of this closeout.

## Original protocol

Base HEADd4abeec. User authorized continuation after two-seed capacity check.
Hypothesis: raw Tanh MSE can reward vector shrinkage when direction is ambiguous;
unit prediction loss removes that mechanism for norms above eps. It does not
guarantee better boundary support or eliminate noisy supervision.

Change only direction_loss(normalize=True). Gather correct TRAIN candidate as
before. Nonzero unit target: normalize prediction with eps1e-6, then coordinate
mean squared error. Undefined zero targets keep original raw-vector penalty.
For two unit vectors, perpoint MSE=2(1-cos(theta))/3. Uniform point averaging;
ownership weightedCE+.5Dice unchanged, direction coefficient .3 unchanged.
Gradient/loss scale inevitably changes; this is not calibrated loss weighting.
No architecture, target generation, optimizer, XYZ or inference modification.
Default raw preserves prior behavior; checkpoint protocol records the setting.

Fixed width2,TRAIN4/historicalDEV2,inward targets,grid1,bias2,agreementOFF,
3200steps,strict deterministic. Paired none/learned for seeds42 and43.
Raw controls already complete direction_capacity_w2_20261025_s42/_s43.
New outputs direction_unit_w2_20261025_s42/_s43 are RUNNING; no efficacy claim.
32 direction tests pass including scale invariance, cosine equivalence,
zero-target compatibility, ignored pixels and finite zero-prediction gradients.
Next full audits, explicit same-target angle comparison,TRAIN fit,error/projection
figures and matched two-seed summary. Preserve checkpoints and all earlier runs.
