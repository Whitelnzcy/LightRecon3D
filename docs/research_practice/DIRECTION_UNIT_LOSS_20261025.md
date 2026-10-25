# Unit-direction MSE control

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
