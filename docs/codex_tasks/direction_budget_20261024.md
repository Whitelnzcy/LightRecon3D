# Paired fixed-route training budget check

Continue direction propagation; do not change model family. Reach and bias pilots
are seed-sensitive. Restore grid1,bias2; compare none vs learned at200/800updates,
same TRAIN4,historicalDEV2,seed42,batch4,loss and strict deterministic settings.
Paired200/800 models must match bitwise throughout first200updates. No parameter
selection using DEV; all checkpoints frozen before evaluation. Report all four
arms, damage, boundary and total errors. This is a bounded budget diagnostic,
not evidence of independent generalization. Do not overwrite previous outputs.
