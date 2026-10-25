# Unit-direction loss control on fixed width2

User authorized following capacity verification. Fixed width2,TRAIN4/historical
DEV2,inward targets,uniform averaging,3200updates,seed42 then43,none/learned pair.
Change only raw Tanh MSE -> unit prediction MSE for nonzero unit targets.
Zero targets retain original raw penalty. eps1e-6 prevents division by zero.
Same .3 coefficient; normalized vector MSE equals2(1-cos)/3 when unit vectors.
The loss scale/gradient changes intrinsically: do not call coefficients calibrated.
Compare against direction_capacity_w2_20261025_s42 and_s43. No XYZ/model/GT
inference changes. Outputs direction_unit_w2_20261025_s42 and_s43. Audit full
metrics,damage,direction accuracy and fit; no promotion from training loss alone.
