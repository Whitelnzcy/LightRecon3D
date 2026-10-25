# Two-end direction consistency in the same propagation operator

Only multiply existing weights by (1+dot(unit_d_i,unit_d_j))/2; zero directions
neutral. Same grid1,bias2,radius.02,loss,initialization,batches. Oracle and
oracle_agreement,800steps,seed42,strict deterministic,TRAIN4/historicalDEV2.
Both arms GT-assisted diagnostic only. No new backbone, XYZ movement or graph cut.
Do not promote if this oracle diagnostic does not support boundary benefit.
