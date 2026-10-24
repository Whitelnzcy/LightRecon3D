# Support bias within fixed direction propagation

Fix grid1; vary only output support bias0/1/2, not candidate inputs. TRAIN4,DEV2,
200steps,seed42,batch4,same loss and initial weights. Strict deterministic
algorithms, cuBLAS workspace, TF32 off. Duplicate bias2 model must have bitwise
equal weights every step and equal predictions. This is not an independent seed.
No GT in prediction. Report boundary/total errors and damage. No mainline
promotion without repeated evidence; preserve previous outputs and research line.
