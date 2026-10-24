# Frozen mechanism diagnostics after weak direction pilot

Two bounded checks authorized by user continuing last proposal.
1.TRAIN0/view0 GT-free patches0,1,64,65; copied DS/DC v2 weights,200steps each,
direction MSE only, AdamW.001. Report initial/final angle andMSE. No DEV use of
these overfit weights. This tests memorization, not generalization/ownership.
2.Frozen original v2DS/DC on historicalDEV2: learned, oracleGT,zero,random
directions. Oracle broadcast identically to all candidate slots; no GT candidate
routing. All interventions separately labeled, never promoted as inference.
Random full-view unit vectors fixedseed42+view. Same sourceXYZ/candidates/weights.
Oracle intervention is not a mathematical upper bound: distribution shift and
jointly trained decoder can limit benefit. Validate learned baseline reproduction.
