# Longer fixed-direction training, not a new research route

Base f8ad9ca plus target audit. Same TRAIN4/historicalDEV2, uniform inward
direction MSE coefficient.3, grid1/bias2/agreementOFF, strict deterministic,
seed42, paired none/learned. Increase only updates800->3200. Preserve old runs.
Output `local_outputs/direction_inward_long_20261025_s42`.

Verify first800 logged updates against previous800-step run where available.
Evaluate total/boundary errors and newly damaged correct points, audit pre-GT
predictions and input mappings, render error changes/projection diagnostic,
evaluate boundary direction quality. Repeat with seed43 if feasible before
claiming stability. No XYZ updates, new backbone, graph cuts or method switch.
