# Capacity control

User requested original vs2x channels, each none/learned propagation. Fixed
TRAIN4/historicalDEV2, inward uniform targets, loss unchanged, seed42,3200steps,
grid1,bias2,agreementOFF,deterministic. Original3200 completed/audited reference:
local_outputs/direction_inward_long_20261025_s42. New width2 output:
local_outputs/direction_capacity_w2_20261025_s42. Different widths cannot share
identical tensors; within each width the two arms share identical initialization
and batches. Do not claim independent-test or stability from a single seed.
Checkpoint schema4 explicitly records width; old checkpoints default width1.
Audit metrics/coordinates, direction quality and training fit before promotion.
