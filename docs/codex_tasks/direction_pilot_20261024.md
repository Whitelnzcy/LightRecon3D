# Frozen direction pilot protocol (before training)

TRAIN4 scenes0--3, historicalDEV199/190;200steps,seed42,batch4,64px patches.
D0 distance interpolation, DS shared predicted3D direction, DC candidate-conditioned
predicted3D direction. Same initialization/parameters/optimizer/ownership loss.
All arms add0.3MSE for same global-nearest-boundary unit-vector target, zero on
boundary. TRAIN only: GT-correct candidate selection for loss, never inference.
Wrong candidate directions receive ownership supervision, not direct direction
targets. Frozen RGB boundary + normalized globalXYZ offsets. Grid4neighbors and
center, same resolution, unlike paper's hierarchical3D kNN: mechanism adaptation,
NOT original reproduction. Radius0.02scene diagonal fixed, no sweep. XYZ/IDs and
candidate pool unchanged. Freeze all heads before DEV; save predictions beforeGT.
Audit full-domain errors/damage and mappings. No independent test/parity claim.

Implementation correction after v1: direction head lacked explicit XYZ despite
supervising global directions. v2 adds normalizedXYZ3 channels to all arms,
same budget/loss. v1 preserved, not used for model selection. Old checkpoints
load with include_xyz=False. This does not establish rotation equivariance.
