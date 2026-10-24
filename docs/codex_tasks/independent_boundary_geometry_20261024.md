# Independent boundary / geometry input ablation

User authorized after questioning the circularity of plane-derived lines.
This takes priority over the proposed keep/reassign head; that head is NOT added.

Freeze before training: TRAIN scenes00000--00003, historical DEV00199/00190;
same seed42, 200 updates, batch4, 64px patches, existing weighted CE+Dice,
same candidate pool, global XYZ, initial weights and observed-pixel applicability.
N: existing RGB/mask/residual only. B: add frozen RGB-only boundary probability.
G: add RAW plane intersections (no RGB-refined lines or matching scores).
BG: add both. Same six-channel extra branch; absent evidence replaced by zero
additions to repeated baseline inputs, keeping all channel weights active.
Additive channel fusion is a limitation, not an optimal architecture claim.

Boundary checkpoint is full3997_e50 best.pt; infer RGB again consistently for all
30 views. Record source/weight/code hashes, explicit halfpixel crop mappings.
Old M3 used RGB-refined intersections, so it was NOT a pure geometry control.
Raw intersections from predicted planes and recovered global-pointmap projection;
10degree near-parallel exclusion only, no GT contact classification.

Freeze all four heads before DEV load; save predictions before DEV GT read.
Evaluate total errors, boundary errors and damaged correct points, per scene.
Historical DEV and upstream-seen TRAIN are not independent test data.
Do not tune loss/threshold or enlarge training based on this single pilot.
