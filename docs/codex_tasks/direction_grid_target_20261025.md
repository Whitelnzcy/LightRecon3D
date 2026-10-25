# Align training boundary topology with the registered image operator

Evidence: TRAIN0 view1 target diagnostic shows a broad 3D kNN boundary cloud at
the cabinet; only43.16% of eligible TRAIN0 3D boundary points lie within5px of
image label transitions. This is not claimed to be the sole failure cause.

Single factor: boundary definition k4 global XYZ -> valid differing GT labels
on each registered image's four-neighbor grid. Still learn unit 3D global XYZ
directions: interior away from global nearest boundary; boundary toward global
nearest same-plane interior. Only which points count as boundary changes.
No GT inference, no point movement, no change to architecture/propagation/loss.
TRAIN4/historicalDEV2, seed42,800steps, uniform loss,paired none/learned,
deterministic. Compare with inward800. Keep old target available and default.
Audit input and full metrics; score direction quality against explicitly named
common target fields; do not compare angles defined with different targets.
