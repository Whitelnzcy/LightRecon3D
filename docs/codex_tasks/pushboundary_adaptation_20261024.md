# PushBoundary adaptation: first verify supervision

User approved another trial borrowing existing boundary-direction propagation.
First read official paper3.2--3.4 and preprocessing code. Build evaluation-only
direction target audit on two historical DEV scenes; no new training yet.
Compare global nearest boundary with same-plane nearest boundary, BOTH defined
from GT solely for supervision feasibility. Boundary points retain zero vectors.
Report angular differences and candidate coverage; do not call oracle targets
predictions or claim reconstruction improvements. k4 on full valid predictedXYZ.
No alteration of XYZ/IDs, no geometry contact gate, no threshold tuning.
If worthwhile, next implement learned shared versus candidate-conditioned
direction propagation under same budget; not a full original-paper reproduction.
