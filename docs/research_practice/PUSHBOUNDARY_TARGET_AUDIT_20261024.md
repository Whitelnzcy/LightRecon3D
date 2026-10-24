# PushBoundary adaptation: supervision feasibility, not a trained result

Base HEAD5e2cd7fb7cf106512e361b04517701815206d0ef.
User approved trying an existing-method adaptation. No new training this turn.

Sources inspected2026-10-24:
- https://arxiv.org/html/2212.12402 Sections3.2--3.4 and4.
- https://github.com/shenglandu/PushBoundary (PointNet++/KPConv implementations).
- https://raw.githubusercontent.com/shenglandu/PushBoundary/main/PointNet2_Backbone/preprocessing_boundaries.py

The paper uses predicted boundaries/directions to weight decoder interpolation.
Official preprocessing marks mixed-label k4 neighborhoods as boundaries and
sets direction to zero on boundary points. Others point away from the closest
boundary. We use mathematically unit-normalized vectors, not verbatim source;
the source shown updates dx,dy,dz sequentially while recomputing denominators.
This audit is not a verified reproduction of released preprocessed datasets.

Implementation: boundary_direction_propagation.py follows the paper's spatial
and directional weight combination; explicit spatial fallback added for all-zero
weights. No backbone integration yet. Shared or candidate-conditioned direction
axes are supported; no claim of original benchmark reproduction or novelty.

Supervision audit on full valid predicted XYZ with attached GT instance IDs:
global nearest boundary versus nearest boundary within the same GT plane.
Both are oracle SUPERVISION targets, not predictions. Original semantic labels
are not available/used; this compares two INSTANCE-target adaptations. It cannot
prove the original semantic method fails on same-category walls.

|scene|valid points|k4 boundary points|wrong-near-boundary eligible|direction differs>30deg|
|---|---:|---:|---:|---:|
|00199|648848|6010|15546|585(3.76%)|
|00190|712434|18521|37279|4455(11.95%)|

Combined5040/52825=9.54%. Eligible excludes boundary-zero vectors and missing
same-plane anchors. Near-boundary uses existing image-domain5px distance, while
targets use3D kNN; do not conflate definitions. Full valid domain1361282 points.
Angular difference demonstrates target sensitivity, NOT which target is better.
Most eligible points do not differ substantially; candidate-conditioned benefit
must be evaluated by actual learned ownership, not assumed from this audit.
Duplicate multiview XYZ samples can affect kNN boundaries; sensitivity unresolved.

Candidate coverage:00199 all65351 errors represented/editable.00190 has243207
errors,176225 represented/editable,65592 missing matched candidates,1390 other
noneditable represented errors. No direction-only head can assign a missing ID
under a fixed candidate pool. Report full-domain scores and editable subset.

Outputs: local_outputs/pushboundary_target_audit_20261024_v1/{protocol,report}.json,
two schema1 diagnostic NPZs and direction_target_diagnostic.png. Original source
hashes and pixel/view arrays checked; GT only in audit; original arrays untouched.
Audit exited0. Six tests pass (targets3, propagation3); diagnostic figure viewed.

Commands using E:/anaconda/envs/ai/python.exe:
```
-m unittest discover -s tests -p test_pushboundary_targets.py
-m unittest discover -s tests -p test_boundary_direction_propagation.py
audit_pushboundary_targets.py --output local_outputs/pushboundary_target_audit_20261024_v1
visualize_pushboundary_audit.py --run local_outputs/pushboundary_target_audit_20261024_v1
```

Next: integrate the operator into the same ownership decoder; compare shared
direction vs candidate-conditioned direction with identical active capacity,
budget and labels. First use SAME direction target to isolate conditioning;
same-plane-target change is a separate ablation. Frozen RGB boundary is an input,
GT direction is training-only. Preserve zero-boundary behavior baseline before
testing alternatives. Add no-direction baseline. No inference routing by GT.
Current audit does not establish new reconstruction gains or proven innovation.
