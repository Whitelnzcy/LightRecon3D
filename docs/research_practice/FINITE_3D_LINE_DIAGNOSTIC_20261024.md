# Finite 3D lines: support rejection diagnostic

Base HEAD d4bc48b46fda5752998305537ff710de21a26bff. No training or assignment edits.
Task: docs/codex_tasks/finite_line_diagnostic_20261024.md.
Outputs: local_outputs/finite_3d_line_diagnostic_20261024_v1 and v2.
v2 only adds continuous distance and nearest-point records, identical thresholds.

The earlier visual displayed every raw infinite intersection across the image.
This diagnostic explicitly keeps p+t*v in global predicted XYZ and restricts t
to overlap of the two support point sets'1%-99% along-line quantiles. Then tests
proximity to BOTH predicted plane supports, and independent RGB evidence in
registered views. No camera/layout GT read; original points/IDs unchanged.
Support bandwidth fixed at2% robust scene diagonal; RGB0.5, pixel proximity1.5px.
Observation proximity is NOT a physical visibility/occlusion check. Near-parallel
pairs<10degrees excluded. These are diagnostic heuristics, not proposed final rules.

Results: scene00199 has8 nonparallel overlap candidates; scene00190 has41
nonparallel pairs,37 overlap candidates. Across45 sampled lines, NONE has a
sample within the fixed bandwidth of both assigned supports. Therefore RGB
confirmation has no surviving candidates: this does NOT test RGB model quality.
No successful finite-line extraction or reconstruction improvement is claimed.

For00199, bandwidth0.01032316(model coordinates, NOT meters). Pair0/1 closest
joint-support sample is[0.09645355,-0.06522913,0.31567526]. Distances to its two
assigned support clouds:0.00056316 and0.02912616. Plane equation residuals at
the sample are~1e-17. Each original plane fits its own support reasonably:
median normalized residuals for IDs0..4 are0.001585,0.003649,0.004656,0.001492,
0.000140. This supports a finite-support gap at this location, not a demonstrated
coordinate mismatch. It does NOT determine whether gap is caused by wrong
ownership, unobserved contact, reconstruction error or lack of true adjacency.
Minimum joint-distance/bandwidth ratios:00199 range2.80--22.39;00190 range1.66--162.24.
Ratios show the role of the chosen radius; zero survivors is not proof that no
true contacts exist. Thresholds were not widened after observing the failure.

Important consequence: requiring CURRENT assigned supports to already touch
can suppress exactly the missing/overflow boundary that refinement must repair.
Use continuous support distance/uncertainty as features, not a hard prerequisite.
Independent learned RGB evidence remains promising but qualitative appearance
is not 3D correctness. Next experiment should separate RGB/geometry channels,
learn pair-conditioned ownership/finite support, retain a no-line control, and
distinguish visible boundary from occlusion. None implemented in this diagnostic.

Commands with E:/anaconda/envs/ai/python.exe:
```
-m unittest discover -s tests -p test_finite_structure_lines.py
diagnose_finite_structure_lines.py --run local_outputs/independent_evidence_train4_dev2_20261024_v1 --output local_outputs/finite_3d_line_diagnostic_20261024_v2
visualize_spatial_projection_audit.py --run local_outputs/independent_evidence_train4_dev2_20261024_v1 --output local_outputs/finite_3d_line_diagnostic_20261024_v2/projection_audit
```
Both diagnostic runs exit0;3 tests and syntax check pass. Both3D figures inspected.
Projection audit generated: same fixed sources with previously verified P95
0.1230/0.1085px. Report contains per-pair global line representations, rejection
details and source SHA256. NPZ schema1 contains samples/masks (no surviving
segments); results kept locally. Existing archival changes untouched.
