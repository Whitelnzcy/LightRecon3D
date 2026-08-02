# Support to bounded evaluator contract map

Date: 2026-07-27

This map records code behavior before the E-contract repair.  Code facts take
precedence over earlier task descriptions.

## Data sources and coordinate contract

### Global cache

- Writer: `export_stage3_scene_plane_fusion.py`, DUSt3R global-alignment path.
- Reader: `global_plane_baselines.load_global_cache`.
- Fields after `min_conf=1.0`:
  - `points`: `(N,3)`, DUSt3R globally aligned XYZ;
  - `view_indices`: `(N,)`;
  - `pixel_xy`: `(N,2)`, integer `(x,y)` in
    `dust3r_aligned_pointmap`;
  - `confidence`, `colors`: `(N,)` and `(N,3)`.
- Pressure-scene `N`: 580,948 / 692,840 / 667,162 for
  `00184/00195/00197`.

### Point-aligned GT

- Writer: `build_structured3d_point_aligned_gt.main`.
- It loads the same global cache, applies the same confidence/finite mask, and
  stores `source_cache_indices`, `view_indices`, `pixel_xy`, `points`,
  `point_plane_ids`, `plane_normals`, and `plane_offsets`.
- Pressure-scene GT arrays have the same `N`, point order, and unique registry
  keys as the filtered cache.  Read-only validation found exact key
  uniqueness and `np.allclose(points, atol=1e-5)` for all three scenes.

### Sparse predicted support

- Source: Stage3/direct support and later split/identity variants.
- Required registry fields are
  `alignment_view_indices: (R,)` and
  `pointmap_pixel_xy: (R,2)`.
- The current diagnostic variants have `R=80,000`, about 20,000 unique keys
  and about 60,000 duplicate records.  These are repeated observations, not a
  full-cache point registry.

## Actual pre-repair call chain

```text
evaluate_intra_fragment_oracles.py
  pointwise_gt_evaluator_oracle
  -> exact_gt_join(sparse support keys, point-aligned GT keys)
  -> fit plane parameters on sparse labeled support
  -> save sparse support prediction

evaluate_all_fragment_common_cache.py
  -> load_global_cache(..., min_conf=1.0)
  -> load_support_prediction(...)
  -> resolve_support_assignments(...)
  -> build_tangent_bounds(...)
  -> expand_plane_seeds(...)
  -> evaluate_global_plane_baselines.evaluate_arrays(...)
```

The previous “pointwise GT evaluator oracle” was therefore not an exact
support E0: join, sparse-to-dense expansion, repeated assignment, plane
dropping and refit were coupled.

## Join behavior

### Sparse support to GT labels

- Function: `evaluate_bias_aware_fragment_association.exact_gt_join`.
- Packs `(alignment_view_index,x,y)` into an integer key, sorts GT keys, and
  uses `searchsorted`.
- It assumes GT keys are unique and does not audit duplicate GT keys.
- Unmatched records receive `-1`; no nearest-neighbour fallback is used.

### Sparse support to global cache

- Function: `lift_support_prediction_to_global_cache.resolve_support_assignments`.
- The global-cache registry must be one-to-one; duplicate cache keys raise.
- Duplicate positive support records with one label collapse to one key.
- A key with multiple positive labels is counted as conflicting and dropped
  under `conflict_policy="drop"`.
- Unmatched keys are counted and dropped.  No nearest-pixel, nearest-XYZ, or
  array-position fallback is used.
- Output: `source_assignment: (N,)`, `-1` for unassigned cache points.

The function currently reports label conflicts but not the requested split
between duplicate records that resolve to one versus several global point
indices.  Since the cache registry is enforced unique, a matched key can only
resolve to one global point; the new audit must record this explicitly.

## Initial plane parameters

- `evaluate_all_fragment_common_cache.main` reads `plane_normals` and
  `plane_offsets` from the sparse support prediction.
- `assignment_to_primitives` fits the joined sparse supports but, when
  expansion is enabled, its fitted normals/offsets are discarded; only the
  source ID list is used.
- The active expansion therefore starts from parameters stored in the support
  prediction rather than a guaranteed fit over the exact joined full-cache
  seed set.

## Tangent box construction

- Function: `evaluate_all_fragment_common_cache.build_tangent_bounds`.
- For each plane it selects joined cache seeds, uses their median as origin,
  removes normal displacement, and obtains two tangent axes by SVD.
- Bounds are independent lower/upper quantiles in the two tangent axes
  (`trim_quantile=0.01` in the frozen common evaluator).
- Expansion later adds a fixed metric margin (`0.02`) to this rectangle.
- No convex hull, connected-component constraint, scene-scale normalization,
  or local-sampling normalization is used.

## Expansion and refit

- Function: `evaluate_all_fragment_common_cache.expand_plane_seeds`.
- For every cache point it computes absolute distance to every active plane,
  rejects candidates outside the tangent rectangle, then assigns the point to
  the nearest plane if distance is at most `0.03`.
- Planes assigned fewer than 1,000 points are dropped and IDs compacted.
- Every loop ends by SVD-refitting normals and offsets from the entire newly
  assigned support.
- The loop count is `refit_iterations + 1`.  Consequently,
  `refit_iterations=0` still performs one refit.  The existing function cannot
  implement E1's “expansion on, refit off” contract.
- Tangent bounds retain their original world-space axes and extents while
  plane parameters drift.
- There is no seed-retention check, added-point purity check, drift limit,
  connected-component check, rollback, or support freeze after refit.

## Bounded matching and Plane TP

- Function: `evaluate_global_plane_baselines.evaluate_arrays`.
- `iou_matrix` compares predicted and GT label masks over the identical full
  cache.
- Hungarian assignment minimizes `1-IoU`.
- A matched pair is a Plane TP iff IoU is at least `0.5`.
- Plane precision/recall use all nonnegative predicted/GT IDs.
- `support_coverage` is the fraction of GT-labeled points with a nonnegative
  prediction.
- The common evaluator separately reports assignment rate as the fraction of
  all cache points with a nonnegative prediction.
- Fragmentation and overmerge use IoU `>=0.1`.
- The old summary reports only the mean IoU of accepted matches.  It does not
  preserve all Hungarian matched IoUs, their median/minimum, or per-plane
  before/after drift.

## Confirmed task-description differences

1. The previous GT sanity run used sparse support records and was not E0 exact
   full-cache support.
2. “Refit disabled” cannot be obtained by passing zero to the existing
   expansion function.
3. Current expansion is a trimmed tangent rectangle, not a bounded convex-hull
   footprint.
4. Current refit changes both plane parameters and the next iteration's
   support without a guard or rollback.
5. Sparse support duplicates describe repeated provenance records, while the
   filtered global cache and point-aligned GT registries are one-to-one.

These differences require a separate versioned evaluator-contract script.
The existing common evaluator and all old outputs remain unchanged.

## E0-discovered metric contradiction

The exact full-cache join subsequently proved that key join and matching are
identity-preserving on all three pressure scenes:

```text
Plane F1 = 1.0
TP = GT plane count
matched IoU median/minimum = 1.0 / 1.0
GT-support assignment = 1.0
support coverage = 1.0
```

However, the historical common-evaluator `assignment_rate` is
`mean(pred_label >= 0)` over **all** cache points.  Exact GT plane support
intentionally leaves GT-unlabeled/non-plane points at `-1`.  Therefore its
maximum possible full-cache assignment rate is the GT-labeled point rate:

```text
00184  0.959199
00195  0.970761
00197  0.976515
```

All are below the preregistered E0 threshold `0.98`.  This is a data/metric
contract contradiction, not evidence that the exact key join, Hungarian
matcher, or Plane TP rule is broken.  E1--E3 were not run, as required by the
E0 stop rule.

## Frozen denominator correction and E0 rejudgment

The ambiguous acceptance name `assignment_rate` is retired.  Let `C` be all
valid cache points, `G` GT plane points, `A` predicted-plane assigned points,
and `M` points assigned to the prediction matched to their GT instance:

```text
gt_support_assignment_rate      = |A intersection G| / |G|
gt_support_matched_coverage     = |M| / |G|
full_cache_label_rate           = |A| / |C|
gt_plane_label_rate             = |G| / |C|
nonplane_false_assignment_rate  = |A minus G| / |C minus G|
```

`evaluate_global_plane_baselines.evaluate_arrays`'s historical
`support_coverage` formula is `|A intersection G| / |G|`; it is retained as a
legacy field but is not the stricter matched-coverage metric.

The preserved v1 E0 tables were rejudged without loading point clouds or
rerunning join, geometry, matcher, or Plane TP.  Integer results are:

```text
scene   |A|      |G|      |M|      |A minus G|
00184   557245   557245   557245   0
00195   672582   672582   672582   0
00197   651494   651494   651494   0
```

Therefore all three E0 scenes pass the corrected metric contract.  The old
`E-Key/Data Contract` verdict is marked
`superseded_invalid_metric_denominator`; the old summary remains intact.
E1--E3 and L/G/J remain paused pending a later preregistration.

## E1 tangent-bbox safety verdict

The preregistered E1 run used the same point-aligned GT files, global caches,
point ordering, exact `(view,x,y)` join, initial GT plane parameters, matcher,
and TP thresholds as E0. Input file hashes matched the E0 record for all three
pressure scenes, and the reconstructed exact join was elementwise equal to the
GT labels.

E1 used the existing tangent-bound construction with its frozen parameters:

```text
trim quantile          0.01
bbox margin            0.02
normal-distance limit  0.03
minimum plane points   1000
```

It applied one assignment pass in a dedicated code path and performed no
plane fit (`refit_call_count=0`). Overlapping candidates were resolved by
minimum absolute plane residual; exact ties retained the lower original plane
ID, matching `np.argmin`. A plane below the minimum point count was dropped
without point reassignment.

All three scenes failed the frozen safety contract. The decisive result was:

```text
scene   F1      TP       IoU median/min   matched coverage   min seed retention
00184   1.000   13/13    0.921/0.609      0.979              0.795
00195   1.000   10/10    0.897/0.752      0.984              0.910
00197   0.200   1/5      0.392/0.254      0.085              0.277
```

The non-plane false-assignment rates were `0.998`, `0.991`, and `0.925`.
Multi-candidate points were respectively `48.4%`, `26.7%`, and `62.3%` of GT
plane points. In `00197`, 95.1% of all newly assigned points came from another
GT plane.

The mechanical verdict is:

```text
Failed
route: E-Bounded Expansion
```

Therefore the current tangent-bbox expansion is marked **unsafe** for formal
evaluation or default bounded-plane result construction. E0 remains the
trusted exact-support evaluator baseline. E2/E3 and L/G/J were not run or
resumed. The next permitted experiment is only an independently
preregistered E2 candidate bounded-footprint rule.

## E2 candidate bounded-footprint verdict

The preregistered E2 audit was implemented in an isolated path. It reused E0
inputs, exact unique join, plane IDs/parameters, matcher, and metric code.
Three predictions were written and hashed before GT scoring. Fit, weighted
fit, refit, and RANSAC call counts were all zero.

E2 replaced E1's trimmed tangent rectangle and fixed `0.03` distance band
with:

```text
deterministic tangent basis
projected eighth-neighbour local scale
accepted local Delaunay triangles
union clipped seed-local disks
adaptive normal residual threshold
third-neighbour locality gate
reject-all multi-candidate conflict handling
```

No bbox, convex hull, morphology, hole filling, or component bridge fallback
was used.

The run was contract-valid. E2 preserved all E0 seeds and plane identities:

```text
scene   seed retention   deletion/reassignment   F1 and TP
00184   1.0              0 / 0                   1.0, 13/13
00195   1.0              0 / 0                   1.0, 10/10
00197   1.0              0 / 0                   1.0, 5/5
```

It nevertheless failed the safety contract:

```text
scene   added nonplane points   cross-seed intrusion   IoU median/min
00184   4000                    0                      0.9916 / 0.9193
00195   3560                    63                     0.9946 / 0.9031
00197   335                     3224                   0.9995 / 0.9896
```

E0 exact support already contains every GT plane point, so every accepted
non-seed addition was GT nonplane/unlabelled. Total leakage fraction was
therefore `1.0` in every scene, exceeding the frozen `0.05` gate. Continuous
IoU also exceeded the allowed regression in each scene. Raw multi-candidate
rates passed and all such points were rejected, but cross-seed intrusion
failed for 00195 and 00197.

Mechanical verdict:

```text
Failed
contract valid: true
```

No E2 constant may be tuned under this registration. E2 must not be rerun,
and E3/L/G/J/refit/training remain stopped pending a new explicit mechanical
instruction.
