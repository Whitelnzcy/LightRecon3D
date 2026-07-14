# Evidence-Gated Plane Feedback Research Task

Date: 2026-07-14

Status: research and validation task note. No novelty or real-scene improvement
is claimed.

## Objective

Determine whether bounded multi-view planes provide trustworthy structural
feedback to a frozen pointmap foundation model, and whether held-out evidence
can identify harmful plane factors before they corrupt the alignment.

This task follows `docs/codex_tasks/stage3_global_alignment.md`; it does not
replace the required Stage3 coordinate, registry or provenance rules.

## Repository baseline

Implemented:

* DUSt3R global alignment and full pointmap cache export;
* one real P0 server smoke with a correctly rejected/rolled-back update;
* point-aligned Structured3D support/identity GT on the identical cache;
* exact support join by `(alignment_view_index, x, y)`;
* post-hoc PlaneGraph-BA v0 with one bounded Sim(3) correction per view;
* live plane-regularized DUSt3R optimization that can update depth, pose, focal
  and pairwise alignment variables;
* global acceptance/rollback based on plane residual and DUSt3R base loss;
* before/after support coloring, displacement heatmap and non-support movement;
* synthetic tests for accepted update, single-view exclusion and rollback.

Not yet established:

* pose, full-cloud, non-planar and bounded-support improvement;
* a learned or uncertainty-aware cross-view plane identity;
* evidence that the current manual identity is more reliable than a simple
  global geometry baseline;
* evidence that any result transfers to a second pointmap backbone.

## Research question

Current v1 uses fixed plane assignments derived from the same aligned geometry
that it regularizes. The core question is:

```text
Can held-out view evidence predict whether a plane factor will help or harm
foundation-model alignment better than confidence, Huber weighting and a single
whole-update rollback gate?
```

If the answer is no, do not build a larger association network around this idea.

## Proposed diagnostic before implementation

For each plane candidate with observations in at least three views:

1. Hold out one view.
2. Fit/refit the plane using the remaining views only.
3. Evaluate on the held-out view:
   * signed point-to-plane residual;
   * residual relative to DUSt3R confidence;
   * visible support overlap in the plane-local chart;
   * free-space/non-plane contradiction;
   * boundary distance and connected-component agreement.
4. Run a short single-factor feedback step from the same initial alignment.
5. Label the factor helpful/harmful from GT pose/full-cloud/plane change and
   non-support displacement.
6. Measure how well each pre-update score predicts that label.

This diagnostic must use the same global cache and exact pixel/view provenance
for every method. Do not use nearest-neighbour XYZ joins.

## Minimal mechanism if the diagnostic has signal

Add two levels of reversible variables:

```text
s_k     per-plane influence switch in [0, 1]
z_i,k   per-observation assignment including a null/dustbin state
```

Use a loss of the form:

```text
L = L_foundation
  + lambda_plane * sum_k s_k * sum_i z_i,k * rho(plane_incidence_i,k)
  + L_switch_prior
  + L_assignment_prior
```

Initialization and updates must respect:

* no plane factor from a single view;
* a held-out observation is not used to fit the plane that validates it;
* unknown/occluded support is not treated as free space;
* low-confidence evidence cannot dominate merely through point count;
* harmful factors can be disabled without irreversible union-find merges;
* the original DUSt3R loss remains active;
* accepted updates preserve a separate unmodified cache.

Start with deterministic scores and continuous switches. Only train a gate if
the deterministic diagnostic predicts harmful factors above simple baselines.

## Experiment order

### P0. Existing v1 server smoke

Run one retained group in a new output directory. Record:

```text
accepted/rejected
DUSt3R base loss before/after
plane residual mean/p95 before/after
support/full-cloud/non-support displacement
active plane and view counts
before/after PLY and displacement heatmap
```

This is a behavior check, not a metric claim.

### P1. Identical-cache quantitative baseline

Build point-aligned Structured3D GT through the saved registry and compare:

1. original DUSt3R alignment;
2. sequential RANSAC on the complete aligned cloud;
3. Stage1 support plus direct global SVD;
4. Stage1 support plus manual merge;
5. PlaneGraph-BA v0;
6. live plane feedback v1;
7. oracle-GT plane identity feedback as an upper bound.

Report alignment time separately from method time.

Implementation status on 2026-07-14:

* `build_structured3d_point_aligned_gt.py` now attaches visible Structured3D
  layout identities to the exact saved DUSt3R `(view_index, x, y)` cache order.
* It uses the stable cross-view `plane.ID`, not the per-file plane array index,
  and records every resize/crop transform and source layout path.
* GT plane parameters are refitted in the DUSt3R global frame. This supports
  plane identity/support evaluation but is not absolute metric GT; depth/pose
  reconstruction metrics still require Structured3D depth and camera poses.
* Server execution and the remaining P1 baselines are not yet complete.

Server update on 2026-07-14:

* The retained five-view group produced 715,848 filtered cache points, 693,839
  labeled points (96.93%), seven plane identities and one recorded source-cache
  SHA-256. The real GT NPZ/PLY and manifest are complete.
* The next baseline is scalable sequential RANSAC on that exact filtered cache.
  Hypotheses may be scored on a fixed deterministic subset, but final inliers,
  assignments and metrics use the full cache; this approximation is recorded
  in each method configuration.
* The identical-cache RANSAC server run completed in 18.74 seconds. It assigned
  715,695/715,848 points and returned five planes, but only three of seven GT
  planes matched at IoU 0.5: precision 0.60, recall 0.429, matched support IoU
  0.713, normal error 5.68 degrees, one fragmentation excess and two over-merge
  excesses. GT-support coverage was 0.99978. This is evidence that near-complete
  geometric coverage does not recover bounded support identity on this scene;
  it is not yet a cross-scene result.
* The next comparison lifts the already executed Stage2 plus manual-merge
  support labels onto the same filtered cache using only the exact saved
  `(alignment_view_index, x, y)` key. Repeated keys with disagreeing labels are
  dropped and counted instead of being resolved by an XYZ nearest-neighbour
  guess. The resulting GT/RANSAC/support table is still part of P1.
* That support lift produced six planes and 15,972 assigned cache points in
  0.248 seconds. The source contained 80,000 records but only 19,985 unique
  positive `(view,x,y)` keys; 3,431 keys (17.2%) had conflicting manual plane
  labels and were explicitly dropped. Another 582 resolved keys were absent
  from the confidence-filtered cache.
* Its dense-GT IoU row reports zero matches and 2.24% coverage, but this cannot
  be used alone as an identity verdict: the exporter deliberately samples
  sparse support, so dense IoU is strongly coupled to per-plane sampling
  density. The aggregate counts do not provide those per-plane upper bounds.
  P1 therefore retains dense coverage as a separate sampling statistic and
  adds support-conditioned IoU, pairwise identity F1, predicted cluster purity
  and GT completeness. These conditioned scores must always be reported with
  coverage and label rate; they do not treat sparse support as dense output.
* The corrected support-domain audit shows useful single-scene identity signal.
  Manual support has pairwise identity F1 0.819 versus RANSAC 0.704, predicted
  cluster purity 0.897 versus 0.756, and purity/completeness F1 0.875 versus
  0.798. Its three accepted planes have conditioned IoU 0.961 and 0.61-degree
  normal error. It still observes only six of seven GT planes, accepts only
  three, fragments two relationships and over-merges one.
* These manual numbers are provisional and optimistic because the first lift
  dropped 3,431 repeated pixel keys whose manual labels disagreed. The next P1
  audit therefore preserves all 80,000 observation records, maps GT by exact
  `(view,x,y)`, and compares manual merge with a no-cross-candidate-merge
  direct-global-SVD baseline on the same repeated records. This distinguishes
  useful merging from errors hidden by conflict dropping.

### P2. Leave-one-view-out factor audit

Create one row per `(scene, plane candidate, held-out view)` and record the
pre-update evidence, post-update influence and GT helpful/harmful label. Compare:

* DUSt3R confidence only;
* support count only;
* plane residual only;
* Huber residual weighting;
* held-out incidence;
* held-out incidence plus bounded visibility/free-space evidence.

Required outputs are a CSV/NPZ table, calibration plot, ROC/PR curves and a
failure gallery with exact source views and pixels.

### P3. Evidence-gated feedback

Compare from the identical initialization:

1. no plane feedback;
2. fixed factors plus Huber loss;
3. fixed factors plus whole-update rollback (current v1);
4. per-plane switch without held-out evidence;
5. held-out evidence-gated switches;
6. oracle helpful-factor switches.

The main evidence is the gap between items 4 and 5, and the remaining gap to
item 6. A lower plane residual alone is not sufficient.

### P4. Reversible identity, only after P2/P3

If held-out evidence predicts harmful factors, add soft association with a null
state. Compare with manual merge, constrained agglomeration and an oracle
identity. Do not add a neural association model until these deterministic
baselines are complete.

### P5. Backbone transfer

Only after real gains on DUSt3R, test the same factor definition on a second
backbone. Cross-backbone behavior is supporting evidence, not the mechanism.

## Metrics and stop/go rules

Always report:

```text
pose rotation/translation error when GT is available
full-cloud accuracy/completeness/Chamfer
non-planar geometry error
plane precision/recall and normal/offset error
fragmentation and over-merge
bounded-support IoU and boundary F-score
non-support displacement
DUSt3R base objective
runtime and peak memory
```

Go to P3 only if held-out evidence predicts harmful factors consistently better
than confidence/residual-only baselines across scenes, not merely on pooled
points. Continue toward a main contribution only if gated feedback improves
pose or full geometry together with plane metrics and does not trade the gain
for systematic non-planar degradation.

Stop or revise the direction if any of the following holds:

* accepted v1 updates only reduce their own plane residual;
* the oracle-identity or oracle-factor upper bound does not improve real metrics;
* held-out evidence is near chance at identifying harmful factors;
* gains disappear under identical-cache or view-order controls;
* improvements require manual scene-specific thresholds;
* a second backbone cannot expose compatible live optimization variables and
  the claimed contribution depends on DUSt3R internals.

## Required artifacts for every retained run

```text
command line and Git SHA
input manifest and cache checksum
view registry and coordinate convention
method configuration
metrics JSON/CSV
acceptance and rollback record
before/after point cloud visualizations
plane-factor diagnostic table
runtime and hardware note
unresolved errors or rejected groups
```

Do not overwrite previous caches, NPZs, PLYs or visualizations.

## Closest-work checklist before a paper claim

Re-read and compare the exact objective/variables against:

* Planar Bundle Adjustment and Efficient Second-Order Plane Adjustment;
* PAS-SLAM and data-association-free landmark SLAM;
* HSfM;
* MERG3R and TALO;
* Plane-DUSt3R and PLANA3R;
* AlphaTablets, NeuralPlane and CCGS;
* switchable-constraint/robust pose-graph methods.

Use `docs/codex_handoff/STRUCTURED_PLANE_LITERATURE_AUDIT.md` as the living
claim ledger and record every newly occupied claim there.
