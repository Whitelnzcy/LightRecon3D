# Bounded Plane Identity and Support Pivot

Date: 2026-07-15

Status: active signal-audit task. No method novelty or cross-scene gain is
claimed.

## Decision being implemented

Stop treating plane incidence as a correction loss for DUSt3R alignment. Keep
the globally aligned pointmaps frozen and test the only surviving positive
signal: whether exact multi-view provenance can turn fragmented local plane
proposals into more accurate bounded plane identities and supports.

The proposed pipeline is:

```text
frozen DUSt3R aligned pointmaps
-> Stage1/Stage2 bounded local proposals
-> exact (view_index, x, y) observation records
-> provenance-aware positive and negative association evidence
-> reversible clustering with a null state
-> global refit on accepted support only
-> bounded editable plane primitives
```

This task does not change pose, depth, focal length, pairwise transforms or the
saved global cache. It does not use nearest-neighbour XYZ joins.

## Why this pivot is allowed

The feedback task failed its final upper-bound gate: neither manual support nor
GT identity had one of five views whose correction jointly improved fixed-gauge
structural correspondence and plane error.

One component did survive its same-record control. On scene 00180, direct
global SVD left 63 candidate identities with pairwise F1 0.120. Manual
cross-candidate aggregation reduced them to 11 identities and reached pairwise
F1 0.769, versus 0.704 for global RANSAC. This is sufficient to justify a
cross-scene signal audit, but insufficient to justify a paper claim or model.

## Claim boundary and closest occupied work

Plane correspondence, bounded multi-view plane reconstruction and unposed
plane-aware reconstruction are already occupied areas. In particular,
PlaneFormers jointly reasons about sparse-view plane correspondence and pose;
PlaneMVS reconstructs planes from calibrated multi-view images; classic
piecewise-planar work already optimizes bounded multi-view supports; and
Plane-DUSt3R and PLANA3R address unposed planar reconstruction.

Therefore the generic claim "multi-view plane association improves plane
reconstruction" is not available. The only mechanism worth testing is more
specific:

> Exact repeated-pixel provenance and explicit contradiction evidence can
> recover bounded editable plane identities from frozen foundation-model
> pointmaps without modifying the underlying geometry.

This is a hypothesis, not a novelty statement. It must first beat strong
geometry and association baselines across scenes.

## Phase I: cross-scene signal audit before new model code

Use 8-12 retained Structured3D validation groups with different layouts. Every
method for a scene must consume the identical global cache, confidence mask,
view registry and GT join.

Compare:

1. no cross-candidate merge / direct global SVD;
2. current manual merge;
3. global sequential RANSAC plus connected components;
4. constrained agglomeration using only plane geometry;
5. provenance-aware deterministic consensus;
6. GT-identity oracle partition as an upper bound.

Always report per scene rather than only pooled points:

```text
pairwise identity precision / recall / F1
cluster purity and GT completeness
support-conditioned IoU and boundary F-score
dense support coverage and observation label rate
fragmentation and over-merge
accepted / null / conflicting observation counts
normal and point-to-plane residual after frozen-geometry refit
runtime and peak memory
```

All metrics must preserve repeated observation records. Conflict dropping is a
separate ablation, never the primary score.

### Phase-I stop/go gate

Stop this pivot before learned components if any of these holds:

* manual/provenance consensus improves median pairwise F1 over RANSAC by less
  than 0.05;
* it wins against RANSAC on fewer than 70% of retained scenes;
* the gain disappears when conflicting repeated keys are retained;
* acceptable results require scene-specific thresholds;
* identity gains are obtained by collapsing support coverage or increasing
  over-merge;
* the GT-identity oracle has little headroom on bounded support metrics.

## Phase II: deterministic reversible consensus

Only after Phase I confirms cross-scene signal, represent each local proposal
as a node and retain provenance records as observations. Candidate merges may
use:

Positive evidence:

* consistent labels on exact repeated `(view_index, x, y)` keys;
* cross-view coplanarity after global alignment;
* compatible overlap in a plane-local bounded chart;
* agreement under leave-one-view-out refit.

Negative evidence:

* different labels or mutually exclusive ownership of the same exact pixel;
* incompatible same-view overlapping supports;
* normal/offset contradiction beyond uncertainty;
* disconnected bounded charts or boundary contradictions;
* a merge that increases over-merge more than it reduces fragmentation.

Use a reversible constrained agglomeration or signed-graph objective with an
explicit null/dustbin state. Do not use irreversible union-find as the only
state. Preserve the evidence and rejection reason for every accepted or
rejected edge.

## Phase III: learned association only if earned

Do not train a network merely because Phase II exists. A learned gate is
allowed only if deterministic provenance consensus:

1. beats manual merge and RANSAC under the Phase-I gate;
2. leaves a consistent gap to the GT-identity oracle;
3. exposes failure patterns that available features could plausibly predict;
4. has a scene-independent operating point.

The learned comparison must include geometry-only, appearance-only,
provenance-only and combined ablations, plus a null-state and calibration
analysis.

## Immediate implementation sequence

1. Generalize the existing all-record support audit to a batch manifest without
   regenerating pointmaps.
2. Add strict cache/registry/GT checksum validation and per-scene failure rows.
3. Freeze one shared metrics schema and generate JSON plus CSV summaries.
4. Run the existing direct-SVD, manual and RANSAC baselines on the retained
   caches.
5. Make the Phase-I decision before implementing a new clustering algorithm.

Long DUSt3R recaching, training and server execution remain explicit user-run
steps. Local work should first produce auditable scripts and exact commands.

## Required artifacts

```text
Git SHA and command line
batch scene manifest
cache, support and GT checksums
view registry and coordinate convention
per-scene and aggregate metrics
conflict and null-assignment diagnostics
failure gallery with exact views and pixels
runtime and hardware note
explicit stop/go decision
```
