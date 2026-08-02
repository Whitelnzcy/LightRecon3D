# E4 immutable-support rebaseline preregistration

Registration ID:

```text
E4_PREREG_immutable_support_rebaseline_20260727_v1
```

Date: 2026-07-27

## Mechanical decision after E2

Stop E-Bounded Expansion. Preserve E0 only as a non-mutating evaluator sanity
ceiling. Permanently stop the current tangent-bbox expansion, the E2 candidate
footprint expansion, evaluator-side support growth, E3 refit, and pressure-scene
expansion tuning.

The evaluator must not expand, delete, propagate, or reassign support and must
not refit plane parameters.

## Sole scientific question

When evaluator-side support and parameter mutation are prohibited, does the
historical quality difference between these two frozen methods remain?

```text
B: topology-logistic without learned split
M: historical 27K learned split/remerge followed by the frozen identity method
```

The historical mean Plane F1 values are `0.7087` and `0.7247`.

## Frozen inputs and validity gate

Use only the 17 historical scenes and only the artifacts entering the old
bounded expansion/refit:

```text
prediction support
frozen plane IDs
frozen plane parameters
point index / provenance
```

The experiment is valid only if all 17 inputs for both methods:

1. are demonstrably the historical pre-expansion artifacts;
2. have registered historical hashes which match the current files;
3. retain identical support and plane-parameter hashes before and after
   evaluation;
4. are evaluated without expansion, deletion, propagation, reassignment,
   refit, inference, training, or threshold changes.

If either method lacks a verifiable pre-expansion artifact or registered
historical hash, the verdict is `Blocked`. Do not use post-expansion outputs
and do not regenerate a historical artifact by rerunning inference.

## Required metrics if and only if the validity gate passes

Report per scene for B and M:

```text
Plane F1
true-positive plane count
matched IoU mean / median / minimum
partition pairwise precision / recall / F1
fragmentation excess
overmerge excess
gt_support_assignment_rate
gt_support_matched_coverage
full_cache_label_rate
gt_plane_label_rate
nonplane_false_assignment_rate
```

Also report plane-weighted, point-weighted, and per-scene results, immutable
and historical method effects, effect retention, and per-scene sign agreement.

## Mechanical verdicts

`Robust` requires immutable mean Plane F1 gain at least `+0.0080` plus all
registered consistency constraints.

`Artifact` applies if the gain is at most `+0.0040`, becomes negative, fewer
than 10/17 scene directions agree, or a registered core conclusion reverses.

Intermediate results permit exactly one hash/metric-only review. A missing
trusted pre-expansion artifact, missing historical hash, hash mismatch, or
unprovable parameter immutability is `Blocked`.

For `Blocked`, stop method comparison and only restore traceable archival
pre-expansion artifacts. Do not rerun a model to manufacture historical input.

## Required output

```text
results/immutable_support_rebaseline_20260727_v1/
  registration.md
  git_state.json
  input_hashes.json
  source_hashes.json
  immutability_audit.json
  per_scene_metrics.csv
  per_plane_metrics.csv
  historical_vs_immutable.csv
  summary.json
  run.log

docs/codex_returns/CODEX_RETURN_immutable_support_rebaseline_20260727.md
```
