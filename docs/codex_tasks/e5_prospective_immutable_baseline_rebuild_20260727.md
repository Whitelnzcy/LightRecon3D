# E5 prospective immutable baseline rebuild

Registration ID:

```text
E5_PREREG_prospective_immutable_baseline_rebuild_20260727_v1
```

Registered: 2026-07-27

## Authoritative prior state

```text
E0 exact-support evaluator: Passed
E1 tangent-bbox expansion: Failed
E2 candidate bounded-footprint expansion: Failed
E-Bounded Expansion: stopped
E3: prohibited
E4: Blocked
historical B-vs-M +0.0160 Plane-F1 difference: Unverified
L/G/J: paused
```

E4 must not be rerun or reinterpreted. Historical prediction NPZs may be
preserved as archives but are not E5 prediction inputs.

## Sole question

Under a completely non-mutating evaluator and prospective hashes of all code,
configuration, checkpoints, inputs, and predictions, does the old learned
split/remerge method M still consistently outperform topology-logistic B?

This is a baseline-reproduction protocol, not a new algorithm.

## Method-definition gate

B must be the historical topology-logistic method with an authoritative code
path, complete fixed configuration, input contract, inference entry point, and
postprocessing entry point.

M must be the historical 27K learned split/remerge method with its authoritative
checkpoint, model code, complete configuration, split/remerge behavior, input
contract, and inference entry point.

Both methods must generate new prospective predictions. Historical B or M
prediction NPZs cannot substitute for inference.

The verdict is immediately `Blocked`, before a prerun is authorized, if any
checkpoint is missing/unreadable, configuration is incomplete, the
authoritative code path cannot be determined, parameters would need to be
guessed from results, or the historical inference entry point cannot be
reproduced.

## Frozen evaluation set and leakage rule

Use exactly the historical 17 scenes, including 00184, 00195, and 00197. These
three pressure scenes may participate only in final evaluation. They may not
select a checkpoint, threshold, postprocessing rule, debug parameter, or scene
retention decision.

Before formal inference, freeze the source commit/diff, environment, source
hashes, B fixed-parameter/config hash, M checkpoint/config hash, scene order,
view/pixel/point order, common input hashes, deterministic settings, evaluator,
matcher, metric code, and forbidden-call list.

## Formal prediction barrier

Only after a valid prerun manifest may each method run exactly once on each
scene. All 34 predictions must be completed and hashed before GT evaluation.
Every prediction must contain method/scene identity, support assignments, plane
IDs and parameters, point/provenance/view/pixel identity, and the frozen
source/config/checkpoint hashes.

## Immutable evaluator

The evaluator may only read predictions and compute matcher/metric outputs. It
must not create a prediction NPZ or call expansion, plane fitting/refitting,
weighted fitting, RANSAC, support reassignment, support growth/deletion, plane
ID mutation, parameter mutation, or provenance mutation.

Prediction support, plane-ID, parameter, and provenance hashes must remain
identical before and after evaluation.

## Metrics and effect gate

If and only if the complete contract is valid, report the registered plane,
IoU, partition, fragmentation/overmerge, and five denominator-explicit metrics
per scene, per plane, plane-weighted, point-weighted, macro, and micro.

The registered paired effects are M minus B. `Confirmed` requires all of:

```text
mean Delta Plane F1 >= +0.0080
one-sided exact paired sign-test p <= 0.05
mean Delta matched IoU >= +0.0050
aggregate TP_M >= aggregate TP_B
mean matched coverage_M >= mean matched coverage_B - 0.005
fragmentation and overmerge not both worse for M
no pressure scene Delta Plane F1 < -0.05
```

The registered `Failed`, `Inconclusive`, `Invalid`, and `Blocked` routes and
their mechanical actions are exactly as supplied in the user preregistration.
No verdict restores evaluator-side expansion, E3, or L/G/J training.

## Required outputs

```text
results/prospective_immutable_baseline_e5_20260727_v1/
docs/codex_returns/CODEX_RETURN_prospective_immutable_baseline_e5_20260727.md
```

The result directory must preserve the registration, literature gate, scene
and prerun manifests, git/environment/source/input hashes, prediction and
immutability audits, all registered metric/effect/statistical schemas,
summary/log, and B/M prediction directories. Existing E0-E4 and historical
results must not be overwritten.
