# E6 prospective method and split freeze

Registration ID:

```text
E6_PREREG_prospective_method_and_split_freeze_20260728_v1
```

Registered: 2026-07-28

## Authoritative state

```text
E0 exact-support evaluator: Passed
E1 tangent-bbox expansion: Failed
E2 candidate bounded-footprint expansion: Failed
E-Bounded Expansion: stopped
E3: prohibited
E4: Blocked
E5: Blocked
historical B-vs-M effect: Unverified
H0/H1: not tested
L/G/J: paused
```

E5 must not be rerun, repaired, or reinterpreted. Historical predictions,
missing historical source, and the pressure-selected historical checkpoint
cannot be promoted into a prospective baseline.

## Sole E6 question

Before any training or inference, determine whether the repository can freeze:

1. complete, committed, hashable prospective B and M definitions;
2. mutually disjoint training, calibration, selection, and final-evaluation
   scene lists;
3. a uniform M definition for every final-evaluation scene;
4. a prediction freeze barrier;
5. a non-mutating evaluator and forbidden-call contract.

E6 compares no method quality. Its only positive verdict is
`Ready_For_Execution_Preregistration`.

## Legacy quarantine

All scenes known to have entered historical training, threshold/checkpoint
selection, oracle analysis, E0-E5, or manual pressure analysis are quarantined.
If repository provenance cannot prove that a scene was never used for those
purposes, it is quarantined by default.

The historical 17-scene set and pressure scenes `00184`, `00195`, and `00197`
must be present in the quarantine.

## Candidate input contract

A clean candidate must:

* be outside the quarantine;
* have RGB, DUSt3R global-cache, and direct-support inputs;
* have a unique, provenance-resolved scene ID;
* be readable and hashable without reading GT semantic contents;
* not depend on historical B/M predictions.

Only paths, IDs, existence, shape, dtype, hashes, and non-semantic integrity
metadata may be inspected in E6.

## Frozen deterministic split

Salt:

```text
LightRecon3D_E6_clean_split_20260728_v1
```

Sort canonical scene IDs by:

```text
SHA256(salt + "|" + canonical_scene_id)
```

Use exactly the first 77 clean candidates:

```text
1-40   training
41-50  calibration
51-60  selection
61-77  final evaluation
```

No smaller split, alternate salt, or manual replacement is allowed. Fewer than
77 clean candidates is an immediate `Blocked` verdict.

## Method and execution freeze

Only after the candidate/split gate passes may E6 freeze new prospective B and
M configs and committed inference implementations. They must not claim to
recover the historical methods. M must use one code path, checkpoint contract,
and calibration contract for every evaluation scene, with no scene-ID branch
or evaluation whitelist.

The future execution order is:

```text
training
-> checkpoint candidates freeze
-> selection by macro partition pair F1
-> selected checkpoint hash freeze
-> one-time calibration
-> calibrated threshold hash freeze
-> final prediction
-> prediction hash freeze
-> GT load
-> read-only evaluation
```

The evaluator may match and score only. Expansion, refit, RANSAC, support/ID/
parameter/provenance mutation, and GT-guided prediction repair are forbidden.

## E6 prohibitions

E6 does not authorize:

* training or checkpoint creation/selection;
* B/M real-scene inference;
* GT semantic-array loading or quality evaluation;
* expansion, refit, RANSAC, or L/G/J restoration.

Only syntax/import/schema checks, synthetic fixtures, split/hash tests,
forbidden-call counter tests, and synthetic immutability tests are allowed.

## Mechanical stop

Any of the following produces `Blocked` before training or inference:

* fewer than 77 clean candidates;
* incomplete or overlapping split lists;
* unproven scene provenance;
* incomplete or uncommitted B/M definitions;
* non-uniform M;
* missing freeze barrier or non-mutating evaluator contract.

After a Blocked verdict, only provenance or method-definition repair under a
new Registration ID is allowed.

## Required outputs

```text
results/prospective_freeze_e6_20260728_v1/
docs/codex_returns/CODEX_RETURN_prospective_freeze_e6_20260728.md
```

The result directory records the registration, repository/environment state,
quarantine and candidate inventory, explicit split-list status, source/config
freeze status, future selection/calibration/freeze/evaluator contracts,
synthetic-test results, summary, and run log. It must not contain trained
checkpoints, formal B/M predictions, quality metrics, or method-effect
statistics.
