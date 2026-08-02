# E7 prospective clean Structured3D acquisition

Registration ID:

```text
E7_PREREG_clean_scene_acquisition_and_materialization_20260728_v1
```

Registered: 2026-07-28

## Authoritative prior state

E0 is Passed. E1 and E2 Failed. Evaluator-side expansion is stopped, E3 is
prohibited, E4-E6 are Blocked, the historical B-vs-M effect is Unverified, and
L/G/J remain paused. E6 must not be rerun or relaxed.

The local 183 provenance-unknown scenes and historical 17 scenes remain
permanently quarantined. E7 establishes provenance prospectively for scenes
that did not exist locally before the acquisition plan was frozen.

## Sole question

Determine whether at least 77 newly acquired official Structured3D training
scenes can be given complete first-seen, raw-file, derived-input, and source
provenance without reading GT semantics or method-quality results.

E7 does not compare methods and does not authorize B/M definition, training,
checkpoint work, inference, evaluation, expansion, refit, RANSAC, or L/G/J.

## Official source

Only the official Structured3D release is allowed:

* website: <https://structured3d-dataset.org/>
* official code/metadata: <https://github.com/bertjiazheng/Structured3D>
* allowed source IDs: `scene_00200` through `scene_02999`
* validation/test IDs `03000-03499` are excluded
* third-party preprocessed copies are forbidden

The official release requires acceptance of its Terms of Use and explicit
download access. If that access is unavailable, the verdict is
`Blocked_Source_Access`; no alternate source is allowed.

## Pre-acquisition freeze

Before any download, inventory:

* every local Structured3D scene directory;
* scene IDs referenced by configs and code;
* scene IDs and data roots referenced by local outputs;
* historical E0-E6 manifests and documentation;
* existing global-cache/direct-support paths.

Every ID present before E7 is excluded. The inventory records absolute paths,
data root, file-type counts, a metadata directory-tree hash, timestamp, and
inventory sources. Existing annotations are not semantically parsed.

## Candidate order

Candidate universe:

```text
official scene_00200-scene_02999
- every pre-E7 scene ID
- every scene ID in the official invalid-case metadata
```

Salt:

```text
LightRecon3D_E7_clean_acquisition_20260728_v1
```

Sort by:

```text
SHA256(salt + "|" + canonical_scene_id)
```

Freeze the first 128 IDs before any download. Do not replace, append, reorder,
or select them by visual or model quality.

## Acquisition and materialization

For every official archive, record source identity, file name, start/end time,
destination, size, SHA-256, command, exit status, and official checksum status.
For every scene, record first-seen time, source archive, raw directory, byte
counts, content tree hashes, RGB hashes, and annotation hashes.

Annotations remain sealed: E7 may check existence, size, file name, and
byte-level hashes only.

Before materialization, freeze input-preparation code, Git/source hashes,
DUSt3R and Stage1 weight hashes, configs, deterministic view selection,
preprocessing, seeds, schema, and environment. Materialization may run only
DUSt3R global-cache and Stage1 direct-support input preparation. B/M, training,
calibration, evaluation, GT parsing, expansion, refit, and RANSAC remain
forbidden.

## Eligibility and verdicts

A scene is `Clean_Materialized` only if it was frozen before acquisition,
absent before E7, acquired from the official archive with first-seen evidence,
technically complete, semantically sealed, materialized under the frozen input
preparation, and linked through recomputable hashes.

At least 77 eligible scenes gives `Ready_Clean_Pool`. Fewer gives
`Blocked_Insufficient_Clean_Scenes`.

The only verdicts are:

```text
Ready_Clean_Pool
Blocked_Source_Access
Blocked_Insufficient_Source_Universe
Blocked_Input_Preparation_Definition
Blocked_Insufficient_Clean_Scenes
Invalid
```

No verdict automatically authorizes data splitting, B/M definition, training,
checkpoint work, or evaluation.

## Required outputs

```text
results/clean_scene_acquisition_e7_20260728_v1/
docs/codex_returns/CODEX_RETURN_clean_scene_acquisition_e7_20260728.md
```

Blocked stages must be represented by explicit receipts. Success content must
never be fabricated.
