# LightRecon3D Codex Instructions

## Required context

Before modifying any code, always read:

1. `docs/codex_handoff/CURRENT_STATE.md`
2. The task file named by the user under `docs/codex_tasks/`
3. `git status`, `git diff --stat`, and the recent commit history
4. Any existing partial implementation related to the current task

Do not assume a new Codex thread contains the previous thread's conversation history. Repository files and Git state are the source of truth.

## Project objective

LightRecon3D aims to reconstruct bounded editable plane primitives from unordered or weakly posed indoor images.

A bounded plane primitive consists of:

* a visible 2D/3D support region;
* a plane equation represented by normal and offset;
* stable identity or correspondence across views where possible.

The intended high-level pipeline is:

```text
images
-> DUSt3R / MASt3R features and pointmaps
-> Stage1 bounded 2D plane proposals
-> Stage2 local geometry validation and refit
-> Stage3 DUSt3R global alignment
-> global-coordinate plane merge and refit
-> bounded editable plane primitives
```

## Current architectural rule

Do not continue tuning pair-local plane merge thresholds as the main solution.

The next Stage3 implementation must use actual DUSt3R global alignment:

```text
make_pairs
-> inference
-> global_aligner
-> compute_global_alignment
-> scene.get_pts3d()
```

Plane support pixels must be mapped to the globally aligned pointmaps before cross-view or cross-pair plane merging.

## Repository safety

* Do not delete or overwrite checkpoints, feature caches, NPZ outputs, or visualization results.
* Do not run long training or recache DUSt3R features unless explicitly requested.
* Preserve backward compatibility for existing Stage2 NPZ files where practical.
* Use schema versioning for new NPZ formats.
* Do not silently guess coordinate conventions, image ordering, or view IDs.
* Do not assume Stage2 pair-local coordinates are globally comparable.
* Do not modify unrelated files.
* Do not overwrite uncommitted user changes.

## Implementation discipline

Before implementing:

1. Identify the actual Stage2 NPZ writer.
2. Identify the current Stage3 reader and merge/refit code.
3. Inspect the vendored DUSt3R version and confirm the actual API signatures.
4. Trace image preprocessing, resize, crop, mask resolution, and pixel-coordinate mappings.
5. Report the existing data flow and any ambiguity.

After implementing:

1. Run syntax checks and focused tests.
2. Validate shard/path/view mappings.
3. Generate at least one visual global-alignment diagnostic.
4. Update `docs/codex_handoff/CURRENT_STATE.md`.
5. Record changed files, commands, outputs, unresolved issues, and the latest commit SHA.

## Coordinate conventions

Every stored pixel coordinate must have an explicit convention:

* order: `(x, y)` or `(row, col)`;
* source resolution;
* whether coordinates refer to original RGB, resized RGB, Stage1 mask, or DUSt3R input;
* resize and crop transform;
* whether coordinates are integer centers or normalized coordinates.

Never map support pixels into `scene.get_pts3d()` using shape assumptions alone.

## Communication

Be precise and evidence-based.

Clearly distinguish:

* implemented behavior;
* planned behavior;
* inferred behavior;
* unverified assumptions.

Do not report a test, visualization, or metric as completed unless it was actually executed.
