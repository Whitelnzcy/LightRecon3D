# E14 Variable-View Clean Input Materialization Registration

Registration ID:

```text
E14_PREREG_variable_view_clean_input_materialization_20260729_v1
```

Authorization:

```text
execution_authorized = true
protocol_modification_authorized = false
```

E11 remains `Ready_Clean_Raw_Pool`; E12 and E13 remain blocked and must not be
resumed or rejudged. E14 uses the same 128 E11 scenes and order.

For each scene, group every `rgb_rawlight.png` by its direct room below
`2D_rendering`, choose the group with the largest count, break ties by
normalized relative room path, and sort images by normalized relative path.
Use the first five images when at least five exist and all four when exactly
four exist. Never fill across rooms, duplicate images, inspect image content,
use GT, or use model output to select views. The frozen result must contain 124
five-view scenes and the following four four-view scenes:

```text
scene_02465
scene_01698
scene_02008
scene_02208
```

Every input and output schema must preserve the actual `view_count`, view IDs,
relative image paths and input hashes. Four-view inputs must not be padded or
reported as five-view inputs.

Use the independent worktree
`E:\Study\code\LightRecon3D_e14_materialization` on branch
`codex/e14-variable-view-materialization`. Before E11 inference, implement and
test a raw-scene entrypoint independent of Stage2, B/M and GT; commit it; keep
the execution worktree clean; and freeze source/configuration hashes.

Select Stage1 mechanically. Priority one is a compatible, provenance-backed
`stage1_large_80m_train2048_sharded_v1/best.pt`. If absent, rank compatible
pre-E11 checkpoints by documented training scene count, optimizer step,
`best.pt` over `last.pt`, then normalized absolute path. Never use E11 outputs
or quality metrics. Freeze the inventory and selection.

DUSt3R weight SHA-256 is:

```text
9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd
```

Freeze the DUSt3R weight, preprocessing, pair construction, 4/5-view handling,
precision, deterministic settings and output schema.

Before E11 inference, run both a five-view and a four-view smoke using the
historical development scene `scene_00180`. Verify DUSt3R, Stage1,
writer/reader, schemas, pair/view indices and provenance. Do not inspect E11
GT or quality metrics.

Formal outputs:

```text
F:\Structured3D_clean_e11\global_cache_e14\scene_xxxxx\
F:\Structured3D_clean_e11\direct_support_e14\scene_xxxxx\
F:\Structured3D_clean_e11\materialization_staging_e14\scene_xxxxx.tmp\
```

Process all 128 scenes in frozen E11 order. Each scene may have one formal run
and at most two technical retries for CUDA OOM, crash, I/O or system
interruption. A resumed complete scene is skipped only after hash validation.
After the first E11 scene starts, source, checkpoint, view selection,
configuration and schema are immutable.

Each scene records its scene/view identity, room, paths and hashes; global and
direct-support file and key-array hashes; pixel/global-index provenance;
weights/configuration/commit; retry count; and UTC timestamps. Only technical
integrity is inspected.

All of the following remain forbidden and counted as zero: B/M inference or
training, checkpoint effectiveness evaluation, threshold calibration, formal
evaluation, GT matching or visualization, Plane F1, IoU, expansion, refit,
plane fitting, weighted fitting, RANSAC, support mutation and GT semantic
parsing or plane-count queries.

Allowed verdicts are:

```text
Ready_Clean_Input_Pool
Paused_Resumable
Blocked_E11_Input_Mismatch
Blocked_View_Selection_Mismatch
Blocked_Insufficient_Views
Blocked_Materialization_Implementation
Blocked_No_Compatible_Stage1_Checkpoint
Blocked_DUSt3R_Weight_Mismatch
Blocked_Incomplete_Input_Materialization
Invalid
```

Ready requires 128/128 global caches, direct supports and provenance records,
124 five-view scenes, four registered four-view scenes, zero GT semantic
calls, zero forbidden calls and a clean committed execution worktree.

The fixed return is:

```text
docs/codex_returns/CODEX_RETURN_variable_view_clean_input_materialization_e14_20260729.md
```

If Ready, the only next action is to stop and define/run the independent B/M
comparison; no further data acquisition, view selection or input-preparation
audit is allowed.
