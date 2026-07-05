# LightRecon3D Current State

Last updated: 2026-07-05

## 1. Current project state

The current pipeline already contains:

* Stage1 plane instance mask prediction;
* Stage2 pair-local geometry merge/refit;
* a runnable editable 3D plane demonstration;
* cached checkpoints and local outputs.

The current most stable Stage1 checkpoint is:

```text
local_outputs/stage1_large_80m_train2048_sharded_v1/best.pt
```

Reported validation metrics:

```text
val mean IoU              = 0.8431
val leakage               = 0.0621
predicted plane count     = 3.83
plane count abs error     = 0.61
```

Training on the raw 4096 set did not improve validation performance and showed a larger train-validation gap. Data-quality filtering and clean-hard rebalancing are separate ongoing work, but they are not the main subject of the current Stage3 task.

## 2. Confirmed Stage3 problem

The current Stage2/Stage3 geometry pipeline is still mainly pair-local.

Plane parameters or point coordinates generated independently for different image pairs cannot be safely merged unless those images have first been aligned into a common coordinate system.

The next implementation must therefore change the data flow rather than continue tuning local plane-merge thresholds.

## 3. Canonical Stage3 decision

The required flow is:

```text
Stage2 plane proposals
        +
original RGB image paths and support coordinates
        ↓
collect all unique views belonging to the same room / pair_group
        ↓
DUSt3R make_pairs
        ↓
DUSt3R inference
        ↓
global_aligner
        ↓
compute_global_alignment
        ↓
scene.get_pts3d()
        ↓
map each Stage2 plane support into the corresponding aligned pointmap
        ↓
merge and refit planes in a common global coordinate system
        ↓
bounded editable global plane primitives
```

Do not treat pair-local fitted plane parameters as globally comparable.

## 4. Stage2 NPZ schema requirements

The new Stage2 NPZ format must preserve enough information to reconstruct the original views and map plane support pixels into DUSt3R aligned pointmaps.

Minimum required metadata:

```text
schema_version

scene_name
pair_group

rgb_path1
rgb_path2

view_id1
view_id2

pixel_xy1
pixel_xy2
```

Each plane or support entry must be associated with the correct source view.

The implementation should also preserve, where available:

```text
original_hw1
original_hw2

stage1_mask_hw1
stage1_mask_hw2

pixel_coordinate_space
pixel_coordinate_order

plane_ids1
plane_ids2

support_confidence1
support_confidence2

pair-local plane parameters
pair-local refit residuals
```

Preferred coordinate storage:

1. integer support pixels in the original RGB coordinate system; and/or
2. normalized support coordinates in `[0, 1]`.

Do not store only 128×128 Stage1 mask coordinates without also storing the transform back to the original or DUSt3R input image.

## 5. Critical coordinate issue

`pixel_xy` is only useful when its coordinate convention is explicit.

The implementation must verify:

```text
Stage1 mask coordinates
-> original RGB coordinates
-> DUSt3R preprocessed image coordinates
-> scene.get_pts3d()[global_view_index]
```

DUSt3R may resize or crop input images. The Stage3 code must inspect the actual preprocessing used by the repository's DUSt3R version and apply the same transform.

Required metadata or transform information includes:

```text
original width and height
DUSt3R input width and height
resize scale
crop offset
mask width and height
coordinate order
```

Do not use simple shape scaling unless the preprocessing path proves that it is valid.

## 6. Critical view-index issue

The index in:

```python
scene.get_pts3d()[view_index]
```

corresponds to the order of unique images passed into DUSt3R inference/global alignment.

It must not be assumed to equal:

* a Structured3D camera number;
* a Stage2 pair-local view index;
* an NPZ array position;
* a filename suffix.

Stage3 must build an explicit view registry:

```text
canonical_rgb_path
-> global_alignment_view_index
```

Recommended registry record:

```json
{
  "canonical_path": "...",
  "scene_name": "...",
  "pair_group": "...",
  "source_view_id": "...",
  "alignment_view_index": 3,
  "original_hw": [720, 1280],
  "alignment_hw": [512, 512]
}
```

Use canonical absolute paths or normalized repository-relative paths to deduplicate images.

## 7. Room and pair grouping

Stage3 should group Stage2 outputs by a stable room-level key.

Current candidate:

```text
scene_name + pair_group
```

Before coding, verify what `pair_group` represents in the existing dataset.

It must be confirmed whether:

* one `pair_group` corresponds to one room/space;
* several pair groups may refer to the same physical room;
* image paths are sufficient for deduplication;
* Stage2 outputs contain overlapping image pairs.

Within one group:

1. collect all Stage2 NPZ records;
2. collect and deduplicate all RGB paths;
3. create a stable ordered view list;
4. run global alignment once for the complete group;
5. map every Stage2 support to its global view;
6. merge/refit all candidate planes globally.

## 8. DUSt3R global-alignment API

The implementation must inspect the repository's installed or vendored DUSt3R code before assuming exact signatures.

The intended API flow is conceptually:

```python
pairs = make_pairs(images, ...)
output = inference(pairs, model, device, ...)
scene = global_aligner(output, device=device, mode=...)
loss = scene.compute_global_alignment(...)
pts3d = scene.get_pts3d()
```

The actual imports, modes, schedules, iteration arguments, confidence access, and tensor formats must come from the repository's current DUSt3R version.

Do not copy an API call from a different DUSt3R revision without verification.

## 9. Global support mapping

For each Stage2 plane support:

1. find its source RGB path;
2. resolve that path through the view registry;
3. convert support pixels into DUSt3R pointmap coordinates;
4. index `scene.get_pts3d()[alignment_view_index]`;
5. optionally index the corresponding confidence map;
6. discard invalid or low-confidence points;
7. store the global 3D support points and source-view provenance.

Each mapped point should retain provenance:

```text
source NPZ
source image path
source plane/query ID
source pixel coordinate
global alignment view index
global XYZ
confidence
```

## 10. Global plane merge and refit

Cross-view merge decisions must use globally aligned support points.

Candidate compatibility signals:

```text
normal similarity
plane offset similarity in global coordinates
point-to-plane residual
support proximity
support overlap or adjacency
cross-view observation agreement
DUSt3R confidence
```

Do not merge solely because two pair-local normals are similar.

After grouping compatible supports:

```text
concatenate global support points
-> robust plane refit
-> estimate normal and offset
-> reject outliers
-> preserve bounded support
```

The final output should not be an infinite plane only. It must retain bounded support geometry and source-view observations.

## 11. Required implementation phases

### Phase 1: Read-only audit

Locate:

* Stage2 NPZ writer;
* Stage2 schema;
* Stage3 NPZ reader;
* current pair-local merge/refit;
* image loading and preprocessing;
* DUSt3R imports and actual global-alignment API;
* existing scene/pair grouping logic.

Do not modify code before reporting the actual data flow.

### Phase 2: Stage2 metadata/schema update

Add required metadata while preserving compatibility with old files where practical.

Add a schema version and a loader that produces a clear error for missing fields.

### Phase 3: Group and view registry

Implement:

```text
Stage2 records
-> group by room/pair_group
-> deduplicate image paths
-> stable ordered view registry
```

### Phase 4: DUSt3R global alignment

Run global alignment once per group.

Cache the aligned pointmaps and view registry so repeated merge experiments do not rerun DUSt3R unnecessarily.

### Phase 5: Support-to-global-point mapping

Map Stage2 support pixels to aligned global pointmaps, with explicit coordinate transforms and confidence filtering.

### Phase 6: Global merge/refit

Perform plane compatibility testing, clustering, robust global refitting, and bounded-support export.

### Phase 7: Diagnostics

Generate:

```text
group/view registry JSON
global alignment summary
mapped support statistics
unmapped/invalid pixel counts
global plane merge table
3D visualization before merge
3D visualization after merge/refit
```

## 12. Required tests

At minimum:

### Metadata round-trip test

Write and reload a Stage2 NPZ and verify all paths, dimensions, coordinate conventions, and support arrays are unchanged.

### View-registry test

The same image appearing in several Stage2 pairs must map to exactly one global alignment view index.

### Pixel mapping test

Select known image pixels and verify that the same coordinates retrieve the intended points from the aligned pointmap.

### Resize/crop test

Verify coordinate mapping on an image whose aspect ratio causes nontrivial preprocessing.

### Global consistency test

The same visible wall observed in multiple views should form one geometrically consistent support after alignment.

### Failure handling

Groups with missing files, too few views, invalid support coordinates, or failed alignment must be skipped with explicit diagnostics rather than silently producing planes.

## 13. Acceptance criteria

The task is not complete merely because the code executes.

Completion requires:

1. Stage2 NPZ contains sufficient reconstructable metadata.
2. Stage3 groups multiple views from the same room.
3. DUSt3R global alignment actually runs.
4. Every Stage2 support maps through an explicit view registry.
5. Support pixels map into `scene.get_pts3d()` correctly.
6. Cross-pair plane merging happens only in a common global frame.
7. At least one multi-view group has a visualized global point cloud with colored plane supports.
8. At least one global merge/refit result is exported.
9. Coordinate and view mapping tests pass.
10. Existing pair-local Stage2 behavior is not silently broken.

## 14. Current priority

The immediate priority is Phase 1, the read-only data-flow audit.

Do not start by tuning:

```text
normal thresholds
offset thresholds
support overlap thresholds
merge distance thresholds
```

Those parameters are not meaningful until all candidates are represented in a verified common global coordinate system.

## 15. End-of-session handoff protocol

At the end of every Codex work session, update this file with:

```text
date
branch
commit SHA
files changed
commands executed
tests completed
outputs generated
remaining problems
next exact step
```

Never leave important project state only in the Codex chat response.
