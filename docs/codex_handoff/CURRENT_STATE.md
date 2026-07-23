# LightRecon3D Current State

Last updated: 2026-07-15

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

## 16. Session update: 2026-07-05 Phase 2 metadata/schema baseline

Branch: `codex/bounded-support-head`

Base commit SHA: `66066ad51898df0a17c2fd46166df920b96b7932`

Files changed in this session:

```text
dataloaders/s3d_dataset.py
export_stage1_pred_support_teacher_npz.py
export_stage2_learned_region_merge_editables.py
tests/test_stage2_metadata_schema.py
docs/codex_handoff/CURRENT_STATE.md
```

Implemented behavior:

* `Structured3DDataset` now returns per-view `view_id`, original RGB height/width, and resized Stage1 input height/width for pair samples.
* Stage1 teacher export now writes `schema_version=2`, `json_path1/json_path2`, `view_id1/view_id2`, `original_hw1/original_hw2`, `stage1_input_hw1/stage1_input_hw2`, `stage1_mask_hw1/stage1_mask_hw2`, explicit pixel coordinate convention fields, `pixel_xy1`, empty `pixel_xy2`, and per-point `support_source_view` for current view1-only support.
* Existing `pixel_xy` remains as a backward-compatible alias for `pixel_xy1`.
* Stage2 learned merge export now writes `schema_version=2`, records `source_schema_version`, passes through the new global-alignment metadata, and validates per-point metadata lengths before writing output.
* Added a lightweight metadata schema test for Stage2 passthrough constants and per-point length validation.

Commands executed:

```text
python -m py_compile dataloaders/s3d_dataset.py export_stage1_pred_support_teacher_npz.py export_stage2_learned_region_merge_editables.py tests/test_stage2_metadata_schema.py
python tests/test_stage2_metadata_schema.py
```

Tests completed:

```text
py_compile: passed
Stage2 metadata unittest: 3 tests passed
```

Outputs generated:

```text
tests/test_stage2_metadata_schema.py
```

Remaining problems:

* No real Stage1/Stage2 export was run in this local session, so the new NPZ schema has not yet been validated on a real Structured3D sample.
* No DUSt3R global alignment was run; this session intentionally did not enter Phase 4.
* `pixel_xy2` is currently empty because the existing Stage1 export only produces view1 support.
* Stage3 still needs an explicit group/view registry and coordinate transform implementation before global support mapping is trustworthy.
* The worktree still contains inherited unrelated/uncommitted changes from previous sessions.

Next exact step:

Proceed to Phase 3: implement a Stage3 record loader plus `scene_name + pair_group` grouping and a deterministic view registry that deduplicates `rgb_path1/rgb_path2`, validates metadata, and records canonical path -> alignment view index before any global alignment run.

## 17. Session update: 2026-07-05 Phase 3 view-registry dry-run baseline

Branch: `codex/bounded-support-head`

Base commit SHA: `66066ad51898df0a17c2fd46166df920b96b7932`

Files changed in this session:

```text
validate_stage3_view_registry.py
tests/test_stage3_view_registry_validator.py
docs/codex_handoff/CURRENT_STATE.md
```

Implemented behavior:

* Added `validate_stage3_view_registry.py`, a no-model/no-DUSt3R dry-run validator for Stage2 NPZ metadata.
* The validator groups records by `scene_name + pair_group`, deduplicates `rgb_path1/rgb_path2` into a deterministic `canonical_path -> alignment_view_index` registry, and records per-view source path, view id, original hw, Stage1 input hw, and Stage1 mask hw.
* It reports summary metrics including record count, valid record count, groups, unique views, groups with 3+ views, missing required metadata, records with errors, duplicate view conflicts, and unmapped support records.
* It validates `pixel_xy1` shape/range, optional empty `pixel_xy2`, `support_source_view`, and per-record required fields before any DUSt3R run.
* Added synthetic NPZ tests for overlapping-pair view deduplication, invalid support coordinates, and missing required metadata reporting.

Commands executed:

```text
python -m py_compile validate_stage3_view_registry.py tests/test_stage3_view_registry_validator.py
python tests/test_stage3_view_registry_validator.py
python tests/test_stage2_metadata_schema.py
```

Tests completed:

```text
py_compile: passed
Stage3 view-registry validator unittest: 3 tests passed
Stage2 metadata unittest: 3 tests passed
```

How to validate on server after pulling:

```text
python validate_stage3_view_registry.py \
  --input_dir /gemini/data-1/lightrecon_runs/stage2_region_merge_v1/learned_merge_npz \
  --output_json /gemini/data-1/lightrecon_runs/stage3_view_registry_dryrun/view_registry_summary.json \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --check_files
```

Useful acceptance metrics for this phase:

```text
metadata_complete_rate == 1.0
records_with_errors == 0
duplicate_view_conflict_count == 0
unmapped_support_records == 0
groups_with_3plus_views > 0 for multi-view validation subsets
```

Remaining problems:

* The validator only proves metadata/group/view-registry integrity. It does not run DUSt3R and does not validate resize/crop mapping into `scene.get_pts3d()`.
* Real server validation still needs freshly exported schema v2 Stage1/Stage2 NPZs; old local NPZs are expected to fail missing metadata checks.
* Phase 4 should consume this registry when running global alignment, rather than rebuilding implicit view order inside the fusion path.

Next exact step:

Commit and push the Phase 2 + Phase 3 metadata/view-registry baseline, then run the validator on a small fresh Stage2 export on the server. If the dry-run metrics are clean, proceed to Phase 4 global alignment cache implementation.

## 18. Session update: 2026-07-10 research-necessity baseline scaffold

Branch: `codex/bounded-support-head`

Current commit SHA: `5bd3e42` (worktree changes are not committed)

Implemented behavior:

* Stage3 global alignment now writes a method-independent cache containing every aligned pointmap pixel: XYZ, RGB, confidence, explicit alignment view index, `(x,y)` pointmap pixel, registry, and alignment loss.
* Added sequential 3D plane RANSAC on that shared cache. Each infinite-plane inlier set is split with radius-connected Euclidean components before separate SVD refits.
* Baseline output uses the existing editable-plane core fields (`points`, `colors`, `point_plane_ids`, `plane_normals`, `plane_offsets`, and counts), plus parameter JSON/TXT and colored PLY.
* Stage3 gained explicit `--merge_mode none|manual`. With Stage1 support input, `none` is the direct global-SVD ablation and `manual` is the hand-threshold merge ablation. RegionMergeMLP/full-method comparison uses its Stage2 output as input.
* Added point-aligned GT evaluation for plane precision/recall, sign-invariant normal error, point-to-plane residual, coverage/matched IoU, count error, fragmentation, over-merge, and recorded runtime.
* No claim that Stage1 support or the full method beats RANSAC is supported yet.

Files changed/added:

```text
export_stage3_scene_plane_fusion.py
global_plane_baselines.py
evaluate_global_plane_baselines.py
tests/test_global_plane_baselines.py
tests/test_evaluate_global_plane_baselines.py
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile global_plane_baselines.py evaluate_global_plane_baselines.py export_stage3_scene_plane_fusion.py tests/test_global_plane_baselines.py tests/test_evaluate_global_plane_baselines.py
<bundled-python> -m unittest tests.test_global_plane_baselines tests.test_evaluate_global_plane_baselines
```

Results:

```text
py_compile: passed
focused baseline/evaluation tests: 5 passed
```

Unresolved / not executed:

* Real Structured3D val metrics have not been produced locally. The retained showcase manifests reference `/gemini/data-1/...`; this Windows workspace has neither that Structured3D root nor the server DUSt3R weights.
* Existing showcase NPZs contain only mapped Stage1/Stage2 support points (for example 80,000 points), not the full aligned pointmaps, so using them for RANSAC would violate the identical-input requirement.
* A point-aligned Structured3D GT cache writer still needs to be connected to the exact same alignment view/pixel registry before quantitative evaluation. The evaluator intentionally rejects predictions whose point array differs from GT.
* Runtime must be measured in one server job with one cached alignment shared by all methods; alignment time should be reported separately from method time.

Next exact step:

On the server, regenerate the selected val groups once to create the new global-cloud caches, build point-aligned GT labels through the saved `(view_index, x, y)` registry, then run RANSAC, Stage1+SVD, Stage1+manual merge, and the full Stage2 input on exactly those caches. Do not publish comparative claims until the resulting CSV is complete.

## 19. Session update: 2026-07-12 structured-plane novelty audit

Branch: `codex/bounded-support-head`

Current commit SHA: `5bd3e42` (worktree changes are not committed)

Output generated:

```text
docs/codex_handoff/STRUCTURED_PLANE_LITERATURE_AUDIT.md
```

Literature finding:

* Existing work already covers query plane recovery, unknown-pose two-view plane reconstruction, posed multi-view MVS, learned 3D plane tracking/fusion, multi-view-consistent plane embeddings, explicit planar primitive optimization, visible/occluded plane extent, and polygonal primitive assembly.
* Therefore no current Stage1/Stage2/Stage3 component is independently defensible as novel.
* The leading research hypothesis is uncertainty-aware evidence fusion for bounded plane identity and support under unordered, unposed, imperfect foundation-model pointmaps.
* This is explicitly recorded as a hypothesis, not a novelty claim.

Second-pass correction:

* Additional closest works (Plane-DUSt3R, PLANA3R, AlphaTablets, NeuralPlane, NOPE-SAC, PlaneRecTR++, PlanarNeRF and CCGS) substantially weaken uncertainty-only, bounded-support-only and unposed-plane-only novelty claims.
* The recommended main direction is now **PlaneGraph-BA**: a training-free structural adapter for frozen pointmap foundation models. It uses an instance-level trimmed plane graph both as output and as structural landmarks to jointly refine per-view/submap Sim(3), plane identities and plane parameters.
* Uncertainty-aware evidence fusion is retained only as a component, not the main contribution.

Tests/training:

```text
No code test or training was run; this session was a literature and research-design audit.
```

Next exact step:

Complete the shared-global-cloud RANSAC experiment and construct point-aligned GT. Then measure whether DUSt3R confidence, bootstrap plane uncertainty, reprojection disagreement, and visibility/free-space evidence predict hard-merge failures. Implement a non-learned uncertainty-normalized association baseline before training another network.

## 20. Session update: 2026-07-12 PlaneGraph-BA v0 implementation

Branch: `codex/bounded-support-head`

Base commit SHA: `5bd3e42` (new changes are not committed at the time of this handoff entry)

Implemented behavior:

* Added `planegraph_ba.py`, an additive structural-feedback layer for a frozen
  DUSt3R global pointmap cache.
* Stage3 global support output now preserves exact cache join provenance:
  `alignment_view_indices` and `pointmap_pixel_xy` in explicit `(x,y)` DUSt3R
  aligned-pointmap coordinates.
* PlaneGraph-BA joins support observations to the complete global cloud using
  `(alignment_view_index, x, y)`, not floating-point XYZ nearest neighbours.
* v0 fixes one reference view and alternates confidence-weighted global plane
  SVD refits with bounded per-view Sim(3) corrections.
* Point-to-plane updates use Huber weights, quadratic priors around the frozen
  DUSt3R alignment, parameter bounds, damping, and a step-acceptance check.
* Only plane identities observed in at least `min_plane_views` affect alignment;
  single-view planes are still refitted and exported.
* Output uses the existing editable-plane core fields and adds original/corrected
  global points, per-view Sim(3), before/after PLY, and optimization history.
* Stage3's SVD plane helper is now dependency-light, allowing mapping tests to
  run without importing the Torch-dependent Stage2 training module.

Files added/changed for this implementation:

```text
planegraph_ba.py
docs/codex_handoff/PLANEGRAPH_BA_V0.md
export_stage3_scene_plane_fusion.py
tests/test_planegraph_ba.py
tests/test_stage3_scene_plane_fusion_mapping.py
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile planegraph_ba.py export_stage3_scene_plane_fusion.py tests/test_planegraph_ba.py tests/test_stage3_scene_plane_fusion_mapping.py
<bundled-python> -m unittest tests.test_planegraph_ba tests.test_stage3_scene_plane_fusion_mapping tests.test_global_plane_baselines tests.test_evaluate_global_plane_baselines
<bundled-python> planegraph_ba.py --global_cloud_npz <synthetic-cache> --support_npz <synthetic-support> ...
```

Results:

```text
py_compile: passed
focused tests: 9 passed
synthetic CLI smoke: passed
synthetic mean absolute plane residual: 0.0329946 -> 0.0000247
support cache-key mapping: 720 / 720
```

Generated diagnostic:

```text
local_outputs/planegraph_ba_synthetic_smoke/planegraph_ba_before_after.png
local_outputs/planegraph_ba_synthetic_smoke/result/synthetic_room_planegraph_ba_v0_before.ply
local_outputs/planegraph_ba_synthetic_smoke/result/synthetic_room_planegraph_ba_v0_after.ply
```

Scientific limitations / unverified items:

* The synthetic result validates implementation behavior only. No Structured3D
  or real-scene improvement has been measured.
* v0 applies one Sim(3) per view; it does not yet correct intra-view pointmap
  deformation and does not fine-tune DUSt3R.
* The current cache does not retain DUSt3R pair correspondences, so v0 contains
  plane residuals plus foundation-alignment priors, but not an explicit match
  reprojection term.
* Incorrect Stage1/Stage2 plane identity can still bias the optimizer. The
  existing robust loss and bounds limit damage but do not solve association.
* Real acceptance requires pose/geometry/non-planar metrics, not merely lower
  point-to-plane residual.

Next exact step:

Regenerate one Structured3D val group with the new Stage3 provenance, run
PlaneGraph-BA v0 on the shared global cache, and compare original DUSt3R,
post-hoc SVD, sequential RANSAC, manual merge, and learned-support PlaneGraph-BA
using both plane metrics and full/non-planar geometry metrics. If alignment and
plane metrics do not improve together, stop or revise this direction before
adding pointmap deformation or learned uncertainty.

## 21. Session update: 2026-07-13 Stage1 assignment performance patch

Branch: `codex/bounded-support-head`

Implemented behavior:

* Added an exact rectangular Hungarian assignment solver with no SciPy dependency.
* `train_stage1_multiscale_pair.py` now uses this solver instead of the exponential
  subset-DP matcher. The network, costs, loss terms, data, and evaluation cadence
  are unchanged.
* Added interval step time, ETA, and epoch timing to train/validation logs.
* The override is local to the multiscale-pair entry point; inherited user changes
  in the shared Stage1 training files were not modified.

Diagnosis:

* The prior run showed about 610% CPU, 8--10% GPU utilization, and low GPU clocks.
* One step performs up to eight assignments (final, aux32, aux64, benchmark32 for
  two views). With 12 queries the old matcher explored up to 4096 subsets per
  target sequence, repeatedly forcing GPU-to-CPU synchronization.
* Actual server speedup remains unverified until a one-epoch same-cache benchmark.

Files changed:

```text
stage1_fast_assignment.py
train_stage1_multiscale_pair.py
tests/test_stage1_fast_assignment.py
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile stage1_fast_assignment.py train_stage1_multiscale_pair.py tests/test_stage1_fast_assignment.py
PYTHONPATH=. python tests/test_stage1_fast_assignment.py
git diff --check -- stage1_fast_assignment.py train_stage1_multiscale_pair.py tests/test_stage1_fast_assignment.py
```

Results:

```text
py_compile: passed
fast assignment tests: 4 passed, including exhaustive brute-force comparisons
diff check: passed (line-ending warning only)
```

Next exact step:

Let the active server epoch finish and save, stop it with SIGINT, pull this
commit, then run one timed epoch from the saved checkpoint in a new output
directory before scheduling the remaining epochs.

## 22. Session update: 2026-07-13 differentiable plane feedback v1

Branch: `codex/bounded-support-head`

Implemented behavior:

* Added `plane_regularized_alignment.py`. Unlike PlaneGraph-BA v0's cached
  per-view Sim(3) correction, v1 continues optimization inside DUSt3R's live
  `PointCloudOptimizer`, so plane incidence gradients update depth, pose, focal,
  and pairwise alignment parameters while DUSt3R weights remain frozen.
* Plane support joins use explicit `(alignment_view_index, x, y)` pointmap
  provenance. Repeated observations are deduplicated.
* Only multi-view planes with sufficient support drive alignment. Confidence
  weights and per-plane aggregate weights are normalized.
* Added a rollback gate: accept only if robust plane residual improves and the
  original DUSt3R objective stays within a configured degradation limit.
* Stage3 writes the unmodified method-independent cache before feedback and an
  accepted corrected cache separately. It also writes sampled before/after PLY,
  support/full-cloud/non-support displacement, loss history, and acceptance
  diagnostics.
* Before/after diagnostics now dim non-support RGB and color every mapped plane
  support consistently; a blue-to-red displacement heatmap PLY makes small
  structural corrections visible instead of relying on raw RGB inspection.
* Plane-feedback outputs use a distinct filename suffix and should be run in a
  new output directory, preserving prior results.

Files changed/added:

```text
plane_regularized_alignment.py
export_stage3_scene_plane_fusion.py
tests/test_plane_regularized_alignment.py
docs/codex_handoff/PLANE_FEEDBACK_V1.md
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile plane_regularized_alignment.py export_stage3_scene_plane_fusion.py tests/test_plane_regularized_alignment.py
python tests/test_plane_regularized_alignment.py
python tests/test_stage3_scene_plane_fusion_mapping.py
python tests/test_global_plane_baselines.py
python tests/test_evaluate_global_plane_baselines.py
git diff --check -- plane_regularized_alignment.py export_stage3_scene_plane_fusion.py tests/test_plane_regularized_alignment.py
```

Results:

```text
plane-feedback tests: 3 passed (accepted update, single-view exclusion, rollback)
Stage3 mapping test: 1 passed
global RANSAC baseline tests: 3 passed
evaluation tests: 2 passed, with a local SciPy/NumPy binary compatibility warning
py_compile and diff check: passed
```

Scientific status:

* This is implemented behavior with synthetic tests, not evidence of real-scene
  improvement.
* Current cross-record plane identity for the first smoke test still comes from
  manual global geometric merge. Learned/uncertainty-aware association remains
  a required follow-up if the feedback mechanism passes its acceptance gate.
* Point-aligned Structured3D GT and full before/after metrics are still missing.

Next exact step:

Run one retained showcase group on the server with `--plane_feedback`, inspect
the acceptance record, DUSt3R base loss, plane residual, non-support movement,
and before/after PLY. Do not batch all validation groups until this smoke test
passes without harmful geometry movement.

## 23. Session update: 2026-07-14 third-pass innovation archive

Branch: `codex/bounded-support-head`

Current commit SHA: `cbf37ef` (documentation changes from this session are not
committed at the time of this entry)

Files changed/added in this session:

```text
docs/codex_handoff/STRUCTURED_PLANE_LITERATURE_AUDIT.md
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Research finding:

* A third-pass audit checked the exact v1 mechanism against planar bundle
  adjustment, plane SLAM/data association, HSfM, MERG3R, TALO, VGGT-SLAM++ and
  G3T.
* “Plane constraints in BA,” “external structure fed back into DUSt3R,”
  “training-free/model-agnostic adapter,” “latent landmark association” and
  “per-view Sim(3) correction” are not defensible standalone novelty claims.
* The leading hypothesis is now evidence-gated plane feedback: use held-out
  views, bounded visibility/free-space evidence, reversible factor switches and
  per-factor influence checks to prevent self-confirming plane constraints from
  corrupting live pointmap alignment.
* This is explicitly a hypothesis. Switchable constraints and joint data
  association are established; novelty would depend on the exact held-out
  bounded-evidence mechanism inside a frozen pointmap optimizer and on real,
  cross-scene evidence.

Task note generated:

```text
docs/codex_tasks/evidence_gated_plane_feedback.md
```

It records the P0-P5 experiment order, identical-cache baselines, metrics,
required artifacts and stop/go rules. The immediate implementation priority did
not change: first run the existing v1 single-group server smoke, then construct
point-aligned GT before adding a learned association or a larger optimizer.

Commands/tests completed:

```text
git status --short
git diff --stat
git log -6 --oneline --decorate
git diff --check -- docs/codex_handoff/STRUCTURED_PLANE_LITERATURE_AUDIT.md docs/codex_tasks/evidence_gated_plane_feedback.md docs/codex_handoff/CURRENT_STATE.md
```

Results:

```text
Markdown diff check: passed (line-ending conversion warnings only)
No code test, training, server run or new global-alignment visualization was
executed. This session only inspected the repository, audited primary
literature and updated Markdown archives.
```

Inherited unrelated tracked and untracked user changes remain in the worktree
and were not modified by this documentation session.

Unresolved:

* Real-plane-feedback acceptance and geometry improvement remain unverified.
* Point-aligned Structured3D GT remains missing.
* The proposed held-out gate has not been implemented or shown to predict
  harmful plane factors.
* No systematic-review completeness or novelty claim is asserted.

Next exact step:

Run P0 from `docs/codex_tasks/evidence_gated_plane_feedback.md` on one retained
server group without overwriting previous outputs. Archive the exact command,
Git SHA, view registry, acceptance JSON and before/after diagnostics before
deciding whether to implement the held-out factor audit.

## 24. Session update: 2026-07-14 P0 server launch preparation

Branch: `codex/bounded-support-head`

Base commit SHA: `cbf37ef`

Implemented behavior:

* Added an optional repeatable `--path_prefix_map SOURCE=DESTINATION` to Stage3.
  It remaps stored RGB path prefixes at read time without modifying source NPZs,
  allowing the same schema-v2 exports to run across server and Windows roots.
* The same remapped path is used both when registering DUSt3R images and when
  resolving each support point's source view.
* Added focused tests for prefix parsing/remapping and support-to-global-view
  lookup after remapping.
* Added `run_plane_feedback_p0.sh`, a server-ready single-group P0 smoke script
  for the retained `scene_00180` showcase. It records the Git SHA/GPU/log and
  refuses to overwrite an existing output directory.

Files changed/added for this update:

```text
export_stage3_scene_plane_fusion.py
tests/test_stage3_scene_plane_fusion_mapping.py
run_plane_feedback_p0.sh
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile export_stage3_scene_plane_fusion.py tests/test_stage3_scene_plane_fusion_mapping.py
python tests/test_stage3_scene_plane_fusion_mapping.py
PYTHONPATH=. python tests/test_plane_regularized_alignment.py
<Git-for-Windows-bash> -n run_plane_feedback_p0.sh
git diff --check -- export_stage3_scene_plane_fusion.py tests/test_stage3_scene_plane_fusion_mapping.py
```

Results:

```text
py_compile: passed
Stage3 mapping/path-remap tests: 3 passed
plane-feedback tests: 3 passed
P0 shell syntax check: passed
diff check: passed (line-ending conversion warnings only)
```

Execution status:

* A local P0 launch was prepared after confirming all five local RGB files and
  the DUSt3R checkpoint exist, but it was stopped before model execution when
  the run target was changed to the server.
* No P0 acceptance result, metric or new diagnostic visualization is reported.

Next exact step on the server:

```bash
git pull
bash run_plane_feedback_p0.sh
```

If the default output directory already exists, set a new `OUT_DIR`; do not
delete or overwrite the old run.

## 25. Session update: 2026-07-14 P0 `roma` dependency guard

Branch: `codex/bounded-support-head`

Base commit SHA: `74b2a14`

Observed server result:

* The first P0 launch stopped during DUSt3R module import with
  `ModuleNotFoundError: No module named 'roma'`.
* `roma` is an upstream dependency listed in the vendored
  `dust3r/requirements.txt`; the failure occurred before global alignment or
  plane feedback ran.
* Therefore there is still no P0 acceptance decision, metric or diagnostic
  visualization to report.

Implemented behavior:

* `run_plane_feedback_p0.sh` now checks whether the selected server Python can
  import `roma` and installs the missing package before creating any run output.
* The installed `roma` version is printed to the console and retained in the
  run log for reproducibility.
* The default output directory advances from `v1` to `v2`, preserving the
  failed `v1` directory rather than deleting or overwriting it.

Files changed:

```text
run_plane_feedback_p0.sh
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
<Git-for-Windows-bash> -n run_plane_feedback_p0.sh
git diff --check -- run_plane_feedback_p0.sh docs/codex_handoff/CURRENT_STATE.md
```

Results:

```text
P0 shell syntax check: passed
diff check: passed (line-ending conversion warning only)
No model run, metric or visualization was produced locally.
```

Plan status:

* Work remains strictly at P0, the existing-v1 single-scene behavior smoke.
* P1 identical-cache quantitative baselines and point-aligned GT have not
  started.
* P2 leave-one-view-out factor auditing and P3 evidence gating have not
  started; no innovation claim is made from this dependency repair.

Next exact step on the server:

```bash
cd /gemini/code/LightRecon3D
git pull --ff-only origin codex/bounded-support-head
bash run_plane_feedback_p0.sh
```

Retain the generated log, acceptance record, before/after PLY files and
displacement visualization before deciding whether P0 passes.

## 26. Session update: 2026-07-14 P0 `trimesh` export dependency

Branch: `codex/bounded-support-head`

Base commit SHA: `fae720a`

Observed server result:

* The `v2` P0 launch reached `write_dust3r_textured_glb` and then failed with
  `RuntimeError: trimesh is required to export DUSt3R-style textured GLB`.
* `trimesh==4.9.0` is pinned in the repository root `requirements.txt` and is
  also listed by the vendored DUSt3R requirements.
* The exporter writes plane-feedback diagnostics, the main result NPZ and the
  main PLY before attempting the GLB. Therefore the failed `v2` directory may
  contain valid partial P0 evidence, but this has not been inspected and the
  run did not write its final HTML/parameter files/manifest.

Implemented behavior:

* The P0 server script now preflights both `roma==1.5.6` and
  `trimesh==4.9.0`, installing a package when it is missing or its installed
  distribution version differs from the repository pin.
* Both resolved versions are retained in `run.log`.
* The default output advances to `v3`; neither failed `v1` nor partial `v2` is
  deleted or overwritten.

Files changed:

```text
run_plane_feedback_p0.sh
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
<Git-for-Windows-bash> -n run_plane_feedback_p0.sh
git diff --check -- run_plane_feedback_p0.sh docs/codex_handoff/CURRENT_STATE.md
```

Results:

```text
P0 shell syntax check: passed
diff check: passed (line-ending conversion warning only)
No server output, metric or visualization was inspected locally.
```

Plan status:

* The work remains at P0. Reaching the export section is useful behavioral
  evidence but is not a completed P0 result and is not an innovation claim.
* P1-P5 remain pending in the order recorded in
  `docs/codex_tasks/evidence_gated_plane_feedback.md`.

Next exact step on the server:

```bash
cd /gemini/code/LightRecon3D
git pull --ff-only origin codex/bounded-support-head
bash run_plane_feedback_p0.sh
```

After completion, inspect the `v3` manifest and plane-feedback block. Preserve
the partial `v2` directory as failure evidence.

## 27. Session update: 2026-07-14 completed P0 rollback and proposed-state diagnostics

Branch: `codex/bounded-support-head`

Server execution commit SHA: `fae720a3e92eb2c493b861d660d9b2aa632e098e`

Completed server output:

```text
/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3
```

Observed P0 result:

```text
accepted                         = false
observations                     = 17310
active planes                    = 4
base loss before                 = 0.0076693739
base loss proposed               = 0.0080806026
base loss acceptance limit       = 0.0078994651
plane residual mean before       = 0.0058277562
plane residual mean proposed     = 0.0005458354
relative plane improvement       = 0.9063387
support displacement mean/p95    = 0.00748175 / 0.03111751
feedback runtime seconds         = 2.1792
```

Interpretation:

* The proposed feedback update reduced its plane residual by about 90.63%, but
  increased the DUSt3R base objective by about 5.36%, exceeding the configured
  3% tolerance.
* The whole update was correctly rejected and the scene parameters were
  restored. Committed support/full-cloud/non-support displacement is therefore
  zero.
* This is a successful rollback behavior check, not evidence of geometry
  improvement or novelty. It motivates auditing individual harmful factors.

Retained artifacts include the original global-cloud cache, result NPZ,
manifest, HTML, GLB, plane parameter files, textured plane PLY, and the
before/after/displacement PLY diagnostics. The directory totals about 69 MiB.

Diagnostic gap found after the run:

* The optimizer restored the differentiable scene before returning to Stage3.
  Consequently the existing `after.ply` and displacement heatmap correctly
  visualize the committed rollback state, but do not visualize the rejected
  candidate that caused the 5.36% base-loss increase.
* The rejected candidate pointmaps were not retained, preventing full-cloud and
  non-support candidate-motion inspection.

Implemented behavior after the P0 run:

* The optimizer now snapshots all proposed globally aligned pointmaps before a
  possible rollback.
* Stage3 maps those pointmaps through explicit `(alignment_view_index, x, y)`
  provenance and writes separate `proposed.ply` and
  `proposed_displacement_heatmap.ply` files.
* The manifest/NPZ now record proposed diagnostic paths plus proposed
  full-cloud and non-support displacement summaries, while the existing
  `after` fields continue to represent the committed state.
* The P0 server script advances its default output to `v4`, preserving `v3`.

Files changed:

```text
plane_regularized_alignment.py
export_stage3_scene_plane_fusion.py
tests/test_plane_regularized_alignment.py
tests/test_stage3_scene_plane_fusion_mapping.py
run_plane_feedback_p0.sh
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile plane_regularized_alignment.py export_stage3_scene_plane_fusion.py tests/test_plane_regularized_alignment.py tests/test_stage3_scene_plane_fusion_mapping.py
PYTHONPATH=. python tests/test_plane_regularized_alignment.py
python tests/test_stage3_scene_plane_fusion_mapping.py
<Git-for-Windows-bash> -n run_plane_feedback_p0.sh
git diff --check -- <six changed implementation/test/handoff files>
```

Results:

```text
py_compile: passed
plane-feedback tests: 3 passed
Stage3 mapping/provenance tests: 4 passed
P0 shell syntax check: passed
diff check: passed (line-ending conversion warnings only)
```

An initial `python -m unittest tests...` invocation resolved the environment's
installed Ultralytics `tests` package and failed while reading its settings.
Running the repository test files directly avoided that unrelated namespace
collision and produced the passing results above.

Unresolved:

* The new proposed-state diagnostic has not yet been run on the server.
* The exact proposed full-cloud/non-support displacement for `scene_00180` is
  unknown until the `v4` rerun.
* GitHub access from the server is intermittent; no direct SSH control path is
  available to Codex.
* P1 point-aligned GT and identical-cache quantitative baselines remain pending.

Next exact step:

Pull the diagnostic commit when server GitHub access permits and rerun P0 into
the new `v4` directory. Archive the proposed full-cloud/non-support movement and
proposed heatmap, then mark P0 complete and begin P1.

## 28. Session update: 2026-07-14 P1 point-aligned plane GT scaffold

Branch: `codex/bounded-support-head`

Base commit SHA: `72cbed2`

Data-flow finding:

* Each Structured3D view provides a `layout.json` with visible plane polygons
  and a `plane.ID`. For the retained five-view `scene_00180` group, IDs such as
  1, 19, 27, 35, 45 and 71 recur across views, while their stored normals and
  offsets change with camera coordinates.
* Therefore the point-aligned writer uses `plane.ID` only for cross-view
  identity and visible support. It does not compare camera-frame layout plane
  equations directly with DUSt3R global planes.
* The GT plane equations are refitted from identically indexed DUSt3R cache
  points after labeling. This evaluates support/identity in the common cache
  frame; it is not absolute metric 3D ground truth.

Implemented behavior:

* Added `build_structured3d_point_aligned_gt.py`.
* It reads the method-independent global cache, validates explicit `xy` DUSt3R
  pointmap coordinates and the saved view registry, derives each matching
  `layout.json`, and reproduces vendored DUSt3R resize/crop geometry.
* Visible layout polygons are rasterized with their original Structured3D
  plane IDs. A one-pixel boundary band is ignored by default to avoid counting
  rasterization-edge ambiguity as an identity error.
* Labels are gathered in the exact filtered cache order. Source cache indices,
  view indices, pointmap pixels, source plane IDs, transforms, layout paths and
  the SHA-256 of the source cache are retained.
* Added a shared identical-cache finite/confidence filter to
  `global_plane_baselines.py`; the GT writer and RANSAC now cannot silently use
  different point subsets for the same `min_conf`.
* Added `run_plane_feedback_p1_gt.sh`, targeting the retained original P0 `v3`
  global cache and refusing to overwrite its own output directory.

Files added/changed:

```text
build_structured3d_point_aligned_gt.py
global_plane_baselines.py
tests/test_structured3d_point_aligned_gt.py
tests/test_global_plane_baselines.py
run_plane_feedback_p1_gt.sh
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile global_plane_baselines.py build_structured3d_point_aligned_gt.py tests/test_global_plane_baselines.py tests/test_structured3d_point_aligned_gt.py
PYTHONPATH=. python tests/test_global_plane_baselines.py
PYTHONPATH=. python tests/test_structured3d_point_aligned_gt.py
<Git-for-Windows-bash> -n run_plane_feedback_p1_gt.sh
git diff --check -- <P1 implementation/test/script/task/handoff files>
```

Results:

```text
global plane baseline tests: 4 passed
point-aligned Structured3D GT tests: 2 passed
P1 shell syntax check: passed
diff check: passed (line-ending conversion warnings only)
```

The first GT test run exposed an unclosed `np.load` handle during Windows
temporary-directory cleanup. The test now uses a context manager; the repeated
test run passed.

Unresolved:

* No real server GT NPZ/PLY has been generated yet.
* Point-aligned layout labels alone do not provide pose, metric full-cloud,
  non-planar geometry or Chamfer GT. Structured3D depth/camera-pose lifting is
  a separate required P1 component.
* Stage1/direct-SVD/manual-merge/PlaneGraph-BA/live-feedback predictions still
  need adapters that emit assignments on the identical filtered full cache.

Next exact server step after the branch is pulled:

```bash
bash run_plane_feedback_p1_gt.sh
```

The required acceptance checks are: one cache checksum, five registered views,
nonzero labeled points, stable source plane IDs across views, and a visually
sensible colored GT PLY before running comparative baselines.

## 29. Session update: 2026-07-14 real P1 GT and scalable RANSAC preparation

Branch: `codex/bounded-support-head`

Server execution commit SHA: `e444c8a2f730254c22a754e084279b01e53fede4`

Completed server output:

```text
/gemini/data-1/lightrecon_runs/plane_feedback_p1_gt_scene00180_20260714_v1
```

Observed point-aligned GT result:

```text
source cache SHA-256 = 77b745a52bf28d170977f9ffd14da79c11df5e7940c886b4f36e69f0daf32101
filtered points        = 715848
labeled points         = 693839
labeled coverage       = 96.93%
planes                 = 7
source plane IDs       = [0, 1, 19, 27, 29, 35, 45]
registered views       = 5
min confidence         = 1.0
boundary ignore radius = 1 pixel
```

The real GT NPZ, colored PLY, manifest and run log were generated. This passes
the point-aligned plane-label acceptance check. It still does not provide
absolute pose or metric depth/full-cloud GT.

Implemented next P1 baseline:

* Sequential RANSAC now scores hypotheses on at most a fixed deterministic
  subset (`hypothesis_max_points`) and validates/refits the winning hypothesis
  against every remaining full-cache point.
* Large inlier sets use deterministic occupied-voxel connectivity; small sets
  retain the exact radius-connected implementation. The threshold and mode are
  stored in the method configuration.
* Added a server RANSAC entrypoint that runs both an oracle GT self-evaluation
  sanity row and the full-cache RANSAC row against the same GT NPZ.
* Replaced the optional SciPy/greedy assignment path with an exact pure-NumPy
  rectangular Hungarian matcher, preventing environment ABI issues or greedy
  plane-match errors from changing evaluation metrics.

Files changed/added:

```text
global_plane_baselines.py
evaluate_global_plane_baselines.py
tests/test_global_plane_baselines.py
tests/test_evaluate_global_plane_baselines.py
run_plane_feedback_p1_ransac.sh
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Commands/tests completed:

```text
python -m py_compile global_plane_baselines.py evaluate_global_plane_baselines.py tests/test_global_plane_baselines.py tests/test_evaluate_global_plane_baselines.py
PYTHONPATH=. python tests/test_global_plane_baselines.py
PYTHONPATH=. python tests/test_evaluate_global_plane_baselines.py
<Git-for-Windows-bash> -n run_plane_feedback_p1_ransac.sh
git diff --check -- <RANSAC/evaluator/test/script files>
```

Results:

```text
global plane baseline tests: 5 passed
global plane evaluation tests: 3 passed
P1 RANSAC shell syntax check: passed
diff check: passed (line-ending conversion warnings only)
```

Next exact server step after pulling the next commit:

```bash
bash run_plane_feedback_p1_ransac.sh
```

Archive the RANSAC method manifest, CSV/JSON metrics, PLY and runtime before
building the Stage1 support adapters. The P0 `v4` proposed-state rerun remains a
useful diagnostic but no longer blocks the completed P0 behavior verdict.

## 30. Session update: 2026-07-14 real P1 RANSAC and support-label adapter

Branch: `codex/bounded-support-head`

Base commit SHA: `8aa80e8`

Completed server output:

```text
/gemini/data-1/lightrecon_runs/plane_feedback_p1_ransac_scene00180_20260714_v1
```

Observed identical-cache RANSAC result:

```text
runtime seconds            = 18.7362
assigned points            = 715695 / 715848
predicted / GT planes      = 5 / 7
true-positive planes       = 3
plane precision / recall   = 0.6000 / 0.4286
matched support IoU        = 0.71314
GT-support coverage        = 0.999779
normal angular error       = 5.67694 degrees
point-to-plane residual    = 0.00643551
fragmentation excess       = 1
over-merge excess          = 2
```

The GT self-evaluation sanity row returned seven of seven planes, unit
precision/recall/IoU/coverage and zero angular error. RANSAC's almost complete
support coverage together with low plane recall and two over-merges shows that
fitting almost every point is not sufficient to recover bounded plane
identity. This is one retained scene, so it does not establish a general
advantage or a novelty claim.

Implemented next P1 comparison:

* Added `lift_support_prediction_to_global_cache.py`. It maps the existing
  Stage2 plus manual-merge prediction to the exact filtered global cache using
  only `(alignment_view_index, x, y)` provenance.
* The adapter validates the stored `xy` order and
  `dust3r_aligned_pointmap` coordinate space. It rejects a non-unique cache
  registry, collapses duplicate records only when their labels agree, drops
  and counts conflicting keys by default, and counts unmatched keys.
* Active source labels are deterministically remapped to output plane IDs and
  refitted on their identically indexed global-cache points. Cache/support
  SHA-256 values, source-to-output ID mapping and all join diagnostics are
  written to the manifest.
* Added a server script that preflights `roma`, refuses to overwrite outputs,
  builds the support baseline and evaluates GT self, RANSAC and support rows in
  one CSV/JSON table.

Files added/changed:

```text
lift_support_prediction_to_global_cache.py
tests/test_lift_support_prediction_to_global_cache.py
run_plane_feedback_p1_support_baseline.sh
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Focused validation completed:

```text
python -m py_compile lift_support_prediction_to_global_cache.py tests/test_lift_support_prediction_to_global_cache.py
PYTHONPATH=. python tests/test_lift_support_prediction_to_global_cache.py
<Git-for-Windows-bash> -n run_plane_feedback_p1_support_baseline.sh
```

Results:

```text
py_compile: passed
exact provenance-join tests: 6 passed
P1 support shell syntax check: passed
```

The synthetic tests cover shuffled exact joins, agreeing duplicates,
conflicting duplicates, unmatched keys, no positive support and rejection of a
duplicate cache registry. A real support result has not yet been run, so its
plane metrics and duplicate/conflict counts remain unknown.

Next exact server step after pulling the adapter commit:

```bash
bash run_plane_feedback_p1_support_baseline.sh
```

Archive `support_baseline_manifest.json` and
`scene00180_plane_metrics.json`. The result determines whether the current
manual support identity beats full-cloud RANSAC on this scene; it does not yet
complete the remaining P1 methods or authorize P2/P3.

## 31. Session update: 2026-07-14 real support lift and sparse-metric correction

Branch: `codex/bounded-support-head`

Server execution commit SHA: `5a3590a`

Completed server output:

```text
/gemini/data-1/lightrecon_runs/plane_feedback_p1_support_scene00180_20260714_v1
```

Observed exact-join result:

```text
method                         = stage2_manual_merge_support
runtime seconds                = 0.247642
output planes                  = 6
cache points                   = 715848
support records                = 80000
positive support records       = 79831
unique positive (view,x,y)     = 19985
duplicate positive records     = 59846
conflicting unique keys        = 3431 (17.2% of unique positive keys)
conflicting records            = 13719
resolved keys                  = 16554
matched cache keys             = 15972
unmatched resolved keys        = 582
assigned output points         = 15972
```

The cache SHA-256 remained
`77b745a52bf28d170977f9ffd14da79c11df5e7940c886b4f36e69f0daf32101`.
The high repeated-observation count is expected from overlapping pairs, but
the 3,431 disagreeing keys are direct evidence that the current pair/manual
identity is not self-consistent at every repeated pixel. They were dropped,
not guessed or majority-voted.

Metric interpretation correction:

* The first evaluation gave the sparse support prediction zero plane matches
  and 2.2401% GT-support coverage. The zero match count is not sufficient as an
  identity verdict because its IoU compares only 15,972 predicted support
  samples with 693,839 dense GT points. Achievable IoU is strongly coupled to
  the unknown per-plane sampling density, which the aggregate row does not
  expose.
* The 2.2401% value remains useful as full-cache sample coverage, but it must
  not be compared as if the sparse support proposal were a dense segmentation.
* Identity quality must instead be measured on the emitted support domain,
  while always reporting dense coverage and the fraction of assigned samples
  that have a GT label to prevent a trivially tiny prediction from appearing
  successful.

Implemented evaluator correction:

* `evaluate_global_plane_baselines.py` now preserves every original dense-GT
  metric and adds a support-conditioned plane matching block.
* New metrics include eligible observed GT planes, conditioned IoU/precision/
  observed recall/all-GT recall, normal error, fragmentation/over-merge,
  pairwise identity precision/recall/F1, weighted predicted-cluster purity, GT
  completeness, assigned point count and assigned GT-label rate.
* `--min_observed_plane_points` defaults to 64 in the CLI so a plane cannot be
  counted from a few accidental samples.
* Added `run_plane_feedback_p1_support_audit.sh`. It uses the existing GT,
  RANSAC and support NPZ files, records their SHA-256 values, refuses to
  overwrite output and performs only a fast re-evaluation; it does not rerun
  DUSt3R or rebuild any cache.

Files changed/added:

```text
evaluate_global_plane_baselines.py
tests/test_evaluate_global_plane_baselines.py
run_plane_feedback_p1_support_audit.sh
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Focused validation completed:

```text
python -m py_compile evaluate_global_plane_baselines.py tests/test_evaluate_global_plane_baselines.py
PYTHONPATH=. python tests/test_evaluate_global_plane_baselines.py
<Git-for-Windows-bash> -n run_plane_feedback_p1_support_audit.sh
```

Results:

```text
py_compile: passed
global plane evaluation tests: 5 passed
P1 support-audit shell syntax check: passed
```

The two new tests prove that a correct 4/100-point sparse partition retains
4% dense coverage and zero dense-IoU matches while receiving perfect
conditioned identity scores, and that an explicit two-GT-plane over-merge
reduces pairwise precision/F1.

Unresolved:

* The real support-conditioned metrics have not yet been produced.
* Sparse-support partition quality and dense bounded-support completeness are
  different quantities. A full per-view mask/boundary evaluation is still
  required before claiming bounded-support quality.
* This is still P1. Direct global SVD, PlaneGraph-BA, live feedback and oracle
  identity/factor upper bounds remain incomplete; P2/P3 are not authorized.

Next exact server step after pulling the evaluator commit:

```bash
bash run_plane_feedback_p1_support_audit.sh
```

Archive `scene00180_support_conditioned_metrics.json`. This reuses existing
NPZ files and should complete in seconds.

## 32. Session update: 2026-07-14 real conditioned metrics and direct-SVD audit

Branch: `codex/bounded-support-head`

Server execution commit SHA: `1c73908`

Completed server output:

```text
/gemini/data-1/lightrecon_runs/plane_feedback_p1_support_audit_scene00180_20260714_v1
```

Observed support-conditioned comparison:

```text
metric                              RANSAC       manual support
assigned GT-label rate              0.96925      0.97314
observed GT planes                  7            6
conditioned true-positive planes    3            3
conditioned plane precision         0.6000       0.5000
conditioned all-GT recall           0.4286       0.4286
conditioned matched IoU             0.71360      0.96067
conditioned normal error (degrees)  5.67694      0.61006
conditioned fragmentation excess    1            2
conditioned over-merge excess       2            1
pairwise identity precision         0.60885      0.84794
pairwise identity recall            0.83527      0.79215
pairwise identity F1                0.70431      0.81909
predicted-cluster purity            0.75611      0.89687
GT completeness                     0.84516      0.85505
purity/completeness F1              0.79816      0.87546
```

Interpretation:

* On the emitted support domain, the current Stage2/manual pipeline preserves
  plane identity substantially better than full-cloud RANSAC on this one
  scene: pairwise F1 improves by 0.115 and weighted cluster purity by 0.141.
* The correctly matched manual planes are geometrically strong: conditioned
  IoU is 0.961 and normal error is 0.61 degrees.
* This is not a complete success. Manual support misses one GT plane, only
  three of six observed identities pass the 0.5 conditioned-IoU match, and it
  shows two fragmentation excesses plus one over-merge.
* It is also not yet a clean manual-vs-RANSAC claim. The unique-cache adapter
  dropped 3,431 repeated keys with disagreeing manual labels before producing
  these metrics. That safety choice was correct for a single-label cache, but
  it removes exactly the contradictions needed to audit association quality
  and can make manual identity look better than its raw observations.

Implemented next P1 ablation:

* Stage3 now accepts `--global_cloud_cache` in `dust3r_global` mode. It
  reconstructs every cached per-view pointmap through explicit `(view,x,y)`,
  validates complete one-to-one pixel coverage, scene key and image registry,
  and skips model loading/global alignment. Plane feedback is explicitly
  forbidden on this frozen-cache path because no differentiable live scene
  exists.
* A cached `merge_mode=none` run now provides the missing
  `stage2_support_direct_global_svd` baseline on the exact original P0 cache.
  The method and merge mode are stored in the output NPZ.
* Added `evaluate_support_record_partitions.py`. It preserves all repeated
  support records rather than collapsing pixels, maps GT and full-cache RANSAC
  labels by the exact registry key, requires direct/manual record order to be
  identical, reports conflict counts and writes a per-predicted-plane GT
  contingency audit including views, unique keys, dominant identity, purity
  and normal error.
* The old RANSAC NPZ predates explicit coordinate-convention fields. Its use is
  guarded by a required `--allow_legacy_cache_xy` command-line override, which
  is recorded in the audit JSON. The known writer provenance is
  `global_plane_baselines.py`; that writer now stores explicit `xy` and
  `dust3r_aligned_pointmap` fields for all future outputs.
* Added `run_plane_feedback_p1_direct_svd.sh`, which preflights pinned `roma`
  and `trimesh`, refuses overwrites, verifies SHA-256 inputs, runs cached direct
  SVD and evaluates GT, RANSAC, manual and direct-SVD identities on the same
  repeated support records. It does not rerun DUSt3R.

Files changed/added:

```text
export_stage3_scene_plane_fusion.py
evaluate_support_record_partitions.py
global_plane_baselines.py
tests/test_stage3_scene_plane_fusion_mapping.py
tests/test_support_record_partitions.py
run_plane_feedback_p1_direct_svd.sh
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Focused validation completed:

```text
python -m py_compile export_stage3_scene_plane_fusion.py evaluate_support_record_partitions.py evaluate_global_plane_baselines.py global_plane_baselines.py tests/test_stage3_scene_plane_fusion_mapping.py tests/test_support_record_partitions.py
PYTHONPATH=. python tests/test_stage3_scene_plane_fusion_mapping.py
PYTHONPATH=. python tests/test_support_record_partitions.py
PYTHONPATH=. python tests/test_evaluate_global_plane_baselines.py
PYTHONPATH=. python tests/test_lift_support_prediction_to_global_cache.py
PYTHONPATH=. python tests/test_global_plane_baselines.py
<Git-for-Windows-bash> -n run_plane_feedback_p1_direct_svd.sh
```

Results:

```text
Stage3 mapping/cache tests: 5 passed
support-record partition tests: 5 passed
global plane evaluator tests: 5 passed
exact support-lift tests: 6 passed
global plane baseline tests: 5 passed
direct-SVD shell syntax check: passed
```

The record-audit tests include an end-to-end synthetic CLI run that writes and
reloads JSON/CSV while preserving repeated keys, in addition to exact cache
label expansion, conflict counting, per-plane dominant-GT diagnostics and the
explicit legacy-coordinate override guard.

Unresolved:

* The direct-global-SVD and repeated-record server metrics have not yet been
  produced.
* Current evidence is one scene and does not establish robustness, geometry
  improvement or novelty.
* P1 PlaneGraph-BA, live-feedback and oracle-identity/factor upper bounds still
  remain before P2.

Next exact server step after pulling the implementation commit:

```bash
bash run_plane_feedback_p1_direct_svd.sh
```

Archive `scene00180_support_record_partition_audit.json` and the Stage3
manifest. The manual-vs-direct gap will show whether the current merge adds
useful cross-candidate identity or mainly hides/creates association errors.

## 33. Session update: 2026-07-15 repeated-record result and metric P1 gate

Branch: `codex/bounded-support-head`

Starting commit SHA: `dc6b225`

Archived repeated-record result:

* All four methods were evaluated on the same 80,000 Stage2 observation
  records. GT matched 77,672 registry records and labeled 75,248 of them. Six
  of the seven scene GT identities are observed on this support domain.
* Direct global SVD with no cross-candidate merge emitted 63 identities. It
  assigned 79,831 records, but 19,974/19,985 unique positive keys and
  79,820/79,831 positive records had conflicting candidate IDs. Pairwise
  precision/recall/F1 were 0.8425/0.0644/0.1196; purity/completeness/F1 were
  0.8843/0.0934/0.1689.
* Manual merge on the identical records emitted 11 identities. Its pairwise
  precision/recall/F1 were 0.8505/0.7022/0.7693 and
  purity/completeness/F1 were 0.8843/0.8103/0.8457. It therefore removes 52
  candidate fragments and recovers most within-GT pairs without reducing
  purity. Cross-candidate identity aggregation is genuinely useful on this
  scene.
* Manual merge is not solved: only three of six observed GT planes match at
  conditioned IoU 0.5; precision is 3/11, observed recall is 3/6, and the
  output has two fragmentation excesses plus one over-merge. RANSAC has lower
  pairwise F1 (0.7044) and purity (0.7562), but higher recall/completeness
  (0.8354/0.8455) and only five groups. The manual method is purer but more
  fragmented.
* This is single-scene support-identity evidence. It does not demonstrate
  global geometry improvement, pose improvement, generalization or novelty.

Implemented metric P1 stop/go infrastructure:

* `build_structured3d_point_aligned_gt.py` schema v2 now preserves the old
  DUSt3R-frame identity GT and additionally reconstructs metric structural
  points. For every exact `(alignment_view_index,x,y)`, it inverts the
  vendored DUSt3R PIL resize/integer crop with a recorded half-pixel mapping,
  forms a ray from Structured3D half-FOV calibration, intersects the matching
  camera-frame layout plane, and transforms the point to Structured3D world
  coordinates in metres.
* The Structured3D perspective camera convention is explicitly handled as a
  left-handed camera frame (`+x` right, `+y` up, `+z` view direction) mapped to
  the right-handed world frame. Plane equations use `n dot X + d = 0`.
* The writer stores metric camera/world points, validity, world plane
  parameters, camera bases/FOVs, annotation provenance, plane-consistency
  diagnostics and separate DUSt3R-frame and metric-world PLYs.
* `evaluate_structured3d_metric_geometry.py` refuses nearest-XYZ joins. It
  joins predictions by exact registry keys, verifies that the reference NPZ
  SHA-256 equals the GT source-cache checksum, estimates one recorded global
  Sim(3), and reports correspondence error plus GT-plane residual globally and
  per view. Both independent gauge removal and the original reference gauge
  are reported.
* `planegraph_ba.py` now accepts the point-aligned GT's canonical
  `view_indices + pixel_xy` provenance, enabling a dense oracle-identity
  PlaneGraph-BA upper bound without changing the cache.
* `run_plane_feedback_p1_metric_oracle.sh` preflights pinned `roma==1.5.6`,
  refuses overwrite, checks inputs/outputs, builds metric GT, runs manual and
  oracle-identity PlaneGraph-BA, evaluates both against original DUSt3R, and
  writes before/after PLYs and a metrics JSON.

Files changed/added:

```text
build_structured3d_point_aligned_gt.py
evaluate_structured3d_metric_geometry.py
planegraph_ba.py
run_plane_feedback_p1_metric_oracle.sh
tests/test_structured3d_point_aligned_gt.py
tests/test_evaluate_structured3d_metric_geometry.py
tests/test_planegraph_ba.py
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Validation completed:

```text
py_compile: passed
metric GT tests: 3 passed
metric evaluator tests: 2 passed
PlaneGraph-BA tests: 4 passed
Stage3 mapping/cache tests: 5 passed
support-record partition tests: 5 passed
global plane evaluator tests: 5 passed
exact support-lift tests: 6 passed
global plane baseline tests: 5 passed
metric/oracle shell syntax: passed
```

Real-coordinate diagnostic:

```text
scene: scene_00180 / space 445895 / views 0..4
layout-to-world plane comparisons: 34
maximum normal error: 0.0 degrees
maximum offset error: 2.0374e-5 mm
```

Limitations and stop/go rule:

* Metric GT currently covers structural layout planes. It does not measure
  furniture or other non-planar geometry without Structured3D `depth.png`.
* The retained global cache lacks optimized DUSt3R camera poses, so this audit
  reports metric structural point error rather than direct camera-pose error.
* Do not enter P2 until the manual and oracle upper-bound results are read. If
  oracle identity does not improve gauge-invariant metric correspondence and
  GT-plane error, revise or stop this feedback direction even if its internal
  plane residual falls.

Next exact server step after pulling the implementation commit:

```bash
bash run_plane_feedback_p1_metric_oracle.sh
```

Archive these outputs without overwriting them:

```text
scene00180_metric_structural_gt_manifest.json
scene00180_metric_geometry_metrics.json
manual_ba/*_summary.json
oracle_ba/*_summary.json
run.log
```

## 34. Session update: 2026-07-15 metric/oracle result and final P1 rollback gate

Branch: `codex/bounded-support-head`

Server commit SHA: `b973258fa419dbc582dcaf1227ab6ee013f78087`

Completed server run:

```text
/gemini/data-1/lightrecon_runs/plane_feedback_p1_metric_oracle_scene00180_20260715_v1
```

Input integrity and GT:

* Global cache SHA-256 remained
  `77b745a52bf28d170977f9ffd14da79c11df5e7940c886b4f36e69f0daf32101`.
* Manual support SHA-256 remained
  `39d963d24a2eb4a8b951a942c4ffafa23f1235f2a87547236c4ac8a8baaf4cb5`.
* Metric GT contains 715,848 filtered cache points, 693,839 valid/labeled
  structural points, seven plane identities and five views.
* Layout-to-world consistency remained exact within numerical precision:
  maximum normal error 0 degrees and maximum offset error 2.0374e-5 mm.

PlaneGraph-BA behavior:

* Manual support mapped 77,503/79,831 records to 24,759 unique
  `(cache point, manual label)` observations. Eleven planes were present but
  only four were multiview-active. Runtime was 0.537 seconds. Its internal
  overall mean residual fell from 0.0105015 to 0.0103899.
* Dense oracle identity mapped all 693,839 observations. Seven planes were
  present and six were multiview-active. Runtime was 5.743 seconds. Its
  internal mean residual fell from 0.0054363 to 0.0053121, although the median
  residual increased from 0.0009017 to 0.0012978.

Metric stop/go result after a global Sim(3):

```text
method                       correspondence RMSE   mean GT-plane residual
original DUSt3R              0.2644067 m           0.0959640 m
manual PlaneGraph-BA         0.2650402 m           0.0949075 m
oracle-identity PlaneGraph   0.2666520 m           0.0922639 m
```

Interpretation:

* Manual BA improved the target plane residual by 1.06 mm (1.10%) but worsened
  structural correspondence RMSE by 0.63 mm (0.24%).
* Even dense GT identity improved plane residual by 3.70 mm (3.86%) while
  worsening correspondence RMSE by 2.25 mm (0.85%).
* Fixed all-plane PlaneGraph-BA v0 therefore fails the P1 geometry gate. The
  result is the same failure mode as live feedback v1: the structural loss can
  improve while real point correspondence degrades. Correct plane association
  does not by itself make every plane factor geometrically helpful.
* Do not present PlaneGraph-BA v0 as a successful method and do not start the
  full P2 audit from this aggregate result alone.

Implemented final cheap P1 discriminator:

* `evaluate_structured3d_metric_geometry.py` schema v2 now records fixed-gauge
  per-view deltas against original DUSt3R.
* It also computes two explicitly GT-supervised upper bounds from the already
  generated final corrections: a correspondence oracle that keeps a corrected
  view only when its RMSE improves, and a joint-Pareto oracle that keeps it
  only when both correspondence RMSE and mean GT-plane residual improve.
* The oracle switch uses `alignment_view_index` as its unit and requires exact
  identical GT key coverage. It never performs a nearest-neighbour join.
* Added `run_plane_feedback_p1_metric_gate_recheck.sh`. It checks/installs
  `roma==1.5.6`, reuses the existing metric GT and BA NPZs, refuses to
  overwrite outputs, reruns only the evaluator and prints every per-view
  decision plus aggregate oracle metrics.

Files changed/added:

```text
evaluate_structured3d_metric_geometry.py
tests/test_evaluate_structured3d_metric_geometry.py
run_plane_feedback_p1_metric_gate_recheck.sh
docs/codex_tasks/evidence_gated_plane_feedback.md
docs/codex_handoff/CURRENT_STATE.md
```

Validation completed:

```text
metric evaluator tests: 3 passed
metric GT tests: 3 passed
PlaneGraph-BA tests: 4 passed
Stage3 mapping/cache tests: 5 passed
support-record partition tests: 5 passed
global plane evaluator tests: 5 passed
exact support-lift tests: 6 passed
global plane baseline tests: 5 passed
py_compile: passed
metric gate shell syntax: passed
```

Final P1 stop/go rule:

* If the joint-Pareto oracle selects no useful views, or its mixed result does
  not improve both aggregate correspondence RMSE and GT-plane residual, stop
  the plane-feedback main line and pivot to cross-scene bounded-plane identity
  aggregation/output quality.
* If the joint-Pareto oracle has a material aggregate joint gain, then and only
  then proceed to P2 and ask whether held-out non-GT evidence predicts its
  keep/rollback labels.

Next exact server step after pulling the evaluator commit:

```bash
bash run_plane_feedback_p1_metric_gate_recheck.sh
```

This command does not rerun DUSt3R, rebuild GT or rerun either BA optimization.

## 2026-07-15 final plane-feedback gate and research pivot

The evidence-gated plane-feedback main line is now stopped. The final P1
server run used commit `1b01dc8`, the same 715,848-point scene-00180 cache and
693,839 metric layout correspondences, and evaluated the already computed
per-view corrections under the original fixed global Sim(3).

Manual-support PlaneGraph-BA improved aggregate structural correspondence RMSE
by only 0.250 mm while worsening GT-plane mean residual by 3.335 mm. None of
the five views improved both metrics. A correspondence-only oracle selected
views 1, 2 and 4 and improved RMSE by 1.743 mm, but worsened plane error by
3.914 mm; the joint-Pareto oracle selected no views.

GT-identity PlaneGraph-BA worsened aggregate correspondence RMSE by 4.366 mm
and plane residual by 5.708 mm under the fixed gauge. Again, none of five views
jointly improved both metrics. Its correspondence-only oracle selected views
1, 3 and 4, improving RMSE by 0.811 mm while worsening plane error by 6.374 mm;
the joint oracle selected no views.

Decision: do not run P2-P5 and do not tune PlaneGraph-BA or the live plane loss
as the main solution. This is an oracle failure showing an objective conflict,
not merely a weak association or acceptance threshold.

The only retained research signal is cross-candidate identity aggregation on
frozen geometry. On the same 80,000 provenance records, manual aggregation
reduced 63 direct-SVD fragments to 11 and raised pairwise identity F1 from
0.120 to 0.769, above global RANSAC at 0.704. This is still single-scene,
manual-rule evidence.

The next task is `docs/codex_tasks/bounded_plane_identity_pivot.md`. Its first
gate is a multi-scene identical-cache audit; no new neural model is authorized
until the identity gain is cross-scene, conflict-preserving and better than
RANSAC by the pre-registered margin. No pointmap recache or long server run has
been performed for this pivot.

## 2026-07-15 research-practice completion sprint and structural lines

The project scope is now frozen around completing the approved undergraduate
research-practice deliverables by 2026-07-31. The active task is
`docs/codex_tasks/research_practice_completion.md`; the bounded-plane identity
pivot remains one time-bounded experiment rather than a new open-ended main
line. Strong novelty and paper submission are not completion requirements.

Implemented the missing lightweight structural-line path:

* `extract_structural_lines.py` reconstructs every saved DUSt3R pointmap RGB,
  XYZ and confidence image from the complete global cache.
* OpenCV LSD detects 2D lines in the exact aligned-pointmap resolution. Line
  pixels are stored in explicit `(x,y)` order and
  `dust3r_aligned_pointmap` space.
* 3D lifting samples only the same view and exact pointmap pixels, filters
  confidence/invalid geometry, splits invalid or large 3D gaps, keeps the
  longest contiguous run and fits a deterministic PCA segment.
* Optional plane labels are joined by exact `(alignment_view_index,x,y)`.
  Duplicate agreement is collapsed and conflicting duplicate keys are dropped
  and counted rather than guessed.
* Both sides of a line are associated with plane labels as unassigned,
  within-plane, plane-boundary or single-plane-edge evidence.
* Outputs are a schema-versioned NPZ, JSON line ledger, edge PLY, one overlay
  per view and a checksum/config/runtime manifest. The exporter refuses to
  overwrite an existing output directory.
* `run_research_practice_line_smoke.sh` targets the retained scene-00180 cache,
  checks/installs `roma==1.5.6`, checks OpenCV, records checksums/GPU/Git SHA
  and runs without recomputing DUSt3R alignment.

Focused validation completed locally:

```text
python -m py_compile extract_structural_lines.py tests/test_extract_structural_lines.py
python tests/test_extract_structural_lines.py
python tests/test_stage3_scene_plane_fusion_mapping.py
<Git-for-Windows-bash> -n run_research_practice_line_smoke.sh
```

Results:

```text
structural-line tests: 7 passed
Stage3 mapping/cache tests: 5 passed
shell syntax: passed
```

The structural-line tests cover conflict-preserving label rasterization, both
supported prediction schemas, clipped `(x,y)` sampling, two-sided boundary
association, longest-contiguous-run 3D fitting, edge PLY output and one full
synthetic CLI artifact set. The retained synthetic visual diagnostic detected
four lines, lifted all four, classified two as plane boundaries and two as
within-plane, and produced the expected overlay. It is a behavior diagnostic,
not a real-scene metric.

No real server structural-line run, cross-scene batch result, efficiency
benchmark or report result is claimed yet. The next exact step is to push this
implementation and run `bash run_research_practice_line_smoke.sh` on the
retained cache before connecting the multi-scene batch runner.

The first server launch of commit `283c43a` stopped before line extraction.
The runner had defaulted to `/root/miniconda3/bin/python`; installing `roma`
there could not satisfy its runtime import because that base environment had no
`torch`. Its OpenCV fallback also upgraded base-environment NumPy. No global
cache or output directory was modified, although the sibling `_run.log` was
created.

The runner now defaults explicitly to
`/root/miniconda3/envs/lightrecon/bin/python`, refuses to fall back when that
executable or `torch` is missing, installs `roma==1.5.6` with `--no-deps`, and
uses an OpenCV `<5` no-dependency fallback so dependency repair cannot silently
replace NumPy. The real rerun must use a new `v2` output path because the `v1`
run log is retained as failure evidence.

The explicit-Python `v2` launch reached the dependency/version preflight and
confirmed `cv2=4.13.0` and `numpy=2.2.6`, but stopped because the logging probe
read a nonexistent `roma.__version__` attribute. No extraction or output
directory creation occurred; only the sibling `v2_run.log` was written. The
probe now obtains the installed distribution version through
`importlib.metadata.version("roma")`. The next real launch must use `v3`.

## 2026-07-15 real structural-line gate and batch preflight

The real scene-00180 structural-line smoke completed on server commit
`314c7fc`, using the frozen 715,848-point global cache with SHA256
`77b745a52bf28d170977f9ffd14da79c11df5e7940c886b4f36e69f0daf32101`.
It processed five views in 1.3295 seconds, detected 291 2D segments and
exported 289 lifted 3D segments. The two-sided plane association ledger
contained 77 unassigned, 92 within-plane, 3 plane-boundary and 117
single-plane-edge segments.

The engineering gate passed: exact pointmap lifting, line NPZ/JSON, PLY and
all five overlays were produced. The reconstruction-constraint usefulness
gate failed. The overlays were dominated by window/material-frame edges and
duplicate parallel responses, while only three segments were associated with
two distinct planes. The Stage2 plane labels also covered only 15,972 of
715,848 cache points (2.23%). Structural lines are therefore retained as an
auxiliary point/line/plane output and a report ablation. They are not used to
modify global alignment or plane geometry, and no line-detector tuning or new
line model is authorized before the deadline.

Implemented the first W2 batch component:

* `research_practice_batch.py` reads a schema-versioned manifest and performs
  a CPU-only preflight before expensive GPU execution.
* Every Stage2 input file is hashed. Stage2 metadata, image existence, scene
  name, pair group and minimum unique-view count are validated through the
  existing Stage3 view-registry implementation.
* Optional frozen artifacts can be required and checked against an expected
  SHA256. Missing files and checksum mismatches become explicit failure rows.
* Duplicate scene/pair group entries are rejected. Unique view groups and
  unique Structured3D scenes are counted separately so repeated room
  perspectives cannot be reported as independent scenes.
* JSON, UTF-8 CSV and Markdown summaries are written to a new directory; the
  tool refuses to overwrite an existing output.
* `docs/research_practice/manifests/three_group_smoke.json` freezes the first
  three retained groups. They contain three view groups but only two unique
  Structured3D scene IDs, which the report must state explicitly.
* `run_research_practice_batch_preflight.sh` is the single server entrypoint
  for this non-GPU discovery step.

Local validation completed:

```text
research-practice batch tests: 4 passed
python syntax checks: passed
batch preflight shell syntax: passed
```

No three-group server preflight, new global alignment, cross-scene metric or
efficiency number is claimed yet. The next server action is the CPU-only
three-group input preflight. Its archived JSON will freeze the valid view
groups and checksums; cache inventory and any required one-time global
alignment generation follow only for those passing rows.

## 2026-07-15 three-group preflight result and smoke executor

The server preflight completed successfully on commit `147615f`. All three
retained view groups passed exact metadata and image checks:

```text
items                       3
passed_items                3
failed_items                0
Stage2 records             30
unique views per group      5, 5, 5
unique view groups          3
unique Structured3D scenes  2
input bytes                 5,550,864
```

The smoke set is therefore three room/perspective view groups, not three
independent scenes. Group 000 and group 001 both belong to `scene_00180`;
group 002 belongs to `scene_00181`. This smoke can validate the batch machinery
but cannot satisfy the final eight-scene requirement or the cross-scene
promotion gate.

Implemented `execute_research_practice_batch.py` and the server entrypoint
`run_research_practice_batch_smoke.sh`:

* Group 000 reuses the frozen cache with verified SHA256
  `77b745a52bf28d170977f9ffd14da79c11df5e7940c886b4f36e69f0daf32101`.
  Groups 001 and 002 generate one DUSt3R global alignment each. No plane
  feedback is enabled.
* Every method within a group consumes that group's identical global cache.
  The frozen stages are direct per-support global SVD, point-aligned
  Structured3D GT, current manual support merge, sequential RANSAC, exact
  support lift and structural-line output.
* Full-cache metrics compare GT, RANSAC, direct-support conflict-drop and
  manual-support conflict-drop outputs only after exact cache indexing.
* The primary identity audit keeps repeated raw support records and their
  conflicts. Conflict-drop results are labeled as ablations, never as the
  primary score.
* Structural lines produce NPZ, PLY and five overlays per group but do not
  modify points, poses or plane assignments.
* Each subprocess streams to its own log. A failed stage becomes an explicit
  item failure row, while partial outputs and checksums remain available.
* The batch writes `batch_execution.json`, item CSV, concatenated metric
  JSON/CSV, per-method mean/median JSON/CSV and Markdown. Unique view groups
  and independent scene IDs remain separate counters.
* The server launcher fixes the project Python, checks CUDA, NumPy, SciPy and
  torch, checks/installs `roma==1.5.6`, `trimesh==4.9.0` and a pre-5 OpenCV
  build without allowing those repairs to replace NumPy.

Local validation completed:

```text
focused tests: 50 passed
Python syntax checks: passed
smoke shell syntax: passed
```

No three-group execution metrics are claimed yet. The next server interaction
is the one-command smoke run from
`docs/research_practice/manifests/three_group_smoke_execute.json`.

## 2026-07-15 three-group smoke aggregate and gate audit

The supplied `aggregate_method_summary.json` contains completed metric rows for
all three smoke view groups (two independent Structured3D scene IDs). This is a
smoke signal only; the per-group gate and `batch_execution.json` still need to
be audited before any method decision.

Conflict-preserving support-record medians:

```text
method                         pairwise F1  GT coverage  assignment  plane precision  observed recall
global RANSAC                  0.704429     0.999740     0.780800    0.600000         0.500000
raw manual support merge       0.769270     0.994722     0.994400    0.250000         0.500000
raw direct per-support SVD      0.119595     0.994722     0.994400    0.000000         0.000000
```

The difference between the manual and RANSAC aggregate median F1 values is
`+0.064841`, but their means are nearly tied (`0.742417` versus `0.741845`).
Raw manual merge reduces median overmerge excess from 2 to 1, while increasing
fragmentation excess from 1 to 2 and reducing plane precision from 0.60 to
0.25. It also contains a median 9,084 conflicting positive keys and 36,297
conflicting records. Per-group deltas are required to determine the 70% win
rate gate; aggregate medians alone cannot answer it.

The conflict-drop numbers are not valid primary improvements:

```text
method                         pairwise F1  GT coverage  assignment
manual unique conflict-drop    0.844669     0.544519     0.412850
direct unique conflict-drop    0.827462     0.000585     0.000550
```

Both obtain a higher identity score by discarding coverage. Direct
conflict-drop is especially degenerate and must not be shown as a successful
method. These remain coverage-collapse ablations only.

Implemented `audit_research_practice_batch_results.py` and
`run_research_practice_smoke_gate.sh`. The audit reads the archived per-group
metric rows and batch ledger, rejects failed/missing methods, computes the
manual-versus-RANSAC F1 delta for every group, and applies the pre-registered
median gain, group win rate, coverage, assignment and overmerge gates. It also
reports conflict-drop coverage collapse separately and refuses promotion
before eight independent scenes. Outputs are JSON, CSV and Markdown, and the
audit refuses to overwrite an existing directory.

Local audit validation completed:

```text
smoke-gate tests: 4 passed
Python syntax check: passed
shell syntax check: passed
```

No final stop/go decision is claimed from the aggregate summary. The next
server step is the CPU-only smoke gate on the existing batch files; no GPU or
reconstruction rerun is needed.

## 2026-07-16 identity stop gate and learning-guided RANSAC pivot

The CPU-only server gate completed on commit `d8fccba` for all three smoke
view groups (two independent Structured3D scene IDs). The pre-registered raw
manual identity gate failed:

```text
median manual raw F1                  0.769270
median global RANSAC F1               0.704429
median per-group manual delta         0.058468
manual group wins                     2 / 3 (0.666667; required 0.70)
median manual GT coverage             0.994722
median manual assignment              0.994400
median manual plane precision         0.250000
median RANSAC plane precision         0.600000
median manual fragmentation excess    2
median RANSAC fragmentation excess    1
median manual overmerge excess        1
median RANSAC overmerge excess        2
```

The per-group F1 deltas were `+0.064841`, `+0.058468`, and `-0.121594`.
The decision is therefore
`stop_identity_method_promotion_use_strongest_baseline`. Manual aggregation
and conflict-drop are frozen as ablations; no additional identity-threshold
tuning is authorized. Conflict-drop remains invalid as a primary result
because median GT coverage collapses to `0.544519` for manual and `0.000585`
for direct support.

This stop does not remove learned plane support from the project. A narrower
learning-support-guided RANSAC candidate is now implemented:

```text
direct Stage1/Stage2 local plane supports
-> exact (alignment_view_index,x,y) cache mapping
-> robust support-restricted plane hypotheses
-> full frozen-global-cloud consensus scoring
-> DUSt3R-confidence-weighted refit
-> bounded connected components
-> reduced random-RANSAC fallback on uncovered points
```

`guided_plane_ransac.py` deliberately uses the unmerged direct support output,
not failed cross-view manual plane IDs. Duplicate and conflicting support
records are preserved as competing hypotheses. They are neither assigned by
nearest XYZ nor silently dropped. Output uses the same full-cache editable
plane schema as `global_ransac_cc`.

`evaluate_guided_ransac_smoke.py` reuses each archived batch item's verified
global cache, direct support records, point-aligned GT and global-RANSAC
artifact. It runs only the guided method and evaluates an explicit two-path
gate:

* quality: median F1 gain at least `0.02`, wins at least 2/3 groups, coverage
  at least `0.90`, and plane precision/overmerge not worse than RANSAC;
* efficiency: median F1 loss no more than `0.01`, runtime no more than `0.75x`
  RANSAC, coverage at least `0.90`, and precision/overmerge not worse.

Passing either path is only a smoke signal; final promotion still requires at
least eight independent scene IDs. Failure keeps the guided implementation as
an ablation and global RANSAC as the primary engineering baseline.

The existing manifest-driven final executor now includes the guided method in
both full-cache and support-record evaluations. The new server entrypoint is
`run_research_practice_guided_ransac_smoke.sh`. It checks the fixed
`lightrecon` Python, installs `roma==1.5.6` with `--no-deps` if absent, refuses
to overwrite output, and does not require CUDA or rerun DUSt3R.

Local validation completed with the bundled scientific Python runtime:

```text
guided/baseline/lift/batch focused tests: 25 passed
additional metric/audit/batch/mapping tests: 29 passed, 1 skipped
Python syntax checks: passed
guided smoke and batch shell syntax: passed
```

One unrelated Structured3D GT test could not import local `cv2` in the bundled
desktop runtime; it did not execute and is not claimed as a pass. The server
`lightrecon` environment already recorded `cv2=4.13.0`. No real guided-RANSAC
metric is claimed yet. The next exact server action is the one-command guided
smoke on the existing batch, followed by the unique-scene final manifest only
if the gate result justifies it.

## 2026-07-16 deterministic final-scene preparation

The downstream final-set selection step is implemented without starting GPU
work. `prepare_research_practice_final_manifest.py` rebuilds the validation
pair-group inventory from the same `Structured3DDataset` configuration and
selects exactly one eligible pair group per Structured3D scene.

Selection is deterministic and metric-blind:

* scene names are sorted and the first target scene IDs with at least ten
  valid all-pairs records are retained;
* within one scene, a complete existing Stage2 group is preferred, then pair
  group paths are sorted lexicographically;
* no reconstruction score, GT metric or qualitative result participates in
  selection;
* duplicate scene IDs and duplicate pair groups are rejected before output.

The selector reads the archived `selected_groups.tsv` rather than inferring
old group numbers from directories. An existing Stage2 group is considered
ready only when its expected record count is present. Archived global caches
are reused only when a passed `batch_execution.json` item has the exact same
pair-group metadata and a recorded SHA256.

Outputs are a JSON/CSV/Markdown selection plan plus
`final_unique_scenes_execute.json`. Missing Stage1/Stage2 inputs are listed
with deterministic target paths under a separate expansion root; the selector
does not create or run those inputs. The execution manifest may therefore be
precomputed safely while the guided-RANSAC smoke is pending.

`run_research_practice_final_selection_preflight.sh` is a CPU-only server
entrypoint. It fixes the `lightrecon` Python, checks `cv2`, NumPy and torch,
installs `roma==1.5.6` with `--no-deps` if absent, and explicitly records that
CUDA is not used. It refuses to overwrite its output and performs no Stage1,
Stage2, DUSt3R, RANSAC or metric execution.

Local validation completed:

```text
final scene-selection tests: 6 passed
Python syntax checks: passed
selection shell syntax: passed
```

No server scene inventory or exact list of eight scene IDs is claimed yet.
The guided-RANSAC smoke remains the immediate gate; the prepared selector is
the next read-only step and prevents the old eight-view-group/five-scene
counting error from recurring.

## 2026-07-16 guided smoke result and final eight-scene launch path

The supplied server output contains a completed guided-RANSAC smoke from the
existing `research_practice_guided_ransac_smoke_20260716_v1` directory. The
new launch correctly refused to overwrite it; the archived Markdown was then
read successfully. Results on the three retained view groups are:

```text
group                    RANSAC F1  guided F1  delta      RANSAC s  guided s
group_000 / scene_00180  0.704307   0.720109   +0.015802  18.750    5.087
group_001 / scene_00180  0.595275   0.608537   +0.013262   3.933   37.318
group_002 / scene_00181  0.924908   0.926884   +0.001976  10.040   12.918
```

Guided RANSAC improved F1 in all three groups and retained median coverage
`0.999683`, plane precision `0.60`, and overmerge excess `2`. This is a
consistent small accuracy signal, but it did not pass either pre-registered
promotion path:

* median F1 gain was `+0.013262`, below the quality threshold `+0.02`;
* median runtime ratio was `1.286575`, above the efficiency threshold `0.75`.

The archived decision `keep_guided_as_ablation_ransac_primary` is accepted
without post-hoc threshold tuning. Global RANSAC remains the stable primary
baseline. Learning-guided RANSAC remains an implemented method contribution
and final-batch ablation whose honest claim is consistent small smoke F1 gain,
not established superiority or acceleration.

The deterministic selection preflight completed on commit `2e0979b` and
selected eight unique scene IDs and pair groups:

```text
scene_00180  existing Stage2, archived cache
scene_00181  existing Stage2, archived cache
scene_00182  existing Stage2
scene_00184  existing Stage2
scene_00185  existing Stage2
scene_00186  needs frozen Stage1/Stage2 materialization
scene_00187  needs frozen Stage1/Stage2 materialization
scene_00189  needs frozen Stage1/Stage2 materialization
```

Five Stage2 groups are reused, three require materialization, and two global
caches are reused with exact pair-group metadata and SHA256. Scene `00188` was
not silently substituted or manually removed; under the frozen eligibility
rule the next selected eligible scene after `00187` is `00189`.

Implemented `materialize_research_practice_final_inputs.py`:

* reads and hashes the frozen selection plan;
* validates unique item, scene and pair-group identities;
* treats all five existing Stage2 groups as read-only;
* runs the frozen Stage1 teacher-support exporter and learned Stage2 region
  merge only for the three `needs_stage1_stage2` rows;
* freezes the exact dataset indices, validation split, all-pairs strategy,
  Stage1 support limits and Stage2 safety gate;
* validates every final input with the existing exact view-registry and image
  metadata checks;
* preserves per-stage logs and failure rows and refuses any existing planned
  materialization root rather than overwriting a partial run.

`run_research_practice_final_batch.sh` is the single GPU server entrypoint for
the next phase. It checks the fixed `lightrecon` Python and CUDA, installs
`roma==1.5.6` and `trimesh==4.9.0` with `--no-deps` when needed, and performs:

```text
three missing Stage1/Stage2 groups
-> strict eight-independent-scene preflight
-> identical-cache final executor for all eight scenes
```

The final executor includes global RANSAC as primary, guided RANSAC as the
learning-guided ablation, direct/manual/conflict variants, point-aligned GT,
structural lines, full-cache metrics, support-record metrics and per-method
aggregate JSON/CSV. Existing output, preflight, materialization or launcher
paths cause an immediate stop.

Local validation of the new launch path completed:

```text
materialization tests: 5 passed
Python syntax checks: passed
final batch shell syntax: passed
```

No eight-scene accuracy or runtime result is claimed yet. The next authorized
server action is the final one-command GPU batch after this implementation is
committed and pushed.

## 2026-07-16 final eight-scene batch result and paired audit

The user-supplied server output records a complete final batch on commit
`788ed0db701a2d8f26e23ac5cca5557ed7b0a6ae`. Input materialization passed
`8/8`, strict preflight passed `8/8`, and final execution passed `8/8` view
groups from eight independent Structured3D scene IDs. The three missing
Stage1/Stage2 groups (`scene_00186`, `scene_00187`, `scene_00189`) materialized
in `58.188`, `64.898`, and `64.199` seconds. All five existing Stage2 groups
remained read-only.

The supplied aggregate-method JSON gives the following full-cache means:

```text
metric                         global RANSAC  guided RANSAC  delta
partition pairwise F1          0.704999       0.744248       +0.039248
purity/completeness F1         0.776531       0.822015       +0.045484
support-conditioned IoU        0.614389       0.685150       +0.070761
plane precision                0.482540       0.600595       +0.118056
plane recall over all GT       0.415751       0.586584       +0.170833
support coverage               0.997793       0.999610       +0.001818
fragmentation excess           1.875          1.250          -0.625
overmerge excess               2.500          1.125          -1.375
normal angular error (degrees) 2.974294       4.849207       +1.874913
runtime (seconds)              14.938125      13.127324      -1.810800
```

These means show a useful learning-guided signal: better support partition,
plane detection, coverage and structural separation, with worse matched-plane
normal angle. Mean runtime is lower, but median runtime is slightly higher
(`10.187` versus `10.735` seconds), so no acceleration claim is accepted from
the aggregate alone.

The conflict-drop variants remain invalid primary comparisons despite high
conditional scores. Their median full-cache coverage is only approximately
`0.01455` for manual conflict-drop and `0.0000374` for direct conflict-drop.

The supplied attachment is an aggregate-method summary, not the per-scene
`aggregate_metrics.json`. A difference of aggregate medians cannot establish
the pre-registered paired median gain or scene win rate. Therefore final
method promotion is not claimed yet.

Implemented `audit_research_practice_final_results.py` to consume the archived
`aggregate_metrics.json` and `batch_execution.json` without GPU recomputation.
It:

* requires exactly matched RANSAC/guided rows and unique scene IDs;
* reapplies the frozen smoke quality and efficiency thresholds to the eight
  paired scenes;
* reports per-scene wins, median paired gain, coverage, precision, overmerge
  and runtime ratio;
* emits deterministic paired-scene bootstrap intervals and an exact sign test
  as descriptive uncertainty diagnostics only, never as a post-hoc gate;
* preserves input paths and SHA256 and refuses to overwrite an existing audit;
* reports conflict-drop coverage collapse alongside report-ready JSON, CSV and
  Markdown tables.

`run_research_practice_final_audit.sh` is a CPU-only one-command entrypoint. It
runs both the new guided-RANSAC final gate and the existing raw-manual identity
gate on the exact final artifacts. No DUSt3R alignment, model inference,
training, or RANSAC recomputation occurs.

Local validation completed:

```text
final audit + related batch/gate tests: 45 passed
Python syntax check: passed
Git Bash shell syntax check: passed
```

The immediate next server action is this CPU-only paired audit. Its archived
decision determines whether the report promotes learning-guided RANSAC as the
final method or retains it as a positive ablation.

## 2026-07-16 final method promotion and W3 efficiency launcher

The supplied paired audit completed successfully on all eight independent
scenes and returned:

```text
decision: promote_learning_guided_ransac_final
quality path: pass
efficiency path: fail
guided F1 wins: 8/8 scenes
median paired F1 gain: +0.035580
mean paired F1 gain: +0.039248
paired bootstrap 95% interval: [+0.015288, +0.063010]
exact two-sided sign-test p: 0.007812
median coverage: 0.999828
median plane precision: 0.585714 vs 0.535714
median overmerge: 0.5 vs 2.0
median runtime ratio: 1.464814
```

Learning-support-guided RANSAC is now the frozen final method. Global RANSAC
is its deterministic geometric baseline. The honest claim is a cross-scene
quality improvement, not an acceleration claim. The guided method improved
pairwise F1 in all eight scenes; paired mean F1 increased from `0.704999` to
`0.744248`, matched IoU increased from `0.614389` to `0.685150`, and mean
overmerge excess decreased from `2.500` to `1.125`.

Normal error is reported only on seven common valid scene pairs because one
baseline scene had no finite matched-plane angle. On that paired domain the
guided mean is `2.195663` degrees versus `2.974294` degrees. This common-pair
number must not be confused with the separately aggregated guided normal mean
that includes a scene where the baseline value is undefined.

The raw manual identity method failed its final gate: median F1 delta versus
RANSAC was `-0.106110`, and it won only `3/8` scenes. It remains a negative
association ablation. Conflict-drop also remains diagnostic-only because its
median full-cache coverage is `0.014547` for manual and `0.000037` for direct
support.

Implemented `benchmark_research_practice_efficiency.py` and
`run_research_practice_efficiency.sh` for the final W3 server interaction. The
benchmark freezes resolution at `512 x 512` and separates the shared DUSt3R
backbone from the contributed Stage1 support head and Stage2 merge MLP. It
records:

* parameter, trainable-parameter, parameter-byte and checkpoint-byte counts;
* Stage1-head per-image P50/P95 latency and peak/incremental GPU memory;
* DUSt3R pair latency, Stage2 candidate-pair latency and a repeated five-view
  inference-plus-global-alignment benchmark;
* Stage1 support partition accuracy over the exact final eight-scene source
  files, with assignment/coverage reported separately;
* archived per-stage timings for RANSAC, guided RANSAC, lines, evaluation and
  uncached direct-support alignment-plus-export proxies;
* Git SHA, hardware, CUDA/library versions and SHA256 for every controlling
  input and checkpoint.

The benchmark refuses existing output and requires the archived final audit
decision to be `promote_learning_guided_ransac_final`. It performs inference
and timing only; no training, recaching, threshold tuning or result mutation
is allowed.

### First W3 server attempt and cache-schema fix

The first server attempt on commit `f8b4c70` passed dependency checks, loaded
the fixed checkpoints and reached the alignment-source selection step. It then
failed before writing the output directory with:

```text
FileNotFoundError: no final global cache with valid image paths
```

Cause: method-independent global caches written by
`write_global_cloud_cache()` store image paths inside
`dust3r_view_registry_json`; `dust3r_image_paths` exists only in later fused
prediction NPZ files. The efficiency reader incorrectly assumed the latter
field was present in the cache.

`first_alignment_images()` now reads and sorts the canonical cache view
registry by `alignment_view_index`, while retaining `dust3r_image_paths` as a
backward-compatible fallback. A focused test constructs the real registry
schema and verifies deterministic view ordering. The failed `_v1` launcher
log is preserved; the corrected rerun must use a fresh `_v2` output path.

### Second W3 server attempt and environment diagnostics

The corrected `_v2` attempt pulled commit `b5d80c2` successfully but stopped
in the launcher environment preflight before creating a run log or entering
the benchmark. The old preflight combined imports of OpenCV, NumPy, SciPy and
PyTorch with `torch.cuda.is_available()` while redirecting all diagnostic
output to `/dev/null`, so the supplied message cannot distinguish a failed
package import from a temporarily unavailable CUDA device.

The launcher now reports every package version, the PyTorch CUDA build,
`CUDA_VISIBLE_DEVICES`, CUDA device count and availability. It also performs
one real CUDA tensor allocation and synchronization. Import and allocation
tracebacks are retained, and `nvidia-smi` is printed on failure. The benchmark
still requires CUDA and never falls back silently to CPU. Because the second
attempt stopped before `RUN_LOG` creation, the same `_v2` path may be reused
unless a path was created by an external action.

## 2026-07-16 W3 completed and W4 started

The server pull for the next attempt timed out, so the benchmark ran at
`b5d80c2` rather than diagnostic-only commit `123e9d5`. This is valid because
`b5d80c2` contains the cache-reader correction and `123e9d5` does not change
benchmark computation. The run completed at:

```text
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2
```

It evaluated 80 Stage1 pair records from eight independent scenes. Mean
pairwise precision, recall and F1 were `0.883814`, `0.629410` and `0.724166`.
Mean purity/completeness F1 was `0.809942` with `0.998100` assignment and GT
coverage. The Stage1 head has `79,906,572` parameters and a `304.898 MiB`
checkpoint; per-image P50/P95 latency was `75.160/101.401 ms` with `361.899
MiB` incremental peak allocation. It is an added task head, not a tiny head.

The Stage2 MLP has `203,521` parameters and processed 64 candidate pairs in
`0.451 ms` P50. Five-view DUSt3R inference plus 300-step alignment measured
`9.123 s` P50 and `11.922 s` P95 over three repeats. The actual 16:9 images
were loaded at `512 x 288` under the image-size-512 setting. Structural-line
output averaged `1.725 s` across eight scenes.

W3 is complete. No acceleration claim is made: the final method was promoted
through the frozen quality path only. W4 report and defense work is active.
The local report evidence record is
`docs/research_practice/W3_EFFICIENCY_RESULT.md`; canonical values remain in
the server's `efficiency_results.json` and CSV files.

The first Chinese report body now exists at
`docs/research_practice/REPORT_DRAFT.md`. It contains 8,421 Chinese characters
before the bibliography, covering the task, true guided-RANSAC implementation,
frozen protocol, W3/final results, negative ablations, limitations and
reproducibility. `docs/research_practice/REFERENCES.md` records 30 primary
paper sources. The prose was audited to avoid claiming that the 79.9M-parameter
Stage1 head is tiny, that structural lines improve plane metrics, or that the
guided method is universally faster. Remaining W4 work is GB/T 7714 metadata
cleanup, table/figure generation, failure-gallery selection, Word formatting,
slides and the static demo.

## 2026-07-17 mentor-facing stage report with real 3D figures

A mentor-facing stage report was generated without modifying any experiment
result. The new builder is:

```text
docs/research_practice/build_stage_progress_report.py
```

It reads the archived Stage3 showcase NPZ/PLY assets and produces real 3D
projections rather than using the structural-line overlays as reconstruction
evidence. The report contains:

* one scene shown as RGB global point cloud, categorical plane instances and
  bounded textured plane mesh;
* a two-view 3D gallery for `scene_00180`, `scene_00182` and `scene_00185`;
* the final eight-scene paired RANSAC/guided-RANSAC result chart and table;
* W3 model-size and efficiency evidence;
* an explicit limitation statement that eight scenes are a pilot rather than
  a large-scale generalization result;
* a dated plan to download the final per-method PLY/NPZ outputs and expand the
  frozen protocol to at least 30, targeting 50, independent scenes.

Generated deliverables:

```text
output/research_practice_stage_report_20260717/LightRecon3D_阶段性研究进展报告_20260717.docx
output/research_practice_stage_report_20260717/LightRecon3D_阶段性研究进展报告_20260717.pdf
```

Validation completed:

```text
builder py_compile: passed
DOCX structural check: 67 paragraphs, 10 tables, 5 inline figures
DOCX image audit: all 5 figures are inline; no floating anchors
PDF render: 10 A4 pages
visual review: all 10 rendered pages inspected; no clipping, overlap or missing CJK glyphs
PDF text check: title, 3D-result section and 30-50-scene plan present
```

LibreOffice is not installed locally, so the packaged DOCX renderer could not
execute. The independently generated PDF was rendered through Poppler and
passed full visual review. The final-method RANSAC/guided-RANSAC PLY files
remain server-side; the report labels the local Stage3 showcase honestly and
does not present it as the final per-method geometry comparison.

## 2026-07-17 actual final-batch 3D visualization entrypoint

The old Stage3 showcase image is **not** acceptable as the final-method result:
it uses `scene_00180` support/showcase assets, an edge-on camera and a sparse
bounded mesh, rather than the final eight-scene ordinary/guided RANSAC outputs.
It must not be used as the main mentor-facing reconstruction figure.

The following CPU-only render path was added:

```text
visualize_research_practice_final_3d.py
run_research_practice_final_visualization.sh
tests/test_visualize_research_practice_final_3d.py
```

The renderer reads the paths recorded under each passed batch item's
`global_ransac` and `learning_guided_ransac` artifacts. It requires exact
equality of the ordered points and original RGB arrays before drawing a
comparison. Every scene uses one shared PCA canonical frame, an automatically
selected pair of non-redundant oblique cameras and one elevated camera. RGB,
ordinary RANSAC and guided RANSAC use the same camera and z-buffer. Output
contains per-view PNGs, a 3x3 scene sheet, an eight-scene contact sheet, source
hashes and a visualization manifest. Plane colors are explicitly documented as
method-local IDs.

Local validation (synthetic room fixture, not a claimed experiment result):

```text
py_compile: passed
Git Bash shell syntax check: passed
18 focused visualization/audit/batch tests: passed
synthetic multiview image: visually inspected
```

Actual final scene renders have not been generated locally because the final
NPZ files remain on the server. Run there with a fresh output directory:

```bash
OUT_DIR=/gemini/data-1/lightrecon_runs/research_practice_final_3d_visualization_20260717_v1 \
bash run_research_practice_final_visualization.sh
```

The launcher uses `/root/miniconda3/envs/lightrecon/bin/python`, installs
`roma==1.5.6` with `--no-deps` only when missing, requires NumPy/Pillow, refuses
overwrites and does not use CUDA or recompute reconstruction.

## 2026-07-17 all-eligible-validation experiment launcher

The next server experiment is defined as every eligible independent scene in
the frozen `val` split (five views and at least ten Stage2 pair records), rather
than filling a target count with training scenes. The selection script accepts
`--target_scenes 0` for this mode. The one-command entrypoint is:

```text
run_research_practice_all_validation.sh
```

It performs deterministic selection, missing Stage1/Stage2 materialization,
preflight, identical-cache ordinary/guided RANSAC reconstruction and metrics,
paired auditing, actual same-camera 3D visualization, and final collection.
Successful scenes remain usable when another scene fails; failure records are
retained and excluded from paired metrics. `RESUME=1` verifies frozen source
checksums and skips ledger-recorded work without deleting partial artifacts.
The final bundle contains `large_scale_summary.{json,md}`,
`scene_artifact_index.csv`, `failures.csv`, plus a contact sheet and per-scene
3D PNG paths. `roma==1.5.6` and `trimesh==4.9.0` are installed with `--no-deps`
when missing so NumPy is not upgraded.

Local validation before the server run:

```text
py_compile: passed
Git Bash shell syntax check: passed
34 focused selection/materialization/batch/audit/collector/visualization tests: passed
```

No large-scale result is claimed yet; actual scene count and metrics will only
be known after running the launcher against the server dataset inventory.

## 2026-07-17 all-validation result and immutable failed-scene recovery

The all-eligible validation launcher selected 17 independent scenes and
completed 14. On the 14 valid identical-cache comparisons, learning-guided
RANSAC beat ordinary RANSAC in 14/14 scenes. The archived aggregate result is:

```text
pairwise F1       0.644675 -> 0.721296  (+0.076621)
matched IoU       0.500448 -> 0.688247  (+0.187798)
overmerge excess  2.928571 -> 1.500000  (-1.428571)
method runtime s  19.945028 -> 18.356680 (-1.588348)
```

The three missing scenes were traced to infrastructure, not model output:

* `scene_00190`: platform `[bootstrap] Fail to init session`; the wrapper
  returned zero despite producing no Stage1 manifest or NPZ;
* `scene_00194`: platform Server Service connection failure, exit 255;
* `scene_00197`: CUDA device busy/unavailable while moving DUSt3R to GPU.

`materialize_research_practice_final_inputs.py` now verifies a non-empty,
parseable Stage1 manifest and the referenced NPZ files before Stage2 starts.
The non-destructive recovery path is:

```text
recover_research_practice_failed_batch.py
run_research_practice_failed_recovery.sh
```

It derives a three-item retry plan from the original failure ledger, writes all
retry inputs under a new root, and only replaces originally failed rows when a
retry passes. It verifies frozen input/checkpoint, DUSt3R weight, config and
coordinate-contract identity, then recomputes aggregate metrics, audit and 3D
visualization over the combined 17-row ledger. Original failure records and
paths remain embedded in the recovery provenance. No original output is
deleted or overwritten.

## 2026-07-17 report draft updated to the large-scale result

`docs/research_practice/REPORT_DRAFT.md` is now the second Chinese draft. The
body before the evidence index contains 9,446 Chinese characters under a
strict CJK count. It separates the earlier eight-scene confirmation experiment
from the current 17 selected / 14 valid-scene main result and records the three
infrastructure failures instead of treating them as method failures.

The draft now includes the Stage1 to Stage3 data flow, coordinate registry,
learning-guided RANSAC sampling and scoring, point-aligned GT construction,
metric definitions, batch and immutable recovery protocol, quantitative and
qualitative analysis, limitations, and an evidence index. The corresponding
outline has been updated, and
`docs/research_practice/LARGE_SCALE_RESULT_20260717.md` is the compact frozen
record of the current large-scale result and server artifact paths.

The existing 14-scene contact sheet is scientifically valid as a same-camera
3D point-plane assignment comparison, but it is too dense for the main report
figure. The final report should use selected scenes such as `scene_00186`,
`scene_00192`, `scene_00195`, and `scene_00198`, preferably with GT-matched
colors and an RGB / GT / ordinary RANSAC / guided RANSAC layout.

Report commit: `abe3ef2` (`Update report with large-scale validation`), pushed
to `origin/codex/bounded-support-head`. The commit contains only the report
draft, report outline, and large-scale result record. Validation completed:
strict body CJK count `9446`, 30 reference entries, all required sections
present, large-scale values and `[E4]` evidence paths present, no em/en dash,
and `git diff --cached --check` passed before commit.

Remaining report work is reference-metadata cleanup, curated GT-aligned
figures, school identity and title fields, and Word/template rendering. If the
three-scene recovery finishes, its combined ledger should update the evidence
record before the final document is frozen.

## 2026-07-17 failed-scene recovery completed and report result frozen

The immutable recovery completed all three originally failed scene IDs. The
combined ledger now contains 17 passed scenes, 17 unique scene IDs and no
unresolved failure row. Original failure logs and checksums remain in
`recovery_merge.json`; the first 14 successful scenes were not recomputed.

Final recovered-batch result:

```text
pairwise F1       0.632406 -> 0.706348  (+0.073942)
matched IoU       0.489136 -> 0.678036  (+0.188899)
overmerge excess  2.882353 -> 1.470588  (-1.411765)
method runtime s  19.826422 -> 17.892183 (-1.934239)
guided F1 wins    16 / 17
median F1 gain    +0.064718
```

`scene_00194` is the only negative F1 case, approximately `0.813 -> 0.811`.
The report must say 16/17 wins, not universal improvement. The exact two-sided
sign-test value for 16 wins and one loss is `0.000274658203125`.

The controlling artifact roots are:

```text
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_bundle
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_merged/recovery_merge.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_3d_visualization
```

`REPORT_DRAFT.md`, `REPORT_OUTLINE.md`,
`LARGE_SCALE_RESULT_20260717.md`, and the completion task now use the final
17-scene metrics. The original 14/17 batch remains recovery provenance rather
than the final result. The 17-scene contact sheet is suitable for an appendix;
the main report figure still needs selected enlarged cases with GT-matched
colors.

Report update commit: `1925df1` (`Freeze recovered 17-scene report result`),
pushed to `origin/codex/bounded-support-head`. It contains only the report
draft, outline, final large-scale evidence record and research-practice task.
The report body before the evidence index now contains 9,519 Chinese
characters. Validation completed: 30 reference entries, final 17-scene values
and recovery paths present, the `scene_00194` negative case present, no em/en
dash, exact sign-test value independently checked, and staged diff check
passed.

## 2026-07-17 mentor-facing final result report

Added `docs/research_practice/导师阶段结果汇报_20260717.md` as a short report that
can be sent directly to the mentor. It is separate from the old untracked
eight-scene brief and does not overwrite that file. The new report contains
1,610 Chinese characters and covers the final method, 17/17 recovered result,
16/17 F1 wins, the `scene_00194` negative case, visualization interpretation,
Stage1 and DUSt3R efficiency evidence, stopped directions, limitations and the
remaining report/defense work.

Commit: `fd6d3ac` (`Add mentor-facing 17-scene progress report`), pushed to
`origin/codex/bounded-support-head`. Validation: final metrics, completion,
negative case, no-speed claim and model-size caveat present; no em/en dash;
staged diff check passed.

## 2026-07-17 mentor report rewritten to the fixed student-report brief

`docs/research_practice/导师阶段结果汇报_20260717.md` was rewritten with the
user-specified title and six fixed sections. It contains 1,527 Chinese
characters, preserves the complete 17-scene metrics, the `scene_00194`
negative case, failed support/conflict/feedback experiments, the Stage1 model
size caveat and structural-line limitation. The text contains none of the
forbidden headings or transition phrases, no repeated `目前/当前/进一步/主要`,
and no em/en dash. This rewrite is not committed yet.

## 2026-07-17 mentor report now embeds the final 3D result

The mentor report's next step was corrected to focus only on the research
practice report, followed by an optional paper-style draft based on the same
frozen methods and experiments. Defense slides, live demos and related tasks
were removed from the active completion plan.

The supplied 17-scene contact sheet was copied to
`docs/research_practice/figures/all_scenes_final_3d_contact_sheet_17.png` and
embedded directly in `导师阶段结果汇报_20260717.md`. Source and copied SHA256 are
both `77B3EEB005BA174EEBF83E05EE6E0B416ACB760D538BC9D3B729077A04DA056D`.
The embedded image was visually inspected after copying.

Commit: `b975265` (`Embed final results in mentor report`), pushed to
`origin/codex/bounded-support-head`. The report has 1,771 Chinese characters,
contains no forbidden transition phrase or em/en dash, and clearly sequences
the final practice report before the optional paper-style rewrite.

## 2026-07-17 publication-style metric baseline

The 17-scene result is now explicitly treated as an internal pilot rather than
a public-paper comparison. Added `public_partition_metrics()` to
`evaluate_global_plane_baselines.py`. It computes VOI in nats, Rand Index, and
Segmentation Covering in both directions plus their mean on all GT-planar
points. Predicted `-1` points are retained as a single unassigned segment, so
the public-style metrics cannot be improved by dropping difficult points.

Added `recompute_public_plane_metrics.py` and
`run_research_practice_public_metrics.sh`. They read the frozen combined batch
ledger and archived GT/RANSAC NPZ files, verify identical point arrays, and
write per-scene CSV/JSON plus an aggregate Markdown table. They do not rerun
DUSt3R, Stage1, RANSAC, or training.

Added `docs/codex_tasks/public_plane_benchmark.md` to separate the primary
unposed comparison track (ordinary RANSAC, guided RANSAC, Plane-DUSt3R, and a
separate PLANA3R two-view protocol) from posed ScanNet reference methods
(PlanarRecon, AirPlanes, and PlanarSplatting). The document freezes the metric
families and makes large-scale testing the next priority; new training remains
conditional.

Local validation used the bundled workspace Python:

```text
python -m py_compile evaluate_global_plane_baselines.py recompute_public_plane_metrics.py tests/test_evaluate_global_plane_baselines.py tests/test_recompute_public_plane_metrics.py
python -m unittest tests.test_evaluate_global_plane_baselines tests.test_recompute_public_plane_metrics
```

Result: 25 focused and batch-compatibility tests passed; Git Bash syntax check
passed. Commit: `b08da2b` (`Add public plane partition metrics`). Server
execution is still pending; therefore no 17-scene VOI/RI/SC values are reported
yet.

## 2026-07-17 public metrics complete and Plane-DUSt3R preflight

The server completed `run_research_practice_public_metrics.sh` at commit
`b08da2b`. All 17 independent Structured3D scenes passed. The frozen ordinary
RANSAC versus learning-guided RANSAC means are:

```text
pairwise F1:                    0.632406 -> 0.706348 (+0.073942)
matched IoU:                   0.489136 -> 0.678036 (+0.188899)
overmerge excess:              2.882353 -> 1.470588 (-1.411765)
VOI, nats (lower):             1.493965 -> 1.172062 (-0.321904)
Rand Index:                    0.789837 -> 0.834620 (+0.044783)
Segmentation Covering, sym.:   0.561563 -> 0.658550 (+0.096987)
```

Server evidence root:

```text
/gemini/data-1/lightrecon_runs/research_practice_public_metrics_20260717_v1
```

The evidence was added to `PUBLIC_PARTITION_METRICS_20260717.md`, the report
draft, report outline, mentor report, large-scale result record, and public
benchmark task.

Plane-DUSt3R is the first external paper baseline. Official source inspection
confirmed that `evaluate_planedust3r.py` imports `MASt3R.dust3r_extract`,
`NonCuboidRoom`, `plane_merge_planedust3r.py`, and `metric.py`; it reads
`2D_rendering/<room>/perspective/full/*/rgb_rawlight.png`. The official script
hardcodes `save = False` and catches per-room exceptions with a bare
`continue`, so an auditable wrapper is required before a batch claim.

The frozen LightRecon3D batch uses `perspective/empty`, and Plane-DUSt3R's
native output/evaluation is structural room layout rather than the same global
point-cache plane partition. It must first be reproduced in a separate native
metric table. Shared VOI/RI/SC is allowed only after a reviewed output adapter
maps predictions to the same ordered point cache without inventing omitted
labels.

Added:

```text
preflight_plane_dust3r_compatibility.py
run_plane_dust3r_compatibility_preflight.sh
tests/test_preflight_plane_dust3r_compatibility.py
docs/codex_tasks/plane_dust3r_compatibility.md
```

The CPU-only preflight inventories `empty/full` images for all frozen scene
IDs and checks the official repo, submodules and both checkpoints before the
4.41 GB Plane-DUSt3R weight is downloaded. Three new unit tests pass. The old
public-metric test could not be rerun with the default Windows MSYS Python
because that interpreter lacks NumPy; it previously passed at `b08da2b` and no
metric implementation was changed here. WSL/Git Bash is not installed in the
current Windows environment, so shell syntax must be confirmed on the Linux
server.

## 2026-07-17 Plane-DUSt3R preflight result and same-input pivot

The server ran `run_plane_dust3r_compatibility_preflight.sh` at commit
`8e6317e`. The frozen ledger contains 17 independent scenes and every source
group has five `perspective/empty` images, but none of those scene/room groups
contains `perspective/full` images. The official repo, 4.41 GB Plane-DUSt3R
checkpoint and NonCuboidRoom checkpoint are also not installed. The exact
result is:

```text
scenes=17
native_scene_ready=0
repository_ready=false
checkpoints_ready=false
native_smoke_ready=false
native_full_batch_ready=false
identical_input_scenes=0
common_partition_ready=false
```

This rules out reproduction of the paper's native `full`-render table on the
installed data. It does not rule out an external-model comparison. The active
plan is a same-input reproduction: run the official Plane-DUSt3R weights on
the exact frozen `empty` five-view RGB groups, clearly label the protocol
adaptation, and only add VOI/RI/SC if an output adapter can produce labels on
the same ordered global point cache.

Added `materialize_plane_dust3r_same_input.py` and
`run_plane_dust3r_same_input_materialization.sh`. They create a separate
dataset root where the evaluator's expected `perspective/full/<position>`
directories resolve to the frozen `perspective/empty/<position>` directories.
The source Structured3D tree is read-only. The manifest records source mode,
declared mode, scene, room, position IDs and that native-protocol claims are
not allowed. `copy` mode exists only as a fallback when symlinks are not
available.

Added `setup_plane_dust3r_repository.sh`. It clones the official repository
and initializes submodules with HTTP/1.1, validates the three evaluator files
and two submodule directories, and records the main and recursive submodule
commits. It does not download checkpoints, install dependencies, or run GPU
inference. Six focused tests for compatibility preflight plus same-input
materialization pass with the local stdlib-only Python. Linux shell syntax
still needs confirmation on the server because WSL/Git Bash is unavailable on
the Windows workstation.

## 2026-07-17 Plane-DUSt3R external environment and smoke runner

The server completed the same-input materialization: 17 independent scene and
room groups, 85 images total, and positions `0..4` for every item. The dataset
view is:

```text
/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_20260717_v1/dataset
```

The official repository setup also completed. Plane-DUSt3R is pinned to
`9a1ae50650ec6d706bf329352aaaf49efded90a0`; recursive submodule status is
empty because the checked repository contains the MASt3R and NonCuboidRoom
trees directly. The official README requests Python 3.11, PyTorch 2.2.0,
torchvision 0.17.0 and CUDA 11.8. The NonCuboidRoom checkpoint Google Drive ID
is `1DZnnOUMh6llVwhBvb-yo9ENVmN4o42x8`.

Added `prepare_plane_dust3r_external.sh`. It creates an isolated conda prefix
under `/gemini/data-1/lightrecon_envs`, installs the official environment and
requirements, checks `roma==1.5.6`, downloads the 4.41 GB Plane-DUSt3R weight
from the official Hugging Face repository and the NonCuboidRoom weight from
the README link, verifies minimum sizes, and records package inventories plus
checkpoint SHA256. It does not change the existing `lightrecon` environment.

Added `run_plane_dust3r_same_input_smoke.py` and its shell entry point. The
Python wrapper validates the exact external commit and clean working tree,
selects only `scene_00180`, creates an isolated runtime view, and patches
exactly one official `save = False` and one `except ... continue`. The patched
copy uses the CLI save flag and turns a room exception into a logged hard
failure. It records official/patched evaluator hashes, command, runtime,
return code, log and every saved artifact. The external repo remains unchanged.

Eight stdlib-only tests covering the compatibility preflight, same-input
materialization and evaluator patch pass locally. The environment install,
weight downloads and GPU smoke are pending server execution.

## 2026-07-17 accelerated and strict external checkpoint download

The server reported that the external setup download was too slow. The
original script used one `wget` connection and accepted any plane checkpoint
larger than 1 GB. That size-only gate was unsafe because a truncated 4.41 GB
file could exceed 1 GB and then fail later during model loading.

`prepare_plane_dust3r_external.sh` now defaults to a 16-connection `aria2c`
download with continuation enabled. It installs aria2 only inside the isolated
Plane-DUSt3R conda prefix. Existing partial checkpoint bytes are retained and
resumed in place. `DOWNLOAD_BACKEND=hf_xet` uses Hugging Face Hub's current
chunked parallel downloader with `HF_XET_HIGH_PERFORMANCE=1`; `wget` remains a
fallback. `HF_ENDPOINT`, both checkpoint URLs and connection count are
explicit overrides and are recorded in `download_provenance.txt`.

Both checkpoints must now be successfully deserialized by PyTorch on CPU
before setup can pass. There is no minimum-size shortcut. Final byte counts,
SHA256, backend, endpoint, environment and repository commit remain recorded.
The eight existing focused tests still pass; Linux shell execution is pending.

## 2026-07-17 Plane-DUSt3R Python 3.11 dependency repair

The first server setup failed before checkpoint download. The unpinned
`torch` and `torchvision` entries in the official MASt3R/DUSt3R requirement
files caused pip to replace conda PyTorch 2.2.0 with PyTorch 2.13.0 and CUDA
13 packages, while NumPy became 2.4.6. The following official
`NonCuboidRoom/requirements.txt` then requested `scipy==1.3.1`; that release
does not support Python 3.11 and failed while trying to build NumPy 1.14.5.
This polluted only the external prefix
`planedust3r-py311-torch220-cu118`; the `lightrecon` environment was not
changed.

Commit `c0ce23a` (`Fix Plane-DUSt3R Python 3.11 setup`) was pushed to
`origin/codex/bounded-support-head`. It adds
`prepare_plane_dust3r_requirements.py`, which reads the three requirement
files at the pinned external commit, preserves unaffected dependencies, and
writes sanitized requirements, a global constraints file and JSON/Markdown
replacement audits. PyTorch 2.2.0, torchvision 0.17.0, torchaudio 2.2.0 and
CUDA 11.8 remain conda-managed. NumPy is fixed at 1.26.4, SciPy at 1.11.4,
and the other old NonCuboidRoom pins are replaced by Python 3.11 compatible
versions close to their original APIs.

`prepare_plane_dust3r_external.sh` now defaults to the untouched new prefix
`/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118-v2`. It
refuses to repair an existing prefix with incompatible PyTorch, applies the
constraints to every pip step, runs exact version checks plus `pip check`,
and writes the completion marker only after all checks pass. It verifies the
environment again after installing aria2 or Hugging Face dependencies.
`run_plane_dust3r_same_input_smoke.sh` uses the same v2 prefix and checks
PyTorch/CUDA/NumPy/SciPy before starting GPU work.

Local validation:

```text
python -m py_compile prepare_plane_dust3r_requirements.py tests/test_prepare_plane_dust3r_requirements.py
python -m unittest tests.test_preflight_plane_dust3r_compatibility tests.test_materialize_plane_dust3r_same_input tests.test_run_plane_dust3r_same_input_smoke tests.test_prepare_plane_dust3r_requirements
12 tests passed
Git for Windows bash -n: both setup and smoke launchers passed
git diff --cached --check: passed before commit
```

Next exact server step; do not override `ENV_DIR` with the old prefix and do
not run the smoke until setup prints both checkpoint SHA256 values:

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head
git rev-parse --short HEAD

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v2 \
bash prepare_plane_dust3r_external.sh
```

Expected pulled SHA: `c0ce23a`. If setup succeeds, run the one-scene smoke
with a fresh v2 output directory. The old corrupted environment and partial
checkpoint bytes are deliberately retained; aria2 resumes the checkpoint in
place and PyTorch loadability remains the final completeness gate.

## 2026-07-17 Plane-DUSt3R MMCV source-build repair

The server ran the constrained v2 setup. The important first-stage fix was
confirmed: conda PyTorch remained 2.2.0 with CUDA 11.8, and the sanitizer
identified all 17 controlled official requirement lines. No PyTorch 2.13 or
CUDA 13 pip replacement occurred.

The run then failed while preparing `mmcv==1.7.2`. PyPI supplied a source
archive rather than a Python 3.11 wheel. Pip's isolated build environment used
a current setuptools build that did not expose `pkg_resources`, which the
MMCV setup imports. The failure occurred before pip's final install
transaction, so the v2 conda prefix remains reusable; it does not need to be
deleted or rebuilt.

Commit `bedb0cb` (`Fix Plane-DUSt3R MMCV bootstrap`) was pushed to
`origin/codex/bounded-support-head`. The compatibility constraints now include
`setuptools==80.9.0` and `wheel==0.45.1`. The setup first installs and verifies
that build bootstrap, installs wheel-backed compatibility packages so NumPy
becomes 1.26.4, and then installs MMCV alone with `--no-build-isolation`.
Generated requirement artifacts explicitly separate binary packages from the
single source-build package. The completion marker is bumped to `py311v3`, so
the failed v2 attempt cannot be mistaken for a completed setup.

Local validation after the repair:

```text
12 Plane-DUSt3R focused tests passed
prepare_plane_dust3r_external.sh: Git Bash syntax passed
run_plane_dust3r_same_input_smoke.sh: Git Bash syntax passed
git diff --cached --check: passed before commit
```

Next server command, reusing the existing v2 environment but writing a fresh
run ledger:

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head
git rev-parse --short HEAD

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v3 \
bash prepare_plane_dust3r_external.sh
```

Expected pulled SHA: `bedb0cb`. Do not run the GPU smoke until this setup
prints both checkpoint SHA256 values and the final ready message.

## 2026-07-17 Plane-DUSt3R PyTorch/MKL runtime repair

The v3 server run installed the pinned binary compatibility packages and
advanced into the MMCV source build, confirming that the previous
`pkg_resources` bootstrap failure was fixed. It then failed while importing
PyTorch from MMCV metadata generation:

```text
ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

The conda solve had selected MKL 2025.0.0 and Intel OpenMP 2025.0.0 for
PyTorch 2.2.0. Commit `6cf9f82` (`Pin MKL for Plane-DUSt3R PyTorch runtime`)
pins `mkl=2024.0` and `intel-openmp=2024.0`, then runs a real CPU tensor matrix
multiplication before any pip requirement or MMCV build. The full environment
verification repeats that operation. The completion marker and default setup
and smoke output directories were advanced to v4. The existing external v2
environment remains the intended prefix and should be repaired in place.

Local validation after the repair:

```text
12 Plane-DUSt3R focused tests passed
prepare_plane_dust3r_external.sh: Git Bash syntax passed
run_plane_dust3r_same_input_smoke.sh: Git Bash syntax passed
git diff --cached --check: passed before commit
```

Next server command:

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head
git rev-parse --short HEAD

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v4 \
bash prepare_plane_dust3r_external.sh
```

Expected pulled SHA: `6cf9f82`. The setup must print `PyTorch runtime
verified` before compiling MMCV. Do not start the GPU smoke until setup also
prints both checkpoint SHA256 values and the final ready message.

## 2026-07-18 Plane-DUSt3R environment stop point

The server successfully fetched and fast-forwarded to LightRecon3D commit
`6cf9f82e6c8df83fde011f3c99c0533702698e10`. The v4 setup then reached the
new conda repair branch but stopped during solving because the configured
channels do not expose `mkl=2024.0` or `intel-openmp=2024.0`.

No MKL downgrade, MMCV build, checkpoint download, or GPU smoke completed in
that run. The reusable external prefix remains:

```text
/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118-v2
```

Its last verified problem is MKL 2025.0.0 causing PyTorch 2.2.0 to fail with
`libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`. The main `lightrecon`
environment was not modified. Failed setup ledgers v4 and v5 must be retained
and not reused.

Tomorrow's first action is the explicit Anaconda archive install recorded in
`docs/codex_tasks/plane_dust3r_compatibility.md`: install MKL 2023.1.0 and
Intel OpenMP 2023.1.0 into the existing v2 prefix, then verify a real PyTorch
CPU matrix multiplication. Those commands have not yet been executed. Only
after `PyTorch verified` may setup resume with fresh output directory
`plane_dust3r_external_setup_20260718_v6`. Do not run the GPU smoke until
setup records both loadable checkpoint SHA256 values.

## 2026-07-18 local RTX 4060 Ti benchmark bootstrap

The manually transferred bootstrap archive was verified before extraction.
Its outer SHA-256 matched, and all 21 extracted file hashes matched the server
checksum set. It contains the frozen Stage1 epoch-23 checkpoint, Stage2 epoch-9
checkpoint, 17-scene result controls, selection records, public partition
metrics, and W3 efficiency controls. Both checkpoints load locally.

The user specified the local Conda `ai` environment. It reports Python at
`E:\anaconda\envs\ai\python.exe`, PyTorch `2.9.1+cu126`, and a CUDA-visible RTX
4060 Ti with 8187.5 MiB.

A real five-view DUSt3R global-alignment smoke completed on
`scene_00180/445895`, positions 0..4, using 512-pixel loading, 20 ordered pairs,
batch size 1, and 300 alignment iterations. It produced 737,280 finite cache
points, alignment loss `0.0076863365`, peak allocated memory `2740.4 MiB`, peak
reserved memory `2866.0 MiB`, and total model-load plus alignment time
`46.646 s`. The cache is 12,511,798 bytes.

A separate simultaneous backbone + frozen Stage1 + frozen Stage2 smoke also
passed. Two-view inference took `0.613 s`; the 288x512 Stage1 prediction
contained five plane IDs; Stage2 evaluated a 64-candidate dummy batch. Peak
allocated/reserved memory was `2837.0/3004.0 MiB`.

Generated local outputs:

```text
local_outputs/manual_transfer_bootstrap_20260718_v1_extracted/
local_outputs/rtx4060ti_dust3r_smoke_scene00180_20260718_v1/
local_outputs/rtx4060ti_stage_heads_smoke_scene00180_20260718_v3/
```

The global-alignment diagnostic PNG is:

```text
local_outputs/rtx4060ti_dust3r_smoke_scene00180_20260718_v1/scene_00180_445895_global_alignment_diagnostic.png
```

No tracked code was changed for the smoke. Two failed wrapper attempts are
preserved as empty output directories: v1 lost its multiline body under
Windows `conda run`, and v2 passed NumPy `true_shape` values directly to the
backbone. The successful v3 converts images and `true_shape` to CUDA tensors.

Local VRAM feasibility is now established. The next exact step is to create a
minimal manual-transfer bundle for the 17 frozen global caches, ordinary and
guided RANSAC NPZs, and point-aligned GT required for read-only reproduction
and ablations. Do not recache the 17 scenes merely to reproduce archived
metrics.

## 2026-07-18 completed local 17-scene B0/B4 benchmark

The previously planned manual-transfer-only reproduction was superseded with
the user's explicit authorization to continue locally. All commands used the
Conda `ai` interpreter at `E:\anaconda\envs\ai\python.exe`.

The frozen selection contains 17 independent Structured3D development scenes:

```text
scene_00180, scene_00181, scene_00182, scene_00184, scene_00185,
scene_00186, scene_00187, scene_00189, scene_00190, scene_00191,
scene_00192, scene_00193, scene_00194, scene_00195, scene_00197,
scene_00198, scene_00199
```

The executed manifest is:

```text
local_outputs/local_17scene_recache_20260718_v3/selection/final_unique_scenes_execute.json
SHA-256 198b5ac1d4895fd8e9a344c9bbc3991b3cc1df37128fa87224f7514a9cd1a6ef
```

Stage1/Stage2 materialization passed for 17/17 scenes in 673.443033 seconds.
The final benchmark was recovered in two immutable batches because one long
Windows process stopped early: the first five completed items are in
`local_outputs/b17u4`, and the remaining twelve are in
`local_outputs/b12r2`. The source `batch_execution.json` hashes are:

```text
b17u4:  816f9aacebe59728acd19915cb5397e92f71f4aeefdd89f28ba02993cb94af25
b12r2:  a8bf02173e94df1d886b0934f23eeff74bf705a201dec8f3e0f286a313281eee
```

The recovery batch passed 12/12 scenes in 3504.554466 seconds. It recomputed
global alignment except for `scene_00186`, whose already validated local
global cache was reused. The first batch passed 5/5 in 1681.698245 seconds.
The canonical combined ledger is:

```text
local_outputs/local_17scene_benchmark_20260718_v2/
17/17 passed, 0 failed
total retained-batch runtime: 5186.252711 seconds (86.44 minutes)
materialization plus retained batch: 5859.695744 seconds (97.66 minutes)
```

`local_outputs/local_17scene_benchmark_20260718_v1` is retained but is not
canonical. Its derived metadata placed the selection-plan SHA in the
manifest-SHA field. The scene outputs themselves were not modified. Version 2
points to the executed manifest and its verified hash.

The canonical statistical audit is:

```text
local_outputs/local_17scene_benchmark_audit_20260718_v2/
decision: promote_learning_guided_ransac_final
```

Primary local result:

| Metric | B0 ordinary RANSAC | B4 learning-guided RANSAC | Paired change |
|---|---:|---:|---:|
| pairwise F1, mean | 0.636284 | 0.706531 | +0.070247 |
| pairwise F1, median | 0.569644 | 0.659044 | median delta +0.046451 |
| purity/completeness F1, mean | 0.719681 | 0.796301 | +0.076620 |
| matched IoU, mean | 0.517580 | 0.677549 | +0.159968 |
| plane precision, mean | 0.381373 | 0.558226 | +0.176853 |
| recall over all GT planes, mean | 0.369177 | 0.559513 | +0.190336 |
| coverage, mean | 0.998283 | 0.999600 | +0.001317 |
| normal error, mean, degrees | 9.124512 | 4.291560 | -4.832952 |
| overmerge excess, mean | 2.647059 | 1.470588 | -1.176471 |
| runtime, median, seconds | 91.816 | 108.926 | ratio 1.148833 |

B4 wins pairwise F1 on 16/17 scenes. The only local negative is
`scene_00187`: 0.802746 to 0.800094, delta -0.002652. This differs from the
older archived/server result and must be reported as a separate local cache
realization. The exact two-sided sign-test p-value is 0.000274658. The paired
scene-bootstrap mean delta is +0.070247 with exploratory 95% interval
[+0.038461, +0.106371] using 10,000 resamples and seed 20260716.

The Phase 2 quality gate passes: median F1 improves, B4 wins 94.12% of scenes,
coverage remains effectively complete, and multiple confirmatory metrics
improve. The efficiency gate fails: the median runtime ratio is 1.148833 and
does not meet the frozen <=0.75 rule. Retain a quality claim only; do not claim
that guided RANSAC is faster.

Conflict-drop rows remain negative controls. Median full-cache coverage falls
to 0.014761 for manual conflict-drop and 0.000012 for direct conflict-drop.
They must not replace the primary no-drop path.

### Local GPU conclusion

The five-view DUSt3R smoke reserved 2866 MiB, and the backbone plus frozen
Stage1/Stage2 head smoke reserved 3004 MiB. The sequential 17-scene run
completed without OOM. RTX 4060 Ti 8 GB is therefore verified for batch-size-1
512-resolution inference, five-view global alignment, cache-only ablations,
failure analysis, and a resumable 100-scene frozen benchmark from a VRAM
perspective. Runtime and immutable-cache disk use are the constraints. A naive
100-scene extrapolation from the measured materialization and retained-batch
rate is about 9.6 hours, with substantial scene-dependent variance.

This does not make end-to-end DUSt3R or Plane-DUSt3R training a practical
local workload. Those training runs remain server-only.

The global-alignment diagnostic was visually inspected:

```text
local_outputs/rtx4060ti_dust3r_smoke_scene00180_20260718_v1/scene_00180_445895_global_alignment_diagnostic.png
```

Its three viewpoints show a consistent aligned room shell, wall/ceiling
structure, and window region. The plot is sparse and is only a global-frame
sanity check, not a final reconstruction-quality visualization.

### Windows compatibility changes used by the completed run

The local run exposed Windows/SciPy/NumPy instability and MAX_PATH issues in
the retained inference/evaluation path. The current uncommitted compatibility
changes replace affected plane eigensystem fits with PyTorch or OpenCV
eigensolvers, replace problematic matrix-style dot/norm operations with
explicit elementwise reductions, and use deterministic hashed Stage3 output
names with a 64-character maximum. Relevant files are:

```text
train_stage2_region_merge_net.py
export_stage2_learned_region_merge_editables.py
export_stage3_scene_plane_fusion.py
build_structured3d_point_aligned_gt.py
global_plane_baselines.py
guided_plane_ransac.py
extract_structural_lines.py
evaluate_global_plane_baselines.py
evaluate_support_record_partitions.py
tests/test_stage2_plane_fit.py
```

`audit_research_practice_final_results.py` also now renders the actual scene
count in Markdown instead of the hard-coded phrase `eight scenes`. The v2
audit was regenerated after this fix; the numerical result is unchanged.

Focused validation in `ai` after the completed run:

```text
python -m py_compile: 12 changed source/test files passed

test_audit_research_practice_final_results.py: 8 passed
test_stage2_plane_fit.py: 4 passed
test_stage3_scene_plane_fusion_mapping.py: 5 passed
test_structured3d_point_aligned_gt.py: 3 passed
test_global_plane_baselines.py: 5 passed
test_guided_plane_ransac.py: 3 passed
test_extract_structural_lines.py: 7 passed
test_evaluate_global_plane_baselines.py: 7 passed
test_support_record_partitions.py: 5 passed
test_evaluate_guided_ransac_smoke.py: 4 passed
total: 51 tests passed
```

The first combined unittest command imported the `ai` environment's unrelated
Ultralytics `tests` package and failed while reading its AppData settings. The
same ten repository test files were then run with `unittest discover -s tests`
and all 51 passed. This was an environment package-name collision, not a
project test failure.

The current HEAD remains
`6cf9f82e6c8df83fde011f3c99c0533702698e10`. No commit was created. The worktree
already contains extensive unrelated user changes and outputs; preserve them.

### Next exact experiment package

Do not rerun B0/B4 merely to obtain the same table. Use the immutable local
caches to run the remaining Phase 2 package:

1. B0-B5 and O1/O2 component rows;
2. three frozen RANSAC seeds;
3. proposal, consensus, refit, provenance, and conflict-handling controls;
4. a failure gallery centred on `scene_00187`, plus the largest positive and
   slowest scenes;
5. freeze the selected configuration before a 10-scene official-split canary
   and at least 100 eligible Structured3D validation/test scenes.

The 17 scenes remain an internal development set. They support an engineering
and research-practice conclusion, not a public benchmark or broad
generalization claim.

## 2026-07-19 completed 17-scene guided-RANSAC mechanism ablation

The historical B4 implementation was traced before adding component switches.
It uses learned support for proposal generation and learned-candidate
consensus scoring, but its final refit uses ordinary DUSt3R-confidence weights
over global inliers. Historical B4 is therefore the `proposal_consensus` mode,
not the new `all` mode. `guided_plane_ransac.py` now exposes six explicit,
schema-versioned modes: `none`, `proposal_only`, `consensus_only`,
`refit_only`, `proposal_consensus`, and `all`. Support-weighted refit multiplies
the selected coherent-label inlier weights by `1 + support_refit_weight`
(default 1.0) and does not discard points.

`execute_guided_ransac_mechanism_ablation.py` runs these modes from the
canonical immutable batch ledger, validates input and reused-output SHA-256
hashes, checkpoints every completed mode, evaluates against the same GT, and
checks historical B4 semantic equivalence. Recovery runs write new output
directories and reference checksum-valid prior ledgers rather than
overwriting them. The canonical recovered run is:

```text
local_outputs/guided_ransac_mechanism_17scene_seed0_20260719_v3/
source: local_outputs/local_17scene_benchmark_20260718_v2/batch_execution.json
17/17 scene-seed items passed
102/102 mechanism executions passed
0 failed items
17/17 historical B4 equivalence checks passed
executor runtime: 8634.643462 seconds (143.91 minutes)
metric rows: 119 (102 mechanism + 17 archived B0)
```

The first two long Windows runs stopped when the front task was interrupted.
Their completed immutable outputs are retained in v1 and v2. V3 checksum-
validated and reused 33 completed mode outputs, then finished the remaining
work. No cache, NPZ, checkpoint, or visualization was overwritten.

Primary seed-0 mechanism result:

| Mode | Mean pairwise F1 | Median | Mean runtime, s |
|---|---:|---:|---:|
| matched none | 0.635813 | 0.569485 | 101.290 |
| proposal only | 0.707222 | 0.658716 | 105.668 |
| consensus only | 0.635813 | 0.569485 | 95.498 |
| refit only | 0.635643 | 0.576892 | 114.794 |
| historical B4, proposal + consensus | 0.706531 | 0.659044 | 103.685 |
| all | 0.706582 | 0.659198 | 111.492 |

`analyze_guided_ransac_mechanism_ablation.py` performs deterministic paired
scene bootstrap, exact two-sided sign tests, Holm correction over the six
planned F1 contrasts, and Wilcoxon sensitivity tests. Its immutable output is:

```text
local_outputs/guided_ransac_mechanism_statistics_17scene_seed0_20260719_v1/
10,000 bootstrap resamples, seed 20260719
```

Proposal-only versus matched none gains +0.071410 mean F1 and +0.047196 median
F1, wins 16/17 scenes, has bootstrap 95% interval [+0.038900, +0.106941], and
Holm-adjusted sign-test p=0.001648. Consensus-only is exactly identical to
none because consensus has no learned candidates without proposal guidance.
Adding consensus after proposal changes mean F1 by -0.000691 with interval
[-0.002226, +0.000103] and Holm p=0.067383. Refit-only changes mean F1 by
-0.000169 with interval [-0.004269, +0.003653]. Adding support refit after B4
changes mean F1 by only +0.000051; despite 15/17 tiny positive changes and
Holm p=0.009399, its interval [-0.000003, +0.000095] touches zero and the
effect is practically negligible.

Decision: learned proposal generation accounts for essentially all retained
gain. Keep historical B4 for backward compatibility; do not promote `all`.
Report proposal-only as the decisive component row. This is still a
single-seed internal result; add seeds 1 and 2 before closing the robustness
family.

Added or changed for this package:

```text
guided_plane_ransac.py
execute_guided_ransac_mechanism_ablation.py
analyze_guided_ransac_mechanism_ablation.py
tests/test_guided_plane_ransac.py
tests/test_execute_guided_ransac_mechanism_ablation.py
tests/test_analyze_guided_ransac_mechanism_ablation.py
docs/codex_tasks/experiment_benchmark_plan_4060ti.md
docs/codex_handoff/CURRENT_STATE.md
```

Validation in `E:\anaconda\envs\ai\python.exe`:

```text
py_compile for the new analysis script: passed
test_guided_plane_ransac.py: 7 passed
test_execute_guided_ransac_mechanism_ablation.py: 4 passed
test_analyze_guided_ransac_mechanism_ablation.py: 2 passed
total focused mechanism tests: 13 passed
```

The `ai` environment has no pytest. A module-name unittest attempt also
resolved the environment's unrelated Ultralytics `tests` package. Repository
tests were therefore executed with `unittest discover -s tests`, and all 13
focused tests passed. These setup failures are not project test failures.

RTX 4060 Ti conclusion is unchanged: this cache-only ablation completed
locally without OOM and is primarily CPU/runtime bound. The 8 GiB card is
adequate for these ablations and the verified sequential inference path, not
for practical end-to-end training.

Current HEAD remains `6cf9f82e6c8df83fde011f3c99c0533702698e10` and the
worktree remains dirty with extensive pre-existing user changes. No commit was
created. Preserve all unrelated changes and retained outputs.

Next exact package: run the same mechanism table for frozen seeds 1 and 2
using checksum-based recovery from the canonical seed-0 ledger where valid,
then average seeds within scene. In parallel only after that decision table,
complete the remaining B0-B5/O1/O2, provenance/conflict controls, and the
`scene_00187` failure gallery. Do not start the official split before these
internal rows are frozen.

## 2026-07-19 three-seed mechanism robustness family completed

Seeds 1 and 2 are complete in addition to the seed-0 package above. Each seed
has 17/17 passed items, 102/102 passed mechanism executions, and zero failures:

```text
local_outputs/guided_ransac_mechanism_17scene_seed0_20260719_v3/
local_outputs/guided_ransac_mechanism_17scene_seed1_20260719_v1/
local_outputs/guided_ransac_mechanism_17scene_seed2_20260719_v3/
```

The seed-2 v3 recovery reused all checksum-valid predictions from v2 and
resolved the one transient empty-log evaluation failure. Its recovery runtime
was 71.429 seconds. Historical B4 equivalence is explicitly checked against
the archived reference only for seed 0 (17/17); seeds 1 and 2 have no archived
same-seed reference, so no equivalence claim is made for them.

`analyze_guided_ransac_mechanism_ablation.py` now accepts repeated aggregate
CSV inputs and averages stochastic seeds within each scene before cross-scene
bootstrap, sign, Holm, and Wilcoxon statistics. Its schema-v2 output is:

```text
local_outputs/guided_ransac_mechanism_statistics_17scene_seeds012_20260719_v1/
17 independent scene pairs; seeds 0, 1, and 2 averaged within scene
10,000 bootstrap resamples; seed 20260719
```

Three-seed means are 0.637278 for matched none, 0.702667 for proposal only,
0.637355 for consensus only, 0.637201 for refit only, 0.702452 for historical
B4, and 0.702504 for all. Proposal-only versus none gains +0.065389 mean
Pairwise F1, wins 16/17 scenes, and has bootstrap 95% interval
[+0.036570, +0.097369] with Holm-adjusted sign-test p=0.001648. Adding
consensus after proposal changes mean F1 by -0.000215 with interval
[-0.000742, +0.000073]. Adding support refit after historical B4 changes mean
F1 by only +0.000053. The mechanism decision is closed: learned proposal
generation accounts for essentially all useful gain; retain historical B4 for
compatibility and do not promote `all`.

Next exact package: complete the remaining B0-B5/O1/O2 rows, provenance and
conflict controls, and the `scene_00187` failure gallery. Then freeze the
internal decision table before the official-split canary.

## 2026-07-22 completed 17-scene B0--B5/O1/O2 component table

The remaining Phase-2 seed-0 component rows are complete on the canonical
immutable local 17-scene caches. New executors adapt point-aligned GT into an
exact support-guidance registry, generate the missing B1 Stage1-direct row,
run the O1 GT-support/frozen-geometry row, verify the B5 contract, checkpoint
every stage, and refuse output overwrite. Internal output paths use compact
numeric directories because descriptive Stage3 filenames otherwise exceed
the legacy Windows 260-character path limit. The initial long-path run is
retained; it failed before expensive computation on 16 scenes and did not
overwrite any artifact.

Canonical recovery:

```text
local_outputs/mca17_s0_final_20260722_v1/
17/17 passed scenes
0 failed scenes
119 metric rows (17 x 7 evaluated quality rows)
31.178-second checksum-validated final consolidation

local_outputs/mca17_s0_statistics_20260722_v1/
10,000 scene bootstrap resamples, seed 20260722
six planned Pairwise-F1 contrasts with Holm correction
```

Mean/median Pairwise F1:

| Row | Mean | Median |
|---|---:|---:|
| B0 global RANSAC | 0.636273 | 0.568900 |
| B1 Stage1 direct SVD | 0.127858 | 0.123777 |
| B2 Stage2 local refit | 0.133097 | 0.131549 |
| B3 frozen manual merge | 0.721490 | 0.727245 |
| B4 guided RANSAC | 0.706568 | 0.658663 |
| O1 GT support, frozen geometry, seed 0 | 0.719581 | 0.710929 |
| O2 GT identity | 1.000000 | 1.000000 |

B4-B0 is +0.070295 mean F1, wins 16/17 scenes, and has bootstrap
95% interval [+0.038396, +0.106903] with Holm-adjusted sign p=0.000824.
O1-B4 is only +0.013013 with interval [+0.006234, +0.021602], so learned
support is close to perfect-support performance under frozen geometry. The
O2-O1 gap is +0.280419, identifying identity/association as the larger
remaining ceiling.

B4 does not dominate B3: B4-B3 is -0.014922 mean F1, interval
[-0.091005, +0.058917], and W/T/L 8/0/9. B4 instead produces cleaner, less
fragmented instances: mean precision 0.5582 versus 0.2882 and fragmentation
excess 1.2941 versus 4.7059. Report this trade-off and do not claim universal
superiority over manual merging. B2-B1 is a small +0.005239, whereas B3-B2 is
+0.588393; association, not local refit alone, is decisive.

B5 aliases B4 for quality metrics and adds system-output contracts. For all
17 scenes, prediction checksums, exact registry, bounded-component
assignments, and structural lines were present. The structural-line total is
3,100, ranging from 73 to 289 per scene.

Added files:

```text
build_gt_support_guidance.py
execute_main_component_ablation.py
analyze_main_component_ablation.py
tests/test_build_gt_support_guidance.py
tests/test_execute_main_component_ablation.py
tests/test_analyze_main_component_ablation.py
```

Validation in `E:\anaconda\envs\ai\python.exe`:

```text
py_compile for the three new scripts: passed
GT-support adapter tests: 2 passed
component executor tests: 2 passed
paired-statistics tests: 2 passed
total new focused tests: 6 passed
```

The `ai` environment has no pytest; tests were run with `unittest discover -s
tests` to avoid the environment's unrelated Ultralytics `tests` package.
Cache-only O1 was CPU/runtime bound (538.12-second mean) with roughly
100--175 MB child working sets. No OOM occurred; the RTX 4060 Ti 8 GB remains
adequate for these experiments.

Published implementation commits on `codex/bounded-support-head` are
`a464d1f` (`Complete main component ablation table`) and `63eb782` (`Fix
Windows final result rendering`). The branch was pushed through `63eb782` on
2026-07-22. The worktree has extensive unrelated user changes and outputs;
preserve them. Only the files listed above plus this handoff, the experiment
plan, and the separately tested visualization fix belong to these commits.

Next exact package: run the remaining provenance/conflict controls and build
the failure gallery centred on `scene_00187` plus the largest positive and
slowest scenes. Then freeze the internal decision table before an official-
split canary. Official validation/test data are not present locally, and the
17 current scenes remain an internal development set rather than a public
benchmark.
