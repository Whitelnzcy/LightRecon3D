# LightRecon3D Current State

Last updated: 2026-07-14

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
