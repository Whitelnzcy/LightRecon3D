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
