# LightRecon3D Current State

## 2026-10-25 seed43 replicated; boundary-balanced loss trial active

Base HEAD113a94a. Same fixed direction propagation route. Seed43 zero-none
227819/56999/18583,total/boundary/damage; zero-learned226239/59132/14156;
inward-none224366/55922/16699; inward-learned224373/56758/17965.
Inward target boundary benefit repeats, but damage benefit does NOT; propagation
still worse than matched inward-none. Both seed43 runs independently audited.
Represented/editable boundary direction median TRAIN0=88.42deg, DEV19980.60,
DEV19089.25 after inward training (fixed GT-free overlapping tile sample).
New single-factor equal-mean boundary/interior direction loss, coefficient.3,
same everything else, none/learned800steps seed42 RUNNING:
local_outputs/direction_balanced_20261025_s42. 22direction tests passed.
Details docs/research_practice/DIRECTION_REPLICATION_20261025.md. No mainline
promotion; no GT in prediction. Next audit balanced result and boundary quality.

## 2026-10-25 learned inward supervision completed and audited

Implementation checkpoint e0df056. Same direction propagation route. Inward TRAIN
targets,none/learned,800steps,seed42,grid1,bias2,agreementOFF,strict deterministic.
Full DEV total/boundary/damage: inward-none227704/55735/8768;
inward-learned227458/56066/8940. Prior zero-none229797/55926/12620;
zero-learned237959/58456/19762. Target change helps learned vsold, but propagation
still adds331 boundary errors vsmatched inward-none. No mainline promotion.
19tests pass; full audit4results passes, including pre-GT prediction equality.
Four-way figures,error maps,global XYZ projection generated; figures inspected.
Run local_outputs/direction_inward_learned_20261025_s42. All training stopped
normally; no active job. Details docs/research_practice/DIRECTION_INWARD_LEARNED_20261025.md.
Next boundary-vs-interior direction quality and second-seed paired replication;
not executed. Preserve fixed research direction and historical artifacts.

## 2026-10-25 inward oracle audited; learned supervision control active

Base HEADb8422c9. Quota interruption left boundary-target work local; user resumed.
Oracle800 zero target224727/57896/14991 total/boundary/damage; agreement oracle
230721/56994/17785. Edge audit:90.75%/88.40% cross-plane grid edges have a zero
endpoint target. Inward completion only at GT boundary preserves interior targets;
cross-edge raw weights0.791/0.773 ->0.424/0.466, same-plane~0.927/0.931.
Inward oracle800 complete232596/56312/10421, independent audit passed today.
Not deployable, not superior overall to no-propagation800. 19direction tests pass.
New paired none/learned inward TRAIN supervision,800steps,seed42,strict deterministic
RUNNING: local_outputs/direction_inward_learned_20261025_s42. No GT in prediction.
Compare with saved zero-target none800/learned800, same inputs/budget/candidates.
Audit now checks ordinary predictions exactly match pre-GT saved arrays. All prior
outputs preserved; no Stage2/3/XYZ edits. Details DIRECTION_BOUNDARY_ZERO_20261024.md
under docs/research_practice; task direction_inward_learned_20261025.md.

## 2026-10-24 deterministic bias and budget trials completed; oracle800 active

Base HEAD6b0bcc3. Fixed direction-propagation line. Bias0/1/2 seed42 and43
completed/audited; bias1 lowers damage vsbias2 but boundary ranking reverses.
No universal bias winner. Both200step duplicates exact each step/full predictions.
Budget trial grid1,bias2,seed42: DEV total/boundary/damage none200255502/62047/6936;
learned200239788/61509/19152; none800229797/55926/12620;
learned800237959/58456/19762. Longer training helps but direction800 loses to
none800. TRAIN800 totals621495vs621444, damage20424vs30032. Direction problem
not only DEV generalization. Cross-process learned200 matches bias2 bitwise.
Independent DEV audit8results passed;15tests pass. Error/change figures generated.
Oracle800 same protocol currently RUNNING in direction_oracle800_20261024_s42.
GT-assisted diagnostic only, not deployable/upper bound; no mainline replacement.
See docs/research_practice/DIRECTION_BUDGET_20261024.md and prior sensitivity report.

## 2026-10-24 direction sensitivity and reach trials; bias run active

Base HEADb540791. User requests continuous progress on SAME direction line.
Frozen sensitivity: boundary feature relative change1.98/2.25%; zero-direction
label flips0.067/0.048%; removing support bias flips52.9/62.2% (NOT accuracy).
Reach1/4/8 seed42 boundary62788/63405/63238; seed43 64070/63632/63884.
No consistent wider-reach benefit; keep grid1. Both full-domain runs audited.
Strict deterministic replay30steps passed per-step parameter and full prediction
equality; no efficacy claim at30steps. 15 related tests passed. Bias0/1/2 200step
deterministic seed42 experiment RUNNING with duplicate bias2 equality assertions.
Outputs direction_sensitivity_20261024_v1, direction_reach_20261024_s42/s43,
direction_replay_20261024_v1, direction_bias_20261024_s42 under local_outputs.
See docs/research_practice/DIRECTION_SENSITIVITY_REACH_20261024.md.
Next audit bias results and repeat evidence; no mainline promotion yet.

## 2026-10-24 matched direction propagation control completed

Base HEAD268c5e3. User explicitly fixed direction-propagation research line.
TRAIN4,DEV2,seed42,200steps,identical initialization/batches; none/spatial/learned/
oracle-trained direction. Total/boundary/damage: none270929/63883/4293,
spatial299439/63847/2023, learned263904/62676/5392, oracle301963/63729/1554.
Learned saves1207 boundary errors vsnone but adds1099 damages. Oracle is GT-only
diagnostic, not deployable/upper bound. No mainline replacement or robust claim.
12 direction tests pass; independent audit verifies all8 scene/arm metrics and
source mappings. Both figures inspected; fragmented/ragged boundaries remain.
Global XYZ projection audit generated. Outputs direction_control_20261024_v1
under local_outputs; details docs/research_practice/DIRECTION_CONTROL_20261024.md.
Next within SAME line: feature/logit sensitivity diagnostics, then one-factor
neighborhood/bias ablation; not yet executed. Preserve all historical outputs.

## 2026-10-24 direction memorization and frozen intervention completed

Base HEAD70c9639. Four TRAIN patches, 200 direction-only updates on copied DS/DC
weights: median angle 76.75->3.92 / 59.58->3.33 degrees. Memorization only;
overfit weights never used on DEV. Original frozen models tested with learned,
GT-oracle, zero and random directions on historical DEV2. Boundary errors:
DS62900/62884/62916/62910; DC62889/62863/62887/62887. Weak dependence on
direction; do not scale training or promote to mainline on this evidence.
Oracle is GT-assisted diagnostic, NOT an upper bound or deployable result.
Run complete; 2 focused tests passed; independent audit verified all16 full-domain
results and mappings. Comparison figures and global XYZ projection generated.
Outputs local_outputs/direction_mechanism_diagnostic_20261024_v1. Details:
docs/research_practice/DIRECTION_MECHANISM_DIAGNOSTIC_20261024.md.
Next: controlled no/spatial/oracle-trained propagation comparison; not run yet.

## 2026-10-24 learned3D direction pilot completed, no conditioning gain

Base HEAD5352564. Added direction inputs/supervision, D0/DS/DC spatial decoder,
runner --direction-pilot and audits. FixedTRAIN4,200steps,seed42,64px,batch4;
same ownership loss+.3directionMSE, frozen RGB boundary. Prediction uses noGT.
PushBoundary-inspired grid propagation with3D offsets, NOT original reproduction.
v1 lackedXYZ at direction head; preserved. v2 addsXYZ3 all arms, same budget.
Global basis not rotation-equivariant; v1/v2 layer RNGs differ due width change.
Output local_outputs/direction_train4_dev2_20261024_v2 (v1 diagnostic retained).
DEV total/boundary/damage: D0299427/63845/2026; DS271025/62900/2714;
DC273264/62889/3988. DC saves11 boundary errors vsDS but adds1274damages: no
useful gain demonstrated. Do not replace mainline. TRAIN0 fixed80patch direction
median errors69.65/71.97/64.99deg; weak fit, norms0.1285/0.1264/0.1925.
Both runs exit0,42 tests pass; mappings/sources/fullDEV metrics audited. Six
figures/run; bothv2DEV comparisons viewed; projection audit generated. Details
docs/research_practice/DIRECTION_PILOT_20261024.md. No independent test claim.
Next small overfit + oracle direction diagnostic before scaleup; not run yet.

## 2026-10-24 PushBoundary target audit and propagation operator

Base HEAD5e2cd7f. User approved borrowing method and testing a small change.
Read paper/official preprocessing: semantic k4 boundary; boundary direction zero.
Added evaluation-only instance-target audit (global vs same-plane nearest
boundary) and differentiable direction-weighted propagation operator; NOT yet
integrated/trained. FullDEV1361282 points: wrong-near-boundary eligible52825,
5040(9.54%) differ>30deg; no proof same-plane target improves reconstruction.
00190:65592 wrong points missing matched candidate; fixed pool cannot repair.
Six tests pass, audit exit0; target diagnostic figure generated/viewed. Sources,
caveats/commands in docs/research_practice/PUSHBOUNDARY_TARGET_AUDIT_20261024.md.
Output local_outputs/pushboundary_target_audit_20261024_v1. No XYZ/ID changes.
Next same-target/same-capacity shared vs candidate-conditioned direction pilot,
plus no-direction baseline; change target definition separately, no GT inference.

## 2026-10-24 finite 3D support diagnostic completed

Base HEADd4bc48b. User says learned RGB boundaries look accurate; inspected3D
correspondence before further training. New diagnose_finite_structure_lines.py
keeps global p+t*v, clips t by support overlap, records continuous distances.
No GT/XYZ/ownership edits. Outputs finite_3d_line_diagnostic_20261024_v1/v2.
Two historicalDEV scenes:8+37 sampled lines; fixed2% scene-diagonal BOTH-support
radius rejects ALL. This heuristic is NOT suitable as a mandatory repair gate;
does not disprove RGB prior or true contacts. Pair0/1 on199: one support0.00056
away, other0.02913(model units), bandwidth0.01032; equations consistent. Gap
cause (ownership/visibility/geometry/noncontact) not yet resolved. No threshold
widening or successful extraction claim.3 tests+syntax pass; runs exit0;3D and
projection diagnostics generated. See FINITE_3D_LINE_DIAGNOSTIC_20261024.md in
docs/research_practice and task finite_line_diagnostic_20261024.md.
Next: independent RGB evidence + continuous pair-conditioned geometry, learn
finite ownership without requiring existing supports already touch. Not trained.

## 2026-10-24 independent boundary / raw geometry ablation completed

Base HEAD0186271. User questioned circular plane-derived constraints; prioritize
independent RGB-prior experiment over proposed keep/reassign head (not added).
Old M3 lines already included RGB refinement, NOT a pure geometry control.
New N/B/G/BG: neither/RGB boundary/raw intersections/both; same active architecture,
TRAIN4,200steps,batch4,seed42,64px,loss and observed-pixel application.
Frozen RGB-only model inferred all30 views; raw lines contain no RGB correction.
Additive evidence over repeated baseline channels is an explicit pilot limitation.
Output local_outputs/independent_evidence_train4_dev2_20261024_v1.
DEV total/boundary/damage:
N254364/62609/11205; B257236/62344/9208;
G243255/60413/10525; BG241555/60051/11244.
BG boundary4.09% below N, but not uniform per-scene gain and worse than historical
RGB-refined M3 boundary51831. No mainline replacement or broad efficacy claim.
Upstream TRAIN4=train_scenes; DEV199/190=historical test_scenes, already repeatedly
inspected downstream; NOT independent evaluation. 32 tests pass;305 rawlines and
all DEV metrics verified; source/row/candidate invariants pass. Input/output and
global projection figures generated and viewed. Training exited0,43.154s loop.
See docs/research_practice/INDEPENDENT_BOUNDARY_GEOMETRY_20261024.md and task
docs/codex_tasks/independent_boundary_geometry_20261024.md for commands/caveats.
Next: separate evidence representation and learn finite pair-specific ownership;
not more geometric self-confirmation, not yet implemented. No long training.

## 2026-10-24 context audit and gate-supervision trial completed

Base HEAD05df388, same scoped branch. Continued interrupted 2026-10-23 run;
completion.json/checkpoints/predictions verified, did NOT repeat context training.
Old process session expired; original context exit code is no longer retrievable.
See docs/research_practice/LINE_SUPPORT_CONTEXT_PERMISSION_20261024.md.
Added spatial_context_data.py, TRAIN damage diagnosis, permission auxiliary
loss/runner/tests; extended existing runner with context-margins. Defaults
preserve old64 behavior. Original241 deletions/7 modifications left alone.

Context: MC/M3 x64/128 input, same64 supervised/output centers and batch sequence,
TRAIN4,200steps each. Zero halo padding outside image; explicit registry unchanged.
Larger input also changes GroupNorm/padding behavior; NOT equal FLOPs.
Historical DEV total/boundary/damage:
MC64 254366/62610/11209; M364 206658/51831/17418;
MC128 258927/62045/12994; M3128 196868/51886/16394.
128 improves total/damage but NOT aggregate boundary. Scene199 boundary improves
12151->11441 while190 worsens39680->40445. No universal improvement claim.
Output local_outputs/spatial_context_train4_dev2_20261023_v1.
Train86.061s,allocated955.84MiB,reserved1276MiB; excludes prep/eval/CUDA overhead.

TRAIN diagnosis: M364 damages49592, of which41447 agree with predicted line side,
42124 lie in proposed training centers,23589 are>20px from GT boundary. Line-side
agreement is insufficient, but these associations do not prove causal attribution.
Second pre-specified modification: M3P adds0.1 balanced candidate gate BCE only;
same64 model/inputs/batches/200steps, no sweep or inference GT. DEV M3P
212455/53355/18645 vs matched M3206657/51831/17418: REJECT, no mainline update.
Output local_outputs/spatial_permission_train4_dev2_20261024_v1; train12.174s.

Both run audits exactly reproduce full DEV metrics and230 TRAIN line records.
TRAIN diagnosis recomputes all TRAIN metrics. Final29 unit tests+2 projection
checks and8-file syntax checks pass.12 comparison figures+1 projection figure
generated; viewed4 DEV comparisons and projection diagnostic. Original XYZ,
row order,unassigned labels and source hashes preserved. Two targeted tweaks
(context,auxiliary gate loss) now closed; no further loss-weight/context sweep.
Next proposal: explicitly normalized keep-versus-reassign decision, not calling
a multiplier on signed logits an edit probability. Needs a separately frozen
mechanism experiment with matched no-line control; NOT implemented or proven.
Independent split,nearest-work comparison,multiseed and17-scene current-model
comparison remain pending. No claim of completed goal or Plane-DUSt3R parity.

## 2026-10-23 active-capacity and scalar controls completed

Base HEAD a2f562b on codex/line-support-refinement-20261023. Scoped continuation;
prior archival deletions/modifications untouched. No large training or recache.
Extended spatial_line_refiner with MC (same active architecture as M3, no line
input) and MS (single learned scalar line gate). Same TRAIN4/1920 patches,
64px/batch4/seed42, 200steps each MC/MS/M3. All inference frozen before DEV2.
Output: local_outputs/spatial_line_controls_train4_dev2_20261023_v1.

Historical DEV total/boundary/damage: MC254367/62609/11208,
MS222440/55895/17184, M3206657/51831/17418. MS scalar0.542879.
MC is genuinely line-independent and trains the extra branch; applicability
mask differs intentionally (observed pixels vs predicted line-pair mask).
M3 DEV predictions reproduce earlier metrics; TRAIN differs by1 point due to
non-deterministic CUDA floating paths. Do not claim bitwise determinism.

Frozen M3 gate intervention in gate_intervention/: calibrate a constant ONLY
on TRAIN available tiled pixels (0.785678), replace gate without retraining.
Constant total/boundary/damage217184/54505/13531 vs spatial206657/51831/17418.
Spatial gate reduces2674 boundary errors but introduces3887 extra damages.
This supports gate-dependent behavior, not proof of learned finite endpoints,
causal optimality, independent generalization, or superiority to Plane-DUSt3R.
Historical finite still has lower damage8087; do not replace mainline.

Execution: three-arm training39.520s (excludes prep/eval), allocated279.27MiB,
reserved496MiB (not whole GPU usage). 20 unit tests +2 projection checks pass;
syntax checks pass. Independent audit recomputes all DEV metrics and230 TRAIN
line provenance records exactly. Generated6 comparison figures and1 two-view
global XYZ projection diagnostic; viewed both DEV comparisons and diagnostic.
Projection disjoint P95s0.1230/0.1085px: coordinate consistency, NOT depth accuracy.
Commands, hashes and per-scene results: LINE_SUPPORT_PROGRESS_20261023.md.
Next: controlled64/128 context comparison and TRAIN-based protection analysis;
change one factor per experiment. Independent split/nearest-work/multiseed pending.

## 2026-10-23 spatial M1 and first M1/M2/M3 pilot completed

Base commit 16836b104fda0ee68632ce72fb3adb66c4cbbad5; same scoped GitHub branch.
Implemented spatial_support_data.py, spatial_support_refiner.py and plain M1
training/evaluation, then spatial_line_features.py, spatial_line_refiner.py,
train_spatial_line_ablation.py and audit_spatial_line_pilot.py. See task and
docs/research_practice/LINE_SUPPORT_PROGRESS_20261023.md for commands/details.

Actually ran TRAIN4/20views/1,920 GT-free64px patches, 200steps/batch4. Plain
M1 fit errors829570->748464, fixed-probe loss2.393909->1.369852. Then ran all
M1/M2/M3 arms with same initialization, batches/loss/budget, each200steps.
M2 fixed line weight0.5; M3 learned local weight initialized0.5; gates affect
candidate logits and receive nonzero gradients. XYZ and candidates unchanged.

Frozen historical DEV2 totals (total/boundary/damage):
M1 268232/62705/6261; M2 224857/56362/17231; M3 206657/51831/17418.
Historical finite 213028/52014/8087. M3 gives a pilot increment over M2, but
only183 fewer boundary errors than historical finite and9331 extra damages.
DO NOT replace old mainline or claim independent generalization/publication.
M1 has disabled extra branch: effective capacity is not fully matched.
M3 mean gate~0.78 could partly be global strength adjustment: scalar-gate
control is required before claiming learned finite spatial applicability.

Outputs: local_outputs/spatial_support_m1_train4_20261023_v1,
spatial_support_m1_dev2_20261023_v1, spatial_line_m123_train4_dev2_20261023_v1.
All execution entries exited0; 12comparison images generated, sampled4 viewed.
Final checks:18 unit tests plus2 projection checks passed; syntax passed.
Postrun audit verifies230 TRAIN line records and all DEV metrics/source order.
Scene190 M3 has6126 damages >20px from GT boundary, concentrated in few pairs.
Next: capacity-matched no-line and scalar-gate controls; TRAIN damage diagnosis,
then targeted protection supervision. No parameter sweep or big retraining yet.
Independent split/full upstream audit and nearest-work comparison still pending.

## 2026-10-23 line-conditioned support refinement: P0/P1 pilot

User authorized implementation and GitHub progress synchronization. New branch:
codex/line-support-refinement-20261023, base HEAD e2c058f. No new training yet.
Task: docs/codex_tasks/line_conditioned_support_refinement_20261023.md.
Detailed execution record: docs/research_practice/LINE_SUPPORT_PROGRESS_20261023.md.

Added audit_line_support_refinement.py, models/line_support_patches.py and
tests/test_line_support_patches.py. Ten interface tests, two projection checks
and syntax compilation pass. Executed audit into
local_outputs/line_support_audit_20261023_v1 (exit 0): both DEV scenes' saved
control/support metrics exactly reproduced; all ten view transforms, point
registries, dense GT and 5px boundary distances verified. DLT from predicted
XYZ/pixels gives disjoint-check P95 0.1085-0.1316px. Generated 10 diagnostic
figures and 80 prediction-only patch indices; inspected both view0 figures.
No GT labels/camera/layout paths in exported prediction input metadata.

Old scene199 residual errors are nearly all within20px of GT boundary (5 outside).
Scene190 retains65,592 fixed-map-unrepresented points,1,390 preserved unassigned,
31,707 represented errors within5px and95,398 beyond5px (21,940 beyond32px).
Missing matched identity is NOT evidence of physically missing geometry.
All21 available guided item caches are in frozen boundary model's historical
train/dev/test protocol (4/8/9). Independent final test remains unresolved;
Stage1/full upstream split audit is pending. No fresh model gain claimed.

Next: upstream split/nearest-work audit, TRAIN4 patch dataset and spatial M1
fit/profile, then controlled M2/M3; no automatic large recache/training.
Original dirty worktree/outputs preserved. Sync only scoped code/tests/task/
progress documentation, not datasets, checkpoints or unrelated archival changes.

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

## 2026-07-24 Plane-DUSt3R local five-view smoke passed

The pinned official Plane-DUSt3R repository is available at
`E:\Study\code\Plane-DUSt3R`, commit
`9a1ae50650ec6d706bf329352aaaf49efded90a0`. Both required checkpoints are
present and loadable:

```text
checkpoint-best-onlyencoder.pth
bytes 4411401351
sha256 54a0015b2df35de5ca6eeabae114f8e6139c29ed56d5db5eaa380d7c74dbfa56

Structured3D_pretrained.pt
bytes 137254828
sha256 a0e576567ca5f1632fb06c82258f1e05fe41c0ccdd495a84a11ffbb737629384
```

The user-required local `ai` environment is Python 3.12.11 with
PyTorch 2.9.1+cu126. `easydict==1.9` and `tensorboardX==2.1` were installed.
An attempted `mmcv==1.2.1 --no-deps` source build failed because Microsoft
C++ Build Tools were unavailable and briefly overlaid NumPy 2.4.3. NumPy was
immediately restored and verified as 2.2.6; Numba 0.63.1, PyTorch and CUDA
remain importable. MMCV was not installed.

`run_plane_dust3r_same_input_smoke.py` now stages an isolated runtime without
changing the official repository. The runtime provides only the three MMCV
APIs imported by NonCuboidRoom, allows `argparse.Namespace` through PyTorch's
weights-only safe loader, isolates legacy TensorBoard protobuf behavior,
copies code when Windows symlinks are unavailable, forces NonCuboid
`num_workers=0` on Windows, and replaces six POSIX-only path parses in the
runtime copy of `metric.py`. The wrapper can checksum and reuse completed
DUSt3R or room-level intermediate outputs. Official tracked-file changes still
fail the worktree audit; only checkpoints and generated `__pycache__` entries
are allowed.

Validation in `E:\anaconda\envs\ai\python.exe`:

```text
py_compile: passed
tests/test_run_plane_dust3r_same_input_smoke.py: 10 tests passed
both models loaded on RTX 4060 Ti with strict key matching
static model allocation: 2318.9 MiB
static model reservation: 2428.0 MiB
five-view DUSt3R inference/global alignment: passed without OOM
sampled total GPU use during inference: 5421 MiB; 2529 MiB free
```

The first complete same-input smoke passed for
`scene_00180 / room 445895 / positions 0--4`. These are the same five
`perspective/empty` images used by the frozen LightRecon3D experiment,
materialized as an explicitly non-native `perspective/full` compatibility
view. The official `convert_plane_depth.py` generated five evaluation depth
maps in the copied dataset. Canonical successful output:

```text
local_outputs/plane_dust3r_same_input_smoke_scene00180_20260724_v7/
status pass
return code 0
```

Five-view mean native metrics are:

```text
layout IoU             0.5971863801
pixel error            0.1214576226
edge error             7.7542636513
depth RMSE             0.0366002411
scale-aligned RMSE     0.0366833496
plane precision        1.0000000000
plane recall           0.4444444444
```

Generated artifacts include `metric_results.txt`, `dust3r_output.npz`,
`plane_detection.json`, `node_data.json`, `layout.png`, and five
`seg_pred_vis_*.png` images. Intermediate failed outputs v1--v6 are retained
as audit evidence and were not overwritten.

These numbers are a one-room same-input reproduction, not Plane-DUSt3R's
paper-native Structured3D test result and not yet a common-output ranking
against LightRecon3D. Next exact step: promote the Windows-safe runtime and
artifact-reuse behavior into a batch executor, run all 17 frozen rooms, then
report Plane-DUSt3R native layout metrics separately from any defensible
common-point-cache adapter.

## 2026-07-24 Plane-DUSt3R 17-room same-input batch passed

The 17 frozen development scenes were materialized without changing the source
Structured3D tree:

```text
local_outputs/plane_dust3r_same_input_17scene_20260724_v1/
scenes 17
images 85
source render mode perspective/empty
compatibility declaration perspective/full
```

The official `convert_plane_depth.py` generated all 85 evaluation depth maps
in that copied dataset and completed with return code 0.

`run_plane_dust3r_same_input_batch.py` is the new one-process batch executor.
It validates the non-native protocol declaration, official repository commit
and worktree, stages the isolated Windows-safe runtime, loads both models once,
supports checksummed room-output reuse, and writes `batch_manifest.json` plus
`batch_metrics.csv`. The ledger independently re-parses every room's native
metric file and marks missing or malformed artifacts as failures. The Git
read-only audit now scopes `safe.directory` to each command, avoiding any
global Git configuration change.

Validation:

```text
py_compile: passed
batch tests: 4 passed
smoke tests: 11 passed
single-scene batch canary: 1/1 room and 5/5 views passed
```

Canonical complete batch:

```text
local_outputs/plane_dust3r_same_input_batch_17scene_20260724_v1/
status pass
return code 0
runtime 1014.6898072 seconds
rooms 17/17
views 85/85
```

`scene_00180` reused the verified prediction intermediates; the other 16 rooms
ran fresh inference. Observed RTX 4060 Ti use was approximately 6.3 GiB with
approximately 1.6 GiB free during inference, and there was no OOM.

The 17-room native metric aggregate is:

```text
layout IoU                         0.7763029385
pixel error                        0.0781947595
edge error                         7.1339249102
depth RMSE                         0.0749965016
scale-aligned RMSE                 0.0641348465
plane precision, room macro mean   0.5070028011
plane recall, room macro mean      0.4237278245
```

All rooms have one each of `dust3r_output.npz`, `plane_detection.json`,
`node_data.json`, `layout.png`, and `metric_results.txt`, plus five separate
`seg_pred_vis_*.png` files. `scene_00193` has the highest layout IoU
(`0.9636468778`); `scene_00197` is the clearest failure case with layout IoU
`0.3809978764` and plane precision/recall both `0`.

These remain Plane-DUSt3R native layout/depth metrics on the same-input
compatibility data. They are not the paper-native test protocol and must not
be put in the same numeric column as LightRecon3D's current point-cache plane
F1. The next defensible comparison task is a common output-domain
adapter/evaluator, while keeping this native table as a separate external
baseline result.

## 2026-07-24 common-point-cache external baseline evaluation passed

Plane-DUSt3R can be compared to the two LightRecon3D RANSAC baselines without
aligning their independently reconstructed 3D frames. All methods observe the
same five images at the same `288x512` pointmap resolution. The new adapter
uses the explicit key `(alignment_view_index,x,y)` to transfer
Plane-DUSt3R's global structural-plane identities onto the existing ordered
LightRecon3D cache. No spatial nearest-neighbour join is used.

Implementation:

```text
adapt_plane_dust3r_to_point_cache.py
evaluate_plane_dust3r_common_domain_batch.py
visualize_common_point_cache_partitions.py
tests/test_adapt_plane_dust3r_to_point_cache.py
tests/test_evaluate_plane_dust3r_common_domain_batch.py
tests/test_visualize_common_point_cache_partitions.py
```

The adapter reuses the pinned runtime's official polygon reconstruction
function, preserves `node_data.json` global wall identities across views, and
assigns distinct scene-global IDs to floor and ceiling. It samples those
labels at the GT cache's recorded `xy` pixels, then refits each predicted
plane on the same LightRecon3D cache points used by the other methods. The
refit uses predicted support only and never reads GT labels. GT enters only
the unchanged downstream evaluator. The adapter validates that the GT
`source_cache_indices` preserve both view index and pixel coordinates.

The canonical 17-scene result is:

```text
local_outputs/plane_dust3r_common_domain_17scene_20260724_v2/
status pass
scenes 17/17
join key (alignment_view_index,x,y)
nearest-neighbour join false
IoU match threshold 0.5
fragmentation threshold 0.1
minimum predicted/observed plane points 64
```

Mean common-domain metrics:

```text
method                    precision  recall   F1       IoU      VOI↓     RI       SC
ordinary RANSAC           0.3748     0.3594   0.3587   0.5239   1.4813   0.7918   0.5641
learning-guided RANSAC    0.5484     0.5497   0.5427   0.6802   1.1721   0.8347   0.6590
Plane-DUSt3R              0.8591     0.7735   0.8098   0.8305   0.3982   0.9494   0.8752
```

Plane-DUSt3R versus learning-guided RANSAC paired over the 17 scenes:

```text
Plane F1: 14 wins / 2 ties / 1 loss; mean delta +0.2671
matched IoU: 16 / 0 / 1; mean delta +0.1503
VOI: 16 / 0 / 1; mean improvement 0.7739 nats
RI: 16 / 0 / 1; mean delta +0.1147
SC: 16 / 0 / 1; mean delta +0.2163
```

The adapted Plane-DUSt3R assigned-point rate is
`0.9999793` on average, with range `0.9998882--1.0`. Its mean normal angular
error (`6.6412` degrees) is worse than guided RANSAC (`4.2449`) because of
the severe `scene_00197` failure, while its median is better
(`1.1572` versus `2.0448` degrees). `scene_00197` is the only scene where
Plane-DUSt3R loses clearly: its common-domain plane F1 is zero, matching the
failure already visible in the method-native result.

Separate headed visual diagnostics are retained for `scene_00180` and
`scene_00197` under:

```text
local_outputs/plane_dust3r_common_domain_17scene_20260724_v1/visualizations/
```

Each view has separate RGB, GT, guided-RANSAC and Plane-DUSt3R PNG files.
Prediction colors are Hungarian-matched to GT only for display; the adapter
NPZ is generated before and independently of this coloring.

Validation:

```text
py_compile: passed
adapter tests: 3 passed
batch/paired-summary tests: 4 passed
visualization test: 1 passed
single-scene real adapter/evaluator canary: passed
17-scene batch: 17/17 passed
```

Interpretation limits: these 17 scenes are still the frozen internal
development set, not a public test split. The common GT is the visible
Structured3D structural-layout partition, which is especially aligned with
Plane-DUSt3R's room-layout task. This result supports a same-input structural
plane comparison, not a claim that Plane-DUSt3R solves the entire bounded,
editable primitive objective or the paper-native benchmark.

### Clarification: point-support evaluation is not bounded-surface evaluation

The common-domain table above is a **3D ordered-point-cache plane partition**
evaluation. Plane equations are refit on common 3D points and the IoU/F1/RI/SC
metrics compare 3D point membership, so the numbers are not 2D mask metrics.
However, the first diagnostic images were 2D view projections. More
importantly, this evaluation does not yet score continuous bounded plane-patch
polygons, surface area overlap, 3D boundary distance, or plane adjacency.
It must not be described as a complete bounded editable primitive evaluation.

Separate true 3D z-buffer point-cloud views for GT, ordinary RANSAC, guided
RANSAC and Plane-DUSt3R are now available at:

```text
local_outputs/plane_dust3r_common_domain_3d_visuals_20260724_v1/
```

They use identical 3D points, cameras and deterministic 80,000-point samples.
For `scene_00180`, Plane-DUSt3R has common-cache 3D plane F1 `0.727` and
matched support IoU `0.890`; the 3D view shows that it preserves the major
room planes but merges/misses narrow wall planes. `scene_00197` visibly
retains the previously measured global identity failure (F1 and matched IoU
both zero).

The next evaluator must construct bounded 3D patches in a per-plane tangent
frame and report surface/boundary/topology metrics separately from the current
point-partition table.

## 2026-07-24 provenance-aware canonical-plane hypothesis gate

Branch: `codex/bounded-support-head`

Current commit SHA: `1fa7027` (the diagnostic is uncommitted and lives under
the ignored `local_outputs/` tree).

The proposed research hypothesis is that a DUSt3R global pointmap must not be
treated as an ordinary merged point cloud. Each aligned point remains a
view-conditioned observation of a latent canonical plane. A candidate model
would jointly infer plane membership, canonical plane parameters, per-view
plane observation bias and reliability without moving the original pointmap.

A first GT-conditioned, cross-validated diagnostic was executed on all 17
frozen development scenes. For each GT plane with at least 256 points in at
least three views, a canonical normal was fit on a spatially interleaved half
of the pixels. Per-view scalar offsets were estimated on that training half
and evaluated on the held-out half.

Results over 72 eligible planes:

```text
median held-out RMSE ratio after per-view offset       0.8917
mean held-out RMSE ratio                               0.8243
median residual variance explained                     0.2048
mean residual variance explained                       0.2870
planes with >=10% held-out RMSE reduction              51.39%
planes with >=25% held-out RMSE reduction              27.78%
planes with >=25% explained residual variance          43.06%
median per-view normal disagreement                    1.9965 degrees
```

This passes only the weakest phenomenon gate: view provenance explains a
material part of plane thickness for a subset of planes, but not universally.
Several planes receive almost no benefit, and some have large per-view normal
disagreement, so an unconditional free scalar `b_vk` is insufficient.

Diagnostic files:

```text
local_outputs/provenance_bias_diagnostic_20260724.py
local_outputs/bottleneck_diagnosis_20260724.py
```

Commands executed:

```text
E:\anaconda\envs\ai\python.exe local_outputs\provenance_bias_diagnostic_20260724.py
E:\anaconda\envs\ai\python.exe local_outputs\bottleneck_diagnosis_20260724.py
```

No checkpoint, cache, NPZ prediction or visualization was overwritten.

Important prior negative evidence remains binding: PlaneGraph-BA v0 modified
per-view geometry and could improve its plane residual while worsening metric
correspondence, even with GT plane identity. The new hypothesis must therefore
treat view bias as a nuisance variable in canonical-plane inference and leave
the frozen DUSt3R pointmaps unchanged.

Unresolved confounds:

* different views observe different regions of the same plane, so the current
  view effect may partly absorb spatially varying pointmap error;
* a free `b_vk` has a gauge ambiguity with the canonical offset;
* unconstrained bias can incorrectly merge distinct parallel planes;
* scalar offsets cannot explain view-dependent tilt or non-rigid local warp.

Next exact step:

Bin GT-plane observations in a common plane-tangent coordinate system, keep
cells observed by multiple views, and compare held-out residual models with
spatial-cell effects only versus spatial-cell plus view effects. If the view
term survives this overlap control, implement a GT-identity oracle estimator
with zero-mean/shrinkage-constrained view bias and evaluate canonical plane
geometry plus fragmentation/overmerge before authorizing a learned assignment
network.

## 2026-07-24 provenance-aware canonical-plane full gate

The overlap control, GT-identity oracle, shared-bias structure test and
fragment-association tests are complete. The outcome is a qualified result:
view provenance is a real nuisance variable for canonical plane estimation,
but a free per-view/per-plane scalar bias is not sufficient to solve plane
instance identity. It should be retained inside a future structured grouping
model, not promoted to the main method by itself.

### 1. Spatial-overlap-controlled provenance effect

Script:

```text
analyze_provenance_view_bias.py
tests/test_analyze_provenance_view_bias.py
```

Output:

```text
local_outputs/provenance_overlap_control_17scene_20260724_v1/
```

For each GT plane, observations were binned in metric plane-tangent cells.
Only cells observed by multiple connected views were retained. Held-out
prediction compared spatial-cell effects alone against spatial-cell plus a
regularized view effect.

At the canonical 10 cm cell size over 68 eligible planes:

```text
median held-out RMSE ratio, view+cell / cell-only       0.722987
mean held-out RMSE ratio                                0.684782
median incremental explained variance                  0.477274
planes improved                                        100.00%
planes improved by at least 5%                          82.35%
planes improved by at least 10%                         75.00%
```

The result is stable across 5 cm and 20 cm controls:

```text
cell size   planes   median RMSE ratio   median incremental explained variance
0.05 m      54       0.647480            0.580690
0.10 m      68       0.722987            0.477274
0.20 m      69       0.751800            0.434790
```

This closes the main confound in the first diagnostic: the provenance effect
does not disappear after controlling for which part of a plane each view
observes. It remains heterogeneous, including planes with almost no effect.

### 2. Canonical-plane oracle under GT identity

Script:

```text
evaluate_canonical_plane_bias_oracle.py
tests/test_evaluate_canonical_plane_bias_oracle.py
```

Output:

```text
local_outputs/canonical_plane_bias_oracle_17scene_20260724_v2/
```

All methods use the same frozen pointmaps and GT plane identity. A single
scene-level trimmed metric Sim3 is used only for metric evaluation. The bias
variables use a zero-mean gauge and never move the pointmaps.

Over 72 planes:

```text
method                   mean normal   median normal   mean offset   median offset
pooled SVD               5.321 deg    1.742 deg       0.2762 m      0.1134 m
free view bias           4.354 deg    1.569 deg       0.2543 m      0.1143 m
bias + reliability       4.517 deg    1.207 deg       0.2481 m      0.0719 m
```

Paired free-bias versus pooled results:

```text
normal mean delta                        -0.9676 deg
normal wins                              70.8% of planes, 76.5% of scenes
offset mean delta                        -0.0219 m
bias-corrected held-out RMSE delta       -0.0137 m
corrected-RMSE wins                      86.1% of planes, 82.35% of scenes
```

The raw canonical-plane residual becomes worse when measured against the
unmodified biased layers; this is expected and confirms that `b_vk` must be
interpreted as an observation nuisance variable rather than a pointmap
correction.

### 3. A shared view translation does not explain the bias

Script:

```text
evaluate_shared_view_bias_structure.py
tests/test_evaluate_shared_view_bias_structure.py
```

Output:

```text
local_outputs/shared_view_bias_structure_17scene_20260724_v1/
```

The model tested whether offsets for different planes in the same view obey
`b_vk ~= n_k^T t_v`. Leave-one-plane-out prediction over 309 plane-view
observations produced:

```text
zero predictor RMSE                      0.3293 m
shared-translation predictor RMSE        0.4190 m
shared / zero RMSE ratio                 1.2724
absolute-error win rate                  22.33%
incremental explained variance          -61.89%
```

Therefore the effect is not a simple residual camera translation that can be
estimated from the other planes of a view. It is more consistent with
view-by-surface and locally non-rigid pointmap error.

### 4. Pairwise fragment identity gate

Script:

```text
evaluate_bias_aware_fragment_association.py
tests/test_evaluate_bias_aware_fragment_association.py
```

Output:

```text
local_outputs/bias_aware_fragment_association_17scene_20260724_v1/
```

The audit population contains 30,773 pairs of high-purity Stage2 fragments,
including 5,466 same-plane and 25,307 different-plane pairs. Fragment support
was joined exactly to the GT cache through
`(alignment_view_index, pixel_x, pixel_y)`.

```text
pair cost                        ROC AUC    AP
ordinary pooled fit             0.806585   0.765088
free bias, marginalized         0.806681   0.764979
bias + 2x required-bias penalty 0.821015   0.770869
```

On near-parallel, disjoint-view pairs, where the hypothesis should matter
most, pooled and penalized AUC are `0.975509` and `0.978334`. The gain is
real but very small; free marginalization alone is slightly worse.

### 5. Leave-one-scene-out global clustering gate

Script:

```text
analyze_bias_aware_fragment_clustering.py
tests/test_analyze_bias_aware_fragment_clustering.py
```

Canonical output:

```text
local_outputs/bias_aware_fragment_clustering_17scene_20260724_v1/summary.json
```

Each held-out scene uses a single global threshold selected only on the other
16 scenes. Connected components then converts fragment pairs into an actual
global partition, exposing transitive merge failures.

```text
pair cost                        precision   recall     mean F1   median F1
ordinary pooled fit             0.81584     0.57822    0.65774   0.72960
free bias, marginalized         0.82482     0.56239    0.64527   0.70815
bias + 2x required-bias penalty 0.85018     0.57844    0.66971   0.74549
```

Paired against pooled:

```text
free bias             mean F1 delta -0.01247; wins/ties/losses 2/10/5
penalized bias        mean F1 delta +0.01197; wins/ties/losses 6/7/4
```

The original exhaustive threshold implementation and the optimized
incremental union-find implementation produced the same result. The optimized
curve is covered by a direct-equivalence unit test. Four clustering tests
pass.

### Decision

Do not train a network whose main claimed contribution is merely predicting
free `b_vk`. The evidence supports the following narrower role:

```text
global learned plane-instance partition
-> canonical plane refit
-> provenance-conditioned nuisance bias and reliability
-> bounded support and adjacency refinement
```

The main learned object must be global plane identity/partition, using
appearance, image connectivity, 2D support overlap, 3D geometry, normal
compatibility and provenance jointly. The bias term is useful as a
geometry-aware residual and regularizer inside that structured model:

* explaining a small view-conditioned shift should not prevent a valid merge;
* requiring a large or inconsistent shift should be negative merge evidence;
* after grouping, bias/reliability improves the canonical plane parameter.

The next implementation gate is a fragment-graph dataset and leave-one-scene-
out structured identity baseline. It must compare geometry-only,
appearance/2D-connectivity-only, provenance-bias-only and their joint model.
Training on the same 17 development scenes without a held-out scene protocol
is not authorized as evidence.

Validation executed in the `ai` environment:

```text
test_analyze_provenance_view_bias.py                 4 passed
test_evaluate_canonical_plane_bias_oracle.py         4 passed
test_evaluate_shared_view_bias_structure.py          2 passed
test_evaluate_bias_aware_fragment_association.py     3 passed
test_analyze_bias_aware_fragment_clustering.py       4 passed
```

No checkpoint, feature cache, input NPZ or prior visualization was
overwritten. The failed/empty timing-attempt directory
`local_outputs/bias_aware_fragment_clustering_17scene_20260724_v2/` and the
duplicate optimized result in `..._v3/` are retained rather than deleted.

## 2026-07-24 learned global fragment-identity gate

The next gate tested the actual remaining bottleneck: predicting whether two
Stage2 fragments belong to the same physical plane, then converting the edge
scores into a globally consistent partition. This is still an
oracle-conditioned association audit, not an end-to-end plane benchmark.

### Prediction-only fragment graph

Script and test:

```text
build_fragment_graph_features.py
tests/test_build_fragment_graph_features.py
```

Output:

```text
local_outputs/fragment_graph_features_17scene_20260724_v1/
```

The exporter produced 30,773 candidate edges over 17 scenes with 26 features.
The largest scene contains 87 accepted fragments and 3,066 candidate edges.
Feature groups include:

* fitted-plane compatibility, centroid/tangent distance, cross-plane residual,
  3D support gap and covariance shape;
* RGB mean/variance;
* same-view pointmap bounding-box overlap, gap and centroid distance;
* view-set overlap and disjoint provenance;
* bias-corrected residual and the magnitude/range of the required view bias.

All feature values are computed from predicted support, frozen pointmaps, RGB
samples or provenance. GT is used only to retain fragments with at least
`0.8` dominant-plane purity and to label edges. The exact GT join remains
`(alignment_view_index, pixel_x, pixel_y)`. Metric features use a prediction-
only per-scene median point-radius normalizer.

### Scene-held-out learned edge baseline

Script and test:

```text
evaluate_learned_fragment_identity.py
tests/test_evaluate_learned_fragment_identity.py
```

Canonical output:

```text
local_outputs/learned_fragment_identity_17scene_20260724_v3/
```

The model is deliberately small: standardized, class-balanced logistic
regression. Every scene is held out once. For each outer fold, the global
clustering threshold is calibrated using four-fold scene-grouped cross-fitted
predictions from the other 16 scenes; the held-out labels never influence the
model or threshold.

Connected-component fragment partition results:

```text
method                          precision   recall     mean F1   median F1
pooled-fit scalar threshold     0.81584     0.57822    0.65774   0.72960
learned geometry only           0.84155     0.82498    0.82437   0.85039
appearance/connectivity only    0.43190     0.69443    0.50452   0.48642
provenance/bias only            0.69024     0.61915    0.60717   0.65185
joint without bias variables    0.87429     0.83250    0.83907   0.89144
joint all 26 features           0.88389     0.84024    0.85245   0.90047
```

The joint model improves mean fragment-partition F1 by `+0.19471` over the
pooled-fit baseline. It wins 15, ties 0 and loses 2 held-out scenes. Mean
held-out pair ROC AUC/AP are `0.97208/0.93306`. The previously catastrophic
`scene_00197` association F1 improves from `0.0` to `0.5983`.

The comparison also sharpens the role of provenance bias. Bias-only features
are insufficient. Adding the four exclusive bias variables to the full joint
feature set raises connected-component mean F1 from `0.83907` to `0.85245`,
but gains are heterogeneous.

### Component-consistent partition

Script and test:

```text
evaluate_fragment_partition_linkage.py
tests/test_evaluate_fragment_partition_linkage.py
```

Output:

```text
local_outputs/fragment_partition_linkage_17scene_20260724_v1/
```

Connected components can merge complete plane instances through one false
positive edge. Average linkage instead requires the mean cross-component
evidence to remain compatible. Thresholds use the same nested scene-held-out
protocol.

```text
method                          precision   recall     mean F1   median F1
pooled fit + average linkage    0.78210     0.53919    0.61846   0.65865
learned geometry + avg link     0.88257     0.87131    0.86850   0.91045
joint no bias + avg link        0.89351     0.88359    0.88128   0.93855
joint all + avg link            0.91707     0.89473    0.90190   0.94551
```

Important paired contrasts:

```text
pooled average vs pooled connected       -0.03929 F1; 6/0/11
joint average vs pooled connected        +0.24416 F1; 17/0/0
joint average vs joint connected         +0.04945 F1; 11/1/5
joint all vs joint no-bias average        +0.02062 F1; 4/10/3
```

Thus the improvement is not explained by replacing connected components with
average linkage: the raw pooled scalar becomes worse under the same
partitioner. The useful combination is learned multi-cue identity evidence
plus component-level consistency. The view-bias features provide a small,
non-universal auxiliary gain, consistent with the preceding hypothesis gate.

### Compute and interpretation

The current learned baseline is CPU-only and requires zero GPU VRAM. The
canonical six-ablation nested evaluation completes in under one minute on the
local machine; the joint model's accumulated fit time over all nested folds is
about 9.7 seconds. A small graph/edge head at the observed maximum of 87 nodes,
3,066 edges and 26 scalar edge features is comfortably compatible with an
8 GB RTX 4060 Ti when trained on frozen caches. This does not imply that
fine-tuning the DUSt3R backbone would fit in 8 GB.

Do not compare the `0.90190` fragment-partition F1 directly with the
Plane-DUSt3R common-point-cache F1. The former:

* evaluates only Stage2 fragments with GT purity at least `0.8`;
* measures whether those fragments are grouped correctly;
* does not penalize internally mixed/impure fragments that were excluded;
* does not yet reconstruct and score all point supports or bounded surfaces.

The next mandatory gate is inference over all predicted fragments without a
GT-purity filter, followed by point-level and bounded-surface evaluation.
Only after that gate can the learned grouping result be called an end-to-end
improvement.

Validation in the `ai` environment:

```text
test_build_fragment_graph_features.py              3 passed
test_evaluate_learned_fragment_identity.py         4 passed
test_evaluate_fragment_partition_linkage.py        2 passed
```

No existing cache, checkpoint, NPZ prediction or visualization was
overwritten.

## 2026-07-24 all-fragment end-to-end identity gate

The GT-purity-conditioned limitation in the preceding fragment audit has now
been removed from held-out inference. A model is trained only on high-purity
fragments from the other 16 scenes, but every sufficiently large Stage2
fragment in the held-out scene is inferred, clustered and mapped back to all
repeated 3D support observations. Internally mixed fragments are retained and
penalized.

Implementation:

```text
evaluate_all_fragment_identity_end_to_end.py
evaluate_all_fragment_common_cache.py
tests/test_evaluate_all_fragment_identity_end_to_end.py
tests/test_evaluate_all_fragment_common_cache.py
```

The fast end-to-end gate uses the 22 joint geometry/RGB/2D-connectivity/
provenance features without the four expensive pairwise view-bias variables.
It retains the scene-held-out logistic identity model, four-fold
scene-cross-fitted threshold calibration and average-linkage component
consistency.

### Repeated 3D support-record result

Canonical output:

```text
local_outputs/all_fragment_identity_end_to_end_17scene_20260724_v1/
```

Mean over 17 held-out scenes:

```text
metric                                      Stage2 direct   learned identity
support partition pairwise precision        0.88625         0.77902
support partition pairwise recall           0.07252         0.85084
support partition pairwise F1                0.13310         0.80698
partition purity                             0.92727         0.85569
partition completeness                       0.11274         0.89826
purity/completeness F1                       0.19982         0.87420
predicted plane count                       77.82            6.47
observed GT plane count                       7.59            7.59
matched plane precision                       0.000           0.735
matched plane recall                          0.000           0.674
matched IoU                                   0.000           0.791
normal error                                  N/A            10.03 deg
```

Pairwise F1 improves on every scene. The mean `0.80698` is higher under the
same repeated-support metric than the previously frozen rows:

```text
B3 manual merge       0.72149
B4 guided RANSAC      0.70657
O1 GT support/RANSAC  0.71958
```

This closes the most important oracle-population objection to the learned
identity result. The remaining high mean normal error is driven by imperfect
support-plane parameters and motivates the already validated canonical
bias/reliability refit after identity recovery.

### Common full-cache expansion

The initial common-cache lift is retained at:

```text
local_outputs/all_fragment_identity_common_cache_17scene_20260724_v1/
```

It intentionally used the old conflict-drop lift and exposed a coverage
contract mismatch: roughly 80k repeated sparse support records cover only
about 2.2% of the roughly 650k-point common cache after duplicate keys and
conflicts are removed. Full-domain Plane F1 is therefore zero despite
excellent support-conditioned identity. This is a failed adapter, not a
model-quality result.

The corrected deterministic completion treats learned component planes as
canonical seeds, assigns every global point to its nearest seed within the
same frozen `0.03` distance threshold, and performs two SVD refits. It uses no
random RANSAC hypotheses and no GT at inference.

Canonical output:

```text
local_outputs/all_fragment_identity_common_cache_17scene_20260724_v2/
```

Same-domain comparison:

```text
method                    precision  recall   plane F1  IoU     normal   VOI     RI      SC
ordinary RANSAC           0.3748     0.3594   0.3587    0.5239  9.14deg  1.4813  0.7918  0.5641
guided RANSAC             0.5484     0.5497   0.5427    0.6802  4.24deg  1.1721  0.8347  0.6590
learned identity + refit  0.6429     0.5525   0.5881    0.7668  3.95deg  1.1432  0.8755  0.6724
Plane-DUSt3R              0.8591     0.7735   0.8098    0.8305  6.64deg  0.3982  0.9494  0.8752
```

Additional learned-identity common-cache statistics:

```text
assignment rate                         0.96210
support partition pairwise F1           0.73627
purity/completeness F1                  0.80195
point-to-plane residual                 0.00402
mean predicted/GT plane count           6.41 / 7.88
```

The candidate direction therefore passes the practical improvement gate: it
beats guided RANSAC on mean plane F1, matched IoU, normal error, VOI, Rand
index and segmentation covering, while using a deterministic learned
identity partition instead of random plane hypothesis discovery. It does not
yet match Plane-DUSt3R's structural-layout identity quality.

The strongest scenes (`scene_00186`, `scene_00192`, `scene_00199`) achieve
full-cache plane F1 `1.0`. The persistent bad cases are `scene_00191`
(`0.211`), `scene_00190` (`0.286`) and `scene_00187`/`scene_00198`
(`0.308`), primarily due to incorrect whole-plane merging or missing narrow
walls rather than fragmentation.

### Separate 3D visual outputs

Separate headed PNGs, retaining RGB and GT rows plus guided and learned
predictions for three representative scenes, are under:

```text
local_outputs/all_fragment_identity_visuals_17scene_20260724_v1/
  final_000_scene_00180/
  final_010_scene_00192/
  final_014_scene_00197/
```

Each scene contains three camera views and separate files for RGB, GT, guided
RANSAC and learned identity. All renders use the identical common 3D cache and
an identical deterministic 80k display sample. Visual inspection confirms
that `scene_00192` is structurally coherent, `scene_00180` reduces
fragmentation relative to guided RANSAC, and `scene_00197` still merges a
major front-wall structure that GT keeps separate.

### Decision and remaining limits

The research direction is now:

```text
Stage2 support fragments
-> learned multi-cue global identity
-> component-consistent partition
-> deterministic full-cache support expansion
-> provenance-aware canonical plane refit
-> bounded support and topology
```

This is materially different from “use another method's planes and add
RANSAC.” The learned object is global plane identity; geometry recovers
parameters and dense support from that learned partition.

Remaining evidence limits:

* all 17 scenes are an internal development set, although each is held out
  from training and threshold calibration;
* no official Structured3D test split is available locally;
* the end-to-end fast gate has not yet reintroduced the four bias variables;
* deterministic distance expansion is not yet spatially bounded in a plane
  tangent frame and can absorb a nearby parallel/coplanar surface;
* continuous bounded-polygon and adjacency metrics remain unimplemented.

## 2026-07-25 bounded tangent-support expansion

The deterministic learned-identity adapter no longer has to treat each
canonical seed as an infinite plane. `evaluate_all_fragment_common_cache.py`
now optionally estimates a robust 2D bounding box in each plane's tangent
frame from the provenance-resolved seed support. Full-cache assignment
requires both:

```text
absolute point-to-plane distance <= 0.03
projected point lies inside the trimmed seed box + margin
```

The tangent boxes remain fixed during the two SVD refits, so repeated refits
cannot expand the support boundary. A missing or degenerate seed box rejects
expansion for that plane instead of silently reverting to an infinite plane.
The old behavior remains available through `--bound_mode none`.

Focused validation:

```text
set PYTHONPATH=E:\Study\code\LightRecon3D&&
E:\anaconda\envs\ai\python.exe tests\test_evaluate_all_fragment_common_cache.py

Ran 3 tests in 0.374s
OK

E:\anaconda\envs\ai\python.exe -m py_compile
  evaluate_all_fragment_common_cache.py
```

The `ai` environment does not contain pytest. Module-form unittest discovery
also resolves an unrelated site-packages package named `tests`, so the
focused test file was executed directly with the repository on `PYTHONPATH`.

Three 17-scene margin ablations used the same learned fragment predictions,
frozen `0.03` plane distance, 1%/99% tangent support quantiles, two SVD refits,
and no GT at inference:

```text
output                                                        margin
local_outputs/all_fragment_identity_common_cache_17scene_
  20260725_bound002_v1/                                      0.02
local_outputs/all_fragment_identity_common_cache_17scene_
  20260725_bound005_v1/                                      0.05
local_outputs/all_fragment_identity_common_cache_17scene_
  20260725_bound010_v1/                                      0.10
```

Unified mean comparison:

```text
method                    precision  recall   plane F1  IoU     normal   VOI     RI      SC
ordinary RANSAC           0.3748     0.3594   0.3587    0.5239  9.14deg  1.4813  0.7918  0.5641
guided RANSAC             0.5484     0.5497   0.5427    0.6802  4.24deg  1.1721  0.8347  0.6590
unbounded identity        0.6429     0.5525   0.5881    0.7668  3.95deg  1.1432  0.8755  0.6724
bounded identity, 0.02    0.8025     0.6177   0.6871    0.7988  3.16deg  0.8533  0.8997  0.7507
bounded identity, 0.05    0.8025     0.6150   0.6856    0.7728  4.06deg  0.9413  0.8905  0.7317
bounded identity, 0.10    0.7487     0.5812   0.6445    0.7693  3.90deg  1.0243  0.8803  0.7004
Plane-DUSt3R              0.8591     0.7735   0.8098    0.8305  6.64deg  0.3982  0.9494  0.8752
```

For the selected 0.02 setting, assignment rate is `0.93496`, point partition
pairwise F1 is `0.81888`, and purity/completeness F1 is `0.87082`. Relative to
unbounded identity, per-scene plane F1 improves in 9 scenes, is unchanged in
6, and decreases in 2; the mean per-scene improvement is `+0.098997`.

This validates the bounded-support hypothesis: a substantial share of the
remaining unbounded-adapter error came from absorbing points that were close
to the same infinite plane but outside the observed finite surface. The best
bounded model now exceeds guided RANSAC by `+0.1444` plane F1 and `+0.1186`
matched IoU, but remains `0.1227` plane F1 below Plane-DUSt3R.

Separate headed 3D renders (RGB, GT, guided RANSAC, unbounded identity and
bounded identity, three views each) are under:

```text
local_outputs/all_fragment_identity_bounded_visuals_17scene_20260725_v1/
  final_000_scene_00180_retry/
  final_008_scene_00190/
  final_014_scene_00197/
```

Visual inspection shows both the gain and the next bottleneck. In
`scene_00190`, bounding prevents many distant coplanar assignments but leaves
a visible unassigned region, consistent with the coverage/precision tradeoff.
In `scene_00197`, assignment remains 100% and the major front-wall merge is
unchanged. That error is already present in the learned component identity;
no post-hoc bound around a merged seed can split it.

The next method change should therefore target learned split/merge decisions,
not another distance or box-margin sweep. The strongest bounded formulation
would predict an instance-affinity/support field jointly with local tangent
occupancy or boundary evidence, allowing a component to split when a
coplanar geometric discontinuity, provenance inconsistency, or negative
adjacency edge contradicts one global identity. Continuous support polygons
and adjacency-aware metrics are still missing.

Changed source files for this step:

```text
evaluate_all_fragment_common_cache.py
tests/test_evaluate_all_fragment_common_cache.py
docs/codex_handoff/CURRENT_STATE.md
```

Latest repository commit remains `1fa7027`; this bounded-support work and the
preceding experiment scripts/results are not committed in the current dirty
worktree.

No existing cache, checkpoint, input NPZ or prior visualization was
overwritten.

## 2026-07-26 learned dense support partition and global reintegration

The crop-level splitter was extended beyond local neighbor supervision.
`train_intra_fragment_splitter.py` now samples balanced non-local pairs:

```text
same-plane remote positives
different-plane global negatives
nearest different-plane hard negatives
local pixel-neighbor edges
local boundary targets
```

The model remains small (`27,730` parameters). A 600-step run on the local
4060 Ti 8 GB completed in `37.98 s`, peaked at `28.9 MB` allocated CUDA
memory, and selected step 100:

```text
local_outputs/intra_fragment_splitter_global_pairs_20260726_v1/
```

`evaluate_intra_fragment_splitter.py` now evaluates symmetric dense learned
affinity with average-linkage clustering as well as the old local connected
components. Dense pair scores average both directed pair logits. Linkage
trees are cached, and partition method/threshold selection uses training
scenes only.

The first dense attempt with the old locally supervised checkpoint failed:

```text
local_outputs/intra_fragment_splitter_partition_eval_20260726_v1/
dense average-linkage hard-val F1 = 0.4217
```

This is retained as a negative result: local-edge supervision does not
calibrate non-local affinity.

With global-pair training and hard-fragment-only calibration:

```text
local_outputs/intra_fragment_splitter_partition_eval_20260726_v2/
selected method                    dense average linkage
selected affinity                  0.175
hard-val precision                 0.7538
hard-val recall                    0.8512
hard-val F1                        0.7779
predicted components / fragment    2.95
GT components / fragment           3.79
```

The previous local-CC result was F1 `0.7690` with `7.30` predicted
components. Thus the main gain is removal of severe oversegmentation, with a
small F1 gain.

Applying the hard-only threshold to every fragment was unsafe: the first
prediction-only export changed 93 fragments into 236 in scene 00184.
That output is retained only as a failed audit:

```text
local_outputs/intra_fragment_split_support_val3_20260726_v1/
```

Calibration was therefore repeated with equal caps of 64 hard and 64 pure
training fragments:

```text
local_outputs/intra_fragment_splitter_partition_eval_20260726_v3/
```

For full support inference the conservative dense threshold `0.10` was
selected. Although local CC had slightly higher calibration F1, it predicted
`12.65` components per hard validation crop and was rejected as operationally
unsafe. `apply_intra_fragment_splitter.py` performs prediction-only export:

```text
fragment/view support
-> deterministic anchors (maximum 512)
-> learned dense affinity
-> average linkage
-> merge tiny anchor clusters
-> nearest-anchor propagation in source-pixel space
-> refit plane equations
```

No GT field is read by this inference script. The accepted three-scene
support export is:

```text
local_outputs/intra_fragment_split_support_val3_20260726_v3/
```

Output fragment counts:

```text
scene 00184   93 -> 131   split groups 26
scene 00195  105 -> 153   split groups 43
scene 00197   64 ->  76   split groups  9
```

The derived batch retains all 17 scenes; only the three difficult
development scenes point to split support. Topology-logistic global identity
was rerun with the identical LOSO audit features:

```text
local_outputs/all_fragment_identity_split_val3_20260726_v2/
```

Sparse repeated-support comparison against the previous topology-logistic
identity:

```text
scene       plane F1 old -> new   partition F1 old -> new
00184       0.5714 -> 0.5833      0.8305 -> 0.8712
00195       0.5556 -> 0.6316      0.7808 -> 0.8822
00197       0.4000 -> 0.4000      0.5206 -> 0.5216
```

The identical bounded full-cache adapter was then run with distance `0.03`,
tangent margin `0.02`, 1% trim and two refits:

```text
local_outputs/all_fragment_identity_split_val3_bounded_common_20260726_v1/
```

Full-cache comparison:

```text
scene       plane F1 old -> new   IoU old -> new      VOI old -> new
00184       0.6000 -> 0.6364      0.7788 -> 0.7546    0.9243 -> 0.8898
00195       0.5882 -> 0.8235      0.8284 -> 0.8401    0.8156 -> 0.5370
00197       0.4000 -> 0.4000      0.8376 -> 0.8378    1.2280 -> 1.2274
17-scene    0.7087 -> 0.7247
```

Only three scenes changed, so the aggregate improvement is attributable to
the learned split/remerge path. Scene 00184 trades a small IoU loss for
higher plane F1 and lower VOI. Scene 00195 is the strongest positive result.
Scene 00197 remains the main unresolved failure.

New or changed source/test files in this step:

```text
train_intra_fragment_splitter.py
evaluate_intra_fragment_splitter.py
apply_intra_fragment_splitter.py
tests/test_train_intra_fragment_splitter.py
tests/test_evaluate_intra_fragment_splitter.py
tests/test_apply_intra_fragment_splitter.py
docs/codex_handoff/CURRENT_STATE.md
```

Focused verification:

```text
py_compile                                                        passed
test_train_intra_fragment_splitter.py                             5 passed
test_evaluate_intra_fragment_splitter.py                          4 passed
test_apply_intra_fragment_splitter.py                             2 passed
```

Important limitations and next action:

* Threshold `0.10` is calibrated on difficult and pure training crops, but
  the full exporter still splits 43 groups in scene 00195 versus 24 known
  hard fragments. A learned split/no-split gate should replace the current
  fixed trigger.
* Non-anchor propagation currently uses nearest source pixel. A scalable
  sparse affinity propagation or prototype head should be trained directly.
* Only the three difficult development scenes use split support. A proper
  final result requires frozen training/calibration followed by all-scene
  prediction and a separate untouched test set.
* Scene 00197 does not improve, so its error should be audited before scaling
  training. It may require features that distinguish large coplanar but
  disconnected supports rather than stronger pair affinity alone.

Latest repository commit remains `1fa7027`. The worktree contains extensive
pre-existing unrelated changes and untracked results; this step was not
committed. No existing checkpoint, cache, NPZ, visualization or prior output
directory was overwritten.

### 2026-07-26 pause point: million-parameter multiscale splitter

Work was explicitly paused by the user after implementing and training the
larger learned splitter. The active calibration process was terminated; no
Python training/evaluation process remains.

The current `IntraFragmentSplitter` keeps the old `architecture=light`
checkpoint contract and adds `architecture=multiscale`:

```text
13D point features
-> 128D point encoder
-> 3 multiscale EdgeConv-style graph blocks (8/16/32 neighbors)
-> 4 local-attention blocks (4 heads; neighbor-only attention)
-> 64D normalized instance embedding
-> affinity head + boundary head + fragment split/no-split head
```

The enhanced split gate additionally receives local edge-probability mean,
standard deviation, minimum and maximum. `apply_intra_fragment_splitter.py`
now supports both:

```text
--partition_method local_connected_components
--partition_method dense_average_linkage
```

and applies the learned split gate before partitioning.

Capacity and local hardware validation:

```text
parameters                 1,227,587 initially; 1,228,099 with pair-stat gate
512 anchor points
32 local neighbors
FP16 CUDA
peak allocated CUDA        about 229.6 MB
4060 Ti 8 GB               passed comfortably
```

Smoke output:

```text
local_outputs/intra_fragment_splitter_multiscale_smoke_20260726_v1/
20 steps including evaluation: 3.79 s
```

The first 1,000-step run used GT-stratified crops and is rejected for the
split gate because it made pure fragments appear artificially mixed:

```text
local_outputs/intra_fragment_splitter_multiscale_train_20260726_v1/
```

The second run used natural uniform crops:

```text
local_outputs/intra_fragment_splitter_multiscale_train_20260726_v2/
1,000 steps
best step 300
runtime 134.06 s
peak CUDA 229,531,648 bytes
```

Its completed crop partition calibration is:

```text
local_outputs/intra_fragment_splitter_multiscale_partition_eval_20260726_v1/
selected local connected components
affinity 0.75
boundary 0.60
gate 0.20
hard-val precision 0.7000
hard-val recall 0.9746
hard-val F1 0.7932
predicted components 3.19
GT components 3.72
```

This beats the accepted 27K model's hard-val F1 `0.7779`, but its first
full-support export failed the fragmentation safety check:

```text
local_outputs/intra_fragment_split_support_multiscale_val3_20260726_v1/
scene 00184   93 -> 350
scene 00195  105 -> 555
scene 00197   64 ->  89
```

That export must not enter global identity evaluation.

The split gate was then extended with local pair-score statistics and trained
for 800 steps:

```text
local_outputs/intra_fragment_splitter_multiscale_train_20260726_v3/
parameters 1,228,099
best step 800
runtime 120.35 s
peak CUDA 229,572,608 bytes
val-all gate balanced accuracy at 0.5 = 0.5995
```

Important sampling detail: that v3 process was already running when the code
was changed to strict alternating hard/pure fragments, so v3 still used the
older 75%-hard four-fragment batch distribution. The source currently on
disk now alternates:

```text
hard, pure, hard, pure
```

This current sampler has not yet produced a new checkpoint. The next training
output should therefore be a new `v4` directory, not overwrite v3.

A new calibration of the v3 checkpoint was started at:

```text
local_outputs/intra_fragment_splitter_multiscale_partition_eval_20260726_v2/
```

but the user requested a pause before it completed. The process was stopped
and that directory has no valid `summary.json`; treat it as incomplete.

Current resume sequence:

1. Run focused syntax/tests once more.
2. Train `multiscale_train_..._v4` with the current strict 50/50 hard/pure
   sampler and pair-stat gate.
3. Calibrate gate/affinity on training scenes only.
4. Export the three validation scenes and reject immediately if fragment
   counts inflate similarly to the failed 350/555/89 export.
5. Only after passing that safety check, rerun global identity and bounded
   full-cache evaluation.

Focused checks completed during this step:

```text
py_compile train/evaluate/apply splitter                         passed
test_train_intra_fragment_splitter.py                            7 passed
test_evaluate_intra_fragment_splitter.py                         4 passed
test_apply_intra_fragment_splitter.py                            2 passed
```

Latest repository commit is still `1fa7027`. None of this multiscale work is
committed, and no prior checkpoint, support NPZ, cache or result directory
was overwritten.

### 2026-07-27 dual-agent coordination and multiscale v4

Parallel collaboration is now formalized:

```text
docs/collaboration/DUAL_AGENT_PROTOCOL.md
docs/collaboration/ACTIVE_WORK_20260727.md
docs/codex_handoff/WEB_CONTEXT_20260727.md
docs/codex_tasks/web_scene00197_gate_review_20260727.md
docs/web_returns/README.md
```

The uploadable GPT web context package is:

```text
local_outputs/web_collab_context_20260727_v1.zip
```

It contains the task, protocol, current source, handoff and selected result
JSON files. The web lane owns only a returned analysis Markdown; the local
lane retains exclusive ownership of code, GPU runs and outputs.

The strict alternating hard/pure multiscale v4 run completed:

```text
local_outputs/intra_fragment_splitter_multiscale_train_20260727_v4/
parameters     1,228,099
best step      600
runtime        104.73 s
peak CUDA      229,572,608 bytes
```

Calibration performance was improved operationally by replacing the Python
edge-loop union-find in `connected_components` with SciPy sparse graph
connected-components. A local-only calibration now skips dense all-pair
affinity and completed in about 33 seconds:

```text
local_outputs/intra_fragment_splitter_multiscale_partition_eval_20260727_v4_local_fast/
affinity             0.85
boundary             0.60
gate                 0.20
hard-val F1          0.7590
pred components      2.72
GT components        3.72
```

The v4 full-support safety export is:

```text
local_outputs/intra_fragment_split_support_multiscale_val3_20260727_v4/
scene 00184   93 -> 204
scene 00195  105 -> 370
scene 00197   64 ->  90
```

This remains unsafe and must not enter global identity evaluation. The result
shows that crop-level component metrics do not constrain full-support
fragment inflation. The next local task is therefore full-natural-support
gate calibration on training scenes with an explicit fragmentation-inflation
constraint, separating the decision to split from the partition produced
after that decision.

The web context addendum containing these v4 facts is:

```text
docs/codex_handoff/WEB_LOCAL_UPDATE_20260727_V4.md
```

## 2026-07-25 topology-aware fragment identity learning

The learned identity stage now explicitly models finite support topology
instead of relying only on plane equations, centroids, RGB and provenance.
`build_fragment_graph_features.py` adds six prediction-only edge features in
a shared tangent frame:

```text
tangent_support_primary_gap_normalized
tangent_support_primary_overlap
tangent_support_secondary_gap_normalized
tangent_support_secondary_overlap
tangent_support_bbox_gap_normalized
tangent_support_bbox_overlap
```

Endpoints use robust 2%/98% support quantiles. The primary tangent axis follows
the pair's tangent centroid separation; the secondary axis is orthogonal
inside the plane. This lets the classifier distinguish two finite coplanar
supports from two noisy observations of the same physical support.

`evaluate_learned_fragment_identity.py` and
`evaluate_all_fragment_identity_end_to_end.py` additionally support:

```text
--model_kind logistic
--model_kind hist_gradient_boosting
--hard_negative_weight N
```

The hard-negative definition is supervised only during training: different-GT
edges with normal angle below 10 degrees and normalized pooled RMSE below 0.1.
There are 288 such edges among 25,307 negatives in the current audit, so the
option prevents them from being numerically hidden by easy negatives.

Focused validation:

```text
tests/test_build_fragment_graph_features.py                 5 passed
tests/test_evaluate_learned_fragment_identity.py            4 passed
tests/test_evaluate_all_fragment_identity_end_to_end.py     3 passed
py_compile                                                   passed
```

The new 32-feature graph is:

```text
local_outputs/fragment_graph_features_topology_17scene_20260725_v1/
```

All-fragment 17-scene LOSO outputs:

```text
local_outputs/all_fragment_identity_topology_logistic_17scene_20260725_v1/
local_outputs/all_fragment_identity_topology_hgb_17scene_20260725_v1/
local_outputs/all_fragment_identity_topology_hgb_hn4_17scene_20260725_v1/
```

Repeated-support results:

```text
method                         pair F1   plane precision  plane recall  matched IoU
previous logistic              0.80698   0.7351           0.6738        0.79095
topology logistic              0.82407   0.7678           0.6960        0.79185
topology HGB                   0.83600   0.7463           0.7062        0.81038
topology HGB + hard-neg x4     0.83419   0.7502           0.7008        0.80870
```

Hard-negative x4 did not improve over ordinary HGB and is rejected. HGB
achieves the best sparse repeated-support score, but it takes about 419
seconds for the full LOSO run versus 156 seconds for topology logistic.
Everything is CPU-side and fits the local 4060 Ti 8 GB machine without using
GPU memory.

Both viable models were then evaluated with the identical full-cache adapter:
distance 0.03, tangent box margin 0.02, 1% trimming and two SVD refits.

```text
output
local_outputs/all_fragment_identity_topology_logistic_bounded_common_17scene_20260725_v1/
local_outputs/all_fragment_identity_topology_hgb_bounded_common_17scene_20260725_v1/
```

Unified common-domain result:

```text
method                    precision  recall   plane F1  IoU     normal   VOI     RI      SC
guided RANSAC             0.5484     0.5497   0.5427    0.6802  4.24deg  1.1721  0.8347  0.6590
unbounded identity        0.6429     0.5525   0.5881    0.7668  3.95deg  1.1432  0.8755  0.6724
bounded identity          0.8025     0.6177   0.6871    0.7988  3.16deg  0.8533  0.8997  0.7507
topology logistic         0.8200     0.6396   0.7087    0.8000  3.54deg  0.7841  0.9087  0.7708
topology HGB              0.7814     0.6475   0.6975    0.8014  3.01deg  0.8081  0.9060  0.7627
Plane-DUSt3R              0.8591     0.7735   0.8098    0.8305  6.64deg  0.3982  0.9494  0.8752
```

Topology logistic is selected despite HGB's higher sparse score because it
has the best full-cache plane F1 and precision. Relative to the previous
bounded model it improves 3 scenes, ties 12 and loses 2, with mean plane-F1
gain `+0.02155`. Major gains are:

```text
scene_00182  0.3750 -> 0.6667
scene_00191  0.4444 -> 0.6316
scene_00190  0.4211 -> 0.5263
```

Separate headed renders for those scenes are under:

```text
local_outputs/all_fragment_identity_topology_visuals_17scene_20260725_v1/
  final_002_scene_00182_retry/
  final_008_scene_00190/
  final_009_scene_00191/
```

Each directory contains separate RGB, GT, previous-bounded and
topology-learned PNGs for three views. Visual inspection confirms a large
coverage recovery in `scene_00182` (63.8% to 98.1% assignment in the shown
view) and more coherent support identities in `scene_00191`.

The remaining bottleneck is now clearer. The graph treats every Stage2
fragment as atomic, so no edge classifier can split a fragment that already
contains multiple physical planes. Prediction-only descriptor audits show:

```text
scene         impure fragments   point rate inside purity<0.8 fragments
scene_00184   26 / 92            31.3%
scene_00195   24 / 104           24.0%
scene_00197   12 / 63            13.3%
```

The two scenes that regress under topology learning are `scene_00184` and
`scene_00195`, precisely the scenes with the highest atomic-fragment impurity.
`scene_00197` remains at F1 0.4. The next learning improvement should therefore
be an intra-fragment split head or point-level affinity/boundary field before
global fragment clustering, not a larger edge classifier or another threshold
sweep.

Changed source files in this step:

```text
build_fragment_graph_features.py
evaluate_learned_fragment_identity.py
evaluate_all_fragment_identity_end_to_end.py
tests/test_build_fragment_graph_features.py
tests/test_evaluate_all_fragment_identity_end_to_end.py
docs/codex_handoff/CURRENT_STATE.md
```

Latest repository commit remains `1fa7027`. These changes are uncommitted, and
no existing checkpoint, cache, NPZ output or visualization was overwritten.

## 2026-07-25 learned intra-fragment splitting first gate

This step starts actual neural representation learning rather than another
handcrafted fragment-edge classifier. It deliberately stops at a crop-level
gate before changing global prediction NPZs.

### Dataset contract and split

`build_intra_fragment_split_dataset.py` creates a read-only manifest over the
existing direct-support and point-aligned GT NPZs. The join remains the exact
saved provenance tuple:

```text
(alignment_view_index, x, y)
```

No nearest-XYZ GT matching is used. Input features are:

```text
fragment-centered normalized XYZ
RGB
explicit 512-space pointmap pixel XY
alignment view index
signed residual to the current fragment plane
current fragment normal
```

The pressure validation split contains the three known hard scenes:

```text
final_003_scene_00184
final_013_scene_00195
final_014_scene_00197
```

The other 14 scenes are training-only. This is a development pressure split,
not a final unbiased test set.

Canonical manifest:

```text
local_outputs/intra_fragment_split_manifest_17scene_20260725_v2/manifest.json
```

Population:

```text
split   scenes  eligible fragments  hard impure fragments  hard points
train   14      902                 104                    139,479
val      3      209                  43                     32,422
```

All 147 hard impure fragments contain at least one individual source view
whose points map to two or more GT planes. This establishes that the target
is learnable as a pointmap-local boundary problem and is not only a cross-view
association artifact.

### Model and training

`train_intra_fragment_splitter.py` implements a small learned point model:

```text
point MLP
-> PointNet global context
-> view-local pixel-neighbor EdgeConv aggregation
-> 16D normalized point embedding
-> local point-pair affinity head
-> point boundary head
```

Training alternates ordinary and hard impure fragments, always includes a
hard fragment per optimizer step, and mines the nearest different-GT point
for supervised negative affinity edges. Crops are single-view because the
audit proves that every hard fragment has a within-view mixture. GT is used
only during training and evaluation.

Early smoke runs are retained:

```text
local_outputs/intra_fragment_splitter_smoke_20260725_v1/
local_outputs/intra_fragment_splitter_smoke_20260725_v2/
local_outputs/intra_fragment_splitter_smoke_20260725_v3/
```

They exposed two invalid shortcuts:

* ordinary edge F1 was dominated by same-plane local edges;
* a pointwise boundary head without neighbor aggregation collapsed.

The corrected canonical smoke run is:

```text
local_outputs/intra_fragment_splitter_smoke_20260725_v4/
```

It uses 1000 optimizer steps, four fragments per step, 512 points per crop and
AMP on the local RTX 4060 Ti 8 GB. Best checkpoint is step 200:

```text
parameters                              27,730
runtime                                 43.0 seconds
peak CUDA memory                        26,764,288 bytes (~25.5 MiB)
hard-val edge balanced accuracy         0.6979
hard-val same-plane recall              0.8546
hard-val different-plane recall         0.5412
hard-val boundary balanced accuracy     0.6659
hard-val boundary F1                    0.3814
```

This confirms that local training is far below the 8 GB GPU limit. Ordinary
edge F1 is retained only as a secondary statistic; balanced accuracy and
different-plane recall are the relevant anti-merge gates.

### Actual component split evaluation

`evaluate_intra_fragment_splitter.py` converts predicted local affinity edges
to connected components. Affinity and boundary thresholds are calibrated
only on 64 hard training-scene fragments, then frozen for the 43 hard
validation fragments.

Canonical output:

```text
local_outputs/intra_fragment_splitter_component_eval_20260725_v1/
```

Selected thresholds:

```text
affinity >= 0.5
boundary gate disabled (threshold 1.01)
```

The disabled boundary gate is evidence that the current gain comes from the
affinity representation; the boundary head is not yet strong enough to help
the final partition.

Hard validation crop result:

```text
method              precision  recall  pair F1  predicted comps  GT comps
unsplit fragment    0.4512     1.0000  0.6153   1.00             3.79
learned components  0.7086     0.9112  0.7690   7.30             3.79
delta              +0.2574    -0.0888 +0.1538
```

The first research gate therefore passes: learned point affinity can recover
substantial structure that no fragment-level merge classifier can recover.
However, the predicted component count shows clear over-segmentation. It is
not yet authorized as a replacement for global predictions.

The next stage should:

1. reduce over-segmentation with component-consistent affinity training or
   average-linkage/region-growing instead of raw connected components;
2. export split support records for only the three held-out hard scenes;
3. feed those split records into the frozen topology-logistic global identity
   stage;
4. rerun bounded full-cache evaluation and require a gain over plane F1
   `0.7087` before expanding to a full scene-held-out training protocol.

New source files:

```text
build_intra_fragment_split_dataset.py
train_intra_fragment_splitter.py
evaluate_intra_fragment_splitter.py
tests/test_build_intra_fragment_split_dataset.py
tests/test_train_intra_fragment_splitter.py
tests/test_evaluate_intra_fragment_splitter.py
```

Focused validation:

```text
dataset tests       2 passed
training tests      4 passed
evaluation tests    2 passed
py_compile          passed
```

Latest repository commit remains `1fa7027`. All new source and experiment
outputs are uncommitted. No existing checkpoint, cache, NPZ result or
visualization was overwritten.

## 2026-07-27 oracle route arbitration and evaluator audit

GPT's frozen route rules are stored at:

```text
docs/web_returns/WEB_RETURN_oracle_preregister_20260727.md
```

The local implementation added:

```text
evaluate_intra_fragment_oracles.py
tests/test_evaluate_intra_fragment_oracles.py
```

`apply_intra_fragment_splitter.py` now emits schema-v2 full-support
diagnostics without changing the learned gate decision:

* per-group gate probability, decision and skip reason;
* candidate and selected component sizes;
* tiny-component counts;
* nearest-anchor distance p50/p95/max;
* per-point frozen candidate group/component labels;
* separate view-group and partition inflation.

The audit found and fixed a data-contract bug: input support with
`point_plane_ids=-1` was previously treated as a fragment and converted to a
positive plane ID. Contract-fixed validation counts are:

```text
scene 00184  92 valid input fragments -> 191 outputs
scene 00195 104 valid input fragments -> 365 outputs
scene 00197  63 valid input fragments ->  87 outputs
```

The preserved unassigned point counts are respectively `1078`, `448`, and
`96`.

The final split-gain oracle selected:

```text
scene 00184  20 groups versus learned gate 40
scene 00195  31 groups versus learned gate 44
scene 00197  14 groups versus learned gate 15
```

After the identical topology-logistic identity backend and frozen bounded
common-cache evaluator:

```text
method                                  00184   00195   00197 Plane F1
current gate + current identity         0.6000  0.8235  0.4000
split oracle + current identity         0.6667  0.8235  0.4000
current gate + dominant-GT identity     0.7273  0.7000  0.2000
split oracle + dominant-GT identity     0.6364  0.9000  0.2000
```

Under the preregistered rules, both S and I are failed, so the route is E:
pause L/G/J and audit the evaluator/data contract.

The decisive E sanity check assigned the same support records their exact
pointwise GT labels solely for offline evaluator validation:

```text
scene 00184  Plane F1 1.0
scene 00195  Plane F1 1.0
scene 00197  Plane F1 0.0
```

For `scene_00197`, all five GT-labeled support seeds have normal error below
`0.05 degrees` and offset error below `1.4e-4`, but the frozen
`0.03 + tangent_bbox + two refits` expansion produces matched IoUs:

```text
0.2786, 0.4752, 0.3690, 0.2406, 0.2119
```

Therefore current bounded results for `scene_00197` cannot arbitrate between
split and identity methods. The support-to-bounded evaluator fails its own GT
sanity check.

Mapping validation completed:

```text
scene 00184  support-key-in-GT rate 0.78815
scene 00195  support-key-in-GT rate 0.93930
scene 00197  support-key-in-GT rate 0.90535
```

Each 80,000-record support contains 60,000 repeated `(view,x,y)` keys. The
schema-v2 candidate labels reconstruct the formal learned-gate labels exactly
for all three scenes.

Canonical outputs:

```text
local_outputs/intra_fragment_split_support_multiscale_val3_20260727_v6_contract_fix/
local_outputs/intra_fragment_oracles_val3_20260727_v5_evaluator_sanity/
local_outputs/oracle_common_current_learned_contract_val3_20260727_v1/
local_outputs/oracle_common_current_split_contract_val3_20260727_v1/
local_outputs/oracle_common_identity_learned_contract_val3_20260727_v1/
local_outputs/oracle_common_identity_split_contract_val3_20260727_v1/
local_outputs/oracle_common_pointwise_gt_evaluator_val3_20260727_v1/
local_outputs/oracle_route_visuals_scene00197_20260727_v2_contract/
```

Focused validation:

```text
py_compile                                            passed
test_apply_intra_fragment_splitter.py                 5 passed
test_evaluate_intra_fragment_oracles.py               5 passed
test_evaluate_intra_fragment_splitter.py              4 passed
test_train_intra_fragment_splitter.py                 7 passed
```

Invalid attempts are retained:

```text
local_outputs/intra_fragment_oracles_val3_20260727_v1/
  failed because tiny components were rejected by the old >=3-point refit

local_outputs/intra_fragment_oracles_val3_20260727_v2/
local_outputs/intra_fragment_oracles_val3_20260727_v3/
  pre-contract-fix diagnostics; not used for the final decision
```

Changed files in this step:

```text
apply_intra_fragment_splitter.py
evaluate_intra_fragment_oracles.py
tests/test_apply_intra_fragment_splitter.py
tests/test_evaluate_intra_fragment_oracles.py
docs/web_returns/WEB_RETURN_oracle_preregister_20260727.md
docs/collaboration/ACTIVE_WORK_20260727.md
docs/collaboration/ORACLE_ROUTE_DECISION_20260727.md
docs/codex_handoff/CURRENT_STATE.md
```

Latest repository commit remains `1fa7027`; all changes and outputs are
uncommitted. No existing checkpoint, cache, NPZ, result directory or
visualization was overwritten.

Next exact step:

Define and freeze a support-to-bounded evaluator on the 14 training scenes
that handles DUSt3R residual scale, repeated/unmatched provenance keys and
tangent-bound/refit drift. Do not resume model training until the pointwise-GT
sanity oracle achieves Plane F1 at least `0.90`, assignment at least `0.98`,
and zero conflicting keys on each pressure scene.

## 2026-07-27 E0 exact-support evaluator contract audit

GPT's E-route repair preregistration was applied in the required order:
Literature Gate, repository contract map, E0, then the E0 stop decision.

New implementation and focused test:

```text
audit_support_bounded_evaluator_contract.py
tests/test_audit_support_bounded_evaluator_contract.py
```

The implementation constructs E0 from the complete point-aligned GT registry,
not from the earlier 80K sparse support records.  It enforces a one-to-one
global cache registry, deduplicates same-key/same-label records, drops
ambiguous and unmatched keys without fallback, and preserves the exact GT
support without expansion or refit.

Literature and code-contract outputs:

```text
docs/literature/LITERATURE_GATE_support_bounded_20260727.md
docs/audits/SUPPORT_BOUNDED_CONTRACT_MAP_20260727.md
```

The Literature Gate actually read the original CVF papers/pages for
MV-DUSt3R+, PlanarRecon, AirPlanes, PlaneMVS, PolyFit and PlaneRAS.  This work
is explicitly classified as evaluator/data-contract repair, not innovation.

Canonical E0 output:

```text
results/evaluator_contract_e0_20260727_v1/
```

It contains the required config, Git/environment state, input hashes, CSV and
JSON join/residual/expansion/refit/per-plane/per-scene audits, summary and run
log.

E0 result:

```text
scene   Plane F1   TP       IoU median/min   GT assignment   coverage   full-cache assignment
00184   1.0        13/13    1.0 / 1.0        1.0             1.0        0.959199
00195   1.0        10/10    1.0 / 1.0        1.0             1.0        0.970761
00197   1.0         5/5     1.0 / 1.0        1.0             1.0        0.976515
```

E0 therefore fails only the preregistered historical full-cache assignment
threshold `>=0.98`.  Exact GT plane support intentionally leaves
GT-unlabeled/non-plane cache points at `-1`, so the maximum possible
full-cache assignment rate equals the GT-labeled point rate shown above.
Join, Hungarian matching, Plane TP and continuous IoU are all exact.

Mechanical verdict:

```text
E-Key/Data Contract
```

E1, E2 and E3 were not executed.  L/G/J remain stopped.

The raw sparse-support key diagnostic, before identity reconciliation, found:

```text
scene   positive records   unique keys   duplicate records   ambiguous keys
00184   78922              19952         58970               19914
00195   79552              19988         59564               19974
00197   79904              20000         59904               19999
```

Those different labels are pair/view-local fragment IDs; this diagnostic is
not a score for the reconciled identity output.

Residual-scale audit also confirms that the old fixed `0.03` expansion
distance is not a stable local-scale contract.  In `00197`, GT plane 4 has
371,129 points, residual median `0.028909`, p95 `0.058954`, and p95 divided by
median 8-NN distance `90.11`.

Visual diagnostic:

```text
results/evaluator_contract_e0_visual_scene00197_20260727_v1/
```

All five rendered E0 label regions are pixel-identical to their GT renders
below the title header.

Focused validation:

```text
py_compile audit/test files                              passed
test_audit_support_bounded_evaluator_contract.py         5 passed
test_evaluate_all_fragment_common_cache.py               3 passed
test_evaluate_intra_fragment_oracles.py                  5 passed
```

The first test invocation accidentally resolved `tests` to the installed
Ultralytics package and failed on its settings-file permission.  Repository
tests were then rerun with `unittest discover` and all 13 passed.

Fixed-format return:

```text
docs/codex_returns/CODEX_RETURN_evaluator_contract_20260727.md
```

Changed files in this step:

```text
audit_support_bounded_evaluator_contract.py
tests/test_audit_support_bounded_evaluator_contract.py
docs/literature/LITERATURE_GATE_support_bounded_20260727.md
docs/audits/SUPPORT_BOUNDED_CONTRACT_MAP_20260727.md
docs/codex_returns/CODEX_RETURN_evaluator_contract_20260727.md
docs/collaboration/ACTIVE_WORK_20260727.md
docs/codex_handoff/CURRENT_STATE.md
```

Latest repository commit remains `1fa7027`.  No checkpoint, cache, NPZ, old
result or visualization was overwritten.

Next exact step:

Preregister one assignment/coverage metric contract that separates plane
support measured over GT-labeled points from optional full-cache labeling of
GT-unlabeled/non-plane points.  Do not run E1--E3 or resume L/G/J before this
denominator contract is frozen.

## 2026-07-27 corrected assignment/coverage denominator contract

GPT froze a metric-only correction that retires the ambiguous acceptance name
`assignment_rate`.  The definitions are now:

```text
gt_support_assignment_rate      = |A intersection G| / |G|
gt_support_matched_coverage     = |M| / |G|
full_cache_label_rate           = |A| / |C|
gt_plane_label_rate             = |G| / |C|
nonplane_false_assignment_rate  = |A minus G| / |C minus G|
```

Here `C` is the valid full cache, `G` is GT plane support, `A` is all
prediction-assigned points, and `M` is the subset assigned to the prediction
matched to its GT instance.

The preregistration is stored at:

```text
docs/web_returns/WEB_RETURN_assignment_coverage_contract_20260727.md
```

`audit_support_bounded_evaluator_contract.py` now emits the five explicit
metrics and their integer numerators/denominators.  The historical
`support_coverage` formula is retained only as `legacy_support_coverage`; it
equals GT-support assignment, not matched coverage.

The preserved v1 E0 result was rejudged by:

```text
rejudge_support_bounded_e0_metrics.py
```

No global cache, point cloud, support NPZ, geometry, join, matcher, or model
was rerun.  The script used only the existing v1 scene/per-plane JSON tables.

Corrected output:

```text
results/evaluator_contract_e0_20260727_v2_metric_contract/
```

Integer result:

```text
scene   assigned |A|   GT |G|   matched |M|   non-plane false   verdict
00184   557245         557245   557245        0                 passed
00195   672582         672582   672582        0                 passed
00197   651494         651494   651494        0                 passed
```

For all three scenes:

```text
gt_support_assignment_rate       1.0
gt_support_matched_coverage      1.0
nonplane_false_assignment_rate   0.0
full_cache_label_rate            exactly equals gt_plane_label_rate
```

The corrected mechanical verdict is:

```text
E0 Passed
```

The original v1 `summary.json` was not modified or deleted.  An additive
marker was written at:

```text
results/evaluator_contract_e0_20260727_v1/
  VERDICT_SUPERSEDED_BY_METRIC_CONTRACT_20260727.json
```

It marks the old `E-Key/Data Contract` verdict as invalidated by the
contradictory full-cache assignment denominator and points to the v2 summary.

New focused test:

```text
tests/test_rejudge_support_bounded_e0_metrics.py
```

E1--E3 were not executed, and L/G/J were not resumed.

Updated fixed-format return:

```text
docs/codex_returns/CODEX_RETURN_evaluator_contract_20260727.md
```

Latest repository commit remains `1fa7027`.  No old output was overwritten.

Next exact step:

Keep E1--E3 and L/G/J paused until a new evaluator-stage preregistration is
provided.

## 2026-07-27 E1 tangent-bbox expansion safety audit

GPT preregistered E1 after the corrected E0 passed. This round executed only:

```text
same E0 exact GT support and exact (view,x,y) join
+ frozen current tangent-bbox expansion
+ no plane refit
```

The E1-specific literature appendix is:

```text
docs/literature/LITERATURE_GATE_tangent_bbox_e1_20260727.md
```

PlanarRecon, AirPlanes, PlaneMVS, and PolyFit were rechecked. None was found
to accept the repository's trimmed axis-aligned tangent rectangle plus a
single raw point-to-plane threshold as final full-scene support. E1 is
classified as an evaluator safety test of an engineering heuristic, not a
method reproduction or innovation.

Implementation:

```text
evaluate_tangent_bbox_e1_contract.py
tests/test_evaluate_tangent_bbox_e1_contract.py
```

The existing `expand_plane_seeds(refit_iterations=0)` cannot satisfy E1
because it still calls `weighted_fit_plane`. The new E1 path reproduces the
same first-pass bbox/residual assignment and plane-dropping rule but returns
the unchanged initial plane parameters. The focused test mocks
`weighted_fit_plane` to raise if called; the E1 path completes with
`refit_call_count=0`.

Frozen parameters:

```text
distance_threshold       0.03
bound_margin             0.02
bound_trim_quantile      0.01
min_points_per_plane     1000
match_iou                0.5
fragmentation_iou        0.1
```

All GT/cache SHA-256 values matched E0. For every scene, cache/GT
`view_indices`, `pixel_xy`, and point order were equal, and the recomputed
exact join was elementwise equal to the GT labels with no ambiguous or
unmatched key.

Formal output:

```text
results/evaluator_contract_e1_20260727_v1/
```

Result:

```text
scene   F1      TP       IoU median/min   GT assign   matched cov   nonplane false   multi/GT   min seed retain
00184   1.000   13/13    0.921/0.609      0.9982      0.9790        0.9982           0.4836     0.7954
00195   1.000   10/10    0.897/0.752      0.9980      0.9837        0.9911           0.2674     0.9096
00197   0.200   1/5      0.392/0.254      0.9437      0.0850        0.9255           0.6228     0.2767
```

In `00197`, 284,412 of 298,912 added points came from another GT plane
(`95.15%`). Fragmentation and overmerge excess both increased from zero to
two. Plane- and point-weighted IoU both decreased, so their directions were
consistent and did not mask the failure.

Mechanical verdict:

```text
Failed
route: E-Bounded Expansion
```

The current tangent-bbox expansion is unsafe for formal evaluation or default
bounded-plane construction. E0 remains the trusted exact-support evaluator.
E2/E3 were not implemented or run, and L/G/J were not resumed. The only
permitted next experiment is a separately preregistered E2 candidate
bounded-footprint rule.

Visual diagnostic:

```text
results/evaluator_contract_e1_visual_scene00197_20260727_v1/
```

It contains five-view RGB, GT, E0-exact, and E1 bbox/no-refit partition
renders. Visual inspection confirms large cross-plane relabeling in E1.

One initial result directory had a blank scene column only in the residual
audit table. It was preserved without deletion at:

```text
results/evaluator_contract_e1_20260727_v1_invalid_incomplete_residual_scene_field/
```

After fixing the field mapping, the formal v1 directory was regenerated with
identical inputs and parameters. Per-scene CSV rows and every prediction NPZ
array were verified equal between the archived and corrected runs.

Commands executed:

```text
E:\anaconda\envs\ai\python.exe -m py_compile evaluate_tangent_bbox_e1_contract.py tests/test_evaluate_tangent_bbox_e1_contract.py
E:\anaconda\envs\ai\python.exe tests/test_evaluate_tangent_bbox_e1_contract.py
E:\anaconda\envs\ai\python.exe evaluate_tangent_bbox_e1_contract.py
E:\anaconda\envs\ai\python.exe visualize_common_point_cache_partitions.py ...
```

Final focused validation:

```text
E1 no-refit tests                              5 passed
E0 evaluator-contract tests                    6 passed
E0 denominator rejudgment tests                2 passed
common-cache expansion tests                   3 passed
oracle evaluator tests                         5 passed
partition visualizer tests                     1 passed
total                                          22 passed
py_compile                                     passed
E1 formal run                                  completed
E1 artifact integrity                          passed
E1 visual diagnostic                           5 views generated and inspected
```

Fixed-format return:

```text
docs/codex_returns/CODEX_RETURN_evaluator_e1_20260727.md
```

Branch remains `codex/bounded-support-head`; latest commit remains `1fa7027`.
No E0 result, checkpoint, feature cache, or pre-existing visualization was
overwritten.

Next exact step:

Send the E1 fixed-format return to GPT and request only an E2 candidate
bounded-footprint preregistration. Do not implement E2, run E3, or resume
L/G/J without that new preregistration.

## 2026-07-27 E2 candidate bounded-footprint safety audit

GPT supplied the complete preregistration:

```text
E2_PREREG_candidate_bounded_footprint_20260727_v1
```

Only the isolated evaluator-only E2 path was implemented:

```text
evaluate_candidate_bounded_footprint_e2_contract.py
tests/test_evaluate_candidate_bounded_footprint_e2_contract.py
```

Frozen geometry behavior:

```text
E0 exact seed assignment remains immutable
deterministic tangent basis from frozen normal
local scale = median projected eighth-neighbour distance
footprint = accepted Delaunay triangles union clipped local disks
tau = min(0.03, 0.5*s, max(1e-6, q99 + 3*1.4826*MAD))
candidate third-neighbour distance <= 2*s
multi-candidate points rejected without reassignment
```

No bbox, convex-hull fallback, morphology, hole filling, component bridge,
fit, refit, RANSAC, training, E1 prediction, E3, or L/G/J path was used.

Before the formal run, 12 focused synthetic tests passed. A read-only
preflight then verified all three GT/cache SHA-256 values against E0, exact
view/pixel/point order, exact E0 join reproduction, and zero
ambiguous/unmatched keys.

Formal output:

```text
results/evaluator_contract_e2_candidate_bounded_footprint_20260727_v1/
```

The formal run was executed once. All three predictions were written and
hashed before GT scoring:

```text
00184 f7eb3345c25878fd79ae34c2d3bc83071635d6b0a427fcf0775f94be5246f337
00195 33209d8000dd7f9a9cc6b42a734ace3a63ce0ea1186d8701fdf99ccf2d635b3e
00197 a91487b1a4df863804a67c481974a4215bcf3678d21fe2f07e9341fdc385fd4e
```

The freeze barrier was satisfied:

```text
prediction_frozen_before_GT_scoring = true
```

Contract validity passed. E0 matcher and metric functions were reused
directly, and evaluating the exact seed support reproduced the frozen E0
metrics. All forbidden call counters remained zero.

Formal result:

```text
scene   verdict   F1 / TP       IoU median/min   seed retain   added   cross-seed
00184   Failed    1.0 / 13/13   0.9916/0.9193    1.0           4000    0
00195   Failed    1.0 / 10/10   0.9946/0.9031    1.0           3560    63
00197   Failed    1.0 / 5/5     0.9995/0.9896    1.0           335     3224
```

E2 successfully avoided E1 seed deletion and relabeling:

```text
global/per-plane seed retention = 1.0 exactly
seed deletion/reassignment      = 0 / 0
plane IDs and count             unchanged
final multi-assignment          = 0
fragmentation/oversegmentation  = 0 / 0
```

However, because E0 exact support already contains all GT plane points, every
accepted non-seed addition was GT nonplane/unlabelled. Total leakage fraction
was `1.0` in all scenes. The `0.05` leakage gate failed everywhere; 00184 and
00195 also exceeded scene leakage over GT support. Cross-seed intrusion
failed in 00195 and 00197. Continuous median/minimum IoU gates also failed,
despite perfect discrete F1 and TP.

Mechanical verdict:

```text
Failed
contract_valid = true
```

Visual diagnostic:

```text
results/evaluator_contract_e2_visual_scene00184_20260727_v1/
```

Five RGB/GT/E0/E2 view sets were generated and inspected. E2 retains the
plane layout but fills narrow regions that exact E0 leaves unassigned,
consistent with the recorded nonplane leakage.

Final validation:

```text
E2 focused tests                  12 passed
E1 no-refit tests                  5 passed
E0 evaluator-contract tests        6 passed
E0 denominator rejudgment tests    2 passed
common-cache expansion tests       3 passed
oracle evaluator tests             5 passed
partition visualizer tests         1 passed
total                              34 passed
py_compile                         passed
E2 artifact integrity              passed
```

The final read-only artifact audit recomputed the three NPZ file hashes and
verified all required fields, immutable seeds, unique-candidate acceptance,
multi-candidate rejection, no second reassignment, zero fit/refit/RANSAC
counters, table row counts, valid E0 hashes/join, exact matcher/metric reuse,
and the prediction-before-GT freeze barrier.

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_evaluator_e2_candidate_bounded_footprint_20260727.md
```

Latest repository commit remains `1fa7027`; the worktree remains dirty with
inherited changes. E0/E1 outputs, cache files, checkpoints, and prior
visualizations were not overwritten.

Next exact step:

Stop and preserve the complete E2 audit. Do not change registered constants,
rerun E2, implement/run E3, restore L/G/J, refit, or train. Await a new
explicit mechanical instruction.

## 2026-07-27 E4 immutable-support archival gate

Registration:

```text
E4_PREREG_immutable_support_rebaseline_20260727_v1
```

Post-E2 mechanical decision:

* E-Bounded Expansion is stopped.
* E0 remains only as a non-mutating evaluator sanity ceiling.
* E1/E2/E3, evaluator-side support growth/refit, pressure-scene threshold
  tuning, and L/G/J remain stopped.

The sole planned E4 comparison was the historical 17-scene topology-logistic
method B (`Plane F1 = 0.7087`) against the old learned split/remerge method M
(`Plane F1 = 0.7247`) using their immutable pre-expansion support and plane
parameters.

The archival gate found both referenced input directories and all 17 NPZ files
for both methods:

```text
local_outputs/all_fragment_identity_topology_logistic_17scene_20260725_v1/predictions/
local_outputs/all_fragment_identity_split_val3_20260726_v2/predictions/
```

Both historical bounded summaries explicitly reference those directories as
their `support_prediction_dir`. All 34 NPZs contain the required support IDs,
plane IDs, plane parameters, pointmap provenance, and source points.

The blocking failure is historical hash provenance. Neither the prediction
NPZs nor their historical method summaries contain per-scene hashes, and no
existing repository hash manifest covers these 34 files. Hashes computed now
cannot prove identity with the files used on 2026-07-25/26.

Mechanical verdict:

```text
Blocked
contract_valid = false
scientific_effect_interpretation_allowed = false
```

No immutable metrics or B-vs-M effect were computed. Robust/Artifact was not
adjudicated. The current `+0.0160` historical effect remains unverified under
the non-mutating evaluator.

Formal output:

```text
results/immutable_support_rebaseline_20260727_v1/
```

It contains all ten registered files. `input_hashes.json` records current
whole-file, support-array, and plane-parameter hashes as a new archival
baseline while explicitly distinguishing them from missing historical
expected hashes. `per_scene_metrics.csv` contains 34 Blocked rows with blank
metric fields; `per_plane_metrics.csv` contains only its schema.

Read-only/prohibited-action audit:

```text
current file hash stable across audit       34 / 34
evaluation started                          false
prediction output created                   false
metric comparison performed                 false
expansion calls                             0
refit calls                                 0
model inference calls                       0
training calls                              0
GT-guided prediction calls                  0
```

Changed/added:

```text
docs/codex_tasks/e4_immutable_support_rebaseline_20260727.md
audit_immutable_support_rebaseline_e4.py
tests/test_audit_immutable_support_rebaseline_e4.py
results/immutable_support_rebaseline_20260727_v1/
docs/codex_returns/CODEX_RETURN_immutable_support_rebaseline_20260727.md
docs/codex_handoff/CURRENT_STATE.md
```

Validation:

```text
py_compile                              passed
focused E4 archival-audit tests         4 passed
formal archival gate                    Blocked as registered
required formal artifacts               10 / 10
artifact integrity                      passed
```

Commands executed:

```text
E:\anaconda\envs\ai\python.exe -m py_compile audit_immutable_support_rebaseline_e4.py tests/test_audit_immutable_support_rebaseline_e4.py
E:\anaconda\envs\ai\python.exe tests/test_audit_immutable_support_rebaseline_e4.py
E:\anaconda\envs\ai\python.exe audit_immutable_support_rebaseline_e4.py --output-dir results/immutable_support_rebaseline_20260727_v1
```

One module-form unittest attempt imported the unrelated installed
`site-packages/tests` package and failed on an Ultralytics settings permission
check. Direct execution of the repository test file passed all four tests.

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_immutable_support_rebaseline_20260727.md
```

Branch remains `codex/bounded-support-head`; current commit remains
`1fa7027f16630069d3475e0d012affea25e91e81`. Existing worktree changes and all
historical outputs were preserved.

Next exact step:

Stop the method comparison. Search only existing archives, transfer bundles,
logs, or manifests for contemporaneously recorded hashes or independently
immutable copies of the 34 pre-expansion NPZ files. Do not rerun inference to
recreate historical inputs. If no such evidence exists, leave the old method
effect `Unverified`; any prospective immutable baseline rebuild requires a new
explicit preregistration.

## 2026-07-28 E5 prospective immutable baseline gate

Registration:

```text
E5_PREREG_prospective_immutable_baseline_rebuild_20260727_v1
```

E5 was registered as a new prospective baseline experiment rather than an E4
rerun. Historical B/M predictions were prohibited as E5 inputs.

The registered literature gate was completed against the original PlanarRecon,
AirPlanes, PlaneMVS, and PolyFit papers:

```text
docs/literature/LITERATURE_GATE_immutable_evaluation_e5_20260727.md
```

The reviewed systems construct/fuse/refine their support as explicit method
stages and evaluate the resulting predictions. Matching, nearest-neighbor
label transfer, and surface sampling are evaluator correspondence operations;
none reports evaluator-side support growth or plane refit. Exact byte hashes
and forbidden-call auditing are not stated by those papers.

### Method-definition result

The requested M checkpoint exists, loads successfully, and contains 27,730
state parameters:

```text
local_outputs/intra_fragment_splitter_global_pairs_20260726_v1/best.pt
SHA-256 bd8f89bc8c457fd0b54547c775477813b4d5e3a0667f81c08ec28d4faaadfcd1
```

The prospective run was nevertheless blocked before inference for five
registered reasons:

1. M inference source was never committed.
2. The only archived M source copy postdates the historical run and differs
   from the current contract-fixed source; the exact run source is absent.
3. The 27K checkpoint best step was selected using 00184/00195/00197, although
   E5 permits them only in final evaluation.
4. Historical learned splitting ran only on those 3 scenes. The other 14 were
   passed through unchanged and are splitter training scenes, so there is no
   valid `17 scenes x M` evaluation.
5. The shared B/M topology-logistic identity source was also never committed
   and its historical summary registered no source hash.

Key source evidence:

```text
historical split completion       2026-07-26T05:49:46Z
archived apply source timestamp   2026-07-26T08:18:48Z
archived apply SHA-256            513f98bae42349f03bce30e3be44f9dbb8ac83563e598900809764f612a7067d
current apply SHA-256             4d19eeed2c6739e4de1ec35e8a9df017ea8fb5bb57ece7ddbdc7bd9255c20e99
identity source SHA-256           e10f8ba2d7c2f475526a8e1a5edaed95a77fff434215c214572c8efa071856bd
```

Mechanical verdict:

```text
Blocked
contract_valid = false
formal_inference_authorized = false
scientific_effect_interpretation_allowed = false
```

No B/M prediction was produced and no GT array was loaded.

### Input and freeze audit

The historical 17-scene order was extracted without reading historical
predictions. Before stopping, the read-only audit verified:

```text
scenes                                  17
original RGB images                     85
global caches                           17
direct-support common inputs            17
all files present                       yes
all recorded cache/support hashes       match
all original images present/hashed      yes
```

This proves the common inputs are available; it does not repair the method
definition or data-leakage failures.

Denial manifest:

```text
results/prospective_immutable_baseline_e5_20260727_v1/prerun_manifest.json
SHA-256 9619741506aa06abf57ccda941a3ff0751bf3c15e2f835041faa45a342d958d6
```

Prediction/forbidden-call audit:

```text
B predictions                           0
M predictions                           0
GT evaluation loads                     0
expansion/refit/fit/RANSAC calls        0
support/ID/parameter mutations          0
model inference calls                   0
training calls                          0
```

Formal output:

```text
results/prospective_immutable_baseline_e5_20260727_v1/
```

All 21 required files and both empty prediction directories exist.
`per_scene_metrics` contains 34 explicit Blocked rows; per-plane and effect
metrics remain empty. No historical metric was copied into E5.

Added/changed:

```text
docs/codex_tasks/e5_prospective_immutable_baseline_rebuild_20260727.md
docs/literature/LITERATURE_GATE_immutable_evaluation_e5_20260727.md
audit_prospective_immutable_baseline_e5.py
tests/test_audit_prospective_immutable_baseline_e5.py
results/prospective_immutable_baseline_e5_20260727_v1/
docs/codex_returns/CODEX_RETURN_prospective_immutable_baseline_e5_20260727.md
docs/codex_handoff/CURRENT_STATE.md
```

Validation/commands:

```text
E:\anaconda\envs\ai\python.exe -m py_compile audit_prospective_immutable_baseline_e5.py tests/test_audit_prospective_immutable_baseline_e5.py
E:\anaconda\envs\ai\python.exe tests/test_audit_prospective_immutable_baseline_e5.py
E:\anaconda\envs\ai\python.exe audit_prospective_immutable_baseline_e5.py --output-dir results/prospective_immutable_baseline_e5_20260727_v1

py_compile                    passed
focused E5 tests              4 passed
artifact integrity            passed
source hashes                 11 / 11 matched
```

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_prospective_immutable_baseline_e5_20260727.md
```

Branch remains `codex/bounded-support-head`; current commit remains
`1fa7027f16630069d3475e0d012affea25e91e81`. Existing worktree changes,
checkpoints, caches, E0-E4 artifacts, and historical predictions were
preserved.

Next exact step:

Stop E5. Do not use historical predictions, guess the missing historical
source, run the pressure-selected checkpoint as a clean final model, evaluate
it on its 14 training scenes, retrain inside E5, or restore expansion/E3/L/G/J.
A future prospective comparison requires a new preregistration with committed
and hashed B/M inference code, a checkpoint selected on scenes disjoint from
all 17 evaluation scenes, and one complete frozen M definition applicable to
every evaluation scene.

## Session update: 2026-07-28 E6 prospective method/split freeze gate

Registration:

```text
E6_PREREG_prospective_method_and_split_freeze_20260728_v1
```

Branch and commit at audit:

```text
branch  codex/bounded-support-head
HEAD    1fa7027f16630069d3475e0d012affea25e91e81
```

Mechanical verdict:

```text
Blocked
```

E5 remains Blocked, H0/H1 remain untested, and the historical B-vs-M effect
remains Unverified.

E6 was restricted to a metadata-only candidate/provenance gate. It did not
authorize B/M definition work, a source commit, training, checkpoint creation
or loading, real-scene inference, GT semantic-content loading, or quality
evaluation.

Observed scene/input state:

```text
local Structured3D scene directories                200
explicit historical 17-scene quarantine              17
unproven-provenance default quarantine               183
total quarantine                                     200
scenes with RGB/global-cache/direct-support presence  17
clean candidates                                       0
registered requirement                                77
```

Only the historical 17 scenes have both DUSt3R global-cache and direct-support
inputs. Every one is explicitly quarantined by E0-E5/training/pressure
history. The remaining 183 have no affirmative repository ledger proving
non-use, so the E6 default rule also quarantines them.

No GT plane-label content, prediction result, method metric, difficulty rank,
or plane count was read during candidate construction.

The four split files are explicit empty Blocked records:

```text
training       0 / 40
calibration    0 / 10
selection      0 / 10
evaluation     0 / 17
```

The registered salt was unchanged and no scene was manually substituted.
The B/M YAML files inside the formal package are Blocked receipts with
`scientific_method_definition: absent`; they are not runnable method configs.
No scientific source/config freeze or clean source commit was created.

Implemented audit-only files:

```text
docs/codex_tasks/e6_prospective_method_and_split_freeze_20260728.md
audit_prospective_freeze_e6.py
tests/test_audit_prospective_freeze_e6.py
docs/codex_returns/CODEX_RETURN_prospective_freeze_e6_20260728.md
```

Formal package:

```text
results/prospective_freeze_e6_20260728_v1/
```

It contains all 21 registered files and contains no trained checkpoint,
formal prediction, per-scene quality metrics, or method-effect statistics.

Commands:

```text
E:\anaconda\envs\ai\python.exe -m py_compile audit_prospective_freeze_e6.py tests/test_audit_prospective_freeze_e6.py
E:\anaconda\envs\ai\python.exe tests/test_audit_prospective_freeze_e6.py
E:\anaconda\envs\ai\python.exe audit_prospective_freeze_e6.py --output-dir results/prospective_freeze_e6_20260728_v1 --synthetic-test-results tmp/e6_synthetic_test_results.json
```

Validation:

```text
py_compile                      passed
focused E6 synthetic tests      5 passed
formal exit                     2 (registered Blocked stop)
required result files           21 / 21
training/checkpoint calls       0
B/M real-scene inference        0
formal predictions              0
GT semantic-content loads       0
quality metric computations     0
expansion/refit/fit/RANSAC       0
prediction mutations            0
```

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_prospective_freeze_e6_20260728.md
```

Next exact step:

Stop E6 and do not amend or rerun it. A new Registration ID first needs an
external or independently auditable scene-level provenance ledger proving at
least 77 scenes were never used for historical training, calibration,
checkpoint selection, oracle analysis, E0-E5, or manual debugging. Those
scenes must also have materialized, hashable RGB, global-cache, and
direct-support inputs. Do not relax the count/salt, infer non-use from missing
logs, write B/M definitions, train, infer, load evaluation GT, or restore
expansion/E3/L/G/J before that gate is repaired.

## Session update: 2026-07-28 E7 clean scene acquisition gate

Registration:

```text
E7_PREREG_clean_scene_acquisition_and_materialization_20260728_v1
```

Branch and commit at audit:

```text
branch  codex/bounded-support-head
HEAD    1fa7027f16630069d3475e0d012affea25e91e81
```

Mechanical verdict:

```text
Blocked_Source_Access
```

E6 remains Blocked and was not rerun or relaxed. The historical B-vs-M effect
remains Unverified.

Official source audit:

```text
official website       https://structured3d-dataset.org/
official repository    https://github.com/bertjiazheng/Structured3D
official scene count   3500
training split         scene_00000-scene_02999
validation split       scene_03000-scene_03249
test split             scene_03250-scene_03499
errata blob SHA        859568c5df2af217fe6688265bc6db8058dc91d5
```

The official publisher requires the user to complete its agreement form and
then grants download access. The E7 execution context had no authorized
official download URL, no authority-proven official archive, and no relevant
environment access hint. No form was submitted and no license/terms were
accepted on the user's behalf.

Pre-acquisition inventory:

```text
physical local Structured3D directories     200
historical 17                                17
unknown-provenance local scenes             183
conservative scene-like reference tokens    206
GT semantic parses                            0
```

The inventory records file-system metadata, file-type counts, byte counts, and
metadata tree hashes. It does not parse annotations or read prediction/quality
results.

Six text-only tokens occurred beyond the physical 200:

```text
scene_02999 scene_03000 scene_03249 scene_03250 scene_03499 scene_20260
```

The first five are existing official split-boundary mentions.
`scene_20260` is a conservative false-positive prefix from a
`scene_202607...` date token and lies outside the E7 source range.
`scene_02999` was conservatively excluded. Independent recomputation with only
the 200 physical directories plus the official errata gives it rank 357, so
the first 128 frozen candidate IDs are exactly unchanged.

Candidate freeze:

```text
salt                 LightRecon3D_E7_clean_acquisition_20260728_v1
candidate universe   2727
frozen IDs           128 / 128 unique
all IDs in 00200-02999  yes
download before freeze  no
manual replacement      no
```

Acquisition-plan SHA-256:

```text
aa9889c0aea54e59df49ea6f696531278d4ece036535ba2a525e29c02ec5528c
```

The plan hash matches the hash receipt and summary. All 21 registered files
exist.

No acquisition/materialization occurred:

```text
download attempts                     0
archive extraction                    0
raw scenes acquired                   0
annotation semantic parses            0
DUSt3R input-preparation calls         0
Stage1 input-preparation calls         0
global caches/direct support created  0
B/M/training/checkpoint/evaluation     0
expansion/refit/RANSAC                 0
forbidden counter sum                  0
```

Added:

```text
docs/codex_tasks/e7_prospective_clean_scene_acquisition_20260728.md
docs/codex_tasks/e7_official_structured3d_errata_snapshot.txt
audit_clean_scene_acquisition_e7.py
tests/test_audit_clean_scene_acquisition_e7.py
results/clean_scene_acquisition_e7_20260728_v1/
docs/codex_returns/CODEX_RETURN_clean_scene_acquisition_e7_20260728.md
```

Commands:

```text
E:\anaconda\envs\ai\python.exe -m py_compile audit_clean_scene_acquisition_e7.py tests/test_audit_clean_scene_acquisition_e7.py
E:\anaconda\envs\ai\python.exe tests/test_audit_clean_scene_acquisition_e7.py
E:\anaconda\envs\ai\python.exe audit_clean_scene_acquisition_e7.py --output-dir results/clean_scene_acquisition_e7_20260728_v1 --download-search-root C:\Users\admin\Downloads
```

Validation:

```text
py_compile                          passed
focused tests                       5 passed
formal exit                         2 (registered Blocked stop)
required result files              21 / 21
plan hash                           matched
candidate IDs                      128 / 128 unique
independent first-128 recompute     identical
```

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_clean_scene_acquisition_e7_20260728.md
```

Next exact step:

Stop E7 and do not amend or rerun it. The user must independently accept the
official Structured3D Terms of Use and obtain an authorized official download
link. Once access exists, create a new Registration ID and take a fresh
pre-acquisition inventory before any download. Do not use a third-party mirror,
the 183 unknown scenes, official validation/test scenes, B/M, training,
checkpoint selection, evaluation, expansion, refit, RANSAC, or L/G/J.

## Session update: 2026-07-28 E8 preregistration freeze

Registration:

```text
E8_PREREG_official_archive_acquisition_20260728_v1
```

State:

```text
PRE_REGISTERED_NOT_AUTHORIZED
```

The supplied E8 protocol explicitly authorized preregistration only. No E8
inventory, network probe, ZIP central-directory probe, download, extraction,
input-preparation freeze, model call, GT parse, or evaluation was authorized.

The full protocol is preserved byte-for-byte at:

```text
results/official_archive_acquisition_e8_20260728_v1/registration.md
```

Protocol SHA-256:

```text
565324e8ee8ddeb5c527fe463fb5e632b6b0b242a206d431e3e01f70df561799
```

Added:

```text
docs/codex_tasks/e8_official_archive_acquisition_prereg_20260728.md
freeze_official_archive_acquisition_e8.py
tests/test_freeze_official_archive_acquisition_e8.py
results/official_archive_acquisition_e8_20260728_v1/
docs/codex_returns/CODEX_RETURN_official_archive_acquisition_e8_20260728.md
```

The formal preregistration directory contains all 36 registered filenames.
Execution-stage files are explicit `not_executed` receipts rather than empty
success manifests.

Frozen scanner:

```regex
scene_(\d{5})(?!\d)
```

Synthetic scanner validation passed all registered cases, including rejecting
date strings such as `scene_20260728`.

Commands:

```text
E:\anaconda\envs\ai\python.exe -m py_compile freeze_official_archive_acquisition_e8.py tests/test_freeze_official_archive_acquisition_e8.py
E:\anaconda\envs\ai\python.exe tests/test_freeze_official_archive_acquisition_e8.py
E:\anaconda\envs\ai\python.exe freeze_official_archive_acquisition_e8.py --protocol-source C:\Users\admin\.codex\attachments\9c8ec622-7496-4b88-b8fc-f4e56745f7d7\pasted-text.txt --output-dir results/official_archive_acquisition_e8_20260728_v1
```

Validation:

```text
py_compile                         passed
focused preregistration tests      4 passed
required result files             36 / 36
protocol copy SHA-256              matched
scene-scanner synthetic tests      passed
execution verdict issued          false
network/inventory/download         not executed
D archive root created             false
F clean root created               false
forbidden counter sum              0
```

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_official_archive_acquisition_e8_20260728.md
```

Next exact step:

Stop. Wait for a separate, explicit user instruction authorizing execution of
Registration `E8_PREREG_official_archive_acquisition_20260728_v1`. Until that
authorization arrives, do not inventory, probe network/ZIP metadata, create
D/F directories, download, extract, freeze input preparation, or run any model
or evaluator.

## Session update: 2026-07-29 E8 authorized execution

Branch: `codex/bounded-support-head`

Base commit SHA:

```text
1fa7027f16630069d3475e0d012affea25e91e81
```

Registration:

```text
E8_PREREG_official_archive_acquisition_20260728_v1
```

Final mechanical verdict:

```text
Blocked_Official_Metadata_Changed
```

Execution facts:

* Separate explicit authorization was received.
* The corrected pre-acquisition inventory completed with 53,153 records and
  417 unique pre-existing canonical scene IDs.
* The official README and errata were snapshotted and hashed.
* The 128-candidate order was independently generated from the E8 inventory
  and errata with the frozen E7 salt; no E7 candidate file was read.
* Nineteen official remote objects were frozen.
* The first ZIP tail Range read was interrupted and recorded as
  `Paused_Resumable`.
* On the mandatory same-source resume validation, the README
  `Content-Length` differed from its frozen value (`null` versus a supplied
  value). Under E8 section 4 this mechanically changed the final verdict to
  `Blocked_Official_Metadata_Changed`.
* ZIP central-directory mapping, disk preflight, archive download and
  extraction did not execute.
* Neither the D acquisition root nor the F clean root was created.
* All forbidden counters remained zero.

Files added or changed for E8 execution:

```text
execute_official_archive_acquisition_e8.py
tests/test_execute_official_archive_acquisition_e8.py
results/official_archive_acquisition_e8_20260728_v1/
docs/codex_returns/CODEX_RETURN_official_archive_acquisition_e8_20260728.md
docs/codex_handoff/CURRENT_STATE.md
```

Commands and checks:

```text
E:\anaconda\envs\ai\python.exe -m py_compile execute_official_archive_acquisition_e8.py tests\test_execute_official_archive_acquisition_e8.py
E:\anaconda\envs\ai\python.exe -m unittest discover -s tests -p 'test_execute_official_archive_acquisition_e8.py' -v
E:\anaconda\envs\ai\python.exe -m unittest discover -s tests -p 'test_freeze_official_archive_acquisition_e8.py' -v
E:\anaconda\envs\ai\python.exe execute_official_archive_acquisition_e8.py
```

Validation:

```text
E8 execution tests                3 passed
E8 preregistration regression     4 passed
protocol SHA-256                  matched
forbidden counter sum             0
D acquisition root created        false
F clean root created              false
```

Next exact step:

Stop E8. Do not resume or modify its frozen manifests. A new Registration ID
must be preregistered before official metadata can be frozen again or any ZIP
probe, download or extraction can continue.

## Session update: 2026-07-29 E9 preregistration freeze

Registration:

```text
E9_PREREG_stable_official_archive_acquisition_20260729_v1
```

State:

```text
PRE_REGISTERED_NOT_AUTHORIZED
```

E8 remains:

```text
Blocked_Official_Metadata_Changed
```

The new E9 protocol explicitly identifies the E8 stop as an overly strict
response-header identity rule, but does not alter E8's mechanical verdict.
E9 separates text response-body identity from optional HTTP response metadata
and introduces dual independent archive probes with canonical central
directory plus fixed head/tail range hashes.

This turn froze E9 only. It did not execute inventory, retrieve official text,
probe an archive, create D/F directories, download, extract, parse GT, freeze
a complete input-preparation definition, materialize inputs, or run any model
or evaluator.

Protocol:

```text
results/stable_official_archive_acquisition_e9_20260729_v1/registration.md
SHA-256 ee0f73ded7b93389d804c319986e1aa8e7e26330ab60fc50b88363c63cf77e80
```

Files added:

```text
docs/codex_tasks/e9_stable_official_archive_acquisition_prereg_20260729.md
freeze_stable_official_archive_acquisition_e9.py
tests/test_freeze_stable_official_archive_acquisition_e9.py
results/stable_official_archive_acquisition_e9_20260729_v1/
docs/codex_returns/CODEX_RETURN_stable_official_archive_acquisition_e9_20260729.md
```

Commands:

```text
E:\anaconda\envs\ai\python.exe -m py_compile freeze_stable_official_archive_acquisition_e9.py tests\test_freeze_stable_official_archive_acquisition_e9.py
E:\anaconda\envs\ai\python.exe -m unittest discover -s tests -p 'test_freeze_stable_official_archive_acquisition_e9.py' -v
E:\anaconda\envs\ai\python.exe freeze_stable_official_archive_acquisition_e9.py --protocol-source C:\Users\admin\.codex\attachments\9a2035f3-aafa-4dbe-8aa2-250e4ca52dfd\pasted-text.txt
```

Validation:

```text
focused E9 preregistration tests  4 passed
registered files                  31 / 31
protocol SHA-256                  matched
scene scanner synthetic tests     passed
network/download/extraction       not executed
D/F E9 directory creation         false
forbidden counter sum             0
```

Next exact step:

Stop. Wait for a separate explicit user authorization to execute Registration
`E9_PREREG_stable_official_archive_acquisition_20260729_v1`. Until then, do
not inventory, retrieve official text, probe remote archives, create E9 D/F
roots, download, extract, freeze input preparation, or run models/evaluation.

## Session update: 2026-07-29 E9 authorized execution

Registration:

```text
E9_PREREG_stable_official_archive_acquisition_20260729_v1
```

Final mechanical verdict:

```text
Invalid
```

Execution facts:

* The byte-identical frozen E9 protocol received separate execution
  authorization.
* E9 regenerated 58,988 inventory records and classified 208 canonical scene
  IDs as disqualifying.
* README and errata were freshly retrieved, hashed, stored as binary snapshots,
  and rechecked after candidate generation.
* E9 independently generated 128 candidates from a 2,725-scene universe using
  the frozen salt; no E7/E8 candidate file was read as generation input.
* Two independent probes completed for all 19 remote objects. Total length,
  fixed head/tail hashes, central-directory canonical hashes and strong ETags
  matched across both probes.
* Read-only Range interruptions were saved as `Paused_Resumable`; per-object
  checkpoint manifests enabled same-source recovery.
* The perspective indexes uniquely mapped all 128 candidates to formal
  archives 01–14 and none to boundary archives.
* The executor incorrectly included the separate annotation archive in the
  perspective-archive member-path conflict check, yielding 6,994 expected
  perspective/annotation overlaps as false mapping conflicts.
* This was an execution-code/frozen-contract mismatch. The explicit
  authorization required immediate `Invalid`; the executor was not patched
  and resumed.
* Disk preflight, D/F E9 directory creation, archive download, extraction,
  Clean Raw qualification, and input-preparation freeze did not execute.
* All forbidden counters remained zero.

Files added:

```text
execute_stable_official_archive_acquisition_e9.py
tests/test_execute_stable_official_archive_acquisition_e9.py
results/stable_official_archive_acquisition_e9_20260729_v1/implementation_contract_audit.json
results/stable_official_archive_acquisition_e9_20260729_v1/remote_member_indexes_probe_1/
results/stable_official_archive_acquisition_e9_20260729_v1/remote_member_indexes_probe_2/
```

Commands and validation:

```text
E:\anaconda\envs\ai\python.exe -m py_compile execute_stable_official_archive_acquisition_e9.py tests\test_execute_stable_official_archive_acquisition_e9.py
E:\anaconda\envs\ai\python.exe -m unittest discover -s tests -p 'test_execute_stable_official_archive_acquisition_e9.py' -v
E:\anaconda\envs\ai\python.exe -m unittest discover -s tests -p 'test_execute_official_archive_acquisition_e8.py' -v
E:\anaconda\envs\ai\python.exe execute_stable_official_archive_acquisition_e9.py
```

```text
E9 execution tests                4 passed
ZIP/path regression tests         3 passed
protocol SHA-256                  matched
probe 1                           19 / 19
probe 2                           19 / 19
dual hard identity match          passed
candidate perspective mapping     128 / 128
forbidden counter sum             0
D/F E9 roots created              false
archive downloads                 0
```

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_stable_official_archive_acquisition_e9_20260729.md
```

Next exact step:

Stop E9. Do not patch and resume it. Preserve all E9 evidence. A new
Registration must be adjudicated before any further archive acquisition
attempt.
## 2026-07-29 E11 member-level Range acquisition

Registration:

```text
E11_PREREG_RANGE_MEMBER_ACQUISITION_20260729_v1
```

Current execution verdict:

```text
Ready_Clean_Raw_Pool
```

Completed:

* authoritative E11 protocol frozen; SHA-256
  `78bcfdf0b945d1ab8aa73ccaab4a91ccb89cb4c9d7101ceeda6263269d15f30d`;
* all 48 adopted E9 upstream files verified;
* 128/128 P mappings unique, 128/128 PB mappings absent, and 128/128 sealed
  annotation mappings unique;
* frozen member manifest contains 3,265 `rgb_rawlight.png` members,
  4,491,349,817 compressed bytes and 4,490,111,105 uncompressed bytes;
* reduction relative to full perspective archives is 0.972471811;
* 15/15 involved remote objects passed fresh hard-identity verification;
* all 3,265 selected RGB members were acquired by exact HTTP Range, verified
  with compressed SHA-256, uncompressed size, CRC32 and uncompressed SHA-256,
  and atomically written to F;
* no complete perspective archive was downloaded;
* the 39,392,097-byte annotation archive was downloaded to D and permanently
  retained with whole-file SHA-256
  `c3caea0861c7ca1dbdedafe3fa72994264d0d985d1f879d10158cf540bb700bc`;
* 128/128 annotation JSON members were extracted as sealed bytes without JSON
  deserialization;
* RGB basic decoding and all provenance/eligibility checks passed for 128/128
  candidates, producing 128 `Clean_Raw` scenes;
* all forbidden call counters remain zero.

Same-registration engineering repairs were committed independently:

```text
15a3ecb Harden E11 same-source Range retries
5bfa9a7 Chunk E11 remote identity range reads
6f58cc2 Add E11 ZIP suffix range fallback
5b761f7 Match E11 annotation hash to frozen E9 bytes
```

The third repair was supported by a read-only compatibility probe showing
that the official endpoint returned a correct 1 MiB ZIP suffix Range after
repeatedly truncating an explicit EOF Range. The fourth repair corrected only
the byte serialization used for the local recheck of an already-frozen E9
central-directory identity. The local and remote 7,001 canonical annotation
rows were identical.

```text
E11 tests                           5 passed
E9 identity regression tests       4 passed
Clean Raw scenes                   128 / 128
complete perspective archives      0
leftover .part files               0
forbidden calls                    0
```

Mechanical next action:

```text
已建立不少于77个Clean Raw scenes。停止，等待新的输入物化任务。
```
## 2026-07-29 E12 clean input materialization

Registration:

```text
E12_PREREG_clean_input_materialization_20260729_v1
```

Execution verdict:

```text
Blocked_Incomplete_Input_Materialization
```

E11 upstream verification passed:

```text
Clean Raw scene order             128 / 128
RGB provenance hashes             3265 / 3265
sealed annotation byte hashes     128 / 128
GT semantic reads                 0
```

E12 stopped at the mandatory pre-inference freeze gate:

* HEAD was `5b761f76f5bc2c1c08b0ddd7918216511cb767c4`;
* the shared execution worktree had 21 tracked dirty entries and could not be
  recorded as clean without overwriting or absorbing unrelated user changes;
* the DUSt3R weight existed and hashed to
  `9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd`;
* the designated stable Stage1 checkpoint
  `local_outputs/stage1_large_80m_train2048_sharded_v1/best.pt` did not exist
  in the repository or the E11 D/F roots;
* alternate Stage1 checkpoints were not selected because E12 explicitly
  forbids checkpoint selection;
* the existing global-cache writer is embedded in a Stage2-NPZ-driven Stage3
  CLI, and no frozen raw-scene to direct-support entry exists.

No F output or staging directory was created. DUSt3R, Stage1, B/M, training,
evaluation, GT matcher, fitting, RANSAC, and support mutation call counts all
remain zero.

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_clean_input_materialization_e12_20260729.md
```

Do not proceed to B/M. Restore the exact Stage1 checkpoint and provide a
clean, committed raw-scene materialization implementation before a separately
adjudicated resume or new registration.

## 2026-07-29 E13 Stage1 selection and clean input materialization

Registration:

```text
E13_PREREG_stage1_selection_and_clean_input_materialization_20260729_v1
```

Frozen protocol SHA-256:

```text
3817df95447ee5b42a8d030cf6a1428f0f03aa9306feb76bac8ded988abecee0
```

Execution verdict:

```text
Blocked_Insufficient_Views
```

E13 created the required independent worktree and branch without changing the
original dirty worktree:

```text
worktree  E:\Study\code\LightRecon3D_e13_materialization
branch    codex/e13-clean-materialization
HEAD      5b761f76f5bc2c1c08b0ddd7918216511cb767c4
status    clean
```

The registered view rule was applied to the frozen E11 scene order using only
RGB member paths. It requires the lexically first five images from the largest
single-room group and forbids cross-room filling. Results:

```text
E11 scenes checked             128
view-rule pass                 124
view-rule fail                   4
maximum room-group size = 5    124
maximum room-group size = 4      4
```

Blocking scenes:

```text
scene_02465  selected room 2D_rendering/584  4 RGB
scene_01698  selected room 2D_rendering/141  4 RGB
scene_02008  selected room 2D_rendering/208  4 RGB
scene_02208  selected room 2D_rendering/141  4 RGB
```

The stop occurred before checkpoint selection/freeze. A priority-1 compatible
checkpoint was discovered read-only at:

```text
E:\Study\code\LightRecon3D\local_outputs\manual_transfer_bootstrap_20260718_v1_extracted\checkpoints\stage1_large_80m_train2048_sharded_v1\best.pt
```

It was not selected or frozen because the earlier mandatory view gate failed.
No historical-scene smoke inference or new-scene inference ran. No F output or
staging directories were created. GT semantic calls and all forbidden calls
remain zero.

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_stage1_selection_and_clean_input_materialization_e13_20260729.md
```

Mechanical next action: stop E13. Do not patch or rerun it. A new adjudication
is required before changing the five-view single-room rule; B/M remains
forbidden.

## 2026-07-29 E14 variable-view clean input materialization

Registration:

```text
E14_PREREG_variable_view_clean_input_materialization_20260729_v1
```

Execution verdict:

```text
Ready_Clean_Input_Pool
```

E14 superseded only E13's fixed five-view input rule. E12 and E13 remain
blocked and were not resumed or rejudged. The same frozen E11 128-scene order
was used.

The implementation was committed in an independent clean worktree:

```text
worktree  E:\Study\code\LightRecon3D_e14_materialization
branch    codex/e14-variable-view-materialization
commit    23ea0d520ea32fadfebb953dd4a31f5367df35d0
status    clean
```

Frozen view selection:

```text
five-view scenes  124
four-view scenes    4
```

The four-view scenes were exactly `scene_02465`, `scene_01698`,
`scene_02008` and `scene_02208`. No cross-room filling or padding occurred.

Upstream and formal results:

```text
E11 RGB hashes verified           3265 / 3265
sealed annotation byte hashes      128 / 128
global-cache                       128 / 128
direct-support                     128 / 128
provenance                         128 / 128
technical retries                    0
GT semantic calls                    0
forbidden calls                      0
```

All 256 output NPZ whole-file hashes were recomputed after completion. All
schemas, key arrays, scene order, path/input-hash metadata and provenance
checks passed. The staging directory is empty.

Frozen identities:

```text
DUSt3R SHA-256  9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd
Stage1 SHA-256  2edc0ec3387a9a6ebcdd8dec7d1847fb0e38f62e9fc6694a5d7d32b97e3f4ffa
config SHA-256  885acc82bb5338f7df9235ba87170ca3868888c7dfa950d9c0913b67e70f7c07
```

Formal outputs:

```text
F:\Structured3D_clean_e11\global_cache_e14\
F:\Structured3D_clean_e11\direct_support_e14\
```

Fixed return:

```text
docs/codex_returns/CODEX_RETURN_variable_view_clean_input_materialization_e14_20260729.md
```

Mechanical next action: stop. The next registration must directly define and
run the independent B/M comparison. Do not reopen data acquisition, view
selection or input-preparation auditing.
## 2026-08-02 Stage1 DUSt3R-feature versus RGB ablation

The Stage1 representation ablation is implemented but the six full training
runs have not yet been executed. The comparison is parameter matched:

```text
DUSt3R: 768 x 32 x 32 -> 1x1 projection -> shared query-mask head
RGB:       3 x 512 x 512 -> 16x16/16 projection -> shared query-mask head
```

Both input projections contain 196,864 parameters and both complete models
contain 3,491,842 trainable parameters. Training data, loss, sample order,
checkpoint selection and the frozen 80-record W3 validation set are shared.
The preregistered primary metric is support F1, with scene-cluster bootstrap
over the eight validation scenes and three fixed seeds.

Added:

```text
models/stage1_feature_ablation_head.py
run_stage1_feature_ablation.py
summarize_stage1_feature_ablation.py
tests/test_stage1_feature_ablation_head.py
docs/research_practice/STAGE1_FEATURE_ABLATION.md
```

Validation completed:

```text
py_compile                                    passed
Stage1 feature-ablation unit tests            3 passed
RGB real-data 2-train/2-val one-epoch smoke   passed
DUSt3R real-data 2-train/2-val smoke           passed
batch-size-2 real-data training smoke          passed
```

Smoke outputs are under:

```text
results/stage1_feature_ablation_smoke_rgb_20260802/
results/stage1_feature_ablation_smoke_dust3r_20260802/
results/stage1_feature_ablation_smoke_rgb_batch_20260802/
```

These smoke metrics are technical checks only and must not be reported as the
feature-ablation result. The full experiment requires both conditions for
seeds 20260616, 20260617 and 20260618, each with 2,048 training records, 20
epochs and all 80 frozen validation records, followed by the registered
summarizer.

Formal execution started on the local RTX 4060 Ti at
`2026-08-02T09:48:33Z` under:

```text
results/stage1_feature_ablation_20260802_v1/
```

Launcher PID is `46368`. It runs RGB seeds first, followed by DUSt3R seeds,
then invokes the paired summarizer. At the latest check the first RGB run was
active and caching the 2,048 training records. No formal verdict exists yet.
