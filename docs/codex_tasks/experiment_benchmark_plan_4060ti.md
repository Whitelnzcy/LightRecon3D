# LightRecon3D Experiment, Ablation, and Benchmark Plan

Date: 2026-07-18

Status: proposed execution plan based on the frozen 17-scene pilot, the public
plane benchmark track, the failed plane-feedback branch, and the current local
RTX 4060 Ti 8 GB workstation audit.

This plan does not authorize retraining, long DUSt3R recaching, dataset
downloads, or external-baseline installation by itself. Those remain explicit
execution steps after their corresponding gates pass.

## 1. Decision to be supported

The retained claim to test is:

> On frozen globally aligned pointmaps from sparse unposed indoor images,
> learned bounded-plane support can guide geometric consensus to recover more
> accurate bounded plane partitions than geometry-only extraction, while
> remaining practical on a consumer GPU.

The plan must separately establish:

1. the gain is real under identical point caches and scene-level statistics;
2. the gain comes from support guidance rather than conflict dropping,
   coverage collapse, or scene-specific thresholds;
3. the result transfers beyond the 17 repeatedly inspected internal scenes;
4. external methods are compared only under compatible input and output
   protocols;
5. the retained inference path, not large-scale training, is feasible on an
   RTX 4060 Ti 8 GB.

PlaneGraph-BA and live plane feedback remain negative ablations. The final P1
oracle gate showed no view whose correction jointly improved fixed-gauge
structural correspondence and plane error. Do not reopen P2-P5 without new
evidence that invalidates that result.

## 2. Evidence already available

The frozen internal pilot contains 17 independent scenes. Ordinary sequential
RANSAC and learning-support-guided RANSAC consume identical ordered DUSt3R
global point caches.

| Metric | Ordinary RANSAC | Guided RANSAC | Change |
|---|---:|---:|---:|
| Pairwise F1 | 0.632406 | 0.706348 | +0.073942 |
| Matched IoU | 0.489136 | 0.678036 | +0.188899 |
| Overmerge excess | 2.882353 | 1.470588 | -1.411765 |
| VOI, nats | 1.493965 | 1.172062 | -0.321904 |
| Rand Index | 0.789837 | 0.834620 | +0.044783 |
| Symmetric Segmentation Covering | 0.561563 | 0.658550 | +0.096987 |

Guided RANSAC wins pairwise F1 on 16/17 scenes. This is a useful paired pilot,
not a public benchmark result.

The internal pilot has already established two important negative controls:

* support-only manual aggregation is useful for identity but is weaker and
  less consistent than the retained guided RANSAC across scenes;
* lowering a plane-incidence objective does not establish better registered
  geometry, even with GT plane identity or a per-view oracle rollback.

## 3. Dataset and split policy

### 3.1 Existing local data

The workstation contains 200 Structured3D scenes, `scene_00000` through
`scene_00199`, occupying about 11.73 GiB. The project dataloader makes an
internal 90/10 split, so scenes 180-199 form its local validation set and 17 of
them passed the five-view eligibility gate.

Under the official Structured3D split, however, scenes 0-2999 are training
scenes. Therefore the current 17 scenes are an internal development/pilot set,
not official validation or test data.

### 3.2 Required benchmark data

Use the official split for the paper-style extension:

```text
official training:   scene_00000 - scene_02999
official validation: scene_03000 - scene_03249
official test:       scene_03250 - scene_03499
```

Recommended frozen manifests:

| Manifest | Purpose | Minimum size | Preferred size |
|---|---|---:|---:|
| internal pilot | regression and ablation development | existing 17 | existing 17 |
| official validation | thresholds and adapter debugging | 30 eligible scenes | 50 eligible scenes |
| official test | primary synthetic benchmark | 100 eligible scenes | all eligible test scenes |
| real-domain smoke | cross-domain gate | 20 scenes | 20 scenes |
| real-domain test | generalization benchmark | 100 scenes | official protocol size |

Scene selection must be frozen before method results are inspected. Eligibility
may depend only on input completeness, view count, valid GT, and successful
coordinate/registry validation, never on method accuracy.

If only 100 official test scenes are affordable, sample them once with a fixed
seed and stratify by room/plane count using GT metadata without looking at
predictions. Preserve the full eligible inventory and rejection reasons.

### 3.3 Real-data track

The preferred real-domain track is ScanNetV2 using the published plane GT and
visibility protocol. For the primary unposed test, select sparse overlapping
views but do not supply camera poses to LightRecon3D. Poses may be used only by
the evaluator and by an explicitly labelled posed-reference track.

Do not mix the synthetic Structured3D and real ScanNet results into one mean.

## 4. Frozen evaluation contract

Every compared method for a scene must read the same:

```text
ordered RGB view manifest
DUSt3R checkpoint and preprocessing
global point cache and cache checksum
confidence mask
view registry
(alignment_view_index, x, y) point provenance
point-aligned GT and GT checksum
visibility/evaluation mask
```

Run DUSt3R inference and global alignment once per scene. Reuse the immutable
cache for every downstream baseline and ablation. Report alignment time
separately from method time.

The primary experimental unit is a scene, not a point and not an image pair.

## 5. Main comparison table

### 5.1 Primary unposed sparse-view track

| ID | Method | Role |
|---|---|---|
| B0 | DUSt3R global cache + sequential RANSAC + component split | geometry-only primary baseline |
| B1 | Stage1 support + direct global SVD, no cross-candidate merge | fragmented support baseline |
| B2 | B1 + Stage2 local geometry validation/refit | local-refit ablation |
| B3 | B2 + frozen manual merge | hand-threshold association ablation |
| B4 | B2 + learning-support-guided RANSAC proposal/consensus; ordinary DUSt3R-confidence final refit | retained historical method |
| B5 | B4 + exact conflict/provenance diagnostics + bounded export + structural lines | full system output |
| O1 | GT support with frozen geometry | support upper bound |
| O2 | GT plane identity with frozen geometry | identity upper bound |
| NEG | archived PlaneGraph-BA/live feedback result | negative geometry-feedback ablation |

B0-B5 must use a single global operating point selected on official validation.
Scene-specific thresholds are allowed only as an oracle row and cannot enter
the main ranking.

### 5.2 External unposed references

| Method | Protocol | Decision |
|---|---|---|
| Plane-DUSt3R | five sparse unposed views; room-layout output | first external priority; same-input reproduction and native metrics kept separate |
| PLANA3R | unposed two-view metric reconstruction | separate two-view table; do not present as a five-view ranking |

Plane-DUSt3R may share a table only for outputs that can be mapped without
inventing furniture/support labels. Its native room-layout IoU/depth/boundary
metrics remain a separate table.

### 5.3 Posed references

PlanarRecon, AirPlanes, and PlanarSplatting are posed-reference or upper-bound
tracks. Published numbers may be cited as related work but may not be placed
beside current Structured3D numbers as direct experimental comparisons.

Reproduce them only after the official ScanNet mesh, visibility, unit, and
plane-GT evaluator is working. The first practical posed reference should be
the method whose official test split and evaluator can be reproduced with the
least adaptation.

## 6. Required ablations

Run the following first on the 17-scene pilot, then promote only decisive rows
to the official benchmark.

### A. Source of the retained gain

1. no support guidance;
2. predicted support used for proposal sampling only;
3. predicted support used for consensus scoring only;
4. predicted support used for final refit only;
5. proposal + consensus, the retained historical B4 method;
6. proposal + consensus + support-weighted refit, an added mechanism row;
7. GT support guidance upper bound.

This identifies whether Stage1 helps hypothesis generation, assignment, or
parameter estimation.

### B. Local proposal processing

1. Stage2 local refit off/on;
2. radius-connected component split off/on;
3. confidence filtering off/on;
4. bounded support retained versus infinite-plane-only fitting;
5. 2, 3, and 5 input views using pre-registered view subsets.

### C. Provenance and conflict handling

1. preserve repeated records with an explicit null/conflict state, primary;
2. exact repeated-key consensus disabled;
3. conflicting repeated keys dropped, diagnostic only;
4. majority-label conflict resolution, diagnostic only;
5. geometry-only constrained agglomeration;
6. provenance-aware deterministic consensus if Phase-I implementation is
   resumed.

Any score from conflict dropping must report how many records and unique keys
were removed. It cannot be the primary result.

### D. Robustness

1. three frozen RANSAC seeds per scene;
2. three view-order permutations on the internal pilot;
3. one global threshold selected on validation versus a scene-specific oracle;
4. clean/simple versus hard/cluttered scene strata;
5. assigned, unassigned, and invalid-point rates;
6. success/failure rate with every failed scene retained in the ledger.

Average stochastic repeats within each scene before computing cross-scene
statistics. Seeds are not independent scene samples.

### E. Efficiency and consumer-GPU trade-off

1. cache hit versus uncached end-to-end execution;
2. two, three, and five views;
3. Stage1, Stage2, alignment, RANSAC, metrics, and export timing boundaries;
4. peak allocated and peak reserved CUDA memory;
5. CPU RAM, checkpoint bytes, cache bytes, and output bytes per scene;
6. optional reduced-resolution diagnostic only after the 512-resolution
   primary run is established.

Do not change the primary 512-resolution protocol merely to improve the
consumer-GPU result.

## 7. Metrics

### 7.1 Primary plane-partition metrics

```text
pairwise precision / recall / F1
VOI, in nats
Rand Index
Segmentation Covering, pred-to-GT, GT-to-pred, and symmetric
matched IoU
fragmentation excess
overmerge excess
assignment rate
GT coverage
```

Pre-register scene-level pairwise F1 as the primary internal metric. VOI, RI,
SC, matched IoU, and overmerge are required confirmatory metrics.

### 7.2 Bounded support and geometry

```text
support-conditioned IoU
boundary F-score at a declared pixel or metric tolerance
normal angular error
point-to-plane residual mean and p95
Chamfer distance and F-score when mesh GT and visibility masks are valid
large-plane fidelity, accuracy, and planar Chamfer on compatible public data
```

Sparse support metrics must always be accompanied by dense coverage and label
rate.

### 7.3 Pose and system

```text
RRA / RTA / mAA on a compatible unposed protocol
end-to-end runtime
alignment runtime
downstream method runtime
peak allocated and reserved GPU memory
trainable parameter count
checkpoint and cache bytes
valid-scene rate
```

Speed is diagnostic. The existing efficiency gate failed, so no universal
acceleration claim is allowed without new evidence.

## 8. Statistical analysis

For every retained main comparison:

1. report each scene row and the aggregate mean and median;
2. report mean and median paired differences;
3. compute a scene-level paired bootstrap 95% confidence interval;
4. report win/tie/loss counts and an exact sign test;
5. use a paired Wilcoxon test as a secondary sensitivity analysis when the
   sample size is sufficient;
6. apply Holm correction within each ablation family rather than across every
   exploratory diagnostic;
7. report failure rate and do not silently remove failed or OOM scenes.

Do not use millions of points as independent samples for significance.

## 9. RTX 4060 Ti 8 GB feasibility

### 9.1 Verified local facts

```text
GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB
current desktop GPU use during audit: about 1403 MiB
system RAM: 31.8 GiB
free E: drive space during audit: 32.8 GiB
Python: E:\anaconda\python.exe
PyTorch: 2.6.0+cu126
CUDA available: yes
local DUSt3R 512 checkpoint: present, about 2.13 GiB
local Structured3D subset: 200 scenes, about 11.73 GiB
```

The frozen server efficiency run at image size 512 and batch size 1 measured:

| Component | Peak allocated memory |
|---|---:|
| DUSt3R pair backbone, two images | 3084.7 MiB |
| Stage1 support head, one image | 2959.2 MiB |
| Stage2 MLP | 2598.4 MiB process peak, about 1.1 MiB incremental |
| five-view inference and global alignment | 2745.8 MiB |

These measurements strongly suggest that the retained inference path should
fit in 8 GB when run at batch size 1 and one scene at a time. They do not prove
Windows peak reserved memory or allocator behavior, so a local smoke is still
required.

### 9.2 Feasibility by experiment class

| Experiment | 4060 Ti 8 GB judgment | Conditions |
|---|---|---|
| Read archived NPZ and recompute metrics | yes | CPU-only or negligible GPU |
| Cache-only RANSAC and ablations | yes | sequential scenes; CPU RAM is sufficient |
| DUSt3R pair inference at 512 | verified yes | batch size 1; local peak reserved memory remained below 3.0 GiB |
| Five-view global alignment at 512 | verified yes | one scene at a time; 300-iteration local smoke passed |
| Stage1 + Stage2 inference | verified yes | frozen checkpoints synchronized and loaded in the `ai` environment |
| Full 17-scene internal rerun | verified yes | 17/17 passed locally; runtime, rather than VRAM, is the constraint |
| 100-scene frozen inference benchmark | yes in principle | sequential execution, resumable ledger, more disk headroom |
| Stage1 training from cached features | possible but not currently justified | small batch, AMP/gradient accumulation only after an explicit training decision |
| End-to-end DUSt3R or Plane-DUSt3R training | no practical local plan | optimizer/activation memory and duration exceed the intended consumer-GPU scope |
| Plane-DUSt3R inference | unverified/high risk | current official environment is not installed; 4.41 GiB checkpoint and MMCV stack require a one-scene Linux/server smoke |
| PLANA3R/posed external methods | unverified | separate environment and official smoke required |

### 9.3 Original local blockers and current status

1. The canonical Stage1 checkpoint
   `local_outputs/stage1_large_80m_train2048_sharded_v1/best.pt` is not present
   locally. Only the smaller local smoke checkpoint was found.
2. The server's final Stage2 RegionMergeMLP checkpoint was not found under the
   local experiment outputs.
3. The frozen 17-scene full global caches and machine-readable result bundle
   are server paths and are not currently mirrored locally.
4. Server manifests contain `/gemini/...` absolute paths and need an audited
   local path map, not string guessing.
5. Only 32.8 GiB was free on the data drive. This is enough for a smoke and
   careful subset work, but not comfortable for official validation/test,
   external checkpoints, immutable caches, and ScanNet together. Reserve at
   least 100 GiB before a public benchmark because repository policy forbids
   deleting or overwriting retained outputs.
6. Plane-DUSt3R requires an isolated Python 3.11 / PyTorch 2.2.0 / CUDA 11.8 /
   MMCV-compatible environment. The current Windows PyTorch 2.6 environment
   is not a verified substitute.

Status update on 2026-07-18: blockers 1-4 were resolved by the manually
transferred bootstrap archive, audited path remapping, and fresh local
materialization. The complete 17-scene retained path now runs in the `ai`
environment. Disk capacity remains a planning constraint for official-scale
immutable caches, and the isolated Plane-DUSt3R environment remains an
external-method blocker rather than a blocker for B0/B4.

### 9.4 Mandatory local memory smoke

Use one internal development scene and the exact retained five views.

```text
batch_size = 1
image_size = 512
global_alignment_iterations = 300
one scene per process
no training or recaching beyond the smoke output
```

Record:

```text
nvidia-smi before and after
torch.cuda.max_memory_allocated
torch.cuda.max_memory_reserved
wall time per stage
cache size
output checksum and validity
```

Pass the local feasibility gate when the run completes without OOM and peak
reserved memory stays below 6.5 GiB. The threshold leaves headroom for WDDM
and desktop use. If it fails, first close GPU-heavy applications and isolate
stages into separate processes. Do not silently lower image resolution or
change the method before recording the failed primary smoke.

## 10. Execution order and gates

### Phase 0: freeze and synchronize evidence

1. Copy, do not regenerate, the canonical 17-scene JSON/CSV/NPZ bundle and
   final Stage1/Stage2 checkpoints from the server.
2. Verify SHA-256 hashes and create local path maps.
3. Recompute existing metrics read-only and require exact agreement with the
   frozen report.

Gate: no benchmark work until checksums and the 17-scene aggregate table agree.

### Phase 1: local 4060 Ti smoke

1. run one pair-backbone memory smoke;
2. run Stage1 and Stage2 inference smoke;
3. run one five-view global alignment;
4. run B0 and B4 from the same new cache;
5. archive memory, runtime, and cache metadata.

Gate: primary 512-resolution inference fits below the 6.5 GiB reserved-memory
budget, or the local machine is restricted to cache-only experiments.

### Phase 2: internal ablation package

1. run B0-B5 and O1/O2 on the existing 17 scenes from immutable caches;
2. run three RANSAC seeds and robustness controls;
3. run the proposal/consensus/refit component ablation;
4. run conflict/provenance controls without dropping records in the primary
   row;
5. produce scene-level CSV, aggregate JSON, plots, and failure gallery.

Gate: B4 must retain a positive median F1 gain, win at least 70% of scenes,
avoid a coverage collapse, and improve at least two confirmatory partition
metrics. Otherwise stop the paper-style expansion and retain the result as an
engineering project.

### Phase 3: official Structured3D benchmark

1. obtain official validation/test scenes under the dataset agreement;
2. freeze the complete eligibility inventory and manifests;
3. select all thresholds on validation only;
4. run a 10-scene end-to-end canary;
5. run the frozen 100-scene minimum or all eligible official test scenes;
6. evaluate without reading intermediate aggregate results for threshold
   changes.

Gate: B4 must reproduce a consistent scene-level gain with a paired 95% CI
that does not collapse around zero. If it fails, perform failure analysis but
do not retune on test.

### Phase 4: closest external unposed comparison

1. finish the isolated Plane-DUSt3R environment;
2. run one-scene GPU smoke;
3. run five compatibility scenes;
4. run the same frozen manifest when output adaptation is semantically valid;
5. report native room-layout metrics separately from shared partition metrics;
6. add PLANA3R only as a separate two-view protocol after its official adapter
   is verified.

Gate: every external scene must have a success/failure ledger. Silent scene
skipping invalidates the aggregate.

### Phase 5: real-domain generalization

1. implement the official ScanNet mesh/visibility/plane-GT evaluator;
2. freeze 20 real-domain smoke scenes;
3. verify units, visibility, and sparse-view overlap;
4. run B0/B4 and one posed reference;
5. promote to the 100-scene protocol only if at least 90% of smoke scenes are
   valid and failures are not caused by an unresolved adapter error.

This phase, rather than more synthetic training, supplies the first meaningful
generalization evidence.

### Phase 6: conditional training only

Do not train a new Stage1 model unless the frozen model fails official or
cross-domain evaluation and the intended claim explicitly becomes a learned
method claim.

If training is authorized:

1. freeze a clean training split disjoint from validation and test;
2. retain the existing architecture first;
3. use at least three seeds;
4. select checkpoints and thresholds on validation only;
5. compare against the frozen checkpoint under the same test manifest;
6. run training on the server, while the 4060 Ti remains suitable for smoke,
   debugging, and cached-feature experiments.

## 11. Required artifacts for every retained run

```text
Git SHA and dirty-worktree status
full command line
environment package versions
GPU/CPU/RAM record
input and selection manifest
view registry and coordinate convention
checkpoint and input SHA-256 values
global-cache checksum
method configuration and random seed
per-scene metrics row
aggregate metrics JSON/CSV/Markdown
runtime and peak allocated/reserved memory
failure and OOM ledger
visualization index
explicit pass/fail gate decision
```

All output directories are immutable. Recovery runs write new directories and
are merged through a recorded ledger.

## 12. Minimum evidence packages

### Undergraduate/research-practice completion

Already sufficient after artifact integrity checks:

* frozen 17-scene paired result;
* W3 accuracy/latency/memory result;
* B0-B5 ablation table;
* qualitative success and failure cases;
* negative feedback ablation.

### Paper-style internal result

Minimum additional evidence:

* complete 17-scene component and robustness ablations;
* official Structured3D test with at least 100 eligible scenes;
* Plane-DUSt3R same-input comparison with protocol caveats;
* scene-level confidence intervals and failure ledger;
* consumer-GPU memory/runtime smoke.

### Strong paper-style benchmark

Preferred evidence:

* all eligible official Structured3D test scenes;
* 100-scene real ScanNet plane benchmark;
* closest unposed external method and at least one posed reference;
* cross-domain failure analysis;
* official evaluators, released manifests, and reproducible adapters.

## 13. Immediate next action

Phase 0, Phase 1, the primary B0/B4 row, and the single-seed
proposal/consensus/refit mechanism ablation are complete locally. The next
bounded work package is the remaining Phase 2 evidence: two additional frozen
RANSAC seeds, the remaining B0-B5 and O1/O2 rows, robustness controls, and a
failure gallery centred on `scene_00187`. Do not start the official test split
until those rows and their decision table are complete.

## 14. Local 4060 Ti gate result: 2026-07-18

The bootstrap archive and both frozen downstream checkpoints were downloaded
and verified locally. The outer archive SHA-256 matched, and all 21 extracted
file hashes matched the server checksum set. The Stage1 epoch-23 and Stage2
epoch-9 checkpoint payloads both load with local PyTorch.

The smoke used the user-designated Conda `ai` environment:

```text
Python: E:\anaconda\envs\ai\python.exe
PyTorch: 2.9.1+cu126
GPU: NVIDIA GeForce RTX 4060 Ti, 8187.5 MiB
```

Five-view DUSt3R global-alignment smoke:

```text
scene / space: scene_00180 / 445895
input: positions 0..4, perspective/empty/rgb_rawlight.png
preprocessed image size: 512 x 288
ordered pairs: 20
global-alignment iterations: 300
alignment loss: 0.0076863365
model load: 5.332 s
five-view inference + alignment: 41.314 s
total through alignment: 46.646 s
peak allocated: 2740.4 MiB
peak reserved: 2866.0 MiB
cache points: 737,280, all finite
cache bytes: 12,511,798
```

Simultaneous DUSt3R backbone + frozen Stage1 head + frozen Stage2 MLP smoke:

```text
two-view inference: 0.613 s
target pointmap: 288 x 512
Stage1 predicted plane count: 5
Stage2 dummy candidate batch: 64 x 14 -> 64 outputs
peak allocated: 2837.0 MiB
peak reserved: 3004.0 MiB
```

The local inference gate passes with substantial margin below the registered
6.5 GiB reserved-memory ceiling. Cache-only ablations, the retained inference
path, and sequential 100-scene execution are feasible on the 4060 Ti from a
VRAM perspective. Runtime and disk, rather than VRAM, are now the main local
constraints.

Generated evidence:

```text
local_outputs/rtx4060ti_dust3r_smoke_scene00180_20260718_v1/smoke_report.json
local_outputs/rtx4060ti_dust3r_smoke_scene00180_20260718_v1/scene_00180_445895_dust3r_global_cloud_cache.npz
local_outputs/rtx4060ti_dust3r_smoke_scene00180_20260718_v1/scene_00180_445895_global_alignment_diagnostic.png
local_outputs/rtx4060ti_stage_heads_smoke_scene00180_20260718_v3/smoke_report.json
```

The diagnostic PNG was inspected after generation. Its three viewpoints show
a consistent globally aligned room shell, wall/ceiling structure, and window
region. It is a sparse point-cloud diagnostic, not a photorealistic quality
claim.

## 15. Full local 17-scene fresh-cache result: 2026-07-18

The complete local run used the user-designated Conda `ai` environment and
the frozen 17-scene manifest. Stage1 and Stage2 inputs were materialized for
all 17 scenes in 673.443 seconds. The retained global-alignment, B0/B4, and
evaluation batch then passed 17/17 scenes with no OOM or failed item in
5186.253 seconds (86.44 minutes). Together these two measured stages took
97.66 minutes; setup, smoke runs, and report generation are excluded.
The combined ledger references the first five completed items from `b17u4`
and the remaining twelve from `b12r2`; `scene_00186` reused its already
validated local global cache while the other recovery items recomputed global
alignment. No Stage2 NPZ or retained cache was overwritten.

This is a new local fresh-cache result. It must not be substituted silently
for the older archived/server pilot in Section 2 because the per-scene cache
realizations and the single negative scene differ.

| Metric | B0 ordinary RANSAC | B4 guided RANSAC | Change / result |
|---|---:|---:|---:|
| Pairwise F1, mean | 0.636284 | 0.706531 | +0.070247 |
| Pairwise F1, median | 0.569644 | 0.659044 | paired median +0.046451 |
| Purity/completeness F1, mean | 0.719681 | 0.796301 | +0.076620 |
| Matched IoU, mean | 0.517580 | 0.677549 | +0.159968 |
| Plane precision, mean | 0.381373 | 0.558226 | +0.176853 |
| Recall over all GT planes, mean | 0.369177 | 0.559513 | +0.190336 |
| Normal error, mean, degrees | 9.124512 | 4.291560 | -4.832952 |
| Overmerge excess, mean | 2.647059 | 1.470588 | -1.176471 |
| Coverage, mean | 0.998283 | 0.999600 | +0.001317 |
| Runtime, median, seconds | 91.816 | 108.926 | ratio 1.148833 |

B4 wins pairwise F1 on 16/17 local scenes. The only loss is
`scene_00187`, from 0.802746 to 0.800094, a change of -0.002652. The exact
two-sided sign-test p-value is 0.000275. A paired scene-bootstrap estimate of
the mean F1 change is +0.070247 with an exploratory 95% interval of
[+0.038461, +0.106371] using 10,000 resamples and seed 20260716.

The frozen Phase 2 quality gate passes: positive median F1 gain, 94.12% scene
win rate, no coverage collapse, and improvements in multiple confirmatory
partition metrics. The efficiency gate fails because the median B4/B0 runtime
ratio is 1.148833 rather than at most 0.75. The supported claim is therefore
quality improvement, not speed improvement.

Conflict-drop rows remain negative controls. Their median full-cache coverage
is 0.014761 for manual conflict-drop and 0.000012 for direct conflict-drop;
neither should replace the primary no-drop row.

Canonical local evidence:

```text
manifest:
  local_outputs/local_17scene_recache_20260718_v3/selection/final_unique_scenes_execute.json
manifest SHA-256:
  198b5ac1d4895fd8e9a344c9bbc3991b3cc1df37128fa87224f7514a9cd1a6ef
combined batch:
  local_outputs/local_17scene_benchmark_20260718_v2/
statistical audit:
  local_outputs/local_17scene_benchmark_audit_20260718_v2/
```

`local_17scene_benchmark_20260718_v1` is retained but is not canonical: its
derived metadata placed the selection-plan hash in the manifest-hash field.
The underlying scene outputs were not altered. Version 2 corrects provenance
by referencing the executed manifest and its verified SHA-256.

### 15.1 What the 4060 Ti can run

The verified smoke peak was 2866 MiB reserved for five-view DUSt3R global
alignment and 3004 MiB reserved for simultaneous backbone plus Stage1/Stage2
heads. This is well below the 6.5 GiB gate and the physical 8 GiB limit. The
17-scene run then completed sequentially without OOM. Consequently this
machine is suitable for the remaining cache-only ablations, frozen inference,
failure analysis, and a resumable 100-scene benchmark from a VRAM perspective.

At the measured local rate, a naive 100-scene extrapolation is approximately
9.6 hours for materialization plus the retained batch, before adapters and
reporting. This estimate is directional because scene complexity varies.
End-to-end DUSt3R/Plane-DUSt3R training is still outside the practical local
scope; it requires a server-class training environment, while the 4060 Ti is
the inference and ablation machine.

### 15.2 Next experimental package

1. Run B0-B5 and O1/O2 on the same immutable 17-scene caches.
2. Add two frozen seeds to the completed seed-0 mechanism rows and average
   stochastic repeats within each scene before cross-scene statistics.
3. Run the remaining provenance and conflict-handling controls; do not repeat
   the completed proposal/consensus/refit seed-0 row.
4. Produce a scene-level plot and a failure gallery for `scene_00187` plus the
   largest positive and slowest scenes.
5. Freeze the retained configuration, then run a 10-scene official-split
   canary followed by at least 100 eligible Structured3D validation/test
   scenes. Thresholds must be selected on validation only.

## 16. Guided-RANSAC mechanism ablation: 2026-07-19

The historical B4 implementation was audited before defining the switches.
Learned support affected proposal generation and learned-candidate consensus
scoring. Its final plane refit used ordinary DUSt3R confidence over the global
inliers; it did not apply support weights. Therefore the historical B4 row is
`proposal_consensus`, while `all` is a new proposal + consensus +
support-weighted-refit row. The executor verifies historical B4 equivalence on
assignments, plane parameters, point order, view registry, and pixel registry.

The seed-0 cache-only run used the same 17 immutable global clouds, GT files,
geometry thresholds, and evaluation path for all six mechanism modes. It
completed 17/17 scenes, 102/102 mode executions, and 17/17 historical B4
equivalence checks with zero failed items. The measured executor runtime was
8634.643 seconds (143.91 minutes). It used CPU-side cached processing and did
not rerun DUSt3R, Stage1, Stage2, or global alignment; no GPU OOM occurred.

| Mode | Proposal | Consensus | Support refit | Pairwise F1 mean | median | Runtime mean, s |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | 0 | 0 | 0.635813 | 0.569485 | 101.290 |
| proposal only | 1 | 0 | 0 | 0.707222 | 0.658716 | 105.668 |
| consensus only | 0 | 1 | 0 | 0.635813 | 0.569485 | 95.498 |
| refit only | 0 | 0 | 1 | 0.635643 | 0.576892 | 114.794 |
| proposal + consensus, historical B4 | 1 | 1 | 0 | 0.706531 | 0.659044 | 103.685 |
| all | 1 | 1 | 1 | 0.706582 | 0.659198 | 111.492 |

The paired Pairwise-F1 statistics use 10,000 scene bootstrap resamples with
seed 20260719. Holm correction covers the six planned mechanism contrasts.

| Contrast | Mean delta | Median delta | Bootstrap 95% CI | W/T/L | Holm sign-test p |
|---|---:|---:|---:|---:|---:|
| proposal vs none | +0.071410 | +0.047196 | [+0.038900, +0.106941] | 16/0/1 | 0.001648 |
| consensus vs none | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0/17/0 | 1.000000 |
| refit vs none | -0.000169 | +0.000179 | [-0.004269, +0.003653] | 10/0/7 | 1.000000 |
| consensus after proposal | -0.000691 | +0.000007 | [-0.002226, +0.000103] | 11/4/2 | 0.067383 |
| refit after proposal + consensus | +0.000051 | +0.000062 | [-0.000003, +0.000095] | 15/0/2 | 0.009399 |
| all vs none | +0.070769 | +0.047203 | [+0.038621, +0.105954] | 16/0/1 | 0.001648 |

Mechanism conclusion: learned proposal generation accounts for essentially
all retained F1 gain. Consensus-only is structurally inert without learned
proposals, and its incremental effect after proposal is not robust after Holm
correction or bootstrap. Support-weighted refit produces a consistently tiny
increment after B4, but the mean gain is only 0.000051 and its bootstrap
interval touches zero. This is statistically interesting but practically
negligible; it does not justify replacing historical B4 with `all`. Retain B4
for backward compatibility and report proposal-only as the decisive component
ablation. This remains an internal 17-scene, single-seed conclusion pending
the two additional frozen seeds.

Canonical evidence:

```text
execution and aggregate metrics:
  local_outputs/guided_ransac_mechanism_17scene_seed0_20260719_v3/
paired statistical audit:
  local_outputs/guided_ransac_mechanism_statistics_17scene_seed0_20260719_v1/
```

### 16.1 Three-seed closure

Frozen seeds 1 and 2 subsequently completed on the same 17 immutable caches.
All three runs passed 17/17 scenes and 102/102 mechanism executions with zero
failed items. The final seed-2 recovery reused checksum-valid predictions and
completed its previously transient evaluation failure in 71.429 seconds.
Statistics first average seeds 0, 1, and 2 within each scene, then treat the 17
scene means as the independent paired samples.

| Mode | Pairwise F1 mean | Median scene mean | Runtime mean, s |
|---|---:|---:|---:|
| none | 0.637278 | 0.591244 | 63.379 |
| proposal only | 0.702667 | 0.658707 | 76.209 |
| consensus only | 0.637355 | 0.591244 | 63.290 |
| refit only | 0.637201 | 0.597748 | 70.944 |
| proposal + consensus, historical B4 | 0.702452 | 0.659031 | 69.457 |
| all | 0.702504 | 0.659184 | 72.530 |

| Contrast | Mean delta | Median delta | Bootstrap 95% CI | W/T/L | Holm sign-test p |
|---|---:|---:|---:|---:|---:|
| proposal vs none | +0.065389 | +0.049474 | [+0.036570, +0.097369] | 16/0/1 | 0.001648 |
| consensus vs none | +0.000077 | +0.000000 | [+0.000000, +0.000230] | 1/16/0 | 1.000000 |
| refit vs none | -0.000077 | +0.000026 | [-0.002308, +0.001733] | 9/0/8 | 1.000000 |
| consensus after proposal | -0.000215 | +0.000000 | [-0.000742, +0.000073] | 8/2/7 | 1.000000 |
| refit after proposal + consensus | +0.000053 | +0.000061 | [+0.000001, +0.000095] | 15/0/2 | 0.009399 |
| all vs none | +0.065226 | +0.049471 | [+0.036471, +0.097004] | 16/0/1 | 0.001648 |

The three-seed result closes the mechanism robustness family with the same
decision as seed 0. Proposal guidance explains essentially all useful gain.
Consensus and support-weighted refit add effects that are negligible in
magnitude; retain historical B4 for compatibility and do not promote `all`.

Canonical three-seed evidence:

```text
seed 0: local_outputs/guided_ransac_mechanism_17scene_seed0_20260719_v3/
seed 1: local_outputs/guided_ransac_mechanism_17scene_seed1_20260719_v1/
seed 2: local_outputs/guided_ransac_mechanism_17scene_seed2_20260719_v3/
statistics: local_outputs/guided_ransac_mechanism_statistics_17scene_seeds012_20260719_v1/
```
