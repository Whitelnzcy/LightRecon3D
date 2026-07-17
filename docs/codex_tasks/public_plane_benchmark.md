# Public Plane Benchmark Track

Date: 2026-07-17

Status: active after the 17-scene internal pilot. This track does not change the
frozen research-practice result. Its purpose is to determine whether the work
can support a paper-style comparison.

## 1. Evidence boundary

The recovered Structured3D batch contains 17 independent scenes. Ordinary and
learning-support-guided RANSAC read identical ordered DUSt3R global point
caches. Guided RANSAC improves mean point-partition F1 from `0.632406` to
`0.706348`, matched IoU from `0.489136` to `0.678036`, and reduces mean
overmerge excess from `2.882353` to `1.470588`. It wins F1 on 16/17 scenes.

This is an internal paired comparison. It is not a public leaderboard result
because the dataset subset, point-aligned GT domain, output representation and
metrics do not yet match a published benchmark end to end.

## 2. Comparison tracks

Methods with different pose assumptions must not be mixed into one ranking.

### Track A: unposed sparse views (primary)

| Method | Input condition | Role | Integration status |
|---|---|---|---|
| DUSt3R + sequential RANSAC | five unposed views | geometry-only baseline | complete |
| LightRecon3D guided RANSAC | same five unposed views | retained method | complete |
| Plane-DUSt3R, ICLR 2025 | unposed sparse views; Structured3D fine-tuning | closest published DUSt3R-family reference | adapter and checkpoint reproduction required |
| PLANA3R, NeurIPS 2025 | unposed two-view images; metric planar primitives | recent pose-free reference | separate two-view protocol required |

Plane-DUSt3R reconstructs room layout rather than the same unrestricted plane
partition used here. It may enter a shared table only after its output is
converted without inventing missing furniture/support labels and evaluated on
the same held-out scenes. PLANA3R's native two-view setting must remain a
separate table unless an official multi-view evaluation path is reproduced.

### Track B: posed multi-view methods (reference/upper-bound)

| Method | Input condition | Public protocol |
|---|---|---|
| PlanarRecon, CVPR 2022 | posed video | ScanNetV2 plane reconstruction |
| AirPlanes, CVPR 2024 | posed RGB sequence | ScanNetV2 `testplanes`, 100 scenes |
| PlanarSplatting, CVPR 2025 | posed RGB images | ScanNetV2 100 scenes; ScanNet++ 30 scenes |

These methods may be evaluated as posed references. Their published numbers
must not be placed beside the current 17-scene Structured3D numbers as if the
inputs and GT were identical.

Primary sources:

* Plane-DUSt3R: https://openreview.net/forum?id=DugT77rRhW
* PLANA3R: https://papers.nips.cc/paper_files/paper/2025/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html
* AirPlanes: https://openaccess.thecvf.com/content/CVPR2024/html/Watson_AirPlanes_Accurate_Plane_Estimation_via_3D-Consistent_Embeddings_CVPR_2024_paper.html
* PlanarSplatting: https://openaccess.thecvf.com/content/CVPR2025/html/Tan_PlanarSplatting_Accurate_Planar_Surface_Reconstruction_in_3_Minutes_CVPR_2025_paper.html

## 3. Unified evaluation

Every adapter must declare:

```text
dataset and split
scene ID and selected image IDs
number of views
whether camera intrinsics/poses are supplied
metric scale or similarity-aligned scale
predicted geometry representation
plane instance IDs and unassigned mask
visibility/evaluation mask
runtime boundary and hardware
```

The canonical evaluation has three families.

### Plane partition

```text
VOI (lower is better)
Rand Index (higher is better)
Segmentation Covering in both directions and their mean
pairwise precision/recall/F1
matched IoU
fragmentation and overmerge
assignment rate and GT coverage
```

The first three metrics follow the evaluation family used by PlanarRecon,
AirPlanes, NeuralPlane and PlanarSplatting. The local point-aligned version
retains predicted `-1` points as one unassigned segment. It therefore cannot
gain by dropping difficult regions.

### Surface geometry

```text
Chamfer distance
F-score at a declared distance threshold
normal angular error
point-to-plane residual
planar fidelity, accuracy and chamfer for large GT planes when mesh GT exists
```

The current point-aligned cache is suitable for correspondence and plane
partition analysis. Published ScanNet geometry numbers require the official
mesh sampling, visibility masks and metric units; they cannot be reconstructed
from the current Structured3D point subset alone.

### Pose and system

```text
RRA/RTA/mAA when a comparable unposed pose protocol is available
end-to-end runtime
plane-stage runtime
peak GPU memory
trainable parameter count and checkpoint bytes
```

Speed is diagnostic, not the main claim.

## 4. Scale and training decision

The next experiment is large-scale **testing**, not immediate large-scale
training.

1. Recompute VOI/RI/SC for all 17 frozen scenes without rerunning inference.
2. Inventory every eligible held-out Structured3D scene and freeze a manifest
   before looking at results.
3. Run the frozen ordinary/guided methods on all eligible held-out scenes. A
   paper-scale target should be comparable to the 100-scene ScanNetV2 protocol,
   but the exact count is determined by the eligible-scene inventory rather
   than chosen after seeing outcomes.
4. Reproduce Plane-DUSt3R on a small compatibility subset, then the same frozen
   held-out manifest if its official output can be evaluated fairly.
5. Add a separate two-view PLANA3R table only if its official inference and
   metric scale can be reproduced.
6. Add ScanNetV2/ScanNet++ posed-reference experiments only after the canonical
   mesh/visibility evaluator is working.

Do not start a new foundation-model training run. Stage1 is already learned and
its training provenance must be documented. New training is justified only if
the frozen model fails the held-out or cross-dataset evaluation and the paper
claim explicitly becomes a learned-method claim. In that case, freeze a larger
clean training split, retain the architecture, run at least three seeds, and
keep all benchmark test scenes isolated from training and threshold selection.

## 5. Immediate execution

First run the read-only 17-scene public-metric recomputation:

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/research_practice_public_metrics_20260717_v1 \
bash run_research_practice_public_metrics.sh
```

Expected outputs:

```text
public_plane_metrics.json
public_plane_metrics.csv
public_plane_metrics.md
```

This job is CPU-only and reads the archived NPZ files. It does not rerun
DUSt3R, Stage1, RANSAC, or training.
