# Structured Plane Reconstruction Literature Audit

Date: 2026-07-12

Status: first-pass novelty audit. This document records claims supported by the
papers inspected so far; it is not yet a complete systematic review.

## 1. Scope

Target problem:

```text
unordered indoor RGB images without supplied camera poses
-> foundation-model pointmaps and confidence
-> structured global plane primitives
```

A desired primitive is not only an infinite plane equation. It is:

```text
Pi_k = (normal, offset, bounded support, observations, uncertainty)
```

The audit asks which parts are already established in prior work and which
research hypothesis remains defensible for LightRecon3D.

## 2. Closest prior work and consequences

| Work | Setting | Main capability | Consequence for this project |
|---|---|---|---|
| PlaneRCNN, CVPR 2019 | single image | plane masks, normals and depth with nearby-view refinement | Single-view plane instance masks and parameters are not novel. |
| PlaneAE, CVPR 2019 | single image | associative pixel embeddings and arbitrary plane instances | Embedding-based plane grouping is not novel. |
| PlaneTR, ICCV 2021 | single image | query-based plane recovery with structural line cues | Plane queries and structure-aware decoding are not novel. |
| SparsePlanes, ICCV 2021 | two sparse views, unknown pose | jointly reasons about relative camera pose, plane correspondence and reconstruction | “Unknown-pose planar reconstruction” alone is not novel. |
| PlaneMVS, CVPR 2022 | multi-view, known poses | plane-sweep MVS plus semantic plane detection and soft pooling | Combining 2D plane masks with multi-view plane geometry is not novel. |
| PlanarRecon, CVPR 2022 | posed monocular video | fragment-level 3D plane detection, learned tracking and fusion | Learned plane tracking/fusion is not novel. It also handles far-apart surfaces with similar plane parameters using centroid voting. |
| Good Configurations of Planar Primitives, CVPR 2022 | unorganized point cloud | jointly refines plane parameters and discrete point assignments with create/remove/modify operators | Better-than-RANSAC primitive configuration from point clouds is established. |
| AirPlanes, CVPR 2024 | posed RGB sequence | strong geometry+RANSAC baseline and multi-view-consistent plane embeddings | Cross-view-consistent plane embeddings and “semantics complement geometry” are not novel. This is a mandatory baseline/reference. |
| PlanarSplatting, CVPR 2025 | posed multi-view indoor images | directly optimizes 3D plane primitives by differentiable planar splatting | Direct explicit planar primitive optimization is not novel. |
| PlaneRAS, ICCV 2025 | monocular image | reconstructs local 3D planar primitives, aggregates them into global plane instances and splats for supervision | Query-to-3D primitive reconstruction and primitive aggregation are not novel. |
| GSPlane, 2025 preprint | posed images + GS | lifts SAM/normal planar priors into a 3D graph, regularizes planar Gaussians and refines mesh layout | 2D planar prior lifting, 3D planar grouping, structured mesh refinement and editing claims are occupied. |
| Peek-a-Boo, CVPR 2020 | single image, multi-view training | visible and full/occluded plane masks with plane-warping supervision | Visible-vs-occluded plane extent is not a new representation by itself. |
| PolyFit, ICCV 2017 | point cloud | intersects detected planes and selects polygonal faces under manifold/watertight constraints | Polygonal plane assembly and watertight structured output are not novel. |
| Primitive Assembly, ICCV 2023 | point cloud/CAD | candidate primitive patches plus pruning and binary optimization | Structure-aware primitive assembly is not novel. |

## 3. Claims that must not be used as novelty

The following are existing techniques or combinations with close precedents:

1. Predicting plane instances with learned queries.
2. Fitting plane equations with SVD or RANSAC.
3. Combining a 2D plane mask branch with a geometry branch.
4. Tracking or merging planes across views with learned descriptors.
5. Separating distant coplanar surfaces using centroid/location cues.
6. Learning multi-view-consistent plane embeddings.
7. Representing a scene with explicit planar primitives.
8. Predicting visible and occluded/full plane masks.
9. Intersecting planes to obtain polygonal or watertight models.
10. Claiming editability merely because plane parameters can be changed.

## 4. The remaining gap is narrower than the current project description

The clearest difference between the closest methods and the current setting is
the combination of all of the following conditions:

```text
unordered views
+ no supplied camera poses
+ geometry from a general-purpose pointmap foundation model
+ alignment and geometry are uncertain/nonuniform
+ output requires bounded instance identity, not only plane equations
```

This combination is not enough by itself to establish novelty. A new mechanism
must address a failure specifically caused by this setting.

The most promising failure is that foundation-model geometry provides neither
uniform metric reliability nor exact multi-view surface coincidence. Existing
posed-video methods typically assume known poses and build TSDF/voxel/mesh
representations before plane grouping. Simply copying their association logic
onto noisy aligned pointmaps can cause:

* high-confidence and low-confidence observations to contribute equally;
* alignment errors to be mistaken for distinct parallel planes;
* coplanarity to merge disconnected semantic instances;
* visibility gaps to be mistaken for support boundaries;
* repeated pixels/views to dominate a plane refit;
* a single hard merge to irreversibly corrupt identity.

## 5. Second-pass audit: additional closest works

The first candidate in this document is weakened by several additional works
found in the second pass:

| Work | Occupied contribution space |
|---|---|
| PlaneFormers, 2022 | 3D-aware plane tokens jointly reason about sparse-view correspondence and relative pose. |
| NOPE-SAC, 2023 | Learns one-plane pose hypotheses in a RANSAC framework for sparse two-view reconstruction. |
| PlaneRecTR++, 2023 | Unifies plane detection, segmentation, parameter regression, correspondence and pose estimation without initial pose or correspondence supervision. |
| PlanarNeRF, 2024 | Maintains globally consistent cross-frame planar primitives with a global memory bank. |
| AlphaTablets, NeurIPS 2024 | Represents planes as bounded rectangles with alpha maps, differentiably optimizes irregular boundaries, merges primitives and supports editing. |
| NeuralPlane, ICLR 2025 | Distills inconsistent multi-view 2D plane observations into a unified neural coplanarity field without plane annotations. |
| Plane-DUSt3R, ICLR 2025 | Fine-tunes DUSt3R to predict structural plane pointmaps for unposed sparse-view room-layout reconstruction on Structured3D. |
| PLANA3R, NeurIPS 2025 | Feed-forward zero-shot metric planar primitives and relative pose from unposed two-view images; also demonstrates eight-view reconstruction. |
| CCGS, 2025 | Uses pointmap association and piecewise-plane constraints for consistent 3D Gaussian segmentation and editing. |
| SceneScript, ECCV 2024 | Directly generates editable structured architectural commands such as walls, doors and windows. |
| Scenes as Objects, 2026 | Produces instance-structured editable 3D token groups directly from unposed multi-view images. |
| Joint Layout and Registration, ICCV 2017 | Alternates layout estimation and layout-constrained global registration for indoor RGB-D fragments. |

Consequences:

* Uncertainty-aware plane fusion alone is too incremental.
* Bounded alpha support alone is occupied by AlphaTablets.
* Pointmap-based mask association alone is occupied by CCGS.
* Plane-aware unposed reconstruction alone is occupied by Plane-DUSt3R and PLANA3R.
* Joint plane/pose reasoning alone has both classical and learned precedents.

## 6. Ranked innovation candidates

Scores use 1 (weak/expensive) to 5 (strong/feasible).

| Candidate | Novelty | Feasibility | Evaluation clarity | Existing-code reuse | Verdict |
|---|---:|---:|---:|---:|---|
| A. N-view feed-forward trimmed plane-graph foundation model | 5 | 1 | 4 | 2 | Strongest academically, unrealistic with current data/compute. |
| B. PlaneGraph Bundle Adjustment adapter for frozen pointmap foundation models | 4 | 4 | 5 | 5 | Best overall direction. |
| C. Unposed AlphaTablets from DUSt3R/VGGT | 2 | 4 | 4 | 4 | Implementable but likely an incremental combination. |
| D. Uncertainty-aware hard-merge replacement | 2 | 5 | 4 | 5 | Useful ablation/component, insufficient as the main paper. |
| E. Plane relationship graph only | 2 | 3 | 3 | 3 | Crowded by layout, CAD and structured-modeling work. |

## 7. Recommended main innovation: PlaneGraph-BA

Working title:

```text
PlaneGraph-BA: A Training-Free Structural Adapter for 3D Foundation Pointmaps
```

### 7.1 Problem

Given unordered, unposed images, a frozen geometry foundation model produces
pairwise or per-view pointmaps, confidences and an initial global alignment.
These estimates are dense but are not guaranteed to preserve exact planes,
instance identity, boundaries or topology in low-texture indoor regions.

Rather than accepting the alignment as immutable and detecting planes only
afterwards, jointly refine:

```text
per-view/submap Sim(3) corrections
global plane identities
plane normal and offset
point-to-plane assignments
trimmed bounded support
plane adjacency/intersection relations
```

### 7.2 Key distinction

The method is not a new DUSt3R head and does not retrain a foundation model.
It is a plug-in test-time structural bundle-adjustment layer that consumes the
outputs of DUSt3R, MASt3R, VGGT or another pointmap model.

The plane graph is simultaneously:

1. the compact structured output; and
2. a set of structural landmarks that feeds back to correct pointmap alignment.

This closed loop distinguishes the proposal from pipelines that first freeze
geometry and then fit planes, and from plane-specific feed-forward models whose
plane representation is the primary predictor.

### 7.3 Variables

For each view or local submap `v`, optimize a small correction:

```text
T_v in Sim(3)
```

For each plane instance `k`, optimize:

```text
pi_k = (n_k, d_k)
```

For each local observation `i`, maintain a soft assignment:

```text
z_ik = P(observation i belongs to plane k)
```

For each plane, maintain a plane-local alpha/support field and graph edges:

```text
A_k(u,v)
E_kl in {adjacent, intersecting, parallel, independent}
```

### 7.4 Objective

Use an alternating discrete-continuous objective:

```text
E = E_foundation
  + lambda_plane E_point_to_plane
  + lambda_corr E_cross_view_correspondence
  + lambda_boundary E_support_reprojection
  + lambda_graph E_intersection_and_adjacency
  + lambda_complexity E_minimum_description_length
```

`E_foundation` keeps corrected pointmaps close to the frozen model prediction,
weighted by its confidence. `E_plane` flattens only points with sufficiently
strong planar evidence. `E_corr` preserves cross-view matches. `E_boundary`
requires the trimmed support to reproject consistently. `E_graph` regularizes
only high-confidence relationships; it must not impose a global Manhattan-world
assumption. `E_complexity` prevents thousands of redundant micro-primitives.

### 7.5 Specific testable claims

The paper should claim only improvements that are directly measured:

1. PlaneGraph-BA improves pose/global alignment in low-texture planar scenes.
2. It improves plane parameter accuracy and reduces duplicate/fragmented plane
   instances relative to post-hoc RANSAC and hard fusion.
3. It produces fewer primitives at matched reconstruction accuracy.
4. The same adapter improves at least two frozen pointmap backbones without
   backbone-specific training.
5. Optimized trimmed supports enable local plane edits with less collateral
   geometry movement than point-cloud or micro-primitive baselines.

### 7.6 Why it is feasible here

The current repository already provides most initialization and evaluation
infrastructure:

```text
Stage1 local support proposals
DUSt3R global pointmaps and confidence
explicit view registry and pixel provenance
global-cloud cache
plane SVD/refit
editable primitive export
RANSAC and merge ablation scaffold
```

The first prototype does not require new large-scale training. It can optimize
small per-view Sim(3) corrections and plane variables in PyTorch or a nonlinear
least-squares solver on the existing selected validation groups.

## 8. Stronger but currently impractical direction

The strongest clean-slate research direction would directly predict a variable
number of trimmed plane faces and their adjacency graph from arbitrary N unposed
views, trained by differentiable rendering and topology supervision. It would go
beyond PLANA3R's pair-centric micro-primitives and beyond SceneScript's layout
commands by representing general structural and object-supporting planes.

However, it requires:

* scene-level plane identity and adjacency annotations;
* a differentiable polygon/B-Rep renderer;
* large multi-view training corpora;
* substantially more GPU compute;
* careful handling of non-planar surfaces.

It is not the recommended immediate path.

## 9. Candidate research hypothesis retained as a component

Working name:

```text
EvidencePlane: uncertainty-aware bounded plane recovery from unposed images
```

Hypothesis:

> Treating every local plane observation as uncertain evidence over plane
> parameters and plane-local support, rather than as a hard mask or a hard
> merge decision, improves global plane identity and bounded support recovery
> under imperfect foundation-model alignment.

For observation `o_i` from view `v`, retain:

```text
o_i = (
  support pixels,
  aligned XYZ distribution,
  confidence,
  local plane parameter distribution,
  visible/free/unknown evidence in a plane-local chart,
  source-view provenance
)
```

For a candidate global plane, maintain:

```text
Pi_k = (
  posterior normal/offset,
  bounded support evidence field,
  observation membership probabilities,
  parameter uncertainty,
  identity uncertainty
)
```

The intended distinction from existing hard fusion is:

1. Plane association is soft and uncertainty-calibrated.
2. Geometry confidence and cross-view disagreement affect parameter fusion.
3. Plane-local support distinguishes positive, free-space, occluded and
   unobserved evidence instead of taking a mask union or convex hull.
4. Merges remain reversible until scene-level evidence supports one identity.

## 10. Minimal mechanism worth testing

Do not start with another large neural network. Implement a probabilistic or
energy-based prototype on the existing global-alignment cache.

### 6.1 Observation uncertainty

For each local support, bootstrap or weighted-fit a plane and estimate:

```text
mean plane parameter (normal, offset)
covariance / angular and offset uncertainty
support centroid and spatial covariance
DUSt3R confidence distribution
cross-view reprojection disagreement
```

### 6.2 Soft association graph

Create candidate edges only when observations pass broad geometric and
visibility gates. Score an edge using:

```text
uncertainty-normalized plane distance
spatial connectedness
cross-view support reprojection agreement
free-space contradiction
appearance/Stage1 identity evidence
cycle consistency across three or more observations
```

Avoid transitive union-find as the final inference method. Compare correlation
clustering, constrained agglomeration, or discrete-continuous optimization.

### 6.3 Plane-local evidence field

After estimating a plane basis, project each observation into `(u,v)`. Accumulate
four evidence states:

```text
observed planar support
observed free/non-plane region
occluded region
unobserved region
```

Recover one or more connected bounded components. Do not fill unknown space by
default. Export the evidence and uncertainty with the primitive.

## 11. Falsification plan

The idea is rejected unless it beats simpler alternatives on identical global
geometry.

Required methods:

1. sequential 3D RANSAC + Euclidean components;
2. Stage1 support + direct global SVD;
3. Stage1 support + manual geometry merge;
4. AirPlanes-style geometry plus learned/consistent embedding baseline where
   practical;
5. hard global merge/refit (current method);
6. proposed uncertainty/evidence fusion.

Required metrics:

```text
plane precision / recall
normal angular error
point-to-plane residual
support coverage and IoU
plane count error
fragmentation and over-merge
boundary Chamfer / F-score in plane-local coordinates
calibration of predicted uncertainty
runtime and memory
```

Stress tests must explicitly perturb:

```text
number of views
view ordering
alignment noise
point confidence corruption
occlusion fraction
small-plane area
coplanar disconnected instances
```

The proposed contribution is meaningful only if gains concentrate on the
failure modes it claims to solve, especially imperfect alignment and partial
visibility.

## 12. Immediate next steps

1. Finish the shared-global-cloud RANSAC study before extending the model.
2. Add point-aligned Structured3D GT for the selected validation scenes.
3. Build an observation-level diagnostic dataset from Stage1 supports and the
   global pointmaps.
4. Measure empirical uncertainty: bootstrap each support and compare predicted
   confidence with GT plane error.
5. Quantify hard-merge failures: false edge, transitive-chain merge, duplicate
   identity and missing-boundary rates.
6. Implement a non-learned uncertainty-normalized association baseline.
7. Implement PlaneGraph-BA v0 with fixed plane assignments and only per-view
   Sim(3) plus global plane parameter refinement.
8. Compare `initial alignment -> post-hoc plane fit -> PlaneGraph-BA` on both
   pose/global-cloud and plane metrics.
9. Only add learned association or alpha support if the structural feedback
   produces a measurable signal.

## 13. Primary sources inspected

* PlaneRCNN: https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_PlaneRCNN_3D_Plane_Detection_and_Reconstruction_From_a_Single_Image_CVPR_2019_paper.html
* PlaneAE: https://arxiv.org/abs/1902.09777
* PlaneTR: https://arxiv.org/abs/2107.13108
* SparsePlanes: https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Planar_Surface_Reconstruction_From_Sparse_Views_ICCV_2021_paper.html
* PlaneMVS: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_PlaneMVS_3D_Plane_Reconstruction_From_Multi-View_Stereo_CVPR_2022_paper.html
* PlanarRecon: https://openaccess.thecvf.com/content/CVPR2022/html/Xie_PlanarRecon_Real-Time_3D_Plane_Detection_and_Reconstruction_From_Posed_Monocular_CVPR_2022_paper.html
* Good Configurations of Planar Primitives: https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Finding_Good_Configurations_of_Planar_Primitives_in_Unorganized_Point_Clouds_CVPR_2022_paper.html
* AirPlanes: https://openaccess.thecvf.com/content/CVPR2024/papers/Watson_AirPlanes_Accurate_Plane_Estimation_via_3D-Consistent_Embeddings_CVPR_2024_paper.pdf
* PlanarSplatting: https://openaccess.thecvf.com/content/CVPR2025/html/Tan_PlanarSplatting_Accurate_Planar_Surface_Reconstruction_in_3_Minutes_CVPR_2025_paper.html
* PlaneRAS: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_PlaneRAS_Learning_Planar_Primitives_for_3D_Plane_Recovery_ICCV_2025_paper.html
* GSPlane: https://arxiv.org/abs/2510.17095
* Peek-a-Boo: https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_Peek-a-Boo_Occlusion_Reasoning_in_Indoor_Scenes_With_Plane_Representations_CVPR_2020_paper.html
* PolyFit: https://openaccess.thecvf.com/content_iccv_2017/html/Nan_PolyFit_Polygonal_Surface_ICCV_2017_paper.html
* Structure-Aware Primitive Assembly: https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Structure-Aware_Surface_Reconstruction_via_Primitive_Assembly_ICCV_2023_paper.html
* PlaneFormers: https://arxiv.org/abs/2208.04307
* NOPE-SAC: https://arxiv.org/abs/2211.16799
* PlaneRecTR++: https://arxiv.org/abs/2307.13756
* PlanarNeRF: https://arxiv.org/abs/2401.00871
* AlphaTablets: https://arxiv.org/abs/2411.19950
* NeuralPlane: https://neuralplane.github.io/index.html
* Plane-DUSt3R: https://arxiv.org/abs/2502.16779
* PLANA3R: https://arxiv.org/abs/2510.18714
* CCGS: https://arxiv.org/abs/2502.16303
* SceneScript: https://arxiv.org/abs/2403.13064
* Joint Layout Estimation and Global Registration: https://openaccess.thecvf.com/content_iccv_2017/html/Lee_Joint_Layout_Estimation_ICCV_2017_paper.html
* Room Envelopes: https://arxiv.org/abs/2511.03970

## 14. Current novelty verdict

No current LightRecon3D stage is independently novel under this literature
audit. Uncertainty-aware fusion is useful but insufficient as the main idea.

The best balance of novelty and feasibility is PlaneGraph-BA: a training-free,
cross-backbone structural adapter that uses an instance-level trimmed plane
graph to jointly refine frozen foundation-model pointmaps/alignment and output
editable structured geometry.

This remains a research hypothesis, not a novelty claim. The exact combination
has not been found in the papers inspected, but classical plane bundle adjustment,
joint layout-registration and recent plane foundation models are close enough
that novelty depends on the specific formulation and cross-backbone experimental
evidence.
