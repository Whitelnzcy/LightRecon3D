# Plane-Regularized DUSt3R Alignment v1

## Purpose

This is the first implementation that sends the recovered plane structure back
into DUSt3R's differentiable global alignment. It is not a post-hoc SVD refit
and it is not the earlier per-view Sim(3) cache correction.

The frozen DUSt3R network first produces pairwise pointmaps. Standard global
alignment initializes the scene. Stage1/Stage2 support pixels and plane
identities are then mapped by exact `(alignment_view_index, x, y)` provenance
into the live `PointCloudOptimizer`. A second optimization minimizes:

```text
L_total = L_DUSt3R + lambda_plane * L_multiview_plane_incidence
```

Gradients from the plane term reach DUSt3R global-alignment variables including
per-view depthmaps, poses, focals, and pairwise alignment parameters. DUSt3R
network weights remain frozen.

## Safety and scientific controls

* Only planes observed from at least two distinct aligned views provide feedback.
* Exact repeated `(plane, view, x, y)` observations are deduplicated.
* Confidence weighting is normalized and every active plane receives balanced
  aggregate weight.
* Plane incidence uses a robust Huber-like penalty in scene-normalized units.
* The original DUSt3R objective remains in every optimization step.
* All live scene parameters are snapshotted before feedback.
* The proposed update is rolled back unless plane residual improves and the
  original DUSt3R loss stays below a configured relative-increase limit.
* The unmodified global-cloud cache is written before feedback. Accepted
  feedback gets a separate cache and before/after diagnostic PLY files.
* Full-cloud and non-support displacement are reported so improvement cannot be
  claimed from plane residual alone.

## Current limitation

The current Stage3 smoke path obtains cross-record plane identity from the
existing manual global merge. This tests the differentiable feedback mechanism,
but it is not the final learned association design. A failed or rejected update
must be reported as such. Real geometry and plane metrics have not yet been run.

## Single-group server smoke command

```bash
python export_stage3_scene_plane_fusion.py \
  --input_dir <one_showcase_group>/stage2_merge \
  --output_dir <new_output_dir> \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --group_by pair_group \
  --fusion_mode dust3r_global \
  --merge_mode manual \
  --weights_path /gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --include_second_view \
  --plane_feedback
```

The first acceptance gate is:

```text
plane_feedback.accepted == true
plane residual mean/p95 decreases
DUSt3R base loss increase <= configured limit
non-support displacement remains bounded
```

This gate validates behavior only. The research hypothesis still requires
Structured3D geometry, pose, and plane metrics against the original alignment.
