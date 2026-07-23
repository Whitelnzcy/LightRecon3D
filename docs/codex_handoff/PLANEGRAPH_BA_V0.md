# PlaneGraph-BA v0

## Scope

PlaneGraph-BA v0 is an additive structural-feedback layer. It does not replace
Stage1, Stage2, DUSt3R global alignment, or the existing editable-plane output.

```text
frozen DUSt3R global cloud cache
        +
Stage3 plane supports with global pointmap provenance
        -> alternating plane refit / per-view Sim(3) correction
        -> corrected global point cloud + editable plane primitives
```

The current version changes one Sim(3) transform per aligned view. It does not
yet deform individual pointmaps and does not fine-tune DUSt3R weights. This is
intentional: the first experiment tests whether learned cross-view plane
identity is useful as structural feedback without adding another trained model.

## Required inputs

The method-independent global cloud cache is written by
`export_stage3_scene_plane_fusion.py` in `dust3r_global` mode. It contains:

```text
points, colors, confidence, view_indices, pixel_xy
```

The Stage3 support NPZ now additionally contains:

```text
alignment_view_indices
pointmap_pixel_xy
point_plane_ids
```

The join key is exactly `(alignment_view_index, pointmap_x, pointmap_y)`. No
nearest-neighbour or floating-point XYZ matching is used.

## Optimization

The reference view is fixed to remove the global gauge freedom. For every other
view, v0 optimizes a bounded seven-parameter correction:

```text
rotation vector (3), translation (3), log scale (1)
```

Each iteration:

1. transforms mapped plane observations with the current per-view corrections;
2. refits every global plane by confidence-weighted SVD;
3. updates each non-reference view with robust point-to-plane residuals;
4. applies quadratic rotation/translation/scale priors around the frozen
   DUSt3R alignment;
5. rejects steps that do not reduce the fixed-plane robust objective.

Only planes observed by `--min_plane_views` or more views affect view
corrections. Single-view planes are still refitted and exported but cannot
provide cross-view structural feedback.

## Example

```bash
python planegraph_ba.py \
  --global_cloud_npz outputs/scene_dust3r_global_cloud_cache.npz \
  --support_npz outputs/scene_stage3_dust3r_global_fusion_full_pointcloud_editable_planes_data.npz \
  --output_dir outputs/planegraph_ba \
  --scene_key scene \
  --min_conf 1.0 \
  --min_plane_views 2
```

Outputs use the existing plane primitive core fields and include both before
and after PLY visualizations, per-view Sim(3) corrections, plane parameters,
and residual history.

## Scientific status

Passing synthetic tests only verifies implementation behavior. It does not show
that PlaneGraph-BA improves Structured3D or real-scene reconstruction. A valid
claim requires the same-scene comparison against original DUSt3R, post-hoc
SVD, sequential RANSAC, manual merge, and the current full method, including
non-planar reconstruction and pose metrics.
