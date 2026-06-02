# Editable Full-Pointcloud Plane Demo

This demo converts a DUSt3R point cloud into explicit editable plane primitives.

The target behavior is:

1. Generate a full 3D point cloud from DUSt3R.
2. Extract major planes from the full point cloud.
3. Export each plane as an equation:

   ```text
   n_x*x + n_y*y + n_z*z + d = 0
   ```

4. Bind display points to their nearest extracted plane.
5. Edit each plane independently by changing `d`.
6. Move both the transparent plane mesh and the assigned point-cloud points along the plane normal.

## Generate Editable Plane Data

Example:

```bash
CUDA_VISIBLE_DEVICES=2 python export_full_pointcloud_editable_planes.py \
  --root_dir /data/zhucy23u/datasets/Structured3D \
  --weights_path /data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --split val \
  --sample_idx 26 \
  --max_planes 10 \
  --threshold 0.035 \
  --min_inliers 900 \
  --iterations 700 \
  --max_fit_points 65000 \
  --max_display_points 22000 \
  --output_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_points_move
```

Main outputs:

```text
val_000026_full_pointcloud_editable_planes.html
val_000026_full_pointcloud_editable_planes.ply
val_000026_full_pointcloud_plane_params.json
val_000026_full_pointcloud_plane_params.txt
```

The HTML includes one slider per extracted plane. Moving a slider changes the plane offset and moves assigned points along the plane normal.

## Generate Presentation HTML

The presentation page adds a right-side panel with plane equations, inlier counts, displayed point counts, and a focus mode for inspecting one plane at a time.

```bash
python make_editable_planes_presentation.py \
  --input_html /data/zhucy23u/logs/full_pointcloud_editable_planes_points_move/val_000026_full_pointcloud_editable_planes.html \
  --output_html /data/zhucy23u/logs/full_pointcloud_editable_planes_points_move/val_000026_full_pointcloud_editable_planes_presentation.html
```

Recommended local file to inspect:

```text
outputs/val_000026_full_pointcloud_editable_planes_presentation_v2.html
```

## Current Sample 26 Evidence

For `val` sample `26`:

```text
full input point cloud: 262144 points
display points: 22000
extracted major planes: 5
assigned display points: 21975
unassigned display points: 25
```

Extracted plane equations:

```text
plane 0: normal=(-0.015690, 0.998085, 0.059836), d=-0.191830
plane 1: normal=(0.012551, 0.999695, -0.021287), d=0.323495
plane 2: normal=(-0.147149, -0.002859, 0.989110), d=-0.635111
plane 3: normal=(0.973163, 0.001678, 0.230109), d=0.240543
plane 4: normal=(0.996804, -0.068612, 0.040905), d=-0.482586
```

## Research Position

This is currently an automatic point-cloud geometry stage, not yet a fully learned end-to-end primitive predictor.

It is still aligned with the final direction because it demonstrates the desired output surface:

```text
full point cloud -> major plane equations -> editable structure
```

The next research step is to replace or fuse the RANSAC-like extraction with the learned plane parameter head and learned refinement gate.
