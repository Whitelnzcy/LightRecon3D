# Bounded Plane Support Head

This experiment learns finite plane support masks and refits plane equations
analytically from the predicted support points.

## Install on another server

Start from the plane experiment branch, then cherry-pick the support-head
commit:

```bash
git switch param-editable-primitives
git pull --ff-only origin param-editable-primitives
git cherry-pick <support-head-commit>
```

The training data directories referenced by the shell scripts are:

```text
/data/zhucy23u/logs/learned_plane_params_v1/stage1_teacher_train128_v1_line_v3
/data/zhucy23u/logs/learned_plane_params_v1/stage1_teacher_val32_v1_line_v3
```

Adjust `INPUT_DIR`, `OUTPUT_DIR`, and `CKPT` in the scripts for the new server.

## Training sequence

The current hard-boundary experiment continues from the v2 checkpoint:

```bash
bash run_bounded_support_head_train128_v1.sh
bash run_bounded_support_head_train128_v2_boundaryweighted.sh
bash run_bounded_support_head_train128_v3_hardboundary.sh
bash run_bounded_support_head_eval_val32_v3_hardboundary.sh
bash run_bounded_support_head_eval_val32_v3_hardboundary_visuals.sh
```

The v3 loss adds a barrier on confident boundary edges:

```text
L_hard_boundary = -log(1 - dot(p_i, p_j))
```

For patches on opposite sides of a supervised boundary, predicting the same
plane produces a rapidly increasing penalty.

## Method

```text
DUSt3R/MASt3R point and decoder features
-> local planar patches
-> bounded plane support head
-> boundary-aware support assignment
-> SVD plane refit on each predicted support
-> editable finite plane primitive
```

The network learns support assignment. Plane normal and offset are computed
from the final support instead of being treated as the primary learned output.
