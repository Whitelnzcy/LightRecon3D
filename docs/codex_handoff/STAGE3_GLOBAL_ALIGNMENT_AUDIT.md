# Stage3 DUSt3R Global Alignment 数据流审计

日期：2026-07-05

范围：仅执行 Phase 1 只读审计。未启动训练，未调 plane merge 阈值，未修改源码；本次只新增/更新本文档。

## 继承项目状态

- 重新上传后已读取：`AGENTS.md`、`docs/codex_handoff/CURRENT_STATE.md`、`docs/codex_tasks/stage3_global_alignment.md`。
- 最新提交：`66066ad docs: add Codex handoff for DUSt3R global alignment`。
- handoff 明确要求：Stage3 必须实际走 DUSt3R `make_pairs -> inference -> global_aligner -> compute_global_alignment -> scene.get_pts3d()`，并把 Stage2 support pixels 映射到全局 aligned pointmaps 后再做跨视角/跨 pair 的 merge/refit。
- 当前工作区已有前一会话遗留的未提交改动，集中在 Stage2 metadata passthrough、Stage3 global alignment 草稿、训练损失实验和少量 DUSt3R demo 修补。本文档不回滚、不覆盖这些改动。

## Git 状态和 diff 摘要

`git status --short` 显示：删除 `BOUNDED_SUPPORT_HEAD.md`；修改 `dust3r/dust3r/demo.py`、`export_stage2_learned_region_merge_editables.py`、`export_stage3_scene_plane_fusion.py`、`run_stage3_scene_fusion_v1.sh`、`train_amortized_plane_tokens.py`、`train_patch_plane_tokens.py`、`train_stage1_clean_baseline.py`、`train_stage1_plane_masks.py`；新增未跟踪 `docs/codex_handoff/STAGE3_GLOBAL_ALIGNMENT_AUDIT.md`；另有大量未跟踪本地输出、报告、patch、缓存目录，包括 `%SystemDrive%/`、`_packages/`、`badcase_light_report/`、`bug/`、`local_outputs/` 等。

`git diff --stat`：9 tracked files changed, 480 insertions(+), 96 deletions(-)。

最近 10 个提交：

```text
66066ad docs: add Codex handoff for DUSt3R global alignment
075da61 Add Stage3 scene plane fusion
b1223d6 Inline Stage1 teacher mask scoring
25ecd17 Run Stage2 region merge on full validation
8a8b7ee Add learned Stage2 region merge
6bb7dea Add Stage1 quality split tooling
889aba9 Add 4096-sample Stage1 large training script
d7a75f3 Add strict duplicate plane merge for geometry refit
63f0070 Add Stage2 geometry refit export scripts
e371f31 Handle legacy matching in visualization
```

## 文件检查清单

- `AGENTS.md`
- `docs/codex_handoff/CURRENT_STATE.md`
- `docs/codex_tasks/stage3_global_alignment.md`
- `dataloaders/s3d_dataset.py`
- `export_stage1_pred_support_teacher_npz.py`
- `export_stage2_learned_region_merge_editables.py`
- `export_stage2_geometry_refit_editables.py`
- `export_stage3_scene_plane_fusion.py`
- `train_stage2_region_merge_net.py`
- `run_stage3_scene_fusion_v1.sh`
- `dust3r/dust3r/utils/image.py`
- `dust3r/dust3r/image_pairs.py`
- `dust3r/dust3r/inference.py`
- `dust3r/dust3r/cloud_opt/__init__.py`
- `dust3r/dust3r/cloud_opt/base_opt.py`
- `dust3r/dust3r/cloud_opt/optimizer.py`
- diff：`dust3r/dust3r/demo.py`、`train_stage1_clean_baseline.py`、`train_stage1_plane_masks.py`、`train_patch_plane_tokens.py`、`train_amortized_plane_tokens.py`
- 样例 NPZ：`local_outputs/stage1_pred_support_safe_geometry_val32_v1/val_000002_stage1_teacher_full_pointcloud_editable_planes_data.npz`
- 样例 NPZ：`local_outputs/stage2_safe_geometry_bounded_support_val32_v5_display_gate/val_000002_stage1_teacher_bounded_support_head_assignment.npz`

## 当前数据流图

现有代码实际路径：

```text
Structured3DDataset
-> pair samples: img1/img2 + scene_name/pair_group/rgb_path1/rgb_path2
-> Stage1 export: view1 pointmap + predicted support + pixel_xy
-> Stage2 learned merge: read point_plane_ids, refit/merge within one NPZ, passthrough metadata if present
-> Stage3 scene fusion:
   group NPZ files by sample/scene/pair_group/reference_view
   local mode: concatenate pair-local points, compare/refit pair-local planes
   inherited dust3r_global draft:
      collect rgb_path1, optionally rgb_path2
      load_images
      make_pairs
      inference
      global_aligner
      compute_global_alignment
      scene.get_pts3d / scene.get_conf
      map pixel_xy to registered view pointmap
      refit/merge in mapped coordinates
```

目标路径应改为：

```text
Stage2 support pixels
-> recover source RGB views
-> group all views from same scene_name + pair_group
-> build explicit view registry
-> DUSt3R load_images/make_pairs/inference/global_aligner
-> compute_global_alignment
-> map support pixels into scene.get_pts3d()[registered_view_index]
-> merge/refit bounded planes in global coordinates
```

## Stage2 NPZ writer 和当前 schema

当前与 learned Stage2 merge 直接相关的 writer 是 `export_stage2_learned_region_merge_editables.py`。

它当前写出：`points`、`colors`、`original_colors`、`point_plane_ids`、`plane_ids`、`plane_normals`、`plane_offsets`、`plane_inlier_counts`、`active_planes`、`source_plane_groups`、`merge_pairs_json`。

它会在输入存在时透传：`scene_name`、`pair_group`、`rgb_path1`、`rgb_path2`、`sample_idx`、`pixel_xy`、`gt_point_plane_ids`、`point_confidence`、`point_margin`、`line_prob`。

缺口：没有 `schema_version`，没有 `view_id1/view_id2`，没有 `pixel_xy1/pixel_xy2`，没有 `original_hw1/original_hw2`，没有 `stage1_mask_hw1/stage1_mask_hw2`，没有 `pixel_coordinate_space` 和 `pixel_coordinate_order`，也没有逐 support 点的 source view 标识。

## 当前 Stage1 export schema

`export_stage1_pred_support_teacher_npz.py` 当前代码写出：`points (N,3) float32`、`colors (N,3) uint8`、`original_colors (N,3) uint8`、`scene_name`、`pair_group`、`rgb_path1`、`rgb_path2`、`sample_idx`、`pixel_xy (N,2) float32`、`point_plane_ids (N,) int32`、`gt_point_plane_ids (N,) int32`、`plane_normals (P,3) float32`、`plane_offsets (P,) float32`、`plane_inlier_counts (P,) int32`、`plane_ids (P,) int32`、`source_gt_plane_ids (P,) int32`、`line_prob (N,) float32`、`point_confidence (N,) float32`、`point_margin (N,) float32`、`query_ids (P,) int32`、`query_existence (Q,) float32`、`full_mask_counts`、`core_mask_counts`、`fit_mask_counts`。

注意：当前 Stage1 export 只导出 view1 的 support 和 `pixel_xy`，不是双视角 support schema。

## 本地样例 NPZ schema

检查到的旧 Stage1 样例字段：

```text
points (16000,3) float32
colors (16000,3) uint8
original_colors (16000,3) uint8
pixel_xy (16000,2) float32
point_plane_ids (16000,) int32
gt_point_plane_ids (16000,) int32
plane_normals (5,3) float32
plane_offsets (5,) float32
plane_inlier_counts (5,) int32
plane_ids (5,) int32
source_gt_plane_ids (5,) int32
line_prob (16000,) float32
point_confidence (16000,) float32
point_margin (16000,) float32
query_ids (5,) int32
query_existence (8,) float32
full_mask_counts (5,) int32
core_mask_counts (5,) int32
fit_mask_counts (5,) int32
```

该旧样例缺少 `scene_name/pair_group/rgb_path1/rgb_path2/sample_idx`。

检查到的旧 Stage2 bounded-support 样例字段：

```text
points (16000,3) float32
colors (16000,3) uint8
original_colors (16000,3) uint8
assignment (16000,) int32
patch_assignment (625,) int32
point_to_patch (16000,) int32
patch_labels (625,) int32
patch_centroids (625,3) float32
patch_normals (625,3) float32
patch_label_conf (625,) float32
patch_boundary_ce_weight (625,) float32
patch_edge_i (2352,) int32
patch_edge_j (2352,) int32
patch_edge_boundary_conf (2352,) float32
patch_boundary_neighbor (625,8) float32
patch_support_prior (625,8) float32
patch_fit_confidence (625,5) float32
patch_fit_weight (625,5) float32
plane_normals (5,3) float32
plane_offsets (5,) float32
plane_offsets_normalized (5,) float32
active_planes (5,) bool
gt_point_plane_ids (16000,) int32
```

该旧 Stage2 样例使用 `assignment`，不是 Stage3 当前 reader 需要的 `point_plane_ids`，也缺少全局对齐需要的路径、分组、坐标约定字段。

## Stage3 reader 和当前 merge/refit

`export_stage3_scene_plane_fusion.py` 当前 reader：glob 输入 NPZ；用 `group_key()` 按 `sample`、`scene`、`pair_group` 或 `reference_view` 分组；读取 `point_plane_ids`、`plane_normals`、`plane_offsets`；local mode 直接使用输入 `points`；inherited `dust3r_global` mode 使用 `pixel_xy` 和 `rgb_path1` 映射到 DUSt3R aligned pointmap；对每个局部 plane support 用 `fit_plane_np()` 重新拟合；用 angle、offset、mutual residual、centroid distance 做 union-find merge；对 merged support 重新拟合全局 plane。

风险：local mode 仍会把 pair-local points 当作可比较坐标；global mode 虽然已有草稿，但没有显式 schema version、view registry、坐标 transform 证明和双视角 support 结构。

## RGB path、view id、scene、pair_group 传播

- `Structured3DDataset._get_pair_group()` 返回 `.../2D_rendering/<space>/perspective/empty`，基本对应一个 Structured3D space/room 级 group。
- pair sample 包含 `scene_name`、`pair_group`、`rgb_path1`、`rgb_path2`，但不包含显式 `view_id1/view_id2` 输出字段。
- `view_id` 可以从 view 目录名解析，dataset 内部已有 `_get_view_id()`，但当前 batch 返回值没有把 `view_id1/view_id2` 传出来。
- 当前 Stage1 export 已写 `scene_name/pair_group/rgb_path1/rgb_path2/sample_idx/pixel_xy`。
- 当前 Stage2 learned merge 只是在存在时透传这些字段。
- 旧本地 NPZ 不具备这些字段，因此不能直接作为 Stage3 global alignment 的可靠输入。

## mask 分辨率和 pixel 坐标约定

- Stage1 里 `target_hw = point_map.shape[:2]`。
- `pixel_xy` 由 meshgrid 生成：`x = linspace(-1.0, 1.0, target_w)`，`y = linspace(-1.0, 1.0, target_h)`。
- 当前约定可推断为：order `(x, y)`；space `Stage1/DUSt3R point_map raster`；range `[-1, 1]`。
- 当前 Stage3 草稿使用 `col = round((x + 1) * 0.5 * (W - 1))` 和 `row = round((y + 1) * 0.5 * (H - 1))`。
- 这只在 Stage1 point_map raster 与 Stage3 DUSt3R `load_images(..., size=image_size)` 后的 pointmap raster 具有同一 resize/crop 语义时成立。handoff 明确要求不能只靠 shape assumption。

## DUSt3R API 和预处理签名

vendored DUSt3R 实际位置：`dust3r/dust3r/`。

实际函数签名：

- `load_images(folder_or_list, size, square_ok=False, verbose=True, patch_size=16)`
- `make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)`
- `inference(pairs, model, device, batch_size=8, verbose=True)`
- `global_aligner(dust3r_output, device, mode=GlobalAlignerMode.PointCloudOptimizer, **optim_kw)`
- `scene.compute_global_alignment(init=None, niter_PnP=10, **kw)`
- global alignment loop accepts `lr=0.01`, `niter=300`, `schedule='cosine'`, `lr_min=1e-6`
- `scene.get_pts3d(raw=False)`
- `scene.get_conf(mode=None)`

`load_images()` 关键预处理：输入 list 时按传入顺序遍历，并设置 `idx=len(imgs)`；非 224 size 先将长边 resize 到 `size`；然后以中心裁剪到 patch size 的整数倍；当 `square_ok=False` 且 resize 后 `W == H` 时，会把高度裁成 `3/4`，不是简单方形 resize；输出 dict 包含 `img`、`true_shape`、`idx`、`instance`。

这意味着 Stage3 必须记录并复用实际 resize/crop transform；不能直接把 Stage1 normalized coordinate 缩放到 `scene.get_pts3d()` 的 shape。

## view-index 风险

- `BasePCOptimizer` 从 `view1['idx']` 和 `view2['idx']` 建图，要求 indices 连续为 `0..n-1`。
- `scene.get_pts3d()[i]` 对应 DUSt3R loaded image 的 `idx=i`，不是 Structured3D camera id、不是文件名后缀、不是 Stage2 pair-local view 编号。
- 当前 Stage3 草稿默认 `pts3d[index] == image_paths[index]`。以当前 `load_images()` 实现看，list 输入时这通常成立，但仍应显式记录和断言。
- 当前 `collect_group_images()` 默认只收 `rgb_path1`，`rgb_path2` 需要 `--include_second_view`。这不满足 recover source RGB views / group all views 的默认架构要求。
- 当前 support 映射只看 `rgb_path1`。如果未来有 `pixel_xy2` 或双视角支持区域，必须逐 support 绑定 source view。
- path canonicalization 只做 normpath 和 slash 替换，没有处理绝对路径、大小写、repo-relative 路径和跨系统路径差异。

## 缺失字段

最低缺失字段：`schema_version`、`view_id1`、`view_id2`、`pixel_xy1`、`pixel_xy2`、`original_hw1`、`original_hw2`、`stage1_mask_hw1`、`stage1_mask_hw2`、`dust3r_input_hw1`、`dust3r_input_hw2`、`pixel_coordinate_space`、`pixel_coordinate_order`、per-support `source_view_key` 或等价字段、per-support `source_pixel_xy`、pair-local plane params 和 residual 的 provenance 字段。

兼容性缺口：旧 Stage2 NPZ 使用 `assignment`，当前 Stage3 reader 需要 `point_plane_ids`；旧输出缺少 RGB path 和 pair_group，无法自动恢复全局对齐 view registry。

## 坐标系统风险

- `pixel_xy` 当前是 `[-1,1]` normalized `(x,y)`，但 handoff 推荐 `[0,1]` 或 original RGB integer pixels。现有字段名没有写明约定。
- DUSt3R `load_images()` 有 resize + center crop，且 square image 在 `square_ok=False` 时会裁成 4:3；当前 Stage3 草稿没有保存 crop offset 和 scale。
- Stage1 pointmap 的 resolution 与 Stage3 global alignment 重新 load 的 image resolution 可能一致，也可能因参数或预处理差异不一致。
- 当前 mapping 使用 `round`，需要明确 pixel center 约定；否则边界点和下采样点会有 off-by-one 风险。
- confidence filtering 会删除点；所有 per-point arrays 必须同步过滤，否则 assignment 和 points 会错位。

## 坐标映射风险

- 支持点来自 Stage2 `point_plane_ids`，但对应像素应来自 Stage1 `pixel_xy`。如果 Stage2 做过采样、过滤、重排，必须保证 `pixel_xy` 与 `point_plane_ids` 同长度同顺序。
- 当前 Stage2 learned merge 透传 `pixel_xy`，没有显式验证长度与 `point_plane_ids` 一致。
- 当前 global mapping 没有输出 unmapped/invalid pixel counts、mapped row/col、source view index、confidence，这会让错误很难定位。
- 当前 global path 会在 mapped global points 上 refit plane，这是正确方向，但输出没有保留每个 source local plane 的 global refit diagnostics。

## proposed file changes

Phase 2 以后建议改动文件：

- `dataloaders/s3d_dataset.py`：batch 中显式返回 `view_id1/view_id2`、`original_hw1/original_hw2` 或可推导路径。
- `export_stage1_pred_support_teacher_npz.py`：写 `schema_version`、`pixel_xy1`、坐标约定、mask/input/original hw、view id；如需要双视角 support，新增 `pixel_xy2/point_plane_ids2/plane_ids2` 等。
- `export_stage2_learned_region_merge_editables.py`：schema versioning；严格验证并透传 required metadata；保留 pair-local plane params 和 residual/provenance。
- `export_stage3_scene_plane_fusion.py`：新增 Stage2 record loader、group/view registry、DUSt3R preprocessing transform 记录和验证、support-to-global mapper、global diagnostics 输出。
- `run_stage3_scene_fusion_v1.sh`：明确 global mode 参数，但只在 Phase 4 后启用。
- `tests/`：新增 metadata round-trip、view registry、pixel mapping、resize/crop、failure handling 测试。

## 聚焦测试计划

1. Metadata round-trip：写出并重新加载 Stage2 NPZ，检查 schema version、路径、view ids、hw、coordinate convention 和 support arrays 不变。
2. View registry：同一 RGB 出现在多个 Stage2 pair 时，只产生一个 global alignment view index。
3. DUSt3R preprocessing transform：用非方形图和方形图分别验证 resize/crop 后 `true_shape`、scale、crop offset。
4. Pixel mapping：构造已知 normalized/original pixel，验证映射到 `scene.get_pts3d()[registered_index][row,col]` 的 row/col 符合 transform。
5. Legacy compatibility：旧 `assignment` NPZ 应给出清晰错误或兼容转换，不允许静默错误。
6. Failure handling：缺路径、少于两张图、缺 `pixel_xy`、缺 `pair_group`、global alignment 失败时输出明确 diagnostics。
7. Visual smoke：Phase 4/5 后在一个小 group 上输出 view registry JSON、mapped support stats、global colored point cloud。

## 审计结论

当前 inherited Stage3 草稿已经接近目标 API 流程，但还不满足 handoff 的架构验收标准。主要缺口不是阈值，而是 schema、view registry、DUSt3R resize/crop transform 和 per-support source view provenance。

下一步应先做 Phase 2 metadata/schema update，并保持旧 Stage2 文件的显式兼容策略；在这之前不应运行大规模 Stage3 global fusion，也不应调 merge 阈值。
