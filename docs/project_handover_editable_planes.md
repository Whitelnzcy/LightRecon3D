# 项目接手说明：完整点云主平面参数化与可编辑重建

这份文档用于接手当前 `param-editable-primitives` 分支。当前阶段的核心目标是：

```text
DUSt3R/MASt3R 生成完整 3D 点云
-> 从完整点云中提取主要结构平面
-> 每个主要平面给出参数方程 nx*x + ny*y + nz*z + d = 0
-> 将完整点云中的点绑定到对应平面
-> 修改平面 offset d，驱动完整点云中对应点一起移动
```

现阶段先用 RANSAC-like 后处理证明可行性；后续再把 plane head / line head 学习式预测接入，形成更像端到端的结构 primitive 参数化。

## 1. 分支和服务器

GitHub 分支：

```text
param-editable-primitives
```

服务器项目目录：

```text
/home/zhucy23u/projects/LightRecon3D
```

服务器数据集：

```text
/data/zhucy23u/datasets/Structured3D
```

DUSt3R 权重：

```text
/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

Python 环境：

```text
/data/zhucy23u/conda_envs/lightrecon/bin/python
```

当前主要输出目录：

```text
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz
```

批量汇总和 before/after 可视化目录：

```text
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary
```

## 2. 当前最重要的新增代码

### `export_full_pointcloud_editable_planes.py`

用途：

- 调 DUSt3R 对一个 Structured3D 样本生成完整点云。
- 对完整点云做主要平面提取。
- 为每个主要平面输出方程：

  ```text
  nx*x + ny*y + nz*z + d = 0
  ```

- 给完整点云中的每个点分配最近的平面 ID。
- 输出 `.npz`，保存完整点云、颜色、点到平面分配、平面 normal/offset。

主要输出：

```text
val_000026_full_pointcloud_editable_planes_data.npz
val_000026_full_pointcloud_plane_params.txt
val_000026_full_pointcloud_plane_params.json
val_000026_full_pointcloud_editable_planes.html
val_000026_full_pointcloud_editable_planes.ply
```

最关键的是 `.npz`，因为它保存的是完整点云，不只是浏览器采样点。

### `apply_editable_plane_offsets.py`

用途：

- 从完整 `.npz` 读入点云和平面参数。
- 指定 `--edit PLANE_ID:DELTA`。
- 将属于该平面的完整点云点沿平面 normal 移动。
- 输出编辑后的完整 PLY 和 JSON 报告。

例子：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python apply_editable_plane_offsets.py \
  --input_npz /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/val_000026_full_pointcloud_editable_planes_data.npz \
  --output_ply /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/val_000026_edit_plane2_d025_full_points.ply \
  --output_json /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/val_000026_edit_plane2_d025_report.json \
  --edit 2:0.25
```

### `summarize_full_pointcloud_plane_edits.py`

用途：

- 扫描一个目录下所有 `*_full_pointcloud_editable_planes_data.npz`。
- 汇总每个样本的完整点数、平面数量、每个平面方程、每个平面绑定了多少完整点云点。
- 可选：自动编辑最大主平面并导出完整 PLY。

例子：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python summarize_full_pointcloud_plane_edits.py \
  --input_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz \
  --output_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary \
  --export_edits \
  --edit_plane largest \
  --edit_delta 0.25
```

主要看：

```text
full_pointcloud_plane_edit_summary.md
full_pointcloud_plane_edit_summary.csv
full_pointcloud_plane_edit_summary.json
```

### `make_full_pointcloud_edit_comparison.py`

用途：

- 从完整 `.npz` 生成 before/after 对照 HTML。
- 左边显示原始点云，右边显示编辑后的点云。
- 右侧列出完整点数、移动点数、所有主平面方程。
- 浏览器中显示的是采样点，但所有统计和编辑都是基于完整点云计算的。

例子：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python make_full_pointcloud_edit_comparison.py \
  --input_npz /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/val_000026_full_pointcloud_editable_planes_data.npz \
  --output_html /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000026_full_pointcloud_edit_comparison.html \
  --edit_plane largest \
  --edit_delta 0.25 \
  --max_display_points 28000
```

### `make_editable_planes_presentation.py`

用途：

- 把早期交互 HTML 转成更适合汇报的页面。
- 这个不是当前最关键脚本。当前更推荐看 `make_full_pointcloud_edit_comparison.py` 输出的 before/after HTML。

### `train_pseudo_plane_head_from_npz.py`

用途：

- 这是从后处理走向可学习 primitive head 的第一版最小实验。
- 输入当前 RANSAC-like 后处理生成的完整点云 `.npz`。
- 每个点的输入是：

  ```text
  normalized xyz + rgb
  ```

- 伪标签来自 `.npz` 中的：

  ```text
  point_plane_ids
  plane_normals
  plane_offsets
  ```

- 训练一个小 MLP head，让它预测：

  ```text
  per-point plane normal
  per-point plane offset
  valid/confidence
  ```

注意：这还不是最终的 DUSt3R/MASt3R feature head，只是为了确认“RANSAC 后处理结果能否作为 pseudo label 训练一个 plane primitive head”。

## 3. 当前已经跑出的结果

服务器汇总文件：

```text
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/full_pointcloud_plane_edit_summary.md
```

当前已跑样本：

```text
val_000026: 262144 点，5 个主平面，261837 点绑定到平面，编辑最大平面移动 91892 点
val_000027: 262144 点，6 个主平面，259947 点绑定到平面，编辑最大平面移动 77512 点
val_000028: 262144 点，4 个主平面，262046 点绑定到平面，编辑最大平面移动 116714 点
```

before/after HTML：

```text
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000026_full_pointcloud_edit_comparison.html
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000027_full_pointcloud_edit_comparison.html
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000028_full_pointcloud_edit_comparison.html
```

第一版 pseudo plane head 训练结果：

```text
/data/zhucy23u/checkpoints/lightrecon_param/pseudo_plane_head_npz_v1/best_pseudo_plane_head.pt
/data/zhucy23u/checkpoints/lightrecon_param/pseudo_plane_head_npz_v1/pseudo_plane_head_training_summary.json
```

训练设置：

```text
输入样本: val_000026 到 val_000040 的 15 个完整点云 npz
训练/验证: 12 / 3
输入: 每点 normalized xyz + rgb
监督: RANSAC-like plane normal / offset / assignment 伪标签
```

当前 best epoch 是 35，验证指标大概是：

```text
val loss: 0.3539
normal angle error: 44.95 deg
offset MAE: 0.1686
plane distance loss: 0.0861
```

这个结果说明 head 已经能开始拟合伪标签，但还不够好，尤其 normal 误差仍然偏大。它只能算学习头起步实验，不能作为最终方法。下一步应该把输入从简单 xyz/rgb 改成 DUSt3R/MASt3R decoder feature 或点云局部几何特征，并引入 plane-level pooling / assignment head。

编辑后的完整 PLY：

```text
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000026_edit_plane0_d+0.250_full_points.ply
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000027_edit_plane1_d+0.250_full_points.ply
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000028_edit_plane0_d+0.250_full_points.ply
```

本地已经拉回的可看文件：

```text
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\batch_full_plane_edit_summary\full_pointcloud_plane_edit_summary.md
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\batch_full_plane_edit_summary\val_000026_full_pointcloud_edit_comparison.html
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\batch_full_plane_edit_summary\val_000027_full_pointcloud_edit_comparison.html
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\batch_full_plane_edit_summary\val_000028_full_pointcloud_edit_comparison.html
```

## 4. 如何跑一个新样本

假设要跑 `val` 的第 29 个样本：

```bash
cd /home/zhucy23u/projects/LightRecon3D

CUDA_VISIBLE_DEVICES=2 /data/zhucy23u/conda_envs/lightrecon/bin/python export_full_pointcloud_editable_planes.py \
  --root_dir /data/zhucy23u/datasets/Structured3D \
  --weights_path /data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --split val \
  --sample_idx 29 \
  --max_planes 10 \
  --threshold 0.035 \
  --min_inliers 900 \
  --iterations 700 \
  --max_fit_points 65000 \
  --max_display_points 22000 \
  --output_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz
```

然后重新批量汇总：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python summarize_full_pointcloud_plane_edits.py \
  --input_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz \
  --output_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary \
  --export_edits \
  --edit_plane largest \
  --edit_delta 0.25
```

再生成 before/after HTML：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python make_full_pointcloud_edit_comparison.py \
  --input_npz /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/val_000029_full_pointcloud_editable_planes_data.npz \
  --output_html /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/batch_summary/val_000029_full_pointcloud_edit_comparison.html \
  --edit_plane largest \
  --edit_delta 0.25 \
  --max_display_points 28000
```

## 5. 你最可能要改的参数

### `--sample_idx`

换数据样本。

```text
26, 27, 28, 29, ...
```

### `--max_planes`

最多提取几个主平面。室内场景一般先用：

```text
10
```

### `--threshold`

点到平面的距离阈值。越小越严格，平面更准但绑定点更少；越大绑定更多但可能混入噪声。

当前：

```text
0.035
```

可以试：

```text
0.02, 0.03, 0.05
```

### `--min_inliers`

一个平面最少需要多少点。越大越只保留大平面。

当前：

```text
900
```

### `--edit_plane`

编辑哪个平面。

```text
largest
```

表示自动选绑定点最多的主平面。也可以写具体 ID：

```text
0
1
2
```

### `--edit_delta`

平面 offset 改变量。当前用：

```text
0.25
```

小一点可以试：

```text
0.05, 0.10, 0.15
```

## 6. 怎么看结果是不是对的

先看：

```text
full_pointcloud_plane_edit_summary.md
```

重点检查：

- `full points` 是否是完整点云数量，例如 `262144`。
- `major planes` 是否合理。
- `assigned points` 是否接近完整点数。
- 表格里的每个平面是否有 equation。
- `exported edit` 是否移动了大量完整点云点。

再打开：

```text
val_0000XX_full_pointcloud_edit_comparison.html
```

重点看：

- 左右两边是不是同一个点云的 before/after。
- 红色点是不是被编辑的主平面点。
- 右侧 `Plane Equations` 是否列出所有主平面方程。
- `Moved points` 是否和 summary 一致。

如果要用 CloudCompare / MeshLab 看完整点云，打开：

```text
*_full_points.ply
```

还必须同时看原始 RGB。只看 3D 点云颜色分块不够，因为 learned token 可能只是把点云切成几块几何区域，并不一定对应真实墙面/地面/桌面。

当前已经导出 `val_000026` 到 `val_000040` 的 RGB：

```text
/data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/rgb_context
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\rgb_context
```

例如：

```text
val_000037_rgb_resized.png
val_000037_rgb_context.json
```

`*_rgb_context.json` 里记录了对应 Structured3D 的原始 `rgb_rawlight.png` 和 `layout.json` 路径。

可以用脚本重新导出 RGB：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python export_sample_rgb_context.py \
  --root_dir /data/zhucy23u/datasets/Structured3D \
  --split val \
  --sample_indices 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 \
  --output_dir /data/zhucy23u/logs/full_pointcloud_editable_planes_full_npz/rgb_context
```

## 7. 当前还没完成的研究部分

当前这条链路主要是：

```text
DUSt3R 点云 + RANSAC-like 后处理 -> 平面方程 -> 可编辑点云
```

它已经能证明“完整点云主平面参数化和编辑”这件事可行，但还不是最终创新版本。

后续应该继续做：

1. learned plane head：让网络从 DUSt3R/MASt3R 特征中预测平面参数或平面概率。
2. line head：预测主要结构线的参数方程。
3. iterative refinement：用当前 RANSAC-like 结果作为初始化，再用点云误差、平面一致性、线面约束迭代优化。
4. edit consistency：编辑 plane/line 参数后，点云、结构边界、可视化结果保持一致。

早期相关实验文件包括：

```text
train_plane_params.py
evaluate_plane_params_head.py
evaluate_plane_param_refinement.py
visualize_plane_params_head.py
evaluate_plane_clustering.py
oracle_plane_refine.py
```

这些是 plane head/聚类/优化方向的实验脚本，现阶段可以作为后续学习式方案的参考，但现在接手时优先理解 full pointcloud editable planes 这条主线。

### `train_unsupervised_plane_tokens.py`

用途：

- 这是纠正方向后的第一版 plane-token decomposition 实验。
- 不使用 RANSAC plane normal / offset / assignment 作为监督。
- 直接从点云中学习 `K` 个 plane token：

  ```text
  plane token k -> normal n_k + offset d_k
  point i -> soft assignment over K planes
  ```

- loss 由点到平面距离、assignment entropy、plane diversity、coverage 等几何项组成。
- RANSAC 结果只作为 baseline 和对照，不作为主监督。

当前在 `val_000035` 上跑过一版：

```text
/data/zhucy23u/logs/learned_plane_tokens_unsup_v1/val_000035_learned_plane_tokens.json
/data/zhucy23u/logs/learned_plane_tokens_unsup_v1/val_000035_learned_plane_tokens_assignment.npz
/data/zhucy23u/logs/learned_plane_tokens_unsup_v1/val_000035_learned_plane_token_edit_comparison.html
```

训练设置：

```text
num_planes: 4
points used: 80000 sampled points
steps: 1600
supervision: none from RANSAC labels
```

当前结果：

```text
final loss: 0.0086
soft fit: 0.0064
assignment confidence: 0.962
largest learned plane moved points: 58773 / 80000
```

但这版还有明显问题：largest token 覆盖约 73% 点，一个 token 只有约 2.4% 点，说明 learned planes 有不均衡/塌缩倾向。下一步要加强 coverage/diversity，或者引入 plane-token initialization、local smoothness、multi-sample training。

后续已经做了一个 v2 assignment-head 版本：

```text
/data/zhucy23u/logs/learned_plane_tokens_unsup_v2_assign_head/val_000035_learned_plane_tokens.json
/data/zhucy23u/logs/learned_plane_tokens_unsup_v2_assign_head/val_000035_learned_plane_tokens_assignment.npz
/data/zhucy23u/logs/learned_plane_tokens_unsup_v2_assign_head/val_000035_learned_plane_token_edit_comparison.html
```

v2 改动：

```text
新增 point assignment MLP
输入每点 xyz/rgb/radius + plane normal/offset/distance
输出每点到每个 plane token 的 assignment logits
保留 distance logit 作为几何 bias
加强 coverage/balance/assignment margin
```

v2 在 `val_000035` 上的结果：

```text
final loss: 0.0131
soft fit: 0.0086
assignment confidence: 0.998
largest learned plane moved points: 37324 / 80000
```

覆盖比例从 v1 的：

```text
0.7347, 0.1980, 0.0435, 0.0239
```

改善为 v2 的：

```text
0.4666, 0.2406, 0.1706, 0.1223
```

这说明 assignment-head 版本牺牲了一点 fit，但明显缓解了 token collapse，更接近“头自己学出可分离平面”的方向。下一步应该继续加入 local smoothness 和多样本训练。

### `train_multisample_unsupervised_plane_tokens.py`

用途：

- 多样本版 unsupervised plane-token decomposition。
- 每个样本有自己的 `K` 个 plane tokens，输出该样本的平面方程。
- 多个样本共享同一个 point assignment MLP，学习通用的点到 plane token 分配规则。
- 不使用 RANSAC 平面作为监督。

当前已在 `val_000026` 到 `val_000040` 的 15 个样本上训练过：

```text
/data/zhucy23u/logs/learned_plane_tokens_multisample_v1
```

训练设置：

```text
num_samples: 15
num_planes: 4
points per sample: 30000
steps: 1400
shared module: assignment MLP
sample-specific module: plane tokens normal/offset/logit
```

本地已拉回：

```text
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\learned_plane_tokens_multisample_v1
```

主要报告：

```text
multisample_learned_plane_tokens_report.md
multisample_learned_plane_tokens_summary.json
```

从统计看，多样本版在一些样本上 token 覆盖比较均衡：

```text
val_000026: ratios 0.265, 0.246, 0.252, 0.237
val_000037: ratios 0.315, 0.207, 0.235, 0.242
val_000036: ratios 0.250, 0.325, 0.269, 0.156
```

但部分样本仍有 token 偏大：

```text
val_000035: largest ratio 0.681
val_000038: largest ratio 0.636
```

结论：多样本训练已经能让一部分样本学出更均衡的可分离 plane tokens，但仍需要 local smoothness、局部几何特征、以及更好的 coverage/diversity 设计来避免个别样本的大 token 吞并问题。

### `evaluate_learned_plane_token_params.py`

用途：

- 评价 learned plane-token 输出的参数方程质量。
- 不只看 HTML 观感，而是量化：

  ```text
  assigned_point_count / assigned_ratio
  mean residual
  median residual
  p90 residual
  trimmed mean residual
  inlier_ratio@0.02
  inlier_ratio@0.05
  plane normal pair angle relation
  ```

运行示例：

```bash
/data/zhucy23u/conda_envs/lightrecon/bin/python evaluate_learned_plane_token_params.py \
  --input_dir /data/zhucy23u/logs/learned_plane_tokens_multisample_v1 \
  --output_dir /data/zhucy23u/logs/learned_plane_tokens_multisample_v1/eval \
  --pattern "*_multisample_learned_plane_tokens.json" \
  --trim_ratio 0.8 \
  --thresholds 0.02 0.05
```

输出：

```text
learned_plane_token_param_eval.csv
learned_plane_token_normal_relations.csv
learned_plane_token_param_eval_summary.md
learned_plane_token_param_eval_summary.json
```

### 多样本 v2：去掉强均衡，改 anti-collapse + trimmed fit

用户指出真实大墙/地面本来就应该对应大 token，所以不应该强迫每个 token 覆盖接近 `1/K`。这是合理的。

因此 v2 做了：

```text
去掉 balance loss: 不再强制每个 token 覆盖 1/K
加入 dead-token loss: 只防止 token 完全空掉
加入 trimmed fit: 用 residual 最小的 80% 点优化点到平面距离
```

服务器结果：

```text
/data/zhucy23u/logs/learned_plane_tokens_multisample_v2_trimfit
```

本地结果：

```text
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\learned_plane_tokens_multisample_v2_trimfit
```

v1/v2 指标对比：

```text
v1 mean trimmed residual: 0.01019
v1 mean inlier@0.05:      0.9499
v1 mean coverage spread:  0.2844

v2 mean trimmed residual: 0.01168
v2 mean inlier@0.05:      0.9202
v2 mean coverage spread:  0.4123
```

解释：

```text
v2 更允许真实大面对应大 token，但整体 residual 稍差，token spread 变大。
这说明“去强均衡”方向是合理的，但还需要 local smoothness / RGB feature consistency / structure prior，避免大 token 吞并非同一真实平面的区域。
```

### 多样本 v3：加入局部 RGB/空间一致性

目标：回应 `val_000037` 这类可视化中出现的斜切片问题。v2 去掉强均衡以后允许大墙/地面对应大 token，但仅靠点到平面的 residual 容易把空间上不连续的点切成数学上可拟合、结构上不合理的片。v3 在不恢复强均衡的前提下加入 local smoothness：

```text
smooth pair: 在采样点中为每个 anchor 随机找候选邻居，并选空间最近点
smooth weight: exp(-xyz_dist^2 / sigma_xyz^2 - rgb_dist^2 / sigma_rgb^2)
smooth loss: 让局部空间/RGB 相近点的 assignment 分布更接近
```

服务器结果：

```text
/data/zhucy23u/logs/learned_plane_tokens_multisample_v3_smooth
/data/zhucy23u/logs/learned_plane_tokens_multisample_v3b_smooth02
```

本地结果：

```text
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\learned_plane_tokens_multisample_v3_smooth
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\learned_plane_tokens_multisample_v3b_smooth02
```

指标对比：

```text
v1_balance:   trim=0.01019, inlier@0.05=0.9499, spread=0.2844, empty_token_samples=0/15
v2_trimfit:   trim=0.01168, inlier@0.05=0.9202, spread=0.4123, empty_token_samples=0/15
v3_smooth08: trim=0.01405, inlier@0.05=0.8895, spread=0.5309, empty_token_samples=4/15
v3b_smooth02:trim=0.01258, inlier@0.05=0.9027, spread=0.4903, empty_token_samples=2/15
```

`val_000037` 单样本上，v3b 的 trimmed residual 从 v2 的 `0.00364` 降到 `0.00247`，说明平面方程拟合更准；但 spread 到 `0.6755`，仍有大 token 吞并问题。因此结论不是“继续加平滑”，而是：

```text
1. 保留轻量 local smoothness 作为可选项；
2. 加入 effective token count / empty token 惩罚，防止 token 死亡；
3. 引入 RGB 图像边界或 DUSt3R decoder feature，使 token 更贴近真实墙面/地面边界；
4. 后续再加入结构关系指标，例如平行/垂直/共面关系，以及可编辑后点云一致性。
```

### 阶段路线修正：从 per-sample tokens 到 amortized head

当前研究线需要按阶段推进，不能直接把 stage4 的结构约束当成主方法：

```text
Stage 1: per-sample plane tokens + shared assignment MLP
  作用：验证无监督 plane decomposition 是否能跑通。
  局限：每个样本的 plane tokens 是单独优化出来的，更像 test-time fitting，不是真正的预测头。

Stage 2: amortized prediction
  作用：输入点云/feature，由网络直接预测该样本的 plane tokens。
  关键区别：plane tokens 不再是 per-sample nn.Parameter，而是 head(features) 的输出。

Stage 3: weak / auxiliary supervision
  作用：Structured3D GT plane、RANSAC 高置信候选、trimmed SVD refinement 只作为辅助约束。
  注意：这些不是唯一 teacher，不能把方法退回纯 RANSAC 蒸馏。

Stage 4: structure constraints
  作用：local smoothness、RGB/feature consistency、parallel / orthogonal / Manhattan、line-plane consistency。
  注意：这些是提升结构合理性的约束，不是 stage2 的替代品。
```

已新增 stage2 baseline 脚本：

```text
train_amortized_plane_tokens.py
```

服务器结果：

```text
/data/zhucy23u/logs/amortized_plane_tokens_stage2_v1
```

本地结果：

```text
C:\Users\admin\Documents\Codex\2026-06-01\mobaxterm\outputs\amortized_plane_tokens_stage2_v1
```

stage2 baseline 指标：

```text
stage2_amortized_v1: trim=0.01420, inlier@0.05=0.8769, spread=0.3422, empty_token_samples=0/15
val_000037: trimmed residual=0.00701, spread=0.6234
```

和 stage1 对比：

```text
stage1_v1_balance:    trim=0.01019, inlier@0.05=0.9499, spread=0.2844, empty=0/15
stage1_v2_trimfit:    trim=0.01168, inlier@0.05=0.9202, spread=0.4123, empty=0/15
stage1_v3b_smooth02:  trim=0.01258, inlier@0.05=0.9027, spread=0.4903, empty=2/15
stage2_amortized_v1:  trim=0.01420, inlier@0.05=0.8769, spread=0.3422, empty=0/15
```

解释：

```text
stage2 residual 比 stage1 略差是正常的，因为它不再给每个样本单独优化自由 plane parameters。
但它更接近真正要写进论文/项目里的 primitive head：同一套网络从点云特征直接预测 plane equations。
下一步应优先提升 stage2，而不是继续堆 stage1 的 per-sample fitting。
```

### `make_learned_plane_token_comparison.py`

用途：

- 可视化 learned plane tokens 的 before/after 编辑。
- 输入 `train_unsupervised_plane_tokens.py` 输出的 learned assignment `.npz` 和 learned plane `.json`。
- 输出 HTML，用来和 RANSAC 后处理版本对比。

## 8. 推荐接手顺序

1. 打开 `full_pointcloud_plane_edit_summary.md`，先看 26/27/28 三个样本的统计。
2. 打开三个 `*_full_pointcloud_edit_comparison.html`，看 before/after 是否直观。
3. 跑一个新样本，例如 `sample_idx 29`。
4. 改 `--threshold` 和 `--edit_delta`，观察平面数量、绑定点数、编辑效果变化。
5. 再考虑把 learned plane head 接回来。

如果要继续学习头，推荐顺序是：

1. 先复现 `train_pseudo_plane_head_from_npz.py` 的小 MLP 结果。
2. 加入局部几何特征，例如邻域 PCA normal、curvature、点到候选平面的距离。
3. 将输入换成 DUSt3R decoder feature + pointmap 坐标。
4. 从 per-point 参数预测升级到 plane-token 参数预测。
5. 再加 line head，形成 plane/line primitive 联合预测。
