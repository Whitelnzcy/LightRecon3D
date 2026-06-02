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

## 8. 推荐接手顺序

1. 打开 `full_pointcloud_plane_edit_summary.md`，先看 26/27/28 三个样本的统计。
2. 打开三个 `*_full_pointcloud_edit_comparison.html`，看 before/after 是否直观。
3. 跑一个新样本，例如 `sample_idx 29`。
4. 改 `--threshold` 和 `--edit_delta`，观察平面数量、绑定点数、编辑效果变化。
5. 再考虑把 learned plane head 接回来。
