# LightRecon3D

LightRecon3D 从无相机位姿或弱位姿先验的室内图像中恢复平面实例。项目使用 DUSt3R 建立统一三维坐标，再把 Stage1 的二维平面 support 映射到全局点图。输出保留平面参数、有限支撑点和像素来源，便于检查和编辑。

方法实验已经结束。当前仓库保留两个经过前瞻性冻结的最终方法：

```text
B16 = topology-logistic 全局平面身份恢复
M16 = B16 + learned intra-fragment split
```

B16 是默认方法。M16 作为研究扩展和消融实验完整保留。

## 最终流水线

```text
无位姿 RGB 图像
-> DUSt3R 点图与全局对齐
-> Stage1 稀疏平面 support
-> 显式 (view, x, y) 到全局点索引映射
-> topology-logistic 全局身份恢复
-> 可选 learned intra-fragment split
-> 确定性平面参数估计
-> 有限平面 primitive 与来源记录
```

坐标映射不依赖数组形状或文件名猜测。每个 support 点都保存视图编号、点图像素和全局点索引。B16 与 M16 使用相同输入、identity 特征、logistic 模型、聚类阈值、平面参数估计和评价器。两者唯一的区别是是否启用 learned split。

## 128 场景独立验证

最终比较使用 128 个此前未参与训练和调参的 Structured3D 场景，其中 124 个场景使用 5 个视图，4 个场景使用 4 个视图。旧 17 个场景只用于训练共享 identity 模型、冻结阈值和工程测试。

全部 256 份 B16/M16 prediction 在最终 GT 解封前完成并冻结。128 对 B/M support hash 全部一致。随后使用同一组 point-aligned GT 和 non-mutating evaluator 评价，评价前后 prediction hash 不变。

| 指标 | B16 | M16 | M16 - B16 |
|---|---:|---:|---:|
| Plane F1 | 0.000000 | 0.000000 | 0.000000 |
| matched IoU | 0.000000 | 0.000000 | 0.000000 |
| partition pair F1 | 0.358892 | 0.358968 | +0.000076 |
| GT-support assignment | 0.111023 | 0.111023 | 0.000000 |
| full-cache label rate | 0.111012 | 0.111012 | 0.000000 |
| predicted planes, total | 878 | 1292 | +414 |

Partition pair F1 的配对 bootstrap 结果为：

```text
mean delta = +0.0000758
95% CI = [-0.0000552, +0.0002065]
positive / zero / negative scenes = 78 / 1 / 49
```

这个差值没有统计稳定性。M16 产生了更多平面实例，但没有获得完整 GT support 上的有效 TP。

## 如何理解 Plane F1 为 0

Plane F1 为 0 不等于模型完全没有作用。B16 和 M16 的任务域是 Stage1 产生的稀疏 direct support，而注册评价使用完整可见 GT 平面 support。两者的覆盖范围差别很大：

```text
prediction full-cache label rate = 约 11.1%
GT plane label rate = 约 97.2%
```

在这种条件下，稀疏预测与完整 GT 平面的 IoU 很难达到 0.5。Plane F1 和 matched IoU 因此主要暴露 support completeness 问题，不能有效区分 B16 与 M16 的 fragment identity 能力。B/M 的相对比较应看 partition pair F1，但该指标同样没有显示 M16 的稳定收益。

最终结论是：

> B16 作为稳定默认方案。M16 是经过独立验证但未取得显著整体收益的研究扩展。当前主要限制不是 fragment 内部拆分，而是稀疏 support 到完整平面 support 之间的恢复缺口。

## 结果图

报告图位于：

```text
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/figures/
```

其中包括：

- B16/M16 主结果表
- 128 场景 partition pair F1 配对差值分布
- 4 个 B/M 结果近似一致的场景
- 3 个 M 额外拆分但未改善的场景
- 完整 GT support 与稀疏预测 support 的实际视图对比

可重复生成这些图：

```powershell
E:\anaconda\envs\ai\python.exe visualize_e16_final_results.py
```

脚本只读取冻结结果，不运行模型，不修改 prediction、GT 或 evaluator 输出。

## 失败模式

当前瓶颈按实验结果排序如下：

1. Stage1 direct support 覆盖不足。
2. 最终方法没有恢复完整可见平面 support。
3. Learned split 只能调整已有稀疏 support 内部的分区。
4. M16 增加了碎片和预测平面数，但无法补回未分配区域。

详细分析见 `docs/research_practice/E16_FAILURE_ANALYSIS.md`。

## 复现实验证据

E16 的权威结果目录：

```text
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/
```

主要文件：

```text
source_manifest.json
method_B_manifest.json
method_M_manifest.json
prediction_manifest.json
prediction_manifest.sha256
point_aligned_gt_manifest.json
per_scene_metrics.csv
paired_effects.csv
bootstrap_statistics.json
summary.json
forbidden_call_audit.json
```

完整执行回执：

```text
docs/codex_returns/CODEX_RETURN_targeted_GT_sidecar_and_final_BM_e16_20260730.md
```

## 项目状态

模型实验已经关闭。后续工作只整理图表、失败分析和报告，不再执行以下操作：

```text
修改 evaluator
扩张 prediction support
重新训练 M
调整 split 或 identity threshold
增加最终评价场景
恢复 L/G/J
```

旧实验、负结果和 checkpoint 均保留。它们记录了项目从 pair-local 几何、guided RANSAC 到最终 prospective B/M 验证的完整过程。
