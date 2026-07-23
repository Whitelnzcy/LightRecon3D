# LightRecon3D resume evidence

Date: 2026-07-15

This note separates facts that can be used now from results that still require
the final multi-scene experiment.

## Draft rewrite

```text
LightRecon3D：无姿态先验室内场景的结构化三维重建 | PyTorch, DUSt3R, Structured3D

面向弱纹理室内场景，基于 DUSt3R 全局对齐 pointmap 开发有限支撑平面重建流程，完成平面实例预测、局部几何验证、跨视角 support 映射、全局平面拟合及 NPZ/PLY/GLB 可编辑结果导出。

训练 plane-query 多尺度平面实例模型，验证集 mean IoU 为 0.8431，leakage 为 0.0621；建立显式的视角与像素坐标约定，使用 (alignment_view_index, x, y) 精确连接二维 support 和全局三维点，避免直接混合不同图像对的局部坐标。

搭建可复现实验工具，记录 Git SHA、输入 SHA256、运行时间和失败样例；实现结构线检测与三维提升、批量输入预检和 JSON/CSV/Markdown 汇总，并通过对照实验否定了会降低平面损失但损害真实几何的反馈方案。
```

## Remaining AI-like or weak points

* The first sentence lists many stages and reads more like project
  documentation than a resume result.
* "可复现实验工具" is defensible, but the concrete artifacts should replace
  the generic label.
* The negative-result sentence is useful in an interview, but it is too long
  for a one-page resume.
* Cross-scene gains, latency, parameter count, checkpoint size and peak memory
  do not have final archived numbers yet and must not be added now.

## Final resume version usable now

```text
LightRecon3D：无姿态先验室内场景的结构化三维重建 | PyTorch, DUSt3R, Structured3D

基于 DUSt3R 全局对齐 pointmap 开发室内平面重建流程，将二维平面 support 映射到统一三维坐标，完成局部几何验证、跨视角平面拟合和 NPZ/PLY/GLB 可编辑结果导出。

设计并训练 plane-query 多尺度平面实例模型，验证集 mean IoU 为 0.8431，leakage 为 0.0621；以 (alignment_view_index, x, y) 作为精确连接键，解决不同图像对局部坐标不可直接合并的问题。

编写结构线三维提升、输入校验和实验汇总工具，自动保存输入 SHA256、Git 版本、运行时间及失败记录；对平面反馈方案进行固定缓存对照，发现其平面损失下降但几何误差上升，因此停止该路线并保留负结果分析。
```

## Add after the final batch only

Replace one engineering sentence with a measured cross-scene result only when
the archived final CSV exists. Candidate fields are the number of independent
scenes, identity/support/geometry gain over RANSAC, P50/P95 latency, trainable
parameter count, checkpoint size and peak GPU memory. Do not quote the current
single-scene identity F1 as a general result.
