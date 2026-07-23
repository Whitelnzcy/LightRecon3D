# 17场景公开常用平面分组指标记录

日期：2026-07-17

这组指标从已经归档的17场景普通RANSAC和support引导RANSAC NPZ中只读计算，没有重新运行DUSt3R、Stage1、RANSAC或训练。17个独立Structured3D场景全部进入评价，普通方法和引导方法使用相同的point-aligned GT与相同点顺序。

| 指标 | 普通RANSAC | support引导RANSAC | 变化 | 场景数 |
|---|---:|---:|---:|---:|
| pairwise F1 ↑ | 0.632406 | 0.706348 | +0.073942 | 17 |
| matched IoU ↑ | 0.489136 | 0.678036 | +0.188899 | 17 |
| overmerge excess ↓ | 2.882353 | 1.470588 | -1.411765 | 17 |
| Variation of Information（nats）↓ | 1.493965 | 1.172062 | -0.321904 | 17 |
| Rand Index ↑ | 0.789837 | 0.834620 | +0.044783 | 17 |
| Segmentation Covering（symmetric）↑ | 0.561563 | 0.658550 | +0.096987 | 17 |

VOI、RI和Segmentation Covering属于平面实例分组论文常用的评价指标。三个指标与原有F1、matched IoU和overmerge结论方向一致，没有出现内部F1提高但公开常用分组指标恶化的情况。未分配标签`-1`在评价中保留为一个预测分组，方法不能通过删除困难点获得更高分数。VOI使用自然对数，Segmentation Covering取pred-to-GT与GT-to-pred的对称平均。

这仍然是项目冻结的Structured3D 17场景内部成对实验，不是ScanNetV2、ScanNet++或Plane-DUSt3R官方协议上的公开排行榜结果。论文方法只能在数据划分、输入位姿条件、输出域和评价代码对齐后比较。

服务器证据：

```text
git_sha=b08da2b
/gemini/data-1/lightrecon_runs/research_practice_public_metrics_20260717_v1/public_plane_metrics.json
/gemini/data-1/lightrecon_runs/research_practice_public_metrics_20260717_v1/public_plane_metrics.csv
/gemini/data-1/lightrecon_runs/research_practice_public_metrics_20260717_v1/public_plane_metrics.md
```
