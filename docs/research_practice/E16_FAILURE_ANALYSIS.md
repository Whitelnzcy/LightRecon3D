# E16 失败分析与方法定位

## 评价边界

E16 的合同有效。128 个最终场景没有参与共享 identity 训练、阈值校准或 checkpoint 选择。B16 与 M16 的 256 份 prediction 在解析最终 GT 前完成并冻结，128 对 support hash 全部一致。评价器没有扩张、删除或重分配 support，也没有执行 refit、RANSAC 或 GT guided repair。

机械 verdict 是 `Mixed_Result`。科学结论需要再具体一点：

> 当前 B/M 都是在稀疏 direct support 上恢复平面身份，不能直接覆盖完整 GT 平面。M16 相比 B16 几乎没有带来可测的身份恢复提升。

## 主结果

| 指标 | B16 | M16 | M16 - B16 |
|---|---:|---:|---:|
| Plane F1 | 0.000000 | 0.000000 | 0.000000 |
| matched IoU | 0.000000 | 0.000000 | 0.000000 |
| partition pair F1 | 0.358892 | 0.358968 | +0.0000758 |
| GT-support assignment | 0.111023 | 0.111023 | 0.000000 |
| full-cache label rate | 0.111012 | 0.111012 | 0.000000 |
| GT-plane label rate | 0.972479 | 0.972479 | 0.000000 |
| 预测平面总数 | 878 | 1292 | +414 |

Partition pair F1 的配对 bootstrap 95% 区间为 `[-0.0000552, +0.0002065]`，跨过 0。128 个场景的胜、平、负数量是 `78/1/49`。这个结果不能支持 M16 有稳定提升。

## Plane F1 为 0 的含义

预测与 GT 使用同一批全局缓存点，但 support 域并不相同。B16/M16 只给约 11.1% 的缓存点分配平面标签，GT 平面点约占 97.2%。当 IoU 的分母包含完整 GT 平面时，稀疏预测很难达到 0.5 的 TP 门槛。

因此，Plane F1 和 matched IoU 在 E16 中首先测到了 support completeness，而不是 B/M 的局部分区差异。这个结果不能写成 "模型完全没用"，也不能从报告中删去。准确说法是：

> 注册评价暴露了稀疏 direct support 与完整可见平面 support 之间的评价域差异。B16 和 M16 都没有完成 full-support plane recovery，所以 Plane F1 无法区分两者的 fragment identity 能力。

Partition pair F1 可以比较固定 GT 域上的实例关系，但大量未分配点仍然限制了它的解释范围。即使采用这个相对指标，M16 的平均变化也接近 0。

## M16 的额外拆分

M16 的预测平面总数从 878 增加到 1292，多出 414 个实例。增加的实例没有转化成 Plane TP。以下三个场景由冻结指标机械选出，它们的平面数明显增加，同时 partition pair F1 下降：

| 场景 | B16 平面数 | M16 平面数 | 数量变化 | Pair F1 变化 |
|---|---:|---:|---:|---:|
| `scene_00597` | 8 | 20 | +12 | -0.000656 |
| `scene_01767` | 7 | 17 | +10 | -0.000811 |
| `scene_01619` | 5 | 14 | +9 | -0.000741 |

这些例子没有证明 learned split 在所有局部都错误。它们说明当前 split 的输出很难通过后续 identity clustering 转化为更完整、更正确的平面实例。Split 只重排已有 support，无法增加 support 覆盖。

## B/M 近似一致的场景

`scene_02817`、`scene_02785`、`scene_01218` 和 `scene_02976` 的 B/M 最终平面数相同，partition pair F1 差值接近 0。可视化先用 frozen support overlap 对 M 的显示颜色做匹配，再比较左右分区。颜色匹配只用于显示，不改变 prediction ID 或评价结果。

这些场景说明 learned split 经 remerge 后经常回到与 B16 接近的最终分区。它没有形成可重复的独立收益。

## 瓶颈排序

1. Stage1 direct support 覆盖不足。预测只触及完整缓存的一小部分。
2. 最终 B/M 没有包含 full-support recovery，未分配区域不会被补回。
3. Learned split 只能修改稀疏 support 内部的 fragment 边界。
4. M16 产生更多实例，但完整平面 TP、matched IoU 和 coverage 都没有改善。

这个排序来自冻结的 denominator metrics、预测平面数和配对统计，不依赖人工挑选的单场景观感。

## 最终定位

B16 是最终默认方法，理由是结构更简单，而 128 场景结果与 M16 基本相同。M16 作为研究扩展和完整消融保留。报告采用以下结论：

> 学习式 intra-fragment split 在当前稀疏 direct-support 和 identity 恢复框架下，未产生具有统计意义的整体改进。它轻微改变了局部分区一致性，同时显著增加预测平面数量，但没有改善完整平面匹配。

项目不再修改模型或评价器。E16 后只整理可视化、失败分析和报告。

## 图表与证据

```text
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/figures/
  e16_main_results_table.png
  e16_pair_f1_delta_distribution.png
  e16_similar_scene_gallery.png
  e16_extra_split_gallery.png
  e16_support_domain_gap.png
  figure_manifest.json
```

数值来源：

```text
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/summary.json
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/per_scene_metrics.json
results/targeted_GT_sidecar_and_final_BM_e16_20260730_v1/bootstrap_statistics.json
```
