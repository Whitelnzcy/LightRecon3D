# 17场景验证实验记录

日期：2026-07-17

状态：最终合并批次已经完成。冻结验证集共选出17个独立场景，第一次运行完成14个，另外3个因服务器基础设施异常中断。不可覆盖的定向恢复补齐了3个场景，最终17个场景全部通过从输入物化、DUSt3R全局对齐、普通RANSAC、学习support引导RANSAC、评价到三维渲染的完整流程。

## 主结果

所有场景中的两种方法读取完全相同的有序DUSt3R全局点云缓存。学习support引导RANSAC在16个场景中取得更高的配对F1，在1个场景中轻微下降。

| 指标 | 普通RANSAC | 引导RANSAC | 引导方法变化 | 有效场景 |
|---|---:|---:|---:|---:|
| 配对F1 | 0.632406 | 0.706348 | +0.073942 | 17 |
| matched IoU | 0.489136 | 0.678036 | +0.188899 | 17 |
| overmerge excess | 2.882353 | 1.470588 | -1.411765 | 17 |
| VOI（nats，越低越好） | 1.493965 | 1.172062 | -0.321904 | 17 |
| Rand Index | 0.789837 | 0.834620 | +0.044783 | 17 |
| Segmentation Covering（symmetric） | 0.561563 | 0.658550 | +0.096987 | 17 |
| 方法阶段耗时（秒） | 19.826422 | 17.892183 | -1.934239 | 17 |

场景胜率为16/17，中位配对F1增益为`+0.064718`。在把各场景视为独立配对的条件下，16胜1负对应的精确双侧符号检验为`p=0.000274658203125`。该统计只说明当前冻结Structured3D验证协议中的方向较一致，不代表在其他数据集或真实照片上已经达到同样效果。

耗时一行只比较两个平面提取阶段，不是包含DUSt3R推理、全局对齐、输入物化和评价的端到端耗时。虽然17场景的平均值更低，早期8场景冻结效率gate没有通过，因此报告不得将加速写成主要贡献。

## 恢复记录

第一次批处理报告6条失败记录，实际对应3个场景。每个场景分别在物化账本和批处理账本中出现一次。

| 场景 | 首次记录阶段 | 已确认原因 |
|---|---|---|
| scene_00190 | Stage1/Stage2输入物化 | 平台`[bootstrap] Fail to init session`。包装层返回0，但没有生成Stage1 manifest或NPZ |
| scene_00194 | Stage1输入物化 | 平台Server Service连接失败，进程退出码255 |
| scene_00197 | Stage1输入物化 | DUSt3R迁移到GPU时CUDA设备busy/unavailable |

恢复任务使用相同的selection plan、checkpoint、DUSt3R权重和冻结配置，只为这3个scene ID写入新目录。3项输入物化和3项最终批处理全部通过。合并账本包含17个成功场景、0个失败场景，最终`failures.csv`为空。原始失败日志没有删除，`recovery_merge.json`保存了原始批次、恢复批次及控制文件的路径和SHA256。

恢复场景中，`scene_00190`的配对F1从约`0.399`提高到`0.576`，`scene_00197`从`0.514`提高到`0.523`。`scene_00194`从`0.813`降到`0.811`，约为`-0.002`，是17个场景中唯一的负增益。最终报告必须保留这个反例。

## 定性结果

主批次生成了真实三维点云上的平面实例对比。左右方法使用相同点顺序和相机视角，颜色表示各自方法内部的平面编号，不能把左右相同颜色理解为同一平面。

从总览图看，`scene_00186`中引导方法把9个预测平面整理为6个，主要墙面更连续；`scene_00192`由8个变为7个，配对F1提高`0.075`；`scene_00195`的配对F1从`0.376`提高到`0.574`；`scene_00198`从`0.358`提高到`0.612`，是当前增益最大的复杂场景之一。`scene_00181`和`scene_00187`变化很小，可用于说明引导过程没有明显破坏原本已经稳定的几何结果。`scene_00184`存在容易被误解为缺失区域的深色实例，不适合作为正文主图。

当前17场景总览适合作为完整性附录。正文主图应选择`scene_00186`、`scene_00195`和`scene_00198`，同时展示RGB全局点云、点对齐GT、普通RANSAC和引导RANSAC，并使用GT匹配后的统一颜色。`scene_00194`可放入失败或边界案例图，说明引导方法并非逐场景必胜。

## 控制文件

```text
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_bundle/large_scale_summary.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_bundle/large_scale_summary.md
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_bundle/scene_artifact_index.csv
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_bundle/failures.csv
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_merged/recovery_merge.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_merged/combined_batch/batch_execution.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_merged/combined_batch/aggregate_metrics.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_3d_visualization/visualization_manifest.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_3d_visualization/all_scenes_final_3d_contact_sheet.png
```

报告中的17场景数字必须从上述JSON和CSV读取。第一次14场景结果保留为恢复前记录，8场景结果保留为较早的冻结确认实验，两者都不能再称为最终大规模主结果。
