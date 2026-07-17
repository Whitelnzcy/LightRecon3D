# 17场景验证实验记录

日期：2026-07-17

状态：主批次已经完成。冻结验证集共选出17个独立场景，其中14个完成了从输入物化、DUSt3R全局对齐、普通RANSAC、学习support引导RANSAC、评价到三维渲染的完整流程。其余3个场景因服务器基础设施异常未进入有效方法比较。定向恢复任务已经建立，但本记录只保存第一次不可覆盖批次的事实。

## 主结果

所有有效场景中的两种方法读取完全相同的有序DUSt3R全局点云缓存。学习support引导RANSAC在14个有效场景中全部取得更高的配对F1。

| 指标 | 普通RANSAC | 引导RANSAC | 引导方法变化 | 有效场景 |
|---|---:|---:|---:|---:|
| 配对F1 | 0.644675 | 0.721296 | +0.076621 | 14 |
| matched IoU | 0.500448 | 0.688247 | +0.187798 | 14 |
| overmerge excess | 2.928571 | 1.500000 | -1.428571 | 14 |
| 方法阶段耗时（秒） | 19.945028 | 18.356680 | -1.588348 | 14 |

场景胜率为14/14，中位配对F1增益为`+0.069861`。在把各场景视为独立配对的条件下，14次同方向变化对应的精确双侧符号检验为`p=0.0001220703125`。该统计只说明当前冻结Structured3D验证协议中的一致性，不代表在其他数据集或真实照片上已经达到同样效果。

耗时一行只比较两个平面提取阶段，不是包含DUSt3R推理、全局对齐、输入物化和评价的端到端耗时。虽然14场景的平均值更低，早期8场景冻结效率gate没有通过，因此报告不得将加速写成主要贡献。

## 失败记录

第一次批处理报告6条失败记录，实际对应3个场景。每个场景分别在物化账本和批处理账本中出现一次。

| 场景 | 首次记录阶段 | 已确认原因 |
|---|---|---|
| scene_00190 | Stage1/Stage2输入物化 | 平台`[bootstrap] Fail to init session`。包装层返回0，但没有生成Stage1 manifest或NPZ |
| scene_00194 | Stage1输入物化 | 平台Server Service连接失败，进程退出码255 |
| scene_00197 | Stage1输入物化 | DUSt3R迁移到GPU时CUDA设备busy/unavailable |

这3个场景没有产生普通RANSAC与引导RANSAC的有效成对结果，因此不能算作任何一种方法的算法失败，也不能进入14场景均值。主报告应同时给出`14/17`完整执行率，并说明失败来自运行平台。原始失败日志保留，恢复任务写入新目录，不能覆盖第一次批次。

## 定性结果

主批次生成了真实三维点云上的平面实例对比。左右方法使用相同点顺序和相机视角，颜色表示各自方法内部的平面编号，不能把左右相同颜色理解为同一平面。

从总览图看，`scene_00186`中引导方法把9个预测平面整理为6个，主要墙面更连续；`scene_00192`由8个变为7个，配对F1提高`0.075`；`scene_00195`的配对F1从`0.376`提高到`0.574`；`scene_00198`从`0.358`提高到`0.612`，是当前增益最大的复杂场景之一。`scene_00181`和`scene_00187`变化很小，可用于说明引导过程没有明显破坏原本已经稳定的几何结果。`scene_00184`存在容易被误解为缺失区域的深色实例，不适合作为正文主图。

当前14场景总览适合作为完整性附录。正文主图应选择`scene_00186`、`scene_00195`和`scene_00198`，同时展示RGB全局点云、点对齐GT、普通RANSAC和引导RANSAC，并使用GT匹配后的统一颜色。

## 控制文件

```text
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_bundle/large_scale_summary.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_bundle/large_scale_summary.md
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_bundle/scene_artifact_index.csv
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_bundle/failures.csv
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_audit/final_method_audit.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_batch/batch_execution.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_3d_visualization/visualization_manifest.json
/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_3d_visualization/all_scenes_final_3d_contact_sheet.png
```

报告中的14场景数字必须从上述JSON和CSV读取。8场景结果可以保留为较早的冻结确认实验，但不能再称为最终大规模主结果。
