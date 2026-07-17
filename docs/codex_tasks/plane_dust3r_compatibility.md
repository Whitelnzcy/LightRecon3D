# Plane-DUSt3R论文基线兼容性任务

日期：2026-07-17

状态：已完成静态协议审计和服务器预检工具；尚未下载4.41 GB Plane-DUSt3R权重，也尚未报告其推理结果。

## 目标

复现Plane-DUSt3R官方方法，给LightRecon3D补充第一条最接近的无位姿论文基线。比较必须区分两层：

1. **官方原协议结果**：按Plane-DUSt3R仓库读取Structured3D `perspective/full`图像，报告其房间布局IoU、pixel error、boundary error和depth RMSE等原生指标，单独成表。
2. **共同平面分组指标**：只有在官方输出能够无歧义映射到同一个有序DUSt3R全局点缓存后，才计算VOI、RI、Segmentation Covering、pairwise F1和matched IoU。

不得把这两层结果混成同一个排名。

## 已确认的协议差异

| 项目 | LightRecon3D冻结实验 | Plane-DUSt3R官方评测 |
|---|---|---|
| 数据 | 17个Structured3D独立scene ID | Structured3D场景/房间遍历 |
| 图像模式 | `perspective/empty`，每场景5视图 | `perspective/full`，官方脚本读取1至5视图 |
| 位姿输入 | 不提供，DUSt3R全局对齐 | 不提供，DUSt3R系列点图流程 |
| 输出 | 完整全局点缓存上的有界平面实例分组 | 结构房间布局平面、深度和边界 |
| 原生指标 | VOI、RI、SC、F1、IoU、overmerge | 2D layout IoU、pixel/boundary error、depth RMSE等 |

Plane-DUSt3R会抑制物体并恢复结构布局，LightRecon3D评价的是完整全局点云上的平面分组。即使scene ID相同，两者也不是天然同一输出域。

## 外部资源

* 官方代码：https://github.com/justacar/Plane-DUSt3R
* 官方论文：https://openreview.net/forum?id=DugT77rRhW
* 官方权重：https://huggingface.co/yxuan/Plane-DUSt3R
* Plane-DUSt3R checkpoint：`checkpoint-best-onlyencoder.pth`，约4.41 GB
* 额外依赖：NonCuboidRoom代码与`Structured3D_pretrained.pt`

官方评测脚本还需要在正式批量运行前加一层审计包装：仓库版本中的保存开关存在硬编码，异常处理会直接跳过场景。外部基线账本必须记录每个scene的成功、失败阶段和输出路径，不能把异常场景静默排除。

## 执行顺序

1. 在服务器运行只读兼容性预检，统计17个冻结scene的`full`图像是否齐全，并检查官方仓库、子模块和两个checkpoint。
2. 只有至少一个scene具备`full`图像后，才下载大权重并做单场景官方原协议smoke。
3. smoke通过后，冻结Plane-DUSt3R commit、scene ID、视图数、权重SHA256、环境和原生指标配置，运行17场景或所有兼容scene。
4. 保存逐scene失败账本和原生结果，形成独立论文基线表。
5. 审核官方输出是否能在不补造物体平面标签的条件下映射到共同point cache；通过后再实现VOI/RI/SC adapter，否则只保留官方原生指标表并明确任务差异。

服务器预检：

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_compatibility_20260717_v1 \
bash run_plane_dust3r_compatibility_preflight.sh
```

预检是CPU只读任务。即使官方仓库和权重还不存在，它也会写出17个scene的`empty/full`图像清单和缺失项，不会启动训练、GPU推理或下载。
