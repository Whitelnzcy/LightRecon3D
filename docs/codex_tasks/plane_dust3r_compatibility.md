# Plane-DUSt3R论文基线兼容性任务

日期：2026-07-17

状态：服务器预检已完成。17个冻结场景均有5张`perspective/empty`图像，但没有任何`perspective/full`图像；官方仓库和两个checkpoint尚未安装，也尚未报告Plane-DUSt3R推理结果。

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

## 服务器预检结论

2026-07-17在提交`8e6317e`上运行预检，结果为：

```text
scenes=17
empty five-view groups=17/17
official full groups=0/17
official repository=false
Plane-DUSt3R checkpoint=false
NonCuboidRoom checkpoint=false
```

因此无法复现论文原生`perspective/full`结果，也不能把论文表中的数字与当前17场景表直接作差。这不是方法运行失败，而是本地Structured3D渲染模式与官方协议不一致。

后续采用同输入复现：建立独立兼容数据根目录，让官方程序预期的`full/<position>`只读链接到当前冻结的`empty/<position>`。链接不会修改或复制源图像，manifest明确记录实际输入仍为`empty`。Plane-DUSt3R和LightRecon3D由此读取相同scene、room、position和RGB文件。该实验只能称为same-input reproduction，不能称为Plane-DUSt3R论文原生协议。

## 执行顺序

1. **完成：**预检17个冻结scene的`empty/full`图像和外部资源。
2. 生成17场景same-input兼容目录；源数据保持只读。
3. 克隆官方仓库及子模块，记录主仓库和所有子模块commit。
4. 根据官方仓库实际README建立隔离Python环境并下载两个checkpoint，不升级`lightrecon`环境中的NumPy、PyTorch或CUDA。
5. 先在`scene_00180`运行same-input GPU smoke。外部包装器必须取消静默异常并输出逐scene账本。
6. smoke通过后，冻结scene ID、视图数、权重SHA256、环境和配置，运行17场景。
7. 审核官方输出是否能在不补造物体平面标签的条件下映射到共同point cache；通过后再实现VOI/RI/SC adapter。若不能，则报告同输入下各方法各自任务指标，并明确不构成同任务排名。

下一次服务器执行：

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_20260717_v1 \
bash run_plane_dust3r_same_input_materialization.sh

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_repository_setup_20260717_v1 \
bash setup_plane_dust3r_repository.sh
```

第一条命令是CPU只读任务，默认使用绝对软链接。第二条命令只克隆官方代码和子模块，不下载checkpoint、不安装依赖、不启动训练或GPU推理。外部仓库与现有LightRecon3D仓库分开保存。
