# Plane-DUSt3R论文基线兼容性任务

日期：2026-07-17

状态：17场景same-input兼容目录和官方仓库已经准备完成。Plane-DUSt3R固定在commit `9a1ae50650ec6d706bf329352aaaf49efded90a0`。v2隔离环境已创建，Python兼容依赖已安装到MMCV之前，但PyTorch 2.2与MKL 2025运行时冲突；完整依赖和两个checkpoint尚未准备完成，也尚未报告Plane-DUSt3R推理结果。

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

同输入物化已在服务器完成：17个独立scene、17个room、85张RGB全部通过，每组位置均为`0,1,2,3,4`。数据根为：

```text
/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_20260717_v1/dataset
```

官方仓库安装也已完成，主commit为`9a1ae50650ec6d706bf329352aaaf49efded90a0`。`git submodule status --recursive`没有条目，当前`MASt3R`和`NonCuboidRoom`内容由主仓库直接保存。README指定Python 3.11、PyTorch 2.2.0、torchvision 0.17.0和CUDA 11.8；两个checkpoint分别来自Hugging Face和NonCuboidRoom公开Google Drive文件。

## 执行顺序

1. **完成：**预检17个冻结scene的`empty/full`图像和外部资源。
2. **完成：**生成17场景、85张图的same-input兼容目录；源数据保持只读。
3. **完成：**克隆官方仓库并固定commit `9a1ae506`。
4. 根据官方README建立隔离Python 3.11、PyTorch 2.2.0、CUDA 11.8环境并下载两个checkpoint，不升级`lightrecon`环境中的NumPy、PyTorch或CUDA。PyTorch、pip和MMCV约束已经补齐，v4增加MKL 2024.0运行时锁定，待服务器执行。
5. 在`scene_00180`运行same-input GPU smoke。包装器在独立runtime中把官方硬编码`save=False`改为使用命令行保存开关，并把异常后的静默`continue`改为打印堆栈并失败；官方仓库本身不修改。
6. smoke通过后，冻结scene ID、视图数、权重SHA256、环境和配置，运行17场景。
7. 审核官方输出是否能在不补造物体平面标签的条件下映射到共同point cache；通过后再实现VOI/RI/SC adapter。若不能，则报告同输入下各方法各自任务指标，并明确不构成同任务排名。

下一次服务器执行会下载约4.41 GB Plane-DUSt3R权重，并安装隔离conda环境：

```bash
cd /gemini/code/LightRecon3D
git switch codex/bounded-support-head
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v4 \
bash prepare_plane_dust3r_external.sh

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_smoke_scene00180_20260717_v4 \
bash run_plane_dust3r_same_input_smoke.sh
```

环境和checkpoint写入`/gemini/data-1/lightrecon_envs`与`/gemini/pretrain/Plane-DUSt3R`，不改变现有`lightrecon`环境。准备脚本保存conda列表、pip freeze、GPU可用性、官方commit和两个checkpoint的SHA256。若大文件下载中断，换一个新的`OUT_DIR`重跑即可从相同checkpoint路径续传。

大文件下载默认使用隔离conda环境中的`aria2c`进行16路断点续传。已有单连接`wget`部分文件会原地续传，不要删除。脚本不再使用最低文件大小判断完整性，而是调用PyTorch实际加载两个checkpoint；截断文件即使超过1 GB也不能通过。可选后端为`aria2`、`hf_xet`和`wget`。Hugging Face官方线路缓慢时，可以显式设置`HF_ENDPOINT`，但最终仍必须通过PyTorch加载和SHA256记录。

## 2026-07-17依赖安装失败与v2修复

第一次隔离环境安装没有进入checkpoint下载。`MASt3R/dust3r/requirements.txt`中的未固定`torch`和`torchvision`让pip卸载了conda安装的PyTorch 2.2.0，并安装PyTorch 2.13.0、CUDA 13依赖和NumPy 2.4.6。随后`NonCuboidRoom/requirements.txt`要求`scipy==1.3.1`；该版本不支持Python 3.11，pip尝试构建NumPy 1.14.5并失败。旧环境`planedust3r-py311-torch220-cu118`只属于外部基线，已污染但没有影响`lightrecon`。

修复后的安装器默认使用全新的`planedust3r-py311-torch220-cu118-v2`，不删除或修补旧环境。它先固定conda PyTorch 2.2.0、torchvision 0.17.0、torchaudio 2.2.0和CUDA 11.8，再生成三份可审计的清理后依赖文件。清理只控制PyTorch核心包以及NonCuboidRoom的旧版本依赖，其余官方依赖原样保留。NumPy固定为1.26.4，SciPy固定为1.11.4；OpenCV、MMCV、Numba、Pillow等也固定为仍支持Python 3.11且接近原API的版本。每次pip安装都使用约束文件，安装完成后核对包版本、`torch.version.cuda`并执行`pip check`。只有全部检查通过才写入完成标记并开始下载checkpoint。

服务器重跑时不要指定旧`ENV_DIR`，也不要先运行smoke：

```bash
cd /gemini/code/LightRecon3D
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v4 \
bash prepare_plane_dust3r_external.sh
```

## 2026-07-24 统一point-cache评价完成

已实现以下三个工具：

```text
adapt_plane_dust3r_to_point_cache.py
evaluate_plane_dust3r_common_domain_batch.py
visualize_common_point_cache_partitions.py
```

统一连接键为`(alignment_view_index,x,y)`。两边均使用相同五张输入和相同
`288x512`像素网格，因此不需要、也没有使用3D最近邻对齐。Plane-DUSt3R的
官方layout重建逻辑负责生成每视图结构平面标签，`node_data.json`提供跨视图
全局墙面身份；标签采样到LightRecon3D的有序缓存后，仅使用预测支持点重新
拟合平面。GT标签不参与适配或拟合，只进入原有统一评测器。

最终输出：

```text
local_outputs/plane_dust3r_common_domain_17scene_20260724_v2/
status pass
scenes 17/17
match_iou 0.5
fragmentation_iou 0.1
min_plane_points 64
```

17场景均值：

```text
method                    precision  recall   F1       IoU      VOI↓     RI       SC
ordinary RANSAC           0.3748     0.3594   0.3587   0.5239   1.4813   0.7918   0.5641
learning-guided RANSAC    0.5484     0.5497   0.5427   0.6802   1.1721   0.8347   0.6590
Plane-DUSt3R              0.8591     0.7735   0.8098   0.8305   0.3982   0.9494   0.8752
```

Plane-DUSt3R相对学习引导RANSAC的逐场景配对结果：

```text
Plane F1: 14胜/2平/1负，均值提升0.2671
matched IoU: 16胜/0平/1负，均值提升0.1503
VOI: 16胜/0平/1负，平均改善0.7739 nats
RI: 16胜/0平/1负，均值提升0.1147
SC: 16胜/0平/1负，均值提升0.2163
```

`scene_00197`是唯一明确坏例，统一Plane F1为0，与原生指标坏例一致。适配后
Plane-DUSt3R平均标签覆盖率为`0.9999793`。其normal angular error均值因该
坏例为`6.6412°`，差于guided的`4.2449°`；但中位数`1.1572°`优于guided
的`2.0448°`。

分开的带标题可视化位于：

```text
local_outputs/plane_dust3r_common_domain_17scene_20260724_v1/visualizations/
```

每个视图分别保存RGB、GT、guided RANSAC和Plane-DUSt3R PNG。显示颜色才
使用Hungarian匹配；该匹配不参与适配器预测生成。

验证：

```text
adapter tests 3 passed
batch/paired tests 4 passed
visualization test 1 passed
single-scene真实canary passed
17-scene batch 17/17 passed
```

限制：17场景仍是内部开发集；共同GT是可见Structured3D结构layout，天然更
贴近Plane-DUSt3R的房间布局任务。该表可以支持same-input结构平面对比，但
不能扩张为公开test split成绩或完整bounded editable primitive任务排名。

### 三维评价口径更正

上述统一表是同一有序3D点缓存上的平面支持分区评价，不是二维mask评价；
平面法向和offset也在共同3D点上重新拟合。但最初展示图只是二维投影，而且
当前表尚未评价连续有界面片的面积、三维边界、polygon overlap或平面邻接
拓扑，因此不能称为完整bounded 3D primitive评价。

真正的同相机三维z-buffer分开视图已输出到：

```text
local_outputs/plane_dust3r_common_domain_3d_visuals_20260724_v1/
```

下一步需从预测支持点和Structured3D世界几何构建每个平面的有界patch，在
平面切空间中评价surface IoU/Chamfer和boundary distance，并单独评价法向、
offset以及adjacency graph。

## 2026-07-24 17场景批处理完成

已新增`run_plane_dust3r_same_input_batch.py`及其单元测试。批处理只加载一次
两个模型，逐房间核对必需产物，重新解析每个`metric_results.txt`并分别汇总
room-macro和view-micro指标。它还会核对same-input协议声明、官方commit和
worktree；Git的`safe.directory`只在单次只读命令中指定，不修改用户全局配置。

17场景、85张图的复制型兼容数据位于：

```text
local_outputs/plane_dust3r_same_input_17scene_20260724_v1/
```

官方`convert_plane_depth.py`已为85张图生成`plane_depth.png`。完整批次位于：

```text
local_outputs/plane_dust3r_same_input_batch_17scene_20260724_v1/
status pass
return_code 0
runtime_seconds 1014.6898072
complete_rooms 17/17
complete_views 85/85
```

17房间聚合结果：

```text
layout IoU                         0.7763029385
pixel error                        0.0781947595
edge error                         7.1339249102
depth RMSE                         0.0749965016
scale-aligned RMSE                 0.0641348465
plane precision, room macro mean   0.5070028011
plane recall, room macro mean      0.4237278245
```

`scene_00193`的layout IoU最高，为`0.9636468778`。`scene_00197`是最明显
坏例，layout IoU为`0.3809978764`，plane precision和recall均为`0`。所有
17个房间均生成独立的`layout.png`和五张`seg_pred_vis_*.png`，没有OOM；
运行中观测到RTX 4060 Ti总显存约6.3 GiB、剩余约1.6 GiB。

验证结果：

```text
py_compile: passed
tests/test_run_plane_dust3r_same_input_batch.py: 4 passed
tests/test_run_plane_dust3r_same_input_smoke.py: 11 passed
single-scene batch canary: passed
```

协议限制不变：这是17个内部开发场景的same-input外部模型复现，不是
Plane-DUSt3R论文原生test split。其layout/depth原生指标必须与LightRecon3D
当前point-cache plane F1分表。下一步应实现共同输出域适配器与评测器，再做
可直接比较的外部基线表。

## 2026-07-24 本地RTX 4060 Ti五视图smoke通过

本地官方仓库固定在commit
`9a1ae50650ec6d706bf329352aaaf49efded90a0`，两个checkpoint均已完成
SHA256与PyTorch加载校验。`ai`环境为Python 3.12.11、PyTorch
2.9.1+cu126。由于旧MMCV不能在该环境构建，smoke包装器在隔离runtime中只
提供NonCuboidRoom实际导入的三个MMCV接口，并加入Windows复制回退、
`num_workers=0`、Protobuf子进程隔离、PyTorch安全加载白名单以及官方
`metric.py`六处POSIX路径解析的等价修复。官方仓库本身未修改。

`scene_00180 / room 445895 / positions 0--4`的same-input smoke已完成。
五张图仍然来自冻结的`perspective/empty`输入，只在独立副本中声明为
`full`，并运行官方`convert_plane_depth.py`生成指标所需深度图。因此该
结果不能称为论文原生full协议。

```text
canonical output:
local_outputs/plane_dust3r_same_input_smoke_scene00180_20260724_v7/

status: pass
return code: 0
layout IoU mean: 0.5971863801
pixel error mean: 0.1214576226
edge error mean: 7.7542636513
depth RMSE mean: 0.0366002411
scale-aligned RMSE mean: 0.0366833496
plane precision: 1.0000000000
plane recall: 0.4444444444
```

RTX 4060 Ti上两个模型严格匹配并同时驻留，静态分配显存2318.9 MiB；
五视图DUSt3R推理和全局对齐实际通过且未OOM，运行中一次采样总显存
5421 MiB、剩余2529 MiB。当前结论是8GB显卡可以运行该五视图smoke。

下一步是实现17场景批执行与聚合账本。Plane-DUSt3R原生layout指标必须
单独报告；只有输出域映射经过审计后，才能加入LightRecon3D共同点缓存
上的VOI/RI/SC/pairwise F1表。

## 2026-07-18晚间停止点：明日重新配置

今晚停止时，服务器LightRecon3D仓库已经正确更新到提交`6cf9f82e6c8df83fde011f3c99c0533702698e10`，官方Plane-DUSt3R仓库仍固定在`9a1ae50650ec6d706bf329352aaaf49efded90a0`。现有隔离环境路径为：

```text
/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118-v2
```

该环境中的PyTorch仍为2.2.0/CUDA 11.8，但conda此前选中了MKL 2025.0.0，导致导入PyTorch时报`libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`。提交`6cf9f82`尝试固定`mkl=2024.0`和`intel-openmp=2024.0`；服务器已确认运行到这个新安装分支，但当前配置的channels中没有这两个2024.0包，conda在求解阶段以`PackagesNotFoundError`停止。因此今晚没有完成MKL降级，没有进入MMCV构建、checkpoint下载或GPU smoke。

不要删除现有v2环境，也不要把失败的setup输出目录当成成功记录。`plane_dust3r_external_setup_20260717_v4`和`plane_dust3r_external_setup_20260718_v5`都已被失败运行占用；明天使用新的输出目录。现有`lightrecon`环境没有被这次外部基线配置修改。

明天首先直接安装Anaconda官方归档中确实存在的MKL/Intel OpenMP 2023.1.0包：

```bash
cd /gemini/code/LightRecon3D

ENV_DIR=/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118-v2
CONDA_BIN=/root/miniconda3/bin/conda

"${CONDA_BIN}" install -y --prefix "${ENV_DIR}" \
  "https://repo.anaconda.com/pkgs/main/linux-64/intel-openmp-2023.1.0-hdb19cb5_46305.tar.bz2" \
  "https://repo.anaconda.com/pkgs/main/linux-64/mkl-2023.1.0-h213fc3f_46344.tar.bz2"

"${CONDA_BIN}" list --prefix "${ENV_DIR}" | \
grep -E '^(mkl|intel-openmp|pytorch|torchvision|torchaudio)[[:space:]]'

"${ENV_DIR}/bin/python" - <<'PY'
import torch
assert torch.__version__.split("+")[0] == "2.2.0", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
assert float((x @ x.T)[0, 0]) == 5.0
print("PyTorch verified", torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

以上直接包安装和PyTorch验证命令截至今晚尚未在服务器执行，不能记为已通过。只有看到`PyTorch verified`后，才使用新目录继续setup：

```bash
test "$(git rev-parse HEAD)" = \
"6cf9f82e6c8df83fde011f3c99c0533702698e10" || exit 1

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260718_v6 \
bash prepare_plane_dust3r_external.sh
```

当前提交中的脚本仍写有不可用的`mkl=2024.0`规格，但手工降级后PyTorch可以正常导入时，该安装分支会被跳过。setup必须依次通过PyTorch矩阵验证、MMCV安装、完整环境检查、两个checkpoint可加载检查和SHA256记录，之后才允许运行`scene_00180` smoke。若手工降级后PyTorch仍不能导入，停止执行setup，保留完整conda transaction和`conda list`输出，不继续叠加安装。

该命令通过并输出两个checkpoint的SHA256后，才运行`run_plane_dust3r_same_input_smoke.sh`。

## 2026-07-17 v2运行结果与MMCV构建修复

v2在服务器上正确创建了Python 3.11环境，并保持PyTorch 2.2.0、torchvision 0.17.0和CUDA 11.8，没有再次安装PyTorch 2.13或CUDA 13。清理器从三份官方依赖文件中识别出17条受控依赖，说明第一层约束已经生效。

本次失败发生在安装`mmcv==1.7.2`时。MMCV 1.7.2没有Python 3.11预编译wheel，pip进入源码构建；默认隔离构建环境使用新版setuptools，MMCV的`setup.py`导入`pkg_resources`时失败。该错误发生在pip最终安装事务之前，v2环境中的conda PyTorch仍然正确，可以继续复用，不需要重建或删除。

v3将`setuptools==80.9.0`和`wheel==0.45.1`纳入约束。安装器先完成这两个构建工具的引导并验证`pkg_resources`，然后安装有wheel的兼容依赖，最后只对MMCV使用`--no-build-isolation`。这样MMCV在已验证的构建工具和NumPy 1.26.4环境中构建，不再使用pip临时创建的新版setuptools环境。完成标记升级为`py311v3`，旧失败不会被误认为安装完成。

服务器使用新的输出目录继续运行，环境路径仍为v2且不要手工删除：

```bash
cd /gemini/code/LightRecon3D
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v4 \
bash prepare_plane_dust3r_external.sh
```

## 2026-07-17 v3运行结果与MKL运行时修复

v3成功固定了setuptools、wheel、NumPy 1.26.4、SciPy 1.11.4及其他二进制兼容依赖，`pkg_resources`引导也已通过。MMCV使用非隔离源码构建后，`setup.py`能够继续执行，但在导入PyTorch时出现：

```text
libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

服务器conda计划显示环境中安装了`mkl=2025.0.0`。PyTorch官方问题`pytorch/pytorch#123097`记录了相同错误：PyTorch 2.2与MKL 2024.1及以上组合会在导入时缺少该符号，降级到MKL 2024.0.0可以恢复。这说明v3的MMCV构建修复已经生效，新的阻塞点是conda底层数学库版本，而不是Python包解析。

v4把`mkl=2024.0`和`intel-openmp=2024.0`与PyTorch 2.2、CUDA 11.8放入同一次conda求解。任何pip操作之前，安装器都会真实导入PyTorch并执行一次CPU矩阵乘法；只有版本、CUDA编译版本和张量计算全部通过，才生成依赖清单并继续安装。完整环境检查也重复执行同一张量运算，完成标记升级为`py311v4`。

现有v2环境可以原地降级MKL并复用已经下载的Python wheel，不需要删除：

```bash
cd /gemini/code/LightRecon3D
git -c http.version=HTTP/1.1 pull --ff-only origin codex/bounded-support-head

OUT_DIR=/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v4 \
bash prepare_plane_dust3r_external.sh
```
