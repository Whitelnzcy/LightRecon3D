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
