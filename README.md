
````markdown
# LightRecon3D

LightRecon3D 是一个基于 **DUSt3R** 的室内 3D 重建研究项目，目标是在无外参场景下，引入 **结构线（line）** 和 **平面（plane）** 的辅助监督与后续几何约束，缓解大面积无纹理区域（如墙面、地板）带来的点云扭曲、墙面不平整、墙角不锐利等问题。

当前项目处于**第一阶段验证**：已经完成基础工程搭建、DUSt3R 接入、多任务预测头设计、基础 loss 实现、train/val 划分、正式训练脚本与小样本 overfit 测试。

---

## 1. Project Goal

本项目的核心目标是：

- 以 **DUSt3R** 作为无外参 3D 重建骨干网络
- 在其 **decoder** 特征上增加两个辅助分支：
  - 结构线预测分支
  - 平面预测分支
- 利用 2D 结构监督，逐步过渡到 3D 空间中的几何约束：
  - **Coplanarity Loss**
  - **Collinearity Loss**
- 最终提升室内场景中墙面、地面、天花板等平面区域的重建质量，使结果更平整、更稳定、更符合曼哈顿世界先验

---

## 2. Current Status

当前已完成：

- [x] Structured3D 数据读取与预处理
- [x] 基于 `layout.json` 动态生成：
  - `gt_line`
  - `gt_plane`
- [x] DUSt3R 预训练权重加载
- [x] 在 DUSt3R `decoder` 最后一层特征上截取 `dec_features`
- [x] 增加 line / plane 多任务预测头
- [x] 实现基础版本：
  - `line_loss`
  - `plane_loss`
  - `total_loss`
- [x] 完成 train / val 按 scene 划分
- [x] 完成 `debug_train_step.py` 单步训练闭环
- [x] 完成 `train.py` 正式训练脚本
- [x] 完成小样本 smoke test 与 overfit 测试

当前尚未完成：

- [ ] 更强的 line loss（如 Focal + Dice）
- [ ] 更合理的 plane supervision 定义
- [ ] 真正的 3D 几何约束（coplanarity / collinearity）
- [ ] 更完整的可视化与评估体系
- [ ] 云端大规模训练实验

---

## 3. Repository Structure

```text
LightRecon3D/
├── dataloaders/
│   ├── s3d_dataset.py              # Structured3D 数据集读取与 mask 生成
│   └── check.py                    # 数据集划分与读取检查脚本
│
├── models/
│   ├── build_backbone.py           # DUSt3R backbone 加载入口
│   └── lightrecon_net.py           # LightReconModel，包含 line / plane heads
│
├── dust3r/                         # 本地接入的 DUSt3R 源码
│
├── output/                         # 实验输出 / 可视化结果（如有）
│
├── losses.py                       # line / plane / total loss
├── debug_train_step.py             # 单步 forward-loss-backward 调试脚本
├── train.py                        # 正式训练入口
├── visualize_predictions.py        # 预测结果可视化脚本
├── test.py                         # 一些临时测试脚本
└── README.md
````

---

## 4. Dataset

当前一期实验数据集使用 **Structured3D**。

选择原因：

* 自带室内结构化场景
* 可从 `layout.json` 中解析出布局多边形
* 适合生成：

  * 结构线 supervision
  * 平面区域 supervision

当前使用的是基于视角渲染的数据，目录结构大致如下：

```text
data/Structured3D/
├── scene_00000
│   ├── 2D_rendering
│   └── annotation_3d.json
├── scene_00001
├── ...
```

在 `2D_rendering/.../perspective/empty/.../` 下，项目会使用：

* `rgb_rawlight.png`
* `layout.json`

来构建训练样本。

### Dataset Output

当前 `Structured3DDataset` 返回：

* `img`：`[3, H, W]`
* `gt_line`：`[1, H, W]`
* `gt_plane`：`[H, W]`

其中：

* `gt_line` 为结构线二值图
* `gt_plane` 为平面实例标签图

---

## 5. Model Design

### Backbone

本项目以 **DUSt3R** 作为主干网络。

当前策略：

* 使用 DUSt3R 预训练权重初始化
* **冻结 encoder**
* 重点微调：

  * decoder
  * line / plane 预测头

### Feature Interception

当前从 DUSt3R **decoder 最后一层输出**中截取特征：

* 形状为 `(B, S, D)`
* 当前实验中 decoder 特征维度 `D = 768`

然后将其重排为：

* `(B, D, H_feat, W_feat)`

再送入轻量卷积头进行预测。

### Prediction Heads

当前包含两个分支：

* **Line Head**

  * 输出：`pred_line`
  * 形状：`[B, 1, 512, 512]`

* **Plane Head**

  * 输出：`pred_plane`
  * 形状：`[B, 20, 512, 512]`

---

## 6. Loss Design

当前 loss 仍为**第一版基础实现**，主要用于验证训练链路能否跑通。

### Line Loss

当前使用基础版：

* BCEWithLogits
* Dice Loss

### Plane Loss

当前使用基础版：

* Cross Entropy Loss

### Total Loss

形式为：

```text
L_total = λ_line * L_line + λ_plane * L_plane
```

> 说明：
> 当前 loss 设计仍然偏基础，目标是先完成第一阶段验证。
> 后续将重点改进：
>
> * line loss（如 Focal + Dice）
> * 以及 2D → 3D 的几何约束 loss

---

## 7. Training

### 7.1 Debug one step

先进行单步调试：

```bash
python debug_train_step.py
```

用于验证：

* 数据是否能读
* 模型前向是否正常
* loss 是否能计算
* backward 是否成功

---

### 7.2 Formal training

正式训练入口：

```bash
python train.py --freeze_encoder --run_val --num_epochs 1 --small_train_size 32 --small_val_size 8 --log_every 1
```

### 常用参数说明

* `--freeze_encoder`
  冻结 DUSt3R encoder，仅训练 decoder 与新增 heads

* `--run_val`
  每个 epoch 运行验证

* `--small_train_size`
  仅取 train 集前 N 个样本，用于 smoke test / overfit test

* `--small_val_size`
  仅取 val 集前 N 个样本

* `--log_every`
  每多少个 batch 打印一次日志

---

### 7.3 Overfit test

4 样本过拟合测试：

```bash
python train.py --freeze_encoder --run_val --num_epochs 30 --small_train_size 4 --small_val_size 4 --log_every 1
```

当前 overfit 实验结果表明：

* plane 分支可在训练集上快速拟合
* line 分支学习明显更困难
* plane 分支存在明显泛化不足现象

---

## 8. Visualization

预测结果可视化：

```bash
python visualize_predictions.py
```

可视化内容包括：

* Input RGB
* GT Line
* Pred Line
* GT Plane
* Pred Plane

当前可视化表明：

* line 分支开始对结构边界出现一定响应
* 但整体仍然较弱
* plane 分支能够学习一定区域分离趋势
* 但当前监督定义仍较粗糙，容易过拟合

---

## 9. Current Problems

当前主要问题包括：

1. **Line branch is hard to learn**

   * 结构线属于极细粒度目标
   * 当前 line 分支学习速度慢，预测偏弱

2. **Plane supervision is still rough**

   * 当前 `gt_plane` 更接近单图内实例编号
   * 不是全局统一语义类别
   * 容易在小样本上快速记忆，泛化较差

3. **Training is slow on local GPU**

   * 当前本地环境没有 CUDA 版 RoPE2D
   * DUSt3R decoder 前向较慢
   * 本地更适合 smoke test，不适合大规模正式训练

---

## 10. Next Step

下一阶段计划：

* 改进 line loss（优先尝试 **Focal + Dice**）
* 继续分析 plane supervision 的合理性
* 将训练迁移到云端更高算力环境
* 在更完整数据规模上观察趋势
* 逐步接入真正的 3D 几何约束：

  * Coplanarity Loss
  * Collinearity Loss

---

## 11. Acknowledgement

* [DUSt3R](https://github.com/naver/dust3r)
* Structured3D Dataset

---

## 12. Note

本仓库目前仍处于研究开发阶段，代码结构与实验设计会持续调整。
当前版本更偏向“第一阶段验证系统”，主要目标是先确认整条技术路线是否可训练、是否具备继续深入研究的价值。

````

---

