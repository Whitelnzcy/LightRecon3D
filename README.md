# LightRecon3D

LightRecon3D 是一个面向室内场景的三维重建研究项目。项目基于 **DUSt3R** 搭建，当前主要关注一个比较具体的问题：在墙面、地面这类大面积弱纹理区域里，重建结果往往容易出现形变、起伏和边界不清晰。为了解决这个问题，本项目尝试在 DUSt3R 的重建框架中引入 **结构线** 和 **平面区域** 的辅助监督，希望逐步把这些 2D 结构信息转化为更稳定的 3D 几何约束。

目前项目仍然处在第一阶段，重点不是追求最终效果，而是先把整条训练链路搭起来，验证这条路线本身是否可行。

## 项目思路

当前的基本想法是：

- 以 **DUSt3R** 作为无外参重建骨干
- 截取其 **decoder** 的高层特征
- 在此基础上增加两个辅助分支：
  - 结构线预测分支
  - 平面预测分支
- 先利用 2D 监督让模型具备一定的结构感知能力
- 后续再进一步考虑引入更明确的 3D 几何约束，例如共面约束和共线约束

项目的最终目标不是做一个单独的 2D 分割网络，而是希望借助这些结构先验，让室内场景中的墙面更平整、墙角更锐利、整体重建更稳定。

## 当前进展

目前已经完成的内容包括：

- 接入 Structured3D 数据集
- 基于 `layout.json` 生成：
  - `gt_line`
  - `gt_plane`
- 加载 DUSt3R 预训练权重
- 从 DUSt3R 的 decoder 最后一层截取特征
- 增加 line / plane 两个预测头
- 实现基础版本的多任务 loss
- 完成 train / val 按 scene 划分
- 搭建 `debug_train_step.py` 与 `train.py`
- 跑通 forward、loss、backward 与参数更新
- 完成小样本 smoke test 与 overfit 测试

当前的小样本实验结果表明：

- 整体训练链路已经是可运行的
- plane 分支在小样本上较容易拟合训练集
- line 分支明显更难学
- 当前 plane 的监督定义还比较粗糙，存在过拟合和泛化差的问题

## 数据集

当前一期验证数据集使用 **Structured3D**。

选择它的原因比较直接：它本身是室内结构化场景数据集，能够从 `layout.json` 中解析出布局多边形，因此比较适合作为第一阶段的验证数据，用来生成：

- 结构线监督
- 平面区域监督

当前使用的数据目录结构大致如下：

```text
data/Structured3D/
├── scene_00000
├── scene_00001
├── scene_00002
└── ...
```

在每个 scene 下，项目主要使用：

- `rgb_rawlight.png`
- `layout.json`

来构建训练样本。

当前 `Structured3DDataset` 返回：

- `img`：输入图像
- `gt_line`：结构线真值
- `gt_plane`：平面实例真值

需要说明的是，当前的 `gt_plane` 更接近单图内实例编号，还不是特别理想的全局统一平面监督，这也是后续需要继续改进的地方。

## 模型结构

当前模型主要分成两部分。

### DUSt3R backbone

项目使用 DUSt3R 作为主干网络，当前策略是：

- 冻结 encoder
- 重点微调 decoder
- 同时训练新增的 line / plane prediction heads

### 辅助预测头

在 DUSt3R decoder 最后一层特征的基础上，增加两个轻量卷积头：

- `pred_line`
- `pred_plane`

当前做法是先将 `(B, S, D)` 的 token 特征重排为 `(B, D, H, W)`，再通过卷积头输出预测结果，最后上采样回输入图像大小。

## 训练方式

### 单步调试

用于检查整个链路是否打通：

```bash
python debug_train_step.py
```

### 小规模训练

当前正式训练入口：

```bash
python train.py --freeze_encoder --run_val --num_epochs 1 --small_train_size 32 --small_val_size 8 --log_every 1
```

### 4 样本 overfit 测试

用于检查当前模型和 loss 是否具备基本学习能力：

```bash
python train.py --freeze_encoder --run_val --num_epochs 30 --small_train_size 4 --small_val_size 4 --log_every 1
```

## 可视化

项目提供了简单的可视化脚本，用于查看：

- 输入图像
- GT Line / Pred Line
- GT Plane / Pred Plane

运行方式：

```bash
python visualize_predictions.py
```

这部分主要用于快速判断模型到底有没有在学，而不是只看 loss 数字。

## 当前存在的问题

目前比较明确的几个问题是：

### line 分支学习较困难

结构线本身是细粒度目标，而当前 backbone 的原生特征分辨率较低，因此 line 分支的学习明显比 plane 更难。

### plane supervision 仍然比较粗糙

目前使用的是单图内实例编号式标签，这种监督在小样本上容易记忆，但泛化能力不强。

### 本地训练速度较慢

当前本地环境缺少 CUDA 编译版 RoPE2D，DUSt3R decoder 前向较慢，因此目前更适合做小规模开发验证，不适合长时间正式训练。

## 接下来准备做什么

后面的工作会主要围绕这几件事继续推进：

- 改进 line 分支的 loss
- 在云端算力环境上跑更完整的数据规模
- 继续观察 line / plane 分支的学习情况
- 逐步引入更明确的几何约束
- 进一步评估当前 plane supervision 是否需要重新设计

## 仓库结构

```text
LightRecon3D/
├── dataloaders/
│   ├── s3d_dataset.py
│   └── check.py
├── models/
│   ├── build_backbone.py
│   └── lightrecon_net.py
├── dust3r/
├── losses.py
├── debug_train_step.py
├── train.py
├── visualize_predictions.py
└── README.md
```

## 说明

这个仓库目前仍处于持续开发中。当前版本更偏向“第一阶段验证系统”，主要目的是确认这条结构感知增强重建的路线是否真正可训练、是否值得继续往下做。
