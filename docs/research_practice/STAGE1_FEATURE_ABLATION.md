# Stage1特征消融实验

## 实验问题

在相同的Stage1平面预测头和训练预算下，冻结的DUSt3R原生特征是否比直接从RGB像素学习更适合平面支持区域预测。

本实验只比较Stage1输入表示，不修改后续RANSAC、平面拟合或评价规则。

## 对比条件

### DUSt3R特征组

输入为冻结DUSt3R第12层导出的768 x 32 x 32特征。进入查询掩码头前，使用1 x 1卷积投影到256维。

### RGB组

输入为同一张3 x 512 x 512 RGB图像。进入同一个查询掩码头前，使用步长为16的16 x 16卷积生成256 x 32 x 32特征。

两种输入投影均包含768 x 256个权重和256个偏置。投影后的查询嵌入、Transformer decoder、掩码预测、存在性预测和背景预测完全相同，因此两组可训练参数量一致。

RGB组是参数匹配的直接像素基线。实验结论只解释为：在当前数据量、预测头和训练预算下，复用预训练DUSt3R特征是否优于由Stage1头直接从像素学习。该比较不用于否定更大型的专用RGB分割网络。

## 冻结设置

- 训练集：Structured3D训练划分前2048条记录。
- 验证集：W3实验已经冻结的80条配对记录。
- 输入分辨率：512 x 512。
- 查询数：8。
- 隐藏维度：256。
- decoder层数：3。
- 训练轮数：20。
- 优化器：AdamW。
- 学习率：2e-4。
- 权重衰减：1e-4。
- 随机种子：20260616、20260617、20260618。
- checkpoint选择：每个种子只按验证集mean IoU选择最佳epoch。

两组必须使用相同的样本顺序、标签、损失函数、随机种子和评价脚本。不得为单独一组调整学习率、epoch、阈值或损失权重。

## 指标

主指标为验证集support F1。辅助指标包括support precision、support recall、matched mask IoU和plane count absolute error。

所有指标按相同种子和相同样本配对。置信区间以scene为聚类单位进行bootstrap，避免将同一场景中的10条记录当作完全独立样本。

## 判定规则

Strong_DUSt3R_Feature_Advantage必须同时满足：

1. DUSt3R组support F1平均提高至少0.03；
2. support F1的scene-cluster bootstrap 95%区间下界大于0；
3. matched mask IoU平均提高至少0.02；
4. 8个场景中至少6个场景的support F1提高。

若support F1平均提高但未通过全部强条件，判为Weak_or_Inconclusive_DUSt3R_Feature_Advantage。若support F1没有提高，判为No_DUSt3R_Feature_Advantage。

## 执行入口

每个条件和随机种子分别运行run_stage1_feature_ablation.py。六次运行结束后，使用summarize_stage1_feature_ablation.py生成逐记录配对结果、场景聚类bootstrap区间、汇总表和机械判定。
