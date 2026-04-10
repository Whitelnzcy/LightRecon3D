import torch
import torch.nn as nn

class LightReconModel(nn.Module):
    def __init__(self, pretrained_dust3r_model, num_plane_classes=20):
        super().__init__()
        
        # 加载预训练的 DUSt3R 模型作为主干网络
        self.backbone = pretrained_dust3r_model
        
        # 设定主干网络输出的隐藏层特征维度
        hidden_dim = 512 
        
        # 结构线预测分支
        # 执行像素级二分类任务，预测各个像素是否属于结构线，输出单通道特征图
        self.line_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        # 平面实例预测分支
        # 执行像素级多分类任务，预测像素所属的平面类别，输出通道数对应设定的平面类别数
        self.plane_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_plane_classes, kernel_size=1)
        )

    def forward(self, images):
        # 1. 主干网络前向传播
        # 提取点云预测结果以及用于下游任务的中间层特征
        pointmaps, features = self.backbone(images)
        
        # 2. 特征映射与分支预测
        # 将深层特征输入至相应的预测分支，获取结构线与平面的对数几率（logits），用于后续损失函数的计算
        pred_line_logits = self.line_head(features)
        pred_plane_logits = self.plane_head(features)
        
        return {
            'pointmaps': pointmaps,
            'pred_lines': pred_line_logits,
            'pred_planes': pred_plane_logits
        }