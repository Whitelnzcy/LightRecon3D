import torch
import torch.nn as nn
import torch.nn.functional as F

class LightReconModel(nn.Module):
    def __init__(self, dust3r_backbone, patch_size=16, hidden_dim=1024, num_planes=20):
        """
        :param dust3r_backbone: 经过结构适配修改的预训练 DUSt3R 骨干网络
        :param patch_size: Vision Transformer 的 Patch 尺寸（DUSt3R 默认值为 16）
        :param hidden_dim: 解码器输出的特征维度（例如 ViT-Large 为 1024，ViT-Base 为 768）
        :param num_planes: 平面实例的最大类别数（包含背景类别 0 及平面类别 1~19）
        """
        super().__init__()
        self.backbone = dust3r_backbone
        self.patch_size = patch_size

        # 结构线预测分支
        # 轻量级 CNN 结构，输出单通道进行二分类（结构线/非结构线）
        self.line_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # 平面实例预测分支
        # 输出 num_planes 个通道进行多分类
        self.plane_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_planes, kernel_size=1)
        )

    def forward(self, view1, view2):
        # 1. 通过骨干网络提取多视图特征
        res1, res2 = self.backbone(view1, view2)

        # 2. 获取视角 1 的解码器深层特征，形状为 (B, S, D)
        features = res1['dec_features']
        B, S, D = features.shape

        # 3. 动态计算特征图的二维空间分辨率 (H_feat, W_feat)
        # 依据输入图像 view1['img'] 的形状 (B, 3, H, W) 进行推导
        H_img, W_img = view1['img'].shape[2:]
        H_feat = H_img // self.patch_size
        W_feat = W_img // self.patch_size

        # 断言检查：确保序列长度与计算出的空间分辨率匹配
        assert S == H_feat * W_feat, f"序列长度 {S} 与计算的空间维度 {H_feat}x{W_feat} 不匹配！"

        # 4. 序列到图像的空间重排：(B, S, D) -> (B, D, H_feat, W_feat)
        # 保证内存连续性并重构维度，以适配卷积层的输入格式
        feat_cnn = features.contiguous().view(B, H_feat, W_feat, D).permute(0, 3, 1, 2)

        # 5. 输入预测分支进行特征映射
        # 输出特征形状分别为 (B, 1, H_feat, W_feat) 与 (B, num_planes, H_feat, W_feat)
        pred_line_lowres = self.line_head(feat_cnn)
        pred_plane_lowres = self.plane_head(feat_cnn)

        # 6. 空间上采样操作
        # 使用双线性插值将特征图恢复至输入图像的原始分辨率，以对齐 Ground Truth 的尺寸
        pred_line = F.interpolate(pred_line_lowres, size=(H_img, W_img), mode='bilinear', align_corners=False)
        pred_plane = F.interpolate(pred_plane_lowres, size=(H_img, W_img), mode='bilinear', align_corners=False)

        # 7. 将预测结果更新至输出字典
        res1['pred_line'] = pred_line
        res1['pred_plane'] = pred_plane

        # 当前仅保留基于视角 1 的预测结果；视角 2 的处理逻辑可同理在此进行扩展
        return res1, res2