import torch
import torch.nn as nn
# 导入已定义的网络模型
from models.lightrecon_net import LightReconModel

# 1. 构建 DUSt3R 主干网络的模拟模块 (Mock)
# 用于在未加载大规模预训练权重的情况下，独立测试网络的前向传播与特征维度连通性
class MockDUSt3R(nn.Module):
    def forward(self, view1, view2):
        # 获取输入图像的批量大小与空间分辨率
        B = view1['img'].shape[0]
        H, W = view1['img'].shape[2:]
        
        # 依据 Patch size = 16，计算展平后的序列长度 S
        S = (H // 16) * (W // 16)
        # 设定隐层特征维度 D 为 1024
        D = 1024 
        
        # 构造模拟的三维点云输出与解码器深层特征
        res1 = {'pts3d': torch.randn(B, H, W, 3), 'dec_features': torch.randn(B, S, D)}
        res2 = {'pts3d': torch.randn(B, H, W, 3), 'dec_features': torch.randn(B, S, D)}
        return res1, res2

if __name__ == "__main__":
    print("=== 网络特征维度连通性测试开始 ===")
    
    # 1. 构造伪数据以模拟 DataLoader 的输入张量
    # 设定 Batch Size = 2, 通道数 = 3, 图像空间分辨率 = 512x512
    dummy_view1 = {'img': torch.randn(2, 3, 512, 512)}
    dummy_view2 = {'img': torch.randn(2, 3, 512, 512)}

    # 2. 初始化联合网络模型
    mock_backbone = MockDUSt3R()
    # 设定 hidden_dim=1024，需与 Mock 模块输出的特征维度 D 保持严格一致
    model = LightReconModel(mock_backbone, hidden_dim=1024, num_planes=20)
    
    # 3. 执行前向传播测试
    res1, res2 = model(dummy_view1, dummy_view2)
    
    # 4. 输出各层级张量维度信息，验证形状对齐情况
    print("模型前向传播成功，未发生维度异常。")
    print(f"原始图像维度: {dummy_view1['img'].shape}")
    print(f"解码器深层特征维度: {res1['dec_features'].shape}  --> (B, S, D)")
    print(f"结构线预测输出维度: {res1['pred_line'].shape}  --> (B, 1, H, W)")
    print(f"平面实例预测输出维度: {res1['pred_plane'].shape}  --> (B, 20, H, W)")