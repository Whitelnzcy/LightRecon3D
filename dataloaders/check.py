from s3d_dataset import Structured3DDataset
import matplotlib.pyplot as plt
import torch

# 1. 数据集初始化
root = r'F:\Structured3D_Data' # 数据集根目录路径
scenes = ['scene_00000']
dataset = Structured3DDataset(root, scenes)

print(f"数据集扫描完成，有效样本数: {len(dataset)}")

if len(dataset) > 0:
    # 2. 提取单帧样本数据
    sample = dataset[0]
    
    img = sample['image']       # 图像输入张量: [3, 720, 1280]
    line = sample['gt_line']    # 结构线真值张量: [720, 1280]
    plane = sample['gt_plane']  # 平面实例真值张量: [720, 1280]

    # 输出特征张量维度与统计信息
    print(f"图像输入形状: {img.shape}, 数据类型: {img.dtype}, 像素最大值: {img.max()}")
    print(f"结构线真值形状: {line.shape}, 数据类型: {line.dtype}")
    print(f"平面实例真值形状: {plane.shape}, 数据类型: {plane.dtype}, 有效实例类别数: {len(torch.unique(plane))}")

    # 3. 数据与真值可视化
    plt.figure(figsize=(15, 5))
    
    # 将图像张量由通道优先 [C, H, W] 转换为空间优先 [H, W, C] 以适配可视化库
    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(img.permute(1, 2, 0).numpy())
    
    plt.subplot(1, 3, 2)
    plt.title("GT Line Mask")
    plt.imshow(line.numpy(), cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("GT Plane Mask (Instance)")
    # 采用 jet 伪彩色映射对不同平面实例进行着色区分
    plt.imshow(plane.numpy(), cmap='jet') 
    
    plt.show()