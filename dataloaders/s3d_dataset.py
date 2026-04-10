import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class Structured3DDataset(Dataset):
    def __init__(self, root_dir, scene_list, transform=None):
        self.root_dir = root_dir
        self.scenes = scene_list
        # 在类初始化阶段执行数据扫描，构建样本索引列表
        self.samples = self._scan_dataset()

    def _scan_dataset(self):
        """
        遍历指定的场景目录，检索并构建 RGB 图像与相应 JSON 标注文件的配对路径列表。
        """
        samples = []
        for scene in self.scenes:
            scene_dir = os.path.join(self.root_dir, scene, "2D_rendering")
            if not os.path.exists(scene_dir):
                continue
            
            # 递归检索所有符合目标的布局标注文件
            # 路径匹配范式: scene_xxxxx/2D_rendering/<roomID>/perspective/empty/<posID>/layout.json
            search_pattern = os.path.join(scene_dir, "*", "perspective", "empty", "*", "layout.json")
            json_files = glob.glob(search_pattern)
            
            for json_path in json_files:
                # 依据标注文件路径，推演对应渲染图像的物理路径
                rgb_path = os.path.join(os.path.dirname(json_path), "rgb_rawlight.png")
                
                # 验证图像与标注文件的完整性，仅保留有效数据对
                if os.path.exists(rgb_path):
                    samples.append({
                        'json_path': json_path,
                        'rgb_path': rgb_path
                    })
        return samples

    def generate_masks(self, json_data, height=720, width=1280):
        """
        解析基于 JSON 的空间布局几何数据，渲染生成平面实例掩码及结构线二值掩码。
        """
        plane_mask = np.zeros((height, width), dtype=np.uint8)
        line_mask = np.zeros((height, width), dtype=np.uint8)
        
        all_junctions = json_data['junctions']
        
        for i, plane in enumerate(json_data.get('planes', [])):
            for mask_indices in plane.get('visible_mask', []):
                # 根据 Structured3D 的标注规范，基于索引提取交点空间坐标
                pts = [all_junctions[idx]['coordinate'] for idx in mask_indices if idx < len(all_junctions)]
                
                if len(pts) < 3:
                    continue
                    
                poly_points = np.array(pts, np.int32)
                
                cv2.fillPoly(plane_mask, [poly_points], color=i + 1)
                cv2.polylines(line_mask, [poly_points], isClosed=True, color=255, thickness=2)
                
        return plane_mask, line_mask

    def __len__(self):
        # 返回当前数据集所包含的有效样本总数
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 提取当前索引对应的文件路径字典
        sample_info = self.samples[idx]
        
        # 2. 读取图像序列并进行色彩空间对齐
        image_bgr = cv2.imread(sample_info['rgb_path'])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. 反序列化布局标注数据
        with open(sample_info['json_path'], 'r') as f:
            layout_data = json.load(f)
            
        # 4. 根据几何坐标生成密集预测任务所需的 Ground Truth 掩码
        plane_mask, line_mask = self.generate_masks(layout_data)

        # 5. 张量化处理与数据归一化
        # 将图像特征张量的维度由空间优先调整为通道优先 (C, H, W)
        sample = {
            'image': torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0,
            'gt_line': torch.from_numpy(line_mask).float() / 255.0,
            'gt_plane': torch.from_numpy(plane_mask).long()
        }
        
        return sample