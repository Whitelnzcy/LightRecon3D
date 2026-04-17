import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Structured3DDataset(Dataset):
    def __init__(self, root_dir, split="train", train_ratio=0.9, image_size=(512, 512)):
        """
        root_dir: Structured3D 根目录，例如 F:\\Structured3D_Data
        split: "train" 或 "val"
        train_ratio: train 占全部 scene 的比例
        image_size: 输出分辨率 (H, W)
        """
        assert split in ["train", "val"], f"split must be 'train' or 'val', got {split}"

        self.root_dir = root_dir
        self.split = split
        self.train_ratio = train_ratio
        self.image_size = image_size  # (H, W)

        # 1) 自动发现全部 scene
        self.all_scenes = self._discover_all_scenes()

        # 2) 按 scene 做 train / val 划分
        self.scenes = self._split_scenes(self.all_scenes, split=self.split, train_ratio=self.train_ratio)

        # 3) 扫描当前 split 下的所有有效样本
        self.samples = self._scan_dataset()

    def _discover_all_scenes(self):
        """
        自动发现 root_dir 下所有 scene_xxxxx 目录。
        """
        all_items = os.listdir(self.root_dir)
        scenes = []
        

        for name in all_items:
            full_path = os.path.join(self.root_dir, name)
            if os.path.isdir(full_path) and name.startswith("scene_"):
                scenes.append(name)

        scenes = sorted(scenes)

        if len(scenes) == 0:
            raise RuntimeError(f"在 {self.root_dir} 下没有找到任何 scene_* 目录")

        return scenes

    def _split_scenes(self, all_scenes, split="train", train_ratio=0.9):
        """
        按 scene 划分 train / val，避免同一场景的数据泄漏。
        """
        num_scenes = len(all_scenes)
        split_idx = int(num_scenes * train_ratio)

        # 防止极端情况下某个 split 为空
        split_idx = max(1, min(split_idx, num_scenes - 1)) if num_scenes > 1 else num_scenes

        if split == "train":
            scenes = all_scenes[:split_idx]
        else:
            scenes = all_scenes[split_idx:]

        if len(scenes) == 0:
            raise RuntimeError(f"{split} split 为空，请检查 train_ratio={train_ratio} 和 scene 数量")

        return scenes

    def _scan_dataset(self):
        """
        遍历当前 split 中的 scene，收集 layout.json 与 rgb_rawlight.png 的有效配对。
        """
        samples = []

        for scene in self.scenes:
            scene_dir = os.path.join(self.root_dir, scene, "2D_rendering")
            if not os.path.exists(scene_dir):
                continue

            search_pattern = os.path.join(
                scene_dir, "*", "perspective", "empty", "*", "layout.json"
            )
            json_files = glob.glob(search_pattern)

            for json_path in json_files:
                rgb_path = os.path.join(os.path.dirname(json_path), "rgb_rawlight.png")

                if os.path.exists(rgb_path):
                    samples.append({
                        "scene_name": scene,
                        "json_path": json_path,
                        "rgb_path": rgb_path
                    })

        if len(samples) == 0:
            raise RuntimeError(f"{self.split} split 没有扫描到有效样本，请检查数据目录结构")

        return samples

    def generate_masks(self, json_data, height=720, width=1280):
        """
        先在原始分辨率下渲染，再在 __getitem__ 中统一 resize。
        """
        plane_mask = np.zeros((height, width), dtype=np.int32)
        line_mask = np.zeros((height, width), dtype=np.uint8)

        all_junctions = json_data["junctions"]

        for i, plane in enumerate(json_data.get("planes", [])):
            for mask_indices in plane.get("visible_mask", []):
                pts = [
                    all_junctions[idx]["coordinate"]
                    for idx in mask_indices
                    if idx < len(all_junctions)
                ]

                if len(pts) < 3:
                    continue

                poly_points = np.array(pts, dtype=np.int32)

                cv2.fillPoly(plane_mask, [poly_points], color=int(i + 1))
                cv2.polylines(line_mask, [poly_points], isClosed=True, color=255, thickness=2)

        return plane_mask, line_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        image_bgr = cv2.imread(sample_info["rgb_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with open(sample_info["json_path"], "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        plane_mask, line_mask = self.generate_masks(layout_data)

        target_h, target_w = self.image_size

        # 图像用双线性
        image_rgb = cv2.resize(
            image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # mask 用最近邻，避免标签污染
        line_mask = cv2.resize(
            line_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        plane_mask = cv2.resize(
            plane_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

        sample = {
            "img": torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0,   # [3,H,W]
            "gt_line": torch.from_numpy(line_mask).unsqueeze(0).float() / 255.0,   # [1,H,W]
            "gt_plane": torch.from_numpy(plane_mask).long(),                        # [H,W]
        }

        return sample
    
