import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Structured3DDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        train_ratio=0.9,
        image_size=(512, 512),
        input_mode="pair",
        pair_strategy="adjacent",
        pair_max_view_id_gap=None,
    ):
        """
        Structured3D dataset for LightRecon3D.

        input_mode:
            "pair" returns real two-view samples for the DUSt3R interface.
            "single" keeps the previous single-view behavior.

        pair_strategy:
            "adjacent" pairs neighboring renders in the same Structured3D space.
            "all" returns all unordered pairs inside each space.

        pair_max_view_id_gap:
            Optional numeric render-id gap filter. Structured3D does not store
            camera poses in layout.json, so this is only a lightweight guard
            based on the render directory name.
        """
        assert split in ["train", "val"], f"split must be 'train' or 'val', got {split}"
        assert input_mode in ["single", "pair"], (
            f"input_mode must be 'single' or 'pair', got {input_mode}"
        )
        assert pair_strategy in ["adjacent", "all"], (
            f"pair_strategy must be 'adjacent' or 'all', got {pair_strategy}"
        )

        self.root_dir = root_dir
        self.split = split
        self.train_ratio = train_ratio
        self.image_size = image_size
        self.input_mode = input_mode
        self.pair_strategy = pair_strategy
        self.pair_max_view_id_gap = pair_max_view_id_gap

        self.all_scenes = self._discover_all_scenes()
        self.scenes = self._split_scenes(
            self.all_scenes,
            split=self.split,
            train_ratio=self.train_ratio,
        )
        self.samples = self._scan_dataset()

    def _discover_all_scenes(self):
        scenes = []

        for name in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, name)
            if os.path.isdir(full_path) and name.startswith("scene_"):
                scenes.append(name)

        scenes = sorted(scenes)

        if len(scenes) == 0:
            raise RuntimeError(f"No scene_* directories found under {self.root_dir}")

        return scenes

    def _split_scenes(self, all_scenes, split="train", train_ratio=0.9):
        num_scenes = len(all_scenes)
        split_idx = int(num_scenes * train_ratio)
        split_idx = max(1, min(split_idx, num_scenes - 1)) if num_scenes > 1 else num_scenes

        if split == "train":
            scenes = all_scenes[:split_idx]
        else:
            scenes = all_scenes[split_idx:]

        if len(scenes) == 0:
            raise RuntimeError(
                f"{split} split is empty. Check train_ratio={train_ratio} "
                f"and number of scenes={num_scenes}."
            )

        return scenes

    def _scan_dataset(self):
        single_samples = []

        for scene in self.scenes:
            scene_dir = os.path.join(self.root_dir, scene, "2D_rendering")
            if not os.path.exists(scene_dir):
                continue

            search_pattern = os.path.join(
                scene_dir,
                "*",
                "perspective",
                "empty",
                "*",
                "layout.json",
            )

            for json_path in glob.glob(search_pattern):
                rgb_path = os.path.join(os.path.dirname(json_path), "rgb_rawlight.png")

                if os.path.exists(rgb_path):
                    single_samples.append({
                        "scene_name": scene,
                        "json_path": json_path,
                        "rgb_path": rgb_path,
                        "pair_group": self._get_pair_group(json_path),
                        "view_id": self._get_view_id(json_path),
                    })

        if len(single_samples) == 0:
            raise RuntimeError(
                f"{self.split} split has no valid samples. "
                "Check the Structured3D directory layout."
            )

        if self.input_mode == "single":
            return single_samples

        pair_samples = self._build_pair_samples(single_samples)

        if len(pair_samples) == 0:
            raise RuntimeError(
                f"{self.split} split has no valid two-view pairs. "
                "Each pair group needs at least two perspective/empty renders. "
                "Use input_mode='single' to keep the old pseudo-pair path."
            )

        return pair_samples

    def _get_pair_group(self, json_path):
        """
        Group views belonging to one rendered Structured3D space.

        Expected path:
            scene_xxxxx/2D_rendering/<space>/perspective/empty/<view>/layout.json
        """
        view_dir = os.path.dirname(json_path)
        empty_dir = os.path.dirname(view_dir)
        return empty_dir

    def _get_view_id(self, json_path):
        view_dir = os.path.basename(os.path.dirname(json_path))

        try:
            return int(view_dir)
        except ValueError:
            return view_dir

    def _pair_sort_key(self, sample):
        view_id = sample["view_id"]
        if isinstance(view_id, int):
            return (0, view_id)
        return (1, str(view_id), sample["rgb_path"], sample["json_path"])

    def _passes_view_id_gap(self, sample1, sample2):
        if self.pair_max_view_id_gap is None:
            return True

        if self.pair_max_view_id_gap <= 0:
            return True

        view_id1 = sample1["view_id"]
        view_id2 = sample2["view_id"]

        if not isinstance(view_id1, int) or not isinstance(view_id2, int):
            return True

        return abs(view_id2 - view_id1) <= self.pair_max_view_id_gap

    def _build_pair_samples(self, single_samples):
        groups = {}

        for sample in single_samples:
            groups.setdefault(sample["pair_group"], []).append(sample)

        pair_samples = []

        for _, group_samples in sorted(groups.items()):
            group_samples = sorted(
                group_samples,
                key=self._pair_sort_key,
            )

            if len(group_samples) < 2:
                continue

            if self.pair_strategy == "adjacent":
                index_pairs = [(i, i + 1) for i in range(len(group_samples) - 1)]
            else:
                index_pairs = [
                    (i, j)
                    for i in range(len(group_samples))
                    for j in range(i + 1, len(group_samples))
                ]

            for i, j in index_pairs:
                sample1 = group_samples[i]
                sample2 = group_samples[j]

                if not self._passes_view_id_gap(sample1, sample2):
                    continue

                pair_samples.append({
                    "scene_name": sample1["scene_name"],
                    "pair_group": sample1["pair_group"],
                    "view1": sample1,
                    "view2": sample2,
                })

        return pair_samples

    def format_pair_sample(self, idx):
        sample = self.samples[idx]

        if self.input_mode == "single":
            return (
                f"[{idx}] single scene={sample['scene_name']} "
                f"view_id={sample['view_id']} rgb={sample['rgb_path']}"
            )

        view1 = sample["view1"]
        view2 = sample["view2"]
        return (
            f"[{idx}] scene={sample['scene_name']} "
            f"group={sample['pair_group']} "
            f"view1={view1['view_id']} view2={view2['view_id']}\n"
            f"    rgb_path1={view1['rgb_path']}\n"
            f"    rgb_path2={view2['rgb_path']}"
        )

    def generate_masks(self, json_data, height=720, width=1280):
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
                cv2.polylines(
                    line_mask,
                    [poly_points],
                    isClosed=True,
                    color=255,
                    thickness=2,
                )

        return plane_mask, line_mask

    def __len__(self):
        return len(self.samples)

    def _load_single_view(self, sample_info):
        image_bgr = cv2.imread(sample_info["rgb_path"])
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {sample_info['rgb_path']}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        with open(sample_info["json_path"], "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        plane_mask, line_mask = self.generate_masks(layout_data)

        target_h, target_w = self.image_size

        image_rgb = cv2.resize(
            image_rgb,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        line_mask = cv2.resize(
            line_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        plane_mask = cv2.resize(
            plane_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        return {
            "img": torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0,
            "gt_line": torch.from_numpy(line_mask).unsqueeze(0).float() / 255.0,
            "gt_plane": torch.from_numpy(plane_mask).long(),
            "original_hw": torch.tensor([original_h, original_w], dtype=torch.int32),
            "resized_hw": torch.tensor([target_h, target_w], dtype=torch.int32),
        }

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        if self.input_mode == "single":
            sample = self._load_single_view(sample_info)
            sample["rgb_path"] = sample_info["rgb_path"]
            sample["json_path"] = sample_info["json_path"]
            sample["scene_name"] = sample_info["scene_name"]
            sample["pair_group"] = sample_info["pair_group"]
            sample["view_id"] = sample_info["view_id"]
            return sample

        view1_info = sample_info["view1"]
        view2_info = sample_info["view2"]

        view1 = self._load_single_view(view1_info)
        view2 = self._load_single_view(view2_info)

        return {
            "img1": view1["img"],
            "img2": view2["img"],
            "gt_line1": view1["gt_line"],
            "gt_line2": view2["gt_line"],
            "gt_plane1": view1["gt_plane"],
            "gt_plane2": view2["gt_plane"],

            # Backward-compatible aliases for scripts that still inspect view1.
            "img": view1["img"],
            "gt_line": view1["gt_line"],
            "gt_plane": view1["gt_plane"],

            "rgb_path1": view1_info["rgb_path"],
            "rgb_path2": view2_info["rgb_path"],
            "json_path1": view1_info["json_path"],
            "json_path2": view2_info["json_path"],
            "view_id1": view1_info["view_id"],
            "view_id2": view2_info["view_id"],
            "original_hw1": view1["original_hw"],
            "original_hw2": view2["original_hw"],
            "stage1_input_hw1": view1["resized_hw"],
            "stage1_input_hw2": view2["resized_hw"],
            "scene_name": sample_info["scene_name"],
            "pair_group": sample_info["pair_group"],
        }

