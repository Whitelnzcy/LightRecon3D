import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset


def tensor_img_to_uint8(img):
    img_np = img.detach().float().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np, 0.0, 1.0) * 255.0
    return img_np.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser("Export RGB context images for Structured3D sample indices")
    parser.add_argument("--root_dir", default="/data/zhucy23u/datasets/Structured3D")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--sample_indices", nargs="+", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx in args.sample_indices:
        sample = dataset[idx]
        info = dataset.samples[idx]
        rgb = tensor_img_to_uint8(sample["img"])
        rgb_path = out_dir / f"{args.split}_{idx:06d}_rgb_resized.png"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_path), bgr)
        meta = {
            "split": args.split,
            "sample_idx": int(idx),
            "scene_name": info.get("scene_name"),
            "source_rgb_path": info.get("rgb_path"),
            "source_json_path": info.get("json_path"),
            "exported_rgb": str(rgb_path),
            "image_size": [int(args.image_size), int(args.image_size)],
        }
        meta_path = out_dir / f"{args.split}_{idx:06d}_rgb_context.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(meta)
        print(rgb_path)

    manifest_path = out_dir / f"{args.split}_rgb_context_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
