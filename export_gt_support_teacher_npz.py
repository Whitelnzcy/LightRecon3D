import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone


def build_views_from_batch(batch, prefix):
    image1 = batch.get("img1", batch["img"])
    image2 = batch.get("img2", batch["img"])
    batch_size = image1.shape[0]
    shape1 = torch.tensor(image1.shape[-2:], device=image1.device, dtype=torch.long)[None].repeat(
        batch_size, 1
    )
    shape2 = torch.tensor(image2.shape[-2:], device=image2.device, dtype=torch.long)[None].repeat(
        batch_size, 1
    )
    return (
        {
            "img": image1,
            "true_shape": shape1,
            "instance": [f"{prefix}_{index}_view1" for index in range(batch_size)],
        },
        {
            "img": image2,
            "true_shape": shape2,
            "instance": [f"{prefix}_{index}_view2" for index in range(batch_size)],
        },
    )


def resize_label(label, target_hw):
    if label.shape[-2:] == target_hw:
        return label.long()
    return F.interpolate(label.unsqueeze(1).float(), size=target_hw, mode="nearest")[:, 0].long()


def get_pts3d(result):
    for key in ("pts3d", "pts3d_in_other_view", "pointmap", "pred_pts3d"):
        if key in result:
            points = result[key]
            break
    else:
        raise KeyError(f"Cannot find a point map in keys: {list(result.keys())}")
    if points.shape[-1] == 3:
        return points
    if points.shape[1] == 3:
        return points.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Cannot interpret point-map shape {tuple(points.shape)}")


def finite_points(points):
    return torch.isfinite(points).all(dim=-1) & (points.abs().amax(dim=-1) < 1e4)


def plane_boundary(labels):
    valid = labels > 0
    boundary = torch.zeros_like(labels, dtype=torch.float32)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundary[:, 1:][horizontal] = 1.0
    boundary[:, :-1][horizontal] = 1.0
    boundary[1:, :][vertical] = 1.0
    boundary[:-1, :][vertical] = 1.0
    return F.max_pool2d(boundary[None, None], kernel_size=3, stride=1, padding=1)[0, 0]


def fit_plane(points):
    centroid = points.mean(dim=0)
    centered = points - centroid
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / normal.norm().clamp_min(1e-8)
    dominant = torch.argmax(torch.abs(normal))
    if normal[dominant] < 0:
        normal = -normal
    offset = -torch.dot(normal, centroid)
    residual = torch.abs(points @ normal + offset)
    return normal, offset, residual


def image_to_uint8(image, target_hw):
    image = F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)[0]
    if float(image.min()) < -0.05:
        image = (image + 1.0) * 0.5
    return (image.clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).byte()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Export GT bounded-plane support on DUSt3R point maps")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_planes", type=int, default=8)
    parser.add_argument("--min_plane_points", type=int, default=64)
    parser.add_argument("--max_points", type=int, default=24000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode="pair",
    )
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    backbone.eval()

    manifest = []
    end_idx = min(len(dataset), args.start_idx + args.count)
    for sample_idx in range(args.start_idx, end_idx):
        sample = dataset[sample_idx]
        batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items() if torch.is_tensor(value)}
        view1, view2 = build_views_from_batch(batch, prefix=f"gt_support_{sample_idx}")
        result1, _ = backbone(view1, view2)
        point_map = get_pts3d(result1)[0]
        target_hw = point_map.shape[:2]
        labels = resize_label(batch["gt_plane1"], target_hw)[0]
        boundary = plane_boundary(labels)
        colors = image_to_uint8(batch["img1"], target_hw)

        valid = finite_points(point_map)
        candidates = []
        for plane_id in torch.unique(labels):
            plane_id = int(plane_id)
            if plane_id <= 0 or plane_id == 255:
                continue
            count = int(((labels == plane_id) & valid).sum())
            if count >= args.min_plane_points:
                candidates.append((plane_id, count))
        candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = candidates[: args.max_planes]

        flat_points = point_map.reshape(-1, 3)
        flat_colors = colors.reshape(-1, 3)
        flat_labels = labels.reshape(-1)
        flat_boundary = boundary.reshape(-1)
        flat_valid = valid.reshape(-1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, target_hw[0], device=device),
            torch.linspace(-1.0, 1.0, target_hw[1], device=device),
            indexing="ij",
        )
        flat_pixel_xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        valid_indices = torch.nonzero(flat_valid).flatten()
        if args.max_points > 0 and len(valid_indices) > args.max_points:
            sample_positions = torch.linspace(
                0, len(valid_indices) - 1, steps=args.max_points, device=device
            ).long()
            valid_indices = valid_indices[sample_positions]

        sampled_points = flat_points[valid_indices]
        sampled_labels = flat_labels[valid_indices]
        sampled_colors = flat_colors[valid_indices]
        sampled_boundary = flat_boundary[valid_indices]
        sampled_pixel_xy = flat_pixel_xy[valid_indices]

        point_plane_ids = torch.full_like(sampled_labels, -1, dtype=torch.int32)
        normals = []
        offsets = []
        counts = []
        source_ids = []
        fit_rows = []
        for output_id, (source_id, _) in enumerate(candidates):
            full_support = point_map[(labels == source_id) & valid]
            if len(full_support) < args.min_plane_points:
                continue
            normal, offset, residual = fit_plane(full_support)
            normals.append(normal.cpu().numpy().astype(np.float32))
            offsets.append(float(offset))
            point_plane_ids[sampled_labels == source_id] = output_id
            sampled_count = int((sampled_labels == source_id).sum())
            counts.append(sampled_count)
            source_ids.append(source_id)
            fit_rows.append(
                {
                    "plane_id": output_id,
                    "source_gt_plane_id": source_id,
                    "full_support_points": int(len(full_support)),
                    "sampled_support_points": sampled_count,
                    "mean_residual": float(residual.mean()),
                    "p95_residual": float(torch.quantile(residual, 0.95)),
                }
            )
        if not normals:
            print(f"skip sample={sample_idx}: no valid planes")
            continue

        stem = f"{args.split}_{sample_idx:06d}_gt_support_full_pointcloud_editable_planes_data"
        output_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            output_path,
            points=sampled_points.cpu().numpy().astype(np.float32),
            colors=sampled_colors.cpu().numpy().astype(np.uint8),
            original_colors=sampled_colors.cpu().numpy().astype(np.uint8),
            pixel_xy=sampled_pixel_xy.cpu().numpy().astype(np.float32),
            point_plane_ids=point_plane_ids.cpu().numpy().astype(np.int32),
            plane_normals=np.stack(normals).astype(np.float32),
            plane_offsets=np.asarray(offsets, dtype=np.float32),
            plane_inlier_counts=np.asarray(counts, dtype=np.int32),
            plane_ids=np.arange(len(normals), dtype=np.int32),
            source_gt_plane_ids=np.asarray(source_ids, dtype=np.int32),
            line_prob=sampled_boundary.cpu().numpy().astype(np.float32),
        )
        row = {
            "sample_idx": sample_idx,
            "output": str(output_path),
            "points": int(len(sampled_points)),
            "planes": len(normals),
            "background_points": int((point_plane_ids < 0).sum()),
            "boundary_ratio": float(sampled_boundary.mean()),
            "plane_fits": fit_rows,
        }
        manifest.append(row)
        print(json.dumps(row, ensure_ascii=True))

    manifest_path = output_dir / "gt_support_teacher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
