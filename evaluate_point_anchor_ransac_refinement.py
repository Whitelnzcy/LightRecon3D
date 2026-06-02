import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if DUST3R_REPO_ROOT in sys.path:
    sys.path.remove(DUST3R_REPO_ROOT)
sys.path.insert(1, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from evaluate_plane_params_head import geometry_descriptor
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def resize_label(gt_plane, target_hw):
    if gt_plane.shape[-2:] == target_hw:
        return gt_plane.long()
    return F.interpolate(gt_plane.unsqueeze(1).float(), size=target_hw, mode="nearest")[:, 0].long()


def resize_map(x, target_hw):
    if x.shape[-2:] == target_hw:
        return x
    return F.interpolate(x, size=target_hw, mode="nearest")


def angle_deg(pred_n, gt_n):
    v = torch.abs((pred_n * gt_n).sum()).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(v))


def get_pts3d_from_res(res):
    for key in ["pts3d", "pts3d_in_other_view", "pointmap", "pred_pts3d"]:
        if key in res:
            pts = res[key]
            break
    else:
        raise KeyError(f"Cannot find pointmap keys in {list(res.keys())}")
    if pts.shape[-1] == 3:
        return pts
    if pts.shape[1] == 3:
        return pts.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Cannot interpret pointmap shape {pts.shape}")


def clean_points(points):
    finite = torch.isfinite(points).all(dim=1)
    points = points[finite]
    coord_ok = points.abs().amax(dim=1) < 1e4
    return points[coord_ok]


def fit_plane_svd(points, eps=1e-6):
    points = clean_points(points)
    if points.shape[0] < 3:
        return None
    centroid = points.mean(dim=0)
    x = points - centroid
    cov = x.transpose(0, 1) @ x / max(1, points.shape[0] - 1)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov.float())
    normal = normalize(eigvecs[:, 0].to(points.dtype), dim=0)
    offset = -torch.dot(normal, centroid)
    return normal, offset


def fit_plane_3pts(points):
    a, b, c = points[0], points[1], points[2]
    normal = torch.cross(b - a, c - a, dim=0)
    if normal.norm() < 1e-6:
        return None
    normal = normalize(normal, dim=0)
    offset = -torch.dot(normal, a)
    return normal, offset


def align_to_ref(normal, offset, ref_normal):
    sign = torch.where((normal * ref_normal).sum() < 0, -1.0, 1.0)
    return normal * sign, offset * sign


def subsample_points(points, max_points, seed):
    points = clean_points(points)
    if points.shape[0] <= max_points:
        return points
    g = torch.Generator(device=points.device)
    g.manual_seed(seed)
    idx = torch.randperm(points.shape[0], generator=g, device=points.device)[:max_points]
    return points[idx]


def ransac_refine(
    points,
    init_normal,
    init_offset,
    threshold=0.025,
    iterations=128,
    max_points=4096,
    init_pool_quantile=0.85,
    seed=0,
):
    points = subsample_points(points, max_points=max_points, seed=seed)
    if points.shape[0] < 16:
        return init_normal, init_offset, 0, float("inf")

    with torch.no_grad():
        init_dist = torch.abs(points @ init_normal + init_offset)
        q = torch.quantile(init_dist.float(), init_pool_quantile).to(init_dist.dtype)
        pool = points[init_dist <= q]
        if pool.shape[0] < 16:
            pool = points

        g = torch.Generator(device=points.device)
        g.manual_seed(seed)
        best_score = None
        best_n, best_d = init_normal, init_offset
        best_inliers = init_dist <= threshold

        for _ in range(iterations):
            idx = torch.randint(0, pool.shape[0], (3,), generator=g, device=points.device)
            fit = fit_plane_3pts(pool[idx])
            if fit is None:
                continue
            cand_n, cand_d = align_to_ref(fit[0], fit[1], init_normal)
            dist = torch.abs(points @ cand_n + cand_d)
            inliers = dist <= threshold
            count = int(inliers.sum().item())
            if count < 8:
                continue
            mean_dist = float(dist[inliers].mean().item())
            normal_prior = float(torch.abs((cand_n * init_normal).sum()).clamp(0, 1).item())
            score = (count, normal_prior, -mean_dist)
            if best_score is None or score > best_score:
                best_score = score
                best_n, best_d = cand_n, cand_d
                best_inliers = inliers

        if int(best_inliers.sum().item()) >= 8:
            fit = fit_plane_svd(points[best_inliers])
            if fit is not None:
                best_n, best_d = align_to_ref(fit[0], fit[1], init_normal)

        final_dist = torch.abs(points @ best_n + best_d)
        final_inliers = final_dist <= threshold
        mean_dist = float(final_dist[final_inliers].mean().item()) if int(final_inliers.sum()) > 0 else float("inf")
        return best_n, best_d, int(final_inliers.sum().item()), mean_dist


def predict_plane_param(model, token, geom, plane_points, head_type):
    if head_type == "geom_token_point_anchor_conf":
        pred = model.geom_plane_token_conf_head(torch.cat([token, geom], dim=1))[0]
        pred_n = normalize(pred[:3], dim=0)
        pred_d = -torch.dot(pred_n, plane_points.mean(dim=0)) + pred[3]
        confidence = torch.sigmoid(pred[4])
        return pred_n, pred_d, confidence
    if head_type == "geom_token_point_anchor":
        pred = model.geom_plane_token_param_head(torch.cat([token, geom], dim=1))[0]
        pred_n = normalize(pred[:3], dim=0)
        pred_d = -torch.dot(pred_n, plane_points.mean(dim=0)) + pred[3]
        return pred_n, pred_d, None
    raise ValueError(f"Unsupported head type: {head_type}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--plane_offset_scale", type=float, default=1000.0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--param_head_type", default="geom_token_point_anchor_conf", choices=["geom_token_point_anchor", "geom_token_point_anchor_conf"])
    parser.add_argument("--min_pixels", type=int, default=64)
    parser.add_argument("--ransac_threshold", type=float, default=0.025)
    parser.add_argument("--ransac_iterations", type=int, default=128)
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        plane_offset_scale=args.plane_offset_scale,
    )
    if args.num_samples > 0:
        dataset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)
    ckpt = safe_load(args.ckpt_path, device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    rows = []
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
        view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{sample_idx}")
        res1, _ = model(view1, view2)

        feature_map = res1["dec_feature_map"][0]
        target_hw = feature_map.shape[-2:]
        feature_hw = feature_map.permute(1, 2, 0).contiguous()
        pts3d = get_pts3d_from_res(res1)[0]
        gt_plane_pts = resize_label(batch["gt_plane"], pts3d.shape[:2])[0]

        gt_plane = resize_label(batch["gt_plane"], target_hw)[0]
        gt_normal = resize_map(batch["gt_plane_normal"], target_hw)[0].permute(1, 2, 0)
        valid = resize_map(batch["gt_plane_param_valid"], target_hw)[0, 0] > 0.5

        for pid in torch.unique(gt_plane):
            pid_int = int(pid.item())
            if pid_int in (-1, 0, 255):
                continue
            mask = (gt_plane == pid) & valid
            if int(mask.sum().item()) < args.min_pixels:
                continue

            plane_points = clean_points(pts3d[gt_plane_pts == pid])
            if plane_points.shape[0] < 16:
                continue

            token = feature_hw[mask].mean(dim=0, keepdim=True)
            geom = geometry_descriptor(plane_points).to(dtype=token.dtype).view(1, -1)
            pred_n, pred_d, confidence = predict_plane_param(
                model=model,
                token=token,
                geom=geom,
                plane_points=plane_points,
                head_type=args.param_head_type,
            )

            gt_n = normalize(gt_normal[mask].mean(dim=0), dim=0)
            gt_d = -torch.dot(gt_n, plane_points.mean(dim=0))
            pred_n, pred_d = align_to_ref(pred_n, pred_d, gt_n)

            svd_fit = fit_plane_svd(plane_points)
            if svd_fit is None:
                svd_n, svd_d = pred_n, pred_d
            else:
                svd_n, svd_d = align_to_ref(svd_fit[0], svd_fit[1], gt_n)

            ref_n, ref_d, inlier_count, mean_inlier_dist = ransac_refine(
                plane_points,
                pred_n,
                pred_d,
                threshold=args.ransac_threshold,
                iterations=args.ransac_iterations,
                max_points=args.max_points,
                seed=args.seed + sample_idx * 1000 + pid_int,
            )
            ref_n, ref_d = align_to_ref(ref_n, ref_d, gt_n)

            candidates = {
                "pred": (pred_n, pred_d, clean_points(plane_points).shape[0], float("nan")),
                "svd_all": (svd_n, svd_d, clean_points(plane_points).shape[0], float("nan")),
                "ransac_refine": (ref_n, ref_d, inlier_count, mean_inlier_dist),
            }
            for method, (n, d, inliers, mean_dist) in candidates.items():
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "plane_id": pid_int,
                        "method": method,
                        "pixels": int(mask.sum().item()),
                        "points": int(plane_points.shape[0]),
                        "inliers": int(inliers),
                        "mean_inlier_dist": mean_dist,
                        "angle_deg": float(angle_deg(n, gt_n).cpu().item()),
                        "offset_abs_error": float(torch.abs(d - gt_d).cpu().item()),
                        "pred_confidence": float(confidence.cpu().item()) if confidence is not None else "",
                        "nx": float(n[0].cpu().item()),
                        "ny": float(n[1].cpu().item()),
                        "nz": float(n[2].cpu().item()),
                        "offset": float(d.cpu().item()),
                        "gt_nx": float(gt_n[0].cpu().item()),
                        "gt_ny": float(gt_n[1].cpu().item()),
                        "gt_nz": float(gt_n[2].cpu().item()),
                        "gt_offset": float(gt_d.cpu().item()),
                    }
                )

        if (sample_idx + 1) % 16 == 0:
            print(f"evaluated {sample_idx + 1}/{len(dataset)} samples")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {out_path}")
    methods = sorted({r["method"] for r in rows})
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        mean_angle = sum(r["angle_deg"] for r in subset) / len(subset)
        mean_offset = sum(r["offset_abs_error"] for r in subset) / len(subset)
        print(f"{method}: rows={len(subset)} mean_angle_deg={mean_angle:.4f} mean_offset_abs_error={mean_offset:.4f}")


if __name__ == "__main__":
    main()
