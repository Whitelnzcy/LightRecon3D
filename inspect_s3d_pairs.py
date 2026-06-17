import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from dataloaders.s3d_dataset import Structured3DDataset


def parse_args():
    parser = argparse.ArgumentParser(
        "Inspect real Structured3D image pairs before pair-conditioned training"
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--pair_strategy", default="adjacent", choices=("adjacent", "all"))
    parser.add_argument("--pair_max_view_id_gap", type=int, default=None)
    parser.add_argument("--num_pairs", type=int, default=24)
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated dataset indices. Overrides balanced sampling.",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--max_draw_matches", type=int, default=40)
    return parser.parse_args()


def tensor_rgb_to_uint8(image):
    if not torch.is_tensor(image):
        raise TypeError(f"Expected torch.Tensor, got {type(image)!r}")
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def colorize_labels(labels):
    label_array = labels.detach().cpu().numpy().astype(np.int32)
    output = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    unique_ids = [int(value) for value in np.unique(label_array) if value > 0 and value != 255]
    for plane_id in unique_ids:
        hue = (plane_id * 47) % 180
        hsv = np.zeros((*label_array.shape, 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 190
        hsv[..., 2] = 230
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        output[label_array == plane_id] = color
    output[label_array == 255] = (128, 128, 128)
    return output


def add_title(image_rgb, title, height=34):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bar = np.full((height, image_bgr.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(np.concatenate((bar, image_bgr), axis=0), cv2.COLOR_BGR2RGB)


def orb_diagnostics(rgb1, rgb2, max_draw_matches):
    gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    good_matches = []
    if descriptors1 is not None and descriptors2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(descriptors1, descriptors2, k=2)
        for candidates in knn:
            if len(candidates) < 2:
                continue
            first, second = candidates
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)
        good_matches.sort(key=lambda match: match.distance)

    denominator = max(min(len(keypoints1), len(keypoints2)), 1)
    ratio = len(good_matches) / denominator
    match_visualization = cv2.drawMatches(
        cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR),
        keypoints1,
        cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR),
        keypoints2,
        good_matches[:max_draw_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return {
        "orb_keypoints1": len(keypoints1),
        "orb_keypoints2": len(keypoints2),
        "orb_good_matches": len(good_matches),
        "orb_match_ratio": float(ratio),
        "match_visualization_rgb": cv2.cvtColor(match_visualization, cv2.COLOR_BGR2RGB),
    }


def parse_indices(text):
    if not text.strip():
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def balanced_indices(dataset, count, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        key = (sample["scene_name"], sample["pair_group"])
        grouped[key].append(index)

    groups = list(grouped.values())
    rng.shuffle(groups)
    for values in groups:
        rng.shuffle(values)

    selected = []
    cursor = 0
    while len(selected) < min(count, len(dataset)):
        added = False
        for values in groups:
            if cursor < len(values):
                selected.append(values[cursor])
                added = True
                if len(selected) >= min(count, len(dataset)):
                    break
        if not added:
            break
        cursor += 1
    return selected


def view_gap(view1, view2):
    first = view1.get("view_id")
    second = view2.get("view_id")
    if isinstance(first, int) and isinstance(second, int):
        return abs(second - first)
    return ""


def plane_count(labels):
    values = torch.unique(labels)
    return int(((values > 0) & (values != 255)).sum())


def save_pair_figure(path, rgb1, rgb2, labels1, labels2, match_rgb, metadata):
    panel1 = add_title(rgb1, f"View 1: {metadata['view_id1']}")
    panel2 = add_title(rgb2, f"View 2: {metadata['view_id2']}")
    plane1 = add_title(colorize_labels(labels1), f"GT planes view 1: {metadata['plane_count1']}")
    plane2 = add_title(colorize_labels(labels2), f"GT planes view 2: {metadata['plane_count2']}")

    top = np.concatenate((panel1, panel2), axis=1)
    middle = np.concatenate((plane1, plane2), axis=1)
    match_rgb = cv2.resize(match_rgb, (top.shape[1], rgb1.shape[0]))
    match_rgb = add_title(
        match_rgb,
        f"ORB good matches: {metadata['orb_good_matches']} | ratio: {metadata['orb_match_ratio']:.4f}",
    )

    footer = np.full((72, top.shape[1], 3), 248, dtype=np.uint8)
    lines = [
        f"index={metadata['sample_idx']} scene={metadata['scene_name']} gap={metadata['view_gap']}",
        f"group={metadata['pair_group']}",
        f"mean abs RGB diff={metadata['mean_abs_rgb_difference']:.4f} same_path={metadata['same_image_path']}",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            footer,
            text,
            (8, 20 + row * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    canvas = np.concatenate((top, middle, match_rgb, footer), axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode="pair",
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )

    explicit = parse_indices(args.indices)
    if explicit is None:
        indices = balanced_indices(dataset, args.num_pairs, args.seed)
    else:
        indices = [index for index in explicit if 0 <= index < len(dataset)]

    if not indices:
        raise RuntimeError("No valid pair indices were selected")

    rows = []
    for ordinal, index in enumerate(indices):
        raw = dataset.samples[index]
        sample = dataset[index]
        rgb1 = tensor_rgb_to_uint8(sample["img1"])
        rgb2 = tensor_rgb_to_uint8(sample["img2"])
        orb = orb_diagnostics(rgb1, rgb2, args.max_draw_matches)

        row = {
            "sample_idx": index,
            "scene_name": raw["scene_name"],
            "pair_group": raw["pair_group"],
            "view_id1": raw["view1"]["view_id"],
            "view_id2": raw["view2"]["view_id"],
            "view_gap": view_gap(raw["view1"], raw["view2"]),
            "rgb_path1": raw["view1"]["rgb_path"],
            "rgb_path2": raw["view2"]["rgb_path"],
            "same_image_path": raw["view1"]["rgb_path"] == raw["view2"]["rgb_path"],
            "mean_abs_rgb_difference": float(
                np.abs(rgb1.astype(np.float32) - rgb2.astype(np.float32)).mean() / 255.0
            ),
            "plane_count1": plane_count(sample["gt_plane1"]),
            "plane_count2": plane_count(sample["gt_plane2"]),
            "orb_keypoints1": orb["orb_keypoints1"],
            "orb_keypoints2": orb["orb_keypoints2"],
            "orb_good_matches": orb["orb_good_matches"],
            "orb_match_ratio": orb["orb_match_ratio"],
        }
        rows.append(row)

        image_path = output_dir / f"pair_{ordinal:03d}_idx_{index:06d}.png"
        save_pair_figure(
            image_path,
            rgb1,
            rgb2,
            sample["gt_plane1"],
            sample["gt_plane2"],
            orb["match_visualization_rgb"],
            row,
        )
        print(
            f"[{ordinal + 1}/{len(indices)}] idx={index} "
            f"scene={row['scene_name']} views=({row['view_id1']},{row['view_id2']}) "
            f"matches={row['orb_good_matches']} ratio={row['orb_match_ratio']:.4f}",
            flush=True,
        )

    json_path = output_dir / "pair_summary.json"
    csv_path = output_dir / "pair_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ratios = np.asarray([row["orb_match_ratio"] for row in rows], dtype=np.float32)
    print(
        json.dumps(
            {
                "dataset_pairs": len(dataset),
                "inspected_pairs": len(rows),
                "unique_scenes": len({row["scene_name"] for row in rows}),
                "unique_pair_groups": len({row["pair_group"] for row in rows}),
                "same_image_pairs": sum(bool(row["same_image_path"]) for row in rows),
                "orb_match_ratio_mean": float(ratios.mean()),
                "orb_match_ratio_median": float(np.median(ratios)),
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
