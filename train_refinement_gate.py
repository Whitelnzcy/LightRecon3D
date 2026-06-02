import argparse
import csv
import math
from pathlib import Path

import numpy as np


FEATURES = [
    "pred_confidence",
    "inlier_ratio",
    "mean_inlier_dist",
    "dev_pred_ransac_deg",
    "dev_pred_svd_deg",
    "offset_delta",
    "log_pixels",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def dot_abs(a, b):
    return abs(a["nx"] * b["nx"] + a["ny"] * b["ny"] + a["nz"] * b["nz"])


def normal_dev(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_abs(a, b)))))


def read_grouped(path):
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    for row in rows:
        for key in ["sample_idx", "plane_id", "pixels", "points", "inliers"]:
            row[key] = int(float(row[key]))
        for key in [
            "mean_inlier_dist",
            "angle_deg",
            "offset_abs_error",
            "pred_confidence",
            "nx",
            "ny",
            "nz",
            "offset",
        ]:
            row[key] = float(row[key]) if row[key] not in ("", "nan") else math.nan
    grouped = {}
    for row in rows:
        grouped.setdefault((row["sample_idx"], row["plane_id"]), {})[row["method"]] = row
    return [g for g in grouped.values() if {"pred", "ransac_refine", "svd_all"} <= set(g)]


def make_examples(groups):
    xs, ys, meta = [], [], []
    for group in groups:
        pred = group["pred"]
        ran = group["ransac_refine"]
        svd = group["svd_all"]
        pred_score = pred["angle_deg"] / 10.0 + pred["offset_abs_error"] / 0.05
        ran_score = ran["angle_deg"] / 10.0 + ran["offset_abs_error"] / 0.05
        label = 1.0 if ran_score + 0.05 < pred_score else 0.0
        feat = {
            "pred_confidence": pred["pred_confidence"],
            "inlier_ratio": ran["inliers"] / max(1, ran["points"]),
            "mean_inlier_dist": min(ran["mean_inlier_dist"], 0.05),
            "dev_pred_ransac_deg": normal_dev(pred, ran) / 90.0,
            "dev_pred_svd_deg": normal_dev(pred, svd) / 90.0,
            "offset_delta": min(abs(ran["offset"] - pred["offset"]), 0.5),
            "log_pixels": math.log(max(1, pred["pixels"])) / 8.0,
        }
        xs.append([feat[k] for k in FEATURES])
        ys.append(label)
        meta.append(group)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), meta


def metric(meta, use_ransac):
    chosen = []
    accepted = 0
    correct_accept = 0
    for group, use in zip(meta, use_ransac):
        pred = group["pred"]
        ran = group["ransac_refine"]
        pred_score = pred["angle_deg"] / 10.0 + pred["offset_abs_error"] / 0.05
        ran_score = ran["angle_deg"] / 10.0 + ran["offset_abs_error"] / 0.05
        if use:
            accepted += 1
            correct_accept += int(ran_score < pred_score)
            chosen.append(ran)
        else:
            chosen.append(pred)
    angle = sum(r["angle_deg"] for r in chosen) / len(chosen)
    offset = sum(r["offset_abs_error"] for r in chosen) / len(chosen)
    return angle, offset, accepted, correct_accept


def train_logistic(x, y, epochs=4000, lr=0.03, l2=1e-3):
    rng = np.random.default_rng(13)
    w = rng.normal(scale=0.05, size=x.shape[1])
    b = 0.0
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    weights = np.where(y > 0.5, neg / pos, 1.0)
    for _ in range(epochs):
        logits = x @ w + b
        p = sigmoid(logits)
        err = (p - y) * weights
        grad_w = x.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="outputs/point_anchor_conf_ransac_refine_train96.csv")
    parser.add_argument("--val_csv", default="outputs/point_anchor_conf_ransac_refine_val64.csv")
    parser.add_argument("--output_csv", default="outputs/refinement_gate_val64_predictions.csv")
    args = parser.parse_args()

    train_groups = read_grouped(args.train_csv)
    val_groups = read_grouped(args.val_csv)
    x_train, y_train, _ = make_examples(train_groups)
    x_val, y_val, val_meta = make_examples(val_groups)

    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    w, b = train_logistic(x_train_n, y_train)
    train_probs = sigmoid(x_train_n @ w + b)
    val_probs = sigmoid(x_val_n @ w + b)

    best = None
    for q in np.linspace(0.35, 0.9, 56):
        threshold = float(np.quantile(train_probs, q))
        use = val_probs > threshold
        angle, offset, accepted, correct_accept = metric(val_meta, use.tolist())
        score = angle + offset * 15.0
        item = (score, q, threshold, angle, offset, accepted, correct_accept)
        if best is None or item < best:
            best = item

    pred_angle, pred_offset, _, _ = metric(val_meta, [False] * len(val_meta))
    ran_angle, ran_offset, _, _ = metric(val_meta, [True] * len(val_meta))
    threshold = best[2]
    use = val_probs > threshold
    gate_angle, gate_offset, accepted, correct_accept = metric(val_meta, use.tolist())
    label_acc = ((val_probs > threshold).astype(np.float64) == y_val).mean()

    print("features:", ", ".join(FEATURES))
    print(f"weights: {dict(zip(FEATURES, w.round(4)))} bias={b:.4f}")
    print(f"train positives: {int(y_train.sum())}/{len(y_train)}")
    print(f"val positives: {int(y_val.sum())}/{len(y_val)}")
    print(f"pred only: angle={pred_angle:.4f} offset={pred_offset:.4f}")
    print(f"ransac all: angle={ran_angle:.4f} offset={ran_offset:.4f}")
    print(
        "learned gate: "
        f"angle={gate_angle:.4f} offset={gate_offset:.4f} "
        f"accepted={accepted}/{len(val_meta)} correct_accept={correct_accept} "
        f"threshold={threshold:.4f} q={best[1]:.2f} label_acc={label_acc:.4f}"
    )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "sample_idx",
            "plane_id",
            "gate_prob",
            "accept_ransac",
            "target_accept",
            "pred_angle",
            "ransac_angle",
            "pred_offset_error",
            "ransac_offset_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, prob, accept, target in zip(val_meta, val_probs.tolist(), use.tolist(), y_val.tolist()):
            pred = group["pred"]
            ran = group["ransac_refine"]
            writer.writerow(
                {
                    "sample_idx": pred["sample_idx"],
                    "plane_id": pred["plane_id"],
                    "gate_prob": prob,
                    "accept_ransac": int(accept),
                    "target_accept": int(target),
                    "pred_angle": pred["angle_deg"],
                    "ransac_angle": ran["angle_deg"],
                    "pred_offset_error": pred["offset_abs_error"],
                    "ransac_offset_error": ran["offset_abs_error"],
                }
            )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
