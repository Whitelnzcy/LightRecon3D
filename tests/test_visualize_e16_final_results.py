import numpy as np

from visualize_e16_final_results import (
    align_partition_labels,
    paired_scene_rows,
    select_scene_groups,
)


def metric(scene, method, pair_f1, plane_count):
    return {
        "scene_id": scene,
        "method_id": method,
        "partition_pair_f1": pair_f1,
        "predicted_plane_count": plane_count,
    }


def test_pairing_and_mechanical_gallery_selection():
    metrics = [
        metric("same", "B16", 0.4, 5),
        metric("same", "M16", 0.4, 5),
        metric("tiny", "B16", 0.4, 4),
        metric("tiny", "M16", 0.4001, 4),
        metric("split", "B16", 0.4, 5),
        metric("split", "M16", 0.399, 12),
        metric("gain", "B16", 0.4, 5),
        metric("gain", "M16", 0.41, 14),
    ]
    rows = paired_scene_rows(metrics)
    similar, extra = select_scene_groups(rows, similar_count=2, extra_split_count=1)
    assert [row["scene_id"] for row in similar] == ["same", "tiny"]
    assert [row["scene_id"] for row in extra] == ["split"]


def test_display_label_alignment_matches_overlapping_partitions():
    reference = np.asarray([0, 0, 1, 1, 1, 2])
    candidate = np.asarray([7, 7, 9, 9, 8, 8])
    aligned = align_partition_labels(reference, candidate)
    assert aligned[:2].tolist() == [0, 0]
    assert aligned[2:4].tolist() == [1, 1]
