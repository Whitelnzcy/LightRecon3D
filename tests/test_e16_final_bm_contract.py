import hashlib
import json
from pathlib import Path

import numpy as np

from build_structured3d_point_aligned_gt import (
    sealed_sidecar_path,
    structured3d_scene_and_relative,
)
from final_bm_methods_e16 import (
    StandardizedLogistic,
    array_sha256,
    pair_f1,
    sha256_file,
)
from evaluate_final_bm_e16 import evaluate_one


def test_sealed_sidecar_path_preserves_selected_view_parent():
    image = Path(
        "F:/Structured3D_clean_e11/raw_rgb/scene_00426/"
        "2D_rendering/1588/perspective/empty/3/rgb_rawlight.png"
    )
    scene, relative = structured3d_scene_and_relative(image)
    assert scene == "scene_00426"
    assert relative.as_posix().endswith(
        "2D_rendering/1588/perspective/empty/3/rgb_rawlight.png"
    )
    assert sealed_sidecar_path(image, "F:/sealed", "layout.json") == Path(
        "F:/sealed/scene_00426/2D_rendering/1588/"
        "perspective/empty/3/layout.json"
    )


def test_standardized_logistic_learns_separable_data():
    matrix = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    model = StandardizedLogistic(0.2).fit(matrix, labels)
    probability = model.predict_proba(matrix)[:, 1]
    assert np.all(probability[:2] < 0.5)
    assert np.all(probability[2:] > 0.5)


def test_pair_f1_is_exact_for_equal_partitions():
    labels = np.asarray([0, 0, 1, 1])
    assert pair_f1(labels, labels.copy()) == 1.0


def test_array_hash_includes_dtype_and_shape():
    values = np.asarray([1, 2], dtype=np.int32)
    assert array_sha256(values) != array_sha256(values.astype(np.int64))
    assert array_sha256(values) != array_sha256(values.reshape(1, 2))


def test_immutable_evaluator_reports_perfect_prediction(tmp_path):
    cache_path = tmp_path / "global.npz"
    gt_path = tmp_path / "gt.npz"
    prediction_path = tmp_path / "prediction.npz"
    views = np.zeros(4, dtype=np.int16)
    pixels = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    global_indices = np.arange(4, dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    normals = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    offsets = np.zeros(2, dtype=np.float32)
    provenance = json.dumps({"source": "synthetic"}, sort_keys=True)
    np.savez(
        cache_path,
        points=np.zeros((4, 3), dtype=np.float32),
        view_indices=views,
        pixel_xy=pixels,
    )
    np.savez(
        gt_path,
        source_cache_indices=global_indices,
        point_plane_ids=labels,
        plane_normals=normals,
        view_indices=views,
        pixel_xy=pixels,
    )
    np.savez(
        prediction_path,
        global_point_indices=global_indices,
        view_indices=views,
        pixel_xy=pixels,
        plane_ids=labels,
        plane_normals=normals,
        plane_offsets=offsets,
        fragment_provenance_json=np.asarray(provenance),
    )
    receipt = {
        "scene_id": "scene_test",
        "method_id": "B16",
        "view_count": 4,
        "prediction_path": str(prediction_path),
        "whole_file_sha256": sha256_file(prediction_path),
        "support_sha256": array_sha256(
            np.column_stack((views, pixels, global_indices))
        ),
        "plane_id_sha256": array_sha256(labels),
        "plane_parameter_sha256": array_sha256(
            np.column_stack((normals, offsets))
        ),
        "provenance_sha256": hashlib.sha256(provenance.encode("utf-8")).hexdigest(),
    }
    metrics, _ = evaluate_one(
        receipt, {"output_npz": str(gt_path)}, cache_path
    )
    assert metrics["Plane_F1"] == 1.0
    assert metrics["partition_pair_f1"] == 1.0
    assert metrics["gt_support_assignment_rate"] == 1.0
    assert metrics["gt_support_matched_coverage"] == 1.0
    assert metrics["nonplane_false_assigned_point_count"] == 0
    assert metrics["prediction_hash_before"] == metrics["prediction_hash_after"]
