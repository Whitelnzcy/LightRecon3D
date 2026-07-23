import csv
import tempfile
import unittest
from pathlib import Path

from analyze_guided_ransac_mechanism_ablation import (
    COMPARISONS,
    PAIRWISE_F1,
    analyze,
    holm_adjust,
)


class MechanismStatisticsTests(unittest.TestCase):
    def test_holm_adjust_is_monotone_in_sorted_pvalues(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_analysis_pairs_by_item_and_writes_immutable_outputs(self):
        methods = sorted({method for _, reference, method in COMPARISONS for method in (reference, method)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "metrics.csv"
            with source.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "item_id",
                        "scene_name",
                        "seed",
                        "method",
                        PAIRWISE_F1,
                        "runtime_seconds",
                    ],
                )
                writer.writeheader()
                for item_index in range(5):
                    for seed in range(3):
                        for method_index, method in enumerate(methods):
                            writer.writerow(
                                {
                                    "item_id": f"item_{item_index}",
                                    "scene_name": f"scene_{item_index}",
                                    "seed": seed,
                                    "method": method,
                                    PAIRWISE_F1: 0.5 + 0.01 * method_index + 0.001 * seed,
                                    "runtime_seconds": 10 + method_index + seed,
                                }
                            )
            output = root / "analysis"
            result = analyze(source, output, bootstrap_samples=100, bootstrap_seed=7)
            self.assertEqual(len(result["comparisons"]), len(COMPARISONS))
            self.assertEqual(result["comparisons"][0]["pairs"], 5)
            self.assertEqual(result["comparisons"][0]["seeds"], [0, 1, 2])
            self.assertEqual(len(result["mode_summary"]), len(methods))
            self.assertTrue((output / "mechanism_paired_statistics.json").is_file())
            self.assertTrue((output / "mechanism_paired_statistics.csv").is_file())
            self.assertTrue((output / "mechanism_paired_statistics.md").is_file())
            with self.assertRaises(FileExistsError):
                analyze(source, output, bootstrap_samples=100, bootstrap_seed=7)


if __name__ == "__main__":
    unittest.main()
