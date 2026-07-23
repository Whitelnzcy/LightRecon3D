import csv
import tempfile
import unittest
from pathlib import Path

from analyze_main_component_ablation import METHODS, analyze, holm_adjust


class AnalyzeMainComponentAblationTests(unittest.TestCase):
    def test_holm_adjust_is_monotone_in_rank(self):
        adjusted = holm_adjust([0.01, 0.04, 0.02])
        self.assertEqual(adjusted, [0.03, 0.04, 0.04])

    def test_analyze_writes_complete_six_contrast_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "metrics.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["item_id", "scene_name", "method", "support_partition_pairwise_f1", "runtime_seconds"],
                )
                writer.writeheader()
                for item_index in range(5):
                    for method_index, method in enumerate(METHODS):
                        writer.writerow(
                            {
                                "item_id": f"item_{item_index}",
                                "scene_name": f"scene_{item_index}",
                                "method": method,
                                "support_partition_pairwise_f1": 0.1 * method_index + 0.001 * item_index,
                                "runtime_seconds": "nan" if method_index == 1 else 1.0,
                            }
                        )
            output = root / "out"
            result = analyze(source, output, bootstrap_samples=100, bootstrap_seed=7)
            self.assertEqual(len(result["comparisons"]), 6)
            self.assertTrue((output / "component_paired_statistics.json").is_file())
            self.assertEqual(result["comparisons"][2]["wins"], 5)


if __name__ == "__main__":
    unittest.main()
