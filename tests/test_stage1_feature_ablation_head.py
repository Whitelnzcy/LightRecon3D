import unittest
from argparse import Namespace

import torch

from models.stage1_feature_ablation_head import Stage1FeatureAblationHead
from run_stage1_feature_ablation import support_metrics


class Stage1FeatureAblationHeadTest(unittest.TestCase):
    def test_parameter_matched_inputs_produce_same_output_schema(self):
        dust3r = Stage1FeatureAblationHead(input_source="dust3r")
        rgb = Stage1FeatureAblationHead(input_source="rgb")

        self.assertEqual(dust3r.projected_parameter_count(), rgb.projected_parameter_count())
        self.assertEqual(
            sum(parameter.numel() for parameter in dust3r.parameters()),
            sum(parameter.numel() for parameter in rgb.parameters()),
        )

        with torch.no_grad():
            dust3r_output = dust3r(torch.randn(2, 768, 32, 32))
            rgb_output = rgb(torch.randn(2, 3, 512, 512))

        for output in (dust3r_output, rgb_output):
            self.assertEqual(tuple(output["mask_logits"].shape), (2, 8, 32, 32))
            self.assertEqual(tuple(output["background_logits"].shape), (2, 1, 32, 32))
            self.assertEqual(tuple(output["existence_logits"].shape), (2, 8))

    def test_rejects_non_matched_feature_dimension(self):
        with self.assertRaisesRegex(ValueError, "Parameter matching"):
            Stage1FeatureAblationHead(input_source="rgb", feature_dim=512)

    def test_support_metrics_recognize_perfect_single_plane(self):
        labels = torch.zeros(1, 512, 512, dtype=torch.long)
        labels[:, :256] = 1
        mask_logits = torch.full((1, 8, 32, 32), -10.0)
        mask_logits[:, 0, :16] = 10.0
        background_logits = torch.full((1, 1, 32, 32), -10.0)
        background_logits[:, :, 16:] = 10.0
        existence_logits = torch.full((1, 8), -10.0)
        existence_logits[:, 0] = 10.0
        args = Namespace(
            num_queries=8,
            min_plane_pixels=4,
            match_bce_weight=1.0,
            match_dice_weight=2.0,
            match_existence_weight=0.0,
            class_score_weight=0.0,
            existence_threshold=0.5,
        )
        metrics = support_metrics(
            {
                "mask_logits": mask_logits,
                "background_logits": background_logits,
                "existence_logits": existence_logits,
            },
            labels,
            args,
        )
        self.assertAlmostEqual(metrics["support_precision"], 1.0)
        self.assertAlmostEqual(metrics["support_recall"], 1.0)
        self.assertAlmostEqual(metrics["support_f1"], 1.0)
        self.assertAlmostEqual(metrics["matched_mask_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
