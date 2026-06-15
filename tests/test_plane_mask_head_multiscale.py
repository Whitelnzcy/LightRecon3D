import argparse
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.plane_mask_head import PlaneMaskHead
from train_stage1_plane_masks import sample_loss, select_plane_ids


class PlaneMaskHeadMultiscaleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = PlaneMaskHead(
            feature_dim=32,
            encoder_feature_dim=64,
            shallow_feature_dim=32,
            middle_feature_dim=32,
            hidden_dim=32,
            num_queries=4,
            num_decoder_layers=1,
            num_heads=4,
        )
        self.deep = torch.randn(1, 32, 8, 8)
        self.middle = torch.randn(1, 32, 8, 8)
        self.encoder = torch.randn(1, 64, 8, 8)
        self.shallow = torch.randn(1, 32, 8, 8)
        self.image = torch.randn(1, 3, 128, 128)

    def test_shapes_and_zero_refinement_equivalence(self):
        self.head.eval()
        with torch.no_grad():
            output = self.head(
                self.deep,
                middle_feature=self.middle,
                shallow_feature=self.shallow,
                encoder_feature=self.encoder,
                image=self.image,
            )
        self.assertEqual(output["mask_logits_32"].shape, (1, 4, 8, 8))
        self.assertEqual(output["mask_logits_64"].shape, (1, 4, 16, 16))
        self.assertEqual(output["mask_logits_128"].shape, (1, 4, 32, 32))

        class32 = torch.cat(
            (output["mask_logits_32"], output["background_logits_32"]),
            dim=1,
        )
        expected = F.interpolate(
            F.interpolate(
                class32,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            ),
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        actual = torch.cat(
            (output["mask_logits_128"], output["background_logits_128"]),
            dim=1,
        )
        self.assertLess(float((actual - expected).abs().max()), 1e-6)

    def test_old_coarse_state_loads_without_changing_coarse_modules(self):
        source = self.head.state_dict()
        coarse_prefixes = (
            "feature_proj.",
            "query_embed.",
            "decoder.",
            "mask_embed.",
            "existence_head.",
            "background_head.",
        )
        coarse_state = {
            key: value.clone()
            for key, value in source.items()
            if key.startswith(coarse_prefixes)
        }
        target = PlaneMaskHead(
            feature_dim=32,
            encoder_feature_dim=64,
            shallow_feature_dim=32,
            middle_feature_dim=32,
            hidden_dim=32,
            num_queries=4,
            num_decoder_layers=1,
            num_heads=4,
        )
        target.load_coarse_state_dict(coarse_state)
        for key, expected in coarse_state.items():
            self.assertTrue(torch.equal(target.state_dict()[key], expected), key)

    def test_multiscale_loss_backpropagates_to_zero_gates(self):
        args = argparse.Namespace(
            num_queries=4,
            match_bce_weight=1.0,
            match_dice_weight=2.0,
            scale_weights=(1.0, 0.7, 1.0),
            small_plane_max_weight=3.0,
            focal_gamma=2.0,
            tversky_fp_weight=0.7,
            tversky_fn_weight=0.3,
            mask_focal_weight=1.0,
            mask_tversky_weight=2.0,
            partition_weight=1.0,
            boundary_loss_weight=1.0,
            separation_weight=0.5,
            smoothness_weight=0.05,
            boundary_band=3,
            separation_margin=1.0,
            existence_weight=1.0,
            existence_threshold=0.5,
        )
        labels = torch.zeros(1, 128, 128, dtype=torch.long)
        labels[:, 10:100, 10:60] = 1
        labels[:, 20:110, 60:115] = 2
        _, plane_ids = select_plane_ids(labels, (8, 8), 4, 1)

        self.head.train()
        output = self.head(
            self.deep,
            middle_feature=self.middle,
            shallow_feature=self.shallow,
            encoder_feature=self.encoder,
            image=self.image,
        )
        loss, _ = sample_loss(output, labels, plane_ids, args)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.head.alpha64.grad)
        self.assertIsNotNone(self.head.alpha128.grad)
        self.assertGreater(abs(float(self.head.alpha64.grad)), 0.0)
        self.assertGreater(abs(float(self.head.alpha128.grad)), 0.0)

    def test_refinement_gate_suppresses_confident_interiors(self):
        logits = torch.zeros(1, 4, 4, 4)
        logits[:, 0] = 12.0
        confident_gate = self.head._refinement_gate(logits)
        self.assertLess(float(confident_gate.max()), 1e-4)

        ambiguous = torch.zeros(1, 4, 4, 4)
        ambiguous_gate = self.head._refinement_gate(ambiguous)
        self.assertGreater(float(ambiguous_gate.min()), 0.99)


if __name__ == "__main__":
    unittest.main()
