import unittest

import numpy as np
import torch

from train_intra_fragment_splitter import (
    IntraFragmentSplitter,
    balanced_global_pairs,
    fragment_split_target,
    nearest_different_pairs,
    point_and_edge_targets,
    stratified_sample_indices,
)


class IntraFragmentSplitterTest(unittest.TestCase):
    def test_fragment_split_target_uses_purity_threshold(self):
        self.assertEqual(
            fragment_split_target(torch.tensor([0] * 9 + [1])).item(),
            0.0,
        )
        self.assertEqual(
            fragment_split_target(torch.tensor([0] * 7 + [1] * 3)).item(),
            1.0,
        )

    def test_balanced_global_pairs_contains_both_classes(self):
        labels = torch.tensor([0, 0, 1, 1])
        source, target, pair_labels = balanced_global_pairs(labels, 4)
        self.assertEqual(len(source), len(target))
        self.assertEqual(set(pair_labels.tolist()), {0.0, 1.0})

    def test_stratified_sampling_keeps_minority_plane(self):
        labels = np.asarray([0] * 90 + [1] * 10)
        indices = stratified_sample_indices(
            labels, 20, np.random.default_rng(3)
        )
        self.assertEqual(len(indices), 20)
        self.assertIn(0, np.unique(labels[indices]))
        self.assertIn(1, np.unique(labels[indices]))

    def test_edge_and_boundary_targets(self):
        labels = torch.tensor([0, 0, 1])
        neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
        edges, boundary = point_and_edge_targets(labels, neighbors)
        torch.testing.assert_close(
            edges,
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        )
        torch.testing.assert_close(boundary, torch.ones(3))

    def test_model_shapes(self):
        model = IntraFragmentSplitter(input_dim=13, hidden_dim=32, embedding_dim=8)
        features = torch.randn(12, 13)
        xyz = torch.randn(12, 3)
        pixels = torch.randn(12, 2)
        views = torch.zeros(12, dtype=torch.long)
        neighbors = torch.stack(
            (torch.roll(torch.arange(12), 1), torch.roll(torch.arange(12), 2)),
            dim=1,
        )
        embedding, boundary = model.encode(features, neighbors)
        edges = model.edge_logits(
            embedding, xyz, pixels, views, neighbors
        )
        self.assertEqual(embedding.shape, (12, 8))
        self.assertEqual(boundary.shape, (12,))
        self.assertEqual(edges.shape, (12, 2))

    def test_multiscale_model_has_expected_capacity_and_split_gate(self):
        model = IntraFragmentSplitter(
            input_dim=13,
            hidden_dim=128,
            embedding_dim=64,
            architecture="multiscale",
            graph_layers=3,
            attention_layers=4,
            attention_heads=4,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameter_count, 800_000)
        self.assertLess(parameter_count, 2_000_000)
        features = torch.randn(32, 13)
        neighbors = torch.stack(
            [torch.roll(torch.arange(32), shift) for shift in range(1, 17)],
            dim=1,
        )
        embedding, boundary = model.encode(features, neighbors)
        split_logit = model.fragment_split_logit(embedding, boundary)
        self.assertEqual(embedding.shape, (32, 64))
        self.assertEqual(boundary.shape, (32,))
        self.assertEqual(split_logit.shape, ())

    def test_nearest_different_pairs_cross_labels(self):
        xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]]
        )
        labels = torch.tensor([0, 1, 1])
        source, target = nearest_different_pairs(xyz, labels)
        self.assertEqual(len(source), 3)
        self.assertTrue(torch.all(labels[source] != labels[target]))


if __name__ == "__main__":
    unittest.main()
