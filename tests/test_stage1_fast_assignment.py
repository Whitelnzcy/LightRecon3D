import itertools
import unittest

import numpy as np

from stage1_fast_assignment import solve_rectangular_assignment


class Stage1FastAssignmentTest(unittest.TestCase):
    def test_known_rectangular_assignment(self):
        cost = np.asarray(
            [
                [9.0, 2.0, 7.0],
                [6.0, 4.0, 3.0],
                [5.0, 8.0, 1.0],
                [7.0, 6.0, 9.0],
            ]
        )
        queries, targets = solve_rectangular_assignment(cost)
        self.assertEqual(targets.tolist(), [0, 1, 2])
        self.assertEqual(len(set(queries.tolist())), 3)
        self.assertAlmostEqual(float(cost[queries, targets].sum()), 9.0)

    def test_matches_brute_force_on_small_random_matrices(self):
        generator = np.random.default_rng(20260713)
        for query_count in range(1, 7):
            for target_count in range(query_count + 1):
                for _ in range(8):
                    cost = generator.normal(size=(query_count, target_count))
                    queries, targets = solve_rectangular_assignment(cost)
                    solved = float(cost[queries, targets].sum())
                    brute = min(
                        (
                            sum(cost[query, target] for target, query in enumerate(candidate))
                            for candidate in itertools.permutations(
                                range(query_count), target_count
                            )
                        ),
                        default=0.0,
                    )
                    self.assertAlmostEqual(solved, float(brute), places=10)

    def test_empty_targets(self):
        queries, targets = solve_rectangular_assignment(np.empty((12, 0)))
        self.assertEqual(queries.dtype, np.int64)
        self.assertEqual(targets.dtype, np.int64)
        self.assertEqual(queries.size, 0)
        self.assertEqual(targets.size, 0)

    def test_rejects_more_targets_than_queries(self):
        with self.assertRaisesRegex(ValueError, "query_count >= target_count"):
            solve_rectangular_assignment(np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
