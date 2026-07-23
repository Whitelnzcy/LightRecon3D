"""Dependency-light exact assignment used by Stage1 query matching."""

from __future__ import annotations

import numpy as np


def solve_rectangular_assignment(cost_query_target: np.ndarray):
    """Minimize one-to-one query/target cost when queries >= targets.

    Returns query indices ordered by target index, matching the return convention
    of the original exhaustive Stage1 matcher.  This is the rectangular
    shortest-augmenting-path form of the Hungarian algorithm.
    """

    cost = np.asarray(cost_query_target, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be a 2D matrix, got shape={cost.shape}")
    query_count, target_count = cost.shape
    if target_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if query_count < target_count:
        raise ValueError(
            "one-to-one assignment requires query_count >= target_count, "
            f"got {query_count} < {target_count}"
        )
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix contains NaN or infinite values")

    # The algorithm below assigns every row.  Targets are rows and queries are
    # columns, so all targets are assigned while surplus queries remain unused.
    row_cost = cost.T
    row_count, column_count = row_cost.shape
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        column0 = 0
        min_reduced_cost = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)

        while True:
            used[column0] = True
            active_row = matched_row[column0]
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    row_cost[active_row - 1, column - 1]
                    - u[active_row]
                    - v[column]
                )
                if reduced < min_reduced_cost[column]:
                    min_reduced_cost[column] = reduced
                    predecessor[column] = column0
                if min_reduced_cost[column] < delta:
                    delta = min_reduced_cost[column]
                    next_column = column

            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    min_reduced_cost[column] -= delta
            column0 = next_column
            if matched_row[column0] == 0:
                break

        while True:
            previous_column = predecessor[column0]
            matched_row[column0] = matched_row[previous_column]
            column0 = previous_column
            if column0 == 0:
                break

    query_for_target = np.empty(target_count, dtype=np.int64)
    for column in range(1, column_count + 1):
        row = matched_row[column]
        if row != 0:
            query_for_target[row - 1] = column - 1
    return query_for_target, np.arange(target_count, dtype=np.int64)
