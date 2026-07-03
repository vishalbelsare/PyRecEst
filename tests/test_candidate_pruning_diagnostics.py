import unittest

import numpy as np

from pyrecest.utils.candidate_pruning_diagnostics import (
    CandidatePruningDiagnostics,
    candidate_pruning_diagnostics,
)


class TestCandidatePruningDiagnostics(unittest.TestCase):
    def test_default_config_reports_all_finite_entries(self):
        diagnostics = candidate_pruning_diagnostics(
            np.array([[1.0, np.inf], [2.0, 3.0]])
        )

        self.assertIsInstance(diagnostics, CandidatePruningDiagnostics)
        self.assertEqual(diagnostics.shape, (2, 2))
        self.assertEqual(diagnostics.total_entries, 4)
        self.assertEqual(diagnostics.finite_entries, 3)
        self.assertEqual(diagnostics.kept_entries, 3)
        self.assertEqual(diagnostics.pruned_finite_entries, 0)
        self.assertEqual(diagnostics.row_candidate_counts, (1, 2))
        self.assertEqual(diagnostics.column_candidate_counts, (2, 1))
        self.assertEqual(diagnostics.applied_rules, ("all_finite",))
        self.assertEqual(dict(diagnostics.rule_kept_entries), {"all_finite": 3})
        self.assertEqual(diagnostics.to_dict()["finite_retention_fraction"], 1.0)

    def test_row_and_column_top_k_union_reports_rule_counts(self):
        costs = np.array(
            [
                [1.0, 2.0, 10.0],
                [4.0, 3.0, 20.0],
                [30.0, 40.0, 0.5],
            ]
        )

        diagnostics = candidate_pruning_diagnostics(
            costs,
            config={"row_top_k": 1, "column_top_k": 1},
        )

        self.assertEqual(diagnostics.kept_entries, 5)
        self.assertEqual(diagnostics.pruned_finite_entries, 4)
        self.assertEqual(diagnostics.row_candidate_counts, (1, 2, 2))
        self.assertEqual(diagnostics.column_candidate_counts, (2, 2, 1))
        self.assertEqual(diagnostics.applied_rules, ("row_top_k", "column_top_k"))
        self.assertEqual(dict(diagnostics.rule_kept_entries), {"row_top_k": 3, "column_top_k": 3})

    def test_probability_threshold_and_cost_threshold_diagnostics(self):
        costs = np.array([[1.0, 9.0], [2.0, 8.0]])
        probabilities = np.array([[0.1, 0.8], [0.7, 0.2]])

        diagnostics = candidate_pruning_diagnostics(
            costs,
            probability_matrix=probabilities,
            config={"probability_threshold": 0.75, "max_cost": 2.0},
        )

        self.assertEqual(diagnostics.kept_entries, 3)
        self.assertEqual(diagnostics.row_candidate_counts, (2, 1))
        self.assertEqual(diagnostics.column_candidate_counts, (2, 1))
        self.assertEqual(
            diagnostics.applied_rules,
            ("probability_threshold", "max_cost"),
        )
        self.assertEqual(
            dict(diagnostics.rule_kept_entries),
            {"probability_threshold": 1, "max_cost": 2},
        )

    def test_zero_candidate_rows_and_columns_are_reported(self):
        costs = np.array([[100.0, 200.0], [1.0, 2.0], [300.0, 400.0]])

        diagnostics = candidate_pruning_diagnostics(
            costs,
            config={"max_cost": 2.0},
        )

        self.assertEqual(diagnostics.kept_entries, 2)
        self.assertEqual(diagnostics.row_candidate_counts, (0, 2, 0))
        self.assertEqual(diagnostics.column_candidate_counts, (1, 1))
        self.assertEqual(diagnostics.rows_without_candidates, 2)
        self.assertEqual(diagnostics.columns_without_candidates, 0)
        summary = diagnostics.to_dict()
        self.assertEqual(summary["min_row_candidate_count"], 0)
        self.assertEqual(summary["max_row_candidate_count"], 2)

    def test_percentile_rule_and_all_finite_fallback(self):
        percentile_diagnostics = candidate_pruning_diagnostics(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            config={"max_cost_percentile": 50.0},
        )
        self.assertEqual(percentile_diagnostics.kept_entries, 2)
        self.assertEqual(
            percentile_diagnostics.applied_rules,
            ("max_cost_percentile",),
        )

        fallback_diagnostics = candidate_pruning_diagnostics(
            np.array([[1.0, 2.0]]),
            config={},
        )
        self.assertEqual(fallback_diagnostics.kept_entries, 2)
        self.assertEqual(fallback_diagnostics.applied_rules, ("all_finite_fallback",))
        self.assertEqual(
            dict(fallback_diagnostics.rule_kept_entries),
            {"all_finite_fallback": 2},
        )

    def test_probability_threshold_requires_probability_matrix(self):
        with self.assertRaisesRegex(ValueError, "probability_matrix is required"):
            candidate_pruning_diagnostics(
                np.ones((2, 2)),
                config={"probability_threshold": 0.5},
            )

    def test_invalid_cost_matrix_validation_is_reused(self):
        with self.assertRaisesRegex(ValueError, "cost_matrix must be numeric"):
            candidate_pruning_diagnostics([["bad"]])


if __name__ == "__main__":
    unittest.main()
