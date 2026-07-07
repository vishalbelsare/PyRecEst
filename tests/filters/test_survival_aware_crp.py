import unittest

import numpy.testing as npt

from pyrecest.filters.survival_aware_crp import (
    SurvivalAwareCRPAssociationPrior,
    posterior_existence_after_missed_detection,
)


class SurvivalAwareCRPAssociationPriorTest(unittest.TestCase):
    def test_uniform_context_recovers_dirichlet_process_predictive_weights(self):
        prior = SurvivalAwareCRPAssociationPrior(discount=0.0, strength=2.0)

        probabilities = prior.predictive_assignment_probabilities(
            [3, 2],
            clutter_rate=0.0,
        )

        npt.assert_allclose(probabilities, (3.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0, 0.0))
        self.assertTrue(prior.is_dirichlet_process_base)

    def test_pitman_yor_discount_lifts_birth_weight(self):
        prior = SurvivalAwareCRPAssociationPrior(discount=0.5, strength=1.0)

        weights = prior.assignment_weights([3, 2], birth_rate=1.0, clutter_rate=0.0)

        npt.assert_allclose(weights, (2.5, 1.5, 2.0, 0.0))

    def test_time_and_survival_decay_reduce_stale_track_weight(self):
        prior = SurvivalAwareCRPAssociationPrior(
            discount=0.0,
            strength=1.0,
            time_decay=0.5,
        )

        weights = prior.existing_track_weights(
            [10, 10],
            survival_probabilities=[1.0, 0.2],
            time_since_seen=[0.0, 3.0],
        )

        self.assertAlmostEqual(weights[0], 10.0)
        self.assertAlmostEqual(weights[1], 10.0 * 0.5**3 * 0.2)
        self.assertLess(weights[1], weights[0])

    def test_visibility_aware_missed_detection_preserves_occluded_tracks(self):
        visible_posterior = posterior_existence_after_missed_detection(
            predicted_existence=0.8,
            detection_probability=0.9,
            visibility_probability=1.0,
        )
        occluded_posterior = posterior_existence_after_missed_detection(
            predicted_existence=0.8,
            detection_probability=0.9,
            visibility_probability=0.1,
        )

        self.assertLess(visible_posterior, occluded_posterior)
        self.assertAlmostEqual(
            posterior_existence_after_missed_detection(0.8, 0.9, 0.0),
            0.8,
        )

    def test_birth_and_clutter_are_normalized_alternatives(self):
        prior = SurvivalAwareCRPAssociationPrior(discount=0.0, strength=2.0)

        probabilities = prior.predictive_assignment_probabilities(
            [2.0],
            birth_rate=0.5,
            clutter_rate=1.0,
        )

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[-2], 0.0)
        self.assertGreater(probabilities[-1], 0.0)

    def test_association_cost_matrix_uses_context_and_compatibility(self):
        prior = SurvivalAwareCRPAssociationPrior(discount=0.0, strength=1.0)

        costs = prior.association_cost_matrix(
            track_masses=[5.0, 5.0],
            survival_probabilities=[1.0, 0.5],
            compatibility_matrix=[
                [1.0, 0.25],
                [1.0, 1.0],
            ],
        )

        self.assertLess(costs[0][0], costs[1][0])
        self.assertLess(costs[0][0], costs[0][1])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(discount=-0.1)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(discount=1.0)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(discount=0.0, strength=0.0)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(time_decay=1.1)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(minimum_weight=0.0)
        with self.assertRaises(ValueError):
            posterior_existence_after_missed_detection(1.2, 0.9)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior().predictive_assignment_probabilities(
                [0.0],
                birth_rate=0.0,
                clutter_rate=0.0,
            )
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior().association_cost_matrix(
                track_masses=[1.0, 2.0],
                compatibility_matrix=[[1.0]],
            )


if __name__ == "__main__":
    unittest.main()
