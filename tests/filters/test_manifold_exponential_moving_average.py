import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.filters import ManifoldExponentialMovingAverage


def _phi_euclidean(state, xi):
    return state + xi


def _phi_inv_euclidean(state_ref, state):
    return state - state_ref


def _phi_so2(state, xi):
    return state + xi[0]


def _phi_inv_so2(state_ref, state):
    diff = state - state_ref
    return array([(diff + pi) % (2.0 * pi) - pi])


class TestManifoldExponentialMovingAverage(unittest.TestCase):
    def test_update_matches_euclidean_ema(self):
        ema = ManifoldExponentialMovingAverage(
            initial_state=array([0.0, 0.0]),
            alpha=0.25,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
        )

        ema.update(array([4.0, -2.0]))
        npt.assert_allclose(ema.get_point_estimate(), array([1.0, -0.5]))

        ema.update(array([4.0, -2.0]))
        npt.assert_allclose(ema.filter_state, array([1.75, -0.875]))

    def test_first_update_initializes_empty_estimate(self):
        sample = array([2.0, 3.0])
        ema = ManifoldExponentialMovingAverage(
            initial_state=None,
            alpha=0.5,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
        )

        ema.update(sample)

        npt.assert_allclose(ema.get_point_estimate(), sample)

    def test_mutable_input_states_are_copied(self):
        initial_state = np.array([1.0, 2.0])
        ema = ManifoldExponentialMovingAverage(
            initial_state=initial_state,
            alpha=0.5,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
        )

        initial_state[0] = 99.0
        npt.assert_allclose(ema.filter_state, np.array([1.0, 2.0]))

        replacement_state = np.array([3.0, 4.0])
        ema.filter_state = replacement_state
        replacement_state[1] = 99.0
        npt.assert_allclose(ema.filter_state, np.array([3.0, 4.0]))

        first_sample = np.array([5.0, 6.0])
        empty_ema = ManifoldExponentialMovingAverage(
            initial_state=None,
            alpha=0.5,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
        )
        empty_ema.update(first_sample)
        first_sample[0] = 99.0
        npt.assert_allclose(empty_ema.filter_state, np.array([5.0, 6.0]))

    def test_circle_update_uses_shortest_tangent_direction(self):
        ema = ManifoldExponentialMovingAverage(
            initial_state=3.0,
            alpha=0.5,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
        )

        ema.update(-3.0)

        npt.assert_allclose(ema.get_point_estimate(), pi, atol=1e-12)

    def test_alpha_must_be_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            ManifoldExponentialMovingAverage(
                initial_state=0.0,
                alpha=1.1,
                phi=_phi_so2,
                phi_inv=_phi_inv_so2,
            )

        ema = ManifoldExponentialMovingAverage(
            initial_state=0.0,
            alpha=1.0,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
        )
        with self.assertRaises(ValueError):
            ema.alpha = -0.1

    def test_alpha_rejects_boolean_values(self):
        for alpha in (True, False, np.bool_(True), np.array(False)):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(TypeError, "real scalar"):
                    ManifoldExponentialMovingAverage(
                        initial_state=0.0,
                        alpha=alpha,
                        phi=_phi_so2,
                        phi_inv=_phi_inv_so2,
                    )

        ema = ManifoldExponentialMovingAverage(
            initial_state=0.0,
            alpha=0.5,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
        )
        with self.assertRaisesRegex(TypeError, "real scalar"):
            ema.alpha = True

    def test_alpha_rejects_text_values(self):
        for alpha in ("0.5", b"0.5", np.array("0.5")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(TypeError, "real scalar"):
                    ManifoldExponentialMovingAverage(
                        initial_state=0.0,
                        alpha=alpha,
                        phi=_phi_so2,
                        phi_inv=_phi_inv_so2,
                    )

        ema = ManifoldExponentialMovingAverage(
            initial_state=0.0,
            alpha=0.5,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
        )
        with self.assertRaisesRegex(TypeError, "real scalar"):
            ema.alpha = "0.25"


if __name__ == "__main__":
    unittest.main()
