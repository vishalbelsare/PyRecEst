import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array
from pyrecest.models import (
    AdditiveNoiseMeasurementModel,
    AdditiveNoiseTransitionModel,
)


def assert_backend_allclose(test_case, actual, expected):
    test_case.assertTrue(bool(allclose(actual, expected)))


class AdditiveNoiseConstructorOwnershipTest(unittest.TestCase):
    def test_transition_model_copies_explicit_noise_statistics(self):
        noise_mean = np.array([1.0, -1.0])
        noise_covariance = np.array([[2.0, 0.5], [0.5, 3.0]])
        model = AdditiveNoiseTransitionModel(
            lambda state: state,
            noise_mean=noise_mean,
            noise_covariance=noise_covariance,
        )

        noise_mean[:] = 99.0
        noise_covariance[:] = -99.0

        assert_backend_allclose(self, model.noise_mean, array([1.0, -1.0]))
        assert_backend_allclose(
            self,
            model.noise_covariance,
            array([[2.0, 0.5], [0.5, 3.0]]),
        )

    def test_measurement_model_copies_explicit_noise_statistics(self):
        noise_mean = np.array([0.25])
        noise_covariance = np.array([[4.0]])
        model = AdditiveNoiseMeasurementModel(
            lambda state: state,
            noise_mean=noise_mean,
            noise_covariance=noise_covariance,
        )

        noise_mean[:] = 99.0
        noise_covariance[:] = -99.0

        assert_backend_allclose(self, model.noise_mean, array([0.25]))
        assert_backend_allclose(self, model.noise_covariance, array([[4.0]]))


if __name__ == "__main__":
    unittest.main()
