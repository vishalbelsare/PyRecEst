import unittest

from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.smoothers import RauchTungStriebelSmoother


class SmootherMatrixShapeValidationTest(unittest.TestCase):
    def test_rejects_broadcastable_measurement_noise_covariance(self):
        smoother = RauchTungStriebelSmoother()

        with self.assertRaisesRegex(
            ValueError,
            r"meas_noise_covariances.*\(2, 2\)",
        ):
            smoother.filter(
                initial_state=GaussianDistribution(zeros(2), eye(2)),
                measurements=[zeros(2)],
                measurement_matrices=eye(2),
                meas_noise_covariances=array([[1.0]]),
            )

    def test_rejects_broadcastable_process_noise_covariance(self):
        smoother = RauchTungStriebelSmoother()

        with self.assertRaisesRegex(
            ValueError,
            r"sys_noise_covariances.*\(2, 2\)",
        ):
            smoother.filter(
                initial_state=GaussianDistribution(zeros(2), eye(2)),
                measurements=[zeros(2), zeros(2)],
                measurement_matrices=eye(2),
                meas_noise_covariances=eye(2),
                system_matrices=eye(2),
                sys_noise_covariances=array([[1.0]]),
            )

    def test_rectangular_measurement_matrix_remains_supported(self):
        smoother = RauchTungStriebelSmoother()

        filtered_states, predicted_states = smoother.filter(
            initial_state=GaussianDistribution(zeros(2), eye(2)),
            measurements=array([1.0]),
            measurement_matrices=array([[1.0, 0.0]]),
            meas_noise_covariances=array([[1.0]]),
        )

        self.assertEqual(len(filtered_states), 1)
        self.assertEqual(len(predicted_states), 0)
        self.assertEqual(filtered_states[0].mu.shape, (2,))
        self.assertEqual(filtered_states[0].C.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
