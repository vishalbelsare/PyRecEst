import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, diag, eye
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-RBPF tracker tests currently use numpy.testing assertions",
)
class MEMRBPFControlInputValidationTest(unittest.TestCase):
    @staticmethod
    def _tracker():
        return MEMRBPFTracker(
            kinematic_state=array([0.0, 0.0, 1.0, -0.5]),
            covariance=eye(4),
            shape_state=array([0.2, 2.0, 1.0]),
            shape_covariance=diag(array([0.05, 0.1, 0.1])),
            meas_noise_cov=0.05 * eye(2),
            sys_noise=0.01 * eye(4),
            shape_sys_noise=diag(array([0.01, 0.01, 0.01])),
            n_particles=8,
            resampling_threshold=0,
            rng=0,
        )

    def test_rejects_broadcastable_wrong_sized_input_without_mutating_state(self):
        tracker = self._tracker()
        prior_state = tracker.kinematic_state.copy()
        prior_covariance = tracker.covariance.copy()
        prior_system_matrix = tracker.system_matrix.copy()
        prior_sys_noise = tracker.sys_noise.copy()
        prior_theta = tracker.theta.copy()
        prior_axis = tracker.axis.copy()
        prior_axis_covariances = tracker.axis_covariances.copy()

        with self.assertRaisesRegex(ValueError, r"inputs must have shape \(4,\)"):
            tracker.predict_linear(
                system_matrix=2.0 * eye(4),
                sys_noise=3.0 * eye(4),
                inputs=array([2.0]),
            )

        npt.assert_array_equal(tracker.kinematic_state, prior_state)
        npt.assert_array_equal(tracker.covariance, prior_covariance)
        npt.assert_array_equal(tracker.system_matrix, prior_system_matrix)
        npt.assert_array_equal(tracker.sys_noise, prior_sys_noise)
        npt.assert_array_equal(tracker.theta, prior_theta)
        npt.assert_array_equal(tracker.axis, prior_axis)
        npt.assert_array_equal(tracker.axis_covariances, prior_axis_covariances)


if __name__ == "__main__":
    unittest.main()
