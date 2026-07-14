import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.smoothers import SlidingWindowManifoldMeanSmoother


class SlidingWindowWeightStabilityTest(unittest.TestCase):
    def test_rejects_nonfinite_window_weights(self):
        for invalid_weight in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid_weight=invalid_weight), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                SlidingWindowManifoldMeanSmoother(
                    window_size=2,
                    window_weights=array([1.0, invalid_weight]),
                )

    def test_extreme_finite_window_weights_preserve_relative_mass(self):
        backend_dtype = to_numpy(array([1.0])).dtype
        max_finite = np.finfo(backend_dtype).max
        smoother = SlidingWindowManifoldMeanSmoother(
            window_size=2,
            window_weights=array([max_finite, max_finite / 2.0]),
            alignment="trailing",
        )

        smoothed = smoother.smooth([array([0.0]), array([3.0])])

        npt.assert_allclose(to_numpy(smoothed[0]), [0.0])
        npt.assert_allclose(to_numpy(smoothed[1]), [1.0])
        npt.assert_allclose(
            to_numpy(smoother._weights_for_window(0, 2)),
            [2.0 / 3.0, 1.0 / 3.0],
        )


if __name__ == "__main__":
    unittest.main()
