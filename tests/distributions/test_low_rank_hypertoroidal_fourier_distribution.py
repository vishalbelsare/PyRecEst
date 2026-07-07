import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankHypertoroidalFourierDistribution(unittest.TestCase):
    def test_uniform_identity_normalization(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform((3, 3, 3))
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-12)
        npt.assert_allclose(
            dist.coefficient_at_zero(), 1.0 / (2.0 * np.pi) ** 3, atol=1e-12
        )

    def test_high_dimensional_uniform_smoke(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform((3,) * 8)
        self.assertEqual(dist.coeff_shape, (3,) * 8)
        self.assertEqual(dist.tt_ranks, (1,) * 9)
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-12)
        self.assertTrue(np.isfinite(dist.pdf(np.zeros(8))))


if __name__ == "__main__":
    unittest.main()
