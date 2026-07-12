import unittest

from pyrecest.backend import allclose, array, ndim, pi
from pyrecest.distributions import WrappedNormalDistribution


class WrappedNormalLargeSigmaScalarPdfTest(unittest.TestCase):
    def test_large_sigma_scalar_pdf_returns_scalar(self):
        dist = WrappedNormalDistribution(array(0.0), array(100.0))

        value = dist.pdf(array(0.25))

        self.assertEqual(ndim(value), 0)
        self.assertTrue(allclose(value, 1.0 / (2.0 * pi), rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
