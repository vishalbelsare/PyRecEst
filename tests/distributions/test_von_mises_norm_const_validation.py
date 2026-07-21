import math
import unittest

from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution


class VonMisesNormalizationConstantValidationTest(unittest.TestCase):
    def test_constructor_rejects_nonpositive_or_nonfinite_norm_const(self):
        for norm_const in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(norm_const=norm_const):
                with self.assertRaisesRegex(ValueError, "norm_const"):
                    VonMisesDistribution(0.0, 1.0, norm_const=norm_const)

    def test_constructor_accepts_positive_finite_norm_const(self):
        distribution = VonMisesDistribution(0.0, 1.0, norm_const=array(2.0))

        self.assertEqual(distribution.norm_const, 2.0)
        self.assertAlmostEqual(float(distribution.pdf(array(0.0))), math.e / 2.0)


if __name__ == "__main__":
    unittest.main()
