import unittest

import pyrecest.backend
from pyrecest.backend import random


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    "NumPy-specific probability normalization regression",
)
class TestNumpyRandomProbabilityNormalization(unittest.TestCase):
    def test_multinomial_accepts_large_finite_unnormalized_pvals(self):
        sample = random.multinomial(12, [1.0e308, 1.0e308, 1.0e308])

        self.assertEqual(tuple(pyrecest.backend.shape(sample)), (3,))
        self.assertEqual(int(pyrecest.backend.sum(sample)), 12)


if __name__ == "__main__":
    unittest.main()
