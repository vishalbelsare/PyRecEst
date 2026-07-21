import unittest

import pyrecest.backend
from pyrecest.backend import allclose, array
from pyrecest.distributions import CircularUniformDistribution


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ == "numpy",
    "Numerical integration is only supported on the NumPy backend.",
)
class CircularCdfNumericalValidationTest(unittest.TestCase):
    def setUp(self):
        self.distribution = CircularUniformDistribution()

    def test_rejects_invalid_evaluation_points(self):
        invalid_inputs = [
            True,
            [0.1, float("nan")],
            [float("inf")],
            [float("-inf")],
            [1.0 + 0.5j],
            ["0.1"],
        ]

        for xs in invalid_inputs:
            with self.subTest(xs=xs):
                with self.assertRaisesRegex(
                    ValueError, "xs must contain finite real numeric values"
                ):
                    self.distribution.cdf_numerical(xs)

    def test_rejects_invalid_starting_points(self):
        invalid_inputs = [
            True,
            array([0.0]),
            array([0.0, 1.0]),
            float("nan"),
            float("inf"),
            float("-inf"),
            1.0 + 0.5j,
            "0.0",
        ]

        for starting_point in invalid_inputs:
            with self.subTest(starting_point=starting_point):
                with self.assertRaisesRegex(
                    ValueError, "starting_point must be a finite real scalar"
                ):
                    self.distribution.cdf_numerical([0.5], starting_point)

    def test_preserves_valid_scalar_and_vector_queries(self):
        xs = array([0.5, 1.0])
        starting_point = 0.25

        self.assertTrue(
            allclose(
                self.distribution.cdf_numerical(xs, starting_point),
                self.distribution.cdf(xs, starting_point),
                rtol=1e-10,
            )
        )


if __name__ == "__main__":
    unittest.main()
