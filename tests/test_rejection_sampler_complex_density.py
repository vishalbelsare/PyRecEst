import unittest

import numpy as np

from pyrecest.sampling import FibonacciRejectionSampler


class FibonacciRejectionSamplerComplexDensityTest(unittest.TestCase):
    def test_rejects_complex_density_values(self):
        sampler = FibonacciRejectionSampler()

        with self.assertRaisesRegex(ValueError, "real density values"):
            sampler.sample_rejection(
                lambda samples: np.full(samples.shape[0], 0.5 + 1.0j),
                n_candidates=8,
                dim=1,
                max_density=1.0,
            )

    def test_real_density_values_remain_supported(self):
        sampler = FibonacciRejectionSampler()

        samples, info = sampler.sample_rejection(
            lambda candidates: np.full(candidates.shape[0], 0.5),
            n_candidates=8,
            dim=1,
            max_density=1.0,
        )

        self.assertEqual(samples.ndim, 2)
        self.assertEqual(samples.shape[1], 1)
        self.assertEqual(info["n_accepted"], samples.shape[0])


if __name__ == "__main__":
    unittest.main()
