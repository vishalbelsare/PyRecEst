import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, zeros_like
from pyrecest.distributions import AbstractHypertoroidalDistribution, HypertoroidalDiracDistribution


def test_set_mean_accepts_matching_vector_and_preserves_original():
    samples = array([[0.5, 2.0, 0.5], [3.0, 2.0, 0.2], [4.0, 5.0, 5.8]])
    weights = array([0.2, 0.3, 0.5])
    dist = HypertoroidalDiracDistribution(samples, weights)
    target_mean = array([0.25, 1.0, 2.5])

    shifted = dist.set_mean(target_mean)

    npt.assert_allclose(
        AbstractHypertoroidalDistribution.angular_error(
            shifted.mean_direction(), target_mean
        ),
        zeros_like(target_mean),
        atol=1e-6,
    )
    npt.assert_allclose(dist.d, samples)


def test_set_mean_rejects_broadcasting_mean_shapes_for_multidimensional_distribution():
    dist = HypertoroidalDiracDistribution(
        array([[0.5, 2.0, 0.5], [3.0, 2.0, 0.2], [4.0, 5.0, 5.8]]),
        array([0.2, 0.3, 0.5]),
    )

    for malformed_mean in (0.25, [0.25], [0.25, 1.0], [[0.25, 1.0, 2.5]]):
        try:
            dist.set_mean(malformed_mean)
        except ValueError as exc:
            assert "mean must have shape (3,)" in str(exc)
        else:  # pragma: no cover - documents the regression this test guards
            raise AssertionError("malformed mean was accepted")


def test_set_mean_accepts_scalar_for_one_dimensional_distribution():
    dist = HypertoroidalDiracDistribution([0.1, 0.2, 0.3], dim=1)

    shifted = dist.set_mean(0.25)

    npt.assert_allclose(shifted.mean_direction(), array([0.25]), atol=1e-6)
