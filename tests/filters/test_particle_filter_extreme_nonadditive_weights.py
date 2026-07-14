from unittest import mock

import numpy as np
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


def test_nonadditive_prediction_normalizes_extreme_finite_noise_weights():
    particles = array([[10.0], [20.0]])
    samples = array([[1.0], [2.0]])
    backend_dtype = to_numpy(array([1.0])).dtype
    max_weight = np.finfo(backend_dtype).max
    weights = array([max_weight, max_weight / 2.0])
    particle_filter = EuclideanParticleFilter(n_particles=2, dim=1)
    particle_filter.filter_state = LinearDiracDistribution(particles)

    with mock.patch(
        "pyrecest.filters.abstract_particle_filter.random.choice",
        return_value=samples,
    ) as choice_mock:
        particle_filter.predict_nonlinear_nonadditive(
            lambda particle, noise: particle + noise,
            samples,
            weights,
        )

    normalized_weights = choice_mock.call_args.kwargs["p"]
    npt.assert_allclose(to_numpy(normalized_weights), [2.0 / 3.0, 1.0 / 3.0])
    npt.assert_allclose(to_numpy(particle_filter.filter_state.d), [[11.0], [22.0]])
