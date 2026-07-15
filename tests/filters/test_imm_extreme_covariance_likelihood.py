import numpy as np
import numpy.testing as npt
import pytest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="IMM likelihood evaluation is currently supported on the NumPy backend",
)


@pytest.mark.parametrize("scale", [1.0e-200, 1.0e200])
def test_linear_likelihood_stays_finite_for_extreme_covariance_scale(scale):
    covariance = scale * eye(2)
    predicted_state = GaussianDistribution(
        array([0.0, 0.0]), covariance, check_validity=False
    )

    log_likelihood = (
        InteractingMultipleModelFilter._log_linear_measurement_likelihood(
            array([0.0, 0.0]),
            predicted_state,
            eye(2),
            array([[0.0, 0.0], [0.0, 0.0]]),
        )
    )

    determinant_sign, expected_logdet = np.linalg.slogdet(np.asarray(covariance))
    assert determinant_sign > 0.0
    expected = -0.5 * (2.0 * np.log(2.0 * np.pi) + expected_logdet)
    assert np.isfinite(log_likelihood)
    npt.assert_allclose(
        log_likelihood, expected, rtol=1.0e-14, atol=1.0e-14
    )
