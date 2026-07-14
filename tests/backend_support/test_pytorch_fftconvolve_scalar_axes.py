import numpy as np
import pyrecest.backend as backend
import pytest
from scipy.signal import fftconvolve as scipy_fftconvolve


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("axes", [0, -1, [0], np.array(0)])
def test_pytorch_fftconvolve_scalar_axes_match_scipy(mode, axes):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray(2.0)
    second = backend.asarray(3.0)

    actual = backend.to_numpy(
        backend.signal.fftconvolve(first, second, mode=mode, axes=axes)
    )
    expected = scipy_fftconvolve(
        backend.to_numpy(first), backend.to_numpy(second), mode=mode, axes=axes
    )

    assert actual.shape == expected.shape == ()
    assert np.allclose(actual, expected)
