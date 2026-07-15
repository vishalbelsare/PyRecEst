import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, eye, sin, stack, to_numpy
from pyrecest.smoothers import SO3ChordalMeanSmoother


def _z_rotation(angle):
    return array(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_chordal_mean_normalizes_extreme_finite_weights_stably():
    backend_dtype = to_numpy(array([1.0])).dtype
    maximum = np.finfo(backend_dtype).max
    rotations = stack([eye(3), _z_rotation(0.5 * np.pi)], axis=0)

    reference = SO3ChordalMeanSmoother.chordal_mean(
        rotations,
        weights=array([3.0, 1.0]),
    )
    extreme_weights = array(
        np.asarray([maximum, maximum / 3.0], dtype=backend_dtype)
    )

    result = SO3ChordalMeanSmoother.chordal_mean(
        rotations,
        weights=extreme_weights,
    )

    npt.assert_allclose(
        to_numpy(result),
        to_numpy(reference),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
