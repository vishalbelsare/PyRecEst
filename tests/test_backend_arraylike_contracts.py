import pyrecest.backend as backend
import pytest
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_vec_to_diag_accepts_array_like_inputs():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific vec_to_diag array-like regression test")

    result = backend.vec_to_diag([1, 2, 3])

    assert result.shape == (3, 3)
    assert _to_python(result) == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


def test_squeeze_non_singleton_axis_is_noop():
    values = array([[[1], [2]]])

    result = backend.squeeze(values, axis=1)

    assert result.shape == values.shape
    assert _to_python(result) == [[[1], [2]]]
