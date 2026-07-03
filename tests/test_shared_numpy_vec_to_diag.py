import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_shared_numpy_vec_to_diag_accepts_array_like_inputs():
    if backend.__backend_name__ not in {"numpy", "autograd"}:
        pytest.skip("shared NumPy/Autograd vec_to_diag regression test")

    result = backend.vec_to_diag([1, 2, 3])

    assert result.shape == (3, 3)
    assert _to_python(result) == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


def test_shared_numpy_vec_to_diag_accepts_batched_array_like_inputs():
    if backend.__backend_name__ not in {"numpy", "autograd"}:
        pytest.skip("shared NumPy/Autograd vec_to_diag regression test")

    result = backend.vec_to_diag([[1, 2], [3, 4]])

    assert result.shape == (2, 2, 2)
    assert _to_python(result) == [
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]],
    ]
