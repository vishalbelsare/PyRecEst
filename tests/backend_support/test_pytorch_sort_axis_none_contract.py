import pytest

import pyrecest.backend as backend

pytestmark = pytest.mark.backend_portable


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_sort_axis_none_matches_numpy_flattening():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific sort regression")

    values = backend.asarray([[3, 1, 2], [6, 4, 5]])

    assert _to_python(backend.sort(values, axis=None)) == [1, 2, 3, 4, 5, 6]
    assert _to_python(backend.sort(values, axis=1)) == [[1, 2, 3], [4, 5, 6]]
