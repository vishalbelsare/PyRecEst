import pytest

import pyrecest.backend as backend
import pyrecest.evidence  # noqa: F401 ensure runtime backend patches are installed

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


def _as_list(value):
    return pytorch_backend.to_numpy(value).tolist()


def test_raw_pytorch_concatenate_axis_none_flattens_inputs():
    first = pytorch_backend.array([[1, 2], [3, 4]])
    second = pytorch_backend.array([[5], [6]])

    result = pytorch_backend.concatenate((first, second), axis=None)

    assert result.shape == (6,)
    assert _as_list(result) == [1, 2, 3, 4, 5, 6]


def test_public_pytorch_concatenate_axis_none_flattens_inputs_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    result = backend.concatenate(([[1, 2], [3, 4]], [[5], [6]]), axis=None)

    assert result.shape == (6,)
    assert backend.to_numpy(result).tolist() == [1, 2, 3, 4, 5, 6]
