import importlib.util

import numpy as np
import pytest

from tests.support.backend_runner import run_backend_code


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")


def test_pytorch_copy_returns_backend_tensors_for_array_like_inputs():
    _require_torch()

    import pyrecest  # noqa: F401
    import pyrecest._backend.pytorch as raw_pytorch

    scalar_copy = raw_pytorch.copy(1.5)
    assert raw_pytorch.is_array(scalar_copy)
    assert tuple(scalar_copy.shape) == ()
    assert float(scalar_copy) == 1.5

    sequence_copy = raw_pytorch.copy([[1.0, 2.0], [3.0, 4.0]])
    assert raw_pytorch.is_array(sequence_copy)
    assert sequence_copy.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    source = np.array([1.0, 2.0])
    array_copy = raw_pytorch.copy(source)
    source[0] = 99.0
    assert raw_pytorch.is_array(array_copy)
    assert array_copy.tolist() == [1.0, 2.0]


@pytest.mark.backend_portable
def test_top_level_copy_tracks_patched_pytorch_backend_copy():
    _require_torch()

    code = """
import pyrecest
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "pytorch"
assert pyrecest.copy is backend.copy
assert backend.copy is raw_pytorch.copy

copied = pyrecest.copy([[1.0, 2.0], [3.0, 4.0]])
assert backend.is_array(copied)
assert backend.to_numpy(copied).tolist() == [[1.0, 2.0], [3.0, 4.0]]
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
