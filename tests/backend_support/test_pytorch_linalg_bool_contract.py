import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_complex_positive_definite_predicate_returns_python_bool():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

matrix = backend.array(
    [[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]]
)
value = backend.linalg.is_single_matrix_pd(matrix)
assert isinstance(value, bool), type(value)
assert value is True
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_pytorch_linalg_norm_rejects_bool_axis_entries():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

matrix = backend.array([[1.0, 2.0], [3.0, 4.0]])

for axis_value in (False, True, [False], (True,), [np.bool_(False)]):
    try:
        backend.linalg.norm(matrix, axis=axis_value)
    except TypeError:
        pass
    else:
        raise AssertionError(f"norm accepted boolean axis {axis_value!r}")

assert backend.to_numpy(backend.linalg.norm(matrix, axis=[0, 1])).shape == ()
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
