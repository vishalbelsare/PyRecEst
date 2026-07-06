import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_public_pytorch_cross_accepts_plain_two_dimensional_vectors():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest.backend as backend

result = backend.cross([1.0, 0.0], [0.0, 1.0])
assert result.shape == ()
assert float(backend.to_numpy(result)) == 1.0

batched = backend.cross(
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
)
assert backend.to_numpy(batched).tolist() == [1.0, -1.0]
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_cross_accepts_plain_two_dimensional_vectors_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch

result = raw_pytorch.cross([1.0, 0.0], [0.0, 1.0])
assert result.shape == ()
assert float(raw_pytorch.to_numpy(result)) == 1.0

batched = raw_pytorch.cross(
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
)
assert raw_pytorch.to_numpy(batched).tolist() == [1.0, -1.0]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
