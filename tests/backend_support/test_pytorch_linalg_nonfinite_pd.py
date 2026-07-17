"""Regression tests for nonfinite PyTorch positive-definite inputs."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_is_single_matrix_pd_rejects_nonfinite_entries():
    """Nonfinite matrices must not enter the positive-definite cone."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

matrices = [
    backend.array([[float("nan")]]),
    backend.array([[float("inf")]]),
    backend.array([[float("-inf")]]),
    backend.array([[complex(float("inf"), 0.0)]]),
]
for matrix in matrices:
    assert bool(backend.linalg.is_single_matrix_pd(matrix)) is False
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
