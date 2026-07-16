"""Regression tests for finite positive-definite matrix validation."""

from __future__ import annotations

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_numpy_is_single_matrix_pd_rejects_nonfinite_matrices():
    result = run_backend_code(
        "numpy",
        """
import pyrecest.backend as backend

for value in (float("nan"), float("inf"), float("-inf")):
    matrix = backend.array([[value]])
    assert bool(backend.linalg.is_single_matrix_pd(matrix)) is False
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
