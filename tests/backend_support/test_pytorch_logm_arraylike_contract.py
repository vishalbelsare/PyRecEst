"""Regression tests for PyTorch linalg.logm array-like promotion."""

from __future__ import annotations

import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


_LOGM_ARRAYLIKE_CONTRACT = """
import torch
import pyrecest  # noqa: F401  # registers backend compatibility hooks
import pyrecest.backend as backend
import pyrecest._backend.pytorch.linalg as raw_linalg
from pyrecest.backend import linalg


raw_result = raw_linalg.logm([[1, 0], [0, 1]])
assert raw_result.shape == (2, 2)
assert raw_result.dtype.is_floating_point
assert torch.allclose(raw_result, torch.zeros((2, 2), dtype=raw_result.dtype))

public_result = linalg.logm([[1, 0], [0, 1]])
assert public_result.shape == (2, 2)
assert public_result.dtype.is_floating_point
assert bool(backend.allclose(public_result, backend.zeros((2, 2), dtype=public_result.dtype)))

values = torch.eye(2, dtype=torch.float64, requires_grad=True)
grad_result = linalg.logm(values)
grad_result.sum().backward()
assert values.grad is not None
assert values.grad.shape == values.shape
assert bool(torch.isfinite(values.grad).all())

print("ok")
"""


def test_pytorch_logm_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _LOGM_ARRAYLIKE_CONTRACT)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
