"""Regression tests for PyTorch ``logical_and`` device selection."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_logical_and_prefers_existing_non_cpu_device(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        backend_name,
        """
import importlib

import torch

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")

left = torch.tensor([True, False])
right = torch.empty((2,), dtype=torch.bool, device="meta")
raw_result = raw_pytorch.logical_and(left, right)

assert raw_result.device.type == "meta"
assert tuple(raw_result.shape) == (2,)
assert raw_result.dtype == torch.bool

backend = importlib.import_module("pyrecest.backend")
if getattr(backend, "__backend_name__", None) == "pytorch":
    public_result = backend.logical_and(left, right)
    assert public_result.device.type == "meta"
    assert tuple(public_result.shape) == (2,)
    assert public_result.dtype == torch.bool
""",
    )

    assert result.returncode == 0, result.stderr
