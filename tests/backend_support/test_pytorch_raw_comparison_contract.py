"""Regression tests for raw PyTorch backend comparison helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_raw_pytorch_comparisons_accept_numpy_style_array_like_inputs(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        backend_name,
        """
import importlib

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")

right = raw_pytorch.asarray([0, 2, 4])
greater_result = raw_pytorch.greater([1, 2, 3], right)
less_result = raw_pytorch.less(right, [1, 2, 3])
equal_result = raw_pytorch.equal([1, 2, 3], right)
less_equal_result = raw_pytorch.less_equal([1, 2, 3], right)
logical_result = raw_pytorch.logical_or([True, False], raw_pytorch.asarray([False, True]))

assert raw_pytorch.to_numpy(greater_result).tolist() == [True, False, False]
assert raw_pytorch.to_numpy(less_result).tolist() == [True, False, False]
assert raw_pytorch.to_numpy(equal_result).tolist() == [False, True, False]
assert raw_pytorch.to_numpy(less_equal_result).tolist() == [False, True, True]
assert raw_pytorch.to_numpy(logical_result).tolist() == [True, True]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
@pytest.mark.parametrize("helper_name", ["equal", "less_equal"])
def test_raw_pytorch_equality_comparisons_preserve_non_cpu_device(helper_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        f"""
import importlib
import torch

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")
helper = getattr(raw_pytorch, {helper_name!r})
right = torch.empty((3,), device="meta", dtype=torch.float32)

comparison = helper([1.0, 2.0, 3.0], right)
assert tuple(comparison.shape) == (3,)
assert comparison.dtype == torch.bool
assert comparison.device.type == "meta"
""",
    )

    assert result.returncode == 0, result.stderr
