import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_pytorch_arraylike_helpers_accept_numpy_style_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

for candidate in (backend, raw_backend):
    ones = candidate.ones_like([1.0, 2.0])
    assert torch.is_tensor(ones)
    assert candidate.to_numpy(ones).tolist() == [1.0, 1.0]

    zeros = candidate.zeros_like(np.array([1, 2], dtype=np.int64))
    assert torch.is_tensor(zeros)
    assert tuple(zeros.shape) == (2,)
    assert zeros.dtype == candidate.int64
    assert candidate.to_numpy(zeros).tolist() == [0, 0]

    full = candidate.full_like([1, 2], 7)
    assert torch.is_tensor(full)
    assert candidate.to_numpy(full).tolist() == [7, 7]

    empty = candidate.empty_like([1.0, 2.0])
    assert torch.is_tensor(empty)
    assert tuple(empty.shape) == (2,)

    order = candidate.argsort([3.0, 1.0, 2.0], axis=0)
    assert torch.is_tensor(order)
    assert candidate.to_numpy(order).tolist() == [1, 2, 0]

    assert candidate.to_numpy(candidate.isfinite([1.0, float("inf"), float("nan")])).tolist() == [True, False, False]
    assert candidate.to_numpy(candidate.isinf([1.0, float("inf")])).tolist() == [False, True]
    assert candidate.to_numpy(candidate.isnan([1.0, float("nan")])).tolist() == [False, True]
    assert candidate.to_numpy(candidate.isreal([1.0, 2.0])).tolist() == [True, True]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )


@pytest.mark.backend_portable
def test_raw_pytorch_arraylike_helpers_work_under_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

ones = raw_backend.ones_like([1.0, 2.0])
assert torch.is_tensor(ones)
assert raw_backend.to_numpy(ones).tolist() == [1.0, 1.0]

order = raw_backend.argsort([3.0, 1.0, 2.0], axis=0)
assert torch.is_tensor(order)
assert raw_backend.to_numpy(order).tolist() == [1, 2, 0]

assert raw_backend.to_numpy(raw_backend.isfinite([1.0, float("inf")])).tolist() == [True, False]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
