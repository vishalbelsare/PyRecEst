"""Regression tests for PyTorch linalg.logm array-like input promotion."""

from __future__ import annotations

import pytest

import pyrecest.backend as backend
from pyrecest.backend import linalg as public_linalg

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")
pytorch_linalg = pytest.importorskip("pyrecest._backend.pytorch.linalg")


def test_raw_pytorch_logm_accepts_integer_python_lists():
    result = pytorch_linalg.logm([[1, 0], [0, 1]])

    assert pytorch_backend.is_floating(result)
    assert bool(pytorch_backend.allclose(result, pytorch_backend.zeros((2, 2))))


def test_public_pytorch_logm_accepts_array_like_inputs_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    result = public_linalg.logm([[1, 0], [0, 1]])

    assert backend.is_floating(result)
    assert bool(backend.allclose(result, backend.zeros((2, 2))))
