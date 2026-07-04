import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _linspace_device_contract_code(target_module):
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}
right_endpoint = torch.empty((), device="meta")
result = target.linspace(torch.tensor(0.0), right_endpoint, num=3)
assert result.device.type == "meta"
assert tuple(result.shape) == (3,)
print("ok")
"""


def test_raw_pytorch_linspace_prefers_existing_non_cpu_stop_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _linspace_device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_linspace_prefers_existing_non_cpu_stop_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _linspace_device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
