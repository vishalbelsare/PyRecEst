import importlib.util
import os
import subprocess
import sys

import pyrecest
import pytest


def test_backend_tools_report_active_backend():
    active = pyrecest.get_backend_name()

    assert pyrecest.is_backend(active)
    pyrecest.assert_backend(active)


def test_assert_backend_rejects_unexpected_backend():
    active = pyrecest.get_backend_name()
    unexpected = "jax" if active != "jax" else "numpy"

    with pytest.raises(RuntimeError):
        pyrecest.assert_backend(unexpected)


def test_assert_backend_deduplicates_expected_names_in_error_message():
    active = pyrecest.get_backend_name()
    unexpected = "jax" if active != "jax" else "numpy"

    with pytest.raises(RuntimeError) as excinfo:
        pyrecest.assert_backend((unexpected, unexpected))

    assert str(excinfo.value).count(unexpected) == 1


@pytest.mark.parametrize(
    "expected", [(), ("",), " ", (" numpy",), ("numpy ",), ("numpy", 1), 1]
)
def test_assert_backend_rejects_invalid_expected_names(expected):
    with pytest.raises(ValueError, match="expected"):
        pyrecest.assert_backend(expected)


def test_warn_if_backend_env_changed(monkeypatch):
    active = pyrecest.get_backend_name()
    changed = "jax" if active != "jax" else "numpy"
    monkeypatch.setenv("PYRECEST_BACKEND", changed)

    with pytest.warns(RuntimeWarning):
        pyrecest.warn_if_backend_env_changed()

    monkeypatch.setenv("PYRECEST_BACKEND", active)
    pyrecest.warn_if_backend_env_changed()


@pytest.mark.backend_portable
def test_raw_pytorch_diag_accepts_arraylike_when_public_backend_is_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import torch
import pyrecest
import pyrecest._backend.pytorch as raw_pytorch

result = raw_pytorch.diag([1, 2, 3], k=1)
expected = torch.diag(torch.tensor([1, 2, 3]), diagonal=1)
assert torch.equal(result, expected)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
