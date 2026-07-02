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
def test_raw_pytorch_round_accepts_array_like_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

rounded = raw_backend.round([1.2, 2.7])
assert raw_backend.to_numpy(rounded).tolist() == [1.0, 3.0]

rounded_decimals = raw_backend.round([1.24, 2.76], decimals=1)
assert raw_backend.to_numpy(rounded_decimals).tolist() == [1.2, 2.8]

out = raw_backend.empty(2, dtype=raw_backend.float64)
returned = raw_backend.round([1.2, 2.7], out=out)
assert returned is out
assert raw_backend.to_numpy(out).tolist() == [1.0, 3.0]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
