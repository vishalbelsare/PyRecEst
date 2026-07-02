import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_jax_set_default_dtype_sets_and_returns_dtype():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
result = backend.set_default_dtype('float32')
assert result == backend.as_dtype('float32')
result = backend.set_default_dtype('float64')
assert result == backend.as_dtype('float64')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )


@pytest.mark.backend_portable
def test_jax_as_dtype_accepts_numeric_aliases():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import numpy as np
import jax.numpy as jnp
import pyrecest.backend as backend

assert backend.as_dtype('uint8') == jnp.uint8
assert backend.as_dtype(np.dtype('int32')) == jnp.int32
assert backend.as_dtype('float16') == jnp.float16
"""
    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
