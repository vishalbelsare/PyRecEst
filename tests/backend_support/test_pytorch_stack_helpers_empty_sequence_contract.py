import importlib.util
import os
import subprocess
import sys

import pytest


_EMPTY_STACK_HELPER_ERROR = "need at least one array to concatenate"


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
@pytest.mark.parametrize("backend_name", ("pytorch", "numpy"))
def test_pytorch_stack_helpers_empty_sequences_raise_numpy_value_error(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = f"""
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

helpers = (raw_backend,)
if getattr(backend, "__backend_name__", None) == "pytorch":
    helpers = (backend, raw_backend)

for stack_backend in helpers:
    for helper_name in ("hstack", "vstack", "column_stack", "dstack"):
        try:
            getattr(stack_backend, helper_name)([])
        except ValueError as exc:
            assert str(exc) == {_EMPTY_STACK_HELPER_ERROR!r}
        else:
            raise AssertionError(f"{{stack_backend.__name__}}.{{helper_name}}([]) did not raise")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env(backend_name)
    )
