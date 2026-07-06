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
@pytest.mark.parametrize("backend_name", ["numpy", "autograd"])
def test_shared_numpy_triangular_helpers_accept_rectangular_array_like_inputs(
    backend_name,
):
    if backend_name == "autograd" and importlib.util.find_spec("autograd") is None:
        pytest.skip("autograd is not installed")

    code = """
import pyrecest.backend as backend

wide = [[1, 2, 3], [4, 5, 6]]
tall = [[1, 2], [3, 4], [5, 6]]

lower_wide = backend.tril_to_vec(wide)
assert backend.to_numpy(lower_wide).tolist() == [1, 4, 5]

upper_wide = backend.triu_to_vec(wide, k=1)
assert backend.to_numpy(upper_wide).tolist() == [2, 3, 6]

lower_tall = backend.tril_to_vec(tall, k=-1)
assert backend.to_numpy(lower_tall).tolist() == [3, 5, 6]

upper_tall = backend.triu_to_vec(tall)
assert backend.to_numpy(upper_tall).tolist() == [1, 2, 4]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env(backend_name),
    )
