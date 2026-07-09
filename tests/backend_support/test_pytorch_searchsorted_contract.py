import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_searchsorted_runtime_patch_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
import pyrecest.evidence  # noqa: F401 ensure runtime backend patches are installed
import pyrecest._backend.pytorch as raw_pytorch

left = raw_pytorch.searchsorted([1.0, 3.0, 5.0], [0.0, 3.0, 4.0])
assert raw_pytorch.to_numpy(left).tolist() == [0, 1, 2]

right = backend.searchsorted([1.0, 3.0, 5.0], [0.0, 3.0, 4.0], side="right")
assert backend.to_numpy(right).tolist() == [0, 2, 2]

out = backend.empty_like(right)
returned = backend.searchsorted([1.0, 3.0, 5.0], [0.0, 3.0, 4.0], out=out)
assert returned is out
assert backend.to_numpy(out).tolist() == [0, 1, 2]

sorter_result = backend.searchsorted([3.0, 1.0, 5.0], [0.0, 3.0, 4.0], sorter=[1, 0, 2])
assert backend.to_numpy(sorter_result).tolist() == [0, 1, 2]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
