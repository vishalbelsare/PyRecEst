import importlib.util
import os
import subprocess
import sys

import pytest


def _pytorch_env(backend_name):
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
def test_pytorch_split_rejects_non_integer_cut_points():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


def assert_type_error(call, label):
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError(f"{label} accepted non-integer split cut points")


for split_func in (backend.split, raw_pytorch.split):
    for bad_indices in (
        [1.5],
        np.asarray([1.5]),
        ["2"],
        np.asarray(["2"]),
        np.asarray([True]),
    ):
        assert_type_error(
            lambda split_func=split_func, bad_indices=bad_indices: split_func(
                [0, 1, 2, 3], bad_indices
            ),
            split_func.__module__,
        )

    integer_parts = split_func([0, 1, 2, 3], np.asarray([1, 3]))
    assert [raw_pytorch.to_numpy(part).tolist() for part in integer_parts] == [
        [0],
        [1, 2],
        [3],
    ]

    bool_list_parts = split_func([0, 1, 2, 3], [True])
    assert [raw_pytorch.to_numpy(part).tolist() for part in bool_list_parts] == [
        [0],
        [1, 2, 3],
    ]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_pytorch_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_split_is_patched_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert backend.__backend_name__ == "numpy"

try:
    raw_pytorch.split([0, 1, 2, 3], [1.5])
except TypeError:
    pass
else:
    raise AssertionError("raw PyTorch split accepted fractional cut points")

parts = raw_pytorch.split([0, 1, 2, 3], [1, 3])
assert [raw_pytorch.to_numpy(part).tolist() for part in parts] == [
    [0],
    [1, 2],
    [3],
]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_pytorch_env("numpy"),
    )
