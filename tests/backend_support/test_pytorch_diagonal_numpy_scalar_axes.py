from tests.support.backend_runner import run_backend_code


def test_pytorch_diagonal_accepts_numpy_scalar_integer_arguments():
    code = """
import numpy as np
import pyrecest.backend as backend

values = backend.asarray([[1, 2], [3, 4]])

offset_result = backend.diagonal(values, offset=np.array(0))
axis_result = backend.diagonal(
    values,
    axis1=np.array(0, dtype=np.int64),
    axis2=np.array(1, dtype=np.int64),
)

assert backend.to_numpy(offset_result).tolist() == [1, 4]
assert backend.to_numpy(axis_result).tolist() == [1, 4]
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
