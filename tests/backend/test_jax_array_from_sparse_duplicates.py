import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import array_from_sparse  # noqa: E402


def test_array_from_sparse_uses_last_value_for_duplicate_indices():
    indices = jnp.array([[0, 1], [0, 1], [1, 0], [0, 1]])
    data = jnp.array([2, 3, 5, 7])

    dense = array_from_sparse(indices, data, (2, 2))

    expected = jnp.array([[0, 7], [5, 0]])
    assert jnp.array_equal(dense, expected)
