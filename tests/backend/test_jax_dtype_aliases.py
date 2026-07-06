import pytest


def test_jax_as_dtype_accepts_bfloat16_alias():
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax._dtype import as_dtype

    assert as_dtype("bfloat16") == jnp.bfloat16
    assert as_dtype(jnp.bfloat16) == jnp.bfloat16
