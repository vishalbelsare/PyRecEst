import numpy as np
import pytest


@pytest.mark.parametrize("function_name", ["fftshift", "ifftshift"])
def test_jax_fft_shift_accepts_integer_scalar_array_axes(function_name):
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    values = jnp.arange(12.0).reshape(3, 4)
    shift = getattr(fft, function_name)
    expected = shift(values, axes=1)

    for axes in (np.int64(1), np.array(1), jnp.array(1)):
        actual = shift(values, axes=axes)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("function_name", ["fftshift", "ifftshift"])
def test_jax_fft_shift_rejects_boolean_axes(function_name):
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    values = jnp.arange(4.0)
    shift = getattr(fft, function_name)

    for axes in (True, np.bool_(True), np.array(True), jnp.array(True)):
        with pytest.raises(TypeError, match="not boolean"):
            shift(values, axes=axes)
