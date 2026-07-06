import pytest


def test_pytorch_choice_rejects_nonpositive_integer_population_for_empty_draw():
    torch = pytest.importorskip("torch")
    import pyrecest  # noqa: F401  # trigger backend support patches
    from pyrecest._backend.pytorch import random

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(0, size=0)

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(torch.tensor(0), size=(0,))


def test_pytorch_choice_rejects_empty_array_population_for_empty_draw():
    torch = pytest.importorskip("torch")
    import pyrecest  # noqa: F401  # trigger backend support patches
    from pyrecest._backend.pytorch import random

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(torch.empty((0, 3)), size=0, axis=0)

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(torch.empty((3, 0)), size=(0,), axis=1)


def test_jax_choice_rejects_nonpositive_integer_population_for_empty_draw():
    jnp = pytest.importorskip("jax.numpy")
    import pyrecest  # noqa: F401  # trigger backend support patches
    from pyrecest._backend.jax import random

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(0, size=0)

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(jnp.asarray(0), size=(0,))


def test_jax_choice_rejects_empty_array_population_for_empty_draw():
    jnp = pytest.importorskip("jax.numpy")
    import pyrecest  # noqa: F401  # trigger backend support patches
    from pyrecest._backend.jax import random

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(jnp.empty((0, 3)), size=0, axis=0)

    with pytest.raises(ValueError, match="positive integer|non-empty"):
        random.choice(jnp.empty((3, 0)), size=(0,), axis=1)
