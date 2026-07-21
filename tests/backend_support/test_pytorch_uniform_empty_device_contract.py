import pytest


def test_empty_uniform_prefers_existing_non_cpu_bound_device(monkeypatch):
    torch = pytest.importorskip("torch")
    import pyrecest._backend.pytorch.random as raw_random
    from pyrecest.backend_support._random_uniform_empty_contract import (
        _patch_pytorch_uniform_empty_bounds_contract,
    )

    def reject_descending_bounds(*args, **kwargs):
        raise ValueError("Upper bound must be greater than or equal to lower bound")

    monkeypatch.setattr(raw_random, "uniform", reject_descending_bounds)
    _patch_pytorch_uniform_empty_bounds_contract()

    low = torch.tensor(1.0)
    high = torch.tensor(0.0, device="meta")
    result = raw_random.uniform(low, high, size=(0,))

    assert result.shape == (0,)
    assert result.device.type == "meta"
