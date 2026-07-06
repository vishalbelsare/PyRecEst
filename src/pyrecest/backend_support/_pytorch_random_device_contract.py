"""PyTorch random-helper device compatibility hook."""

from __future__ import annotations

import importlib
from functools import wraps


def _raw_pytorch_random_module():
    """Return the raw PyTorch random backend module when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch.random")
    except ModuleNotFoundError:
        return None


def _active_public_random_module():
    """Return the public PyTorch random facade when it is already active."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import failed earlier
        return None
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return None
    return getattr(backend, "random", None)


def _preferred_tensor_device(torch_module, *values, device=None):
    """Return an existing non-CPU tensor device, falling back to any tensor."""
    if device is not None:
        return device
    for value in values:
        if torch_module.is_tensor(value) and value.device.type == "meta":
            return value.device
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def patch_pytorch_random_device_contract() -> None:
    """Patch PyTorch random helpers to preserve existing non-CPU devices."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_random = _raw_pytorch_random_module()
    if raw_random is None:  # pragma: no cover - backend import failed earlier
        return

    public_random = _active_public_random_module()
    if getattr(raw_random, "_pyrecest_random_device_contract", False):
        if public_random is not None:
            public_random.uniform = raw_random.uniform
        return

    def preferred_tensor_device(*values, device=None):
        return _preferred_tensor_device(torch, *values, device=device)

    def randint_device(*values, device=None):
        return preferred_tensor_device(*values, device=device)

    original_uniform = getattr(raw_random, "uniform", None)

    if original_uniform is not None:

        @wraps(original_uniform)
        def uniform(low=0.0, high=1.0, size=None, dtype=None):
            dtype = raw_random._normalize_random_dtype(  # pylint: disable=protected-access
                dtype,
                default=None,
            )
            device = preferred_tensor_device(low, high)
            low = raw_random._validate_uniform_bound(  # pylint: disable=protected-access
                low,
                "low",
                dtype=dtype,
                device=device,
            )
            high = raw_random._validate_uniform_bound(  # pylint: disable=protected-access
                high,
                "high",
                dtype=dtype,
                device=device,
            )
            size = raw_random._uniform_size(  # pylint: disable=protected-access
                size,
                low,
                high,
            )
            raw_random._validate_uniform_bounds(low, high)  # pylint: disable=protected-access
            return (high - low) * torch.rand(size, dtype=dtype, device=device) + low

        uniform._pyrecest_device_contract = True
        raw_random.uniform = uniform
        if public_random is not None:
            public_random.uniform = uniform

    preferred_tensor_device._pyrecest_device_contract = True
    randint_device._pyrecest_device_contract = True
    raw_random._preferred_tensor_device = preferred_tensor_device  # pylint: disable=protected-access
    raw_random._randint_device = randint_device  # pylint: disable=protected-access
    raw_random._normal_device = preferred_tensor_device  # pylint: disable=protected-access
    raw_random._tensor_device = preferred_tensor_device  # pylint: disable=protected-access
    raw_random._pyrecest_random_device_contract = True  # pylint: disable=protected-access
