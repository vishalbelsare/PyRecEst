"""PyTorch ``take`` axis-validation compatibility patch."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as _np


def _is_logical_axis(axis, torch_module) -> bool:
    """Return whether ``axis`` is a logical scalar, not an integer axis."""
    if isinstance(axis, (bool, _np.bool_)):
        return True
    if torch_module.is_tensor(axis):
        return axis.ndim == 0 and axis.dtype == torch_module.bool
    if isinstance(axis, _np.ndarray):
        return axis.shape == () and _np.issubdtype(axis.dtype, _np.bool_)
    return False


def _normalize_take_axis(axis, ndim, torch_module):
    """Normalize NumPy-style take axes without integer-like truthiness casts."""
    if axis is None:
        return None
    if _is_logical_axis(axis, torch_module):
        raise TypeError("an integer is required for the axis")
    try:
        axis = _operator_index(axis)
    except TypeError as exc:
        raise TypeError("an integer is required for the axis") from exc
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def patch_pytorch_take_axis_contract() -> None:
    """Make raw/public PyTorch ``take`` reject non-integer axes like NumPy."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_take = getattr(raw_pytorch, "take", None)
    if original_take is None:
        return
    if getattr(original_take, "_pyrecest_take_axis_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.take = original_take
        return

    def take(a, indices, axis=None, out=None, mode=None):
        values = raw_pytorch.array(a)
        normalized_axis = _normalize_take_axis(axis, values.ndim, torch)
        return original_take(
            values,
            indices,
            axis=normalized_axis,
            out=out,
            mode=mode,
        )

    take.__name__ = getattr(original_take, "__name__", "take")
    take.__doc__ = getattr(original_take, "__doc__", None)
    take._pyrecest_take_axis_contract = True
    raw_pytorch.take = take
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.take = take


__all__ = ["patch_pytorch_take_axis_contract"]
