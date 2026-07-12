"""Runtime patch for integer-like tuple axes in shared PyTorch reductions."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as np

_AXIS_TYPE_MESSAGE = "axis must be an integer or a sequence of integers"


def _is_boolean_axis(value, torch_module) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    return torch_module.is_tensor(value) and value.dtype == torch_module.bool


def _normalize_axis(axis, torch_module):
    """Convert integer-like scalar axes and tuple entries to Python integers."""

    if _is_boolean_axis(axis, torch_module):
        raise TypeError(_AXIS_TYPE_MESSAGE)
    try:
        return _operator_index(axis)
    except TypeError as scalar_error:
        if getattr(axis, "shape", None) == ():
            raise TypeError(_AXIS_TYPE_MESSAGE) from scalar_error
        try:
            axes = tuple(axis)
        except TypeError as iterable_error:
            raise TypeError(_AXIS_TYPE_MESSAGE) from iterable_error

    normalized = []
    for one_axis in axes:
        if _is_boolean_axis(one_axis, torch_module):
            raise TypeError(_AXIS_TYPE_MESSAGE)
        try:
            normalized.append(_operator_index(one_axis))
        except TypeError as item_error:
            raise TypeError(_AXIS_TYPE_MESSAGE) from item_error
    return tuple(normalized)


def patch_pytorch_common_reduction_axis_contract() -> None:
    """Normalize integer-like axes used by ``pyrecest._backend._common``."""

    try:
        import pyrecest._backend._common as common_backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_normalizer = getattr(common_backend, "_normalize_reduction_axes", None)
    if original_normalizer is None or getattr(
        original_normalizer,
        "_pyrecest_common_reduction_axis_scalar_contract",
        False,
    ):
        return

    def _normalize_reduction_axes(axis, ndim_value):
        return original_normalizer(_normalize_axis(axis, torch), ndim_value)

    _normalize_reduction_axes.__name__ = getattr(
        original_normalizer,
        "__name__",
        "_normalize_reduction_axes",
    )
    _normalize_reduction_axes.__doc__ = getattr(original_normalizer, "__doc__", None)
    _normalize_reduction_axes._pyrecest_common_reduction_axis_scalar_contract = True
    common_backend._normalize_reduction_axes = _normalize_reduction_axes
