"""PyTorch sort helpers for backend compatibility."""

from __future__ import annotations

from operator import index as _operator_index


def normalize_sort_axis(axis):
    """Return a sort axis while preserving NumPy's flatten-all sentinel."""
    if axis is None:
        return None
    return _operator_index(axis)


def sort_axis_none(backend_module, torch_module, values, axis=-1):
    """Sort values with NumPy-style ``axis=None`` support."""
    values = backend_module.array(values)
    axis = normalize_sort_axis(axis)
    if axis is None:
        values = torch_module.flatten(values)
        axis = 0
    return torch_module.sort(values, dim=axis).values
