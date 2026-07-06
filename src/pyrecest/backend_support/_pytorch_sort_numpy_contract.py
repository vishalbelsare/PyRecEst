"""PyTorch sort helpers for backend compatibility."""

from __future__ import annotations

from operator import index as _operator_index


def normalize_sort_axis(axis):
    """Return a sort axis while preserving NumPy's flatten-all sentinel."""
    if axis is None:
        return None
    return _operator_index(axis)
