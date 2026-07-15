import operator as _operator

from ._dispatch import _common
from ._dispatch import numpy as _np

_modify_func_default_dtype = _common._modify_func_default_dtype
_allow_complex_dtype = _common._allow_complex_dtype
_BOOLEAN_TYPES = (bool, _np.bool_)
_TEMPORAL_DTYPE_KINDS = "Mm"


def _rand(*dims, size=None):
    """Draw uniform samples with NumPy-style positional and size arguments."""
    if dims:
        if size is not None:
            raise TypeError("Specify either positional dimensions or size, not both.")
        size = dims[0] if len(dims) == 1 else dims
    return _np.random.random(_normalize_size(size))


rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_rand)
)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _is_temporal_scalar_array(value_array):
    return value_array.ndim == 0 and value_array.dtype.kind in _TEMPORAL_DTYPE_KINDS


def _size_type_error():
    return TypeError("size must be None, an integer, or a sequence of integers")


def _normalize_size(size):
    if size is None:
        return None
    if _contains_boolean_value(size) or isinstance(size, (str, bytes)):
        raise _size_type_error()
    try:
        size_array = _np.asarray(size)
    except (TypeError, ValueError) as exc:
        raise _size_type_error() from exc
    if size_array.ndim == 1 and size_array.size == 0:
        return ()
    if size_array.dtype.kind not in "iu":
        raise _size_type_error()
    if size_array.ndim == 0:
        dimension = int(size_array.item())
        if dimension < 0:
            raise ValueError("size dimensions must be non-negative")
        return dimension
    if size_array.ndim != 1:
        raise _size_type_error()
    dimensions = tuple(int(dimension) for dimension in size_array.tolist())
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("size dimensions must be non-negative")
    return dimensions


def _validate_uniform_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        bound_array = _np.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if bound_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(bound_array)):
        raise ValueError("uniform bounds must be finite")
    return bound_array


def _validate_uniform_bounds(low, high):
    low_array = _validate_uniform_bound(low, "low")
    high_array = _validate_uniform_bound(high, "high")
    if _np.any(low_array > high_array):
        raise ValueError("Upper bound must be greater than or equal to lower bound")


def _uniform(low=0.0, high=1.0, size=None):
    _validate_uniform_bounds(low, high)
    try:
        return _np.random.uniform(low, high, _normalize_size(size))
    except OverflowError as exc:
        raise OverflowError("high - low range exceeds valid bounds") from exc


uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_uniform)
)


def _validate_normal_parameter(value, name):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        value_array = _np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if value_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(value_array)):
        raise ValueError(f"{name} must be finite")
    return value_array


def _validate_normal_scale(scale):
    scale_array = _validate_normal_parameter(scale, "scale")
    if _np.any(scale_array < 0):
        raise ValueError("scale must be non-negative")
    return scale_array


def _normal(loc=0.0, scale=1.0, size=None):
    loc = _validate_normal_parameter(loc, "loc")
    scale = _validate_normal_scale(scale)
    return _np.random.normal(loc, scale, _normalize_size(size))


normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_normal)
)
