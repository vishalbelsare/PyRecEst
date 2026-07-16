from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from pyrecest.models.weak_measurement import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
)


def _overflowing_finite_standard_deviation() -> float:
    largest_safe = np.sqrt(np.finfo(float).max)
    return float(np.nextafter(largest_safe, np.inf))


@pytest.mark.parametrize(
    "factory",
    [
        lambda std: diagonal_measurement_covariance([std]),
        lambda std: block_diag_measurement_covariance(trusted_std=[std]),
        lambda std: MaskedLinearMeasurementModel(
            state_dim=1, observed_dims=[0], stds=[std]
        ),
        lambda std: WeakDimensionMeasurementModel(np.eye(1), stds=[std]),
    ],
)
def test_measurement_standard_deviation_overflow_is_rejected(
    factory: Callable[[float], object],
) -> None:
    std = _overflowing_finite_standard_deviation()

    assert np.isfinite(std)
    with pytest.raises(ValueError, match="finite measurement covariance"):
        factory(std)


def test_largest_representable_measurement_variance_remains_supported() -> None:
    std = np.sqrt(np.finfo(float).max)

    covariance = diagonal_measurement_covariance([std])

    assert np.isfinite(covariance).all()
