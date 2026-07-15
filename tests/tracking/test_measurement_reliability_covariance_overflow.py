from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    apply_measurement_reliability,
    scale_covariance_by_reliability,
)


def test_reliability_covariance_scaling_rejects_nonfinite_product() -> None:
    covariance = np.array([[np.finfo(float).max]])

    with pytest.raises(ValueError, match="scaled covariance overflows"):
        scale_covariance_by_reliability(covariance, 0.5)

    with pytest.raises(ValueError, match="scaled covariance overflows"):
        apply_measurement_reliability(covariance, reliability=0.5)


def test_reliability_covariance_scaling_accepts_safe_bound() -> None:
    covariance = np.array([[np.finfo(float).max]])

    scaled, scale = scale_covariance_by_reliability(
        covariance,
        0.5,
        max_scale=1.0,
    )

    assert scale == 1.0
    np.testing.assert_array_equal(scaled, covariance)
