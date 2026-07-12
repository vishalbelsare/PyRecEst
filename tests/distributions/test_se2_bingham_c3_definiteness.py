import numpy as np
import pytest
from pyrecest.distributions import SE2BinghamDistribution


def test_rejects_singular_c3_from_parts_and_full_matrix():
    c1 = np.array([[-3.0, 0.5], [0.5, -1.0]])
    c2 = np.array([[0.1, 0.2], [-0.1, 0.3]])
    singular_c3 = np.array([[-1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="negative definite"):
        SE2BinghamDistribution(c1, c2, singular_c3)

    full_matrix = np.block([[c1, c2.T], [c2, singular_c3]])
    with pytest.raises(ValueError, match="negative definite"):
        SE2BinghamDistribution(full_matrix)
