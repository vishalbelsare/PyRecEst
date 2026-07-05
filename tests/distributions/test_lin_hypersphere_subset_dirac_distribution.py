import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.distributions.cart_prod.lin_hypersphere_subset_dirac_distribution import (
    LinHypersphereSubsetCartProdDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)


def test_constructor_accepts_array_like_inputs_and_marginalizes():
    d = [[1.0, 0.0, 2.0], [0.0, 1.0, 4.0]]
    w = [0.25, 0.75]

    dist = LinHypersphereSubsetCartProdDiracDistribution(1, d, w)
    bounded = dist.marginalize_linear()
    linear = dist.marginalize_periodic()

    npt.assert_allclose(dist.d, array(d))
    npt.assert_allclose(dist.w, array(w))
    assert dist.bound_dim == 1
    assert dist.lin_dim == 1

    assert isinstance(bounded, HypersphericalDiracDistribution)
    npt.assert_allclose(bounded.d, array([[1.0, 0.0], [0.0, 1.0]]))
    npt.assert_allclose(bounded.w, array(w))

    assert isinstance(linear, LinearDiracDistribution)
    npt.assert_allclose(linear.d, array([[2.0], [4.0]]))
    npt.assert_allclose(linear.w, array(w))
