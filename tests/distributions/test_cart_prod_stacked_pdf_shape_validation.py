import pytest

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


def _stacked_gaussian_distribution():
    return CartProdStackedDistribution(
        [
            GaussianDistribution(array([0.0, 0.0]), eye(2)),
            GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
        ]
    )


@pytest.mark.parametrize(
    "xs",
    [
        array([1.0, 2.0, 3.0, 4.0, 5.0, 99.0]),
        array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 99.0],
                [0.5, 1.5, 2.5, 3.5, 4.5, -99.0],
            ]
        ),
    ],
)
def test_pdf_rejects_surplus_coordinates(xs):
    distribution = _stacked_gaussian_distribution()

    with pytest.raises(ValueError, match="trailing dimension 5"):
        distribution.pdf(xs)
