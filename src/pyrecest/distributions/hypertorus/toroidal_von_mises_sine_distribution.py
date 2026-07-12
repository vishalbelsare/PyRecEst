# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
import math

from pyrecest.backend import array, cos, exp, sin
from scipy.special import gammaln, ive

from .abstract_toroidal_bivar_vm_distribution import (
    AbstractToroidalBivarVMDistribution,
    validate_scalar_parameter,
)

_SERIES_RTOL = 1e-12
_SERIES_MIN_TERMS = 10
_SERIES_MAX_TERMS = 10000


def _ive_over_power(order, concentration):
    """Return exp(-concentration) * I_order(concentration) / concentration**order."""
    concentration = float(concentration)
    if order == 0:
        return float(ive(0, concentration))
    if concentration == 0.0:
        return math.exp(-order * math.log(2.0) - float(gammaln(order + 1.0)))
    return float(ive(order, concentration)) / concentration**order


def _adaptive_positive_series_sum(term):
    """Sum a nonnegative series until the tail term is negligible."""
    total = 0.0
    terms = []
    for order in range(_SERIES_MAX_TERMS + 1):
        contribution = float(term(order))
        if not math.isfinite(contribution):
            raise FloatingPointError("Bivariate von Mises series term is not finite")
        terms.append(contribution)
        total += contribution
        if order >= _SERIES_MIN_TERMS and contribution <= _SERIES_RTOL * max(
            1.0, abs(total)
        ):
            return math.fsum(terms)
    raise RuntimeError("Bivariate von Mises series did not converge")


class ToroidalVonMisesSineDistribution(AbstractToroidalBivarVMDistribution):
    """Bivariate von Mises sine model on the torus.

    References
    ----------
    Singh, H., Hnizdo, V., & Demchuk, E. (2002). Probabilistic model for
    two dependent circular variables. Biometrika, 89(3), 719-723.
    """

    def __init__(self, mu, kappa, lambda_):
        AbstractToroidalBivarVMDistribution.__init__(self, mu, kappa)
        lambda_ = validate_scalar_parameter(lambda_, "lambda_")
        self.lambda_ = lambda_
        self.C = math.exp(-self.log_norm_const)

    def _scaled_norm_const(self):
        """Return the normalizer with exp(kappa[0] + kappa[1]) factored out."""
        lambda_sq_over_four = float(self.lambda_) ** 2 / 4.0
        kappa0 = float(self.kappa[0])
        kappa1 = float(self.kappa[1])

        def s(order):
            return (
                math.comb(2 * order, order)
                * lambda_sq_over_four**order
                * _ive_over_power(order, kappa0)
                * _ive_over_power(order, kappa1)
            )

        return 4.0 * math.pi**2 * _adaptive_positive_series_sum(s)

    @property
    def log_norm_const(self):
        return (
            float(self.kappa[0])
            + float(self.kappa[1])
            + math.log(self._scaled_norm_const())
        )

    @property
    def norm_const(self):
        try:
            return math.exp(self.log_norm_const)
        except OverflowError:
            return math.inf

    def pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.dim:
            raise ValueError(
                f"xs must have trailing dimension {self.dim}, got {xs.shape}."
            )
        return exp(
            self.kappa[0] * (cos(xs[..., 0] - self.mu[0]) - 1.0)
            + self.kappa[1] * (cos(xs[..., 1] - self.mu[1]) - 1.0)
            + self._coupling_term(xs)
            - math.log(self._scaled_norm_const())
        )

    def _coupling_term(self, xs):
        return (
            self.lambda_ * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )
