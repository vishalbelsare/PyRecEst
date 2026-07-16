import numpy as np
import pytest

from pyrecest.evidence import EvidenceComputationMode, resolve_evidence_computation_mode


class FailingArrayConversion:
    def __init__(self, error_type):
        self.error_type = error_type

    def __array__(self, dtype=None):
        del dtype
        raise self.error_type("array conversion failed")


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EvidenceComputationMode(return_smoothed=value),
        lambda value: EvidenceComputationMode(terminal_posterior=value),
        lambda value: EvidenceComputationMode.from_return_smoothed(value),
        lambda value: resolve_evidence_computation_mode(return_smoothed=value),
    ],
)
@pytest.mark.parametrize("error_type", [RuntimeError, OverflowError])
def test_evidence_bool_flags_normalize_array_conversion_errors(factory, error_type):
    with pytest.raises(ValueError, match="must be a bool"):
        factory(FailingArrayConversion(error_type))


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EvidenceComputationMode(return_smoothed=value),
        lambda value: EvidenceComputationMode(terminal_posterior=value),
        lambda value: EvidenceComputationMode.from_return_smoothed(value),
        lambda value: resolve_evidence_computation_mode(return_smoothed=value),
    ],
)
@pytest.mark.parametrize("masked_value", [True, False])
def test_evidence_bool_flags_reject_masked_scalars(factory, masked_value):
    with pytest.raises(ValueError, match="must be a bool"):
        factory(np.ma.array(masked_value, mask=True))


def test_evidence_bool_flags_accept_unmasked_masked_scalars():
    full = EvidenceComputationMode.from_return_smoothed(np.ma.array(True, mask=False))
    evidence_only = EvidenceComputationMode.from_return_smoothed(
        np.ma.array(False, mask=False)
    )

    assert full.mode == "full_smoothing"
    assert evidence_only.mode == "evidence_only"
