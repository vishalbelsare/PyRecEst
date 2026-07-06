# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Identity-dependent CRP-style association for managed tracks.

The helpers in this module add a lightweight Bayesian-nonparametric identity
prior on top of ordinary linear/Gaussian association scores.  They are designed
to plug into :class:`pyrecest.filters.TrackManager` rather than to replace the
existing track lifecycle machinery.

The score is intentionally explicit and diagnostic-oriented:

``log_score = log p(y | track) + log p_D - log clutter + log identity_memory``

where the identity-memory term can depend on hits, misses, survival probability,
and optional user-provided appearance or visibility callbacks.  The resulting
negative log score is passed to the existing assignment solver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import log
from typing import Any

import numpy as np

from .association_hypotheses import (
    AssociationHypothesis,
    MeasurementAxis,
    association_result_from_hypotheses,
    filter_hypotheses,
    linear_gaussian_association_hypotheses,
)
from .association_hypotheses import (
    _coerce_measurements,  # pylint: disable=protected-access
)


@dataclass(frozen=True)
class IdentityDependentCRPParams:
    """Parameters for identity-dependent CRP association scoring.

    Parameters
    ----------
    concentration : float, optional
        Positive CRP-style concentration used for unmatched-measurement birth
        costs when ``birth_log_prior`` is not supplied.
    count_power : float, optional
        Exponent applied to the historical hit-count memory term. ``0`` removes
        the rich-get-richer contribution.
    count_offset : float, optional
        Nonnegative offset added to a track's hit count before the logarithm.
    miss_decay : float, optional
        Nonnegative exponential aging penalty per consecutive missed detection.
    survival_probability : float, optional
        Survival probability applied once per consecutive missed detection in
        the identity prior.
    detection_probability : float, optional
        Detection probability used in association and missed-detection costs.
    clutter_intensity : float, optional
        Positive clutter intensity used to score detection-vs-clutter evidence.
    birth_log_prior : float, optional
        Additive log prior for a new identity.  If omitted,
        ``log(concentration)`` is used.
    measurement_log_likelihood_weight : float, optional
        Nonnegative multiplier for the linear/Gaussian measurement log
        likelihood.
    appearance_weight : float, optional
        Multiplier for optional appearance/re-identification callback scores.
    visibility_weight : float, optional
        Multiplier for optional visibility/map/sensor-geometry callback scores.
    minimum_probability : float, optional
        Small positive clipping constant for probabilities and intensities.
    """

    concentration: float = 1.0
    count_power: float = 1.0
    count_offset: float = 0.0
    miss_decay: float = 1.0
    survival_probability: float = 0.99
    detection_probability: float = 0.95
    clutter_intensity: float = 1e-9
    birth_log_prior: float | None = None
    measurement_log_likelihood_weight: float = 1.0
    appearance_weight: float = 1.0
    visibility_weight: float = 1.0
    minimum_probability: float = 1e-12


ScoreCallback = Callable[..., float | None]


def identity_dependent_crp_hypotheses(
    tracks: Sequence[Any],
    measurements: Any,
    measurement_matrix: Any,
    meas_noise: Any,
    params: IdentityDependentCRPParams | dict[str, Any] | None = None,
    gates=None,
    *,
    measurement_axis: MeasurementAxis = "auto",
    appearance_log_likelihood: ScoreCallback | None = None,
    visibility_log_likelihood: ScoreCallback | None = None,
    metadata_builder: Callable[..., dict[str, Any] | None] | None = None,
    include_rejected: bool = False,
    strict_backend: bool = False,
) -> list[AssociationHypothesis]:
    """Build identity-dependent CRP association hypotheses.

    The returned hypotheses retain the Gaussian innovation diagnostics from
    :func:`linear_gaussian_association_hypotheses`, but their ``cost`` and
    ``log_likelihood`` fields contain the identity-aware association score.
    Optional callbacks are additive log-score functions and are called with
    keyword arguments including ``track``, ``measurement``, ``track_index``,
    ``measurement_index``, and ``base_hypothesis``.
    """

    params = _normalize_params(params)
    measurement_matrix_array = np.asarray(measurement_matrix, dtype=float)
    measurement_dim = int(measurement_matrix_array.shape[0])
    measurement_vectors = _coerce_measurements(
        measurements,
        measurement_axis=measurement_axis,
        measurement_dim=measurement_dim,
    )

    base_hypotheses = linear_gaussian_association_hypotheses(
        tracks,
        measurement_vectors,
        measurement_matrix_array,
        meas_noise,
        gates=gates,
        measurement_axis="sequence",
        include_rejected=include_rejected,
        strict_backend=strict_backend,
    )

    scored_hypotheses: list[AssociationHypothesis] = []
    for base_hypothesis in base_hypotheses:
        track_index = int(base_hypothesis.track_index)
        measurement_index = int(base_hypothesis.measurement_index)  # type: ignore[arg-type]
        track = tracks[track_index]
        measurement = measurement_vectors[measurement_index]

        terms = _identity_dependent_log_terms(
            track=track,
            measurement=measurement,
            track_index=track_index,
            measurement_index=measurement_index,
            base_hypothesis=base_hypothesis,
            params=params,
            appearance_log_likelihood=appearance_log_likelihood,
            visibility_log_likelihood=visibility_log_likelihood,
        )
        log_total_score = sum(terms.values())
        cost = _negative_log_score(log_total_score)

        metadata = dict(base_hypothesis.metadata or {})
        metadata.update(terms)
        metadata["identity_dependent_crp_log_score"] = log_total_score
        if metadata_builder is not None:
            extra_metadata = metadata_builder(
                track=track,
                measurement=measurement,
                track_index=track_index,
                measurement_index=measurement_index,
                base_hypothesis=base_hypothesis,
                log_terms=dict(terms),
                log_score=log_total_score,
                cost=cost,
            )
            if extra_metadata:
                metadata.update(extra_metadata)

        scored_hypotheses.append(
            replace(
                base_hypothesis,
                cost=cost,
                log_likelihood=log_total_score,
                metadata=metadata,
            )
        )

    if include_rejected:
        return scored_hypotheses
    return [hypothesis for hypothesis in scored_hypotheses if hypothesis.accepted]


def build_identity_dependent_crp_associator(
    measurement_matrix: Any,
    meas_noise: Any,
    params: IdentityDependentCRPParams | dict[str, Any] | None = None,
    *,
    gates=None,
    measurement_axis: MeasurementAxis = "auto",
    appearance_log_likelihood: ScoreCallback | None = None,
    visibility_log_likelihood: ScoreCallback | None = None,
    birth_log_likelihood: ScoreCallback | None = None,
    metadata_builder: Callable[..., dict[str, Any] | None] | None = None,
):
    """Create a :class:`TrackManager`-compatible identity-dependent associator."""

    default_params = _normalize_params(params)

    def associator(tracks: Sequence[Any], measurements: Any, **kwargs):
        effective_measurement_matrix = kwargs.get(
            "measurement_matrix",
            measurement_matrix,
        )
        effective_params = _normalize_params(kwargs.get("params", default_params))
        effective_measurement_axis = kwargs.get("measurement_axis", measurement_axis)
        effective_measurement_matrix_array = np.asarray(
            effective_measurement_matrix,
            dtype=float,
        )
        effective_measurement_dim = int(effective_measurement_matrix_array.shape[0])
        measurement_vectors = _coerce_measurements(
            measurements,
            measurement_axis=effective_measurement_axis,
            measurement_dim=effective_measurement_dim,
        )

        hypotheses = identity_dependent_crp_hypotheses(
            tracks,
            measurement_vectors,
            effective_measurement_matrix_array,
            kwargs.get("meas_noise", meas_noise),
            effective_params,
            gates=kwargs.get("gates", gates),
            measurement_axis="sequence",
            appearance_log_likelihood=kwargs.get(
                "appearance_log_likelihood",
                appearance_log_likelihood,
            ),
            visibility_log_likelihood=kwargs.get(
                "visibility_log_likelihood",
                visibility_log_likelihood,
            ),
            metadata_builder=kwargs.get("metadata_builder", metadata_builder),
            include_rejected=kwargs.get("include_rejected", False),
            strict_backend=kwargs.get("strict_backend", False),
        )

        if kwargs.get("post_gates") is not None:
            hypotheses = filter_hypotheses(
                hypotheses,
                kwargs["post_gates"],
                accepted_only=True,
            )

        unassigned_track_cost = kwargs.get(
            "unassigned_track_cost",
            identity_dependent_crp_missed_detection_costs(tracks, effective_params),
        )
        unassigned_measurement_cost = kwargs.get(
            "unassigned_measurement_cost",
            identity_dependent_crp_birth_costs(
                measurement_vectors,
                effective_params,
                birth_log_likelihood=kwargs.get(
                    "birth_log_likelihood",
                    birth_log_likelihood,
                ),
            ),
        )

        return association_result_from_hypotheses(
            hypotheses,
            num_tracks=len(tracks),
            num_measurements=len(measurement_vectors),
            missing_cost=kwargs.get("missing_cost", np.inf),
            unassigned_track_cost=unassigned_track_cost,
            unassigned_measurement_cost=unassigned_measurement_cost,
        )

    return associator


def identity_dependent_crp_missed_detection_costs(
    tracks: Sequence[Any],
    params: IdentityDependentCRPParams | dict[str, Any] | None = None,
) -> np.ndarray:
    """Return per-track missed-detection costs for the association solver."""

    params = _normalize_params(params)
    log_missed_detection = _safe_log_probability(
        1.0 - float(params.detection_probability),
        "1 - detection_probability",
        params.minimum_probability,
    )
    costs = []
    for track in tracks:
        misses = _track_misses(track)
        survival_log = misses * _safe_log_probability(
            float(params.survival_probability),
            "survival_probability",
            params.minimum_probability,
        )
        costs.append(_negative_log_score(log_missed_detection + survival_log))
    return np.asarray(costs, dtype=float)


def identity_dependent_crp_birth_costs(
    measurements: Sequence[Any],
    params: IdentityDependentCRPParams | dict[str, Any] | None = None,
    *,
    birth_log_likelihood: ScoreCallback | None = None,
) -> np.ndarray:
    """Return per-measurement new-identity costs for the association solver."""

    params = _normalize_params(params)
    birth_log_prior = _birth_log_prior(params)
    costs = []
    for measurement_index, measurement in enumerate(measurements):
        log_score = birth_log_prior
        log_score += _call_score_callback(
            birth_log_likelihood,
            measurement=np.asarray(measurement, dtype=float).reshape(-1),
            measurement_index=measurement_index,
        )
        costs.append(_negative_log_score(log_score))
    return np.asarray(costs, dtype=float)


def _identity_dependent_log_terms(
    *,
    track: Any,
    measurement: np.ndarray,
    track_index: int,
    measurement_index: int,
    base_hypothesis: AssociationHypothesis,
    params: IdentityDependentCRPParams,
    appearance_log_likelihood: ScoreCallback | None,
    visibility_log_likelihood: ScoreCallback | None,
) -> dict[str, float]:
    if base_hypothesis.log_likelihood is None:
        log_measurement_likelihood = -_as_log_score(base_hypothesis.cost, "base_hypothesis.cost")
    else:
        log_measurement_likelihood = _as_log_score(
            base_hypothesis.log_likelihood,
            "base_hypothesis.log_likelihood",
        )

    log_detection = _safe_log_probability(
        float(params.detection_probability),
        "detection_probability",
        params.minimum_probability,
    )
    log_clutter_correction = -_safe_log_positive(
        float(params.clutter_intensity),
        "clutter_intensity",
        params.minimum_probability,
    )
    log_identity_memory = _identity_memory_log_score(track, params)

    callback_kwargs = {
        "track": track,
        "measurement": np.asarray(measurement, dtype=float).reshape(-1),
        "track_index": track_index,
        "measurement_index": measurement_index,
        "base_hypothesis": base_hypothesis,
    }
    log_appearance = _call_score_callback(
        appearance_log_likelihood,
        **callback_kwargs,
    )
    log_visibility = _call_score_callback(
        visibility_log_likelihood,
        **callback_kwargs,
    )

    return {
        "identity_dependent_crp_log_measurement_likelihood": (
            float(params.measurement_log_likelihood_weight) * log_measurement_likelihood
        ),
        "identity_dependent_crp_log_detection": log_detection,
        "identity_dependent_crp_log_clutter_correction": log_clutter_correction,
        "identity_dependent_crp_log_memory": log_identity_memory,
        "identity_dependent_crp_log_appearance": float(params.appearance_weight) * log_appearance,
        "identity_dependent_crp_log_visibility": float(params.visibility_weight) * log_visibility,
    }


def _identity_memory_log_score(track: Any, params: IdentityDependentCRPParams) -> float:
    hits = _track_hits(track)
    misses = _track_misses(track)
    count = max(hits + float(params.count_offset), float(params.minimum_probability))
    count_term = float(params.count_power) * log(count)
    aging_term = -float(params.miss_decay) * misses
    survival_term = misses * _safe_log_probability(
        float(params.survival_probability),
        "survival_probability",
        params.minimum_probability,
    )
    return count_term + aging_term + survival_term


def _normalize_params(
    params: IdentityDependentCRPParams | dict[str, Any] | None,
) -> IdentityDependentCRPParams:
    if params is None:
        params = IdentityDependentCRPParams()
    elif isinstance(params, dict):
        params = IdentityDependentCRPParams(**params)
    elif not isinstance(params, IdentityDependentCRPParams):
        raise TypeError("params must be IdentityDependentCRPParams, a dict, or None")

    _validate_params(params)
    return params


def _validate_params(params: IdentityDependentCRPParams) -> None:
    _require_positive(params.concentration, "concentration")
    _require_nonnegative(params.count_power, "count_power")
    _require_nonnegative(params.count_offset, "count_offset")
    _require_nonnegative(params.miss_decay, "miss_decay")
    _require_probability(params.survival_probability, "survival_probability")
    _require_probability(params.detection_probability, "detection_probability")
    _require_positive(params.clutter_intensity, "clutter_intensity")
    _require_nonnegative(params.measurement_log_likelihood_weight, "measurement_log_likelihood_weight")
    _require_positive(params.minimum_probability, "minimum_probability")
    if float(params.minimum_probability) >= 0.5:
        raise ValueError("minimum_probability must be less than 0.5")
    if params.birth_log_prior is not None:
        _as_log_score(params.birth_log_prior, "birth_log_prior")


def _track_hits(track: Any) -> float:
    hits = getattr(track, "hits", 1)
    return max(float(hits), 0.0)


def _track_misses(track: Any) -> float:
    misses = getattr(track, "misses", 0)
    return max(float(misses), 0.0)


def _birth_log_prior(params: IdentityDependentCRPParams) -> float:
    if params.birth_log_prior is not None:
        return _as_log_score(params.birth_log_prior, "birth_log_prior")
    return _safe_log_positive(
        float(params.concentration),
        "concentration",
        params.minimum_probability,
    )


def _call_score_callback(callback: ScoreCallback | None, **kwargs) -> float:
    if callback is None:
        return 0.0
    value = callback(**kwargs)
    if value is None:
        return 0.0
    return _as_log_score(value, "callback log score")


def _negative_log_score(log_score: float) -> float:
    log_score = _as_log_score(log_score, "log_score")
    if np.isneginf(log_score):
        return float("inf")
    return -float(log_score)


def _safe_log_probability(value: float, name: str, eps: float) -> float:
    _require_probability(value, name)
    return log(min(max(float(value), float(eps)), 1.0 - float(eps)))


def _safe_log_positive(value: float, name: str, eps: float) -> float:
    _require_positive(value, name)
    return log(max(float(value), float(eps)))


def _as_log_score(value: Any, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a scalar numeric log score")
    try:
        score = float(value_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar numeric log score") from exc
    if np.isnan(score) or np.isposinf(score):
        raise ValueError(f"{name} must be finite or negative infinity")
    return score


def _require_probability(value: Any, name: str) -> None:
    value = _as_finite_scalar(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_positive(value: Any, name: str) -> None:
    value = _as_finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(value: Any, name: str) -> None:
    value = _as_finite_scalar(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _as_finite_scalar(value: Any, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a finite scalar number")
    try:
        scalar = float(value_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


__all__ = [
    "IdentityDependentCRPParams",
    "build_identity_dependent_crp_associator",
    "identity_dependent_crp_birth_costs",
    "identity_dependent_crp_hypotheses",
    "identity_dependent_crp_missed_detection_costs",
]
