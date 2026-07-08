# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""Survival-aware association priors for explicit track management.

The utilities in this module implement a small, CRP-inspired prior for track-to-
measurement association. They are intentionally framed as association scoring
helpers rather than as a classical Dirichlet process: the resulting assignment
scores depend on track state, lifecycle metadata, visibility, and survival, so the
partition prior is not exchangeable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import exp, log
from typing import Any

import numpy as np

from .association_hypotheses import (
    AssociationHypothesis,
    MeasurementAxis,
    association_result_from_hypotheses,
    linear_gaussian_association_hypotheses,
)
from .track_manager import AssociationResult

FactorSpec = float | Callable[..., float]


@dataclass(frozen=True)
class SurvivalAwareAssociationConfig:
    """Configuration for survival-aware track-attractiveness association.

    The existing-track score applied to a Gaussian association hypothesis is

    ``track_mass * existence * survival * detection * visibility``.

    ``track_mass`` is based on the managed track's hit count and is discounted by
    ``mass_decay ** misses``. ``existence_probability=None`` means that the
    helper first looks for ``track.metadata[metadata_existence_key]`` and then
    falls back to one.
    """

    survival_probability: FactorSpec = 0.99
    detection_probability: FactorSpec = 0.95
    visibility_probability: FactorSpec = 1.0
    existence_probability: FactorSpec | None = None
    mass_decay: float = 1.0
    mass_power: float = 1.0
    birth_weight: float = 1.0
    clutter_weight: float = 1.0
    minimum_probability: float = 1.0e-12
    metadata_existence_key: str = "existence_probability"

    def __post_init__(self) -> None:
        _validate_probability_spec(self.survival_probability, "survival_probability")
        _validate_probability_spec(self.detection_probability, "detection_probability")
        _validate_probability_spec(self.visibility_probability, "visibility_probability")
        if self.existence_probability is not None:
            _validate_probability_spec(self.existence_probability, "existence_probability")
        _validate_probability(self.mass_decay, "mass_decay", allow_zero=True)
        if float(self.mass_power) < 0.0 or not np.isfinite(float(self.mass_power)):
            raise ValueError("mass_power must be finite and nonnegative")
        if float(self.birth_weight) < 0.0 or not np.isfinite(float(self.birth_weight)):
            raise ValueError("birth_weight must be finite and nonnegative")
        if float(self.clutter_weight) < 0.0 or not np.isfinite(float(self.clutter_weight)):
            raise ValueError("clutter_weight must be finite and nonnegative")
        if float(self.birth_weight) + float(self.clutter_weight) <= 0.0:
            raise ValueError("birth_weight and clutter_weight cannot both be zero")
        if float(self.minimum_probability) <= 0.0 or not np.isfinite(float(self.minimum_probability)):
            raise ValueError("minimum_probability must be finite and positive")
        if not isinstance(self.metadata_existence_key, str):
            raise ValueError("metadata_existence_key must be a string")


def survival_aware_track_log_prior(
    track: Any,
    measurement: Any | None = None,
    *,
    measurement_index: int | None = None,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> float:
    """Return the log prior attractiveness of assigning a measurement to a track."""

    config = SurvivalAwareAssociationConfig() if config is None else config
    mass = _effective_track_mass(track, config)
    existence = _resolve_existence_probability(track, measurement, measurement_index=measurement_index, step=step, config=config)
    survival = _resolve_probability(config.survival_probability, "survival_probability", track, measurement, measurement_index=measurement_index, step=step)
    detection = _resolve_probability(config.detection_probability, "detection_probability", track, measurement, measurement_index=measurement_index, step=step)
    visibility = _resolve_probability(config.visibility_probability, "visibility_probability", track, measurement, measurement_index=measurement_index, step=step)
    score = max(float(config.minimum_probability), mass * existence * survival * detection * visibility)
    return log(score)


def survival_aware_missed_detection_costs(
    tracks: Sequence[Any],
    *,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> np.ndarray:
    """Return per-track GNN costs for leaving tracks unassigned.

    A missed detection is cheap when the target has low existence, low detection
    probability, or low visibility. It is expensive when the target is likely to
    exist and should have been visible.
    """

    config = SurvivalAwareAssociationConfig() if config is None else config
    costs = []
    for track in tracks:
        existence = _resolve_existence_probability(track, None, measurement_index=None, step=step, config=config)
        detection = _resolve_probability(config.detection_probability, "detection_probability", track, None, measurement_index=None, step=step)
        visibility = _resolve_probability(config.visibility_probability, "visibility_probability", track, None, measurement_index=None, step=step)
        missed_probability = 1.0 - existence * detection * visibility
        missed_probability = max(float(config.minimum_probability), min(1.0, missed_probability))
        costs.append(-log(missed_probability))
    return np.asarray(costs, dtype=float)


def survival_aware_unassigned_measurement_cost(config: SurvivalAwareAssociationConfig | None = None) -> float:
    """Return the GNN cost for treating a measurement as birth or clutter."""

    config = SurvivalAwareAssociationConfig() if config is None else config
    return -log(max(float(config.minimum_probability), float(config.birth_weight) + float(config.clutter_weight)))


def apply_survival_aware_prior_to_hypotheses(
    hypotheses: Sequence[AssociationHypothesis],
    tracks: Sequence[Any],
    measurements: Sequence[Any] | None = None,
    *,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> list[AssociationHypothesis]:
    """Return hypotheses with survival-aware log priors folded into their costs."""

    config = SurvivalAwareAssociationConfig() if config is None else config
    adjusted_hypotheses = []
    for hypothesis in hypotheses:
        if hypothesis.is_missed_detection:
            adjusted_hypotheses.append(hypothesis)
            continue

        track_index = int(hypothesis.track_index)
        measurement_index = int(hypothesis.measurement_index)
        measurement = None if measurements is None else measurements[measurement_index]
        prior_log_score = survival_aware_track_log_prior(
            tracks[track_index],
            measurement,
            measurement_index=measurement_index,
            step=step,
            config=config,
        )
        if hypothesis.log_likelihood is not None:
            log_score = float(hypothesis.log_likelihood) + prior_log_score
        elif hypothesis.probability is not None and float(hypothesis.probability) > 0.0:
            log_score = log(float(hypothesis.probability)) + prior_log_score
        elif hypothesis.cost is not None:
            log_score = -float(hypothesis.cost) + prior_log_score
        else:
            log_score = prior_log_score

        metadata = dict(hypothesis.metadata or {})
        metadata["survival_aware_log_prior"] = prior_log_score
        adjusted_hypotheses.append(
            replace(
                hypothesis,
                cost=-log_score,
                probability=exp(log_score) if log_score < 700.0 else float("inf"),
                metadata=metadata,
            )
        )
    return adjusted_hypotheses


def survival_aware_linear_gaussian_association_hypotheses(
    tracks,
    measurements,
    measurement_matrix,
    meas_noise,
    gates=None,
    *,
    measurement_axis: MeasurementAxis = "auto",
    include_rejected: bool = False,
    metadata_builder: Callable[..., dict[str, Any] | None] | None = None,
    strict_backend: bool = False,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> list[AssociationHypothesis]:
    """Build linear/Gaussian hypotheses and apply a survival-aware prior."""

    measurement_vectors = _coerce_measurements_for_prior(measurements, measurement_matrix, measurement_axis)
    hypotheses = linear_gaussian_association_hypotheses(
        tracks,
        measurements,
        measurement_matrix,
        meas_noise,
        gates=gates,
        measurement_axis=measurement_axis,
        include_rejected=include_rejected,
        metadata_builder=metadata_builder,
        strict_backend=strict_backend,
    )
    return apply_survival_aware_prior_to_hypotheses(
        hypotheses,
        tracks,
        measurement_vectors,
        step=step,
        config=config,
    )


def build_survival_aware_linear_gaussian_hypothesis_associator(
    measurement_matrix,
    meas_noise,
    *,
    config: SurvivalAwareAssociationConfig | None = None,
    gates=None,
    missing_cost: float = np.inf,
    unassigned_measurement_cost: float | Sequence[float] | None = None,
    measurement_axis: MeasurementAxis = "auto",
):
    """Create a :class:`TrackManager`-compatible survival-aware associator."""

    config = SurvivalAwareAssociationConfig() if config is None else config

    def associator(tracks, measurements, **kwargs) -> AssociationResult:
        effective_config = kwargs.get("config", config)
        effective_measurement_matrix = kwargs.get("measurement_matrix", measurement_matrix)
        effective_measurement_axis = kwargs.get("measurement_axis", measurement_axis)
        hypotheses = survival_aware_linear_gaussian_association_hypotheses(
            tracks,
            measurements,
            effective_measurement_matrix,
            kwargs.get("meas_noise", meas_noise),
            gates=kwargs.get("gates", gates),
            measurement_axis=effective_measurement_axis,
            include_rejected=kwargs.get("include_rejected", False),
            metadata_builder=kwargs.get("metadata_builder", None),
            strict_backend=kwargs.get("strict_backend", False),
            step=kwargs.get("step", None),
            config=effective_config,
        )
        measurement_vectors = _coerce_measurements_for_prior(
            measurements,
            effective_measurement_matrix,
            effective_measurement_axis,
        )
        if unassigned_measurement_cost is None:
            default_unassigned_measurement_cost = survival_aware_unassigned_measurement_cost(effective_config)
        else:
            default_unassigned_measurement_cost = unassigned_measurement_cost
        return association_result_from_hypotheses(
            hypotheses,
            num_tracks=len(tracks),
            num_measurements=len(measurement_vectors),
            missing_cost=kwargs.get("missing_cost", missing_cost),
            unassigned_track_cost=kwargs.get(
                "unassigned_track_cost",
                survival_aware_missed_detection_costs(tracks, step=kwargs.get("step", None), config=effective_config),
            ),
            unassigned_measurement_cost=kwargs.get("unassigned_measurement_cost", default_unassigned_measurement_cost),
        )

    return associator


def _effective_track_mass(track: Any, config: SurvivalAwareAssociationConfig) -> float:
    hits = max(1, int(getattr(track, "hits", 1)))
    misses = max(0, int(getattr(track, "misses", 0)))
    mass = float(hits) ** float(config.mass_power)
    mass *= float(config.mass_decay) ** misses
    return max(float(config.minimum_probability), mass)


def _resolve_existence_probability(
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
    config: SurvivalAwareAssociationConfig,
) -> float:
    if config.existence_probability is not None:
        return _resolve_probability(config.existence_probability, "existence_probability", track, measurement, measurement_index=measurement_index, step=step)

    metadata = getattr(track, "metadata", None)
    if isinstance(metadata, dict) and config.metadata_existence_key in metadata:
        return _validate_probability(metadata[config.metadata_existence_key], f"metadata[{config.metadata_existence_key!r}]", allow_zero=True)
    if hasattr(track, "existence_probability"):
        return _validate_probability(getattr(track, "existence_probability"), "track.existence_probability", allow_zero=True)
    return 1.0


def _resolve_probability(
    spec: FactorSpec,
    name: str,
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
) -> float:
    value = _resolve_value(spec, track, measurement, measurement_index=measurement_index, step=step)
    return _validate_probability(value, name, allow_zero=True)


def _resolve_value(spec: FactorSpec, track: Any, measurement: Any | None, *, measurement_index: int | None, step: int | None) -> float:
    if not callable(spec):
        return float(spec)

    call_attempts = (
        lambda: spec(track=track, measurement=measurement, measurement_index=measurement_index, step=step),
        lambda: spec(track, measurement, measurement_index, step),
        lambda: spec(track, measurement),
        lambda: spec(track),
    )
    last_error: TypeError | None = None
    for attempt in call_attempts:
        try:
            return float(attempt())
        except TypeError as exc:
            last_error = exc
    raise TypeError("Callable probability factors must accept keyword arguments, (track, measurement, measurement_index, step), (track, measurement), or track") from last_error


def _validate_probability_spec(spec: FactorSpec, name: str) -> None:
    if callable(spec):
        return
    _validate_probability(spec, name, allow_zero=True)


def _validate_probability(value: Any, name: str, *, allow_zero: bool) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a scalar probability")
    probability = float(value_array.item())
    lower_ok = probability >= 0.0 if allow_zero else probability > 0.0
    if not lower_ok or probability > 1.0 or not np.isfinite(probability):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return probability


def _coerce_measurements_for_prior(measurements, measurement_matrix, measurement_axis: MeasurementAxis) -> list[np.ndarray]:
    measurement_matrix = np.asarray(measurement_matrix, dtype=float)
    measurement_dim = int(measurement_matrix.shape[0])
    try:
        array = np.asarray(measurements, dtype=float)
    except (TypeError, ValueError):
        return [np.asarray(measurement, dtype=float).reshape(-1) for measurement in measurements]
    if array.ndim == 1:
        return [array.reshape(-1)]
    if array.ndim != 2:
        return [np.asarray(measurement, dtype=float).reshape(-1) for measurement in measurements]
    if measurement_axis == "columns":
        return [array[:, index].reshape(-1) for index in range(array.shape[1])]
    if measurement_axis in ("rows", "sequence"):
        return [array[index, :].reshape(-1) for index in range(array.shape[0])]
    if measurement_axis != "auto":
        raise ValueError("measurement_axis must be 'auto', 'columns', 'rows', or 'sequence'")
    columns_match = array.shape[0] == measurement_dim
    rows_match = array.shape[1] == measurement_dim
    if columns_match and not rows_match:
        return [array[:, index].reshape(-1) for index in range(array.shape[1])]
    if rows_match and not columns_match:
        return [array[index, :].reshape(-1) for index in range(array.shape[0])]
    if columns_match and rows_match and array.shape == (1, 1):
        return [array[:, 0].reshape(-1)]
    if columns_match and rows_match:
        raise ValueError("Ambiguous measurement array orientation for measurement_axis='auto'. Pass measurement_axis='columns' or 'rows' explicitly.")
    raise ValueError("Neither axis of measurements matches the measurement dimension inferred from measurement_matrix")


__all__ = [
    "SurvivalAwareAssociationConfig",
    "apply_survival_aware_prior_to_hypotheses",
    "build_survival_aware_linear_gaussian_hypothesis_associator",
    "survival_aware_linear_gaussian_association_hypotheses",
    "survival_aware_missed_detection_costs",
    "survival_aware_track_log_prior",
    "survival_aware_unassigned_measurement_cost",
]
