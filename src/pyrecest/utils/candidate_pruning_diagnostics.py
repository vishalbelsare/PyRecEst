"""Diagnostics for pairwise association candidate pruning.

The helpers in this module summarize the generic pruning mask produced by
:mod:`pyrecest.utils.candidate_pruning`.  They intentionally operate only on
numeric cost/probability matrices and pruning configs; downstream projects should
interpret the diagnostics in their own domain-specific context.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .candidate_pruning import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    candidate_pruning_config_from_mapping,
)


@dataclass(frozen=True)
class CandidatePruningDiagnostics:
    """Summary of a candidate-pruning mask for one pairwise matrix."""

    shape: tuple[int, int]
    total_entries: int
    finite_entries: int
    kept_entries: int
    pruned_finite_entries: int
    finite_retention_fraction: float
    finite_pruned_fraction: float
    row_candidate_counts: tuple[int, ...]
    column_candidate_counts: tuple[int, ...]
    rows_without_candidates: int
    columns_without_candidates: int
    applied_rules: tuple[str, ...]
    rule_kept_entries: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON/CSV-friendly diagnostic summary."""

        return {
            "n_rows": int(self.shape[0]),
            "n_columns": int(self.shape[1]),
            "total_entries": int(self.total_entries),
            "finite_entries": int(self.finite_entries),
            "kept_entries": int(self.kept_entries),
            "pruned_finite_entries": int(self.pruned_finite_entries),
            "finite_retention_fraction": float(self.finite_retention_fraction),
            "finite_pruned_fraction": float(self.finite_pruned_fraction),
            "rows_without_candidates": int(self.rows_without_candidates),
            "columns_without_candidates": int(self.columns_without_candidates),
            "min_row_candidate_count": _safe_min(self.row_candidate_counts),
            "max_row_candidate_count": _safe_max(self.row_candidate_counts),
            "min_column_candidate_count": _safe_min(self.column_candidate_counts),
            "max_column_candidate_count": _safe_max(self.column_candidate_counts),
            "applied_rules": ";".join(self.applied_rules),
            "rule_kept_entries": dict(self.rule_kept_entries),
        }


def candidate_pruning_diagnostics(
    cost_matrix: Any,
    *,
    probability_matrix: Any | None = None,
    config: CandidatePruningConfig | Mapping[str, Any] | None = None,
) -> CandidatePruningDiagnostics:
    """Return diagnostics for PyRecEst candidate pruning.

    The returned counts are mask-level diagnostics, not assignment results.
    ``rule_kept_entries`` records the number of entries selected by each active
    criterion independently; because pruning combines criteria by union, these
    per-rule counts are not expected to sum to ``kept_entries``.
    """

    finite_mask = candidate_mask_from_costs(cost_matrix, config=None)
    keep_mask = candidate_mask_from_costs(
        cost_matrix,
        probability_matrix=probability_matrix,
        config=config,
    )
    cfg = candidate_pruning_config_from_mapping(config)
    applied_rules = _applied_rule_names(cfg)
    rule_kept_entries = _rule_kept_entries(
        cost_matrix,
        probability_matrix=probability_matrix,
        config=cfg,
        applied_rules=applied_rules,
    )

    finite_entries = int(np.sum(finite_mask))
    kept_entries = int(np.sum(keep_mask))
    pruned_finite_entries = int(finite_entries - kept_entries)
    row_counts = tuple(int(value) for value in np.sum(keep_mask, axis=1))
    column_counts = tuple(int(value) for value in np.sum(keep_mask, axis=0))
    retention_fraction = (
        float(kept_entries) / float(finite_entries) if finite_entries else 0.0
    )
    pruned_fraction = (
        float(pruned_finite_entries) / float(finite_entries) if finite_entries else 0.0
    )
    return CandidatePruningDiagnostics(
        shape=tuple(int(v) for v in keep_mask.shape),
        total_entries=int(keep_mask.size),
        finite_entries=finite_entries,
        kept_entries=kept_entries,
        pruned_finite_entries=pruned_finite_entries,
        finite_retention_fraction=retention_fraction,
        finite_pruned_fraction=pruned_fraction,
        row_candidate_counts=row_counts,
        column_candidate_counts=column_counts,
        rows_without_candidates=int(sum(value == 0 for value in row_counts)),
        columns_without_candidates=int(sum(value == 0 for value in column_counts)),
        applied_rules=applied_rules,
        rule_kept_entries=rule_kept_entries,
    )


def _applied_rule_names(
    config: CandidatePruningConfig | None,
) -> tuple[str, ...]:
    if config is None:
        return ("all_finite",)
    rules: list[str] = []
    if config.always_keep_finite:
        rules.append("always_keep_finite")
    if config.row_top_k is not None:
        rules.append("row_top_k")
    if config.column_top_k is not None:
        rules.append("column_top_k")
    if config.probability_threshold is not None:
        rules.append("probability_threshold")
    if config.max_cost is not None:
        rules.append("max_cost")
    if config.max_cost_percentile is not None:
        rules.append("max_cost_percentile")
    if not rules:
        rules.append("all_finite_fallback")
    return tuple(rules)


def _rule_kept_entries(
    cost_matrix: Any,
    *,
    probability_matrix: Any | None,
    config: CandidatePruningConfig | None,
    applied_rules: tuple[str, ...],
) -> dict[str, int]:
    if config is None:
        return {"all_finite": int(np.sum(candidate_mask_from_costs(cost_matrix)))}

    counts: dict[str, int] = {}
    for rule in applied_rules:
        rule_config = _single_rule_config(config, rule)
        counts[rule] = int(
            np.sum(
                candidate_mask_from_costs(
                    cost_matrix,
                    probability_matrix=probability_matrix,
                    config=rule_config,
                )
            )
        )
    return counts


def _single_rule_config(
    config: CandidatePruningConfig,
    rule: str,
) -> CandidatePruningConfig:
    kwargs: dict[str, Any] = {"large_cost": config.large_cost}
    if rule == "always_keep_finite":
        kwargs["always_keep_finite"] = True
    elif rule == "row_top_k":
        kwargs["row_top_k"] = config.row_top_k
    elif rule == "column_top_k":
        kwargs["column_top_k"] = config.column_top_k
    elif rule == "probability_threshold":
        kwargs["probability_threshold"] = config.probability_threshold
    elif rule == "max_cost":
        kwargs["max_cost"] = config.max_cost
    elif rule == "max_cost_percentile":
        kwargs["max_cost_percentile"] = config.max_cost_percentile
    elif rule == "all_finite_fallback":
        pass
    else:  # pragma: no cover - defensive programming for future rule names
        raise ValueError(f"unknown candidate-pruning rule {rule!r}")
    return CandidatePruningConfig(**kwargs)


def _safe_min(values: tuple[int, ...]) -> int | None:
    return None if not values else int(min(values))


def _safe_max(values: tuple[int, ...]) -> int | None:
    return None if not values else int(max(values))


__all__ = (
    "CandidatePruningDiagnostics",
    "candidate_pruning_diagnostics",
)
