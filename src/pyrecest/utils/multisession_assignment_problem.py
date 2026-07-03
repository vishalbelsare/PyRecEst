"""Stable problem/run containers for multi-session assignment.

The containers in this module are intentionally domain-neutral: they carry
pairwise cost matrices, session sizes, scalar solver knobs, optional metadata,
and the resulting PyRecEst assignment result.  Dataset-specific code can convert
its observations into these generic containers and then call PyRecEst's
multi-session assignment solver without introducing domain-specific APIs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .multisession_assignment import (
    MultiSessionAssignmentResult,
    PairwiseCostsInput,
    SessionSizesInput,
    solve_multisession_assignment,
)

SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class MultiSessionAssignmentProblem:
    """Domain-neutral input bundle for multi-session assignment.

    Parameters mirror :func:`pyrecest.utils.solve_multisession_assignment` so the
    object can be used as a lightweight, serializable boundary between
    domain-specific preprocessing and PyRecEst's generic assignment solver.
    """

    pairwise_costs: PairwiseCostsInput
    session_sizes: SessionSizesInput | None = None
    start_cost: float = 0.0
    end_cost: float = 0.0
    gap_penalty: float = 0.0
    cost_threshold: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def solve(self) -> MultiSessionAssignmentResult:
        """Solve this assignment problem with PyRecEst's default solver."""

        return solve_multisession_assignment(
            self.pairwise_costs,
            self.session_sizes,
            start_cost=float(self.start_cost),
            end_cost=float(self.end_cost),
            gap_penalty=float(self.gap_penalty),
            cost_threshold=self.cost_threshold,
        )

    def solved(self) -> "MultiSessionAssignmentRun":
        """Return a run container containing this problem and its solved result."""

        return MultiSessionAssignmentRun(problem=self, result=self.solve())

    def with_pairwise_costs(
        self,
        pairwise_costs: PairwiseCostsInput,
        *,
        metadata_update: Mapping[str, Any] | None = None,
    ) -> "MultiSessionAssignmentProblem":
        """Return a copy with replaced pairwise costs and optional metadata update."""

        metadata = dict(self.metadata)
        if metadata_update:
            metadata.update(dict(metadata_update))
        return replace(self, pairwise_costs=pairwise_costs, metadata=metadata)


@dataclass(frozen=True)
class MultiSessionAssignmentRun:
    """Assignment problem plus the corresponding solver result."""

    problem: MultiSessionAssignmentProblem
    result: MultiSessionAssignmentResult

    @classmethod
    def solve(
        cls,
        problem: MultiSessionAssignmentProblem,
    ) -> "MultiSessionAssignmentRun":
        """Solve ``problem`` and return the problem/result container."""

        return cls(problem=problem, result=problem.solve())

    @property
    def n_tracks(self) -> int:
        """Return the number of recovered tracks."""

        return len(self.result.tracks)

    @property
    def n_matched_edges(self) -> int:
        """Return the number of selected pairwise edges."""

        return len(self.result.matched_edges)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact JSON/CSV-friendly run summary."""

        return {
            "n_tracks": int(self.n_tracks),
            "n_matched_edges": int(self.n_matched_edges),
            "total_cost": float(self.result.total_cost),
            "start_cost": float(self.problem.start_cost),
            "end_cost": float(self.problem.end_cost),
            "gap_penalty": float(self.problem.gap_penalty),
            "cost_threshold": (
                None
                if self.problem.cost_threshold is None
                else float(self.problem.cost_threshold)
            ),
        }


def session_edge_pairs(
    num_sessions: int,
    *,
    max_gap: int = 1,
) -> tuple[SessionEdge, ...]:
    """Return forward session edges admitted by a max-gap policy.

    ``max_gap=1`` yields consecutive edges only.  Larger values admit skip edges
    up to that number of session steps.
    """

    num_sessions = _positive_or_zero_integer(num_sessions, name="num_sessions")
    max_gap = _positive_integer(max_gap, name="max_gap")
    return tuple(
        (source, target)
        for source in range(max(0, num_sessions - 1))
        for target in range(source + 1, min(num_sessions, source + max_gap + 1))
    )


def _positive_or_zero_integer(value: Any, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    try:
        value_float = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not value_float.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value_float)


__all__ = (
    "MultiSessionAssignmentProblem",
    "MultiSessionAssignmentRun",
    "SessionEdge",
    "session_edge_pairs",
)
