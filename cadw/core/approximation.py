"""ApproximationInfo + BudgetTier (v0.9.176, DEC-067/068).

ApproximationInfo — describes any auto-adjustment (window, solver, output,
subsample, ...). Attached to results as ``result.approximation: list | None``.

BudgetTier — 2-tier model: operational (fits) or insufficient (no variant fits).
See AI_CONSTITUTION §4.0 R2, DEC-067.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


_APPROX_VALID_TYPES = frozenset({
    "window_adjusted",     # window auto-reduced to fit budget
    "window_truncated",    # W_eff < n (lost_range in predict_in_sample)
    "solver_switch",       # closed -> sgd
    "streaming_switch",    # batch -> online
    "subsample",           # n reduced to cap (chains, boruta)
    "output_spilled",      # memory -> auto (SpillArray)
    "round_by_round",      # GBDT iteration-level spill
    "blocked",             # ALS/NMF K-blocked
    "sparse",              # dense -> sparse components
    "recursive",           # direct -> recursive forecast
})

_APPROX_VALID_AXES = frozenset({"n", "d", "K", "V", "model", "other"})


@dataclass
class ApproximationInfo:
    """One auto-adjustment event.

    ``type``         — see _APPROX_VALID_TYPES.
    ``axis``         — which axis was adjusted: n, d, K, V, model, other.
    ``requested``    — user's requested value (None = unbounded/auto).
    ``effective``    — actual value used.
    ``reason``       — "budget_limit" | "n_smaller_than_requested" | "explicit".
    ``lost_range``   — for window_truncated: (start, end) of lost range.
    ``accuracy_impact`` — "none" | "negligible" | "estimated: ~X%" | "documented <ref>".
    """

    type: str
    axis: str
    requested: Any = None
    effective: Any = None
    reason: str = "budget_limit"
    lost_range: tuple[int, int] | None = None
    accuracy_impact: str = "none"

    def __post_init__(self) -> None:
        if self.type not in _APPROX_VALID_TYPES:
            raise ValueError(
                f"ApproximationInfo: invalid type {self.type!r}; "
                f"expected one of {sorted(_APPROX_VALID_TYPES)}"
            )
        if self.axis not in _APPROX_VALID_AXES:
            raise ValueError(
                f"ApproximationInfo: invalid axis {self.axis!r}; "
                f"expected one of {sorted(_APPROX_VALID_AXES)}"
            )

    def __str__(self) -> str:
        return (
            f"ApproximationInfo(type={self.type!r}, axis={self.axis!r}, "
            f"requested={self.requested!r}, effective={self.effective!r}, "
            f"accuracy_impact={self.accuracy_impact!r})"
        )


class BudgetTier(Enum):
    """2-tier insufficient model (DEC-067).

    OPERATIONAL          — budget >= min_ram(best_variant); works (with auto-switch if needed).
    INSUFFICIENT_NO_ALT  — budget < min_ram(best_variant); honest refuse with options.
    """

    OPERATIONAL = "operational"
    INSUFFICIENT_NO_ALT = "insufficient_no_alt"

class ApproximationMixin:
    """Mixin for result classes: discloses approximations (v0.9.178).

    Any model/result class inherits this to expose:
      self.approximation: list[ApproximationInfo] | None

    - None: exact (no adjustments).
    - [info]: one auto-adjustment.
    - [info1, info2]: cascaded (e.g., window_adjusted + window_truncated).

    Helpers:
      _clear_approximation() — reset before fit (idempotent).
      _record_approximation(info) — append + emit ApproximationWarning
      (R6: silent approximation = AP24 violation).

    Users:
      - cadw.ts.models.ForecasterBase (all TS forecasters)
      - cadw.ts.thf.TemporalHierarchy

    See DEC-067/068, AI_CONSTITUTION §4.0 R6, AP24.
    """

    # Class-level default; instance attr set on first record.
    approximation: "list[ApproximationInfo] | None" = None

    def _clear_approximation(self) -> None:
        """Reset approximation to None (call at start of fit)."""
        self.approximation = None

    def _record_approximation(self, info: "ApproximationInfo") -> None:
        """Append one ApproximationInfo + emit ApproximationWarning.

        Parameters
        ----------
        info : ApproximationInfo
            The adjustment record (window_adjusted, solver_switch, ...).

        Raises
        ------
        TypeError
            If info is not ApproximationInfo.
        """
        if not isinstance(info, ApproximationInfo):
            raise TypeError(
                f"_record_approximation: expected ApproximationInfo, "
                f"got {type(info).__name__}"
            )

        if self.approximation is None:
            self.approximation = []
        self.approximation.append(info)

        # Emit warning (caller can suppress via warnings.filterwarnings)
        import warnings as _warnings
        from cadw.core.warnings import ApproximationWarning
        _warnings.warn(
            ApproximationWarning(str(info)),
            stacklevel=3,
        )

    def _record_approximations_batch(
        self,
        infos: "list[ApproximationInfo]",
    ) -> None:
        """Append multiple ApproximationInfo + emit one summary warning.

        Use when N > 1 approximations happen in one fit/detect call
        (e.g., per-series window adjustment for N series). Avoids N
        warnings, keeps one actionable message.

        Parameters
        ----------
        infos : list[ApproximationInfo]
            Non-empty list. No-op if empty.

        Raises
        ------
        TypeError
            If any element is not ApproximationInfo.
        """
        if not infos:
            return
        for info in infos:
            if not isinstance(info, ApproximationInfo):
                raise TypeError(
                    f"_record_approximations_batch: expected ApproximationInfo, "
                    f"got {type(info).__name__}"
                )

        if self.approximation is None:
            self.approximation = []
        self.approximation.extend(infos)

        import warnings as _warnings
        from cadw.core.warnings import ApproximationWarning
        if len(infos) == 1:
            msg = str(infos[0])
        else:
            msg = (
                f"{len(infos)} approximations applied "
                f"(first: {infos[0]})"
            )
        _warnings.warn(ApproximationWarning(msg), stacklevel=3)

