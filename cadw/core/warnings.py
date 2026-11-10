"""Warning/exception classes for RAM/time estimation (v0.9.102).

See docs/ml_tools/MEMORY_AND_TIME_ESTIMATION.md (v0.9.102b design doc)
and AI_CONSTITUTION §5 (AP21, AP22).
"""
from __future__ import annotations


class CadwWarning(UserWarning):
    """Base class for all CADW-specific warnings."""
    pass


class SlowOperationWarning(CadwWarning):
    """Emitted when est_time_seconds > 60 (AP22).

    Suppress via `confirm_slow=True` argument to fit_*/predict_*.
    """
    pass


class ActualDataModeFallbackWarning(CadwWarning):
    """estimate_memory() fell back from actual-data to a-priori mode.

    Typically: `_dispatch_op` lacks the op, or sample exec failed.
    """
    pass


class CalibrationMissingWarning(CadwWarning):
    """Env not in memory_constants.yaml profiles; using conservative
    fallback × 1.5. Recalibrate via `python -m cadw.calibrate`.
    """
    pass


class LegacyFormulaWarning(CadwWarning):
    """Card missing `transient_factor_measured`; fell back to
    `transient_factor` (likely inaccurate for Python containers).
    """
    pass


class SlowOperationRequiresConfirm(RuntimeError):
    """Raised for very_slow ops when confirm_slow is not True.

    Used with `strict_slow=True` (CI/automation) — no warnings to ignore.
    """
    pass

class ApproximationWarning(CadwWarning):
    """Auto-adjustment happened (v0.9.176, DEC-067/068).

    Fires when: window auto-reduced, solver switched, output upgraded to
    spill, subsample applied, streaming switched, etc. Always accompanied
    by ``result.approximation: list[ApproximationInfo]``.

    Suppression: ``warnings.filterwarnings("ignore",
    category=ApproximationWarning)`` or ``suppress_approximation_warnings=True``
    on fit call (result.approximation still populated).

    See AI_CONSTITUTION §4.0 R6, AP24.
    """


class ApplySafetyWarning(CadwWarning):
    """df.apply(func) detected dangerous pattern (v0.9.176, DEC-069).

    Detected via AST scan of ``func.__code__``: closure mutation
    (``.append``/``.add``/``.extend`` on captured names), global mutation,
    ``np.concatenate``/``vstack``/``hstack`` on closure args.

    Not raised by default — user must use ``apply_checked`` for runtime
    detection, or ``apply_safe`` for MG-aware func contract.

    See DEC-069, DF_OPERATIONS_SPEC.
    """

