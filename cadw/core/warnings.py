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
