"""Uniform training-log helper for cadw.ml models (PINGIS v0.9.31).

Format (one line per iteration/one-shot fit):
  [fit] <model_name> | iter=N/M | loss=X.XXXXXX | state_bytes=B | rss=Y MiB

Opt-in: gated by env CADW_LOG=1 (default off in production, on in tests
via `CADW_LOG=1`). Never raises — logging must not break a fit.
"""
from __future__ import annotations

import os


def _enabled(verbose: bool | None) -> bool:
    if verbose is not None:
        return bool(verbose)
    return os.environ.get("CADW_LOG", "0") == "1"


def _rss_mib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def log_fit(
    model: str,
    *,
    iter_i: int | None = None,
    iter_n: int | None = None,
    loss: float | None = None,
    state_bytes: int | None = None,
    verbose: bool | None = None,
) -> None:
    """Emit one uniform `[fit]` line. Iterless fit: iter_i=None.

    Example:
      log_fit("gbdt", iter_i=3, iter_n=50, loss=0.123456,
              state_bytes=4096)
      -> [fit] gbdt | iter=3/50 | loss=0.123456 | state_bytes=4096 | rss=123.4 MiB
    """
    if not _enabled(verbose):
        return
    parts = [f"[fit] {model}"]
    if iter_i is not None and iter_n is not None:
        parts.append(f"iter={iter_i}/{iter_n}")
    if loss is not None:
        parts.append(f"loss={float(loss):.6f}")
    if state_bytes is not None:
        parts.append(f"state_bytes={int(state_bytes)}")
    parts.append(f"rss={_rss_mib():.1f} MiB")
    try:
        print(" | ".join(parts), flush=True)
    except Exception:
        pass
