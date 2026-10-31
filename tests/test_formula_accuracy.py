"""v0.9.103: verify ram_formula matches measure_peak.

Fixed batch (per STABILITY_PROTOCOL §4). Matrix varies non-n axes
(S, w for matrix_profile). OOC check at n=10^4..10^6 (all > S+w-1).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import measure_peak
from cadw.ml._memory_estimate import _load_cards, evaluate_formula


def _load_cards_fresh():
    _load_cards.cache_clear()
    return _load_cards()


def _make_ts_df(n: int):
    rng = np.random.RandomState(42)
    return CadwDataFrame.from_table(
        pa.table({"value": rng.randn(n).tolist()}),
        memory_guard=MemoryGuard(max_memory_mb=500),
    )


def _has_formula(op: str) -> bool:
    card = _load_cards_fresh().get(op)
    return bool(card and card.get("ram_formula"))


# ── Matrix profile: formula × S × w (n=10^5 fixed) ─────────────


@pytest.mark.parametrize("S", [1024, 2048, 4096])
@pytest.mark.parametrize("w", [16, 32, 64])
def test_matrix_profile_formula_accuracy(S, w):
    """9 cells: S x w, n=10^5 (>> S+w-1). predicted vs measured ±30%."""
    if not _has_formula("fit_matrix_profile"):
        pytest.skip("fit_matrix_profile: no ram_formula")

    from cadw.ml.matrix_profile import fit_matrix_profile

    n = 10**5
    df = _make_ts_df(n)
    predicted = evaluate_formula("fit_matrix_profile", S=S, w=w)["bytes"]

    r = measure_peak(
        fit_matrix_profile, df, "value",
        window=w, max_samples=S,
        memory_guard=MemoryGuard(max_memory_mb=500),
    )
    measured = r["total_bytes"]

    ratio = predicted / max(measured, 1.0)
    assert 0.7 <= ratio <= 1.4, (
        f"S={S} w={w}: predicted {predicted/1e6:.2f}MB, "
        f"measured {measured/1e6:.2f}MB, ratio {ratio:.2f}"
    )


# ── Matrix profile: OOC (peak independent of n once n > S+w-1) ──


def test_matrix_profile_ooc():
    """peak(n=10^6) / peak(n=10^4) < 2.0 — bounded state by S, not n."""
    if not _has_formula("fit_matrix_profile"):
        pytest.skip("fit_matrix_profile: no ram_formula")

    from cadw.ml.matrix_profile import fit_matrix_profile

    S, w = 4096, 32
    peaks = []
    for n in (10**4, 10**5, 10**6):
        df = _make_ts_df(n)
        peak = measure_peak(
            fit_matrix_profile, df, "value",
            window=w, max_samples=S,
            memory_guard=MemoryGuard(max_memory_mb=500),
        )["total_bytes"]
        peaks.append(peak)

    ratio = peaks[-1] / max(peaks[0], 1.0)
    assert ratio < 2.0, (
        f"peak n=10^6 / n=10^4 = {ratio:.2f}, expected < 2.0. "
        f"peaks={peaks}"
    )
