"""v0.9.97: df.estimate_memory — pre-flight RAM estimate."""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _make_df():
    return CadwDataFrame.from_table(
        pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}),
        memory_guard=MemoryGuard(max_memory_mb=100),
    )


def test_linear_fits_small():
    df = _make_df()
    r = df.estimate_memory("fit_linear", d=100)
    assert r["fits"] is True
    assert r["est_ram_mb"] < 1.0
    assert r["available_mb"] > 0
    assert isinstance(r["bottleneck"], str)


def test_linear_does_not_fit_large():
    df = _make_df()
    r = df.estimate_memory("fit_linear", d=100_000)
    assert r["fits"] is False
    assert any("sgd" in a for a in r["alternatives"])


def test_binseg_does_not_fit_typical_n():
    """binseg is O(n) with 3 prefix arrays (tf=3.0 in YAML).

    At n=10^7: 10^7 * 8 * 3 = 240 MB > 100 MB guard (default).
    NOT refuse_at_guard — binseg has a bounded formula, just a
    large constant factor.
    """
    df = _make_df()
    r = df.estimate_memory("fit_binseg", n=10_000_000)
    assert r["fits"] is False
    assert r["est_ram_mb"] is not None
    assert r["est_ram_mb"] > 100.0
    assert any("pelt" in a for a in r["alternatives"])


def test_missing_axis_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="missing axis"):
        df.estimate_memory("fit_linear")  # no d=


def test_unknown_op_raises():
    df = _make_df()
    with pytest.raises(KeyError, match="not in YAML cards"):
        df.estimate_memory("nonexistent_model_xyz", d=100)


def test_all_verified_cards_have_dims():
    """Sanity: every verified card declares bounded_state_dims
    (except n_scalability='refuse_at_guard')."""
    from cadw.ml._memory_estimate import _load_cards
    cards = _load_cards()
    assert len(cards) > 50, f"expected >=50 cards, got {len(cards)}"
    for op, card in cards.items():
        if card.get("n_scalability") == "refuse_at_guard":
            continue
        assert card.get("bounded_state_dims") is not None, (
            f"{op}: missing bounded_state_dims"
        )

# ── v0.9.101: LDA card dims fix (V-axis, tf=12) ─────────────────────


def test_lda_vocab_fits_small():
    """d=10^4 raw unique tokens → est ~0.96 MB, fits 100 MB guard."""
    df = _make_df()
    r = df.estimate_memory("fit_lda", d=10**4)
    assert r["fits"] is True
    assert 0.5 < r["est_ram_mb"] < 2.0
    assert "d^1" in r["bottleneck"]


def test_lda_vocab_refuses_large():
    """d=10^7 raw unique tokens → est ~960 MB > 100 MB → fits=False."""
    df = _make_df()
    r = df.estimate_memory("fit_lda", d=10**7)
    assert r["fits"] is False
    assert 500.0 < r["est_ram_mb"] < 2000.0
    assert "d^1" in r["bottleneck"]

