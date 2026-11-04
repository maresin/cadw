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

# ── v0.9.118: LDA formula (v0.9.117: k, d, avg_word_len) ──
# Old (v0.9.101) formula: d * 96 + 1024 (vocab dict only).
# New (v0.9.117) formula:
#   k*d*f64 + d*list_ptr + d*py_str_overhead + d*avg_word_len
#   + d*dict_entry + 512
# At k=10, d=10^4, avg_word_len=24: est = 2.77 MB (was 0.96 MB).
# At k=10, d=3*10^6, avg_word_len=24: est = 831 MB (was 960 MB).


def test_lda_vocab_fits_small():
    """v0.9.118: k=10, d=10^4, avg_word_len=24 -> est ~2.77 MB, fits 100 MB."""
    df = _make_df()
    r = df.estimate_memory("fit_lda", k=10, d=10**4, avg_word_len=24)
    assert r["fits"] is True
    assert 2.0 < r["est_ram_mb"] < 4.0
    # bottleneck is the full ram_formula string (v0.9.118: not a
    # dominant-term summary). est_ram_mb range already validates
    # formula correctness; here just sanity that bottleneck is non-empty
    # and mentions 'd' (the LDA vocab axis).
    assert r["bottleneck"] and "d" in r["bottleneck"]


def test_lda_vocab_refuses_large():
    """v0.9.118: k=10, d=3*10^6, avg_word_len=24 -> est ~831 MB > 100 MB."""
    df = _make_df()
    r = df.estimate_memory("fit_lda", k=10, d=3*10**6, avg_word_len=24)
    assert r["fits"] is False
    assert 500.0 < r["est_ram_mb"] < 2000.0

