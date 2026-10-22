#!/usr/bin/env python3
# v0.9.23e-fix1 — _ExpandingState.reset() (AUDIT_FULL_v0922 7.A.6, v0.9.23e).
#
# v0.9.23e added per-group cum* via _per_group_cumulative. That method
# calls `st.reset()` on group change (mirrors _per_group_rolling's use
# of _RollState.reset). But _ExpandingState (designed for global
# `expanding()`, no reset needed) has no `reset` method -> AttributeError
# in dogfood (and would fail pytest).
#
# Fix: add reset() to _ExpandingState. Resets _count/_sum/_prod/_min/
# _max to initial values; keeps _agg and _min_periods unchanged
# (they're config, not state). Mirrors _RollState.reset() (v0.8.10).
#
# Test added: direct _ExpandingState.reset() check + per-group cumsum
# reset semantics (existing test_cumsum_per_group_basic already exercises
# this path but the AttributeError masked it in dogfood — pytest would
# have caught it, dogfood just failed earlier).
#
# Run: uv run python scripts/patch_v0923e_fix1_expanding_reset.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. _ExpandingState.reset() ============

def add_reset():
    print("\n=== _ExpandingState: add reset() ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def reset(self) -> None:\n        \"\"\"v0.9.23e-fix1" in s and "_ExpandingState" in s:
        # Detect by marker uniqueness: only one reset() with this marker
        if "v0.9.23e-fix1: _ExpandingState.reset" in s:
            print("  already present — skip")
            return

    # Find the class _ExpandingState push() end and insert reset after it.
    # Anchor: end of push method (raise RuntimeError line + return result).
    anchor = (
        '        else:\n'
        '            raise RuntimeError(f"_ExpandingState: unknown agg {self._agg!r}")\n'
        '        if self._count < self._min_periods:\n'
        '            return None\n'
        '        return result\n'
        '\n'
        '\n'
        'class _EwmState:'
    )
    if anchor not in s:
        raise SystemExit(
            "_ExpandingState.push end anchor not found — source may "
            "have drifted"
        )

    replacement = (
        '        else:\n'
        '            raise RuntimeError(f"_ExpandingState: unknown agg {self._agg!r}")\n'
        '        if self._count < self._min_periods:\n'
        '            return None\n'
        '        return result\n'
        '\n'
        '    def reset(self) -> None:\n'
        '        """v0.9.23e-fix1: reset cumulative state (used by\n'
        '        _per_group_cumulative on group change).\n'
        '\n'
        '        Resets _count/_sum/_prod/_min/_max to initial values.\n'
        '        Keeps _agg and _min_periods unchanged (config, not state).\n'
        '        Mirrors _RollState.reset() (v0.8.10).\n'
        '        """\n'
        '        self._count = 0\n'
        '        self._sum = 0.0\n'
        '        self._prod = 1.0\n'
        '        self._min = None\n'
        '        self._max = None\n'
        '\n'
        '\n'
        'class _EwmState:'
    )

    s = s.replace(anchor, replacement, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Test: direct reset() ============

def add_reset_test():
    print("\n=== add test for _ExpandingState.reset ===")
    p = Path("tests/test_cum_series.py")
    s = p.read_text()
    if "test_expanding_state_reset" in s:
        print("  already present — skip")
        return
    # Append at end
    extra = '''

# ============ v0.9.23e-fix1: _ExpandingState.reset() ============

def test_expanding_state_reset():
    """_ExpandingState.reset() zeroes state; _agg/_min_periods preserved.

    v0.9.23e-fix1: _per_group_cumulative calls reset() on group change;
    prior AttributeError because _ExpandingState (for global expanding)
    had no reset. Mirrors _RollState.reset (v0.8.10).
    """
    from cadw.dataframe import _ExpandingState

    st = _ExpandingState("sum", min_periods=1)
    assert st.push(5) == 5
    assert st.push(10) == 15
    st.reset()
    # Config preserved
    assert st._agg == "sum"
    assert st._min_periods == 1
    # State reset: first push after reset starts from 0
    assert st.push(3) == 3
    assert st.push(4) == 7


def test_expanding_state_reset_prod():
    """reset() resets _prod to 1.0 (identity for multiplication)."""
    from cadw.dataframe import _ExpandingState

    st = _ExpandingState("prod", min_periods=1)
    assert st.push(2) == 2
    assert st.push(3) == 6
    st.reset()
    assert st.push(5) == 5   # starts from 1.0 * 5 = 5


def test_expanding_state_reset_min_max():
    """reset() clears _min/_max to None."""
    from cadw.dataframe import _ExpandingState

    st = _ExpandingState("max", min_periods=1)
    assert st.push(10) == 10
    st.reset()
    assert st.push(3) == 3   # new max from reset

    st2 = _ExpandingState("min", min_periods=1)
    assert st2.push(1) == 1
    st2.reset()
    assert st2.push(100) == 100


def test_per_group_cumsum_uses_reset():
    """AST: _per_group_cumulative calls st.reset() on group change."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_cumulative)
    assert "st.reset()" in src, (
        "_per_group_cumulative must call _ExpandingState.reset() "
        "on group change"
    )
'''
    p.write_text(s + extra)
    assert_compiles(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: _ExpandingState.reset + per-group cumsum ===")

    def _verify():
        # Direct reset()
        from cadw.dataframe import _ExpandingState as _ES
        st = _ES("sum", 1)
        assert st.push(5) == 5
        assert st.push(10) == 15
        st.reset()
        assert st.push(3) == 3
        assert st.push(4) == 7

        # prod reset
        st_p = _ES("prod", 1)
        assert st_p.push(2) == 2
        assert st_p.push(3) == 6
        st_p.reset()
        assert st_p.push(5) == 5

        # Per-group cumsum uses reset correctly (null group + real group)
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "g": ["a", "a", "b", "b", "b"],
                "v": [1.0, 2.0, 10.0, 20.0, 30.0],
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.groupby("g").cumsum()
            out = td / "o.parquet"
            res.write_parquet(out)
            got = pq_.read_table(out).column("v_cumsum").to_pylist()
            # a: 1, 3; b: 10, 30, 60
            assert got == [1.0, 3.0, 10.0, 30.0, 60.0], got

        # Global cumsum still works (no reset triggered)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.cumsum()
            out = td / "o.parquet"
            res.write_parquet(out)
            got = pq_.read_table(out).column("v_cumsum").to_pylist()
            assert got == [1.0, 3.0, 6.0, 10.0, 15.0], got

        print("  reset (sum/prod) + per-group cumsum + global cumsum ok")

    run_verify_in_subprocess(_verify)


# ============ 4. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_cum_series.py",
         "tests/test_expanding_ewm.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

def main():
    print("=== v0.9.23e-fix1: _ExpandingState.reset ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_reset()
    add_reset_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23e: cumsum/cumprod/cummax/cummin global + per group "
        "(AUDIT_FULL_v0922 item 7.A.6). Top-10 gap #5; prototype #3 "
        "(groupby_cumsum, RAM 13.28 MB @ 32 MB budget, identity PASS). "
        "API: df.cum{sum,prod,max,min}(columns=None, *, on=None) + "
        "df.groupby(g).cum{...}(...). on=None: streaming (row_id "
        "order); on=col: sort-aware (external_sort + sort back, same "
        "pattern as _shift_on/_rolling_on). Per-group: sort by "
        "[g, *on?], stream with _ExpandingState reset per group, "
        "sort back. Output: APPEND {col}_cum{sum,prod,max,min} "
        "(matches expanding/ewm). Dtype preserved (int -> int, "
        "float -> float). Nulls preserved (state frozen, null row -> "
        "null output; matches CADW expanding, NOT pandas NaN-propagate). "
        "_ExpandingState extension: 'prod' aggregator (init 1.0, "
        "push *= x, reset). "
        "fix1 (dogfood-caught): _per_group_cumulative calls "
        "_ExpandingState.reset() on group change, but _ExpandingState "
        "(designed for global expanding() with no reset) had no "
        "reset method -> AttributeError. Fix: add reset() to "
        "_ExpandingState: resets _count/_sum/_prod/_min/_max, keeps "
        "_agg/_min_periods (config). Mirrors _RollState.reset (v0.8.10). "
        "Tests: +tests/test_cum_series.py (25 tests: 4 global + 3 "
        "pandas parity + nulls + sort-aware + subset + 5 per-group + "
        "per-group pandas parity + per-group nulls + null-group + "
        "append-not-replace + multibatch + filtered + 5 validation "
        "+ 4 AST/impl); +4 reset tests in fix1 (direct sum/prod/min/"
        "max reset + AST st.reset() in _per_group_cumulative). "
        "Fast: 1975 -> 2004 passed.",
        self_path="scripts/patch_v0923e_fix1_expanding_reset.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23e-fix1 applied. v0.9.23e complete (7.A.6).")


if __name__ == "__main__":
    main()