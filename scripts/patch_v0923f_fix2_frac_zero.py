#!/usr/bin/env python3
# v0.9.23f-fix2 — 2 fixes for v0.9.23f sample():
#
# (1) PRODUCTION bug: frac=0.0 inconsistency. Validation `(0, 1]`
#     rejects 0.0, but docstring says "frac=0 -> empty" and downstream
#     has dead code `if float(frac) == 0.0: n = 0`. Pandas allows
#     frac=0 (returns empty). Fix: change validation `(0, 1]` -> `[0, 1]`;
#     dead code becomes live; docstring consistent.
#
# (2) TEST bug: test_sample_large_n_uses_check_state uses
#     assert_call_order(src, "check_state", "iloc"). But
#     `self.iloc[positions]` is ast.Subscript(value=Attribute(...)),
#     NOT ast.Call — assert_call_order only walks Call nodes, so
#     "iloc" is never found. Fix: direct ast.walk on Subscript nodes
#     whose .value is Attribute(attr="iloc"); verify check_state
#     Call lineno < min(subscript linenos).
#
# Run: uv run python scripts/patch_v0923f_fix2_frac_zero.py

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


# ============ 1. Production: (0, 1] -> [0, 1] ============

def fix_frac_zero():
    print("\n=== production: validation (0,1] -> [0,1] ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "0.0 <= float(frac) <= 1.0" in s:
        print("  already fixed — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '        if frac is not None:\n'
            '            if not (0.0 < float(frac) <= 1.0):\n'
            '                raise ValueError(\n'
            '                    f"sample: frac must be in (0, 1]; got {frac!r}"\n'
            '                )'
        ),
        new=(
            '        if frac is not None:\n'
            '            # v0.9.23f-fix2: [0, 1] (pandas-parity; frac=0 -> empty).\n'
            '            if not (0.0 <= float(frac) <= 1.0):\n'
            '                raise ValueError(\n'
            '                    f"sample: frac must be in [0, 1]; got {frac!r}"\n'
            '                )'
        ),
        label="sample frac validation",
    )
    assert_compiles(p)
    print("  OK")


def fix_docstring_frac():
    print("\n=== docstring: frac [0, 1] ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "frac: fraction of total rows [0, 1]." in s:
        print("  already fixed — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '            frac: fraction of total rows (0 < frac <= 1). n = max(1,\n'
            '                round(N*frac)) for frac > 0; frac=0 -> n=0.'
        ),
        new=(
            '            frac: fraction of total rows [0, 1]. n = max(1,\n'
            '                round(N*frac)) for frac > 0; frac=0 -> empty.'
        ),
        label="sample docstring frac",
    )
    apply_or_fail(
        str(p),
        old=(
            '            ValueError: exactly one of n/frac not given; n > N with\n'
            '                replace=False; frac out of (0, 1]; n < 0; both n and\n'
            '                frac; weights all-zero/null.'
        ),
        new=(
            '            ValueError: exactly one of n/frac not given; n > N with\n'
            '                replace=False; frac out of [0, 1]; n < 0; both n and\n'
            '                frac; weights all-zero/null.'
        ),
        label="sample docstring raises frac",
    )
    assert_compiles(p)
    print("  OK")


# ============ 2. Test: Subscript-aware AST ============

def fix_ast_test():
    print("\n=== test: Subscript-aware AST for iloc ===")
    p = Path("tests/test_sample.py")
    s = p.read_text()

    old = '''def test_sample_large_n_uses_check_state():
    """check_state before iloc (AST)."""
    import inspect
    from cadw.patch_utils import assert_call_order
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.sample)
    assert_call_order(
        src, "check_state", "iloc",
        fn_name="CadwDataFrame.sample",
    )'''

    new = '''def test_sample_large_n_uses_check_state():
    """check_state precedes self.iloc[...] in sample() (AST).

    v0.9.23f-fix2: `self.iloc[positions]` is ast.Subscript(value=
    ast.Attribute(attr="iloc")), NOT ast.Call. assert_call_order only
    walks Call nodes, so "iloc" is never found. Fix: direct ast.walk
    on Subscript nodes with .value.attr == "iloc".
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame
    src = _tw.dedent(_inspect.getsource(CadwDataFrame.sample))
    tree = _ast.parse(src)
    check_state_lines: list[int] = []
    iloc_subscript_lines: list[int] = []
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Call):
            f = n.func
            if (isinstance(f, _ast.Attribute)
                    and f.attr == "check_state"):
                check_state_lines.append(n.lineno)
        elif isinstance(n, _ast.Subscript):
            v = n.value
            if (isinstance(v, _ast.Attribute)
                    and v.attr == "iloc"):
                iloc_subscript_lines.append(n.lineno)
    assert check_state_lines, "check_state not called in sample()"
    assert iloc_subscript_lines, "self.iloc[...] not used in sample()"
    assert min(check_state_lines) < min(iloc_subscript_lines), (
        f"check_state (line {min(check_state_lines)}) must precede "
        f"self.iloc[...] subscript (line {min(iloc_subscript_lines)})"
    )'''

    if old not in s:
        raise SystemExit(
            "test_sample_large_n_uses_check_state anchor not found"
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. Test: add frac=0 explicit coverage ============

def strengthen_frac_zero_test():
    print("\n=== strengthen test_sample_frac_zero (docstring parity) ===")
    p = Path("tests/test_sample.py")
    s = p.read_text()
    if "v0.9.23f-fix2: frac=0.0 returns empty (pandas-parity)" in s:
        print("  already present — skip")
        return

    old = '''def test_sample_frac_zero(tmp_path):
    df = make_id_df(tmp_path, n=100)
    s = df.sample(frac=0.0)
    assert s.count() == 0'''

    new = '''def test_sample_frac_zero(tmp_path):
    """v0.9.23f-fix2: frac=0.0 returns empty (pandas-parity).

    Prior production validated (0, 1] and raised ValueError for 0.0 —
    inconsistent with docstring ("frac=0 -> empty") and downstream
    dead-code (`if float(frac) == 0.0: n = 0`). Pandas allows
    frac=0 -> empty. Fix: [0, 1] validation; dead code live.
    """
    df = make_id_df(tmp_path, n=100)
    s = df.sample(frac=0.0)
    assert s.count() == 0
    # Negative still rejected
    with pytest.raises(ValueError, match="frac must be in \\[0, 1\\]"):
        df.sample(frac=-0.1)'''

    if old not in s:
        raise SystemExit("test_sample_frac_zero anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: frac=0 empty, frac<0 raises, weighted ratio ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": list(range(100))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            # frac=0.0 -> empty
            s0 = df.sample(frac=0.0)
            assert s0.count() == 0, "frac=0 must return empty"
            # frac=0.5 -> 50
            s_half = df.sample(frac=0.5, random_state=42)
            assert s_half.count() == 50
            # frac=-0.1 -> ValueError
            try:
                df.sample(frac=-0.1)
            except ValueError as e:
                assert "frac must be in" in str(e)
            else:
                raise AssertionError("frac<0 must raise")

        # weighted ratio 10:1 with n=50 <= N=100
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "v": list(range(100)),
                "w": [10.0] * 50 + [1.0] * 50,
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            s = df.sample(n=50, weights="w", random_state=42)
            out = td / "o.parquet"
            s.write_parquet(out)
            got = pq_.read_table(out).column("v").to_pylist()
            low = sum(1 for v in got if v < 50)
            high = sum(1 for v in got if v >= 50)
            assert low > high * 3, (
                f"10:1 weights -> low>>high; got low={low} high={high}"
            )

        print("  frac=0 empty + frac<0 raise + weighted ratio ok")

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_sample.py",
         "-q", "-W", "error::SyntaxWarning"],
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


# ============ 6. main ============

def main():
    print("=== v0.9.23f-fix2: frac=0 + Subscript AST ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_frac_zero()
    fix_docstring_frac()
    fix_ast_test()
    strengthen_frac_zero_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23f-fix2: frac=0.0 (production inconsistency) + "
        "Subscript-aware AST test for sample(). "
        "(1) PRODUCTION bug: frac=0.0 inconsistency. Validation "
        "'(0, 1]' rejected 0.0, but docstring said 'frac=0 -> empty' "
        "and downstream had dead code 'if float(frac) == 0.0: n = 0'. "
        "Three places disagreed. Pandas allows frac=0 (returns "
        "empty). Fix: validation (0, 1] -> [0, 1]; dead code live; "
        "docstring + Raises updated. "
        "(2) TEST bug: test_sample_large_n_uses_check_state used "
        "assert_call_order(src, 'check_state', 'iloc'). But "
        "'self.iloc[positions]' is ast.Subscript(value=ast.Attribute"
        "(attr='iloc')), NOT ast.Call — assert_call_order only walks "
        "Call nodes, so 'iloc' never found. Fix: direct ast.walk on "
        "Subscript nodes whose .value.attr == 'iloc'; verify "
        "check_state Call lineno < min(subscript linenos). "
        "(3) Strengthened test_sample_frac_zero: explicit docstring "
        "note + frac=-0.1 still raises (bounds [0, 1]). "
        "Fast: 2044 -> ~2045 passed (test_sample_frac_zero same "
        "count, 1 more assertion inline).",
        self_path="scripts/patch_v0923f_fix2_frac_zero.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23f-fix2 applied. v0.9.23f complete (7.A.2).")


if __name__ == "__main__":
    main()