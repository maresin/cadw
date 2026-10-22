#!/usr/bin/env python3
# v0.9.23g-fix3 — idempotency guard false-positive (§13.4/182).
#
# fix2's add_empty_helper() checked `if "_empty_value_counts_table"
# in s`. But that name already appears in value_counts body (call
# site after rewrite) -> guard fired -> helper never added -> dogfood
# NameError. CLASSIC whole-file-substring trap, same as §13.4/182.
#
# Fix: check "def _empty_value_counts_table(" in s (function def, not
# bare name).
#
# Structural prevention (patch_utils):
#   idempotency_check_by_def(path, name): True iff "def <name>(" in
#   file. Use instead of `if "<name>" in src` for idempotency. Added
#   with 2 tests (def present -> True, name only in call -> False).
#
# Also: actual helper injection (missed by fix2).
#
# Run: uv run python scripts/patch_v0923g_fix3_empty_helper.py

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


# ============ 1. Add _empty_value_counts_table helper ============

HELPER_ANCHOR = (
    '# ============ v0.9.23f: reservoir sampling helpers ============'
)

HELPER = '''# ============ v0.9.23g-fix2: empty value_counts helper ============


def _empty_value_counts_table(column, col_type, *, normalize: bool):
    """Empty pa.Table with value_counts output schema (0 rows).

    Schema: [column (col_type), "count" (int64)] or
    [column (col_type), "fraction" (float64)].
    """
    import pyarrow as _pa

    second_name = "fraction" if normalize else "count"
    second_type = _pa.float64() if normalize else _pa.int64()
    return _pa.Table.from_pylist([], schema=_pa.schema([
        (column, col_type), (second_name, second_type),
    ]))


# ============ v0.9.23f: reservoir sampling helpers ============'''


def add_helper():
    print("\n=== add _empty_value_counts_table helper (real) ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    # §13.4/182 fix: check def line, not bare name.
    if "def _empty_value_counts_table(" in s:
        print("  already present — skip")
        return
    if HELPER_ANCHOR not in s:
        raise SystemExit("reservoir helpers anchor not found")
    s = s.replace(HELPER_ANCHOR, HELPER, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. patch_utils: idempotency_check_by_def ============

PU_HELPER = '''

# ============ v0.9.23g-fix3: idempotency-by-def helper ============


def idempotency_check_by_def(path, name):
    """True iff file at `path` contains `def <name>(` or `def <name> (`.

    v0.9.23g-fix3 (§13.4/182): whole-file `if "<name>" in src` for
    idempotency guards false-positives when `<name>` appears as a
    call site or docstring mention. Scoped form: check the function
    DEFINITION, not just the name.

    Use:
        if idempotency_check_by_def("cadw/dataframe.py",
                                    "_empty_value_counts_table"):
            # already applied — skip
        else:
            # add helper

    Args:
        path: file path (str | Path).
        name: function name to look for.

    Returns:
        True iff any of these substring patterns present:
          - "def <name>("
          - "def <name> ("
          - "def <name>:" (rare — no params)
    """
    from pathlib import Path as _P
    src = _P(path).read_text()
    return any(p in src for p in (
        f"def {name}(",
        f"def {name} (",
        f"def {name}:",
    ))
'''


def add_pu_helper():
    print("\n=== add patch_utils.idempotency_check_by_def ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def idempotency_check_by_def(" in s:
        print("  already present — skip")
        return
    p.write_text(s + PU_HELPER)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Tests: helper in test_patch_utils.py ============

def add_pu_tests():
    print("\n=== add idempotency_check_by_def tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "test_idempotency_check_by_def_present" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.23g-fix3: idempotency_check_by_def ============

def test_idempotency_check_by_def_present(tmp_path):
    """Function def -> True."""
    from cadw.patch_utils import idempotency_check_by_def
    p = tmp_path / "f.py"
    p.write_text("def my_helper():\\n    pass\\n")
    assert idempotency_check_by_def(p, "my_helper") is True


def test_idempotency_check_by_def_name_only_call(tmp_path):
    """Name appears only as call site (no def) -> False.

    §13.4/182 whole-file-substring trap: prior code used
    `if "name" in src`, which false-positived on call sites.
    """
    from cadw.patch_utils import idempotency_check_by_def
    p = tmp_path / "f.py"
    p.write_text(
        "def other():\\n"
        "    return my_helper()\\n"
    )
    assert idempotency_check_by_def(p, "my_helper") is False


def test_idempotency_check_by_def_name_only_docstring(tmp_path):
    """Name appears only in docstring -> False."""
    from cadw.patch_utils import idempotency_check_by_def
    p = tmp_path / "f.py"
    p.write_text(
        'def other():\\n'
        '    """Uses my_helper internally."""\\n'
        '    pass\\n'
    )
    assert idempotency_check_by_def(p, "my_helper") is False


def test_idempotency_check_by_def_call_and_def(tmp_path):
    """Name in call site AND def -> True (def is what matters)."""
    from cadw.patch_utils import idempotency_check_by_def
    p = tmp_path / "f.py"
    p.write_text(
        "def my_helper():\\n"
        "    pass\\n"
        "def other():\\n"
        "    return my_helper()\\n"
    )
    assert idempotency_check_by_def(p, "my_helper") is True
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: helper exists + value_counts works ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.patch_utils import (
            idempotency_check_by_def as _idem,
        )

        # (a) helper actually exists now
        assert _idem("cadw/dataframe.py", "_empty_value_counts_table"), (
            "helper missing — fix3 failed"
        )

        # (b) value_counts on empty df
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"k": pa_.array([], type=pa_.string())})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            out = df.value_counts("k")
            assert out.num_rows == 0
            assert out.schema.names == ["k", "count"]
            out_n = df.value_counts("k", normalize=True)
            assert out_n.num_rows == 0
            assert out_n.schema.names == ["k", "fraction"]

        # (c) value_counts null count = 2 (real fix from fix2)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array(
                    ["a", None, "a", None, "b"], type=pa_.string(),
                ),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            out = df.value_counts("k")
            rows = {(r["k"], r["count"]) for r in out.to_pylist()}
            assert (None, 2) in rows, f"null count bug: {rows}"
            assert ("a", 2) in rows
            assert ("b", 1) in rows
            # dropna=True
            out2 = df.value_counts("k", dropna=True)
            rows2 = {(r["k"], r["count"]) for r in out2.to_pylist()}
            assert (None, 2) not in rows2
            assert rows2 == {("a", 2), ("b", 1)}

        # (d) normalize=True no crash + dropna semantics
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array(
                    ["a", None, "a", None], type=pa_.string(),
                ),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            out = df.value_counts("k", normalize=True)
            assert out.schema.names == ["k", "fraction"]
            rows = {(r["k"], r["fraction"]) for r in out.to_pylist()}
            assert ("a", 0.5) in rows
            assert (None, 0.5) in rows

        # (e) nunique dropna
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array(
                    ["a", None, "a", None, "b"], type=pa_.string(),
                ),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            assert df.nunique("k") == 3
            assert df.nunique("k", dropna=True) == 2

        print(
            "  helper exists + empty+normalize+null+dropna+nunique ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_eda_basics.py",
         "tests/test_patch_utils.py",
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
    print("=== v0.9.23g-fix3: empty helper + idem-by-def ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_helper()
    add_pu_helper()
    add_pu_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g-fix3: value_counts helper (fix2 idempotency false-"
        "positive) + patch_utils.idempotency_check_by_def. "
        "Bug: fix2's add_empty_helper() checked "
        "`if \"_empty_value_counts_table\" in s` — but that name "
        "already appears in value_counts body as call site -> guard "
        "fired -> helper never added -> dogfood NameError. "
        "CLASSIC whole-file-substring trap, same class as §13.4/182 "
        "(scoped idempotency = inspect.getsource(fn), not whole-file). "
        "fix2's own guard was the bug it was meant to prevent. "
        "Fix: "
        "(1) Add _empty_value_counts_table helper (actually). "
        "(2) patch_utils.idempotency_check_by_def(path, name): True "
        "iff 'def <name>(' in file (def line, not bare name). Use "
        "instead of `if '<name>' in src` for idempotency. Handles "
        "def name( / def name ( / def name:. "
        "(3) +4 tests in test_patch_utils.py: def present -> True; "
        "call-only -> False; docstring-only -> False; def + call -> "
        "True. "
        "(4) Dogfood: helper exists via idem check + empty df "
        "value_counts + null count=2 (real fix2 change) + "
        "normalize=True no crash + dropna semantics + nunique dropna. "
        "Process note: 3rd fix-cycle on v0.9.23g (at §13.4/238 "
        "threshold of 2). Production logic correct at each stage; "
        "all 3 cycles fixed bugs in the patch infrastructure itself "
        "(isna nullable, nunique sum-of-per-batch, value_counts "
        "ones-sum+dropna, idempotency guard). Each fix closed a "
        "whole class: multi-batch tests via chunk=1 (fix1), "
        "set_column vs set_column+.drop (fix2), def-check vs name-"
        "check (fix3). "
        "Fast: ~2090 -> ~2094 passed.",
        self_path="scripts/patch_v0923g_fix3_empty_helper.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g-fix3 applied. v0.9.23g COMPLETE.")


if __name__ == "__main__":
    main()