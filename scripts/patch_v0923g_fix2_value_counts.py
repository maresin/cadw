#!/usr/bin/env python3
# v0.9.23g-fix2 — 2 production bugs + dropna param in value_counts.
#
# (1) value_counts(normalize=True) CRASH: KeyError "Column 'count'
#     not found". Code: result.set_column(1, "fraction", frac).drop(
#     ["count"]). But set_column REPLACES the column at index 1 ->
#     name becomes "fraction", "count" no longer exists -> .drop fails.
#     Fix: set_column(1, "fraction", frac) WITHOUT .drop.
#
# (2) value_counts null count WRONG (silent). groupby_aggregate with
#     ("count") gives non-null count per group (CountOp semantics:
#     skip nulls). Null group → count=0. But value_counts must count
#     SIZE (every row, incl. nulls). Fix: use sum of a constant-1
#     column ("__vco") — each row contributes 1, null group gets
#     correct size.
#
# (3) Add dropna param (default False, matches CADW groupby default
#     per §13.4/218 null-key preservation; dropna=True for pandas-
#     parity). Same for nunique (default False, param added).
#
# (4) Empty input explicit handling (groupby_aggregate raises ValueError
#     on empty input; value_counts should return empty table).
#
# +tests: value_counts null chunk=1 (regression for #2),
#          value_counts dropna=True, value_counts empty,
#          nunique dropna=True, nunique empty.
#
# Run: uv run python scripts/patch_v0923g_fix2_value_counts.py

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


# ============ 1. Rewrite value_counts ============

OLD_VC = '''        from cadw.ops.groupby import groupby_aggregate as _gba

        if column not in self._columns:
            raise KeyError(
                f"value_counts: column {column!r} not in {self._columns}"
            )

        result = _gba(
            self._iter_filtered_batches(),
            [column],
            [column],
            ["count"],
            self._memory_guard,
            self._spill_dir / "value_counts",
            log=False,
        )
        # Rename {column}_count -> count
        result = result.rename_columns([column, "count"])
        if sort:
            # In-memory sort: K rows (bounded by cardinality); safe.
            import pyarrow.compute as _pc
            order = _pc.sort_indices(
                result, sort_keys=[("count", "descending")],
            )
            result = result.take(order)
        if normalize:
            n = self.count()
            if n == 0:
                # No rows — empty table, fraction column = count cast to f64
                import pyarrow.compute as _pc
                frac = _pc.cast(result.column("count"), pa.float64())
                return result.set_column(
                    1, "fraction", frac,
                ).drop(["count"])
            import pyarrow.compute as _pc
            frac = _pc.divide(
                _pc.cast(result.column("count"), pa.float64()),
                pa.scalar(float(n), type=pa.float64()),
            )
            return result.set_column(1, "fraction", frac).drop(["count"])
        return result'''

NEW_VC = '''        import pyarrow.compute as _pc
        from cadw.ops.groupby import groupby_aggregate as _gba

        if column not in self._columns:
            raise KeyError(
                f"value_counts: column {column!r} not in {self._columns}"
            )

        col_type = self._dataset.schema.field(column).type
        # Empty-input fast path: groupby_aggregate raises ValueError
        # on empty input (v0.9.23g-fix2).
        if self.count() == 0:
            return _empty_value_counts_table(
                column, col_type, normalize=normalize,
            )

        # v0.9.23g-fix2: ones-sum, NOT count. groupby_aggregate(count)
        # gives non-null count per group (CountOp skips nulls) -> null
        # group gets 0. value_counts must count SIZE (every row,
        # including nulls). Ones-sum: each row contributes 1.
        def _with_one():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name="value_counts.input",
                )
                if b.num_rows == 0:
                    continue
                if dropna:
                    mask = _pc.is_valid(b.column(column))
                    b = b.filter(mask)
                    if b.num_rows == 0:
                        continue
                one = pa.array([1] * b.num_rows, type=pa.int64())
                yield b.append_column("__vco", one)

        try:
            result = _gba(
                _with_one(),
                [column],
                ["__vco"],
                ["sum"],
                self._memory_guard,
                self._spill_dir / "value_counts",
                log=False,
            )
        except ValueError as e:
            # groupby_aggregate raises ValueError on empty input. With
            # dropna=True and all-null column, filtered is empty ->
            # gba sees empty -> ValueError. Return empty table.
            if "empty" in str(e).lower():
                return _empty_value_counts_table(
                    column, col_type, normalize=normalize,
                )
            raise
        result = result.rename_columns([column, "count"])
        if sort:
            order = _pc.sort_indices(
                result, sort_keys=[("count", "descending")],
            )
            result = result.take(order)
        if normalize:
            n = self.count()
            if n == 0:
                # Shouldn't reach (empty handled above); defensive.
                return _empty_value_counts_table(
                    column, col_type, normalize=True,
                )
            frac = _pc.divide(
                _pc.cast(result.column("count"), pa.float64()),
                pa.scalar(float(n), type=pa.float64()),
            )
            # v0.9.23g-fix2: set_column replaces index 1 -> no .drop.
            return result.set_column(1, "fraction", frac)
        return result'''

def rewrite_value_counts():
    print("\n=== rewrite value_counts (ones-sum + dropna + no-drop) ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "v0.9.23g-fix2: ones-sum, NOT count" in s:
        print("  already fixed — skip")
        return
    if OLD_VC not in s:
        raise SystemExit("value_counts body anchor not found")
    s = s.replace(OLD_VC, NEW_VC, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Signature: add dropna to value_counts ============

def add_dropna_to_vc_signature():
    print("\n=== add dropna param to value_counts signature ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if 'def value_counts(\n        self,\n        column: str,\n        *,\n        normalize: bool = False,\n        sort: bool = True,\n        dropna: bool = False,' in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '    def value_counts(\n'
            '        self,\n'
            '        column: str,\n'
            '        *,\n'
            '        normalize: bool = False,\n'
            '        sort: bool = True,\n'
            '    ) -> pa.Table:'
        ),
        new=(
            '    def value_counts(\n'
            '        self,\n'
            '        column: str,\n'
            '        *,\n'
            '        normalize: bool = False,\n'
            '        sort: bool = True,\n'
            '        dropna: bool = False,\n'
            '    ) -> pa.Table:'
        ),
        label="value_counts_dropna_signature",
    )
    assert_compiles(p)
    print("  OK")


# ============ 3. _empty_value_counts_table helper ============

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


def add_empty_helper():
    print("\n=== add _empty_value_counts_table helper ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "_empty_value_counts_table" in s:
        print("  already present — skip")
        return
    if HELPER_ANCHOR not in s:
        raise SystemExit("reservoir helpers anchor not found")
    s = s.replace(HELPER_ANCHOR, HELPER, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. nunique: add dropna param ============

OLD_NUNIQUE_SIG = '''    def nunique(self, column=None):
        """Count distinct values. Int for one column, dict for all.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.3, Top-10 gap #7).

        Args:
            column: str → int (nunique of that column); None → dict[str,
                int] for all columns.
        """'''

NEW_NUNIQUE_SIG = '''    def nunique(self, column=None, *, dropna: bool = False):
        """Count distinct values. Int for one column, dict for all.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.3, Top-10 gap #7).

        Args:
            column: str → int (nunique of that column); None → dict[str,
                int] for all columns.
            dropna: if True, null is NOT counted as a distinct value
                (pandas-parity). Default False (CADW groupby default
                per §13.4/218 — null-key preservation).
        """'''

def add_dropna_to_nunique_sig():
    print("\n=== add dropna param to nunique signature ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if OLD_NUNIQUE_SIG in s:
        s = s.replace(OLD_NUNIQUE_SIG, NEW_NUNIQUE_SIG, 1)
        p.write_text(s)
        assert_compiles(p)
        print("  OK")
    elif "def nunique(self, column=None, *, dropna: bool = False):" in s:
        print("  already present — skip")
    else:
        raise SystemExit("nunique signature anchor not found")


# ============ 5. nunique body: use dropna ============

OLD_NUNIQUE_BODY = '''        if column is not None:
            if column not in self._columns:
                raise KeyError(
                    f"nunique: column {column!r} not in {self._columns}"
                )
            return _nunique_one(column)

        # All columns: per-column groupby_aggregate.
        out: dict = {}
        for c in self._columns:
            out[c] = _nunique_one(c)
        return out'''

NEW_NUNIQUE_BODY = '''        if column is not None:
            if column not in self._columns:
                raise KeyError(
                    f"nunique: column {column!r} not in {self._columns}"
                )
            return _nunique_one(column, dropna)

        # All columns: per-column groupby_aggregate.
        out: dict = {}
        for c in self._columns:
            out[c] = _nunique_one(c, dropna)
        return out'''

def update_nunique_body():
    print("\n=== nunique body: pass dropna ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if OLD_NUNIQUE_BODY not in s:
        if "return _nunique_one(column, dropna)" in s:
            print("  already fixed — skip")
            return
        raise SystemExit("nunique body anchor not found")
    s = s.replace(OLD_NUNIQUE_BODY, NEW_NUNIQUE_BODY, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 6. _nunique_one: accept dropna ============

OLD_NUNIQUE_ONE = '''        def _nunique_one(col_name: str) -> int:
            # groupby_aggregate over [col] → one row per unique value.
            # Number of output rows = global unique count.
            result = _gba(
                self._iter_filtered_batches(),
                [col_name],
                [col_name],
                ["count"],
                self._memory_guard,
                self._spill_dir / f"nunique_{col_name}",
                log=False,
            )
            return result.num_rows'''

NEW_NUNIQUE_ONE = '''        def _nunique_one(col_name: str, dropna_inner: bool) -> int:
            # groupby_aggregate over [col] → one row per unique value.
            # Number of output rows = global unique count.
            if dropna_inner:
                import pyarrow.compute as _pc
                def _filtered():
                    for b in self._iter_filtered_batches():
                        self._memory_guard.check_batch(
                            b, name=f"nunique[{col_name}]",
                        )
                        mask = _pc.is_valid(b.column(col_name))
                        f = b.filter(mask)
                        if f.num_rows > 0:
                            yield f
                it = _filtered()
            else:
                it = self._iter_filtered_batches()
            try:
                result = _gba(
                    it,
                    [col_name],
                    [col_name],
                    ["count"],
                    self._memory_guard,
                    self._spill_dir / f"nunique_{col_name}",
                    log=False,
                )
            except ValueError as e:
                # groupby_aggregate raises on empty input. dropna=True
                # + all-null column -> filtered is empty -> 0 uniques.
                if "empty" in str(e).lower():
                    return 0
                raise
            return result.num_rows'''

def update_nunique_one():
    print("\n=== _nunique_one: accept dropna ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _nunique_one(col_name: str, dropna_inner: bool) -> int:" in s:
        print("  already present — skip")
        return
    if OLD_NUNIQUE_ONE not in s:
        raise SystemExit("_nunique_one anchor not found")
    s = s.replace(OLD_NUNIQUE_ONE, NEW_NUNIQUE_ONE, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 7. Regression tests ============

def add_regression_tests():
    print("\n=== add value_counts null / dropna / empty regression tests ===")
    p = Path("tests/test_eda_basics.py")
    s = p.read_text()
    if "test_value_counts_null_multibatch" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.23g-fix2: value_counts null/dropna/empty ============

def test_value_counts_null_multibatch(tmp_path):
    """v0.9.23g-fix2 SILENT BUG regression: null count via size.

    Prior used groupby(count) — CountOp skips nulls -> null group
    count=0. Fix: ones-sum -> every row contributes 1 -> null group
    count = number of nulls. chunk=1 (worst case).
    """
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    out = df.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert ("a", 2) in rows
    assert ("b", 1) in rows
    assert (None, 2) in rows, f"null count must be 2, got {rows}"


def test_value_counts_dropna_true(tmp_path):
    """dropna=True excludes null group (pandas-parity)."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    out = df.value_counts("k", dropna=True)
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert rows == {("a", 2), ("b", 1)}


def test_value_counts_dropna_true_all_null(tmp_path):
    """dropna=True on all-null column -> empty table."""
    df = make_df(tmp_path, {
        "k": pa.array([None, None], type=pa.string()),
    })
    out = df.value_counts("k", dropna=True)
    assert out.num_rows == 0
    assert out.schema.names == ["k", "count"]


def test_value_counts_empty_df(tmp_path):
    """Empty df -> empty table with proper schema (no crash)."""
    df = make_df(tmp_path, {
        "k": pa.array([], type=pa.string()),
    })
    out = df.value_counts("k")
    assert out.num_rows == 0
    assert out.schema.names == ["k", "count"]


def test_value_counts_normalize_null_multibatch(tmp_path):
    """normalize with nulls: fraction includes null count in denominator."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None], type=pa.string(),
        ),
    }, chunk=1)
    out = df.value_counts("k", normalize=True)
    rows = {(r["k"], r["fraction"]) for r in out.to_pylist()}
    # Each value (a, null) has 2/4 = 0.5
    assert ("a", 0.5) in rows
    assert (None, 0.5) in rows


def test_nunique_dropna_true(tmp_path):
    """dropna=True excludes null (pandas-parity)."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    assert df.nunique("k", dropna=True) == 2


def test_nunique_dropna_true_all_null(tmp_path):
    """dropna=True on all-null column -> 0."""
    df = make_df(tmp_path, {
        "k": pa.array([None, None], type=pa.string()),
    })
    assert df.nunique("k", dropna=True) == 0


def test_nunique_dropna_false_default(tmp_path):
    """Default dropna=False: null counted as distinct (CADW groupby)."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    assert df.nunique("k") == 3  # a, b, null
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 8. Dogfood ============

def dogfood():
    print("\n=== dogfood: value_counts null/dropna/normalize/empty ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # (a) null count now correct (2), not 0
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

        # (b) dropna=True excludes null
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
            out = df.value_counts("k", dropna=True)
            rows = {(r["k"], r["count"]) for r in out.to_pylist()}
            assert (None, 2) not in rows
            assert rows == {("a", 2), ("b", 1)}

        # (c) normalize=True no longer crashes
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"k": ["a", "a", "b", "b"]})
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
            assert ("b", 0.5) in rows

        # (d) empty df
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
            # normalize=True also empty
            out_n = df.value_counts("k", normalize=True)
            assert out_n.num_rows == 0
            assert out_n.schema.names == ["k", "fraction"]

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
            assert df.nunique("k") == 3          # a, b, null
            assert df.nunique("k", dropna=True) == 2  # a, b

        print(
            "  null count ok + dropna=True ok + normalize no crash + "
            "empty ok + nunique dropna ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 9. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_eda_basics.py",
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


# ============ 10. main ============

def main():
    print("=== v0.9.23g-fix2: value_counts bugs + dropna ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    rewrite_value_counts()
    add_dropna_to_vc_signature()
    add_empty_helper()
    add_dropna_to_nunique_sig()
    update_nunique_body()
    update_nunique_one()
    add_regression_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g-fix2: value_counts nulls/normalize/dropna + nunique "
        "dropna. "
        "Two production bugs: "
        "(1) normalize=True CRASH: KeyError 'Column count not found'. "
        "result.set_column(1, 'fraction', frac).drop(['count']) — "
        "set_column REPLACES column at index 1 (name becomes "
        "'fraction'), old 'count' gone, .drop fails. Fix: "
        "set_column without .drop. "
        "(2) value_counts null count WRONG (silent). groupby_"
        "aggregate(count) = CountOp semantics (skip nulls) -> null "
        "group count=0. value_counts needs SIZE (all rows). Fix: "
        "ones-sum via __vco constant-1 column -> every row "
        "contributes 1 -> null group = number of nulls. "
        "Design gap closed: "
        "(3) dropna=False param on value_counts and nunique. Default "
        "False (CADW groupby default §13.4/218, null-key "
        "preservation); dropna=True for pandas-parity. "
        "(4) Empty-input explicit handling (_empty_value_counts_"
        "table helper; groupby_aggregate raises ValueError on "
        "empty). "
        "(5) All-null column + dropna=True -> empty table (not "
        "crash). "
        "+tests/test_eda_basics.py (8 new: value_counts_null_"
        "multibatch chunk=1 (SILENT BUG regression), value_counts_"
        "dropna_true, value_counts_dropna_true_all_null, value_counts_"
        "empty_df, value_counts_normalize_null_multibatch, nunique_"
        "dropna_true, nunique_dropna_true_all_null, nunique_dropna_"
        "false_default). "
        "Note on CADW vs pandas default: pandas value_counts/nunique "
        "default dropna=True; CADW default dropna=False (consistent "
        "with CADW groupby). Users wanting pandas-parity pass "
        "dropna=True. "
        "Fast: ~2082 -> ~2090 passed.",
        self_path="scripts/patch_v0923g_fix2_value_counts.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g-fix2 applied. v0.9.23g COMPLETE (7.A.5 + 7.A.3).")


if __name__ == "__main__":
    main()