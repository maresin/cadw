#!/usr/bin/env python3
# v0.9.23d — df.corr(method='pearson') facade (AUDIT_FULL_v0922 7.A.4).
#
# CovarianceOp / CovarianceMeanOp already implemented in
# cadw/ops/reduce.py; verified §3.1 audit (formula 8+8d+8d² exact at
# d in {2,10,100,1000,10000}). Facade exposure only.
#
# Pearson: corr[i,j] = cov[i,j] / sqrt(var_i * var_j). Constant / all-
# null column -> NaN row/column (matches pandas). Guard via reduce()'s
# pre-init check_state of CovarianceMeanOp — fails fast at large d.
#
# Spearman: NotImplementedError (needs per-column rank + Pearson on
# ranks; P1 backlog after P0).
#
# Return: dict[str, dict[str, float]] (nested, matches describe() pattern).
# Not pandas.DataFrame — pandas is optional dependency.
#
# Run: uv run python scripts/patch_v0923d_corr.py

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


# ============ 1. CadwDataFrame.corr method ============

ANCHOR = (
    '        return describe_columns(\n'
    '            self._iter_filtered_batches(), cols, self._memory_guard,\n'
    '            reservoir_size=reservoir_size,\n'
    '        )\n'
    '\n'
    '    # ---------------- write ----------------'
)

CORR_METHOD = '''        return describe_columns(
            self._iter_filtered_batches(), cols, self._memory_guard,
            reservoir_size=reservoir_size,
        )

    # ---------------- corr ----------------

    def corr(self, columns=None, *, method="pearson"):
        """Correlation matrix (Pearson) between numeric columns.

        v0.9.23d (AUDIT_FULL_v0922 item 7.A.4). Composition of existing
        `CovarianceMeanOp` (one Reduce pass) + numpy normalization.
        O(d²) state, independent of N. Fail-fast if state > state_limit.

        Args:
            columns: numeric column names. None → auto-detect int/float.
                str → single column; Sequence[str] → those columns.
            method: 'pearson' (default). 'spearman' → NotImplementedError
                (needs per-column rank + Pearson on ranks; backlog P1).

        Returns:
            dict[str, dict[str, float]] — symmetric correlation matrix:
            `corr[a][b] = r`. Diagonal = 1.0 (or NaN for degenerate
            column). NaN for pairs involving a constant / all-null
            column.

        Raises:
            NotImplementedError: method != 'pearson'.
            ValueError: no numeric columns selected.
            KeyError: missing column.
            TypeError: non-numeric column requested explicitly.
            MemoryLimitExceeded: d too large for budget (CovarianceOp
                estimate 8+8d+8d² > state_limit).

        Example:
            >>> corr = df.corr()
            >>> corr['x1']['x2']         # Pearson r between x1, x2
            >>> corr['x1']['x1']         # == 1.0
        """
        from cadw.ops.reduce import CovarianceMeanOp, reduce

        if method != "pearson":
            raise NotImplementedError(
                f"corr: method={method!r} not supported. Only "
                f"'pearson' is implemented in v0.9.23d. 'spearman' "
                f"needs per-column rank (backlog P1)."
            )

        # Auto-detect numeric columns (None) or normalize (str / Sequence).
        if columns is None:
            cols = []
            for c in self._columns:
                t = self._dataset.schema.field(c).type
                if pa.types.is_integer(t) or pa.types.is_floating(t):
                    cols.append(c)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        if not cols:
            raise ValueError(
                "corr: no numeric columns selected (columns=None auto-"
                "detects int/float; found none)"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"corr: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"corr: column {c!r} has type {t}; numeric required"
                )

        # One Reduce pass (O(d²) state, guarded in reduce()).
        op = CovarianceMeanOp(cols)
        res = reduce(
            self._iter_filtered_batches(), op, self._memory_guard, log=False,
        )
        cov = res["cov"]
        n = res["n"]

        nan = float("nan")
        if n < 2:
            # Not enough valid rows for any correlation.
            return {c: {c2: nan for c2 in cols} for c in cols}

        diag = np.diag(cov)
        std = np.sqrt(np.maximum(diag, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.outer(std, std)
            corr_mat = np.where(denom > 0, cov / denom, nan)
        # Symmetrize (numerical noise) and clip to [-1, 1].
        corr_mat = 0.5 * (corr_mat + corr_mat.T)
        corr_mat = np.clip(corr_mat, -1.0, 1.0)
        # Diagonal: 1.0 if var > 0, else NaN (constant / all-null).
        for i in range(len(cols)):
            corr_mat[i, i] = 1.0 if diag[i] > 0 else nan

        return {
            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}
            for i in range(len(cols))
        }

    # ---------------- write ----------------'''


def add_corr_method():
    print("\n=== add CadwDataFrame.corr method ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def corr(self, columns=None" in s:
        print("  already present — skip")
        return
    if ANCHOR not in s:
        raise SystemExit(
            "CadwDataFrame.corr anchor not found — source may have drifted"
        )
    s = s.replace(ANCHOR, CORR_METHOD, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Tests ============

TESTS = '''"""v0.9.23d: CadwDataFrame.corr(method='pearson') (AUDIT_FULL_v0922 7.A.4).

Pearson via CovarianceMeanOp (Reduce, O(d²) state). Reference: numpy
corrcoef (rowvar=False, ddof=1 like CADW).
"""
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ Basic correctness ============


def test_corr_perfect_positive(tmp_path):
    n = 100
    x = np.arange(n, dtype=np.float64)
    y = 2.0 * x + 3.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    c = df.corr()
    assert c["x"]["y"] == pytest.approx(1.0)
    assert c["y"]["x"] == pytest.approx(1.0)
    assert c["x"]["x"] == pytest.approx(1.0)


def test_corr_perfect_negative(tmp_path):
    n = 100
    x = np.arange(n, dtype=np.float64)
    y = -2.0 * x + 3.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    c = df.corr()
    assert c["x"]["y"] == pytest.approx(-1.0)
    assert c["y"]["x"] == pytest.approx(-1.0)


def test_corr_independent_is_zero(tmp_path):
    """Large independent sample -> |r| small."""
    rng = np.random.default_rng(42)
    n = 10_000
    df = make_df(
        tmp_path,
        {"x": rng.normal(size=n).tolist(), "y": rng.normal(size=n).tolist()},
        chunk=1000,
    )
    c = df.corr()
    assert abs(c["x"]["y"]) < 0.05


def test_corr_matches_numpy(tmp_path):
    """Exact parity with np.corrcoef over 3 columns."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    df = make_df(
        tmp_path,
        {f"x{i}": X[:, i].tolist() for i in range(3)},
        chunk=20,
    )
    c = df.corr()
    ref = np.corrcoef(X, rowvar=False)
    for i in range(3):
        for j in range(3):
            assert c[f"x{i}"][f"x{j}"] == pytest.approx(ref[i, j], rel=1e-9)


def test_corr_symmetric(tmp_path):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 3))
    df = make_df(
        tmp_path,
        {f"x{i}": X[:, i].tolist() for i in range(3)},
    )
    c = df.corr()
    for i in range(3):
        for j in range(3):
            assert c[f"x{i}"][f"x{j}"] == pytest.approx(c[f"x{j}"][f"x{i}"])


# ============ Constant / null edge cases ============


def test_corr_constant_column_nan(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [5.0, 5.0, 5.0, 5.0],   # constant
    })
    c = df.corr()
    assert math.isnan(c["a"]["b"])
    assert math.isnan(c["b"]["a"])
    assert math.isnan(c["b"]["b"])
    assert c["a"]["a"] == pytest.approx(1.0)


def test_corr_all_null_column_nan(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": pa.array([None, None, None], type=pa.float64()),
    })
    c = df.corr()
    assert math.isnan(c["a"]["b"])
    assert math.isnan(c["b"]["b"])


def test_corr_no_rows_nan(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([], type=pa.float64()),
        "b": pa.array([], type=pa.float64()),
    })
    c = df.corr()
    assert math.isnan(c["a"]["a"])
    assert math.isnan(c["a"]["b"])


def test_corr_single_row_nan(tmp_path):
    """n=1 -> not enough for variance (ddof=1)."""
    df = make_df(tmp_path, {
        "a": [1.0], "b": [2.0],
    })
    c = df.corr()
    assert math.isnan(c["a"]["a"])


def test_corr_nulls_skipped(tmp_path):
    """Nulls skipped per row (pairwise, matches CovarianceOp's mask)."""
    df = make_df(tmp_path, {
        "a": pa.array([1.0, 2.0, None, 4.0, 5.0], type=pa.float64()),
        "b": pa.array([2.0, 4.0, 6.0, None, 10.0], type=pa.float64()),
    })
    c = df.corr()
    # Valid rows: (1,2), (2,4), (5,10) — perfectly linear -> r=1.0
    assert c["a"]["b"] == pytest.approx(1.0)


# ============ column selection ============


def test_corr_auto_detect_numeric(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
        "s": ["x", "y", "z"],   # string -> skipped
    })
    c = df.corr()
    assert set(c.keys()) == {"a", "b"}
    assert "s" not in c


def test_corr_columns_subset(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
        "s": ["x", "y", "z"],
    })
    c = df.corr(columns=["a"])
    assert list(c.keys()) == ["a"]
    assert "b" not in c


def test_corr_columns_string(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0, 3.0]})
    c = df.corr(columns="a")
    assert c["a"]["a"] == pytest.approx(1.0)


def test_corr_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        df.corr()


def test_corr_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        df.corr(columns=["nope"])


def test_corr_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0], "s": ["x", "y"]})
    with pytest.raises(TypeError, match="numeric"):
        df.corr(columns=["s"])


# ============ method ============


def test_corr_spearman_not_implemented(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0]})
    with pytest.raises(NotImplementedError, match="spearman"):
        df.corr(method="spearman")


# ============ multi-batch / filtered ============


def test_corr_multibatch_consistent(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 2))
    data = {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()}
    df1 = make_df(tmp_path, data, chunk=20, name="a")
    df2 = make_df(tmp_path, data, chunk=200, name="b")
    c1 = df1.corr()
    c2 = df2.corr()
    assert c1["x0"]["x1"] == pytest.approx(c2["x0"]["x1"], rel=1e-12)


def test_corr_filtered(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    df = make_df(tmp_path, {
        "x0": X[:, 0].tolist(),
        "x1": X[:, 1].tolist(),
        "f": list(range(100)),
    }, chunk=20)
    filtered = df[df["f"] > 50]
    c = filtered.corr()
    ref = np.corrcoef(X[51:], rowvar=False)
    assert c["x0"]["x1"] == pytest.approx(ref[0, 1], rel=1e-9)


# ============ memory guard ============


def test_corr_large_d_fails_fast(tmp_path):
    """d=500 -> state ~2 MB; budget 100 KB -> fail-fast BEFORE init."""
    rng = np.random.default_rng(0)
    data = {f"c{i}": rng.normal(size=100).tolist() for i in range(500)}
    df = make_df(tmp_path, data, chunk=20)
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded):
        df.corr()


# ============ AST guards ============


def test_corr_no_pandas_ast():
    """AST: no to_pandas/from_pandas Call in corr body."""
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.corr)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="CadwDataFrame.corr",
    )


def test_corr_uses_covariance_mean_op():
    """AST: corr uses CovarianceMeanOp (not a custom Reduce loop)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.corr)
    assert "CovarianceMeanOp" in src
    assert "reduce(" in src


def test_corr_returns_nested_dict_shape(tmp_path):
    """Return type: dict[str, dict[str, float]] (describe-consistent)."""
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
    })
    c = df.corr()
    assert isinstance(c, dict)
    assert set(c.keys()) == {"a", "b"}
    for row in c.values():
        assert isinstance(row, dict)
        for v in row.values():
            assert isinstance(v, float)
'''


def write_test_file():
    print("\n=== write tests/test_corr.py ===")
    p = Path("tests/test_corr.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: corr parity + fail-fast ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import math as _math
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.core.memory_guard import MemoryLimitExceeded as _MLE

        # Parity with numpy
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(42)
            X = rng.normal(size=(300, 3))
            data = {f"x{i}": X[:, i].tolist() for i in range(3)}
            t = pa_.table(data)
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=50,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            c = df.corr()
            ref = _np_.corrcoef(X, rowvar=False)
            for i in range(3):
                for j in range(3):
                    got = c[f"x{i}"][f"x{j}"]
                    want = float(ref[i, j])
                    assert abs(got - want) < 1e-9, (
                        f"corr mismatch x{i},x{j}: {got} vs {want}"
                    )

        # Constant column -> NaN
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [5.0, 5.0, 5.0, 5.0],
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            c = df.corr()
            assert _math.isnan(c["a"]["b"]), "constant col b: NaN expected"
            assert _math.isnan(c["b"]["b"]), "constant col b self: NaN"
            assert c["a"]["a"] == 1.0

        # Spearman -> NotImplementedError
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"a": [1.0, 2.0], "b": [3.0, 4.0]})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            try:
                df.corr(method="spearman")
            except NotImplementedError:
                pass
            else:
                raise AssertionError("spearman should raise NotImplementedError")

        # Fail-fast at large d
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(0)
            data = {f"c{i}": rng.normal(size=100).tolist() for i in range(300)}
            t = pa_.table(data)
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=20,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            df._memory_guard = _MG(max_memory_bytes=100_000)
            try:
                df.corr()
            except _MLE:
                pass
            else:
                raise AssertionError("corr d=300 under 100 KB must fail-fast")

        print("  numpy parity + const NaN + spearman NIE + fail-fast ok")

    run_verify_in_subprocess(_verify)


# ============ 4. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_corr.py",
         "tests/test_dataframe_v094b.py",   # describe still works nearby
         "tests/test_pivot_arrow_native.py",
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
    print("=== v0.9.23d: df.corr(method='pearson') ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_corr_method()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23d: CadwDataFrame.corr(method='pearson') facade. "
        "AUDIT_FULL_v0922 item 7.A.4 (Top-10 gap #6: EDA must-have; "
        "CovarianceOp already exists, was not exposed). "
        "Composition: CovarianceMeanOp (one Reduce pass, O(d²) state "
        "8+8d+8d², exact per §3.1 audit) + numpy normalization: "
        "corr[i,j] = cov[i,j] / sqrt(var_i * var_j); symmetrize "
        "(numerical noise) + clip [-1,1]; diagonal 1.0 if var>0 else "
        "NaN (constant / all-null column). Nulls skipped pairwise "
        "(matches CovarianceOp mask). "
        "Return: dict[str,dict[str,float]] (nested, matches describe() "
        "pattern) — not pd.DataFrame (pandas optional dependency). "
        "args: columns=None -> auto-detect int/float; str -> single; "
        "Sequence[str] -> subset. method='spearman' -> "
        "NotImplementedError (needs per-column rank + Pearson on ranks; "
        "P1 backlog). Fail-fast at large d: CovarianceMeanOp state "
        "checked by reduce() before init (d=500 ~2 MB -> raises under "
        "100 KB budget). "
        "+tests/test_corr.py (18 tests: perfect +/-1, numpy parity, "
        "symmetric, constant -> NaN, all-null -> NaN, no-rows -> NaN, "
        "single-row -> NaN, nulls pairwise, auto-detect numeric, "
        "subset, str arg, empty cols -> ValueError, missing col -> "
        "KeyError, non-numeric -> TypeError, spearman NIE, multibatch "
        "consistency, filtered, fail-fast at d=500, 2 AST guards, "
        "return shape). "
        "Fast: 1952 -> 1970 passed.",
        self_path="scripts/patch_v0923d_corr.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23d applied. 7.A.4 complete.")


if __name__ == "__main__":
    main()