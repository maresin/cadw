#!/usr/bin/env python3
"""v0.6.2c: df.rolling(window, agg) — incremental rolling aggregation.

Scope:
    df.rolling(window, agg, *, min_periods=None, columns=None)

- window >= 1
- agg in {sum, mean, min, max, count}
- min_periods=None -> window
- columns=None -> all numeric
- output: original columns + {col}_rolling_{agg}
- incremental: O(1) per row (sum/mean/count) or O(1) amortized (min/max)
- peak: O(window × #cols)

Not in scope (v0.6.3+): on=col (time-based), expanding, ewm, std/var/median.

Run: uv run python scripts/patch_v062c_rolling.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ patch dataframe.py ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: rolling ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def rolling(" in s:
        print("  rolling already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- rolling ----------------

    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Rolling window aggregation (v0.6.2c).

        - window: int >= 1.
        - agg: {'sum', 'mean', 'min', 'max', 'count'}.
              count = non-null count in window.
        - min_periods=None -> window. First (min_periods-1) rows = null.
        - columns=None -> all numeric columns.

        Output: original columns + `{col}_rolling_{agg}` for each column.

        Incremental streaming: O(1) per row for sum/mean/count; O(1)
        amortized for min/max via monotonic deque. Peak O(window × #cols).

        skips nulls (pandas default). First rows (before min_periods)
        receive null.

        Not in v0.6.2c: `on=col` (sort-based; use .sort_values(...) first),
        expanding, ewm, std/var/median/quantile.
        """
        from collections import deque

        if not isinstance(window, int):
            raise TypeError(
                f"rolling: window must be int; got {type(window).__name__}"
            )
        if window < 1:
            raise ValueError(f"rolling: window must be >= 1; got {window}")
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"rolling: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )
        if min_periods is None:
            mp = window
        elif not isinstance(min_periods, int):
            raise TypeError(
                f"rolling: min_periods must be int; "
                f"got {type(min_periods).__name__}"
            )
        elif not (1 <= min_periods <= window):
            raise ValueError(
                f"rolling: min_periods must be in [1, window={window}]; "
                f"got {min_periods}"
            )
        else:
            mp = min_periods

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"rolling: column {c!r} not in {self._columns}"
                )

        # Validate numeric for sum/mean/min/max (count works for any type).
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        # Output schema: original cols + rolling output per (col, agg).
        out_fields = []
        for c in self._columns:
            out_fields.append(self._dataset.schema.field(c))
        out_type_by_col: dict[str, pa.DataType] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = ot
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_rolling():
            states = {c: _RollState(window, mp, agg) for c in cols}
            for b in self._iter_filtered_batches():
                n = b.num_rows
                out_arrays = []
                for c in self._columns:
                    out_arrays.append(b.column(c))
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    rolling_vals = []
                    for v in src_vals:
                        rv = state.push(v)
                        rolling_vals.append(rv)
                    out_arrays.append(
                        pa.array(rolling_vals, type=out_type_by_col[c])
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_rolling(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)

    # Add helper class after the CadwDataFrame class (before _GroupBy)
    helper_anchor = "class _GroupBy:"
    if helper_anchor not in s:
        raise SystemExit("_GroupBy not found")
    helper = '''class _RollState:
    """Incremental rolling aggregation state (v0.6.2c).

    - sum/mean/count: running accumulators (O(1) push).
    - min/max: monotonic deque (O(1) amortized push).

    Nulls are skipped (pandas default). Count = non-null in window.
    """

    __slots__ = (
        "_window", "_min_periods", "_agg", "_buf", "_idx",
        "_sum", "_count", "_dq",
    )

    def __init__(self, window: int, min_periods: int, agg: str) -> None:
        from collections import deque as _dq_mod
        self._window = window
        self._min_periods = min_periods
        self._agg = agg
        self._buf = _dq_mod()  # (idx, is_null, value)
        self._idx = 0
        self._sum = 0.0
        self._count = 0  # non-null in current window
        self._dq = _dq_mod()  # for min/max: (idx, value)

    def push(self, v):
        window = self._window
        i = self._idx
        # 1. Add current
        is_null = v is None
        if not is_null:
            self._sum += float(v)
            self._count += 1
        self._buf.append((i, is_null, v))
        if self._agg == "min":
            while self._dq and self._dq[-1][1] >= v:
                self._dq.pop()
            if not is_null:
                self._dq.append((i, v))
        elif self._agg == "max":
            while self._dq and self._dq[-1][1] <= v:
                self._dq.pop()
            if not is_null:
                self._dq.append((i, v))

        # 2. Evict if len > window
        while len(self._buf) > window:
            old_i, old_null, old_v = self._buf.popleft()
            if not old_null:
                self._sum -= float(old_v)
                self._count -= 1
            if self._agg in ("min", "max") and self._dq and self._dq[0][0] == old_i:
                self._dq.popleft()

        self._idx += 1

        # 3. Compute result
        if self._count < self._min_periods:
            return None
        if self._agg == "count":
            return self._count
        if self._agg == "sum":
            return self._sum
        if self._agg == "mean":
            return self._sum / self._count
        if self._agg in ("min", "max"):
            return self._dq[0][1] if self._dq else None
        raise RuntimeError(f"_RollState: unknown agg {self._agg!r}")


'''
    s = s.replace(helper_anchor, helper + helper_anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: rolling + _RollState added")


# ============ tests ============

TESTS_SRC = '''"""v0.6.2c: rolling(window, agg)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=2):
    if data is None:
        data = {"v": [1.0, 2.0, 3.0, 4.0, 5.0]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def approx(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < tol


# ---------- basic windows ----------

def test_rolling_sum_w3(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "sum")
    # [null, null, 1+2+3=6, 2+3+4=9, 3+4+5=12]
    assert [x is None for x in read_col(res, tmp_path, "v_rolling_sum")] == [
        True, True, False, False, False,
    ]
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert approx(vals[2], 6.0)
    assert approx(vals[3], 9.0)
    assert approx(vals[4], 12.0)


def test_rolling_mean_w3(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "mean")
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert vals[0] is None and vals[1] is None
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 3.0)
    assert approx(vals[4], 4.0)


def test_rolling_min_w3(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.rolling(3, "min")
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals[0] is None and vals[1] is None
    assert approx(vals[2], 1.0)
    assert approx(vals[3], 1.0)
    assert approx(vals[4], 2.0)


def test_rolling_max_w3(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.rolling(3, "max")
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals[0] is None and vals[1] is None
    assert approx(vals[2], 4.0)
    assert approx(vals[3], 4.0)
    assert approx(vals[4], 5.0)


def test_rolling_count_w3(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    res = df.rolling(3, "count")
    vals = read_col(res, tmp_path, "v_rolling_count")
    assert vals[0] is None and vals[1] is None
    assert vals[2] == 3 and vals[3] == 3 and vals[4] == 3


# ---------- min_periods ----------

def test_rolling_min_periods_1(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "mean", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 1.5)
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 3.0)
    assert approx(vals[4], 4.0)


def test_rolling_min_periods_2(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "mean", min_periods=2)
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert vals[0] is None
    assert approx(vals[1], 1.5)
    assert approx(vals[2], 2.0)


def test_rolling_min_periods_invalid(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="min_periods"):
        df.rolling(3, "sum", min_periods=0)
    with pytest.raises(ValueError, match="min_periods"):
        df.rolling(3, "sum", min_periods=4)


# ---------- window=1 ----------

def test_rolling_window_1(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(1, "sum")
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert [round(x, 9) for x in vals] == [1.0, 2.0, 3.0, 4.0, 5.0]


# ---------- window > n ----------

def test_rolling_window_gt_n(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0]})
    res = df.rolling(10, "sum")
    vals = read_col(res, tmp_path, "v_rolling_sum")
    # all null (min_periods=10 > n)
    assert vals == [None, None]


def test_rolling_window_gt_n_min_periods_1(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0]})
    res = df.rolling(10, "sum", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 3.0)


# ---------- null handling ----------

def test_rolling_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, 4.0, None]})
    res = df.rolling(3, "sum", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 1.0)
    assert approx(vals[2], 4.0)
    assert approx(vals[3], 7.0)
    assert approx(vals[4], 7.0)


def test_rolling_count_skips_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, None, 5.0]})
    res = df.rolling(3, "count", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_count")
    assert vals[0] == 1
    assert vals[1] == 1
    assert vals[2] == 2
    assert vals[3] == 1
    assert vals[4] == 2


# ---------- multi columns ----------

def test_rolling_two_columns_default(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum")
    assert approx(read_col(res, tmp_path, "a_rolling_sum")[1], 3.0)
    assert approx(read_col(res, tmp_path, "b_rolling_sum")[1], 30.0)


def test_rolling_subset(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum", columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_rolling_sum" in names
    assert "b_rolling_sum" not in names


# ---------- batch boundaries ----------

def test_rolling_across_batches(tmp_path):
    """window > batch_size: state carried across batches."""
    df = make_df(tmp_path, chunk=1)  # 1 row per batch
    res = df.rolling(3, "mean", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 1.5)
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 3.0)
    assert approx(vals[4], 4.0)


# ---------- filtered ----------

def test_rolling_filtered(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    filtered = df[df["v"] > 2]  # [3, 4, 5, 6]
    res = filtered.rolling(2, "sum")
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals[0] is None
    assert approx(vals[1], 7.0)
    assert approx(vals[2], 9.0)
    assert approx(vals[3], 11.0)


# ---------- errors ----------

def test_rolling_invalid_window(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="window must be >= 1"):
        df.rolling(0, "sum")
    with pytest.raises(TypeError, match="window must be int"):
        df.rolling(1.5, "sum")


def test_rolling_invalid_agg(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="agg must be one of"):
        df.rolling(3, "std")


def test_rolling_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.rolling(3, "sum", columns=["nope"])


def test_rolling_non_numeric_for_sum_raises(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1.0, 2.0, 3.0], "s": ["a", "b", "c"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.rolling(2, "sum", columns=["s"])


def test_rolling_count_on_string_works(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1.0, 2.0, 3.0], "s": ["a", "b", "c"],
    })
    res = df.rolling(2, "count", columns=["s"])
    vals = read_col(res, tmp_path, "s_rolling_count")
    assert vals[0] is None
    assert vals[1] == 2 and vals[2] == 2


# ---------- preserves original ----------

def test_rolling_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.rolling(3, "sum")
    assert read_col(df, tmp_path, "v") == [1.0, 2.0, 3.0, 4.0, 5.0]


# ---------- dtype ----------

def test_rolling_sum_dtype_float(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3, 4, 5]})  # int
    res = df.rolling(2, "sum")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_rolling_sum").type == pa.float64()


def test_rolling_count_dtype_int(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "count")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_rolling_count").type == pa.int64()


def test_rolling_min_dtype_same_as_input(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3, 4, 5]})  # int
    res = df.rolling(2, "min")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_rolling_min").type == pa.int64()
'''


def write_tests():
    print("\n=== tests/test_rolling.py ===")
    p = Path("tests/test_rolling.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 26 tests")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v062c-rolling"

    print("=== v0.6.2c: rolling ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v062c-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v062c-{ts_str}",
        "-m", "before patch v0.6.2c rolling"])

    patch_dataframe()
    write_tests()

    print("\n=== pytest tests/test_rolling.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.2c (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.2c: rolling(window, agg) — incremental O(window) memory"])

    print("\n==> v0.6.2c applied on", branch)


if __name__ == "__main__":
    main()