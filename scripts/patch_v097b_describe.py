#!/usr/bin/env python3
# v0.9.7b — DescribeOp fused (R5: 109× speedup, 12.1× RSS reduction).
#
# Audit reference (bench_r5.py, N=2M, 2 cols):
#   current describe:  2.47s, RSS 19.3 MB  (5 Reduce + 1 sort per col)
#   fused single-pass: 0.02s, RSS  1.6 MB  (1 Reduce, Welford+reservoir)
#   109.2× speedup, 12.1× RSS reduction.
#
# Design:
#   - cadw/ops/describe.py::DescribeOp — ReduceOp, per-column state:
#     {n, mean, M2 (Welford), min, max, reservoir (list), seen}.
#   - Reservoir (Algorithm R, size=10000): quantiles exact for N ≤ 10k,
#     approximate above (§11.5 documented). min/max/count/mean/std exact.
#   - std uses ddof=1 (matches existing StdOp default).
#   - Facade CadwDataFrame.describe delegates to describe_columns
#     (thin); no self.quantile call (no per-col external_sort).
#   - Backward compat: existing tests small N (exact); output float
#     everywhere (3.0 == 3 passes int-input assertions).
#
# Two-level tests (§13.4/190):
#   - facade: describe delegates to describe_columns; no .quantile
#   - impl: DescribeOp uses Welford (M2) + reservoir (list + Algorithm R)
#   - correctness: mean/std/min/max exact vs numpy; quantiles exact
#     for N ≤ reservoir_size, approx (via reservoir) above
#   - backward-compat: existing test_dataframe_v094b2 describe tests
#     pass unchanged (N=5/3/2 all ≤ reservoir_size → exact)
#
# Scaffold v2 used: write_content_checked, is_patched,
# assert_no_call_in_source, parse_method_source. Whitelist cleanup:
# remove stale `describe` entry (no longer has high-risk calls).
#
# Run: uv run python scripts/patch_v097b_describe.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    write_content_checked, assert_compiles, commit_with_guards,
    assert_no_call_in_source, assert_call_order, is_patched,
)


# ============ 1. new file cadw/ops/describe.py ============

DESCRIBE_LINES = [
    '"""Fused single-pass describe (v0.9.7b, R5: 109× speedup benchmark).',
    '',
    'Single Reduce pass per column group: Welford (mean/M2/std) +',
    'min/max + reservoir (Algorithm R) for quantiles. Replaces 5 Reduce',
    'passes + 1 external_sort per column.',
    '',
    'Reference (bench_r5.py, N=2M, 2 cols): current 2.47s / 19.3 MB;',
    'fused 0.02s / 1.6 MB (109×, 12.1× RSS). Speedup: single pass over',
    'data (no 5× re-scan) and no external_sort per column (reservoir',
    'for quantiles).',
    '',
    'Semantics (§11.5 documented):',
    '  - count/mean/std/min/max: exact (Welford + min/max).',
    '  - quantiles (25%/50%/75%): exact if N ≤ reservoir_size (all',
    '    values kept); approximate (reservoir) above. Default',
    '    reservoir_size=10000. For exact large-N, use df.quantile().',
    '  - Output dtype float64 for all fields (int input also → float;',
    '    3.0 == 3 in Python, existing tests pass).',
    '  - Empty / all-null column → NaN for all stat fields; count=0.',
    '"""',
    'from __future__ import annotations',
    '',
    'import random',
    'from typing import Iterable, Sequence',
    '',
    'import numpy as np',
    'import pyarrow as pa',
    'import pyarrow.compute as pc',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.ops.reduce import ReduceOp, reduce',
    '',
    '',
    '_DEFAULT_RESERVOIR = 10_000',
    '_DEFAULT_SEED = 42',
    '_STAT_KEYS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")',
    '',
    '',
    'class DescribeOp(ReduceOp):',
    '    """Per-column Welford (mean/M2/std) + min/max + reservoir.',
    '',
    '    State: list of d dicts, each {n, mean, M2, min, max,',
    '    reservoir (list), seen}. Bounded by d × (5 × 8 + reservoir_size',
    '    × 8) bytes; independent of N.',
    '    """',
    '',
    '    def __init__(',
    '        self,',
    '        columns: Sequence[str],',
    '        *,',
    '        reservoir_size: int = _DEFAULT_RESERVOIR,',
    '        seed: int = _DEFAULT_SEED,',
    '    ) -> None:',
    '        if not isinstance(reservoir_size, int) or reservoir_size < 1:',
    '            raise ValueError(',
    '                f"DescribeOp: reservoir_size must be int >= 1; "',
    '                f"got {reservoir_size!r}"',
    '            )',
    '        self.columns = list(columns)',
    '        self.d = len(self.columns)',
    '        self.reservoir_size = int(reservoir_size)',
    '        self.seed = int(seed)',
    '        self._batch_count = 0',
    '',
    '    def estimate_state_nbytes(self) -> int:',
    '        # Per column: 5 float64 (n not counted; mean/M2/min/max) =',
    '        # 5 × 8 = 40 B + reservoir (reservoir_size × 8 B).',
    '        return self.d * (40 + self.reservoir_size * 8)',
    '',
    '    def init(self):',
    '        return [',
    '            {',
    '                "n": 0, "mean": 0.0, "M2": 0.0,',
    '                "min": None, "max": None,',
    '                "reservoir": [], "seen": 0,',
    '            }',
    '            for _ in range(self.d)',
    '        ]',
    '',
    '    def update(self, state, batch):',
    '        if batch.num_rows == 0:',
    '            return state',
    '        rng = random.Random(self.seed + self._batch_count)',
    '        self._batch_count += 1',
    '        for j, col in enumerate(self.columns):',
    '            s = state[j]',
    '            arr_f = pc.cast(batch.column(col), pa.float64())',
    '            x_np = arr_f.to_numpy(zero_copy_only=False)',
    '            mask = ~np.isnan(x_np)',
    '            x_np = x_np[mask]',
    '            n_b = int(len(x_np))',
    '            if n_b == 0:',
    '                continue',
    '            mean_b = float(x_np.mean())',
    '            M2_b = float(((x_np - mean_b) ** 2).sum())',
    '            n_a = int(s["n"])',
    '            if n_a == 0:',
    '                s["mean"] = mean_b',
    '                s["M2"] = M2_b',
    '            else:',
    '                n_total = n_a + n_b',
    '                delta = mean_b - s["mean"]',
    '                s["mean"] = s["mean"] + delta * n_b / n_total',
    '                s["M2"] = (',
    '                    s["M2"] + M2_b',
    '                    + delta * delta * n_a * n_b / n_total',
    '                )',
    '            s["n"] = n_a + n_b',
    '            mn_b = float(x_np.min())',
    '            mx_b = float(x_np.max())',
    '            s["min"] = mn_b if s["min"] is None else min(s["min"], mn_b)',
    '            s["max"] = mx_b if s["max"] is None else max(s["max"], mx_b)',
    '            # Reservoir (Algorithm R)',
    '            for v in x_np.tolist():',
    '                s["seen"] += 1',
    '                if len(s["reservoir"]) < self.reservoir_size:',
    '                    s["reservoir"].append(v)',
    '                else:',
    '                    jj = rng.randint(0, s["seen"] - 1)',
    '                    if jj < self.reservoir_size:',
    '                        s["reservoir"][jj] = v',
    '        return state',
    '',
    '    def merge(self, a, b):',
    '        """Best-effort merge (reduce is sequential; merge rarely',
    '        called). Welford merge + reservoir pool + uniform downsample.',
    '        """',
    '        rng = random.Random(self.seed)',
    '        out = []',
    '        for sa, sb in zip(a, b):',
    '            n_a, n_b = int(sa["n"]), int(sb["n"])',
    '            n = n_a + n_b',
    '            if n_a == 0:',
    '                out.append(dict(sb))',
    '                continue',
    '            if n_b == 0:',
    '                out.append(dict(sa))',
    '                continue',
    '            delta = sb["mean"] - sa["mean"]',
    '            mean = sa["mean"] + delta * n_b / n',
    '            M2 = (sa["M2"] + sb["M2"]',
    '                  + delta * delta * n_a * n_b / n)',
    '            mn = sa["min"] if sa["min"] is not None else sb["min"]',
    '            if sa["min"] is not None and sb["min"] is not None:',
    '                mn = min(sa["min"], sb["min"])',
    '            mx = sa["max"] if sa["max"] is not None else sb["max"]',
    '            if sa["max"] is not None and sb["max"] is not None:',
    '                mx = max(sa["max"], sb["max"])',
    '            pool = list(sa["reservoir"]) + list(sb["reservoir"])',
    '            if len(pool) > self.reservoir_size:',
    '                rng.shuffle(pool)',
    '                pool = pool[: self.reservoir_size]',
    '            out.append({',
    '                "n": n, "mean": mean, "M2": M2,',
    '                "min": mn, "max": mx,',
    '                "reservoir": pool,',
    '                "seen": sa["seen"] + sb["seen"],',
    '            })',
    '        return out',
    '',
    '    def state_nbytes(self, state):',
    '        return sum(',
    '            40 + len(s["reservoir"]) * 8',
    '            for s in state',
    '        )',
    '',
    '    def finalize(self, state):',
    '        result = {}',
    '        for j, col in enumerate(self.columns):',
    '            s = state[j]',
    '            n = int(s["n"])',
    '            if n == 0:',
    '                result[col] = {k: float("nan") for k in _STAT_KEYS}',
    '                result[col]["count"] = 0',
    '                continue',
    '            if n > 1:',
    '                std = float(np.sqrt(s["M2"] / (n - 1)))',
    '            else:',
    '                std = float("nan")',
    '            res = np.asarray(s["reservoir"], dtype=np.float64)',
    '            if len(res) == 0:',
    '                q25 = q50 = q75 = float("nan")',
    '            else:',
    '                q25, q50, q75 = np.percentile(res, [25, 50, 75])',
    '            result[col] = {',
    '                "count": n,',
    '                "mean": float(s["mean"]),',
    '                "std": std,',
    '                "min": float(s["min"]) if s["min"] is not None else float("nan"),',
    '                "25%": float(q25),',
    '                "50%": float(q50),',
    '                "75%": float(q75),',
    '                "max": float(s["max"]) if s["max"] is not None else float("nan"),',
    '            }',
    '        return result',
    '',
    '',
    'def describe_columns(',
    '    batch_iter: Iterable[pa.RecordBatch],',
    '    columns: Sequence[str],',
    '    memory_guard: MemoryGuard,',
    '    *,',
    '    reservoir_size: int = _DEFAULT_RESERVOIR,',
    ') -> dict:',
    '    """One Reduce pass over batch_iter; returns {col: {stat: value}}.',
    '',
    '    Empty columns list → {}. Independent of N; state bounded by',
    '    d × reservoir_size (checked pre-init by reduce()).',
    '    """',
    '    if not columns:',
    '        return {}',
    '    op = DescribeOp(columns, reservoir_size=reservoir_size)',
    '    return reduce(batch_iter, op, memory_guard, log=False)',
    '',
    '',
    '__all__ = ["DescribeOp", "describe_columns"]',
    '',
]


def create_describe_module():
    print("\n=== create cadw/ops/describe.py ===")
    p = Path("cadw/ops/describe.py")
    if p.exists():
        # Validate before skip.
        src = p.read_text()
        missing = [
            s for s in ("def describe_columns(", "class DescribeOp(")
            if s not in src
        ]
        if not missing:
            try:
                assert_compiles(p)
                print("  exists + valid — skip")
                return
            except SystemExit:
                print("  exists but broken — force-write")
        else:
            print(f"  exists but missing symbols {missing} — force-write")
    write_content_checked(p, DESCRIBE_LINES)
    print("  written + compiled")


# ============ 2. replace CadwDataFrame.describe ============

NEW_DESCRIBE_LINES = [
    '    def describe(self, columns=None, *, reservoir_size=10000):',
    '        """Numeric summary — fused single-pass (v0.9.7b, R5: 109×).',
    '',
    '        Single Reduce pass per column group: Welford (mean/M2/std)',
    '        + min/max + reservoir (Algorithm R) for quantiles. Replaces',
    '        5 Reduce passes + 1 external_sort per column.',
    '',
    '        Peak RAM = batch + O(d × reservoir_size × 8B), independent',
    '        of N. Default reservoir_size=10000 → quantiles exact for',
    '        N ≤ reservoir_size, approximate above (§11.5).',
    '',
    '        Args:',
    '            columns: numeric columns (None → auto-detect int/float).',
    '            reservoir_size: reservoir capacity per column',
    '                (>= 1, default 10000). Larger → more exact quantiles,',
    '                more memory (d × reservoir_size × 8B).',
    '',
    '        Returns {col: {count, mean, std, min, 25%, 50%, 75%, max}}.',
    '        Empty / all-null column → NaN for stat fields, count=0.',
    '',
    '        For exact quantiles at any N, use `df.quantile()`',
    '        (external_sort per column; slower but exact).',
    '        """',
    '        from cadw.ops.describe import describe_columns',
    '',
    '        if not isinstance(reservoir_size, int) or reservoir_size < 1:',
    '            raise ValueError(',
    '                f"describe: reservoir_size must be int >= 1; "',
    '                f"got {reservoir_size!r}"',
    '            )',
    '        if columns is None:',
    '            cols = []',
    '            for c in self._columns:',
    '                t = self._dataset.schema.field(c).type',
    '                if (pa.types.is_integer(t)',
    '                        or pa.types.is_floating(t)):',
    '                    cols.append(c)',
    '        elif isinstance(columns, str):',
    '            cols = [columns]',
    '        else:',
    '            cols = list(columns)',
    '        for c in cols:',
    '            if c not in self._columns:',
    '                raise KeyError(',
    '                    "describe: column " + repr(c) + " not in "',
    '                    + str(self._columns)',
    '                )',
    '        return describe_columns(',
    '            self._iter_filtered_batches(), cols, self._memory_guard,',
    '            reservoir_size=reservoir_size,',
    '        )',
]


def replace_describe_facade():
    print("\n=== dataframe.py: replace describe facade ===")
    import ast as _ast
    from cadw.dataframe import CadwDataFrame

    if is_patched(CadwDataFrame.describe, "v0.9.7b"):
        print("  already v0.9.7b — skip")
        return

    p = Path("cadw/dataframe.py")
    s = p.read_text()
    tree = _ast.parse(s)
    cls = None
    for node in tree.body:
        if isinstance(node, _ast.ClassDef) and node.name == "CadwDataFrame":
            cls = node
            break
    if cls is None:
        raise SystemExit("CadwDataFrame class not found")
    start = end = None
    for node in cls.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "describe":
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("CadwDataFrame.describe not found")

    print(f"  replacing lines {start}..{end}")
    lines = s.split("\n")
    write_content_checked(
        p,
        lines[:start - 1] + NEW_DESCRIBE_LINES + lines[end:],
    )
    print("  OK")


# ============ 3. whitelist cleanup ============

def update_whitelist():
    print("\n=== test_arrow_call_memory_guard.py: remove stale describe entry ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()

    # The describe facade no longer has high-risk calls → remove
    # whitelist entry. Anchor variants (older/newer comment text).
    variants = [
        '    # v0.9.4b-2: describe() delegates to self.quantile (guarded via external_sort).\n'
        '    "cadw/dataframe.py:CadwDataFrame.describe":\n'
        '        "delegates to self.quantile (guarded via external_sort)",\n',
        '    "cadw/dataframe.py:CadwDataFrame.describe":\n'
        '        "delegates to self.quantile (guarded via external_sort)",\n',
        '    "cadw/dataframe.py:CadwDataFrame.describe":\n'
        '        "delegates to describe_columns (fused Welford+reservoir)",\n',
    ]
    removed = 0
    for v in variants:
        if v in s:
            s = s.replace(v, "", 1)
            removed += 1
            break
    if removed:
        p.write_text(s)
        assert_compiles(p)
        print(f"  removed stale entry")
    else:
        print("  entry not found (may already be gone) — skip")


# ============ 4. tests ============

NEW_TEST_LINES = [
    '"""v0.9.7b: DescribeOp fused (R5: 109× vs 5-Reduce + sort)."""',
    'from __future__ import annotations',
    '',
    'import ast',
    'import inspect',
    'import time',
    '',
    'import numpy as np',
    'import pyarrow as pa',
    'import pyarrow.parquet as pq',
    'import pytest',
    '',
    'from cadw import CadwDataFrame, MemoryGuard',
    'from cadw.core.memory_guard import MemoryLimitExceeded',
    'from cadw.ops.describe import DescribeOp, describe_columns',
    'from cadw.patch_utils import (',
    '    parse_method_source, assert_no_call_in_source,',
    ')',
    '',
    '',
    'def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):',
    '    t = pa.table(data)',
    '    p = tmp_path / f"{name}.parquet"',
    '    pq.write_table(t, p)',
    '    return CadwDataFrame.from_parquet(',
    '        p, tmp_path / f"ds_{name}", chunk_size=chunk,',
    '        memory_guard=MemoryGuard(max_memory_mb=mem_mb),',
    '        spill_dir=tmp_path / f"spill_{name}",',
    '    )',
    '',
    '',
    '# ============ correctness (existing behaviour preserved) ============',
    '',
    '',
    'def test_describe_basic_5_values(tmp_path):',
    '    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]})',
    '    desc = df.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 5',
    '    assert d["mean"] == pytest.approx(3.0)',
    '    assert d["std"] == pytest.approx(np.sqrt(2.5))',
    '    assert d["min"] == 1.0',
    '    assert d["25%"] == pytest.approx(2.0)',
    '    assert d["50%"] == pytest.approx(3.0)',
    '    assert d["75%"] == pytest.approx(4.0)',
    '    assert d["max"] == 5.0',
    '',
    '',
    'def test_describe_int_input(tmp_path):',
    '    df = make_df(tmp_path, {"x": [1, 2, 3, 4, 5]})',
    '    desc = df.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 5',
    '    assert d["min"] == 1.0',
    '    assert d["max"] == 5.0',
    '',
    '',
    'def test_describe_filtered(tmp_path):',
    '    df = make_df(tmp_path, {"x": [1, 2, 3, 4, 5]})',
    '    f = df[df["x"] > 2]',
    '    desc = f.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 3',
    '    assert d["min"] == 3.0',
    '    assert d["max"] == 5.0',
    '',
    '',
    'def test_describe_subset(tmp_path):',
    '    df = make_df(tmp_path, {"a": [1, 2], "b": [10, 20], "s": ["x", "y"]})',
    '    desc = df.describe(columns=["a"])',
    '    assert "a" in desc',
    '    assert "b" not in desc',
    '    assert "s" not in desc',
    '',
    '',
    'def test_describe_all_numeric_default(tmp_path):',
    '    df = make_df(tmp_path, {"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})',
    '    desc = df.describe()',
    '    assert "i" in desc',
    '    assert "f" in desc',
    '    assert "s" not in desc',
    '',
    '',
    'def test_describe_missing_column(tmp_path):',
    '    df = make_df(tmp_path, {"x": [1]})',
    '    with pytest.raises(KeyError, match="nope"):',
    '        df.describe(columns=["nope"])',
    '',
    '',
    '# ============ empty / null ============',
    '',
    '',
    'def test_describe_empty_column_nan(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "x": pa.array([], type=pa.float64()),',
    '    })',
    '    desc = df.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 0',
    '    assert np.isnan(d["mean"])',
    '    assert np.isnan(d["std"])',
    '    assert np.isnan(d["min"])',
    '',
    '',
    'def test_describe_all_null_column_nan(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "x": pa.array([None, None, None], type=pa.float64()),',
    '    })',
    '    desc = df.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 0',
    '    assert np.isnan(d["mean"])',
    '',
    '',
    'def test_describe_empty_no_numeric_cols(tmp_path):',
    '    df = make_df(tmp_path, {"s": ["a", "b"]})',
    '    assert df.describe() == {}',
    '',
    '',
    'def test_describe_nulls_skipped(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "x": pa.array([1.0, None, 3.0, None, 5.0], type=pa.float64()),',
    '    })',
    '    desc = df.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 3',
    '    assert d["mean"] == pytest.approx(3.0)',
    '    assert d["min"] == 1.0',
    '    assert d["max"] == 5.0',
    '',
    '',
    '# ============ multi-column ============',
    '',
    '',
    'def test_describe_multiple_columns(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "a": [1.0, 2.0, 3.0, 4.0],',
    '        "b": [100.0, 200.0, 300.0, 400.0],',
    '    })',
    '    desc = df.describe()',
    '    assert desc["a"]["mean"] == pytest.approx(2.5)',
    '    assert desc["b"]["mean"] == pytest.approx(250.0)',
    '    assert desc["a"]["min"] == 1.0',
    '    assert desc["b"]["max"] == 400.0',
    '',
    '',
    'def test_describe_multibatch_chunk_1(tmp_path):',
    '    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, chunk=1)',
    '    desc = df.describe()',
    '    d = desc["x"]',
    '    assert d["count"] == 5',
    '    assert d["mean"] == pytest.approx(3.0)',
    '    assert d["std"] == pytest.approx(np.sqrt(2.5))',
    '',
    '',
    '# ============ reservoir semantics ============',
    '',
    '',
    'def test_describe_reservoir_small_n_exact(tmp_path):',
    '    # N <= reservoir_size → all values in reservoir → exact.',
    '    df = make_df(tmp_path, {"x": list(range(100))})',
    '    d = df.describe(reservoir_size=200)["x"]',
    '    # Exact percentiles of 0..99',
    '    assert d["25%"] == pytest.approx(24.75, abs=1e-9)',
    '    assert d["50%"] == pytest.approx(49.5, abs=1e-9)',
    '    assert d["75%"] == pytest.approx(74.25, abs=1e-9)',
    '',
    '',
    'def test_describe_reservoir_large_n_approx(tmp_path):',
    '    # N > reservoir_size → approximate; within tolerance.',
    '    rng = np.random.default_rng(0)',
    '    data = rng.normal(size=20000)',
    '    df = make_df(tmp_path, {"x": data.tolist()}, chunk=2000)',
    '    d = df.describe(reservoir_size=1000)["x"]',
    '    # mean/std exact (Welford)',
    '    assert d["mean"] == pytest.approx(float(data.mean()), rel=1e-9)',
    '    assert d["std"] == pytest.approx(float(data.std(ddof=1)), rel=1e-9)',
    '    # quantiles approximate — within 5% of true',
    '    true_q = np.percentile(data, [25, 50, 75])',
    '    assert abs(d["25%"] - true_q[0]) < 0.1',
    '    assert abs(d["50%"] - true_q[1]) < 0.1',
    '    assert abs(d["75%"] - true_q[2]) < 0.1',
    '',
    '',
    'def test_describe_reservoir_invalid(tmp_path):',
    '    df = make_df(tmp_path, {"x": [1.0]})',
    '    with pytest.raises(ValueError, match="reservoir_size"):',
    '        df.describe(reservoir_size=0)',
    '',
    '',
    'def test_describe_tight_budget_fail_fast(tmp_path):',
    '    # Large d + large reservoir → state > state_limit → MemoryLimitExceeded',
    '    # BEFORE any batch read (via reduce estimate_state_nbytes).',
    '    df = make_df(tmp_path, {',
    '        "a": list(range(1000)),',
    '        "b": list(range(1000)),',
    '    }, chunk=100, mem_mb=1)',
    '    df._memory_guard = MemoryGuard(max_memory_bytes=50_000)',
    '    with pytest.raises(MemoryLimitExceeded):',
    '        df.describe(reservoir_size=50_000)',
    '',
    '',
    '# ============ numpy equivalence (exact fields) ============',
    '',
    '',
    'def test_describe_mean_std_match_numpy(tmp_path):',
    '    rng = np.random.default_rng(42)',
    '    data = rng.normal(size=500)',
    '    df = make_df(tmp_path, {"x": data.tolist()}, chunk=50)',
    '    d = df.describe()["x"]',
    '    assert d["mean"] == pytest.approx(float(data.mean()), rel=1e-12)',
    '    assert d["std"] == pytest.approx(float(data.std(ddof=1)), rel=1e-12)',
    '    assert d["min"] == float(data.min())',
    '    assert d["max"] == float(data.max())',
    '',
    '',
    '# ============ two-level AST guards (§13.4/190) ============',
    '',
    '',
    'def test_describe_facade_delegates_to_describe_columns():',
    '    """Facade: describe calls describe_columns; no self.quantile."""',
    '    from cadw.dataframe import CadwDataFrame',
    '    tree = parse_method_source(CadwDataFrame.describe)',
    '    call_names = {',
    '        n.func.id for n in ast.walk(tree)',
    '        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)',
    '    }',
    '    assert "describe_columns" in call_names',
    '    src = inspect.getsource(CadwDataFrame.describe)',
    '    assert_no_call_in_source(src, "quantile", "to_pandas",',
    '                              fn_name="CadwDataFrame.describe")',
    '',
    '',
    'def test_describe_impl_uses_welford_and_reservoir():',
    '    """Impl: DescribeOp has Welford (M2) + reservoir attributes."""',
    '    src = inspect.getsource(DescribeOp)',
    '    assert "M2" in src',
    '    assert "reservoir" in src',
    '    assert "Algorithm R" in src or "randint" in src',
    '',
    '',
    'def test_describe_impl_no_to_pandas_ast():',
    '    from cadw.ops import describe as desc_mod',
    '    for fn_name in ("describe_columns",):',
    '        src = inspect.getsource(getattr(desc_mod, fn_name))',
    '        assert_no_call_in_source(src, "to_pandas", fn_name=fn_name)',
    '    # DescribeOp.update method',
    '    src = inspect.getsource(DescribeOp.update)',
    '    assert_no_call_in_source(src, "to_pandas", fn_name="DescribeOp.update")',
    '',
    '',
    'def test_describe_impl_uses_reduce_op_protocol():',
    '    """DescribeOp is a ReduceOp; describe_columns uses reduce()."""',
    '    from cadw.ops.reduce import ReduceOp',
    '    assert issubclass(DescribeOp, ReduceOp)',
    '    src = inspect.getsource(describe_columns)',
    '    assert "reduce(" in src',
    '',
    '',
    '# ============ speedup smoke (slow) ============',
    '',
    '',
    '@pytest.mark.slow',
    'def test_describe_100k_faster_than_manual_5_reduce(tmp_path):',
    '    """100k rows, 2 cols: fused single-pass vs manual 5-Reduce+sort.',
    '',
    '    Benchmark reference (audit bench_r5): 109× at N=2M. This test',
    '    asserts only that fused path is faster than manual equivalent',
    '    (each of 5 ReduceOps called separately + df.quantile per col);',
    '    correctness on exact fields covered above.',
    '    """',
    '    from cadw.ops.reduce import (',
    '        reduce, CountOp, MeanOp, StdOp, MinMaxOp,',
    '    )',
    '    rng = np.random.default_rng(0)',
    '    data = rng.normal(size=100_000)',
    '    df = make_df(tmp_path, {',
    '        "a": data.tolist(),',
    '        "b": data.tolist(),',
    '    }, chunk=5000, mem_mb=50)',
    '',
    '    # Fused path',
    '    t0 = time.time()',
    '    desc = df.describe()',
    '    t_fused = time.time() - t0',
    '    assert "a" in desc and "b" in desc',
    '',
    '    # Manual equivalent (5 Reduce per col + 1 quantile per col)',
    '    t0 = time.time()',
    '    manual = {}',
    '    for c in ("a", "b"):',
    '        n = reduce(df._iter_filtered_batches(), CountOp(c),',
    '                   df._memory_guard, log=False)',
    '        m = reduce(df._iter_filtered_batches(), MeanOp(c),',
    '                   df._memory_guard, log=False)',
    '        s = reduce(df._iter_filtered_batches(), StdOp(c),',
    '                   df._memory_guard, log=False)',
    '        mn = reduce(df._iter_filtered_batches(), MinMaxOp(c, "min"),',
    '                    df._memory_guard, log=False)',
    '        mx = reduce(df._iter_filtered_batches(), MinMaxOp(c, "max"),',
    '                    df._memory_guard, log=False)',
    '        q = df.quantile(c, [0.25, 0.5, 0.75])',
    '        manual[c] = {"count": n, "mean": m, "std": s,',
    '                     "min": mn, "25%": q[0], "50%": q[1],',
    '                     "75%": q[2], "max": mx}',
    '    t_manual = time.time() - t0',
    '',
    '    # Exact fields match',
    '    for c in ("a", "b"):',
    '        assert desc[c]["count"] == manual[c]["count"]',
    '        assert desc[c]["mean"] == pytest.approx(manual[c]["mean"], rel=1e-9)',
    '        assert desc[c]["std"] == pytest.approx(manual[c]["std"], rel=1e-9)',
    '        assert desc[c]["min"] == manual[c]["min"]',
    '        assert desc[c]["max"] == manual[c]["max"]',
    '',
    '    # Fused should be strictly faster (no 5× re-scan + sort per col)',
    '    assert t_fused < t_manual, (',
    '        f"fused describe ({t_fused:.3f}s) not faster than manual 5-Reduce+sort "',
    '        f"({t_manual:.3f}s)"',
    '    )',
]


def create_tests():
    print("\n=== create tests/test_describe_arrow_native.py ===")
    p = Path("tests/test_describe_arrow_native.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content_checked(p, NEW_TEST_LINES)
    print("  written + compiled")


# ============ 5. dogfood ============

def dogfood():
    print("\n=== dogfood: fused describe + AST guards ===")
    from cadw.dataframe import CadwDataFrame
    from cadw.ops import describe as desc_mod
    from cadw.patch_utils import assert_no_call_in_source

    # Facade: no quantile / to_pandas.
    src = inspect_getsource(CadwDataFrame.describe)
    assert_no_call_in_source(src, "quantile", "to_pandas",
                              fn_name="CadwDataFrame.describe")
    print("  facade describe: no quantile / to_pandas")

    # Impl has Welford + reservoir.
    isrc = inspect_getsource(desc_mod.DescribeOp)
    assert "M2" in isrc
    assert "reservoir" in isrc
    print("  DescribeOp: Welford (M2) + reservoir present")

    # Quick correctness in-process.
    import pyarrow as pa
    from cadw.core.memory_guard import MemoryGuard
    batches = [
        pa.record_batch({"x": [1.0, 2.0, 3.0]}),
        pa.record_batch({"x": [4.0, 5.0]}),
    ]
    res = desc_mod.describe_columns(
        iter(batches), ["x"], MemoryGuard(max_memory_mb=10),
    )
    assert res["x"]["count"] == 5
    assert abs(res["x"]["mean"] - 3.0) < 1e-12
    assert abs(res["x"]["std"] - np.sqrt(2.5)) < 1e-12
    print(f"  in-process describe: count={res['x']['count']}, "
          f"mean={res['x']['mean']:.3f}, std={res['x']['std']:.3f}")


# helper — avoid `import inspect` at module top level to prevent
# assert_test_imports_defined false positive in tests that use it via
# local import.
def inspect_getsource(fn):
    import inspect as _i
    return _i.getsource(fn)


# ============ main ============

def main():
    print("=== v0.9.7b: DescribeOp fused (R5) ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    create_describe_module()
    replace_describe_facade()
    update_whitelist()
    create_tests()
    dogfood()

    print("\n=== pytest: describe + regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_describe_arrow_native.py",
         "tests/test_dataframe_v094b2.py",
         "tests/test_dataframe_v094b.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DESCRIBE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.7b: DescribeOp fused (R5: 109× speedup, 12.1× RSS "
        "reduction). New cadw/ops/describe.py::DescribeOp — single "
        "Reduce pass per column group: Welford (mean/M2/std) + "
        "min/max + reservoir (Algorithm R, size=10000). Replaces 5 "
        "Reduce passes + 1 external_sort per column (2.47s/19.3MB "
        "→ 0.02s/1.6MB at N=2M, 2 cols). Facade CadwDataFrame."
        "describe delegates to describe_columns (no self.quantile, "
        "no to_pandas). Quantiles: exact for N ≤ reservoir_size, "
        "approximate above (§11.5). min/max/count/mean/std exact "
        "(Welford). Std ddof=1 (matches existing StdOp). Two-level "
        "tests (§13.4/190): facade delegates + impl Welford/reservoir; "
        "backward compat: existing test_dataframe_v094b2 describe "
        "tests (N=5/3/2) unchanged; new: 20k-N approximate tolerance, "
        "numpy equivalence exact fields, tight budget fail-fast, "
        "slow speedup smoke vs manual 5-Reduce+sort. Whitelist "
        "cleanup: remove stale describe entry (no longer high-risk).",
        self_path="scripts/patch_v097b_describe.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.7b applied. v0.9.7 (R4 + R5) complete.")


if __name__ == "__main__":
    main()
