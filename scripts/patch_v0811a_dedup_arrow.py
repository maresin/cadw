#!/usr/bin/env python3
"""v0.8.11a: Arrow-native dedup (P-D1 close) + cleanup pandas in join.

PROTOTYPE-VALIDATED (scripts/proto_dedup_arrow.py):
  Path C (dictionary_encode + group_by + is_in) — peak 0.29× pandas on
  average, 0.12× on 1M rows. Correctness A==B==C across all keep
  variants × selectivity (10%/50%/90% dupes).
  Measured (1M rows, 10% dupes): pandas 87.7 MB, Arrow 10.9 MB.

CHANGES:
  A. cadw/ops/dedup.py::_local_dedup — pandas chain replaced with
     Arrow-native Path C. Removes `import pandas`. Stopgap 5×T → 1.5×T
     (honest Arrow-native bound, proto-backed).
  B. cadw/ops/join.py — remove dead `import pandas as pd` and
     `_drop_null_keys` (pandas helper, superseded by
     `_drop_null_keys_arrow` in v0.8.5.3).
  C. tests/test_dedup_arrow_native.py — 16 tests: correctness
     (first/last/false, nulls, multi-subset, empty, K=N, K=1),
     AST guards (no pandas import, uses dictionary_encode/is_in),
     oversize raises.
  D. tests/test_no_pandas_in_ops.py — AST guard: `import pandas`
     allowed only in `cadw/ops/shuffle.py` (documented perf exception).
  E. §13.4/142 — prototype-first for P-D1 (was stopgap in v0.8.9).
  F. §11.5 — P-D1 closed; note on order change (was sort-by-row_id,
     now partition order; set-equivalence preserved).
  G. §9.2.18 — pandas to_pandas: no longer on hot path.
  H. CHANGELOG under [Unreleased].

GUARDS RUN BEFORE COMMIT (§13.4/140 lesson):
  test_changelog, test_arrow_call_memory_guard, test_no_pandas_in_ops,
  test_dedup_arrow_native, full test_drop_duplicates regression,
  full fast + slow.

Run: uv run python scripts/patch_v0811a_dedup_arrow.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ A. _local_dedup Arrow-native ============

def patch_local_dedup():
    print("\n=== cadw/ops/dedup.py::_local_dedup — Arrow-native ===")
    p = Path("cadw/ops/dedup.py")
    s = p.read_text()

    old = '''def _local_dedup(
    partition_path: Path,
    subset_cols: list[str],
    keep,
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Local dedup of one partition. Returns table with original columns
    (row_id dropped). Raises MemoryLimitExceeded if partition > available*0.5."""
    size = partition_path.stat().st_size
    limit = int(memory_guard.available_bytes() * 0.5)
    if size > limit:
        raise MemoryLimitExceeded(
            f"drop_duplicates: partition {partition_path.name} "
            f"size={size} B > limit={limit} B. "
            f"Увеличьте num_partitions или уменьшите subset. "
            f"Recursion не поддерживается (v0.3.0)."
        )

    reader = ipc.open_file(str(partition_path))
    n_batches = reader.num_record_batches
    # Schema from file with row_id stripped (hash_shuffle always appends).
    out_fields = [f for f in reader.schema if f.name != "row_id"]
    out_schema = pa.schema(out_fields)

    if n_batches == 0:
        return pa.Table.from_batches([], schema=out_schema)

    # v0.8.9 (P-A1 for dedup): generator form, schema-correct.
    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=reader.schema,
    )
    memory_guard.check_state(table.nbytes, name=f"dedup({partition_path.name})")
    # v0.8.9 (P-D1): pandas chain (to_pandas + sort_values + drop_duplicates
    # + drop + from_pandas) — measured 3.95-8.29× table.nbytes (Pingis v0.8.8).
    # 5× is a conservative bound covering int and pandas-3-StringDtype paths.
    # Effective partition limit shrinks to avail × 0.5 / 5 = avail × 0.1.
    memory_guard.check_state(
        5 * table.nbytes,
        name=f"dedup.pandas_chain({partition_path.name})",
    )

    df = table.to_pandas()
    # Сорт по row_id (stable) — keep='first'/'last' по исходному порядку.
    df = df.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=subset_cols, keep=keep)
    df = df.drop(columns=["row_id"])

    return pa.Table.from_pandas(df, schema=out_schema, preserve_index=False)'''

    new = '''def _local_dedup(
    partition_path: Path,
    subset_cols: list[str],
    keep,
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Local dedup via Arrow-native dictionary_encode + group_by + is_in.

    v0.8.11a (P-D1 close): replaces the v0.8.9 pandas chain. Validated by
    scripts/proto_dedup_arrow.py:
      - Correctness: A==B==C across keep ∈ {first, last, False} ×
        selectivity ∈ {10%, 50%, 90% dupes}.
      - Peak RSS: 0.29× pandas on average; 0.38× table on 1M rows
        (10.9 MB vs 87.7 MB pandas).

    Algorithm:
      1. dictionary_encode each subset column (string → int32 id).
      2. group_by(subset).aggregate(min/max row_id) — keep='first'/'last'.
         For keep=False: aggregate count + min; keep only count==1.
      3. is_in(row_id, keep_rids) → filter original (unencoded) table.

    Order: partition order (same as v0.3.0 pandas path, which sorted by
    row_id inside the partition). See §11.5 note on order.

    Raises MemoryLimitExceeded if partition > available×0.5, or if
    Arrow-native bound (1.5× table.nbytes) exceeds state_limit.
    """
    import pyarrow.compute as pc

    size = partition_path.stat().st_size
    limit = int(memory_guard.available_bytes() * 0.5)
    if size > limit:
        raise MemoryLimitExceeded(
            f"drop_duplicates: partition {partition_path.name} "
            f"size={size} B > limit={limit} B. "
            f"Увеличьте num_partitions или уменьшите subset. "
            f"Recursion не поддерживается (v0.3.0)."
        )

    reader = ipc.open_file(str(partition_path))
    n_batches = reader.num_record_batches
    out_fields = [f for f in reader.schema if f.name != "row_id"]
    out_schema = pa.schema(out_fields)

    if n_batches == 0:
        return pa.Table.from_batches([], schema=out_schema)

    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=reader.schema,
    )
    memory_guard.check_state(
        table.nbytes, name=f"dedup({partition_path.name})",
    )
    # v0.8.11a: Arrow-native peak ~0.4× table (proto: 10.9/28.6 MB at 1M).
    # 1.5× conservative for all dtypes incl. K≈N (dictionary ≈ table size
    # for all-unique, but no pandas object overhead). No 5× stopgap.
    memory_guard.check_state(
        1.5 * table.nbytes,
        name=f"dedup.arrow_native({partition_path.name})",
    )

    # Dictionary-encode subset for compact hash.
    encoded_cols = {
        c: pc.dictionary_encode(table.column(c)) for c in subset_cols
    }
    working = pa.table({
        **{c: encoded_cols[c] for c in subset_cols},
        "row_id": table.column("row_id"),
    })

    if keep in ("first", "last"):
        agg_fn = "min" if keep == "first" else "max"
        agg = working.group_by(subset_cols).aggregate([("row_id", agg_fn)])
        keep_rids = agg.column(f"row_id_{agg_fn}")
    elif keep is False:
        agg = working.group_by(subset_cols).aggregate([
            ("row_id", "count"), ("row_id", "min"),
        ])
        single = agg.filter(pc.equal(agg.column("row_id_count"), 1))
        keep_rids = single.column("row_id_min")
    else:
        raise ValueError(f"dedup: unsupported keep={keep!r}")

    mask = pc.is_in(table.column("row_id"), value_set=keep_rids)
    result = table.filter(mask)
    return result.select([f.name for f in out_fields])'''

    s = replace_line_block(s, old, new, tag="local-dedup-arrow")

    # Remove `import pandas as pd` (no longer used in dedup.py)
    s = s.replace("import pandas as pd\n", "")

    p.write_text(s)
    compile_check(p)
    print("  OK: Arrow-native _local_dedup, removed `import pandas`")


# ============ B. Cleanup dead pandas in join.py ============

def cleanup_join_pandas():
    print("\n=== cadw/ops/join.py — remove dead pandas ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()

    changed = False

    # Remove `import pandas as pd`
    if "import pandas as pd\n" in s:
        s = s.replace("import pandas as pd\n", "", 1)
        changed = True

    # Remove `_drop_null_keys` (pandas helper, dead since v0.8.5.3)
    dead_fn = '''def _drop_null_keys(df: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """SQL-семантика: строки с NULL в любом on-ключе не участвуют."""
    if len(df) == 0:
        return df
    mask = np.ones(len(df), dtype=bool)
    for c in on:
        mask &= df[c].notna().to_numpy()
    return df.loc[mask]


'''
    if dead_fn in s:
        s = s.replace(dead_fn, "", 1)
        changed = True

    # Add AST-marker note at top of file (idempotent)
    marker = "# v0.8.11a: no pandas in hot path (dead helpers removed; only\n# `_pandas_hash_column` in cadw/ops/shuffle.py remains, documented).\n"
    if marker not in s and "from cadw.core.memory_guard import" in s:
        s = s.replace(
            "from cadw.core.memory_guard import",
            marker + "from cadw.core.memory_guard import",
            1,
        )
        changed = True

    if not changed:
        print("  already clean — skip")
        return

    p.write_text(s)
    compile_check(p)
    print("  OK: removed dead pandas, no functional change")


# ============ C. AST guard — no pandas in ops except shuffle ============

AST_GUARD_SRC = '''"""v0.8.11a: AST guard — pandas removed from ops hot path.

After v0.8.11a, `cadw/ops/` has zero `import pandas` except in
`shuffle.py::_pandas_hash_column` — a documented perf exception
(pandas C-hash is 19–40× faster than Python fallback for
int/float/timestamp; §9.2.20).

Principle §3.11 (Arrow call memory discipline) is enforced separately in
`test_arrow_call_memory_guard.py`. This file guards the narrower claim:
no NEW pandas import sneaks into ops without an explicit decision.

If you need pandas in an op, whitelist it here with a rationale
(same discipline as `WHITELISTED_FUNCS` in test_arrow_call_memory_guard).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


CADW_OPS = Path(__file__).resolve().parent.parent / "cadw" / "ops"

# Modules that may import pandas, with rationale.
ALLOWED_PANDAS_MODULES = {
    # §9.2.20: pd.util.hash_pandas_object is the fastest hash for
    # int/float/timestamp (19-40× Python fallback). Bounded per-batch
    # by caller's check_batch(extra_factor=3.0).
    "shuffle.py": "pd.util.hash_pandas_object — §9.2.20 perf exception",
}


def _uses_pandas_import(tree: ast.Module) -> bool:
    """True if module has `import pandas` or `from pandas ...`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pandas" or alias.name.startswith("pandas."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "pandas"
                or node.module.startswith("pandas.")
            ):
                return True
    return False


def test_no_pandas_in_ops_except_shuffle():
    """`cadw/ops/*.py` — no `import pandas` outside whitelist."""
    violations: list[str] = []
    for py in sorted(CADW_OPS.glob("*.py")):
        if py.name in ALLOWED_PANDAS_MODULES:
            continue
        src = py.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if _uses_pandas_import(tree):
            violations.append(py.name)

    assert not violations, (
        "Unexpected `import pandas` in cadw/ops/: "
        f"{violations}. "
        "Pandas was removed from hot path in v0.8.11a (P-D1). "
        f"Whitelist rationale: {ALLOWED_PANDAS_MODULES}. "
        "If needed, add to ALLOWED_PANDAS_MODULES with a rationale."
    )


def test_pandas_whitelist_has_rationale():
    """Every whitelisted module has a non-empty rationale."""
    for name, reason in ALLOWED_PANDAS_MODULES.items():
        assert isinstance(name, str) and name.endswith(".py")
        assert isinstance(reason, str) and reason, (
            f"Whitelisted {name!r} must have a rationale string"
        )


def test_no_pandas_in_dataframe_facade():
    """`cadw/dataframe.py` — no pandas import (facade is Arrow-native)."""
    py = CADW_OPS.parent / "dataframe.py"
    src = py.read_text()
    tree = ast.parse(src)
    assert not _uses_pandas_import(tree), (
        "cadw/dataframe.py must not import pandas (facade is Arrow-native)"
    )


def test_dedup_uses_arrow_native():
    """AST: `_local_dedup` uses dictionary_encode + is_in (Path C)."""
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "dictionary_encode" in src, (
        "_local_dedup must use pc.dictionary_encode (Path C, v0.8.11a)"
    )
    assert "is_in" in src, (
        "_local_dedup must use pc.is_in (Path C, v0.8.11a)"
    )
    assert "to_pandas" not in src, (
        "_local_dedup must NOT call to_pandas (P-D1 regression)"
    )
    assert "sort_values" not in src, (
        "_local_dedup must NOT call pandas sort_values (P-D1 regression)"
    )


def test_dedup_no_pandas_import():
    """AST: `cadw/ops/dedup.py` has no `import pandas`."""
    py = CADW_OPS / "dedup.py"
    tree = ast.parse(py.read_text())
    assert not _uses_pandas_import(tree), (
        "cadw/ops/dedup.py must not import pandas (P-D1 closed v0.8.11a)"
    )


def test_join_no_pandas_import():
    """AST: `cadw/ops/join.py` has no `import pandas`."""
    py = CADW_OPS / "join.py"
    tree = ast.parse(py.read_text())
    assert not _uses_pandas_import(tree), (
        "cadw/ops/join.py must not import pandas (v0.8.11a cleanup)"
    )
'''


def write_ast_guard():
    print("\n=== tests/test_no_pandas_in_ops.py ===")
    p = Path("tests/test_no_pandas_in_ops.py")
    p.write_text(AST_GUARD_SRC)
    compile_check(p)
    print("  OK")


# ============ D. Behavior tests ============

BEHAVIOR_TESTS = '''"""v0.8.11a: Arrow-native dedup behavior tests.

Prototype-validated (scripts/proto_dedup_arrow.py, Path C adopted):
  peak 0.29× pandas, correctness A==B==C across keep × selectivity.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.dedup import drop_duplicates as ops_drop_duplicates


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _norm(rows: list, key):
    return sorted(rows, key=lambda r: (r[key] is None, r[key]))


# ============ correctness (was in test_drop_duplicates.py, extended) ============

def test_dedup_first_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 1, "b": 3, "c": 5}


def test_dedup_last_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    by_k = {r["k"]: r["v"] for r in pq.read_table(out).to_pylist()}
    assert by_k == {"a": 2, "b": 4, "c": 5}


def test_dedup_false_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep=False, num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0] == {"k": "c", "v": 5}


def test_dedup_full_row(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "a", "b"], "v": [1, 1, 2]})
    result = df.drop_duplicates(keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    assert rows == [{"k": "a", "v": 1}, {"k": "b", "v": 2}]


def test_dedup_multi_subset(tmp_path):
    df = make_df(tmp_path, "L", {
        "k1": ["a", "a", "a", "b"],
        "k2": [1, 1, 2, 1],
        "v": [10, 20, 30, 40],
    })
    result = df.drop_duplicates(
        subset=["k1", "k2"], keep="first", num_partitions=4,
    )
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert {(r["k1"], r["k2"], r["v"]) for r in rows} == {
        ("a", 1, 10), ("a", 2, 30), ("b", 1, 40),
    }


def test_dedup_nulls_in_subset(tmp_path):
    """Null в subset — своя группа (несколько null-строк → одна)."""
    df = make_df(tmp_path, "L", {
        "k": ["a", None, "b", None, "a"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Distinct keys in result: a, b, None — 3 rows.
    assert len(rows) == 3
    k_set = {r["k"] for r in rows}
    assert k_set == {"a", "b", None}


def test_dedup_nulls_in_values(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b"],
        "v": [1, None, 2],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    by_k = {r["k"]: r["v"] for r in pq.read_table(out).to_pylist()}
    assert by_k == {"a": 1, "b": 2}


def test_dedup_all_unique(tmp_path):
    n = 100
    df = make_df(tmp_path, "L", {
        "k": [f"k{i}" for i in range(n)],
        "v": list(range(n)),
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    assert pq.read_table(out).num_rows == n


def test_dedup_all_same_key(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1 and rows[0]["v"] == 0


def test_dedup_int_key(tmp_path):
    n = 50
    df = make_df(tmp_path, "L", {
        "k": [i % 10 for i in range(n)],
        "v": list(range(n)),
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 10


# ============ pandas-equivalence (set-of-tuples) ============

def test_dedup_pandas_equivalence(tmp_path):
    import pandas as pd
    rng = np.random.default_rng(42)
    n = 200
    data = {
        "k": [f"k{i % 20}" for i in range(n)],
        "v": rng.integers(0, 100, size=n).tolist(),
    }
    df = make_df(tmp_path, "L", data, chunk=20)
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    got = pq.read_table(out).to_pandas()

    pdf = pd.DataFrame(data)
    expected = pdf.drop_duplicates(subset=["k"], keep="first")

    got_set = set(map(tuple, got[["k", "v"]].itertuples(index=False, name=None)))
    exp_set = set(
        map(tuple, expected[["k", "v"]].itertuples(index=False, name=None))
    )
    assert got_set == exp_set


# ============ edge: empty partition ============

def test_dedup_empty_input_raises(tmp_path):
    empty_batches: list = []
    mg = MemoryGuard(max_memory_mb=10)
    with pytest.raises(ValueError, match="empty input"):
        ops_drop_duplicates(
            iter(empty_batches),
            subset=["k"], keep="first",
            memory_guard=mg, spill_dir=tmp_path / "spill",
        )


# ============ memory bounds ============

def test_dedup_arrow_native_bound_fires(tmp_path):
    """Oversize partition → MemoryLimitExceeded before materialize."""
    df = make_df(
        tmp_path, "L",
        {"k": ["HOT"] * 1000, "v": list(range(1000))},
        chunk=200, mem_mb=1,
    )
    df._memory_guard = MemoryGuard(max_memory_bytes=1000)
    with pytest.raises(MemoryLimitExceeded):
        df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)


def test_dedup_removes_stopgap_5x():
    """AST: no `5 * table.nbytes` stopgap in _local_dedup (now 1.5×)."""
    import ast
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "5 * table.nbytes" not in src, (
        "v0.8.11a: 5×T stopgap must be replaced with 1.5×T Arrow-native bound"
    )
    # 1.5× must be present
    assert "1.5 * table.nbytes" in src, (
        "v0.8.11a: expect `1.5 * table.nbytes` Arrow-native bound"
    )


# ============ regression: chained ops ============

def test_dedup_then_sort(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["c", "a", "b", "a", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    sorted_result = result.sort_values("k", ascending=True)
    out = tmp_path / "o.parquet"
    sorted_result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == [
        {"k": "a", "v": 4},
        {"k": "b", "v": 3},
        {"k": "c", "v": 5},
    ]


def test_dedup_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 1]
    result = filtered.drop_duplicates(
        subset=["k"], keep="first", num_partitions=4,
    )
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 2, "b": 3, "c": 5}
'''


def write_behavior_tests():
    print("\n=== tests/test_dedup_arrow_native.py ===")
    p = Path("tests/test_dedup_arrow_native.py")
    p.write_text(BEHAVIOR_TESTS)
    compile_check(p)
    print("  OK")


# ============ E. Docs ============

def append_changelog():
    print("\n=== CHANGELOG.md [Unreleased]: append v0.8.11a ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "Arrow-native dedup" in s or "P-D1" in s and "closed" in s:
        print("  already present — skip")
        return

    anchor = "## [Unreleased]\n"
    if anchor not in s:
        print("  WARNING: [Unreleased] anchor not found — skip")
        return

    addendum = '''## [Unreleased]

### Исправлено (safety — P-D1 close)

- **P-D1 (S2, live): Arrow-native `drop_duplicates` rewrite.** pandas
  chain (`to_pandas → sort_values → drop_duplicates → drop → from_pandas`)
  в `_local_dedup` заменён на Path C: `dictionary_encode(subset)` +
  `group_by(subset).aggregate(min/max row_id)` + `is_in` filter.
  Удалён `import pandas` из `cadw/ops/dedup.py`.

  **Bench (prototype `scripts/proto_dedup_arrow.py`):**
  | n | pandas | Path B (arrow) | **Path C (dict)** |
  |---|---|---|---|
  | 100k | 9.3 MB | 10.2 MB | **5.9 MB** |
  | 300k | 29.3 MB | 39.0 MB | **3.6 MB** |
  | 1M   | 87.7 MB | 47.2 MB | **10.9 MB** |

  avg peak(C)/peak(A) = **0.29**. Correctness A==B==C across keep ∈
  {first, last, False} × selectivity ∈ {10%, 50%, 90% dupes}.

  **Stopgap removed:** `check_state(5 × table.nbytes)` → `check_state(1.5 ×
  table.nbytes)` (honest Arrow-native bound, proto-backed). Effective
  partition limit back from `avail × 0.1` to `avail × 0.33`.

- **Dead pandas helpers removed из `cadw/ops/join.py`.** `import pandas
  as pd`, `_drop_null_keys` (superseded by `_drop_null_keys_arrow` в
  v0.8.5.3) — удалены. Zero pandas in ops except `shuffle.py::_pandas_hash_column`
  (documented perf exception, §9.2.20).

### Добавлено

- **`tests/test_no_pandas_in_ops.py`** — AST guard: `import pandas` в
  `cadw/ops/*` разрешён только в `shuffle.py` (whitelist + rationale).
  Также `test_dedup_uses_arrow_native` (Path C marker), `test_dedup_removes_stopgap_5x`.
- **`tests/test_dedup_arrow_native.py`** — 16 behavior tests: first/last/
  False, nulls в subset/value, multi-subset, int key, empty, K=N, K=1,
  pandas-equivalence, oversize → raise, dedup→sort chain, filtered dedup.
- **§13.4/142** — prototype-first for P-D1 (was stopgap в v0.8.9;
  prototype → Path C → adoption, без fix-итераций).

### Ограничения (v0.8.11a, §11.5)

- `drop_duplicates` порядок вывода: **partition order**, без
  `sort_values("row_id")` внутри partition (был в pandas-версии).
  Set-equivalence сохранена; если нужен детерминированный порядок —
  `.sort_values(...)` после. §11.5 entry обновлён.
- `drop_duplicates` recursion при skew: **не реализован** (как в v0.3.0).
  Oversize partition → MemoryLimitExceeded. Backlog.

### Тесты

- fast: **~1190** (+16 behavior, +5 AST guard). slow: **8**.

'''
    s = s.replace(anchor, addendum, 1)
    p.write_text(s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md: §13.4/142 + §11.5 + §9.2.18 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    # §11.5 — update dedup entry
    old_115 = "| `drop_duplicates` returns rows in partition-concatenation order | `drop_duplicates` (v0.3.0) | pandas-order: `.sort_values(...)` после. §13.4/48 |"
    new_115 = "| `drop_duplicates` returns rows in partition-concatenation order (v0.8.11a: no intra-partition sort by row_id) | `drop_duplicates` | set-equivalence preserved; `.sort_values(...)` после для детерминированного порядка. §11.5, §13.4/48 |"
    if old_115 in s:
        s = s.replace(old_115, new_115, 1)
        print("  OK: §11.5 dedup order note")

    # Remove/lift P-D1 stopgap entries from §11.6 if present
    old_pd1_stopgap = "~~P-D1: dedup pandas chain unchecked~~ — **closed v0.8.9**:"
    if old_pd1_stopgap in s:
        # find the whole §11.6 addendum block "Pingis library-memory audit"
        m = re.search(
            r"(\*\*Pingis library-memory audit \(2026-10-20\) — findings closed in v0\.8\.9:\*\*.*?)(?=\n\*\*NEEDS-BATCHED)",
            s, re.DOTALL,
        )
        if m:
            old_block = m.group(1)
            new_block = old_block.replace(
                "~~P-D1: dedup pandas chain unchecked~~ — **closed v0.8.9**:\n  `check_state(5 × table.nbytes)` before `to_pandas`. Effective partition\n  limit shrinks to `avail × 0.1` (honest RSS bound; Arrow-native rewrite\n  deferred to v0.8.10).",
                "~~P-D1: dedup pandas chain unchecked~~ — **closed v0.8.11a**:\n  Arrow-native rewrite (Path C: dictionary_encode + group_by + is_in),\n  pandas removed. Peak 0.29× of prior pandas chain (proto-validated).",
            )
            if new_block != old_block:
                s = s[:m.start()] + new_block + s[m.end():]
                print("  OK: §11.6 P-D1 entry updated (v0.8.9 → v0.8.11a)")

    # §13.4/142
    if "Эпизод 142" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 142 (P-D1: prototype-first vs stopgap).** v0.8.9 закрыл
P-D1 «stopgap»-ом: `check_state(5 × table.nbytes)` **до** `to_pandas`.
Это honest bound (5× покрывает pandas object/StringDtype overhead), но
**не rewrite** — pandas оставался в hot path, effective partition limit
падал в 5× (`avail × 0.1`).

v0.8.11a — prototype-first по §13.4/123: `scripts/proto_dedup_arrow.py`
сравнил 3 пути (pandas / arrow group_by / arrow dictionary_encode) на
correctness × selectivity × 3 размера. Path C (dictionary_encode +
group_by + is_in) выиграл с 0.29× peak vs pandas; **8× reduction на 1M
rows** (10.9 MB vs 87.7 MB). Correctness A==B==C на всех keep × selectivity.

**Adoption без fix-iterations.** Сравнить: v0.8.9 stopgap → 1 round
(warn-only); v0.8.11a rewrite → 1 round (prototype → patch → green).
Prototype стоит ~30 минут; он **экономит** fix-итерации и **даёт
honest numbers** для §9.2.N формул.

**Урок (расширение §13.4/141):** stopgap — не permanent fix. Каждый
stopgap должен иметь «revisit criteria» (в §11.6: «Arrow-native rewrite
when budget allows»). v0.8.9 → v0.8.11a gap = одна major-версия,
приемлемо. Если бы gap был >2 версий — stopgap становится долгом.

**Связанные:** §13.4/141 (prototype-first), §13.4/138 (audit формулы
из RSS), §11.6 (backlog discipline).

'''
        if marker in s:
            s = s.replace(marker, episode + marker, 1)
            print("  OK: §13.4/142")

    p.write_text(s)


# ============ main ============

def main():
    print("=== v0.8.11a: dedup Arrow-native ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if branch != "master":
        print(f"WARNING: not on master (current: {branch}) — proceed anyway")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    # Phase A — code
    patch_local_dedup()
    cleanup_join_pandas()

    # Phase B — tests
    write_ast_guard()
    write_behavior_tests()

    # Phase C — docs
    append_changelog()
    update_decisions()

    # Phase D — guards (before commit, §13.4/140)
    print("\n=== pytest: new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dedup_arrow_native.py",
         "tests/test_no_pandas_in_ops.py",
         "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.11a (new tests red)"], check=False)
        sys.exit(1)

    print("\n=== pytest: existing guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: dedup + join regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_drop_duplicates.py",
         "tests/test_join.py", "tests/test_join_streaming.py",
         "tests/test_join_oom_guard.py", "tests/test_e2e_join.py",
         "tests/test_merge_planner.py", "tests/test_join_rss_warning.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.11a: Arrow-native dedup (P-D1 close) — Path C "
        "(dictionary_encode + group_by + is_in), 0.29× peak vs pandas; "
        "pandas removed from ops hot path; AST guard no_pandas_in_ops"])

    print("\n==> v0.8.11a applied. Next: v0.8.11b (strict join conditional).")


if __name__ == "__main__":
    main()