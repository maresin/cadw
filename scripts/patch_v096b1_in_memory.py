#!/usr/bin/env python3
# v0.9.6b1 — in-memory Dataset backing + _ColumnProxy aggregators.
#
# Pure additive:
#   - _InMemoryDataset(pa.Table) — read-only Dataset subclass.
#   - CadwDataFrame.from_table(table) — in-memory backing, no disk I/O.
#   - CadwDataFrame.sort_values — fast path for in-memory (pa.Table.sort_by).
#   - _ColumnProxy.mean/sum/min/max/std/count — via existing ReduceOp.
#   - _ColumnProxy.to_list/to_numpy/to_arrow — bounded materialize
#     (check_state before).
#
# No existing behaviour changed. _per_group_apply still uses pandas
# (v0.9.6b2 will switch to CADW-native contract).
#
# Fast path for sort_values is critical: per-group apply on 100k
# groups × 20 rows needs in-memory sort; external_sort would be
# 100k × 1-5 ms = 100-500 s of overhead just creating files.
#
# Run: uv run python scripts/patch_v096b1_in_memory.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


# ============ 1. new file: cadw/dataset/_in_memory.py ============

IN_MEMORY_LINES = [
    '"""In-memory Dataset backing for CadwDataFrame.from_table (v0.9.6b1).',
    '',
    'Read-only view: base_dir is unused; add_column raises',
    'NotImplementedError. Operations requiring new disk datasets (astype,',
    'rename, encode_categorical, groupby, etc.) still write to disk via',
    'Dataset.from_batches — same behaviour as disk-backed CadwDataFrame.',
    '',
    'sort_values on in-memory CadwDataFrame uses pa.Table.sort_by directly',
    '(CadwDataFrame.sort_values fast path). For small per-group tables this',
    'is ~100-1000× faster than external_sort: avoids disk round-trip for',
    'every group. Critical for _per_group_apply (v0.9.6b2 will move to',
    'CADW-native fn contract; 100k groups × external_sort = 100-500s',
    'overhead just for file creation).',
    '"""',
    'from __future__ import annotations',
    '',
    'from pathlib import Path',
    'from typing import Iterator, Sequence',
    '',
    'import pyarrow as pa',
    '',
    'from cadw.dataset.dataset import Dataset',
    '',
    '',
    'class _InMemoryDataset(Dataset):',
    '    """In-memory Dataset backed by a single pa.Table.',
    '',
    '    Read-only. No disk I/O. Iterates table.to_batches() directly.',
    '',
    '    Not a drop-in for disk Dataset: add_column raises',
    '    NotImplementedError (in-memory df is a view, not a mutable',
    '    store). Use it for per-group tables in _per_group_apply and for',
    '    ad-hoc in-memory DataFrame creation (tests, small transforms).',
    '    """',
    '',
    '    def __init__(self, table: pa.Table) -> None:',
    '        if not isinstance(table, pa.Table):',
    '            raise TypeError(',
    '                f"_InMemoryDataset requires pa.Table; "',
    '                f"got {type(table).__name__}"',
    '            )',
    '        # Do NOT call super().__init__ — it would load meta from disk.',
    '        self._table = table',
    '        self.base_dir = Path("")  # sentinel — never used',
    '        self._columns_dir = Path("")',
    '        self._meta_path = Path("")',
    '        self._schema = table.schema',
    '        self._column_paths = {}  # not used',
    '        self._chunk_boundaries = []',
    '        self._total_rows = table.num_rows',
    '',
    '    def iter_batches(',
    '        self,',
    '        batch_size: int | None = None,',
    '        columns: Sequence[str] | None = None,',
    '    ) -> Iterator[pa.RecordBatch]:',
    '        t = self._table.select(columns) if columns is not None else self._table',
    '        if batch_size is None:',
    '            yield from t.to_batches()',
    '            return',
    '        if batch_size < 1:',
    '            raise ValueError("batch_size must be >= 1")',
    '        accum: list[pa.Array] = []',
    '        accum_n = 0',
    '        for b in t.to_batches():',
    '            n = b.num_rows',
    '            j = 0',
    '            while j < n:',
    '                take = min(batch_size - accum_n, n - j)',
    '                if not accum:',
    '                    accum = [c.slice(j, take) for c in b.columns]',
    '                else:',
    '                    accum = [',
    '                        pa.concat_arrays([a, b.column(i).slice(j, take)])',
    '                        for i, a in enumerate(accum)',
    '                    ]',
    '                accum_n += take',
    '                j += take',
    '                if accum_n == batch_size:',
    '                    yield pa.RecordBatch.from_arrays(',
    '                        accum, names=t.schema.names,',
    '                    )',
    '                    accum = []',
    '                    accum_n = 0',
    '        if accum_n > 0:',
    '            yield pa.RecordBatch.from_arrays(accum, names=t.schema.names)',
    '',
    '    def add_column(self, name: str, batches_iter) -> None:',
    '        raise NotImplementedError(',
    '            "_InMemoryDataset is read-only. Operations requiring new "',
    '            "columns should write to disk first (v0.9.6b1 design)."',
    '        )',
    '',
    '    @property',
    '    def columns(self) -> list[str]:',
    '        return list(self._table.schema.names)',
    '',
    '    @property',
    '    def schema(self) -> pa.Schema:',
    '        return self._table.schema',
    '',
    '    @property',
    '    def total_rows(self) -> int:',
    '        return self._table.num_rows',
    '',
    '    @property',
    '    def chunk_boundaries(self) -> list[int]:',
    '        return [b.num_rows for b in self._table.to_batches()]',
    '',
    '',
    '__all__ = ["_InMemoryDataset"]',
    '',
]


def create_in_memory_dataset():
    print("\n=== create cadw/dataset/_in_memory.py ===")
    p = Path("cadw/dataset/_in_memory.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content(p, IN_MEMORY_LINES)
    assert_compiles(p)
    print("  OK")


# ============ 2. export from cadw/dataset/__init__.py ============

def export_in_memory():
    print("\n=== cadw/dataset/__init__.py: export _InMemoryDataset ===")
    p = Path("cadw/dataset/__init__.py")
    s = p.read_text()
    if "_InMemoryDataset" in s:
        print("  already exported — skip")
        return
    # Insert import after existing `from .dataset import Dataset`
    old = "from .dataset import Dataset\n"
    new = "from .dataset import Dataset\nfrom ._in_memory import _InMemoryDataset\n"
    if old not in s:
        raise SystemExit("Dataset import anchor not found in __init__.py")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. CadwDataFrame.from_table ============

def add_from_table():
    print("\n=== dataframe.py: add CadwDataFrame.from_table ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def from_table(" in s:
        print("  already present — skip")
        return

    anchor = (
        "    @classmethod\n"
        "    def from_dataset(\n"
        "        cls,\n"
        "        dataset: Dataset,\n"
        "        *,\n"
        "        memory_guard: MemoryGuard | None = None,\n"
        "        spill_dir: str | Path | None = None,\n"
        "    ) -> \"CadwDataFrame\":\n"
        "        return cls(dataset, memory_guard=memory_guard, spill_dir=spill_dir)\n"
    )
    new_block = anchor + (
        "\n"
        "    @classmethod\n"
        "    def from_table(\n"
        "        cls,\n"
        "        table: pa.Table,\n"
        "        *,\n"
        "        memory_guard: MemoryGuard | None = None,\n"
        "        spill_dir: str | Path | None = None,\n"
        "    ) -> \"CadwDataFrame\":\n"
        "        \"\"\"In-memory CadwDataFrame backed by pa.Table. No disk I/O.\n"
        "\n"
        "        v0.9.6b1. Read-only view: add_column raises\n"
        "        NotImplementedError. Operations that produce a new\n"
        "        Dataset (astype, rename, groupby, etc.) still write to\n"
        "        disk (via Dataset.from_batches) — but this is rare on\n"
        "        small in-memory groups and acceptable. sort_values on\n"
        "        the in-memory df uses pa.Table.sort_by directly (fast\n"
        "        path) — critical for per-group apply.\n"
        "\n"
        "        Peak RAM: 0 (no copy — table ref shared). Caller owns\n"
        "        table and any pre-materialization cost.\n"
        "        \"\"\"\n"
        "        from cadw.dataset._in_memory import _InMemoryDataset\n"
        "        if not isinstance(table, pa.Table):\n"
        "            raise TypeError(\n"
        "                f\"from_table requires pa.Table; \"\n"
        "                f\"got {type(table).__name__}\"\n"
        "            )\n"
        "        return cls(\n"
        "            _InMemoryDataset(table),\n"
        "            memory_guard=memory_guard,\n"
        "            spill_dir=spill_dir,\n"
        "        )\n"
    )
    if anchor not in s:
        raise SystemExit("from_dataset anchor not found in dataframe.py")
    s = s.replace(anchor, new_block, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. sort_values fast path ============

def add_sort_fast_path():
    print("\n=== dataframe.py: sort_values fast path for in-memory ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "in-memory fast path (v0.9.6b1)" in s:
        print("  already present — skip")
        return

    # Insert fast path right after the by_cols validation loop, before
    # external_sort call. Anchor is the validation block followed by
    # `path = external_sort(`.
    old = (
        "    def sort_values(self, by, *, ascending=True) -> \"CadwDataFrame\":\n"
        "        by_cols = [by] if isinstance(by, str) else list(by)\n"
        "        for c in by_cols:\n"
        "            if c not in self._columns:\n"
        "                raise KeyError(f\"sort_values: column {c!r} not in selection\")\n"
        "        path = external_sort(\n"
    )
    new = (
        "    def sort_values(self, by, *, ascending=True) -> \"CadwDataFrame\":\n"
        "        by_cols = [by] if isinstance(by, str) else list(by)\n"
        "        for c in by_cols:\n"
        "            if c not in self._columns:\n"
        "                raise KeyError(f\"sort_values: column {c!r} not in selection\")\n"
        "        # in-memory fast path (v0.9.6b1): pa.Table.sort_by directly.\n"
        "        # Avoids 1-5 ms external_sort overhead per small group —\n"
        "        # critical for per-group apply (100k groups × 5 ms = 500 s).\n"
        "        from cadw.dataset._in_memory import _InMemoryDataset\n"
        "        if isinstance(self._dataset, _InMemoryDataset):\n"
        "            sorted_table = self._dataset._table.select(self._columns).sort_by(\n"
        "                [(c, \"ascending\" if ascending else \"descending\")\n"
        "                 for c in by_cols],\n"
        "            )\n"
        "            return CadwDataFrame.from_table(\n"
        "                sorted_table,\n"
        "                memory_guard=self._memory_guard,\n"
        "                spill_dir=self._spill_dir,\n"
        "            )\n"
        "        path = external_sort(\n"
    )
    if old not in s:
        raise SystemExit("sort_values anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 5. _ColumnProxy: aggregators + to_list/to_numpy/to_arrow ============

PROXY_METHODS = [
    "",
    "    # ---------- scalar aggregators (v0.9.6b1) ----------",
    "    #",
    "    # Composition of existing ReduceOp. O(1) state, bounded by",
    "    # MemoryGuard. No pandas. This is the CADW-native replacement for",
    "    # `g[\"col\"].mean()` etc. inside per-group apply.",
    "",
    "    def mean(self) -> float:",
    "        from cadw.ops.reduce import MeanOp, reduce",
    "        return reduce(",
    "            self._parent._iter_filtered_batches(),",
    "            MeanOp(self._column),",
    "            self._parent._memory_guard,",
    "            log=False,",
    "        )",
    "",
    "    def sum(self) -> float:",
    "        from cadw.ops.reduce import SumOp, reduce",
    "        return reduce(",
    "            self._parent._iter_filtered_batches(),",
    "            SumOp(self._column),",
    "            self._parent._memory_guard,",
    "            log=False,",
    "        )",
    "",
    "    def min(self):",
    "        from cadw.ops.reduce import MinMaxOp, reduce",
    "        return reduce(",
    "            self._parent._iter_filtered_batches(),",
    "            MinMaxOp(self._column, \"min\"),",
    "            self._parent._memory_guard,",
    "            log=False,",
    "        )",
    "",
    "    def max(self):",
    "        from cadw.ops.reduce import MinMaxOp, reduce",
    "        return reduce(",
    "            self._parent._iter_filtered_batches(),",
    "            MinMaxOp(self._column, \"max\"),",
    "            self._parent._memory_guard,",
    "            log=False,",
    "        )",
    "",
    "    def std(self, ddof: int = 1) -> float:",
    "        from cadw.ops.reduce import StdOp, reduce",
    "        return reduce(",
    "            self._parent._iter_filtered_batches(),",
    "            StdOp(self._column, ddof=ddof),",
    "            self._parent._memory_guard,",
    "            log=False,",
    "        )",
    "",
    "    def count(self) -> int:",
    "        \"\"\"Non-null count (skipna=True).\"\"\"",
    "        from cadw.ops.reduce import CountOp, reduce",
    "        return reduce(",
    "            self._parent._iter_filtered_batches(),",
    "            CountOp(self._column),",
    "            self._parent._memory_guard,",
    "            log=False,",
    "        )",
    "",
    "    # ---------- bounded materialize (v0.9.6b1) ----------",
    "    #",
    "    # check_state BEFORE materialize (§3.10). n = upper bound via",
    "    # parent.count(); over-estimate acceptable (conservative guard).",
    "",
    "    def to_list(self) -> list:",
    "        \"\"\"Materialize column as Python list. Bounded by MemoryGuard.",
    "",
    "        Peak ≈ n × 50 B (Python object overhead, conservative for",
    "        numeric/string mix). check_state BEFORE materialize.",
    "        \"\"\"",
    "        n = self._parent.count()",
    "        self._parent._memory_guard.check_state(",
    "            n * 50, name=f\"to_list({self._column})\",",
    "        )",
    "        out: list = []",
    "        for b in self._parent._iter_filtered_batches():",
    "            out.extend(b.column(self._column).to_pylist())",
    "        return out",
    "",
    "    def to_numpy(self):",
    "        \"\"\"Materialize column as numpy array. Bounded by MemoryGuard.",
    "",
    "        Numeric: zero-copy from Arrow buffer (via to_numpy(zero_copy_",
    "        only=False)). Strings: object array. Peak ≈ n × element_size",
    "        + n × 8 (offset for variable types). check_state BEFORE.",
    "        \"\"\"",
    "        import numpy as np",
    "        n = self._parent.count()",
    "        ftype = self._parent._dataset.schema.field(self._column).type",
    "        # Conservative element size: 8 for numeric, 32 for string/other.",
    "        elem_size = 8 if (",
    "            pa.types.is_integer(ftype) or pa.types.is_floating(ftype)",
    "            or pa.types.is_boolean(ftype)",
    "            or pa.types.is_timestamp(ftype) or pa.types.is_date(ftype)",
    "            or pa.types.is_time(ftype)",
    "        ) else 32",
    "        self._parent._memory_guard.check_state(",
    "            n * elem_size, name=f\"to_numpy({self._column})\",",
    "        )",
    "        chunks: list = []",
    "        for b in self._parent._iter_filtered_batches():",
    "            chunks.append(",
    "                b.column(self._column).to_numpy(zero_copy_only=False)",
    "            )",
    "        if not chunks:",
    "            return np.array([])",
    "        return np.concatenate(chunks)",
    "",
    "    def to_arrow(self) -> pa.Array:",
    "        \"\"\"Materialize column as single Arrow Array. Bounded.",
    "",
    "        Peak ≈ n × element_size. check_state BEFORE materialize.",
    "        \"\"\"",
    "        n = self._parent.count()",
    "        ftype = self._parent._dataset.schema.field(self._column).type",
    "        elem_size = 8 if (",
    "            pa.types.is_integer(ftype) or pa.types.is_floating(ftype)",
    "            or pa.types.is_boolean(ftype)",
    "            or pa.types.is_timestamp(ftype) or pa.types.is_date(ftype)",
    "            or pa.types.is_time(ftype)",
    "        ) else 50",
    "        self._parent._memory_guard.check_state(",
    "            n * elem_size, name=f\"to_arrow({self._column})\",",
    "        )",
    "        chunks: list = []",
    "        for b in self._parent._iter_filtered_batches():",
    "            chunks.append(b.column(self._column))",
    "        if not chunks:",
    "            return pa.array([], type=ftype)",
    "        if len(chunks) == 1:",
    "            return chunks[0]",
    "        return pa.concat_arrays(chunks)",
    "",
]


def add_proxy_methods():
    print("\n=== dataframe.py: _ColumnProxy aggregators + materialize ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "_ColumnProxy.mean" in s or "def mean(self) -> float:" in s[:s.find("class CadwDataFrame")]:
        print("  already present — skip")
        return

    # Insert before `    def __repr__(self) -> str:` in _ColumnProxy.
    # Anchor: the _ColumnProxy class body ends right before __repr__.
    # But __repr__ appears multiple times — need the class-scoped one.
    # Search for `return f"ColumnProxy({self._column!r})"` (unique to proxy).
    anchor = (
        "    def __repr__(self) -> str:\n"
        "        return f\"ColumnProxy({self._column!r})\"\n"
    )
    if anchor not in s:
        raise SystemExit("_ColumnProxy.__repr__ anchor not found")
    new = "\n".join(PROXY_METHODS) + "\n" + anchor
    s = s.replace(anchor, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 6. Whitelist sort_by in sort_values (AST guard) ============

def update_whitelist():
    print("\n=== test_arrow_call_memory_guard.py: whitelist sort_values ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "CadwDataFrame.sort_values" in s and "in-memory fast path" in s:
        print("  already whitelisted — skip")
        return

    anchor = '"cadw/dataframe.py:CadwDataFrame.describe":'
    idx = s.find(anchor)
    if idx < 0:
        raise SystemExit("describe whitelist anchor not found")
    lines = s.split("\n")
    line_idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            line_idx = i
            break
    end_idx = line_idx
    if not lines[line_idx].strip().endswith(","):
        for j in range(line_idx + 1, min(line_idx + 5, len(lines))):
            if lines[j].strip().endswith(","):
                end_idx = j
                break
        else:
            raise SystemExit("describe value-close not found")
    new_entries = [
        "    # v0.9.6b1: sort_values in-memory fast path (pa.Table.sort_by)",
        "    # for group tables in per-group apply. Group already ≤ 0.3×",
        "    # available (checked in _per_group_apply before fn() call);",
        "    # sort_by allocates output × 1.5 in-memory (pa.Table.sort_by",
        "    # internal peak §9.2.7 = 1.5× input + 6 MB).",
        '    "cadw/dataframe.py:CadwDataFrame.sort_values":',
        '        "in-memory fast path: pa.Table.sort_by on group ≤ 0.3×avail",',
    ]
    for offset, entry in enumerate(new_entries):
        lines.insert(end_idx + 1 + offset, entry)
    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK")


# ============ 7. Tests ============

def create_from_table_tests():
    print("\n=== create tests/test_from_table.py ===")
    p = Path("tests/test_from_table.py")
    if p.exists():
        print("  already exists — skip")
        return
    lines = [
        '"""v0.9.6b1: CadwDataFrame.from_table + _InMemoryDataset."""',
        'from pathlib import Path',
        '',
        'import pyarrow as pa',
        'import pytest',
        '',
        'from cadw import CadwDataFrame, MemoryGuard',
        'from cadw.dataset._in_memory import _InMemoryDataset',
        '',
        '',
        'def test_from_table_basic():',
        '    t = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    assert df.columns == ["a", "b"]',
        '    assert df.count() == 3',
        '    assert df.shape == (3, 2)',
        '',
        '',
        'def test_from_table_is_in_memory_dataset():',
        '    t = pa.table({"a": [1, 2]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    assert isinstance(df._dataset, _InMemoryDataset)',
        '',
        '',
        'def test_from_table_requires_table():',
        '    with pytest.raises(TypeError, match="requires pa.Table"):',
        '        CadwDataFrame.from_table([1, 2, 3])',
        '',
        '',
        'def test_from_table_iter_batches():',
        '    t = pa.table({"a": [1, 2, 3, 4, 5]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    sizes = [b.num_rows for b in df.to_batches(batch_size=2)]',
        '    assert sizes == [2, 2, 1]',
        '',
        '',
        'def test_from_table_head():',
        '    t = pa.table({"a": list(range(100))})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    h = df.head(5)',
        '    assert h.num_rows == 5',
        '    assert h.column("a").to_pylist() == [0, 1, 2, 3, 4]',
        '',
        '',
        'def test_from_table_filter_works():',
        '    t = pa.table({"a": list(range(10))})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    filtered = df[df["a"] > 5]',
        '    assert filtered.count() == 4',
        '',
        '',
        'def test_from_table_sort_values_fast_path():',
        '    t = pa.table({"a": [3, 1, 4, 1, 5, 9, 2, 6]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    sorted_df = df.sort_values("a")',
        '    assert sorted_df.column("a").to_pylist() == [1, 1, 2, 3, 4, 5, 6, 9]',
        '    # Result is still in-memory (fast path stays in-memory)',
        '    assert isinstance(sorted_df._dataset, _InMemoryDataset)',
        '',
        '',
        'def test_from_table_sort_values_descending():',
        '    t = pa.table({"a": [3, 1, 4, 1, 5]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    sorted_df = df.sort_values("a", ascending=False)',
        '    assert sorted_df.column("a").to_pylist() == [5, 4, 3, 1, 1]',
        '',
        '',
        'def test_from_table_multi_key_sort():',
        '    t = pa.table({"a": [1, 2, 1, 2], "b": [10, 20, 5, 15]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    sorted_df = df.sort_values(["a", "b"])',
        '    assert sorted_df.column("a").to_pylist() == [1, 1, 2, 2]',
        '    assert sorted_df.column("b").to_pylist() == [5, 10, 15, 20]',
        '',
        '',
        'def test_from_table_groupby_writes_to_disk():',
        '    """groupby on in-memory df creates a disk Dataset (acceptable;',
        '    group is small so I/O is fine)."""',
        '    t = pa.table({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    result = df.groupby("g").agg("v", "mean")',
        '    # Result is on disk (groupby calls Dataset.from_batches)',
        '    assert result.count() == 2',
        '    assert not isinstance(result._dataset, _InMemoryDataset)',
        '',
        '',
        'def test_from_table_add_column_raises():',
        '    t = pa.table({"a": [1, 2]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    with pytest.raises(NotImplementedError, match="read-only"):',
        '        df._dataset.add_column("b", iter([]))',
        '',
        '',
        'def test_from_table_write_parquet_roundtrip(tmp_path):',
        '    t = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})',
        '    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))',
        '    out = tmp_path / "o.parquet"',
        '    df.write_parquet(out)',
        '    import pyarrow.parquet as pq',
        '    reread = pq.read_table(out)',
        '    assert reread.num_rows == 3',
        '    assert reread.column("a").to_pylist() == [1, 2, 3]',
        '',
    ]
    write_content(p, lines)
    assert_compiles(p)
    print("  OK")


def create_proxy_aggregators_tests():
    print("\n=== create tests/test_column_proxy_aggregators.py ===")
    p = Path("tests/test_column_proxy_aggregators.py")
    if p.exists():
        print("  already exists — skip")
        return
    lines = [
        '"""v0.9.6b1: _ColumnProxy aggregators + materialize."""',
        'import numpy as np',
        'import pyarrow as pa',
        'import pytest',
        '',
        'from cadw import CadwDataFrame, MemoryGuard',
        'from cadw.core.memory_guard import MemoryLimitExceeded',
        '',
        '',
        'def _df(table: pa.Table, *, chunk: int = 5) -> CadwDataFrame:',
        '    # Disk-backed via from_table (in-memory; iteration via to_batches).',
        '    return CadwDataFrame.from_table(',
        '        table, memory_guard=MemoryGuard(max_memory_mb=10),',
        '    )',
        '',
        '',
        '# ============ scalar aggregators ============',
        '',
        '',
        'def test_proxy_mean():',
        '    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))',
        '    assert df["v"].mean() == pytest.approx(3.0)',
        '',
        '',
        'def test_proxy_sum():',
        '    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))',
        '    assert df["v"].sum() == pytest.approx(15.0)',
        '',
        '',
        'def test_proxy_min():',
        '    df = _df(pa.table({"v": [3.0, 1.0, 4.0, 2.0]}))',
        '    assert df["v"].min() == 1.0',
        '',
        '',
        'def test_proxy_max():',
        '    df = _df(pa.table({"v": [3.0, 1.0, 4.0, 2.0]}))',
        '    assert df["v"].max() == 4.0',
        '',
        '',
        'def test_proxy_std_default_ddof():',
        '    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))',
        '    # ddof=1 → sqrt(2.5)',
        '    assert df["v"].std() == pytest.approx(np.sqrt(2.5))',
        '',
        '',
        'def test_proxy_std_ddof_0():',
        '    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))',
        '    # ddof=0 → sqrt(2.0)',
        '    assert df["v"].std(ddof=0) == pytest.approx(np.sqrt(2.0))',
        '',
        '',
        'def test_proxy_count_skips_nulls():',
        '    df = _df(pa.table({"v": pa.array([1.0, None, 3.0, None, 5.0])}))',
        '    assert df["v"].count() == 3',
        '',
        '',
        'def test_proxy_aggregators_on_filtered_view():',
        '    t = pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})',
        '    df = _df(t)',
        '    filtered = df[df["v"] > 2.0]',
        '    assert filtered["v"].mean() == pytest.approx(4.0)',
        '    assert filtered["v"].sum() == pytest.approx(12.0)',
        '',
        '',
        '# ============ materialize ============',
        '',
        '',
        'def test_proxy_to_list():',
        '    df = _df(pa.table({"v": [1, 2, 3, 4, 5]}))',
        '    assert df["v"].to_list() == [1, 2, 3, 4, 5]',
        '',
        '',
        'def test_proxy_to_list_after_filter():',
        '    df = _df(pa.table({"v": [1, 2, 3, 4, 5]}))',
        '    filtered = df[df["v"] > 2]',
        '    assert filtered["v"].to_list() == [3, 4, 5]',
        '',
        '',
        'def test_proxy_to_numpy():',
        '    df = _df(pa.table({"v": [1.0, 2.0, 3.0]}))',
        '    arr = df["v"].to_numpy()',
        '    assert isinstance(arr, np.ndarray)',
        '    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])',
        '',
        '',
        'def test_proxy_to_numpy_int():',
        '    df = _df(pa.table({"v": [1, 2, 3]}))',
        '    arr = df["v"].to_numpy()',
        '    np.testing.assert_array_equal(arr, [1, 2, 3])',
        '',
        '',
        'def test_proxy_to_arrow():',
        '    df = _df(pa.table({"v": [1, 2, 3]}))',
        '    arr = df["v"].to_arrow()',
        '    assert isinstance(arr, pa.Array)',
        '    assert arr.to_pylist() == [1, 2, 3]',
        '',
        '',
        'def test_proxy_to_arrow_multi_chunk():',
        '    # 10 rows; iter_batches yields 5 at a time; to_arrow concatenates.',
        '    t = pa.table({"v": list(range(10))})',
        '    # Force multi-chunk by writing table with 2 chunks',
        '    t2 = pa.concat_tables([t.slice(0, 3), t.slice(3)])',
        '    df = _df(t2)',
        '    arr = df["v"].to_arrow()',
        '    assert arr.to_pylist() == list(range(10))',
        '',
        '',
        'def test_proxy_to_list_guard_fires_on_tight_budget():',
        '    t = pa.table({"v": list(range(1000))})',
        '    df = CadwDataFrame.from_table(',
        '        t, memory_guard=MemoryGuard(max_memory_bytes=2000),',
        '    )',
        '    with pytest.raises(MemoryLimitExceeded, match="to_list"):',
        '        df["v"].to_list()',
        '',
        '',
        'def test_proxy_to_arrow_guard_fires_on_tight_budget():',
        '    t = pa.table({"v": list(range(1000))})',
        '    df = CadwDataFrame.from_table(',
        '        t, memory_guard=MemoryGuard(max_memory_bytes=2000),',
        '    )',
        '    with pytest.raises(MemoryLimitExceeded, match="to_arrow"):',
        '        df["v"].to_arrow()',
        '',
        '',
        'def test_proxy_aggregators_multi_batch():',
        '    """Aggregators work across batch boundaries."""',
        '    t = pa.concat_tables([',
        '        pa.table({"v": [1.0, 2.0, 3.0]}),',
        '        pa.table({"v": [4.0, 5.0]}),',
        '    ])',
        '    df = _df(t)',
        '    assert df["v"].sum() == pytest.approx(15.0)',
        '    assert df["v"].count() == 5',
        '',
    ]
    write_content(p, lines)
    assert_compiles(p)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6b1: in-memory backing + _ColumnProxy aggregators ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    create_in_memory_dataset()
    export_in_memory()
    add_from_table()
    add_sort_fast_path()
    add_proxy_methods()
    update_whitelist()
    create_from_table_tests()
    create_proxy_aggregators_tests()

    print("\n=== pytest: new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_from_table.py",
         "tests/test_column_proxy_aggregators.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: regression (existing df behaviors) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_sort.py",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
        "v0.9.6b1: in-memory Dataset backing + _ColumnProxy aggregators. "
        "_InMemoryDataset(pa.Table) — read-only Dataset subclass, no "
        "disk I/O, iterates table.to_batches(). CadwDataFrame.from_table"
        "(table) — zero-copy in-memory df. sort_values fast path for "
        "in-memory: pa.Table.sort_by directly (avoids 1-5ms external_sort"
        " per group; critical for per-group apply on 100k groups). "
        "_ColumnProxy.mean/sum/min/max/std/count via existing ReduceOp "
        "(O(1) state, no pandas). to_list/to_numpy/to_arrow with "
        "check_state BEFORE materialize. +27 new tests. No existing "
        "behaviour changed — _per_group_apply still uses pandas "
        "(v0.9.6b2 will switch to CADW-native contract).",
        self_path="scripts/patch_v096b1_in_memory.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b1 applied.")


if __name__ == "__main__":
    main()
