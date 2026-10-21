#!/usr/bin/env python3
# v0.9.6b2 — _per_group_apply: CADW-native contract, no pandas.
#
# Behaviour change (BREAKING for pd.Series/pd.DataFrame users):
#   fn: CadwDataFrame -> scalar | dict[str, scalar] | CadwDataFrame
#     - scalar    : one row, {group_cols: ..., result_col: scalar}
#     - dict      : one row, {group_cols: ..., **dict}   (was pd.Series)
#     - CadwDataFrame: multi-row, group_cols injected (was pd.DataFrame)
#
# Removed from dataframe.py: `import pandas as pd` in _emit_group,
# _detect_mode, _normalize_scalar; `gt.to_pandas()` in _flush_group.
#
# Added: CadwDataFrame.__len__ (same as count()) — used by
# `lambda g: len(g)` per-group patterns.
#
# Tests updated: test_apply.py (rename+rewrite 2 series/dataframe tests,
# fix 2 .iloc usages, tighten exception match, AST check), plus AST
# guard test_no_pandas_in_per_group_apply. test_apply_e2e.py unchanged
# (already uses g["v"].mean()).
#
# Run: uv run python scripts/patch_v096b2_native_apply.py

from __future__ import annotations
import ast
import os, subprocess, sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    assert_compiles, commit_with_guards,
)


# ============ 1. __len__ for CadwDataFrame ============

def add_len():
    print("\n=== dataframe.py: CadwDataFrame.__len__ ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def __len__(self) -> int:" in s:
        print("  already present — skip")
        return

    anchor = (
        "    def count(self) -> int:\n"
        "        \"\"\"Точное число строк с учётом фильтров (сканирует).\"\"\"\n"
        "        if not self._row_filters:\n"
        "            return self._dataset.total_rows\n"
        "        n = 0\n"
        "        for b in self._iter_filtered_batches():\n"
        "            n += b.num_rows\n"
        "        return n\n"
    )
    new = anchor + (
        "\n"
        "    def __len__(self) -> int:\n"
        "        \"\"\"Row count (v0.9.6b2) — same as count().\n"
        "\n"
        "        Enables `len(df)` in per-group apply patterns\n"
        "        (`lambda g: len(g)`). O(1) unfiltered, O(N) filtered\n"
        "        (same cost as count()).\n"
        "        \"\"\"\n"
        "        return self.count()\n"
    )
    if anchor not in s:
        raise SystemExit("count() anchor not found")
    s = s.replace(anchor, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Replace _per_group_apply method ============

NEW_PER_GROUP_APPLY = '''    def _per_group_apply(
        self,
        group_cols: list[str],
        on_cols: list[str],
        fn,
        result_col: str,
        memory_guard: "MemoryGuard",
    ) -> "CadwDataFrame":
        """Composite-sort apply(fn) on each group. v0.9.6b2 CADW-native.

        Args:
            group_cols: group keys. Non-empty.
            on_cols: within-group sort keys (empty → row_id order).
            fn: callable(group_df: CadwDataFrame) → scalar | dict[str,scalar]
                | CadwDataFrame. Uniform mode across groups.
            result_col: column name for scalar mode.
            memory_guard: per-group memory check before fn().

        Peak RAM = batch + 2 × max(group_size × row_size) + fn's output.
        Bounded per group, independent of N.

        No pandas. For multi-metric per-group output use a dict
        (`{"mean": ..., "max": ...}`); for multi-row output use
        `CadwDataFrame.from_table(pa_table)`.

        fn's own alloc is user responsibility (§3.9 extended).
        """
        import pyarrow as pa
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.core.memory_guard import MemoryLimitExceeded

        if not group_cols:
            raise ValueError(
                "_per_group_apply: group_cols must be non-empty"
            )
        if not callable(fn):
            raise TypeError(
                f"_per_group_apply: fn must be callable; "
                f"got {type(fn).__name__}"
            )
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: on column {oc!r} not in "
                    f"{self._columns}"
                )
        if not isinstance(result_col, str) or not result_col:
            raise ValueError(
                f"_per_group_apply: result_col must be non-empty str; "
                f"got {result_col!r}"
            )

        if self.count() == 0:
            out_fields = [
                self._dataset.schema.field(c) for c in group_cols
            ]
            out_fields.append(pa.field(result_col, pa.float64(), nullable=True))
            out_schema = pa.schema(out_fields)
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_apply" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        out_schema_box: list = []
        out_mode_box: list = []
        buf_rows: list = []
        BATCH_ROWS = 1000
        applied_path = (
            self._spill_dir / "per_group_apply" / "applied.arrow"
        )
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        writer_box: list = []

        def _flush_buf():
            nonlocal buf_rows
            if not buf_rows:
                return
            if not writer_box:
                writer_box.append(
                    ArrowIPCBatchWriter(str(applied_path),
                                        out_schema_box[0])
                )
            table = pa.Table.from_pylist(buf_rows,
                                         schema=out_schema_box[0])
            for b in table.to_batches():
                writer_box[0].write_batch(b)
            buf_rows = []

        def _emit_group(key: tuple, result):
            nonlocal buf_rows

            mode = _detect_mode(result)
            if not out_mode_box:
                out_mode_box.append(mode)
                out_schema_box.append(
                    _build_out_schema(group_cols, self, mode,
                                      result, result_col)
                )
            elif mode != out_mode_box[0]:
                raise RuntimeError(
                    f"_per_group_apply: mixed return modes across "
                    f"groups: first group returned {out_mode_box[0]!r}, "
                    f"group {key!r} returned {mode!r}. fn() must "
                    f"return same type for all groups."
                )

            if mode == "scalar":
                row = {gc: v for gc, v in zip(group_cols, key)}
                row[result_col] = _normalize_scalar(result)
                buf_rows.append(row)
            elif mode == "dict":
                row = {gc: v for gc, v in zip(group_cols, key)}
                for k, v in result.items():
                    row[str(k)] = _normalize_scalar(v)
                buf_rows.append(row)
            else:  # CadwDataFrame
                for b in result._iter_filtered_batches():
                    for rec in b.to_pylist():
                        row = dict(rec)
                        for gc, v in zip(group_cols, key):
                            row.setdefault(gc, v)
                        buf_rows.append(row)
            if len(buf_rows) >= BATCH_ROWS:
                _flush_buf()

        _UNSET = object()
        prev_key = _UNSET
        group_buf: list = []
        max_group_limit = int(
            memory_guard.available_bytes() * 0.3
        )

        def _flush_group():
            nonlocal group_buf
            if not group_buf:
                return
            est_pre = sum(b.nbytes for b in group_buf) * 2
            if est_pre > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: group size ~{est_pre // 2} B "
                    f"× 2 (materialize + overhead) = {est_pre} B > "
                    f"0.3 × avail = {max_group_limit} B. Split by "
                    f"more group cols or use groupby().agg() if fn "
                    f"reduces. See §11.5."
                )
            memory_guard.check_state(
                est_pre, name="per_group_apply.group",
            )
            gt = pa.Table.from_batches(group_buf)
            est_actual = gt.nbytes * 2
            if est_actual > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: actual group size "
                    f"{gt.nbytes} B × 2 = {est_actual} B > 0.3 × avail "
                    f"= {max_group_limit} B (pre-estimate was "
                    f"{est_pre} B; buffer overhead). See §11.5."
                )
            # v0.9.6b2: CADW-native group. No `to_pandas()`.
            gdf = CadwDataFrame.from_table(
                gt,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )
            result = fn(gdf)
            _emit_group(tuple(prev_key), result)
            group_buf.clear()

        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                memory_guard.check_batch(
                    b, name="per_group_apply.sorted"
                )
                n = b.num_rows
                if n == 0:
                    continue
                group_vals = {
                    gc: b.column(gc).to_pylist() for gc in group_cols
                }
                start = 0
                for i in range(1, n):
                    gk = tuple(group_vals[gc][i] for gc in group_cols)
                    prev_gk = tuple(
                        group_vals[gc][i - 1] for gc in group_cols
                    )
                    if gk != prev_gk:
                        gk_start = tuple(
                            group_vals[gc][start] for gc in group_cols
                        )
                        piece = b.slice(start, i - start)
                        if gk_start != prev_key:
                            if prev_key is not _UNSET:
                                _flush_group()
                            prev_key = gk_start
                        group_buf.append(piece)
                        start = i
                gk_start = tuple(
                    group_vals[gc][start] for gc in group_cols
                )
                piece = b.slice(start, n - start)
                if gk_start != prev_key:
                    if prev_key is not _UNSET:
                        _flush_group()
                    prev_key = gk_start
                group_buf.append(piece)

            _flush_group()
            _flush_buf()
        finally:
            if writer_box:
                writer_box[0].close()

        if not out_schema_box:
            raise RuntimeError(
                "_per_group_apply: no groups produced output — "
                "unreachable after empty check"
            )

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                yield b

        new_ds = Dataset.from_batches(
            out_schema_box[0], _iter_applied(),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )'''

NEW_HELPERS_BLOCK = '''# ============ per-group apply helpers (v0.9.6b2, CADW-native) ============

def _detect_mode(result) -> str:
    """Classify fn() output: 'scalar' / 'dict' / 'dataframe'.

    v0.9.6b2: CADW-native contract. fn returns one of:
      - scalar (int/float/str/bool/None, or numpy scalar): single-value row
      - dict[str, scalar]: multi-value single row (replaces pd.Series)
      - CadwDataFrame: multi-row output (replaces pd.DataFrame)

    No pandas. For pandas-like multi-row output, wrap in
    CadwDataFrame.from_table(pa_table).
    """
    from cadw.dataframe import CadwDataFrame
    import numpy as np

    if isinstance(result, CadwDataFrame):
        return "dataframe"
    if isinstance(result, dict):
        return "dict"
    if result is None:
        return "scalar"
    if isinstance(result, (bool, int, float, str)):
        return "scalar"
    if isinstance(result, (np.integer, np.floating, np.bool_)):
        return "scalar"
    raise TypeError(
        f"_per_group_apply: fn returned {type(result).__name__}; "
        f"expected scalar, dict[str, scalar], or CadwDataFrame "
        f"(v0.9.6b2 CADW-native contract). For multi-metric output "
        f"use a dict; for multi-row output use CadwDataFrame.from_table("
        f"pa_table)."
    )


def _normalize_scalar(v):
    """Convert numpy scalar / NaN to Python scalar / None.

    v0.9.6b2: no pandas. NaN detection via float comparison (`v != v`),
    not pd.isna. pd.NA / pd.NaT not supported (fn should return None).
    """
    import numpy as np

    if v is None:
        return None
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        v = v.item()
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


def _infer_arrow_type_from_value(v):
    import pyarrow as pa
    import numpy as np

    if v is None:
        return pa.float64()
    if isinstance(v, (np.integer, int)):
        return pa.int64()
    if isinstance(v, (np.floating, float)):
        return pa.float64()
    if isinstance(v, (np.bool_, bool)):
        return pa.bool_()
    if isinstance(v, str):
        return pa.string()
    return pa.float64()


def _build_out_schema(group_cols, df, mode, first_result, result_col):
    """Output pa.Schema from group_cols + first group's result shape.

    v0.9.6b2: CADW-native. Modes:
      - 'scalar': group_cols + result_col (inferred type)
      - 'dict':   group_cols + first_result keys (inferred types)
      - 'dataframe': group_cols + first_result.columns (skipping
        group_cols if already present in df.columns)
    """
    import pyarrow as pa

    fields = [df._dataset.schema.field(c) for c in group_cols]
    if mode == "scalar":
        v = _normalize_scalar(first_result)
        fields.append(pa.field(
            result_col, _infer_arrow_type_from_value(v), nullable=True,
        ))
    elif mode == "dict":
        for k in first_result:
            v = _normalize_scalar(first_result[k])
            fields.append(pa.field(
                str(k), _infer_arrow_type_from_value(v), nullable=True,
            ))
    else:  # CadwDataFrame
        for c in first_result.columns:
            if c in group_cols:
                continue
            fields.append(first_result._dataset.schema.field(c))
    return pa.schema(fields)'''


def _replace_class_method(src: str, class_name: str, method_name: str,
                          new_block: str) -> str:
    tree = ast.parse(src)
    cadw_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cadw_class = node
            break
    if cadw_class is None:
        raise SystemExit(f"class {class_name} not found")
    start = end = None
    for node in cadw_class.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == method_name):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit(f"{class_name}.{method_name} not found")
    lines = src.split("\n")
    return "\n".join(lines[:start - 1] + [new_block] + lines[end:])


def _replace_module_functions(src: str, func_names: list[str],
                              new_block: str) -> str:
    tree = ast.parse(src)
    spans = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            # Backtrack to eat preceding blanks/comments for clean span.
            spans.append((node.lineno, node.end_lineno))
    if not spans:
        raise SystemExit(f"none of {func_names} found")
    start = min(s[0] for s in spans)
    end = max(s[1] for s in spans)
    lines = src.split("\n")
    return "\n".join(lines[:start - 1] + [new_block] + lines[end:])


def replace_method_and_helpers():
    print("\n=== dataframe.py: replace _per_group_apply + helpers ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "v0.9.6b2" in s and "CADW-native" in s:
        print("  already v0.9.6b2 — skip")
        return

    print("  replacing CadwDataFrame._per_group_apply (AST span)")
    s = _replace_class_method(
        s, "CadwDataFrame", "_per_group_apply", NEW_PER_GROUP_APPLY,
    )

    print("  replacing _detect_mode.._build_out_schema (AST span)")
    s = _replace_module_functions(
        s,
        ["_detect_mode", "_normalize_scalar",
         "_infer_arrow_type_from_value", "_build_out_schema"],
        NEW_HELPERS_BLOCK,
    )

    p.write_text(s)
    assert_compiles(p)
    print("  py_compile OK")

    print("  running import + no-pandas smoke test...")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import inspect, cadw.dataframe as df_mod; "
         "src = inspect.getsource(df_mod); "
         "assert 'import pandas' not in src, 'pandas import in dataframe.py'; "
         "assert 'to_pandas' not in inspect.getsource(df_mod.CadwDataFrame._per_group_apply), '_per_group_apply calls to_pandas'; "
         "assert hasattr(df_mod.CadwDataFrame, '__len__')"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print("  STDERR: " + r.stderr.strip()[-800:])
        raise SystemExit("smoke test failed")
    print("  smoke OK")


# ============ 3. Update _GroupBy.apply docstring ============

def update_groupby_apply_docstring():
    print("\n=== dataframe.py: _GroupBy.apply docstring ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "CADW-native contract" in s and "_GroupBy" in s:
        # very rough check
        if "fn: callable(group_df: CadwDataFrame)" in s:
            print("  already updated — skip")
            return

    old = (
        '        """Per-group apply (v0.9.1e). Prototype-validated (11 cases).\n'
        '\n'
        '        Args:\n'
        '            fn: callable(group_df: pd.DataFrame) → scalar |\n'
        '                pd.Series | pd.DataFrame. Uniform mode across\n'
        '                groups.\n'
    )
    new = (
        '        """Per-group apply (v0.9.6b2: CADW-native, no pandas).\n'
        '\n'
        '        Args:\n'
        '            fn: callable(group_df: CadwDataFrame) → scalar |\n'
        '                dict[str, scalar] | CadwDataFrame. Uniform\n'
        '                mode across groups. For pd.Series-like output\n'
        '                use a dict; for pd.DataFrame-like output use\n'
        '                CadwDataFrame.from_table(pa_table).\n'
    )
    if old not in s:
        print("  anchor not found — skip (non-fatal)")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. Update tests ============

def update_test_apply():
    print("\n=== test_apply.py: update for CADW-native contract ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()

    # 4a. Series → dict
    old_series = (
        'def test_apply_series_multi_output(tmp_path):\n'
        '    df = make_df(tmp_path, {\n'
        '        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],\n'
        '    })\n'
        '    out = df.groupby("g").apply(\n'
        '        lambda g: pd.Series({"mean": g["v"].mean(),\n'
        '                             "max": g["v"].max()}),\n'
        '    )\n'
        '    t = read_table(out, tmp_path)\n'
        '    assert set(t.schema.names) == {"g", "mean", "max"}\n'
        '    rows = {r["g"]: (r["mean"], r["max"]) for r in t.to_pylist()}\n'
        '    assert rows == {"a": (15.0, 20), "b": (150.0, 200)}\n'
    )
    new_dict = (
        'def test_apply_dict_multi_output(tmp_path):\n'
        '    """v0.9.6b2: dict[str,scalar] replaces pd.Series."""\n'
        '    df = make_df(tmp_path, {\n'
        '        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],\n'
        '    })\n'
        '    out = df.groupby("g").apply(\n'
        '        lambda g: {"mean": g["v"].mean(), "max": g["v"].max()},\n'
        '    )\n'
        '    t = read_table(out, tmp_path)\n'
        '    assert set(t.schema.names) == {"g", "mean", "max"}\n'
        '    rows = {r["g"]: (r["mean"], r["max"]) for r in t.to_pylist()}\n'
        '    assert rows == {"a": (15.0, 20), "b": (150.0, 200)}\n'
    )
    if old_series in s:
        s = s.replace(old_series, new_dict, 1)
        print("  test_apply_series_multi_output → test_apply_dict_multi_output")
    else:
        print("  WARNING: series test anchor not found")

    # 4b. DataFrame → CadwDataFrame
    old_df = (
        'def test_apply_dataframe_expand(tmp_path):\n'
        '    df = make_df(tmp_path, {\n'
        '        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 10, 20],\n'
        '    })\n'
        '    def fn(g):\n'
        '        return pd.DataFrame({\n'
        '            "v_rank": np.arange(1, len(g) + 1),\n'
        '            "v_shift": g["v"].tolist(),\n'
        '        })\n'
        '    out = df.groupby("g").apply(fn)\n'
        '    t = read_table(out, tmp_path)\n'
        '    assert set(t.schema.names) == {"g", "v_rank", "v_shift"}\n'
        '    assert t.num_rows == 5\n'
    )
    new_cadwdf = (
        'def test_apply_cadwdf_expand(tmp_path):\n'
        '    """v0.9.6b2: CadwDataFrame replaces pd.DataFrame."""\n'
        '    import pyarrow as pa\n'
        '    df = make_df(tmp_path, {\n'
        '        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 10, 20],\n'
        '    })\n'
        '    def fn(g):\n'
        '        n = len(g)\n'
        '        return CadwDataFrame.from_table(\n'
        '            pa.table({\n'
        '                "v_rank": np.arange(1, n + 1),\n'
        '                "v_shift": g["v"].to_list(),\n'
        '            }),\n'
        '            memory_guard=g._memory_guard,\n'
        '            spill_dir=g._spill_dir,\n'
        '        )\n'
        '    out = df.groupby("g").apply(fn)\n'
        '    t = read_table(out, tmp_path)\n'
        '    assert set(t.schema.names) == {"g", "v_rank", "v_shift"}\n'
        '    assert t.num_rows == 5\n'
    )
    if old_df in s:
        s = s.replace(old_df, new_cadwdf, 1)
        print("  test_apply_dataframe_expand → test_apply_cadwdf_expand")
    else:
        print("  WARNING: dataframe test anchor not found")

    # 4c. .iloc → to_list()[0] (2 spots)
    iloc_old_1 = (
        '    out = df.groupby("g").apply(lambda g: g["v"].iloc[0])\n'
        '    t = read_table(out, tmp_path)\n'
        '    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}\n'
        '    assert rows == {"a": 10, "b": 20, "c": 30}\n'
    )
    iloc_new_1 = (
        '    out = df.groupby("g").apply(lambda g: g["v"].to_list()[0])\n'
        '    t = read_table(out, tmp_path)\n'
        '    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}\n'
        '    assert rows == {"a": 10, "b": 20, "c": 30}\n'
    )
    if iloc_old_1 in s:
        s = s.replace(iloc_old_1, iloc_new_1, 1)
        print("  single_row: .iloc[0] → .to_list()[0]")

    iloc_old_2 = (
        '    out = df.groupby("g").apply(\n'
        '        lambda g: g["v"].iloc[0], on="ts",\n'
        '    )\n'
    )
    iloc_new_2 = (
        '    out = df.groupby("g").apply(\n'
        '        lambda g: g["v"].to_list()[0], on="ts",\n'
        '    )\n'
    )
    if iloc_old_2 in s:
        s = s.replace(iloc_old_2, iloc_new_2, 1)
        print("  on_col: .iloc[0] → .to_list()[0]")

    # 4d. Mixed mode: pd.Series → dict
    mixed_old = (
        '        if call_count[0] == 1:\n'
        '            return g["v"].sum()  # scalar\n'
        '        return pd.Series({"x": 1})  # series\n'
    )
    mixed_new = (
        '        if call_count[0] == 1:\n'
        '            return g["v"].sum()  # scalar\n'
        '        return {"x": 1}  # dict (was pd.Series)\n'
    )
    if mixed_old in s:
        s = s.replace(mixed_old, mixed_new, 1)
        print("  mixed_modes: pd.Series → dict")

    # 4e. Exception match tightening
    old_match = 'with pytest.raises(TypeError, match="expected scalar"):'
    new_match = 'with pytest.raises(TypeError, match="expected scalar, dict"):'
    if old_match in s:
        s = s.replace(old_match, new_match, 1)
        print("  unsupported_type: tighten exception match")

    # 4f. AST test update
    old_ast = (
        'def test_apply_ast_uses_composite_sort():\n'
        '    """AST: _per_group_apply external_sort by group+on; check before to_pandas."""\n'
        '    import inspect\n'
        '    from cadw.dataframe import CadwDataFrame\n'
        '    src = inspect.getsource(CadwDataFrame._per_group_apply)\n'
        '    assert "external_sort" in src\n'
        '    assert "to_pandas" in src\n'
        '    assert "check_state" in src\n'
        '    assert src.index("check_state") < src.index("to_pandas")\n'
    )
    new_ast = (
        'def test_apply_ast_uses_composite_sort():\n'
        '    """AST: _per_group_apply external_sort by group+on; check before group materialize."""\n'
        '    import inspect\n'
        '    from cadw.dataframe import CadwDataFrame\n'
        '    src = inspect.getsource(CadwDataFrame._per_group_apply)\n'
        '    assert "external_sort" in src\n'
        '    assert "to_pandas" not in src, (\n'
        '        "v0.9.6b2: _per_group_apply must not use pandas"\n'
        '    )\n'
        '    assert "CadwDataFrame.from_table" in src, (\n'
        '        "v0.9.6b2: group must be materialized as CadwDataFrame"\n'
        '    )\n'
        '    assert "check_state" in src\n'
        '    assert src.index("check_state") < src.index("from_table"), (\n'
        '        "check_state must precede group materialization"\n'
        '    )\n'
    )
    if old_ast in s:
        s = s.replace(old_ast, new_ast, 1)
        print("  ast_test: update for CADW-native")

    # 4g. New test: CadwDataFrame with embedded group col
    if "def test_apply_cadwdf_group_col_embedded" not in s:
        new_test = (
            "\n\n"
            "def test_apply_cadwdf_group_col_embedded(tmp_path):\n"
            '    """User returns CadwDataFrame with group col already\n'
            '    present — output has group col exactly once."""\n'
            "    import pyarrow as pa\n"
            '    df = make_df(tmp_path, {"g": ["a", "a", "b"], "v": [1, 2, 3]})\n'
            "    def fn(g):\n"
            '        vals = g["v"].to_list()\n'
            '        gval = g["g"].to_list()[0]\n'
            "        return CadwDataFrame.from_table(\n"
            '            pa.table({"g": [gval] * len(vals), "val": vals}),\n'
            "            memory_guard=g._memory_guard,\n"
            "            spill_dir=g._spill_dir,\n"
            "        )\n"
            '    out = df.groupby("g").apply(fn)\n'
            "    t = read_table(out, tmp_path)\n"
            '    assert t.schema.names.count("g") == 1\n'
            '    assert "val" in t.schema.names\n'
            "    assert t.num_rows == 3\n"
        )
        s = s + new_test
        print("  + test_apply_cadwdf_group_col_embedded")

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def update_no_pandas_in_ops():
    print("\n=== test_no_pandas_in_ops.py: tighten dataframe facade + per_group_apply guard ===")
    p = Path("tests/test_no_pandas_in_ops.py")
    s = p.read_text()

    old_facade = (
        'def test_no_pandas_in_dataframe_facade():\n'
        '    """`cadw/dataframe.py` — no MODULE-LEVEL pandas import.\n'
        '\n'
        '    v0.9.1e-fix2 (§13.4/155): lazy `import pandas as pd` inside\n'
        '    `_emit_group` (used by `_per_group_apply` to construct pd.Series/\n'
        '    pd.DataFrame from user fn output) is legitimate — keeps `import\n'
        '    cadw` light. Only top-level import is forbidden.\n'
        '    """\n'
        '    py = CADW_OPS.parent / "dataframe.py"\n'
        '    src = py.read_text()\n'
        '    tree = ast.parse(src)\n'
        '    assert not _uses_pandas_import_at_module_level(tree), (\n'
        '        "cadw/dataframe.py must not import pandas at module level "\n'
        '        "(lazy import inside functions OK — see §13.4/155)"\n'
        '    )\n'
    )
    new_facade = (
        'def test_no_pandas_in_dataframe_facade():\n'
        '    """`cadw/dataframe.py` — no pandas import anywhere (v0.9.6b2).\n'
        '\n'
        '    v0.9.6b2: _per_group_apply is CADW-native (fn returns\n'
        '    scalar | dict | CadwDataFrame). No `import pandas` at\n'
        '    module level or inside functions. Remaining pandas touch\n'
        '    in dataframe.py is `Table.to_pandas()` in pivot_table —\n'
        '    that is a METHOD, not an import; addressed separately\n'
        '    (v0.9.6c PivotOp).\n'
        '    """\n'
        '    py = CADW_OPS.parent / "dataframe.py"\n'
        '    src = py.read_text()\n'
        '    tree = ast.parse(src)\n'
        '    assert not _uses_pandas_import(tree), (\n'
        '        "cadw/dataframe.py must not import pandas anywhere "\n'
        '        "(v0.9.6b2: _per_group_apply is CADW-native). "\n'
        '        "If a new lazy import is genuinely required, add a "\n'
        '        "whitelist rationale — do not weaken this test."\n'
        '    )\n'
    )
    if old_facade in s:
        s = s.replace(old_facade, new_facade, 1)
        print("  tightened test_no_pandas_in_dataframe_facade")
    else:
        print("  WARNING: facade test anchor not found")

    if "def test_no_pandas_in_per_group_apply" not in s:
        new_test = (
            "\n\n"
            "def test_no_pandas_in_per_group_apply():\n"
            '    """_per_group_apply + helpers must not import pandas\n'
            '    (v0.9.6b2)."""\n'
            "    import ast as _ast\n"
            "    import inspect as _inspect\n"
            "    import textwrap as _textwrap\n"
            "    from cadw import dataframe as df_mod\n"
            "\n"
            "    targets = [\n"
            '        "_per_group_apply", "_detect_mode", "_normalize_scalar",\n'
            '        "_build_out_schema", "_infer_arrow_type_from_value",\n'
            "    ]\n"
            "    for name in targets:\n"
            "        fn = getattr(df_mod, name, None)\n"
            "        if fn is None:\n"
            "            continue\n"
            "        src = _inspect.getsource(fn)\n"
            "        tree = _ast.parse(_textwrap.dedent(src))\n"
            "        for node in _ast.walk(tree):\n"
            "            if isinstance(node, _ast.Import):\n"
            "                for alias in node.names:\n"
            "                    assert not alias.name.startswith('pandas'), (\n"
            "                        f'{name} imports pandas; must be '\n"
            "                        'CADW-native'\n"
            "                    )\n"
            "            elif isinstance(node, _ast.ImportFrom):\n"
            "                if node.module and node.module.startswith('pandas'):\n"
            "                    raise AssertionError(\n"
            "                        f'{name} imports from pandas; must be '\n"
            "                        'CADW-native'\n"
            "                    )\n"
        )
        s = s + new_test
        print("  + test_no_pandas_in_per_group_apply")

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6b2: CADW-native _per_group_apply ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_len()
    replace_method_and_helpers()
    update_groupby_apply_docstring()
    update_test_apply()
    update_no_pandas_in_ops()

    print("\n=== pytest: apply tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** APPLY TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: no-pandas guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** AST GUARDS FAILED ***")
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
        "v0.9.6b2: CADW-native _per_group_apply — no pandas. "
        "fn contract: CadwDataFrame -> scalar | dict[str,scalar] | "
        "CadwDataFrame (replaces pd.Series/pd.DataFrame). "
        "Removed `import pandas as pd` from _emit_group/_detect_mode/"
        "_normalize_scalar and `gt.to_pandas()` from _flush_group. "
        "Added CadwDataFrame.__len__ (same as count()) for "
        "`lambda g: len(g)` patterns. Updated test_apply.py "
        "(series→dict, dataframe→CadwDataFrame, .iloc→.to_list()[0]); "
        "tightened AST test. Added test_no_pandas_in_per_group_apply "
        "guard. Tightened test_no_pandas_in_dataframe_facade to "
        "no-import-anywhere.",
        self_path="scripts/patch_v096b2_native_apply.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b2 applied.")


if __name__ == "__main__":
    main()