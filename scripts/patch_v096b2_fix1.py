#!/usr/bin/env python3
# v0.9.6b2-fix1 — replace_method_and_helpers false-positive guard.
#
# Root cause: idempotency guard `if "v0.9.6b2" in s and "CADW-native"
# in s` matched because:
#   (a) add_len() (already applied) added "v0.9.6b2" to __len__ docstring;
#   (b) "CADW-native" already present in _ColumnProxy.mean docstring
#       (from v0.9.6b1 PROXY_METHODS: "the CADW-native replacement for").
# Guard fired → pandas-based _per_group_apply and _detect_mode/
# _normalize_scalar/_build_out_schema kept. 6 apply tests failed
# (dict/mixed/unsupported type + CadwDataFrame expansion + AST).
#
# Fix: idempotency guard uses inspect.getsource(target) per-function,
# not whole-file substring scan (§13.4/182). Then replace:
#   - CadwDataFrame._per_group_apply (AST span in class)
#   - module-level _detect_mode.._build_out_schema (AST span)
#
# add_len() already applied in v0.9.6b2 (skip if present).
# test_apply.py / test_no_pandas_in_ops.py already updated in
# v0.9.6b2 — no test edits here.
#
# Run: uv run python scripts/patch_v096b2_fix1.py

from __future__ import annotations
import ast
import inspect
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


# ============ new method source (same as v0.9.6b2) ============

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


# ============ AST span replace helpers ============

def _replace_class_method(src: str, class_name: str, method_name: str,
                          new_block: str) -> str:
    tree = ast.parse(src)
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cls = node
            break
    if cls is None:
        raise SystemExit(f"class {class_name} not found")
    start = end = None
    for node in cls.body:
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
            spans.append((node.lineno, node.end_lineno))
    if not spans:
        raise SystemExit(f"none of {func_names} found")
    start = min(s[0] for s in spans)
    end = max(s[1] for s in spans)
    lines = src.split("\n")
    return "\n".join(lines[:start - 1] + [new_block] + lines[end:])


# ============ main patch ============

def replace_per_group_apply():
    print("\n=== dataframe.py: replace _per_group_apply (scoped guard) ===")
    from cadw.dataframe import CadwDataFrame
    current_src = inspect.getsource(CadwDataFrame._per_group_apply)

    # Scoped idempotency: check ONLY the method, not the whole file.
    if "CadwDataFrame.from_table" in current_src and "to_pandas" not in current_src:
        print("  already CADW-native — skip")
        return

    print("  current _per_group_apply is pandas-based → replacing")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    s = _replace_class_method(
        s, "CadwDataFrame", "_per_group_apply", NEW_PER_GROUP_APPLY,
    )
    p.write_text(s)
    assert_compiles(p)
    print("  py_compile OK")


def replace_helpers():
    print("\n=== dataframe.py: replace _detect_mode.._build_out_schema ===")
    import cadw.dataframe as df_mod
    # Scoped: check _detect_mode source for pandas import.
    try:
        dm_src = inspect.getsource(df_mod._detect_mode)
    except AttributeError:
        print("  _detect_mode not found — abort")
        raise

    if "import pandas" not in dm_src and "pd.Series" not in dm_src:
        print("  already CADW-native — skip")
        return

    print("  helpers are pandas-based → replacing")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    s = _replace_module_functions(
        s,
        ["_detect_mode", "_normalize_scalar",
         "_infer_arrow_type_from_value", "_build_out_schema"],
        NEW_HELPERS_BLOCK,
    )
    p.write_text(s)
    assert_compiles(p)
    print("  py_compile OK")


def smoke_test():
    print("\n=== smoke test: no pandas in dataframe.py execution path ===")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import inspect, cadw.dataframe as df_mod; "
         "src = inspect.getsource(df_mod); "
         "assert 'import pandas' not in src, 'pandas import remains'; "
         "m = inspect.getsource(df_mod.CadwDataFrame._per_group_apply); "
         "assert 'to_pandas' not in m, '_per_group_apply calls to_pandas'; "
         "assert 'CadwDataFrame.from_table' in m, 'from_table missing'; "
         "dm = inspect.getsource(df_mod._detect_mode); "
         "assert 'import pandas' not in dm, '_detect_mode imports pandas'; "
         "assert 'dict' in dm, '_detect_mode does not recognise dict'; "
         "assert hasattr(df_mod.CadwDataFrame, '__len__')"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print("  STDERR: " + r.stderr.strip()[-1200:])
        raise SystemExit("smoke test failed")
    print("  OK")


def main():
    print("=== v0.9.6b2-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    replace_per_group_apply()
    replace_helpers()
    smoke_test()

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

    print("\n=== pytest: from_table/proxy regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_from_table.py",
         "tests/test_column_proxy_aggregators.py", "-q"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** FROM_TABLE/PROXY REGRESSION FAILED ***")
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
        "v0.9.6b2-fix1: actually replace _per_group_apply + helpers "
        "(v0.9.6b2 guard was false-positive: 'v0.9.6b2' present in "
        "__len__ docstring from add_len(), 'CADW-native' present in "
        "_ColumnProxy.mean docstring from v0.9.6b1). Scoped "
        "idempotency guard uses inspect.getsource(target), not "
        "whole-file substring scan (§13.4/182). _per_group_apply now "
        "CADW-native: fn gets CadwDataFrame, returns scalar | dict | "
        "CadwDataFrame. _detect_mode/_normalize_scalar/_build_out_schema "
        "no longer import pandas. Smoke test asserts no 'import pandas' "
        "and no 'to_pandas' in _per_group_apply.",
        self_path="scripts/patch_v096b2_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b2 complete (fix1).")


if __name__ == "__main__":
    main()