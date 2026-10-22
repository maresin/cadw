#!/usr/bin/env python3
"""Read-only scanner for Phase B (dataframe.py split decision).

Prints:
  1. Current state (lines, bytes, HEAD)
  2. Every top-level symbol (class/function/assign) with LOC
  3. Every CadwDataFrame method with LOC (verbose: + call graph)
  4. External imports from cadw.dataframe in cadw/ + tests/ + scripts/
  5. AST guard tests scanning cadw.dataframe (R1 risk)
  6. Proposed 10-module split with LOC estimate + rationale

No files changed. Exit codes:
  0 — analysis complete
  2 — datafile too small / already split (re-run criterion check)

Usage:
    uv run python scripts/service/diagnose_phase_b_split.py
    uv run python scripts/service/diagnose_phase_b_split.py --verbose
"""
from __future__ import annotations
import argparse
import ast
import subprocess
import sys
from pathlib import Path

# Robust root discovery — walk up until pyproject.toml.
# Bug (v0.9.25f-diagnostic-fix1): prior `parent.parent` from
# `scripts/service/file.py` = `scripts/` not project root
# (needs 3 parents). §13.4/256: marker-file discovery
# immune to script depth / moves.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parent
for _ in range(8):
    if (ROOT / "pyproject.toml").exists():
        break
    ROOT = ROOT.parent
else:
    print(
        f"ERROR: no pyproject.toml above {_HERE}",
        file=sys.stderr,
    )
    raise SystemExit(2)
CADW = ROOT / "cadw"
DF_PATH = CADW / "dataframe.py"


def _scan_symbols(src: str) -> dict:
    tree = ast.parse(src)
    out = {"top_level": [], "methods": [], "helpers": []}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            loc = node.end_lineno - node.lineno + 1
            out["helpers"].append(
                (node.name, node.lineno, node.end_lineno, loc)
            )
            out["top_level"].append(
                ("func", node.name, node.lineno, node.end_lineno, loc)
            )
        elif isinstance(node, ast.ClassDef):
            loc = node.end_lineno - node.lineno + 1
            out["top_level"].append(
                ("class", node.name, node.lineno, node.end_lineno, loc)
            )
            if node.name == "CadwDataFrame":
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef,
                                      ast.AsyncFunctionDef)):
                        m_loc = m.end_lineno - m.lineno + 1
                        out["methods"].append(
                            (m.name, m.lineno, m.end_lineno, m_loc)
                        )
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("__"):
                    loc = node.end_lineno - node.lineno + 1
                    out["top_level"].append(
                        ("assign", t.id, node.lineno, node.end_lineno, loc)
                    )
    return out


def _scan_call_graph(src: str, symbols: dict) -> dict:
    """method/function → sorted set of other top-level symbols it calls."""
    tree = ast.parse(src)
    all_toplevel = {s[1] for s in symbols["top_level"]}
    all_methods = {m[0] for m in symbols["methods"]}
    universe = all_toplevel | all_methods

    refs: dict = {}

    def _visit(fn):
        calls: set = set()
        for child in ast.walk(fn):
            if isinstance(child, ast.Name) and child.id in universe:
                calls.add(child.id)
            elif isinstance(child, ast.Attribute) and child.attr in all_methods:
                calls.add(child.attr)
        refs[fn.name] = calls

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CadwDataFrame":
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _visit(m)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _visit(node)
    return refs


def _scan_external_uses() -> dict:
    """who imports what from cadw.dataframe / cadw."""
    out = {"cadw/": set(), "tests/": set(), "scripts/": set()}
    for base in [CADW, ROOT / "tests", ROOT / "scripts"]:
        prefix = base.name + "/"
        for py in base.rglob("*.py"):
            if py == DF_PATH:
                continue
            try:
                tree = ast.parse(py.read_text())
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module == "cadw.dataframe":
                    for a in node.names:
                        out.setdefault(prefix, set()).add(a.name)
                elif node.module == "cadw" and prefix != "cadw/":
                    for a in node.names:
                        if a.name in {"CadwDataFrame", "concat"}:
                            out.setdefault(prefix, set()).add(a.name)
    return out


def _scan_ast_guards() -> list:
    """tests that inspect.getsource(cadw.dataframe) for AST invariants."""
    guards = []
    for py in (ROOT / "tests").rglob("test_*.py"):
        try:
            src = py.read_text()
            tree = ast.parse(src)
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if "inspect.getsource" in seg and (
                "dataframe" in seg or "df_mod" in seg
            ):
                guards.append(
                    (py.relative_to(ROOT).as_posix(), node.name)
                )
    return sorted(set(guards))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not DF_PATH.exists():
        if (CADW / "dataframe").is_dir():
            print("dataframe.py already split (directory exists) — "
                  "Phase B done? Exit 2.")
            return 2
        print(f"ERROR: {DF_PATH} not found", file=sys.stderr)
        return 2

    src = DF_PATH.read_text()
    lines = src.split("\n")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"=== Phase B: dataframe.py split diagnostic ===")
    print(f"  HEAD: {head}")
    print(f"  file: {DF_PATH.relative_to(ROOT)}")
    print(f"  bytes: {DF_PATH.stat().st_size:,}")
    print(f"  lines: {len(lines):,}\n")

    symbols = _scan_symbols(src)
    refs = _scan_call_graph(src, symbols)

    print(f"--- Top-level symbols (sorted by LOC) ---")
    for kind, name, start, end, loc in sorted(
        symbols["top_level"], key=lambda x: -x[4],
    ):
        print(f"  {loc:5d} L  {kind:7s} {name:<40s} ({start}-{end})")

    method_loc = sum(m[3] for m in symbols["methods"])
    print(f"\n--- CadwDataFrame methods: {len(symbols['methods'])} total, "
          f"{method_loc:,} LOC ---")
    if args.verbose:
        for name, start, end, loc in sorted(
            symbols["methods"], key=lambda x: -x[3],
        ):
            called = refs.get(name, set())
            named_methods = sorted(called)  # informational
            called_str = (
                f" -> {', '.join(named_methods[:6])}"
                + (" …" if len(named_methods) > 6 else "")
                if named_methods else ""
            )
            print(f"    {loc:5d} L  {name:<50s} ({start}-{end})"
                  f"{called_str}")

    print(f"\n--- Module-level helpers: {len(symbols['helpers'])} ---")
    for name, start, end, loc in sorted(
        symbols["helpers"], key=lambda x: -x[3],
    ):
        print(f"  {loc:5d} L  {name}  ({start}-{end})")

    uses = _scan_external_uses()
    print(f"\n--- External imports from cadw.dataframe (+ cadw) ---")
    for scope in ("cadw/", "tests/", "scripts/"):
        items = sorted(uses.get(scope, ()))
        if items:
            print(f"  {scope}")
            for name in items:
                print(f"    {name}")

    guards = _scan_ast_guards()
    print(f"\n--- AST-guard tests scanning cadw.dataframe: "
          f"{len(guards)} ---")
    for py, fn in guards:
        print(f"  {py}::{fn}")
    print("  R1 RISK: after split into cadw/dataframe/ package, "
          "inspect.getsource(cadw.dataframe) = __init__.py only. "
          "These tests will silently lose coverage unless re-targeted "
          "to cadw/dataframe/*.py rglob.")

    print("\n=== Proposed split (10 modules) ===\n")
    print("""\
  1. cadw/dataframe/__init__.py     (~60 L)   class composition, re-exports,
                                              backward-compat aliases,
                                              __all__
  2. cadw/dataframe/_core.py        (~250 L)  __init__, properties,
                                              _iter_filtered_batches,
                                              _new_dataset_dir,
                                              _wrap_result_table,
                                              _repr_html_, plot, ts accessors
  3. cadw/dataframe/_lazy.py        (~900 L)  __getitem__, _with_filter,
                                              head, count, to_batches,
                                              _iter_arrow_batches,
                                              iloc/loc/at/iat, set_index,
                                              reset_index, index property,
                                              _iter_filtered_batches_with_positions
  4. cadw/dataframe/_io.py          (~1100 L) write_parquet, astype, dropna,
                                              drop, rename, melt, pivot_table,
                                              crosstab, encode_categorical
  5. cadw/dataframe/_groupby.py     (~700 L)  groupby, _groupby_impl, unique,
                                              drop_duplicates; _GroupBy class
  6. cadw/dataframe/_series.py      (~1200 L) shift, diff, rolling, expanding,
                                              ewm, resample, rank, quantile,
                                              nlargest, nsmallest, sample,
                                              corr, describe,
                                              cumsum/cumprod/cummax/cummin
  7. cadw/dataframe/_ml.py          (~250 L)  all fit_* (12 methods) —
                                              thin wrappers over cadw.ml.*
  8. cadw/dataframe/_per_group.py   (~900 L)  _per_group_apply/shift/diff/
                                              rolling/cumulative, _detect_mode,
                                              _normalize_scalar,
                                              _infer_arrow_type_from_value,
                                              _build_out_schema,
                                              _group_boundaries_arrow
  9. cadw/dataframe/_state.py       (~200 L)  _ArrowTail, _RollState,
                                              _ExpandingState, _EwmState
 10. cadw/dataframe/_utils.py       (~250 L)  concat, _concat_batches,
                                              _convert_batch,
                                              _apply_filter_with_positions,
                                              _empty_value_counts_table,
                                              _reservoir_positions,
                                              _weighted_reservoir_positions,
                                              _ColumnProxy
    """)

    print("=== Risk assessment ===")
    print("  R1 HIGH    AST guards need retargeting (see list above).")
    print("  R2 HIGH    Backward-compat re-exports in __init__.py.")
    print("  R3 MEDIUM  Circular imports (_ColumnProxy/_GroupBy/_detect_mode).")
    print("  R4 MEDIUM  _state.py shared by _series.py + _per_group.py.")
    print("  R5 LOW     __all__ preserved (CadwDataFrame, concat).")
    print("  R6 LOW     MRO cross-deps (_core._repr_html_ -> _lazy.head).")
    print()
    print("=== Expected fix-cycles: 5-8 (structural; analogous to "
          "v0.9.20 4-feature cascade = 5 cycles). ===")
    print("=== Estimated effort: 3-4h (roadmap §11.6 A1). ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())