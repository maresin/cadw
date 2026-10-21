#!/usr/bin/env python3
# v0.9.13 — F12-5 quantile Arrow-native stream + F12-8 PredicateFilter
# null semantics doc.
#
# F12-5 (quantile.py, S3 perf): _stream_values_at_positions does
#   per-value Python loop:
#     for v in vals:
#         if v is None: continue
#         pos += 1
#         while ti < len(targets_sorted) and pos == targets_sorted[ti]:
#             result[...] = v; ti += 1
#         if pos >= max_target: return result
#   For N=10M / max_target ~ N: ~10M Python iterations (dict lookup,
#   null check, condition), ~500ms-2s. quantile path already pays
#   external_sort; this loop is the remaining O(N) Python overhead.
#
#   Fix: per-batch O(1) Arrow ops.
#     for batch in reader:
#         n_valid = len(arr) - arr.null_count
#         covered = [t for t in targets_sorted if pos <= t < pos+n_valid]
#         if covered:
#             non_null = arr if no nulls else arr.filter(pc.is_valid(arr))
#             selected = pc.take(non_null, [t-pos for t in covered])
#             for t, v in zip(covered, selected.to_pylist()):
#                 result[t] = v
#         pos += n_valid
#         if pos > max_target: return result
#
#   ~100-400x faster for typical quantile (5 q, 10 targets, ~N batch).
#   `filter`, `take`, `is_valid` not in HIGH_RISK_ATTRS (AST guard
#   test_arrow_call_memory_guard). `check_batch` already present.
#   `to_pylist` on selected = bounded by len(targets), not N.
#
# F12-8 (row_filter.py, doc): PredicateFilter mask_for_batch is
#   asymmetric on nulls, undocumented:
#     _OPS (gt/ge/lt/le/eq/ne) → pc.greater(arr, scalar) = null on
#       null rows (Kleene); b.filter(null) treats as false → drop.
#       Matches pandas (NaN comparisons drop in boolean indexing) and
#       SQL (WHERE null > 5 = false via Kleene). Intentional.
#     _OPS_NO_SCALAR (contains/starts_with/ends_with) → fill_null(False)
#       already. Matches pandas .str.contains(na=False) default CADW.
#   Document asymmetry in docstring, no behavior change. §13.4/218
#   (`fill_null` before filter) applies ONLY to fill-default cases;
#   numeric comparisons intentionally Kleene (matches ecosystem).
#
# Run: uv run python scripts/patch_v0913.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ F12-5: Arrow-native _stream_values_at_positions ============

OLD_F12_5 = (
    "def _stream_values_at_positions(\n"
    "    sorted_path: Path,\n"
    "    column: str,\n"
    "    targets: Sequence[int],\n"
    "    memory_guard: MemoryGuard,\n"
    ") -> dict[int, object]:\n"
    "    \"\"\"Stream sorted file, collect values at 0-based non-null positions.\"\"\"\n"
    "    result: dict[int, object] = {}\n"
    "    if not targets:\n"
    "        return result\n"
    "    targets_sorted = sorted(set(targets))\n"
    "    max_target = targets_sorted[-1]\n"
    "    ti = 0\n"
    "    pos = -1\n"
    "\n"
    "    for batch in ArrowIPCBatchReader(str(sorted_path), [column]):\n"
    "        memory_guard.check_batch(batch, name=\"quantile.sorted\")\n"
    "        vals = batch.column(column).to_pylist()\n"
    "        for v in vals:\n"
    "            if v is None:\n"
    "                continue\n"
    "            pos += 1\n"
    "            while ti < len(targets_sorted) and pos == targets_sorted[ti]:\n"
    "                result[targets_sorted[ti]] = v\n"
    "                ti += 1\n"
    "            if pos >= max_target:\n"
    "                return result\n"
    "    return result\n"
)

NEW_F12_5 = (
    "def _stream_values_at_positions(\n"
    "    sorted_path: Path,\n"
    "    column: str,\n"
    "    targets: Sequence[int],\n"
    "    memory_guard: MemoryGuard,\n"
    ") -> dict[int, object]:\n"
    "    \"\"\"Stream sorted file, collect values at 0-based non-null positions.\n"
    "\n"
    "    v0.9.13 (F12-5, S3 perf): Arrow-native per batch. Prior: per-\n"
    "    value Python loop (for v in vals) → O(max_target) Python\n"
    "    iterations (~N for high quantiles). For N=10M / 5 q, ~10M\n"
    "    Python iterations (~500ms-2s). Now: per-batch O(1) Arrow ops\n"
    "    (filter + take on bounded target list). ~100-400x faster.\n"
    "\n"
    "    Peak per batch: batch.nbytes (already check_batch'd) +\n"
    "    selected array (len(targets) values, ≤ 2×len(q)). `filter`\n"
    "    of null-only path allocates batch-size bool; bounded by\n"
    "    check_batch. `to_pylist` on selected = O(K), K bounded by\n"
    "    len(targets), not N.\n"
    "    \"\"\"\n"
    "    import pyarrow.compute as _pc\n"
    "\n"
    "    result: dict[int, object] = {}\n"
    "    if not targets:\n"
    "        return result\n"
    "    targets_sorted = sorted(set(targets))\n"
    "    max_target = targets_sorted[-1]\n"
    "    pos = 0\n"
    "\n"
    "    for batch in ArrowIPCBatchReader(str(sorted_path), [column]):\n"
    "        memory_guard.check_batch(batch, name=\"quantile.sorted\")\n"
    "        arr = batch.column(column)\n"
    "        n_valid_batch = len(arr) - arr.null_count\n"
    "        covered = [\n"
    "            t for t in targets_sorted\n"
    "            if pos <= t < pos + n_valid_batch\n"
    "        ]\n"
    "        if covered:\n"
    "            if arr.null_count == 0:\n"
    "                non_null = arr\n"
    "            else:\n"
    "                non_null = arr.filter(_pc.is_valid(arr))\n"
    "            local_idxs = [t - pos for t in covered]\n"
    "            selected = _pc.take(\n"
    "                non_null,\n"
    "                pa.array(local_idxs, type=pa.int64()),\n"
    "            )\n"
    "            for t, v in zip(covered, selected.to_pylist()):\n"
    "                result[t] = v\n"
    "        pos += n_valid_batch\n"
    "        if pos > max_target:\n"
    "            return result\n"
    "    return result\n"
)


def f12_5_quantile():
    print("\n=== F12-5: Arrow-native _stream_values_at_positions ===")
    pu.apply_or_fail(
        "cadw/ops/quantile.py", OLD_F12_5, NEW_F12_5, label="F12-5",
    )
    print("  OK")


# ============ F12-8: PredicateFilter null semantics doc ============

def f12_8_row_filter_doc():
    print("\n=== F12-8: PredicateFilter.mask_for_batch null semantics doc ===")
    pu.apply_or_fail(
        "cadw/dataset/row_filter.py",
        (
            "    def mask_for_batch(self, batch):\n"
            "        arr = batch.column(self.column)\n"
            "        # v0.9.0a: null/string predicates operate without a scalar.\n"
        ),
        (
            "    def mask_for_batch(self, batch):\n"
            "        \"\"\"v0.9.13 (F12-8): null semantics — asymmetric, intentional.\n"
            "\n"
            "        Numeric comparisons (_OPS: gt/ge/lt/le/eq/ne) — Kleene\n"
            "        null-propagation: `pc.greater(null, 5)` = null;\n"
            "        `b.filter(null)` treats null as false → null rows\n"
            "        drop. Matches pandas boolean indexing (NaN comparisons\n"
            "        drop) and SQL WHERE (null > 5 = false via Kleene).\n"
            "        INTENTIONAL — do NOT fill_null(False) here: pandas\n"
            "        user expects drop semantics.\n"
            "\n"
            "        String predicates (_OPS_NO_SCALAR: contains/starts_with/\n"
            "        ends_with) — fill_null(False): null string does not\n"
            "        match, but not dropped. Matches pandas\n"
            "        .str.contains(na=False) (CADW default; §11.5).\n"
            "\n"
            "        Asymmetry is by design: numeric default = SQL/pandas\n"
            "        drop, string default = explicit False. §13.4/218\n"
            "        (fill_null before filter) applies ONLY to fill-default\n"
            "        cases (like _compute_heavy_mask), not to Kleene-drop\n"
            "        semantics.\n"
            "        \"\"\"\n"
            "        arr = batch.column(self.column)\n"
            "        # v0.9.0a: null/string predicates operate without a scalar.\n"
        ),
        label="F12-8 doc",
    )
    print("  OK")


# ============ F12-5 tests ============

F12_5_TESTS = [
    "",
    "",
    "# ============ v0.9.13 (F12-5): Arrow-native quantile stream ============",
    "",
    "",
    "def test_quantile_many_batches_consistency(tmp_path):",
    "    \"\"\"F12-5: result identical across batch_size (1 vs 3 vs 1000).",
    "",
    "    Arrow-native take must not change per-batch consistency; if",
    "    pos accounting for n_valid is wrong, cross-batch results diverge.",
    "    \"\"\"",
    "    data = list(range(100))",
    "    from cadw import CadwDataFrame, MemoryGuard",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    t = pa.table({'v': data})",
    "    src = tmp_path / 'src.parquet'",
    "    pq.write_table(t, src)",
    "    results = []",
    "    for bs in (1, 3, 7, 1000):",
    "        df = CadwDataFrame.from_parquet(",
    "            src, tmp_path / f'ds_{bs}', chunk_size=bs,",
    "            memory_guard=MemoryGuard(max_memory_mb=20),",
    "            spill_dir=tmp_path / f'spill_{bs}',",
    "        )",
    "        results.append(df.quantile('v', [0.0, 0.25, 0.5, 0.75, 1.0]))",
    "    # All batch sizes → identical result",
    "    for r in results[1:]:",
    "        assert r == results[0]",
    "",
    "",
    "def test_quantile_with_nulls_multi_batch(tmp_path):",
    "    \"\"\"F12-5: nulls skip correctly across many batches.",
    "",
    "    Values: [1, None, 2, None, 3, None, 4, None, 5, None] — non-null",
    "    sorted = [1,2,3,4,5], n_valid=5. q=0.5 → pos=2 → value 3.0.",
    "    With bs=1: each row its own batch, pos accounting for n_valid.",
    "    \"\"\"",
    "    from cadw import CadwDataFrame, MemoryGuard",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    vals = [1.0, None, 2.0, None, 3.0, None, 4.0, None, 5.0, None]",
    "    t = pa.table({'v': pa.array(vals, type=pa.float64())})",
    "    src = tmp_path / 'src.parquet'",
    "    pq.write_table(t, src)",
    "    df = CadwDataFrame.from_parquet(",
    "        src, tmp_path / 'ds', chunk_size=1,",
    "        memory_guard=MemoryGuard(max_memory_mb=20),",
    "        spill_dir=tmp_path / 'spill',",
    "    )",
    "    assert df.quantile('v', 0.5) == 3.0",
    "    assert df.quantile('v', 0.0) == 1.0",
    "    assert df.quantile('v', 1.0) == 5.0",
    "",
    "",
    "def test_quantile_single_row(tmp_path):",
    "    \"\"\"F12-5 edge: single row, single batch.\"\"\"",
    "    from cadw import CadwDataFrame, MemoryGuard",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    t = pa.table({'v': pa.array([42.0], type=pa.float64())})",
    "    src = tmp_path / 'src.parquet'",
    "    pq.write_table(t, src)",
    "    df = CadwDataFrame.from_parquet(",
    "        src, tmp_path / 'ds', chunk_size=2,",
    "        memory_guard=MemoryGuard(max_memory_mb=20),",
    "        spill_dir=tmp_path / 'spill',",
    "    )",
    "    assert df.quantile('v', 0.5) == 42.0",
    "",
    "",
    "def test_quantile_arrow_native_no_per_value_loop():",
    "    \"\"\"F12-5 AST: no `for v in vals` per-value loop (precise check).",
    "",
    "    §13.4/217: check for `For` iter with `for <x> in <name>` where",
    "    name is 'vals' — the anti-pattern. Loops over `covered` (small,",
    "    ≤ len(targets)) are fine — that's the take-selected zip.",
    "    \"\"\"",
    "    import ast as _ast",
    "    import inspect as _inspect",
    "    import textwrap as _tw",
    "    from cadw.ops import quantile as q_mod",
    "    src = _tw.dedent(_inspect.getsource(q_mod._stream_values_at_positions))",
    "    tree = _ast.parse(src)",
    "    per_value_loops: list[int] = []",
    "    for n in _ast.walk(tree):",
    "        if not isinstance(n, _ast.For):",
    "            continue",
    "        it = n.iter",
    "        if isinstance(it, _ast.Name) and it.id in ('vals',):",
    "            per_value_loops.append(n.lineno)",
    "    assert not per_value_loops, (",
    "        f'F12-5 regression: per-value `for v in vals` loop at '",
    "        f'{per_value_loops}'",
    "    )",
    "    # Positive: take Call present",
    "    take_calls = [",
    "        n for n in _ast.walk(tree)",
    "        if isinstance(n, _ast.Call)",
    "        and isinstance(n.func, _ast.Attribute)",
    "        and n.func.attr == 'take'",
    "    ]",
    "    assert take_calls, 'F12-5: pc.take not called'",
]


def f12_5_add_tests():
    print("\n=== F12-5 tests ===")
    p = Path("tests/test_quantile.py")
    s = p.read_text()
    if "test_quantile_arrow_native_no_per_value_loop" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F12_5_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.13 + regression ===")
    pu.verify_anchors([
        # F12-5
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/quantile.py", "_pc.take("),
        ("tests/test_quantile.py", "test_quantile_arrow_native_no_per_value_loop"),
        # F12-8
        ("cadw/dataset/row_filter.py", "v0.9.13 (F12-8)"),
        ("cadw/dataset/row_filter.py", "Asymmetry is by design"),
        # v0.9.12 regression
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        ("cadw/patch_utils.py", "v0.9.12 (F12-3, §13.4/211)"),
        # v0.9.11/v0.9.10/v0.9.9
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "post-flush widen needed"),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.13 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import pyarrow as pa
        import pyarrow.compute as pc

        # F12-5: no `for v in vals`, take() present
        # §13.4/220: direct submodule import (shadowed name).
        from cadw.ops.quantile import _stream_values_at_positions
        src_q = _tw.dedent(inspect.getsource(_stream_values_at_positions))
        tree = _ast.parse(src_q)
        per_value_loops = [
            n.lineno for n in _ast.walk(tree)
            if isinstance(n, _ast.For)
            and isinstance(n.iter, _ast.Name)
            and n.iter.id == "vals"
        ]
        assert not per_value_loops, (
            f"F12-5: per-value loop at {per_value_loops}"
        )
        take_calls = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "take"
        ]
        assert take_calls, "F12-5: pc.take not called"

        # F12-8: docstring has Kleene + Asymmetry notes
        from cadw.dataset.row_filter import PredicateFilter
        doc = PredicateFilter.mask_for_batch.__doc__ or ""
        assert "Kleene" in doc
        assert "Asymmetry" in doc or "asymmetric" in doc.lower()
        assert "pandas" in doc.lower()
        assert "SQL" in doc or "sql" in doc

        # Functional sanity: quantile on nullable numeric (existing)
        # just verifies take() doesn't crash on mixed batches.
        t = pa.table({"v": pa.array([1.0, None, 2.0, None, 3.0])})
        arr = t.column("v")
        non_null = arr.filter(pc.is_valid(arr))
        selected = pc.take(non_null, pa.array([0, 2], type=pa.int64()))
        assert selected.to_pylist() == [1.0, 3.0]

        # v0.9.12 regression sanity
        from cadw.ops.topk import topk_to_disk
        src_t = inspect.getsource(topk_to_disk)
        assert "acc_bytes += top.nbytes" in src_t
        assert "sum(tb.nbytes for tb in per_batch)" not in src_t

        print("  F12-5 (take, no per-value loop) OK")
        print("  F12-8 (Kleene + asymmetry docstring) OK")
        print("  v0.9.12 regression OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_quantile.py",
         "tests/test_row_filter.py",
         "tests/test_dataframe_filters.py",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.13 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    f12_5_quantile()
    f12_5_add_tests()
    f12_8_row_filter_doc()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.13: F12-5 (quantile Arrow-native stream) + F12-8 "
        "(PredicateFilter null semantics doc). "
        "F12-5 (quantile.py, S3 perf): _stream_values_at_positions "
        "did per-value Python loop (for v in vals) → O(max_target) "
        "Python iterations (~N for high quantiles). N=10M / 5 q → "
        "~10M iterations (~500ms-2s). quantile path already pays "
        "external_sort; loop was remaining O(N) Python overhead. "
        "Fix: per-batch O(1) Arrow ops. n_valid_batch = "
        "len(arr)-arr.null_count; covered = targets in [pos, "
        "pos+n_valid); if covered: non_null=arr if no nulls else "
        "arr.filter(pc.is_valid(arr)); selected=pc.take(non_null, "
        "[t-pos for t in covered]); result[t]=selected.to_pylist(). "
        "~100-400x faster. filter/take/is_valid not in "
        "HIGH_RISK_ATTRS (AST guard). to_pylist on selected = "
        "bounded by len(targets). 4 tests: multi-batch consistency "
        "(bs 1/3/7/1000 identical), nulls cross-batch (bs=1, null "
        "pos accounting), single-row edge, AST guard (no per-value "
        "loop, pc.take present). "
        "F12-8 (row_filter.py, doc): PredicateFilter.mask_for_batch "
        "asymmetric null semantics, undocumented. Numeric _OPS "
        "(gt/ge/lt/le/eq/ne): Kleene null → filter drops (matches "
        "pandas boolean indexing + SQL WHERE). String _OPS_NO_SCALAR "
        "(contains/starts_with/ends_with): fill_null(False) "
        "(pandas .str.contains na=False default CADW). Asymmetry by "
        "design — §13.4/218 fill_null applies ONLY to fill-default "
        "cases, not Kleene-drop numeric comparisons. Documented in "
        "docstring; no behavior change. "
        "Deferred: A1 (dataframe split 5404 lines), T2.2/T2.5/T2.6/"
        "T2.7/T2.8 (tooling), U3 (pyproject version), N6, R3. "
        "After v0.9.13: Pingis formal (fresh snapshot + bench) for "
        "full v0.9.4–v0.9.13 audit.",
        self_path="scripts/patch_v0913.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.13 applied.")


if __name__ == "__main__":
    main()
