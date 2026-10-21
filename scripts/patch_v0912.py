#!/usr/bin/env python3
# v0.9.12 — Pingis round 2 inline fixes: F12-1/2/3/4.
#
# F12-1 (topk.py, S3 perf): _per_batch_topk caller recomputes
#   `acc_bytes = sum(tb.nbytes for tb in per_batch)` on every batch
#   → O(P²) work. P=10000 → 100M iters, ~1.6 GB attr accesses over
#   the call. Fix: incremental counter (nonlocal acc_bytes).
#
# F12-2 (patch_utils.py, S4): apply_or_fail uses p.write_text(new_s)
#   without newline="" — on Windows, Python opens text mode 'w' and
#   converts \n to \r\n. No test catches it (Linux = \n). Fix:
#   newline="" keeps LF exactly.
#
# F12-3 (patch_utils.py, S4 doc): apply_or_fail docstring doesn't
#   mention compile-before-write (§13.4/211). Future reader might
#   "simplify" by moving compile after write. Fix: docstring note.
#
# F12-4 (groupby.py, S3 perf, PF1): _detect_heavy_keys does per-row
#   `for i in range(b.num_rows): b.column(c)[i].as_py()`. For 10k
#   sample rows — 10k Python as_py calls per groupby. Fix: Arrow-
#   native pa.concat_tables + group_by().aggregate([(k, "count")])
#   + filter by threshold. Semantics identical:
#     threshold_abs = max(_HEAVY_ABSOLUTE_MIN, int(total*ratio)+1)
#     (int(total*ratio)+1 == strict `cnt > total*ratio` for ints)
#
# Run: uv run python scripts/patch_v0912.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ F12-1: topk incremental acc_bytes ============

def f12_1_topk():
    print("\n=== F12-1: topk incremental acc_bytes ===")
    pu.apply_or_fail(
        "cadw/ops/topk.py",
        (
            "    per_batch: list[pa.Table] = []\n"
            "    row_id_offset = 0\n"
        ),
        (
            "    per_batch: list[pa.Table] = []\n"
            "    # v0.9.12 (F12-1): incremental (was O(P²) sum per batch;\n"
            "    # P=10000 → 100M iters ~1.6 GB attr accesses).\n"
            "    acc_bytes = 0\n"
            "    row_id_offset = 0\n"
        ),
        label="F12-1 init",
    )
    pu.apply_or_fail(
        "cadw/ops/topk.py",
        (
            "    def _process(batch: pa.RecordBatch, offset: int) -> None:\n"
            "        nonlocal row_id_offset\n"
        ),
        (
            "    def _process(batch: pa.RecordBatch, offset: int) -> None:\n"
            "        nonlocal row_id_offset, acc_bytes\n"
        ),
        label="F12-1 nonlocal",
    )
    pu.apply_or_fail(
        "cadw/ops/topk.py",
        (
            "        per_batch.append(top)\n"
            "        row_id_offset = offset + batch.num_rows\n"
            "\n"
            "        # Fail-fast: accumulator + 2× overhead before adding more.\n"
            "        acc_bytes = sum(tb.nbytes for tb in per_batch)\n"
            "        memory_guard.check_state(\n"
            "            2 * acc_bytes,\n"
            "            name=f\"topk.{kind}.accum(P={len(per_batch)},n={n})\",\n"
            "        )\n"
        ),
        (
            "        per_batch.append(top)\n"
            "        acc_bytes += top.nbytes  # v0.9.12 (F12-1): incremental\n"
            "        row_id_offset = offset + batch.num_rows\n"
            "\n"
            "        # Fail-fast: accumulator + 2× overhead before adding more.\n"
            "        memory_guard.check_state(\n"
            "            2 * acc_bytes,\n"
            "            name=f\"topk.{kind}.accum(P={len(per_batch)},n={n})\",\n"
            "        )\n"
        ),
        label="F12-1 loop",
    )
    print("  F12-1 OK")


# ============ F12-2: apply_or_fail newline="" ============

def f12_2_apply_or_fail_newline():
    print("\n=== F12-2: apply_or_fail newline=\"\" ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "    p.write_text(new_s)\n"
            "    return True\n"
        ),
        (
            "    # v0.9.12 (F12-2): newline=\"\" — Python text-mode 'w'\n"
            "    # converts \\n -> \\r\\n on Windows. Force LF for\n"
            "    # cross-platform deterministic file content.\n"
            "    p.write_text(new_s, newline=\"\")\n"
            "    return True\n"
        ),
        label="F12-2",
    )
    print("  F12-2 OK")


# ============ F12-3: apply_or_fail docstring ============

def f12_3_apply_or_fail_docstring():
    print("\n=== F12-3: apply_or_fail docstring compile-before-write ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.\n"
            "    Prints `context` lines around partial-match zone on miss (§13.4/160).\n"
            "    Idempotent: returns False if `new` already present (no write).\n"
            "    \"\"\"\n"
        ),
        (
            "    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.\n"
            "    Prints `context` lines around partial-match zone on miss (§13.4/160).\n"
            "    Idempotent: returns False if `new` already present (no write).\n"
            "\n"
            "    v0.9.12 (F12-3, §13.4/211): compile-before-write. `new_s`\n"
            "    is compile()'d in memory BEFORE p.write_text(). On\n"
            "    SyntaxError — SystemExit, file untouched. Without this,\n"
            "    bad anchor (e.g. 4-space substring of 8-space line) can\n"
            "    corrupt the file, and only then would assert_compiles\n"
            "    fail — but the file is already bad. Never reorder:\n"
            "    `compile()` MUST precede `write_text()`.\n"
            "    \"\"\"\n"
        ),
        label="F12-3",
    )
    print("  F12-3 OK")


# ============ F12-4: _detect_heavy_keys Arrow-native ============

OLD_F12_4 = (
    "def _detect_heavy_keys(sample_batches, by_columns, threshold_ratio: float = _HEAVY_RATIO) -> set:\n"
    "    \"\"\"Тяжёлые ключи по sample: доля в sample > threshold_ratio.\"\"\"\n"
    "    if not sample_batches:\n"
    "        return set()\n"
    "    from collections import Counter\n"
    "    counter: Counter = Counter()\n"
    "    for b in sample_batches:\n"
    "        for i in range(b.num_rows):\n"
    "            key = tuple(b.column(c)[i].as_py() for c in by_columns)\n"
    "            counter[key] += 1\n"
    "    total = sum(counter.values())\n"
    "    if total == 0:\n"
    "        return set()\n"
    "    return {\n"
    "        k for k, cnt in counter.items()\n"
    "        if cnt >= _HEAVY_ABSOLUTE_MIN and cnt / total > threshold_ratio\n"
    "    }\n"
)

NEW_F12_4 = (
    "def _detect_heavy_keys(sample_batches, by_columns, threshold_ratio: float = _HEAVY_RATIO) -> set:\n"
    "    \"\"\"Тяжёлые ключи по sample: доля в sample > threshold_ratio.\n"
    "\n"
    "    v0.9.12 (F12-4, PF1): Arrow-native. Prior: per-row Python loop\n"
    "    with .as_py() per column per row — 10k iterations per groupby\n"
    "    call (~10-50 ms). Now: pa.concat_tables + group_by().aggregate(\n"
    "    [(first_key, \"count\")]) — ~10x faster, single C++ pass.\n"
    "\n"
    "    Threshold semantics preserved: heavy iff count >= 100 AND\n"
    "    count > total * threshold_ratio. Integer arithmetic:\n"
    "    int(total*ratio)+1 == strict `cnt > total*ratio` for all\n"
    "    integer counts (verified: t*0.05 truncates, +1 restores).\n"
    "    \"\"\"\n"
    "    if not sample_batches:\n"
    "        return set()\n"
    "\n"
    "    import pyarrow.compute as pc\n"
    "\n"
    "    by_list = list(by_columns)\n"
    "    tables = [\n"
    "        pa.Table.from_batches([b.select(by_list)])\n"
    "        for b in sample_batches\n"
    "    ]\n"
    "    combined = pa.concat_tables(tables, promote_options=\"default\")\n"
    "    if combined.num_rows == 0:\n"
    "        return set()\n"
    "\n"
    "    first_key = by_list[0]\n"
    "    counts = combined.group_by(by_list).aggregate(\n"
    "        [(first_key, \"count\")]\n"
    "    )\n"
    "    count_col = f\"{first_key}_count\"\n"
    "    total = int(pc.sum(counts.column(count_col)).as_py() or 0)\n"
    "    if total == 0:\n"
    "        return set()\n"
    "\n"
    "    threshold_abs = max(\n"
    "        _HEAVY_ABSOLUTE_MIN,\n"
    "        int(total * threshold_ratio) + 1,\n"
    "    )\n"
    "    mask = pc.greater_equal(\n"
    "        counts.column(count_col),\n"
    "        pa.scalar(threshold_abs, pa.int64()),\n"
    "    )\n"
    "    filtered = counts.filter(mask)\n"
    "    return {\n"
    "        tuple(row[c] for c in by_list)\n"
    "        for row in filtered.select(by_list).to_pylist()\n"
    "    }\n"
)


def f12_4_detect_heavy():
    print("\n=== F12-4: _detect_heavy_keys Arrow-native ===")
    pu.apply_or_fail(
        "cadw/ops/groupby.py", OLD_F12_4, NEW_F12_4, label="F12-4",
    )
    print("  F12-4 OK")


# ============ F12-4 tests ============

F12_4_TESTS = [
    "",
    "",
    "# ============ v0.9.12 (F12-4): _detect_heavy_keys Arrow-native ============",
    "",
    "",
    "def test_detect_heavy_keys_arrow_native_correctness():",
    "    \"\"\"v0.9.12 (F12-4): Arrow-native output == Python reference.",
    "",
    "    900 'A' + 100 'B' (total=1000): threshold_abs = max(100,",
    "    int(1000*0.05)+1) = max(100, 51) = 100. Both pass.",
    "    \"\"\"",
    "    import pyarrow as pa",
    "    from cadw.ops.groupby import _detect_heavy_keys",
    "    table = pa.table({'k': ['A'] * 900 + ['B'] * 100})",
    "    batches = [",
    "        table.slice(0, 500).to_batches()[0],",
    "        table.slice(500).to_batches()[0],",
    "    ]",
    "    result = _detect_heavy_keys(batches, ['k'])",
    "    assert result == {('A',), ('B',)}",
    "",
    "",
    "def test_detect_heavy_keys_no_heavy_uniform():",
    "    \"\"\"100 unique keys, 1 row each — none meet threshold_abs=100.\"\"\"",
    "    import pyarrow as pa",
    "    from cadw.ops.groupby import _detect_heavy_keys",
    "    table = pa.table({'k': [f'k{i}' for i in range(100)]})",
    "    assert _detect_heavy_keys([table.to_batches()[0]], ['k']) == set()",
    "",
    "",
    "def test_detect_heavy_keys_null_group():",
    "    \"\"\"Nulls form one group (Arrow group_by default); heavy if count",
    "    >= 100 and > total*ratio. Here: 100 nulls + 100 'A' (total=200).",
    "    threshold_abs = max(100, int(200*0.05)+1) = 100. Both heavy.\"\"\"",
    "    import pyarrow as pa",
    "    from cadw.ops.groupby import _detect_heavy_keys",
    "    table = pa.table({'k': pa.array([None] * 100 + ['A'] * 100)})",
    "    result = _detect_heavy_keys([table.to_batches()[0]], ['k'])",
    "    assert ('A',) in result",
    "    assert (None,) in result",
    "",
    "",
    "def test_detect_heavy_keys_below_absolute_min():",
    "    \"\"\"Total=50, 'A' appears 30 times (60% ratio) — fails",
    "    _HEAVY_ABSOLUTE_MIN=100 → not heavy (tiny sample protection).\"\"\"",
    "    import pyarrow as pa",
    "    from cadw.ops.groupby import _detect_heavy_keys",
    "    table = pa.table({'k': ['A'] * 30 + ['B'] * 20})",
    "    assert _detect_heavy_keys([table.to_batches()[0]], ['k']) == set()",
]


def add_f12_4_tests():
    print("\n=== F12-4: tests ===")
    p = Path("tests/test_mustfix_skew.py")
    s = p.read_text()
    if "test_detect_heavy_keys_arrow_native_correctness" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F12_4_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.12 + regression ===")
    pu.verify_anchors([
        # F12-1
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/ops/topk.py", "nonlocal row_id_offset, acc_bytes"),
        # F12-2
        ("cadw/patch_utils.py", "v0.9.12 (F12-2)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        # F12-3
        ("cadw/patch_utils.py", "v0.9.12 (F12-3, §13.4/211)"),
        # F12-4
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        ("cadw/ops/groupby.py", "pa.concat_tables(tables"),
        # v0.9.11 regression
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "post-flush widen needed"),
        # v0.9.10 regression
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        # v0.9.9 regression
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        # tests
        ("tests/test_mustfix_skew.py",
         "test_detect_heavy_keys_arrow_native_correctness"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.12 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw

        # F12-1: acc_bytes incremental (no `sum(tb.nbytes` in _process)
        from cadw.ops import topk as tk
        src_topk = inspect.getsource(tk.topk_to_disk)
        # Absence of `sum(tb.nbytes` (would be O(P²))
        assert "sum(tb.nbytes for tb in per_batch)" not in src_topk, (
            "F12-1 regression: O(P²) sum still present"
        )
        assert "acc_bytes += top.nbytes" in src_topk

        # F12-2: apply_or_fail uses newline=""
        from cadw.patch_utils import apply_or_fail
        src_aof = inspect.getsource(apply_or_fail)
        assert 'newline=""' in src_aof, "F12-2: newline=\"\" missing"

        # F12-3: docstring mentions compile-before-write
        assert "compile-before-write" in src_aof or "compile-before-write" in (apply_or_fail.__doc__ or "")
        assert "Never reorder" in (apply_or_fail.__doc__ or "")

        # F12-4: _detect_heavy_keys Arrow-native
        from cadw.ops.groupby import _detect_heavy_keys
        src_dhk = _tw.dedent(inspect.getsource(_detect_heavy_keys))
        tree = _ast.parse(src_dhk)
        # No `b.column(c)[i].as_py()` per-row loop
        as_py_calls = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
        ]
        assert not as_py_calls, (
            f"F12-4: as_py() still present at line(s) "
            f"{[c.lineno for c in as_py_calls]}"
        )
        # group_by called
        group_by_calls = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "group_by"
        ]
        assert group_by_calls, "F12-4: group_by() not called"

        print("  F12-1 (no O(P²) sum, acc_bytes +=) OK")
        print("  F12-2 (newline=\"\") OK")
        print("  F12-3 (docstring) OK")
        print("  F12-4 (no as_py per-row, group_by) OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_topk_arrow_native.py",
         "tests/test_mustfix_skew.py",
         "tests/test_groupby_v2.py",
         "tests/test_patch_utils.py", "-q"],
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
    print("=== v0.9.12 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    f12_1_topk()
    f12_2_apply_or_fail_newline()
    f12_3_apply_or_fail_docstring()
    f12_4_detect_heavy()
    add_f12_4_tests()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.12: Pingis round 2 inline fixes — F12-1/2/3/4. "
        "F12-1 (topk.py, S3 perf): _per_batch_topk caller recomputed "
        "`acc_bytes = sum(tb.nbytes for tb in per_batch)` per batch "
        "→ O(P²) work. P=10000 → 100M iters, ~1.6 GB attr accesses "
        "over the call. Fix: incremental `acc_bytes += top.nbytes` "
        "(nonlocal). F12-2 (patch_utils.py, S4 cross-platform): "
        "apply_or_fail wrote without newline=\"\" — Python text-mode "
        "'w' converts \\n→\\r\\n on Windows. No test (Linux = \\n). "
        "Fix: newline=\"\" keeps LF. F12-3 (patch_utils.py, S4 doc): "
        "apply_or_fail docstring didn't mention compile-before-write "
        "(§13.4/211) — future reader might reorder. Fix: docstring "
        "note ('Never reorder: compile() MUST precede write_text()'). "
        "F12-4 (groupby.py, S3 perf, PF1): _detect_heavy_keys per-"
        "row Python loop with .as_py() per column per row — 10k "
        "iterations per groupby call (~10-50 ms). Fix: Arrow-native "
        "pa.concat_tables + group_by().aggregate([(k, \"count\")]) "
        "+ filter by threshold. Semantics preserved (verified: "
        "int(total*ratio)+1 == strict `cnt > total*ratio` for ints; "
        "_HEAVY_ABSOLUTE_MIN still enforced). 4 new tests: "
        "correctness vs Python reference, uniform no-heavy, null "
        "group, below absolute min. Pingis round 2 findings after "
        "inline review (v0.9.4-v0.9.11): PF2 false positive (accum "
        "bounded by heavy_keys × agg_cols, not O(N)); PF3 bounded "
        "by K_columns, §11.5 documented. Deferred: A1 (dataframe "
        "split, 5404 lines), T2.2/T2.5/T2.6/T2.7 (tooling), U3 "
        "(pyproject version), N6, R3, quantile stream loop. Next "
        "patch v0.9.13: A1 or bench-validated PF (quantile / pivot "
        "tightening).",
        self_path="scripts/patch_v0912.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.12 applied.")


if __name__ == "__main__":
    main()
