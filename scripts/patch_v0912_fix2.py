#!/usr/bin/env python3
# v0.9.12-fix2 — data-loss in _compute_heavy_mask + test fix. v0.9.12 COMPLETE.
#
# v0.9.12-fix1 dogfood green. pytest failed test_detect_heavy_keys_null_group:
#   assert (None,) in result  →  got only {('A',)}
#
# Root is TWO issues, one of which is a serious PRE-EXISTING data-loss
# bug that F12-4 exposed:
#
# (1) test_detect_heavy_keys_null_group was MY test (v0.9.12 F12-4),
#     asserting old behavior (nulls included in heavy). New Arrow-native
#     _detect_heavy_keys excludes nulls because pc.greater_equal(null_count,
#     threshold) → null → filter treats as false. This is actually the
#     CORRECT contract: nulls can't be isolated via pc.equal(scalar(value))
#     (Kleene: null==anything = null). Test wrong, not code.
#
# (2) SERIOUS pre-existing bug in _compute_heavy_mask (v0.3.4, 3 months
#     in prod, untested): 
#       mask = pa.array([False] * n, type=pa.bool_())
#       for key in heavy_keys:
#           k_mask = pa.array([True] * n, type=pa.bool_())
#           for col, v in zip(by_columns, key):
#               k_mask = pc.and_(k_mask, pc.equal(batch.column(col), pa.scalar(v)))
#           mask = pc.or_(mask, k_mask)
#       return mask
#
#     For any row where by_col is null: pc.equal(null, non_null) = null.
#     and_(True, null) = null. or_(False, null) = null. Final mask = null
#     on null-key rows. batch.filter(mask) treats null as false → drops
#     from heavy_batch. invert(mask)=null → drops from light_batch too.
#
#     RESULT: any groupby with null-key data that triggers skew path
#     SILENTLY DROPS null-key rows. F12-4's detection just exposed it
#     in test form. Nulls NEVER appear in any output — not in heavy,
#     not in light.
#
# Fix:
#   (a) test_detect_heavy_keys_null_group: accept new contract (nulls
#       not isolated as heavy). Rename to ..._excluded to encode it.
#   (b) _compute_heavy_mask: fill_null(mask, False) before return.
#       Nulls explicitly → light path → shuffled by _NULL_SENTINEL
#       → grouped by pyarrow → preserved.
#   (c) New E2E test: groupby with nulls + skew → null group survives.
#   (d) New unit test: _compute_heavy_mask returns no null values.
#
# §13.4/218 (new): Kleene-logic (null-propagating) arrow predicates in
# presence of nulls must be explicitly defaulted (fill_null or
# pc.coalesce), not left as null. Null in boolean mask = "not sure" =
# filter drops = silent data loss. Rule: any boolean mask from
# pc.equal/and_/or_/not_equal over nullable input must be terminated
# with fill_null(default) before filter.
#
# State on disk (from v0.9.12, all applied):
#   F12-1 topk incremental acc_bytes
#   F12-2 newline="" (apply_or_fail)
#   F12-3 docstring compile-before-write
#   F12-4 _detect_heavy_keys Arrow-native
#   4 F12-4 tests (3 pass, 1 wrong assertion)
#   commit never reached (pytest failed)
#
# Run: uv run python scripts/patch_v0912_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ Fix 1: test_detect_heavy_keys_null_group (test contract) ============

OLD_TEST = '''def test_detect_heavy_keys_null_group():
    """Nulls form one group (Arrow group_by default); heavy if count
    >= 100 and > total*ratio. Here: 100 nulls + 100 'A' (total=200).
    threshold_abs = max(100, int(200*0.05)+1) = 100. Both heavy."""
    import pyarrow as pa
    from cadw.ops.groupby import _detect_heavy_keys
    table = pa.table({'k': pa.array([None] * 100 + ['A'] * 100)})
    result = _detect_heavy_keys([table.to_batches()[0]], ['k'])
    assert ('A',) in result
    assert (None,) in result'''

NEW_TEST = '''def test_detect_heavy_keys_null_group_excluded():
    """Nulls are NOT isolated as heavy keys (v0.9.12 F12-4 contract).

    Rationale (§13.4/218): null-key groups can't be isolated via
    pc.equal(col, scalar(None)) — null==anything returns null (Kleene
    logic), which filter treats as false → null-heavy mask drops
    everything. So nulls ALWAYS go through light path (shuffled by
    _NULL_SENTINEL, grouped by pyarrow) — see _compute_heavy_mask
    fill_null fix. Asserting (None,) NOT in heavy enforces this.

    Case: 100 nulls + 100 'A' (total=200). threshold_abs=max(100,
    int(200*0.05)+1)=100. 'A': 100>=100 & 0.5>0.05 → heavy. Nulls:
    same count but excluded by pc.greater_equal returning null.
    """
    import pyarrow as pa
    from cadw.ops.groupby import _detect_heavy_keys
    table = pa.table({'k': pa.array([None] * 100 + ['A'] * 100)})
    result = _detect_heavy_keys([table.to_batches()[0]], ['k'])
    assert ('A',) in result
    assert (None,) not in result, (
        "nulls must NOT be in heavy_keys (F12-4 contract): they go "
        "through light path via _compute_heavy_mask fill_null"
    )'''


def fix_test():
    print("\n=== fix test_detect_heavy_keys_null_group ===")
    pu.apply_or_fail(
        "tests/test_mustfix_skew.py", OLD_TEST, NEW_TEST,
        label="test null_group",
    )
    print("  OK")


# ============ Fix 2: _compute_heavy_mask fill_null(False) ============

def fix_compute_heavy_mask():
    print("\n=== fix _compute_heavy_mask: fill_null(mask, False) ===")
    pu.apply_or_fail(
        "cadw/ops/groupby.py",
        (
            "def _compute_heavy_mask(batch: pa.RecordBatch, by_columns, heavy_keys: set) -> pa.Array:\n"
            "    n = batch.num_rows\n"
            "    if not heavy_keys or n == 0:\n"
            "        return pa.array([False] * n, type=pa.bool_())\n"
            "    mask = pa.array([False] * n, type=pa.bool_())\n"
            "    for key in heavy_keys:\n"
            "        k_mask = pa.array([True] * n, type=pa.bool_())\n"
            "        for col, v in zip(by_columns, key):\n"
            "            k_mask = pc.and_(k_mask, pc.equal(batch.column(col), pa.scalar(v)))\n"
            "        mask = pc.or_(mask, k_mask)\n"
            "    return mask\n"
        ),
        (
            "def _compute_heavy_mask(batch: pa.RecordBatch, by_columns, heavy_keys: set) -> pa.Array:\n"
            "    \"\"\"Boolean mask: True where row's key is in heavy_keys.\n"
            "\n"
            "    v0.9.12-fix2 (§13.4/218): fill_null(False). Prior code\n"
            "    returned mask with null values on rows where any by_col\n"
            "    was null: pc.equal(null, non_null) = null (Kleene logic),\n"
            "    and_(True, null) = null, or_(False, null) = null. Then\n"
            "    batch.filter(null) treats null as false → null-key rows\n"
            "    dropped from heavy_batch AND from light_batch\n"
            "    (invert(null) = null). Silent data loss: any groupby\n"
            "    with nulls triggering skew path lost all null-key rows\n"
            "    (v0.3.4 to v0.9.12-fix1, ~3 months, no test).\n"
            "\n"
            "    Fix: fill_null(False) — null-key rows explicitly → light\n"
            "    path (invert(False)=True), shuffled by _NULL_SENTINEL,\n"
            "    grouped by pyarrow, preserved in output. Non-null data\n"
            "    unaffected (mask has no nulls when no nulls in input).\n"
            "    \"\"\"\n"
            "    n = batch.num_rows\n"
            "    if not heavy_keys or n == 0:\n"
            "        return pa.array([False] * n, type=pa.bool_())\n"
            "    mask = pa.array([False] * n, type=pa.bool_())\n"
            "    for key in heavy_keys:\n"
            "        k_mask = pa.array([True] * n, type=pa.bool_())\n"
            "        for col, v in zip(by_columns, key):\n"
            "            k_mask = pc.and_(k_mask, pc.equal(batch.column(col), pa.scalar(v)))\n"
            "        mask = pc.or_(mask, k_mask)\n"
            "    # v0.9.12-fix2 (§13.4/218): Kleene null → False, not dropped.\n"
            "    return pc.fill_null(mask, pa.scalar(False))\n"
        ),
        label="compute_heavy_mask fill_null",
    )
    print("  OK")


# ============ Fix 3: E2E + unit tests for null preservation ============

NEW_TESTS = [
    "",
    "",
    "# ============ v0.9.12-fix2 (§13.4/218): null-key preservation ============",
    "",
    "",
    "def test_compute_heavy_mask_no_null_values():",
    "    \"\"\"§13.4/218: _compute_heavy_mask returns bool with NO nulls,",
    "    even when input has null keys (was silent data loss).\"\"\"",
    "    import pyarrow as pa",
    "    from cadw.ops.groupby import _compute_heavy_mask",
    "    table = pa.table({'k': pa.array([None, 'A', 'B', None, 'A']),",
    "                      'v': [1.0, 2.0, 3.0, 4.0, 5.0]})",
    "    batch = table.to_batches()[0]",
    "    mask = _compute_heavy_mask(batch, ['k'], {('A',)})",
    "    assert mask.type == pa.bool_()",
    "    assert mask.null_count == 0, (",
    "        f'mask has nulls: {mask.null_count} — §13.4/218 regression'",
    "    )",
    "    assert mask.to_pylist() == [False, True, False, False, True]",
    "",
    "",
    "def test_groupby_skew_preserves_null_keys(tmp_path):",
    "    \"\"\"§13.4/218 E2E: null-key rows survive skew path.",
    "",
    "    Before v0.9.12-fix2: 500 null keys + 400 'A' + 100 'B' with",
    "    skew triggered → null rows dropped from both heavy_batch and",
    "    light_batch (filter treats null mask as false). Result had",
    "    only {A: 400, B: 100}, nulls lost.",
    "",
    "    After: nulls → light path (invert(fill_null(False))=True),",
    "    shuffled by _NULL_SENTINEL, grouped → {None: 500, A: 400,",
    "    B: 100}.",
    "    \"\"\"",
    "    import pyarrow as pa",
    "    from cadw.core.memory_guard import MemoryGuard",
    "    from cadw.ops.groupby import groupby_aggregate",
    "    n = 1000",
    "    table = pa.table({",
    "        'k': pa.array([None] * 500 + ['A'] * 400 + ['B'] * 100),",
    "        'v': [1.0] * n,",
    "    })",
    "    def it():",
    "        for i in range(0, n, 100):",
    "            yield table.slice(i, 100).to_batches()[0]",
    "    guard = MemoryGuard(max_memory_bytes=200_000)",
    "    result = groupby_aggregate(",
    "        it(), ['k'], ['v'], ['count'],",
    "        guard, tmp_path, num_partitions=4, log=False,",
    "    )",
    "    rows = {r['k']: r['v_count'] for r in result.to_pylist()}",
    "    assert rows.get(None) == 500, (",
    "        f'null-key rows dropped (was silent data loss pre-fix2): '",
    "        f'got {rows}, expected None=500'",
    "    )",
    "    assert rows.get('A') == 400",
    "    assert rows.get('B') == 100",
    "    assert sum(rows.values()) == n",
]


def add_tests():
    print("\n=== add null-preservation tests ===")
    p = Path("tests/test_mustfix_skew.py")
    s = p.read_text()
    if "test_compute_heavy_mask_no_null_values" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.12-fix2 + regression ===")
    pu.verify_anchors([
        # fix2
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/ops/groupby.py", "pc.fill_null(mask, pa.scalar(False))"),
        ("tests/test_mustfix_skew.py", "test_detect_heavy_keys_null_group_excluded"),
        ("tests/test_mustfix_skew.py", "test_compute_heavy_mask_no_null_values"),
        ("tests/test_mustfix_skew.py", "test_groupby_skew_preserves_null_keys"),
        # v0.9.12 regression
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        ("cadw/patch_utils.py", "v0.9.12 (F12-3, §13.4/211)"),
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        # v0.9.11 regression
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "post-flush widen needed"),
        # v0.9.10/v0.9.9 regression
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.12-fix2 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import pyarrow as pa

        # fix2: _compute_heavy_mask has fill_null, no bare `return mask`
        from cadw.ops.groupby import _compute_heavy_mask
        src_chm = _tw.dedent(inspect.getsource(_compute_heavy_mask))
        tree_chm = _ast.parse(src_chm)
        # fill_null Call present
        fill_calls = [
            n for n in _ast.walk(tree_chm)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "fill_null"
        ]
        assert fill_calls, (
            "fix2: _compute_heavy_mask missing pc.fill_null (data loss risk)"
        )
        # No bare `return mask` (must return fill_null(mask))
        for n in _ast.walk(tree_chm):
            if not isinstance(n, _ast.Return):
                continue
            if (isinstance(n.value, _ast.Name) and n.value.id == "mask"):
                raise AssertionError(
                    f"fix2: bare `return mask` at line {n.lineno} — "
                    f"Kleene nulls in mask will drop rows in filter"
                )

        # Behavior test: fill_null → no nulls, correct values
        table = pa.table({
            "k": pa.array([None, "A", "B", None, "A"]),
        })
        batch = table.to_batches()[0]
        mask = _compute_heavy_mask(batch, ["k"], {("A",)})
        assert mask.null_count == 0, (
            f"fix2: mask has {mask.null_count} nulls (data loss)"
        )
        assert mask.to_pylist() == [False, True, False, False, True]

        # F12-1/2/3/4 unchanged
        from cadw.ops import topk as tk
        src_topk = inspect.getsource(tk.topk_to_disk)
        assert "sum(tb.nbytes for tb in per_batch)" not in src_topk
        assert "acc_bytes += top.nbytes" in src_topk

        from cadw.patch_utils import apply_or_fail
        src_aof = inspect.getsource(apply_or_fail)
        assert 'newline=""' in src_aof
        assert "compile-before-write" in src_aof
        assert "Never reorder" in (apply_or_fail.__doc__ or "")

        from cadw.ops.groupby import _detect_heavy_keys
        src_dhk = _tw.dedent(inspect.getsource(_detect_heavy_keys))
        tree_dhk = _ast.parse(src_dhk)
        per_row_as_py = [
            n.lineno for n in _ast.walk(tree_dhk)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
            and isinstance(n.func.value, _ast.Subscript)
        ]
        assert not per_row_as_py, (
            f"F12-4 regression: per-row .as_py at {per_row_as_py}"
        )

        print("  fix2 (_compute_heavy_mask fill_null, no bare return mask) OK")
        print("  fix2 behavior (mask null_count=0 on null input) OK")
        print("  F12-1/2/3/4 regression OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_mustfix_skew.py",
         "tests/test_groupby_v2.py",
         "tests/test_topk_arrow_native.py",
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
    print("=== v0.9.12-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_test()
    fix_compute_heavy_mask()
    add_tests()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.12-fix2: silent data loss in skew path + F12-4 test "
        "fix. v0.9.12-fix1 dogfood green, pytest failed "
        "test_detect_heavy_keys_null_group. TWO issues: "
        "(1) test wrong — MY test (v0.9.12 F12-4) asserted old "
        "behavior (nulls in heavy_keys); new Arrow-native excludes "
        "them because pc.greater_equal(null_count, threshold) → null "
        "→ filter treats as false. Correct new contract: nulls NOT "
        "in heavy (can't be isolated via pc.equal(scalar(None))). "
        "Renamed test_detect_heavy_keys_null_group → "
        "_null_group_excluded. "
        "(2) SERIOUS pre-existing data-loss (v0.3.4, 3 months, "
        "untested) in _compute_heavy_mask: pc.equal(null, non_null) "
        "= null (Kleene), and_/or_ propagate null, filter treats "
        "null as false → null-key rows dropped from BOTH "
        "heavy_batch AND light_batch (invert(null)=null). Any "
        "groupby with nulls triggering skew path SILENTLY LOST "
        "null-key rows. F12-4 test just exposed it. Fix: "
        "pc.fill_null(mask, pa.scalar(False)) — nulls explicitly → "
        "light path (invert(False)=True), shuffled by _NULL_SENTINEL, "
        "grouped by pyarrow, preserved. Non-null data unaffected. "
        "§13.4/218 (new): Kleene-logic (null-propagating) arrow "
        "predicates in presence of nulls MUST be explicitly defaulted "
        "(fill_null / coalesce), not left as null. Null in boolean "
        "mask = 'not sure' = filter drops = silent data loss. Rule: "
        "any boolean mask from pc.equal/and_/or_/not_equal over "
        "nullable input must terminate with fill_null(default) before "
        "filter. Tests: test_detect_heavy_keys_null_group_excluded "
        "(contract), test_compute_heavy_mask_no_null_values (unit: "
        "mask.null_count==0), test_groupby_skew_preserves_null_keys "
        "(E2E: 500 nulls + 400 'A' + 100 'B' with skew → {None:500, "
        "A:400, B:100}, sum=1000). v0.9.12 COMPLETE: F12-1 topk "
        "incremental acc_bytes; F12-2 newline=\"\"; F12-3 docstring; "
        "F12-4 Arrow-native _detect_heavy_keys; fix2 fill_null on "
        "_compute_heavy_mask (pre-existing data-loss). Deferred: A1 "
        "(dataframe split), T2.2/T2.5/T2.6/T2.7 (tooling), U3, N6, "
        "R3, quantile stream loop.",
        self_path="scripts/patch_v0912_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.12-fix2 applied. v0.9.12 COMPLETE.")


if __name__ == "__main__":
    main()
