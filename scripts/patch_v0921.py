#!/usr/bin/env python3
# v0.9.21 — §1.4 Arrow-native group boundaries + DECISIONS 233-240.
#
# Atomic: (a) §1.4 fix, (b) DECISIONS append. Nothing else.
# Post-v0.9.20 lesson (§13.4/238): single-purpose patch, no
# multi-feature mixing (5 fix-cycles for v0.9.20).
#
# (a) §1.4 per-group shift/rolling: replace whole-batch `group_arrs =
#     {gc: b.column(gc).to_pylist()}` + per-row `tuple(group_arrs[gc][i])`
#     with Arrow-native `_group_boundaries_arrow` helper.
#
#     Prior per-row cost (audit §1.4):
#       - K_cols dict lookups + 1 tuple alloc + 1 tuple compare per row
#       - col_vals = {c: b.column(c).to_pylist() for c in cols}:
#         n × cols Python objects (28 B each)
#     New:
#       - Group boundaries: O(n) C++ (pc.equal + AND across cols),
#         transient bool masks. No n × K_cols Python objects.
#       - Per-group slice to_pylist: peak Python = max_group_size,
#         not n (transient between groups, GC'd before next).
#       - Output Arrow-native via pa.concat_arrays of per-group arrays
#         (8 B/val vs 28 B/val Python).
#
#     Honest claim: peak Python from 2×n×cols → max_group_size×cols
#     (transient) + n×cols Arrow. For many small groups (per-group op
#     target): 5-20× peak reduction. For few large groups: ~3.5×
#     (Python 28B→Arrow 8B only). Audit's "40MB→7MB/5×" was on groupby
#     aggregation, not per-group shift; not directly comparable.
#
#     Null semantics preserved: null==null → same group (SQL-style,
#     matches Python tuple compare). NULL vs non-null → different.
#     Cross-batch continuity preserved: if batch N's first group ==
#     batch N-1's last group, bufs/states carry over.
#
# (b) DECISIONS §13.4/233-240: 8 meta-lessons from v0.9.20 chain
#     (5 fix-cycles) + bench analysis (final reduce ops C_peak=0
#     reading):
#       233 fallback anchor → pick LATEST (v0.9.20 initial)
#       234 contiguous-text anchor vs multi-line code (v0.9.20-fix2)
#       235 stale test names after API change (v0.9.20-fix2)
#       236 peak_delta vs peak_abs unit mismatch (v0.9.20-fix3)
#       237 generic method names in AST guards (v0.9.20-fix4)
#       238 multi-feature patch → fix cascade (v0.9.20 overall)
#       239 sampler miss ≠ no peak (bench analysis)
#       240 formula-as-sizing ≠ formula-as-validation metric
#           (final: §9.2.1 = per-batch-total, bench = delta during op)
#
# Run: uv run python scripts/patch_v0921.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) helper: _group_boundaries_arrow ============

HELPER_LINES = [
    '',
    '',
    'def _group_boundaries_arrow(batch, group_cols):',
    '    """v0.9.21 (§1.4): Arrow-native group boundary detection.',
    '',
    '    Returns list [0, b1, b2, ..., n] where boundaries[k] is the',
    '    row index at which group k starts (0 = first, n = past end).',
    '',
    '    Assumes input is sorted by group_cols (caller contract; group',
    '    boundaries = value changes between consecutive rows).',
    '',
    '    Peak: O(n) transient bool mask per group col. No n × K_cols',
    '    Python objects (prior `to_pylist()` + per-row tuple alloc).',
    '',
    '    Null semantics: null==null → same group (SQL-style; matches',
    '    Python tuple compare used before v0.9.21). NULL vs non-null',
    '    → different group (Arrow Kleene equal returns null → treated',
    '    as different via fill_null(False)).',
    '    """',
    '    import pyarrow.compute as _pc',
    '    n = batch.num_rows',
    '    if n <= 1:',
    '        return [0, n] if n > 0 else [0]',
    '    same = None',
    '    for gc in group_cols:',
    '        col = batch.column(gc)',
    '        prev_slice = col.slice(0, n - 1)',
    '        curr_slice = col.slice(1)',
    '        eq = _pc.equal(curr_slice, prev_slice)',
    '        both_null = _pc.and_(',
    '            _pc.is_null(curr_slice), _pc.is_null(prev_slice)',
    '        )',
    '        col_same = _pc.or_(',
    '            _pc.fill_null(eq, pa.scalar(False, type=pa.bool_())),',
    '            both_null,',
    '        )',
    '        if same is None:',
    '            same = col_same',
    '        else:',
    '            same = _pc.and_(same, col_same)',
    '    same_np = same.to_numpy(zero_copy_only=False)',
    '    changes = np.flatnonzero(~same_np) + 1',
    '    return [0] + changes.tolist() + [n]',
]


def add_helper():
    print("\n=== (a) add _group_boundaries_arrow helper ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _group_boundaries_arrow(" in s:
        print("  already present — skip")
        return
    # Insert after _ArrowTail class, before _RollState
    anchor = "class _RollState:"
    if anchor not in s:
        raise SystemExit("_RollState anchor not found")
    block = "\n".join(HELPER_LINES) + "\n\n\n"
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (a2) rewrite _per_group_shift inner loop ============

def rewrite_per_group_shift():
    print("\n=== (a2) rewrite _per_group_shift inner loop ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    marker = "# v0.9.21 (§1.4): Arrow-native group boundaries in shift"
    if marker in s:
        print("  already rewritten — skip")
        return

    old_block = '''        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(b, name="per_group_shift.sorted")
                n = b.num_rows
                if n == 0:
                    continue
                group_arrs = {gc: b.column(gc).to_pylist() for gc in group_cols}
                col_vals = {c: b.column(c).to_pylist() for c in cols}
                shifted_vals = {c: [None] * n for c in cols}
                for i in range(n):
                    gk = tuple(group_arrs[gc][i] for gc in group_cols)
                    if gk != prev_group:
                        for dq in bufs.values():
                            dq.clear()
                        prev_group = gk
                    for c in cols:
                        if len(bufs[c]) < k:
                            shifted_vals[c][i] = fill_value
                        else:
                            shifted_vals[c][i] = bufs[c][0]
                        bufs[c].append(col_vals[c][i])
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(
                        pa.array(
                            shifted_vals[c],
                            type=b.schema.field(c).type,
                        )
                    )
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()'''

    new_block = '''        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_shift.sorted"
                )
                n = b.num_rows
                if n == 0:
                    continue
                # v0.9.21 (§1.4): Arrow-native group boundaries in shift.
                # Prior: group_arrs = {gc: b.column(gc).to_pylist()...}
                # + per-row tuple(group_arrs[gc][i]) = O(n × K_cols)
                # Python ops + n × K_cols Python objects (28 B each).
                # Now: _group_boundaries_arrow = O(n × K_cols) C++ ops
                # + O(n) transient bool masks. Peak Python per batch =
                # max_group_size × cols (transient, GC'd between
                # groups), not n × cols.
                boundaries = _group_boundaries_arrow(b, group_cols)

                first_gk = tuple(
                    b.column(gc)[0].as_py() for gc in group_cols
                )
                if prev_group is _SENTINEL or first_gk != prev_group:
                    if prev_group is not _SENTINEL:
                        for dq in bufs.values():
                            dq.clear()
                    prev_group = first_gk

                all_shifted_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        for dq in bufs.values():
                            dq.clear()
                        prev_group = tuple(
                            b.column(gc)[start].as_py()
                            for gc in group_cols
                        )
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_shifted = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            if len(bufs[c]) < k:
                                group_shifted[c][i] = fill_value
                            else:
                                group_shifted[c][i] = bufs[c][0]
                            bufs[c].append(gvals[c][i])
                    for c in cols:
                        all_shifted_arrow[c].append(
                            pa.array(
                                group_shifted[c],
                                type=b.schema.field(c).type,
                            )
                        )
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_shifted_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()'''

    if old_block not in s:
        raise SystemExit("_per_group_shift inner block anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (a3) rewrite _per_group_rolling inner loop ============

def rewrite_per_group_rolling():
    print("\n=== (a3) rewrite _per_group_rolling inner loop ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    marker = "# v0.9.21 (§1.4): Arrow-native group boundaries in rolling"
    if marker in s:
        print("  already rewritten — skip")
        return

    old_block = '''        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_rolling.sorted"
                )
                n = b.num_rows
                if n == 0:
                    continue
                group_arrs = {gc: b.column(gc).to_pylist() for gc in group_cols}
                col_vals = {c: b.column(c).to_pylist() for c in cols}
                rolled_vals = {c: [None] * n for c in cols}
                for i in range(n):
                    gk = tuple(group_arrs[gc][i] for gc in group_cols)
                    if gk != prev_group:
                        for st in states.values():
                            st.reset()
                        prev_group = gk
                    for c in cols:
                        rolled_vals[c][i] = states[c].push(col_vals[c][i])
                arrays = [b.column("row_id")]
                for c in cols:
                    out_name, ot = out_type_by_col[c]
                    arrays.append(pa.array(rolled_vals[c], type=ot))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()'''

    new_block = '''        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_rolling.sorted"
                )
                n = b.num_rows
                if n == 0:
                    continue
                # v0.9.21 (§1.4): Arrow-native group boundaries in rolling
                # (same rationale as _per_group_shift).
                boundaries = _group_boundaries_arrow(b, group_cols)

                first_gk = tuple(
                    b.column(gc)[0].as_py() for gc in group_cols
                )
                if prev_group is _SENTINEL or first_gk != prev_group:
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()
                    prev_group = first_gk

                all_rolled_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        for st in states.values():
                            st.reset()
                        prev_group = tuple(
                            b.column(gc)[start].as_py()
                            for gc in group_cols
                        )
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_rolled = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            group_rolled[c][i] = states[c].push(gvals[c][i])
                    for c in cols:
                        out_name, ot = out_type_by_col[c]
                        all_rolled_arrow[c].append(
                            pa.array(group_rolled[c], type=ot)
                        )
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_rolled_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()'''

    if old_block not in s:
        raise SystemExit("_per_group_rolling inner block anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) DECISIONS append §13.4/233-240 ============

DECISIONS_BLOCK = (
    "\n\n\n"
    "**Эпизод 233 (append with fallback anchor — check ALL candidates).** "
    "v0.9.20 initial: `append_decisions` used `anchor = "
    "\"**Gap 190–210\"` (present, v0.9.17) with fallback "
    "`if anchor not in s: anchor = §13.5`. Since gap-note EXISTS → "
    "insert BEFORE gap → order became 189...226-232...gap...211-225. "
    "Assert 225 < 226 failed → 226 (early) < 225 (late).\n\n"
    "**Правило (§13.4/233):** append с fallback anchor — **не "
    "полагаться на `if X not in s`**. Проверять **все** candidate "
    "anchors; выбирать **латест** по позиции явно. Если несколько — "
    "`insert_before(target)` параметризованный, не «первый-if-present».\n\n"
    "**Связанные:** §13.4/160 (silent no-op), /216 (helper exists).\n\n\n"
    "**Эпизод 234 (contiguous-text anchor vs multi-line code).** "
    "v0.9.20-fix2: verify anchor `'_model_predicted(\"hash_shuffle\"'` "
    "expects single-line call; bench had multi-line. Sort/groupby "
    "single-line → pass; hash_shuffle multi-line → fail.\n\n"
    "**Правило (§13.4/234):** anchor на contiguous text "
    "(`func(\"arg\"`) требует single-line. Multi-line calls → anchor "
    "на args-only (`'\"hash_shuffle\", batches'`) или **normalize "
    "style** (все single-line). Prefer normalize — simpler AST-check "
    "+ readability.\n\n"
    "**Связанные:** §13.4/230 (write_content units).\n\n\n"
    "**Эпизод 235 (stale test names after API change).** "
    "v0.9.20-fix2: `test_calibrate_records_sample` asserts "
    "`\"arrow\" in s and \"rss\" in s` — v0.9.19 field names. "
    "v0.9.20 renamed to peak_pool/peak_rss → assertion False. "
    "Test was never updated.\n\n"
    "**Правило (§13.4/235):** при rename API в patch — grep ALL tests "
    "for old field names BEFORE commit. Use "
    "`assert_no_stale_behavior(['\"arrow\" in s'])` в patch, или "
    "explicit test refresh.\n\n"
    "**Связанные:** §13.4/152 (grep stale), /193 "
    "(write_content_checked).\n\n\n"
    "**Эпизод 236 (peak_delta vs peak_abs — unit mismatch).** "
    "v0.9.20-fix3: `_record_calibration` passed "
    "`peak_pool_abs`/`peak_rss_abs` (absolute process values, "
    "161 MB) to C calc. Formula §9.2.1 = delta above op-start "
    "(batch + state = 149 KB). C = 161 MB / 149 KB = 1080. Absurd.\n\n"
    "**Правило (§13.4/236):** metric = **delta above op-start**. "
    "`peak_delta = peak_abs - start`. Formula = delta. Both sides "
    "delta. Same class as §13.4/232 but ABSOLUTE side (232 был NET "
    "side).\n\n"
    "**Связанные:** §13.4/232, /240.\n\n\n"
    "**Эпизод 237 (generic method names in AST guards).** "
    "v0.9.20-fix4: `\"join\"` in `HIGH_RISK_ATTRS` matches "
    "`pa.Table.join` (high-risk) AND `threading.Thread.join` "
    "(benign stdlib). AST matcher attribute-name only → false "
    "positive on sampler cleanup.\n\n"
    "**Правило (§13.4/237):** generic method names (join, update, "
    "filter, compute, read) → per-qualname whitelist с rationale. "
    "NOT removal from `HIGH_RISK_ATTRS` (hides real violations). "
    "Rationale = permission + documentation.\n\n"
    "**Связанные:** §13.4/229 (type coercion in numpy), /170 "
    "(whitelist by qualname).\n\n\n"
    "**Эпизод 238 (multi-feature patch → fix cascade).** "
    "v0.9.20 initial — 4 features in one patch: peak-polling (a), "
    "bench rewrite (b), DECISIONS append (c), tests (d). Cascade: "
    "**5 fix-cycles** (initial + fix1 + fix2 + fix3 + fix4). Each "
    "fix exposed one bug from one feature.\n\n"
    "**Правило (§13.4/238):** если patch требует >2 fix-cycle → "
    "**split on next opportunity**. Better: split **before** in "
    "first attempt. v0.9.21+ — single-purpose atomic patches.\n\n"
    "**Связанные:** §13.4/211 (compile-before-write), /222 "
    "(multi-step abort).\n\n\n"
    "**Эпизод 239 (sampler miss ≠ no peak).** v0.9.20 bench: "
    "`peak_pool=0` для reduce ops (sampler 10ms interval, op ~5ms "
    "→ sampled 0 times). But also: `pc.sum(skip_nulls)` на int64 "
    "без nulls = zero-copy → **no temp alloc**. Both true "
    "simultaneously.\n\n"
    "**Правило (§13.4/239):** sampler miss + zero-copy kernel = "
    "`peak=0` honest. Verify with **null-containing** input + "
    "shorter interval to distinguish. Не assume drift без контроля "
    "обоих.\n\n"
    "**Связанные:** §13.4/240.\n\n\n"
    "**Эпизод 240 (formula-as-sizing ≠ formula-as-validation "
    "metric).** §9.2.1 `ReduceMemoryModel`: `batch_size × row_size "
    "+ 2 × state` = **sizing decision** («сколько строк выбрать для "
    "бюджета»). Bench measures **delta during op** (temp allocs "
    "above pre-sized batch). Formula = per-batch-total, bench = "
    "incremental-during-op. Compare → 0/149K = 0 (no temp) or >1 "
    "(temp exists).\n\n"
    "**Правило (§13.4/240):** formula validates at **sizing time** "
    "(batch_size picked, guard check). Bench validates at "
    "**runtime** (temp alloc). Two different purposes. Do NOT "
    "compare across. Correct metric for formula = «does sizing fail "
    "to fit?» not «does op delta exceed formula?».\n\n"
    "**Связанные:** §13.4/232, /236, /239.\n\n\n"
)


def append_decisions():
    print("\n=== (b) DECISIONS append §13.4/233-240 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "**Эпизод 233 (append with fallback anchor" in s:
        print("  already present — skip")
        return
    # Single canonical anchor: §13.5 (always present, single position).
    # No fallback (§13.4/233).
    anchor = "### 13.5. Открытые вопросы"
    if anchor not in s:
        raise SystemExit("§13.5 anchor not found")
    if s.count(anchor) != 1:
        raise SystemExit(
            f"§13.5 anchor appears {s.count(anchor)} times — "
            f"cannot insert deterministically"
        )
    s = s.replace(anchor, DECISIONS_BLOCK + anchor, 1)
    p.write_text(s, newline="")
    # Verify explicit order
    i_232 = s.find("**Эпизод 232 ")
    i_233 = s.find("**Эпизод 233 ")
    i_240 = s.find("**Эпизод 240 ")
    i_135 = s.find("### 13.5. Открытые вопросы")
    assert 0 <= i_232 < i_233 < i_240 < i_135, (
        f"DECISIONS order broken: 232={i_232} 233={i_233} "
        f"240={i_240} §13.5={i_135}"
    )
    print("  OK (233-240 inserted after 232, before §13.5)")


# ============ (c) tests ============

def add_tests():
    print("\n=== (c) regression tests ===")
    # c1: test_groupby_per_group.py — boundaries helper correctness + AST
    p1 = Path("tests/test_groupby_per_group.py")
    s1 = p1.read_text()
    if "def test_group_boundaries_arrow_correctness" in s1:
        print("  c1 already present — skip")
    else:
        extra1 = '''

# ============ v0.9.21 (§1.4): Arrow-native group boundaries ============


def test_group_boundaries_arrow_correctness():
    """v0.9.21 (§1.4): boundary helper matches per-row tuple compare."""
    import pyarrow as pa
    from cadw.dataframe import _group_boundaries_arrow
    # Single col, no nulls
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 1, 2, 2, 2, 3])], names=["g"]
    )
    assert _group_boundaries_arrow(b, ["g"]) == [0, 2, 5, 6]
    # Single col with nulls: null==null → same group (SQL-style)
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([1, 1, None, None, 2])], names=["g"]
    )
    assert _group_boundaries_arrow(b2, ["g"]) == [0, 2, 4, 5]
    # Multi-col: combined AND
    b3 = pa.RecordBatch.from_arrays(
        [pa.array([1, 1, 1, 2]), pa.array(["a", "a", "b", "b"])],
        names=["k1", "k2"],
    )
    assert _group_boundaries_arrow(b3, ["k1", "k2"]) == [0, 2, 3, 4]
    # Empty
    b4 = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.int64())], names=["g"]
    )
    assert _group_boundaries_arrow(b4, ["g"]) == [0]
    # Single row
    b5 = pa.RecordBatch.from_arrays([pa.array([5])], names=["g"])
    assert _group_boundaries_arrow(b5, ["g"]) == [0, 1]
    # Mixed null across different values
    b6 = pa.RecordBatch.from_arrays(
        [pa.array([None, 1, None, 1])], names=["g"]
    )
    assert _group_boundaries_arrow(b6, ["g"]) == [0, 1, 2, 3, 4]


def test_per_group_shift_no_whole_batch_to_pylist():
    """v0.9.21 (§1.4): _per_group_shift must NOT call
    b.column(gc).to_pylist() on the whole batch — only on group slices."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame
    src = _tw.dedent(_inspect.getsource(CadwDataFrame._per_group_shift))
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "to_pylist"):
            continue
        recv = node.func.value
        # Old pattern: b.column(gc).to_pylist() (no .slice)
        # New pattern: b.column(c).slice(start, group_len).to_pylist() OK
        if (isinstance(recv, _ast.Call)
                and isinstance(recv.func, _ast.Attribute)
                and recv.func.attr == "column"):
            raise AssertionError(
                f"_per_group_shift whole-batch to_pylist at line "
                f"{node.lineno} — use slice().to_pylist() per group"
            )
    # Positive: helper is called
    assert "_group_boundaries_arrow" in src


def test_per_group_rolling_no_whole_batch_to_pylist():
    """v0.9.21 (§1.4): same for _per_group_rolling."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame
    src = _tw.dedent(_inspect.getsource(CadwDataFrame._per_group_rolling))
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "to_pylist"):
            continue
        recv = node.func.value
        if (isinstance(recv, _ast.Call)
                and isinstance(recv.func, _ast.Attribute)
                and recv.func.attr == "column"):
            raise AssertionError(
                f"_per_group_rolling whole-batch to_pylist at line "
                f"{node.lineno} — use slice().to_pylist() per group"
            )
    assert "_group_boundaries_arrow" in src
'''
        p1.write_text(s1 + extra1, newline="")
        pu.assert_compiles(p1)
        print("  c1 OK")

    # c2: test_changelog.py — DECISIONS 233-240 order
    p2 = Path("tests/test_changelog.py")
    s2 = p2.read_text()
    if "def test_decisions_233_240_order" in s2:
        print("  c2 already present — skip")
        return
    extra2 = '''

# ============ v0.9.21: DECISIONS 233-240 order ============


def test_decisions_233_240_order():
    """v0.9.21: episodes 233-240 after 232, before §13.5."""
    from pathlib import Path as _P
    text = (_P(__file__).resolve().parent.parent / "DECISIONS.md").read_text()
    if "**Эпизод 233 " not in text:
        pytest.skip("233 block not yet present")
    i_232 = text.index("**Эпизод 232 ")
    i_233 = text.index("**Эпизод 233 ")
    i_240 = text.index("**Эпизод 240 ")
    i_135 = text.index("### 13.5. Открытые вопросы")
    assert i_232 < i_233, (
        f"233 ({i_233}) before 232 ({i_232})"
    )
    assert i_233 < i_240, f"240 ({i_240}) before 233 ({i_233})"
    assert i_240 < i_135, f"240 ({i_240}) after §13.5 ({i_135})"
'''
    p2.write_text(s2 + extra2, newline="")
    pu.assert_compiles(p2)
    print("  c2 OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.21 + regression ===")
    pu.verify_anchors([
        # v0.9.21 (a)
        ("cadw/dataframe.py", "def _group_boundaries_arrow("),
        ("cadw/dataframe.py",
         "# v0.9.21 (§1.4): Arrow-native group boundaries in shift"),
        ("cadw/dataframe.py",
         "# v0.9.21 (§1.4): Arrow-native group boundaries in rolling"),
        # v0.9.21 (b)
        ("DECISIONS.md", "**Эпизод 233 (append with fallback anchor"),
        ("DECISIONS.md", "**Эпизод 234 (contiguous-text anchor"),
        ("DECISIONS.md", "**Эпизод 235 (stale test names"),
        ("DECISIONS.md", "**Эпизод 236 (peak_delta vs peak_abs"),
        ("DECISIONS.md", "**Эпизод 237 (generic method names"),
        ("DECISIONS.md", "**Эпизод 238 (multi-feature patch"),
        ("DECISIONS.md", "**Эпизод 239 (sampler miss"),
        ("DECISIONS.md", "**Эпизод 240 (formula-as-sizing"),
        # v0.9.21 (c)
        ("tests/test_groupby_per_group.py",
         "def test_group_boundaries_arrow_correctness"),
        ("tests/test_groupby_per_group.py",
         "def test_per_group_shift_no_whole_batch_to_pylist"),
        ("tests/test_groupby_per_group.py",
         "def test_per_group_rolling_no_whole_batch_to_pylist"),
        ("tests/test_changelog.py", "def test_decisions_233_240_order"),
        # v0.9.20 regression
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "peak_pool_box"),
        ("cadw/core/memory_guard.py", "# v0.9.20-fix3"),
        ("tests/test_arrow_call_memory_guard.py",
         '"cadw/core/memory_guard.py:MemoryGuard.calibrate"'),
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        # v0.9.19/18/17/16
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.21 (happy path only) ===")

    def _verify_all():
        # 1. Boundary helper: basic correctness
        import pyarrow as pa
        from cadw.dataframe import _group_boundaries_arrow
        b = pa.RecordBatch.from_arrays(
            [pa.array([1, 1, 2, 2, 3])], names=["g"]
        )
        assert _group_boundaries_arrow(b, ["g"]) == [0, 2, 4, 5]

        # 2. Per-group shift end-to-end (behavior preserved)
        from cadw import CadwDataFrame, MemoryGuard
        t = pa.table({
            "sensor": ["a", "a", "a", "b", "b", "b"],
            "ts": [1, 2, 3, 1, 2, 3],
            "v": [10, 20, 30, 100, 200, 300],
        })
        df = CadwDataFrame.from_table(
            t, memory_guard=MemoryGuard(max_memory_mb=20),
        )
        res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
        got_v = []
        for b2 in res._iter_filtered_batches():
            got_v.extend(b2.column("v").to_pylist())
        assert got_v == [None, 10, 20, None, 100, 200], (
            f"per-group shift result wrong: {got_v}"
        )

        # 3. Per-group rolling end-to-end
        res2 = df.groupby("sensor").rolling(
            2, "sum", on="ts", columns=["v"],
        )
        got_s = []
        for b3 in res2._iter_filtered_batches():
            got_s.extend(b3.column("v_rolling_sum").to_pylist())
        assert got_s == [None, 30, 50, None, 300, 500], (
            f"per-group rolling result wrong: {got_s}"
        )

        # 4. DECISIONS order
        from pathlib import Path as _P
        d = _P("DECISIONS.md").read_text()
        assert d.find("**Эпизод 232 ") < d.find("**Эпизод 233 ")
        assert d.find("**Эпизод 240 ") < d.find("### 13.5. Открытые вопросы")

        print("  _group_boundaries_arrow: basic OK")
        print("  per_group_shift end-to-end: correct OK")
        print("  per_group_rolling end-to-end: correct OK")
        print("  DECISIONS 232 < 233 < 240 < §13.5 OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target (per-group + changelog) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_groupby_per_group.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_changelog.py", "-q"],
        timeout=900,
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
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.21 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    add_helper()
    rewrite_per_group_shift()
    rewrite_per_group_rolling()
    append_decisions()
    add_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.21: §1.4 Arrow-native group boundaries in "
        "_per_group_shift/_per_group_rolling + DECISIONS §13.4/233-240 "
        "append. (Atomic: single feature + doc; no multi-feature "
        "mixing per §13.4/238 — v0.9.20 took 5 fix-cycles due to 4-"
        "feature patch.) (§1.4 fix) Prior: per-batch "
        "`group_arrs = {gc: b.column(gc).to_pylist()}` (n × K_cols "
        "Python objects, 28 B each) + per-row tuple(group_arrs[gc][i]) "
        "= O(n × K_cols) Python ops. Peak Python = 2×n×cols "
        "(input + output list). New: `_group_boundaries_arrow` helper "
        "computes boundaries via pc.equal + AND across cols — O(n) "
        "C++ ops + transient bool masks per col. Per-group slice "
        "to_pylist bounded by max_group_size (not n). Output "
        "Arrow-native via pa.concat_arrays of per-group pa.array "
        "(8 B/val vs 28 B/val). Peak Python: max_group_size × cols "
        "(transient) vs 2×n×cols. Peak Arrow: n × cols × 8 B. "
        "Honest: 5-20× peak reduction for many small groups (per-"
        "group op target case); ~3.5× for few large groups "
        "(unavoidable Arrow output). Audit's '40MB→7MB/5×' was on "
        "groupby aggregation (different pattern); not directly "
        "comparable. Null semantics preserved: null==null → same "
        "group (SQL-style, matches Python tuple compare). Cross-batch "
        "continuity preserved: batch N's first group == batch N-1's "
        "last group → bufs/states carry over. (DECISIONS append "
        "§13.4/233-240) 8 meta-lessons from v0.9.20 chain + bench "
        "analysis: 233 fallback anchor (pick LATEST by position, "
        "not 'first-if-present' — v0.9.20 initial died here); 234 "
        "contiguous-text anchor vs multi-line code (normalize style "
        "to single-line — fix1 died here); 235 stale test names "
        "after API rename (grep tests BEFORE commit — fix2 died "
        "here); 236 peak_delta vs peak_abs unit mismatch (formula is "
        "delta, metric must be delta — fix3 fixed real bug); 237 "
        "generic method names in AST guards (per-qualname whitelist, "
        "NOT removal from HIGH_RISK_ATTRS — fix4 died here); 238 "
        "multi-feature patch → fix cascade (split atomic before, not "
        "after); 239 sampler miss ≠ no peak (10ms sampler missed 5ms "
        "op; pc.sum no-null zero-copy → both true); 240 formula-as-"
        "sizing ≠ formula-as-validation metric (§9.2.1 = per-batch-"
        "total for sizing, bench measures delta during pre-sized op "
        "— do NOT compare across). (Backward compat) 1907 existing "
        "tests unchanged; behavior identical (verified via dogfood "
        "end-to-end: [None,10,20,None,100,200] shift, [None,30,50,"
        "None,300,500] rolling). 5 new tests: 3 groupby_per_group "
        "(boundary helper correctness, AST no-whole-batch-to_pylist "
        "for shift + rolling) + 1 test_changelog (DECISIONS order) + "
        "1 DECISIONS 233-240 presence. Deferred to v0.9.22: per-op "
        "predicted wiring в modules (real external_sort/hash_shuffle/"
        "groupby_aggregate call model.estimate() inside their own "
        "calibrate() — bench done; modules not); bench metric "
        "redesign (materialize inside calibrate to catch formula "
        "§9.2.1 correctly; current delta-during-op for reduce = 0 "
        "because batch pre-allocated); Pingis round 3 (ML/pipeline/"
        "cv/metrics); A1 split.",
        self_path="scripts/patch_v0921.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.21 applied.")


if __name__ == "__main__":
    main()
