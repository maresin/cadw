#!/usr/bin/env python3
"""v0.8.11b: strict join conditional + schema-order audit.

STRICT JOIN (v0.8.11b):
  merge() enforcement of `40 MB + 5 × predicted_output <= state_limit`.
  v0.8.10-pre: warn only.
  v0.8.11b:   strict when budget >= 160 MB (= 4 × overhead),
              warn when below (backward compat with mem_mb=20 tests).

  Threshold rationale (§9.2.6):
    - budget < 160 MB: 40 MB overhead > 25% budget → check would fire
      trivially for tiny joins → warn only, per-batch guards remain.
    - budget >= 160 MB: overhead <= 25% budget → strict, real guarantee.
  Secondary `state_limit >= 40 MB` check protects against reservation
  drop (available_bytes shrinks → trivially-fired raise). Warn instead.

SCHEMA-ORDER AUDIT (from §13.4/145 backlog):
  Verified 4 call sites of `Table.from_batches(gen, schema=...)`:
    - join._safe_materialize_partition: OK (explicit out_schema)
    - sort._write_run: OK (table.schema preserved through take)
    - groupby._read_partition_safe: OK (no .select on batches)
    - dedup._local_dedup: fixed v0.8.11a-fix3
  No additional bugs. 4 AST guards lock the invariant.

CHANGES:
  A. cadw/core/memory_guard.py: public `state_limit_bytes` property.
  B. cadw/ops/join.py: `_JOIN_STRICT_BUDGET_THRESHOLD_BYTES = 4 * overhead`.
  C. cadw/dataframe.py::merge: strict_mode branch (raise vs warn).
  D. tests/test_join_rss_warning.py: replace 2 AST tests (Try-based →
     strict_mode-based).
  E. tests/test_join_strict.py: 3 tests (threshold, raise, pass).
  F. tests/test_schema_order_audit.py: 4 AST guards.
  G. CHANGELOG [Unreleased]: append v0.8.11b.
  H. DECISIONS §9.2.6 + §11.5 + §13.4/145 update.

GUARDS (before commit, §13.4/140+143):
  test_join_strict, test_join_rss_warning (updated),
  test_schema_order_audit, test_changelog, test_arrow_call_memory_guard,
  test_join + e2e + streaming regression, fast + slow.

Run: uv run python scripts/patch_v0811b_strict_join.py
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


# ============ A. MemoryGuard.state_limit_bytes ============

def add_state_limit_property():
    print("\n=== cadw/core/memory_guard.py: state_limit_bytes property ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "def state_limit_bytes(" in s:
        print("  already present — skip")
        return

    old = '''    @property
    def budget_bytes(self) -> int:
        return self._budget

    def available_bytes(self) -> int:
        return max(0, self._budget - self._reserved)'''

    new = '''    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def state_limit_bytes(self) -> int:
        """Current state_size limit: `available_bytes() × state_safety`.

        Public since v0.8.11b — callers (e.g., `merge()` strict-mode
        decision) need to compare against state_limit without going
        through `check_state` (which raises).
        """
        return self._state_limit()

    def available_bytes(self) -> int:
        return max(0, self._budget - self._reserved)'''

    s = replace_line_block(s, old, new, tag="state-limit-prop")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ B. _JOIN_STRICT_BUDGET_THRESHOLD_BYTES ============

def add_strict_threshold_const():
    print("\n=== cadw/ops/join.py: _JOIN_STRICT_BUDGET_THRESHOLD_BYTES ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()
    if "_JOIN_STRICT_BUDGET_THRESHOLD_BYTES" in s:
        print("  already present — skip")
        return

    anchor = "_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024\n"
    if anchor not in s:
        raise SystemExit("_JOIN_FIXED_OVERHEAD_BYTES anchor not found")

    add = (
        anchor
        + '''
# v0.8.11b: strict-mode threshold for merge() whole-operation RSS check.
# Below threshold: warn only (fixed 40 MB overhead dominates small
# budgets; strict check would trivially fire for tiny joins).
# At/above: raise MemoryLimitExceeded (real guarantee).
# Secondary condition in merge(): state_limit must also be >=
# _JOIN_FIXED_OVERHEAD_BYTES (protects against reservation drop).
# See §9.2.6.
_JOIN_STRICT_BUDGET_THRESHOLD_BYTES = 4 * _JOIN_FIXED_OVERHEAD_BYTES
'''
    )
    s = s.replace(anchor, add, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ C. merge() strict_mode branch ============

def patch_merge_strict_branch():
    print("\n=== cadw/dataframe.py::merge — strict_mode branch ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "_JOIN_STRICT_BUDGET_THRESHOLD_BYTES" in s:
        print("  already present — skip")
        return

    old = '''                try:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    from cadw.ops.join import _JOIN_FIXED_OVERHEAD_BYTES
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                    )
                    total_rss_estimate = (
                        _JOIN_FIXED_OVERHEAD_BYTES + 5 * predicted_output
                    )
                    self._memory_guard.check_state(
                        total_rss_estimate,
                        name="join.whole_operation_rss",
                    )
                except MemoryLimitExceeded:
                    logger.warning(
                        "merge: predicted whole-operation RSS delta "
                        "~%.0f MiB (output %.1f MiB + %.0f MiB Arrow pool) "
                        "exceeds state_limit. Per-batch guards will still "
                        "enforce; actual peak may exceed budget. Bump "
                        "MemoryGuard budget if possible. See §9.2.6.",
                        total_rss_estimate / (1024 * 1024),
                        predicted_output / (1024 * 1024),
                        _JOIN_FIXED_OVERHEAD_BYTES / (1024 * 1024),
                    )'''

    new = '''                from cadw.ops.join import (
                    _JOIN_FIXED_OVERHEAD_BYTES,
                    _JOIN_STRICT_BUDGET_THRESHOLD_BYTES,
                )
                predicted_output = (
                    left_chars.n_rows
                    * params.avg_matches
                    * params.output_row_size
                )
                total_rss_estimate = (
                    _JOIN_FIXED_OVERHEAD_BYTES + 5 * predicted_output
                )
                # v0.8.11b: strict when budget is large enough that the
                # fixed 40 MB overhead is meaningful. Threshold:
                # budget >= 4×overhead (160 MB) AND
                # state_limit >= overhead (available >= 2×overhead).
                # Below: warn only — raise would fire trivially for
                # tiny joins (40 MB overhead >> state_limit at small
                # budgets). See §9.2.6.
                state_limit = self._memory_guard.state_limit_bytes
                budget = self._memory_guard.budget_bytes
                strict_mode = (
                    budget >= _JOIN_STRICT_BUDGET_THRESHOLD_BYTES
                    and state_limit >= _JOIN_FIXED_OVERHEAD_BYTES
                )
                if strict_mode:
                    self._memory_guard.check_state(
                        total_rss_estimate,
                        name="join.whole_operation_rss",
                    )
                elif total_rss_estimate > state_limit:
                    logger.warning(
                        "merge: predicted whole-operation RSS delta "
                        "~%.0f MiB (output %.1f MiB + %.0f MiB Arrow "
                        "pool) exceeds state_limit (%.0f MiB) but "
                        "budget %.0f MiB < %.0f MiB strict threshold. "
                        "Per-batch guards still enforce; actual peak "
                        "may exceed budget. Bump MemoryGuard budget "
                        "to enable strict enforcement. See §9.2.6.",
                        total_rss_estimate / (1024 * 1024),
                        predicted_output / (1024 * 1024),
                        _JOIN_FIXED_OVERHEAD_BYTES / (1024 * 1024),
                        state_limit / (1024 * 1024),
                        budget / (1024 * 1024),
                        _JOIN_STRICT_BUDGET_THRESHOLD_BYTES / (1024 * 1024),
                    )'''

    s = replace_line_block(s, old, new, tag="merge-strict-branch")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ D. Update existing AST tests in test_join_rss_warning ============

def update_existing_ast_tests():
    print("\n=== tests/test_join_rss_warning.py: replace 2 AST tests ===")
    p = Path("tests/test_join_rss_warning.py")
    s = p.read_text()
    if "test_merge_ast_uses_strict_mode" in s:
        print("  already present — skip")
        return

    old_block = '''def test_merge_ast_guard_warning_present():
    """AST: merge() contains a try/except around whole_operation_rss check."""
    import ast
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod.CadwDataFrame.merge)
    tree = ast.parse(inspect.cleandoc(src))

    found_try = False
    found_check = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            unparsed = ast.unparse(node)
            if "join.whole_operation_rss" in unparsed:
                found_try = True
                found_check = True
                break
    assert found_try, (
        "merge() must have try/except around whole_operation_rss "
        "check_state (v0.8.10-pre warning pattern)"
    )


def test_merge_ast_does_not_raise_whole_op():
    """AST: whole_operation_rss must be inside try (warning), not bare."""
    import ast
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod.CadwDataFrame.merge)
    tree = ast.parse(inspect.cleandoc(src))

    # Every `check_state(..., name="join.whole_operation_rss")` must have
    # a parent Try node.
    def _find_parent_try(target: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if child is target:
                        return True
        return False

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "check_state"):
            unparsed = ast.unparse(node)
            if "join.whole_operation_rss" in unparsed:
                assert _find_parent_try(node), (
                    "check_state(name='join.whole_operation_rss') must be "
                    "inside try (warn, not raise)"
                )'''

    new_block = '''def test_merge_ast_uses_strict_mode():
    """AST: merge() references strict-mode threshold (v0.8.11b).

    Replaces v0.8.10-pre `test_merge_ast_guard_warning_present`
    (try/except pattern) — v0.8.11b uses explicit if/elif branch on
    `strict_mode` instead of try/except.
    """
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod.CadwDataFrame.merge)
    assert "_JOIN_STRICT_BUDGET_THRESHOLD_BYTES" in src, (
        "v0.8.11b: merge() must reference _JOIN_STRICT_BUDGET_THRESHOLD_BYTES"
    )
    assert "strict_mode" in src, (
        "v0.8.11b: merge() must have strict_mode branch"
    )


def test_merge_ast_check_state_guard_name():
    """AST: check_state uses name='join.whole_operation_rss'.

    Replaces v0.8.10-pre `test_merge_ast_does_not_raise_whole_op` —
    v0.8.11b DOES raise (in strict mode), so the guard name is the
    stable marker across both warn/strict modes.
    """
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod.CadwDataFrame.merge)
    assert 'name="join.whole_operation_rss"' in src, (
        "v0.8.11b: check_state must use guard name 'join.whole_operation_rss'"
    )'''

    s = replace_line_block(s, old_block, new_block, tag="ast-tests-strict")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ E. tests/test_join_strict.py ============

STRICT_TESTS = '''"""v0.8.11b: strict join enforcement (budget >= 160 MB → raise)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, name, data, chunk=3, mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_strict_budget_threshold_constant():
    """Threshold = 4 × overhead (160 MB)."""
    from cadw.ops.join import (
        _JOIN_FIXED_OVERHEAD_BYTES,
        _JOIN_STRICT_BUDGET_THRESHOLD_BYTES,
    )
    assert _JOIN_STRICT_BUDGET_THRESHOLD_BYTES == 4 * _JOIN_FIXED_OVERHEAD_BYTES
    assert _JOIN_STRICT_BUDGET_THRESHOLD_BYTES == 160 * 1024 * 1024


def test_state_limit_bytes_property():
    """MemoryGuard exposes state_limit_bytes = available × state_safety."""
    g = MemoryGuard(max_memory_mb=200)  # state_safety=0.5 default
    assert g.state_limit_bytes == 100 * 1024 * 1024


def test_strict_mode_raises_with_large_budget(tmp_path):
    """Budget 200 MB (>= 160 threshold), predicted_output=28.8 MB
    → total=184 MB > state_limit=100 MB → MemoryLimitExceeded.

    Data: 30K rows each side, K_true=100 per side (300 dupes/key).
    sample=3000 (chunk) > K_true → K_sample=100, K_est=100 × 30000/3000=1000.
    avg_matches = 30000 × 30000 / 1000 / 30000 = 30.
    predicted = 30000 × 30 × 32 (row_bytes_out) = 28.8 MB.
    """
    n = 30000
    k_true = 100
    left = make_df(
        tmp_path, "L",
        {"k": [i % k_true for i in range(n)], "a": list(range(n))},
        chunk=3000, mem_mb=200,
    )
    right = make_df(
        tmp_path, "R",
        {"k": [i % k_true for i in range(n)], "b": list(range(n))},
        chunk=3000, mem_mb=200,
    )
    with pytest.raises(MemoryLimitExceeded, match="whole_operation_rss"):
        left.merge(right, on="k")


def test_strict_mode_passes_small_join(tmp_path):
    """Budget 200 MB (strict mode), 1000 × 1000 PK join → predicted tiny,
    check passes, no raise."""
    left = make_df(
        tmp_path, "L",
        {"k": list(range(1000)), "a": list(range(1000))},
        chunk=100, mem_mb=200,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(1000)), "b": list(range(1000))},
        chunk=100, mem_mb=200,
    )
    result = left.merge(right, on="k")
    assert result.shape[0] == 1000
'''


def write_strict_tests():
    print("\n=== tests/test_join_strict.py ===")
    p = Path("tests/test_join_strict.py")
    p.write_text(STRICT_TESTS)
    compile_check(p)
    print("  OK")


# ============ F. tests/test_schema_order_audit.py ============

SCHEMA_AUDIT_TESTS = '''"""v0.8.11b: schema-order audit — verify no latent reorder bugs.

§13.4/145 (discovery of schema-order brittleness in _local_dedup)
prompted an audit of every `Table.from_batches(gen, schema=...)` in
`cadw/ops/`. This file documents the findings and guards against
regressions.

Findings:
- join._safe_materialize_partition: explicit `schema=out_schema` where
  out_schema order matches `.select(keep_cols)`. OK.
- sort._write_run: `ArrowIPCBatchWriter(table.schema)` — table.schema
  derived from pa.Table.from_batches, no reorder through take. OK.
- sort._merge_keys: explicit `schema=schema` in RecordBatch.from_arrays.
  OK (covered by sort tests, not re-asserted here).
- groupby._read_partition_safe: `_iter_batches` yields reader batches
  as-is (no `.select(...)`). schema=_schema matches reader exactly. OK.
- dedup._local_dedup: fixed in v0.8.11a-fix3 (explicit select_schema).

No additional bugs. Four AST guards below lock the invariants.
"""
import inspect

from cadw.ops import dedup as dedup_mod
from cadw.ops import groupby as gb_mod
from cadw.ops import join as join_mod
from cadw.ops import sort as sort_mod


def test_safe_materialize_passes_explicit_schema():
    """join._safe_materialize_partition: explicit `schema=out_schema`."""
    src = inspect.getsource(join_mod._safe_materialize_partition)
    assert "schema=out_schema" in src, (
        "_safe_materialize_partition must pass schema=out_schema explicitly "
        "(no reliance on first-batch order matching generator order)"
    )


def test_read_partition_safe_no_select_on_batches():
    """groupby._read_partition_safe: no `.select()` on batches."""
    src = inspect.getsource(gb_mod._read_partition_safe)
    assert ".select(" not in src, (
        "_read_partition_safe must not call .select() on batches — "
        "it yields reader batches as-is, so schema=_schema matches"
    )


def test_write_run_uses_table_schema():
    """sort._write_run: writer schema from table.schema (stable through take)."""
    src = inspect.getsource(sort_mod._write_run)
    assert "ArrowIPCBatchWriter(str(path), table.schema)" in src, (
        "_write_run must pass table.schema to ArrowIPCBatchWriter"
    )


def test_dedup_explicit_select_schema():
    """dedup._local_dedup: explicit select_schema (v0.8.11a-fix3)."""
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "select_schema = pa.schema(" in src, (
        "_local_dedup must build explicit select_schema (v0.8.11a-fix3)"
    )
    assert "schema=select_schema" in src, (
        "_local_dedup must pass schema=select_schema"
    )
'''


def write_schema_audit_tests():
    print("\n=== tests/test_schema_order_audit.py ===")
    p = Path("tests/test_schema_order_audit.py")
    p.write_text(SCHEMA_AUDIT_TESTS)
    compile_check(p)
    print("  OK")


# ============ G. CHANGELOG [Unreleased] append ============

def append_changelog():
    print("\n=== CHANGELOG.md [Unreleased]: append v0.8.11b ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.8.11b — strict join" in s:
        print("  already present — skip")
        return

    anchor = "- fast: **~1190** (+16 behavior, +5 AST guard). slow: **8**.\n"
    if anchor not in s:
        print("  WARNING: v0.8.11a test-count anchor not found — skip")
        return

    addendum = anchor + '''
### v0.8.11b — strict join conditional + schema-order audit

#### Изменено

- **`merge()` strict join enforcement.** `budget >= 160 MB` →
  `MemoryLimitExceeded` on whole-operation RSS over-budget (was: warn).
  Threshold: `4 × _JOIN_FIXED_OVERHEAD_BYTES = 160 MB`, plus secondary
  `state_limit >= 40 MB` (protects against reservation drop).
  Below threshold → warn only (backward compat with `mem_mb=20` tests;
  fixed 40 MB overhead dominates small budgets — strict check would fire
  trivially even for tiny joins). See §9.2.6.

- **`MemoryGuard.state_limit_bytes`** — public property. Callers need
  `state_limit` without raising `MemoryLimitExceeded` (used by merge's
  strict-mode decision).

- **`cadw/ops/join.py::_JOIN_STRICT_BUDGET_THRESHOLD_BYTES`** = `4 ×
  _JOIN_FIXED_OVERHEAD_BYTES` = 160 MB. Module constant, documented.

#### Добавлено (schema-order audit — §13.4/145 backlog)

- **`tests/test_schema_order_audit.py`** — 4 AST guards for the audit
  findings:
  - `join._safe_materialize_partition`: explicit `schema=out_schema`.
  - `sort._write_run`: `ArrowIPCBatchWriter(table.schema)`.
  - `groupby._read_partition_safe`: no `.select()` on batches.
  - `dedup._local_dedup`: `select_schema` (v0.8.11a-fix3).
  No additional bugs found across 4 audited call sites.

- **`tests/test_join_strict.py`** — 4 tests: threshold constant, public
  property, strict raise at budget 200 MB, strict pass small join.

#### Ограничения (v0.8.11b, §11.5)

- **Join strict enforcement requires `budget >= 160 MB`.** Below — warn
  only, per-batch guards still enforce per-batch memory. For guaranteed
  strict enforcement, set `MemoryGuard(max_memory_mb=200)` or higher.
  Rationale: 40 MB fixed overhead is non-trivial below 160 MB budget;
  raise would fire trivially for tiny joins.

#### Тесты (v0.8.11b)

- fast: **~1198** (+8), slow: **8**.

### Тесты (итого v0.8.11)

- fast: **~1198**, slow: **8**.

'''
    s = s.replace(anchor, addendum, 1)
    p.write_text(s)
    print("  OK")


# ============ H. DECISIONS updates ============

def update_decisions():
    print("\n=== DECISIONS.md: §9.2.6 + §11.5 + §13.4/145 update ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    # §9.2.6 addendum
    anchor_926 = "**Enforcement note (§11.5 addendum).** v0.8.9 per-batch pre-check —"
    if anchor_926 in s and "v0.8.11b (strict enforcement policy)" not in s:
        addendum = '''**v0.8.11b (strict enforcement policy).** `merge()` now enforces
`40 MB + 5 × predicted_output <= state_limit` **strictly** (raise) when
budget is large enough. Threshold:

    strict_mode = (budget >= 4 × _JOIN_FIXED_OVERHEAD_BYTES = 160 MB)
                  AND (state_limit >= _JOIN_FIXED_OVERHEAD_BYTES = 40 MB)

Rationale:
- **Budget < 160 MB:** 40 MB fixed overhead is > 25% of budget (> 50%
  of state_limit). Strict check would fire trivially for tiny joins,
  breaking backward compat with `mem_mb=20` tests and making CADW
  unusable on constrained machines. **Warn only**; per-batch guards
  in `_process_partition_pair` still enforce per-batch memory.
  Honest limitation documented (§11.5).
- **Budget >= 160 MB:** overhead ≤ 25% of budget (≤ 50% of state_limit),
  leaving room for meaningful output estimate. **Strict: raise.** Real
  guarantee.
- **Secondary check `state_limit >= overhead`:** protects against
  reservation drop. If `available_bytes` shrinks such that
  `state_limit < 40 MB`, warn (not raise) — raise would fire trivially
  regardless of output size.

**Enforcement note (§11.5 addendum).** v0.8.9 per-batch pre-check —'''
        s = s.replace(anchor_926, addendum, 1)
        print("  OK: §9.2.6 strict-mode note")

    # §11.5 addendum
    anchor_115 = "### 11.5. Осознанные ограничения (документированы)"
    if anchor_115 in s and "Join strict enforcement" not in s:
        # Insert after §11.5 header line, before first table row
        insert_after = anchor_115 + "\n\n"
        addendum = insert_after + '''**v0.8.11b:** join strict enforcement requires `budget >= 160 MB`
(§9.2.6). Below threshold — warn only, per-batch guards still enforce
per-batch memory. For guaranteed strict enforcement, set
`MemoryGuard(max_memory_mb=200)` or higher.

'''
        s = s.replace(anchor_115 + "\n\n", addendum, 1)
        print("  OK: §11.5 strict-budget note")

    # §13.4/145 update (audit completed)
    anchor_145 = "**Backlog §11.6 (v0.9.0+):** audit всех `Table.from_batches(gen, schema=...)`"
    if anchor_145 in s and "v0.8.11b (audit completed)" not in s:
        # Find the whole §13.4/145 episode and append update at its end
        m = re.search(
            r"(Эпизод 145.*?)(?=\n\*\*Связанные:|\n###|\Z)",
            s, re.DOTALL,
        )
        if m:
            update = m.group(1) + '''

**Update v0.8.11b (audit completed).** Verified 4 additional call sites
of `Table.from_batches(gen, schema=...)` in `cadw/ops/`:
- `join._safe_materialize_partition`: explicit `schema=out_schema`,
  order matches `.select(keep_cols)`. OK.
- `sort._write_run`: `ArrowIPCBatchWriter(table.schema)` — `table.schema`
  preserved through `pc.take(order)`. OK.
- `sort._merge_keys`: explicit `schema=schema` in `RecordBatch.from_arrays`.
  OK.
- `groupby._read_partition_safe`: `_iter_batches` yields reader batches
  as-is (no `.select(...)`), `schema=_schema` matches reader order
  exactly. OK.
- `dedup._local_dedup`: fixed v0.8.11a-fix3.

**No additional schema-order bugs found.** 4 AST guards in
`tests/test_schema_order_audit.py` lock the invariants.
'''
            s = s[:m.start(1)] + update + s[m.end(1):]
            print("  OK: §13.4/145 audit completed")

    p.write_text(s)


# ============ main ============

def main():
    print("=== v0.8.11b: strict join + schema-order audit ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    add_state_limit_property()
    add_strict_threshold_const()
    patch_merge_strict_branch()
    update_existing_ast_tests()
    write_strict_tests()
    write_schema_audit_tests()
    append_changelog()
    update_decisions()

    print("\n=== pytest: strict join + schema audit ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join_strict.py",
         "tests/test_schema_order_audit.py",
         "tests/test_join_rss_warning.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: join regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join.py", "tests/test_join_streaming.py",
         "tests/test_join_oom_guard.py", "tests/test_e2e_join.py",
         "tests/test_merge_planner.py",
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
        "v0.8.11b: strict join (budget >= 160 MB -> raise, below -> warn); "
        "MemoryGuard.state_limit_bytes; schema-order audit (§13.4/145); "
        "4 AST guards + 4 strict-join tests"])

    print("\n==> v0.8.11b applied. Ready to tag v0.8.11.")
    print()
    print("SQUASH PLAN (P0 safety complete):")
    print("  git reset --soft v0.8.10")
    print("  git commit -m 'v0.8.11: P0 safety - Arrow-native dedup, strict join'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.8.11] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.8.11 -m 'v0.8.11: P0 safety complete'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()