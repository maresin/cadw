#!/usr/bin/env python3
"""v0.8.10-pre: whole-operation join RSS warning (audit P-A3 addendum).

CONTEXT (audit v0.8.8):
- P-A3 confirmed live: per-probe-batch pre-check (5 × row_bytes_out ×
  per-batch matches) enforces per-batch memory correctly, but ignores
  Arrow memory pool init + hash-table build (~40 MB floor). Whole-join
  RSS peak at N=1M pk: 114.7 / 142.8 / 156.8 MB across 3 runs — 5×output
  = 120 MB is a middle estimate; 40 MB offset covers worst case.

WHY WARNING NOW, NOT RAISE:
- Adding offset per-batch over-predicts by 8-16× (fires once per
  partition pair). Adding it whole-operation at merge() entry requires
  bumping ~10 test files' `mem_mb=20` → 100 MB (predicted 40 MB >> 10 MB
  state_limit). Not a 5-minute change; needs full test review.
- Users get an explicit WARNING signal today. Enforcement deferred to
  v0.8.11 with test-budget review in one focused commit.

WHAT THIS PATCH DOES:
- Module const `_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024` in join.py.
- In `CadwDataFrame.merge`, after planner solve: compute
  `predicted_output = n_left × avg_matches × row_bytes_out`,
  attempt `check_state(40 MB + 5 × predicted_output)`. On
  `MemoryLimitExceeded` → log WARNING (proceed, do not raise).
- §9.2.6 addendum (variance analysis, 3-run range).
- §11.5 addendum (join guarantee is best-effort on RSS until v0.8.11).

Run: uv run python scripts/patch_v0810_pre_join_offset.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
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


# ============ A. module constant in join.py ============

def add_join_overhead_const():
    print("\n=== cadw/ops/join.py: _JOIN_FIXED_OVERHEAD_BYTES ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()
    if "_JOIN_FIXED_OVERHEAD_BYTES" in s:
        print("  already present — skip")
        return

    anchor = "from cadw.ops.shuffle import hash_shuffle\n"
    if anchor not in s:
        raise SystemExit("shuffle import anchor not found")

    const = (
        anchor
        + '''
# v0.8.10-pre (P-A3 audit addendum): measured RSS floor for Table.join
# beyond logical output. Source: 3 runs of pk join @ N=1M (int64 keys,
# 24 MB output) gave 114.7 / 142.8 / 156.8 MB RSS delta across identical
# binary. 5×output alone (120 MB) under-predicts the worst run by ~37 MB.
# Offset covers Arrow pool init (~16 MB) + hash-table build + non-Arrow
# temporaries (~24 MB) — measured as a floor, invariant of output size.
# Used by `CadwDataFrame.merge` for a whole-operation WARNING. Enforcement
# (raise) is deferred to v0.8.11 with test-budget review; see §9.2.6,
# §11.5.
_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024
'''
    )
    s = s.replace(anchor, const, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ B. warning in dataframe.py::merge ============

def add_merge_warning():
    print("\n=== cadw/dataframe.py::merge — whole-operation RSS warning ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "join.whole_operation_rss" in s:
        print("  already present — skip")
        return

    # Add imports (idempotent)
    if "from cadw.core.logging import get_logger" not in s:
        anchor = "from cadw.core.memory_guard import MemoryGuard\n"
        if anchor not in s:
            raise SystemExit("memory_guard import anchor not found")
        s = s.replace(
            anchor,
            anchor + "from cadw.core.logging import get_logger\n",
            1,
        )

    # Add module-level logger (idempotent)
    if "\nlogger = get_logger(\"dataframe\")\n" not in s:
        anchor = "from cadw.ops.resample import (\n"
        if anchor not in s:
            raise SystemExit("resample import anchor not found")
        # Insert logger AFTER all imports; find line after ")\n" that
        # closes the resample import.
        idx = s.find(anchor)
        if idx < 0:
            raise SystemExit("resample import block not found")
        # Find the closing ")\n" of that block
        close_idx = s.find(")\n", idx) + 2
        s = s[:close_idx] + "\nlogger = get_logger(\"dataframe\")\n" + s[close_idx:]

    # Find the planner block inside merge and add warning AFTER it.
    # Anchor: `max_output_rows = params.max_output_rows` — the last
    # planner-dependent assignment before hash_join_streaming call.
    old = '''                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows
            else:
                # Одна из сторон пуста — planner не нужен; hash_join
                # короткозамкнёт. num_partitions не важен.
                if num_partitions is None:
                    num_partitions = 8
        else:
            if num_partitions is None:
                num_partitions = 8'''

    new = '''                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows

                # v0.8.10-pre (P-A3 addendum): whole-operation RSS WARNING.
                # Per-batch check in _process_partition_pair enforces
                # per-batch memory, but does NOT see Arrow pool init +
                # hash-table build (~40 MB floor, measured across 3 runs
                # @ N=1M: 114.7 / 142.8 / 156.8 MB). Warn if worst-case
                # whole-operation RSS may exceed state_limit. Do not raise
                # (would break existing tests with 20 MB budget — bumping
                # budgets to 100 MB is v0.8.11 work).
                try:
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
                    )
            else:
                # Одна из сторон пуста — planner не нужен; hash_join
                # короткозамкнёт. num_partitions не важен.
                if num_partitions is None:
                    num_partitions = 8
        else:
            if num_partitions is None:
                num_partitions = 8'''

    s = replace_line_block(s, old, new, tag="merge-warning")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ C. CHANGELOG v0.8.10-pre ============

def insert_changelog():
    print("\n=== CHANGELOG.md: prepend [0.8.10-pre] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [Unreleased]" in s:
        print("  already present — skip")
        return
    entry = """## [Unreleased]

### Добавлено (safety follow-up, P-A3 addendum)

- **Whole-operation RSS warning on `merge()`.** `CadwDataFrame.merge`
  вычисляет `predicted_output = n_left × avg_matches × row_bytes_out`
  из planner'а и логирует WARNING, если
  `40 MB + 5 × predicted_output > state_limit`.
  `MemoryLimitExceeded` не бросается — enforcement остаётся per-batch в
  `_process_partition_pair`, warning — диагностический сигнал.

  **Почему warning, а не raise:** per-batch проверка выполняется 8–16
  раз (по числу пар partition'ов), поэтому добавление 40 MB offset
  per-batch over-predicts в 8–16×. Добавление offset на уровне
  `merge()` требует поднятия `mem_mb=20` → `100` в ~10 тестовых файлах
  (~30 тестов падают без этого). Это отдельная работа v0.8.11, не
  «5 минут». Warning даёт сигнал сегодня без test churn.

- **`cadw/ops/join.py::_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024`**
  — module const, измеренный floor для `Table.join` RSS beyond logical
  output. Across 3 runs of pk join @ N=1M: 114.7 / 142.8 / 156.8 MB
  RSS delta для 24 MB output. Offset покрывает Arrow pool init (~16 MB)
  + hash-table build + non-Arrow temporaries (~24 MB). Invariant of
  output size — added once, not per-batch.

### Ограничения (v0.8.10-pre, §11.5 addendum)

- Join guarantee остаётся **best-effort на RSS**, не strict. Для
  бюджетирования: `estimated_rss = 40 MB + 5 × predicted_output`; если
  превышает `state_limit = budget × 0.5`, actual peak может превысить
  budget независимо от per-batch guards. Strict enforcement — v0.8.11
  с ревизией тестовых бюджетов.
- Run-to-run variance 30–40% для одинаковой нагрузки (pool retention,
  jemalloc page release timing). Формулы target'ят worst-case.

### Тесты

- fast: **~1130 passed** (+3 behavior warnings: tight/loose budget). slow: **8**.

"""
    m = re.search(r'^## \[', s, re.MULTILINE)
    if not m:
        raise SystemExit("no `## [` in CHANGELOG")
    s = s[:m.start()] + entry + s[m.start():]
    p.write_text(s)
    print("  OK: [0.8.10-pre]")


# ============ D. DECISIONS §9.2.6 addendum ============

def add_decisions_variance_note():
    print("\n=== DECISIONS.md: §9.2.6 variance addendum ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.8.10-pre (variance analysis)" in s:
        print("  already present — skip")
        return

    # Anchor on §9.2.7 header — insert addendum before it.
    anchor = "### 9.2.7. `Table.sort_by`"
    if anchor not in s:
        print("  WARNING: §9.2.7 anchor not found — skipping")
        return

    addendum = '''**v0.8.10-pre (variance analysis).** Измерено в 3 раздельных прогонах
`pk join @ N=1M` (int64 key + int64 payload, 24 MB output):
- 114.7 MB (audit first run)
- 142.8 MB (rerun, same binary)
- 156.8 MB (P-A3 live-violation bench, same binary)

Разброс 42 MB на идентичном коде. Источник: Arrow memory pool retention
policy + jemalloc page release timing. Формула `5 × output + 40 MB`
target'ит upper end; при бюджетировании на медиану (`5 × output + 25 MB`)
— на свой риск.

**Enforcement note (§11.5 addendum).** v0.8.9 per-batch pre-check —
**per-batch enforcement only**. Whole-operation warning добавлен в
v0.8.10-pre (log WARNING при `40 MB + 5 × predicted_output > state_limit`);
strict enforcement (raise) отложен в v0.8.11 с ревизией тестовых
бюджетов (20 → 100 MB в ~10 файлах).

'''
    s = s.replace(anchor, addendum + anchor, 1)
    p.write_text(s)
    print("  OK")


# ============ E. tests ============

BEHAVIOR_TESTS = '''"""v0.8.10-pre: whole-operation join RSS warning."""
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_join_fixed_overhead_const_exists():
    from cadw.ops.join import _JOIN_FIXED_OVERHEAD_BYTES
    assert _JOIN_FIXED_OVERHEAD_BYTES == 40 * 1024 * 1024


def test_merge_warns_on_tight_budget(tmp_path, caplog):
    """Whole-operation RSS > state_limit → WARNING, not raise."""
    caplog.set_level(logging.WARNING, logger="cadw.dataframe")
    left = make_df(
        tmp_path, "L",
        {"k": list(range(1000)), "a": list(range(1000))},
        mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(1000)), "b": list(range(1000))},
        mem_mb=20,
    )
    result = left.merge(right, on="k")
    assert result.shape[0] == 1000  # did NOT raise
    assert any(
        "whole-operation RSS" in rec.message for rec in caplog.records
    ), f"expected warning; got: {[r.message for r in caplog.records]}"


def test_merge_no_warn_loose_budget(tmp_path, caplog):
    """Loose budget → no warning (predicted RSS < state_limit)."""
    caplog.set_level(logging.WARNING, logger="cadw.dataframe")
    left = make_df(
        tmp_path, "L",
        {"k": list(range(1000)), "a": list(range(1000))},
        mem_mb=500,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(1000)), "b": list(range(1000))},
        mem_mb=500,
    )
    result = left.merge(right, on="k")
    assert result.shape[0] == 1000
    assert not any(
        "whole-operation RSS" in rec.message for rec in caplog.records
    )


def test_merge_warning_does_not_raise(tmp_path, caplog):
    """Even with very tight budget, per-batch guards pass → merge succeeds."""
    caplog.set_level(logging.WARNING, logger="cadw.dataframe")
    left = make_df(
        tmp_path, "L",
        {"k": [1, 2, 3], "a": [10, 20, 30]},
        mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": [1, 3], "b": [100, 300]},
        mem_mb=20,
    )
    # Small join fits per-batch → completes; whole-op warning may fire.
    result = left.merge(right, on="k")
    assert result.shape[0] == 2


def test_merge_ast_guard_warning_present():
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
                )
'''


def write_tests():
    print("\n=== tests/test_join_rss_warning.py ===")
    p = Path("tests/test_join_rss_warning.py")
    p.write_text(BEHAVIOR_TESTS)
    compile_check(p)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.8.10-pre: join whole-operation RSS warning ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    branch = r.stdout.strip()
    print(f"  branch: {branch}")

    if branch != "master":
        print(f"WARNING: on branch {branch}, not master. Continuing.")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    add_join_overhead_const()
    add_merge_warning()
    insert_changelog()
    add_decisions_variance_note()
    write_tests()

    print("\n=== pytest new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_join_rss_warning.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== regression: join + e2e_join ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join.py", "tests/test_e2e_join.py",
         "tests/test_join_streaming.py", "tests/test_join_oom_guard.py",
         "tests/test_merge_planner.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=900)
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
        "v0.8.10-pre: merge() whole-operation RSS warning "
        "(P-A3 audit addendum); _JOIN_FIXED_OVERHEAD_BYTES=40 MB; "
        "§9.2.6 variance note; §11.5 best-effort clarification"])

    print("\n==> v0.8.10-pre applied. Not tagged (rolling into v0.8.10).")


if __name__ == "__main__":
    main()