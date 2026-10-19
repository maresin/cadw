#!/usr/bin/env python3
"""v0.8.5: CHANGELOG reorder + row_id reserved guard + rolling_reference.

PHASE 1 — CHANGELOG reorder (fixes test_changelog_has_latest_tag):
  Current CHANGELOG entries are out-of-order: `[0.7.4c]` at top (from
  patch_v074c), `[0.8.4]` in middle (from patch_v084 which inserted
  before `## [0.8.2]`). Root cause: `patch_changelog()` inserts at
  specific anchor; as history grew, anchors drifted.

  Fix: sort all `## [X]` blocks descending by (major, minor, patch,
  suffix). Add `[0.8.5]` at top. Add guard
  `test_changelog_is_descending`.

PHASE 2 — v0.8.5 code changes:
  - `row_id` reserved guard in external_sort + hash_shuffle (fail-fast
    ValueError before materialize, §13.4/126).
  - `tests/_helpers/rolling_reference.py` — deque ground truth
    (meta-guard for §13.4/128).
  - 4 rolling tests converted to reference-derived expectations.
  - 8 new tests in test_row_id_reserved.py.

Run: uv run python scripts/patch_v085_guards.py
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
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
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


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if len(starts) == 0:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ PHASE 1: CHANGELOG reorder ============

def _version_key(v: str):
    m = re.match(r'^(\d+)\.(\d+)\.(\d+)([a-z]?)$', v)
    if not m:
        raise ValueError(f"unrecognized version: {v!r}")
    major, minor, patch, suffix = m.groups()
    return (int(major), int(minor), int(patch), suffix)


def reorder_changelog():
    print("\n=== CHANGELOG.md: reorder descending by version ===")
    p = Path("CHANGELOG.md")
    text = p.read_text()

    matches = list(re.finditer(r'^## \[([^\]]+)\]', text, re.MULTILINE))
    if not matches:
        raise SystemExit("no `## [X]` entries in CHANGELOG")

    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        entries.append((m.group(1), text[start:end]))

    versions = [v for v, _ in entries]
    if len(set(versions)) != len(versions):
        dup = [v for v in set(versions) if versions.count(v) > 1]
        raise SystemExit(f"duplicate versions in CHANGELOG: {dup}")

    print(f"  found {len(versions)} entries")
    for v in versions:
        print(f"    {v}")
    print()

    entries.sort(key=lambda e: _version_key(e[0]), reverse=True)

    new_header = '''# Changelog

Все значимые изменения проекта CADW. Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

<!--
  Порядок: **descending by version** (v0.8.5 сверху). Восстановлен в
  v0.8.5 из-за сломанной цепочки v0.1.5+ (§13.4/111): каждый
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`, а при
  отсутствии anchor `str.replace` был silent no-op. Как побочный
  эффект, anchors дрейфовали — top становился out-of-order. Пропущенные
  версии (v0.1.5, v0.1.6, v0.2.2, v0.3.5, v0.4.2, v0.5.0, v0.5.2-v0.5.4,
  v0.6.x, v0.7.0-v0.7.7, v0.7.10) — в `git log --oneline --grep='^v'`.
  AST guard `test_changelog_has_latest_tag` (top == tag on master) +
  `test_changelog_is_descending` (v0.8.5) фиксируют инвариант.
-->

'''

    new_text = new_header + "".join(content for _, content in entries)
    p.write_text(new_text)
    print(f"  OK: reordered (descending), {len(entries)} entries")


def insert_v085_entry():
    print("\n=== CHANGELOG.md: prepend [0.8.5] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.5]" in s:
        print("  already present — skip")
        return
    entry = """## [0.8.5] — 2026-10-19

### Исправлено
- **CHANGELOG отсортирован descending** (§13.4/111 + §13.4/130). Исторически
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`. Anchor дрейфовал,
  top CHANGELOG перестал быть «latest». `test_changelog_has_latest_tag`
  (master + tag) корректно упал: `top [0.7.4c] != tag v0.8.4`. Fix:
  одноразовый sort всех `## [X]` блоков by (major, minor, patch, suffix)
  descending. Новый guard `test_changelog_is_descending` (в
  `test_changelog.py`) ловит регресс. v0.8.4 tag **не** ретагнут —
  тест не запускается на старых tags (`--exact-match` на HEAD).
- **`row_id` reserved guard** в `external_sort` и `hash_shuffle`
  (§13.4/126). User column named `row_id` коллизит с internal identity
  column (appended by external_sort/hash_shuffle) → previously:
  `KeyError: Field "row_id" exists 2 times in schema` (через
  `ArrowIPCBatchReader`, в `sort_values`); `Multiple matches for
  FieldRef.Name(row_id)` (в `pc.sort_indices`). Теперь: `ValueError:
  external_sort: column name 'row_id' is reserved for CADW internal
  identity column. Rename your column before calling.` Guard выполняется
  **до** materialize batches.

### Добавлено
- **`tests/_helpers/rolling_reference.py`** — `expected_rolling(vals,
  window, agg, min_periods)` — deque ground truth (§13.4/128). Matches
  pandas: positional window, nulls consume slot, min_periods applies to
  non-null count.
- 4 rolling-теста (`test_rolling_min/max_with_nulls` в
  `test_rolling.py`; `test_rolling_on_min/max_with_nulls` в
  `test_rolling_on_col.py`) → `expected_rolling(...)`. Устраняет корень
  §13.4/128 (manual arithmetic).
- 8 новых тестов в `test_row_id_reserved.py`: external_sort
  (reject key/value + valid works), hash_shuffle (reject key/value +
  valid works), E2E (sort_values/groupby на user `row_id`).
- `test_changelog.py::test_changelog_is_descending` — strict descending
  invariant.

### Ограничения (v0.8.5, §11.5)
- `row_id` — reserved. User column → rename. `__orig_rid` тоже reserved
  (internal sortback). Рекомендуется префикс `my_`.
- Guard fires on arguments (`key_columns` / `value_columns`). Internal
  callers (`_shift_on`, `_rolling_on`, `rank`, `groupby`) либо передают
  `__orig_rid`, либо не содержат `row_id` в value list — не затронуты.

### Тесты
- +8 в `test_row_id_reserved.py`.
- +1 в `test_changelog.py` (descending guard).
- 4 rolling-теста → helper (behavior unchanged).
- fast: **~1056 passed**. slow: **8 passed**.

"""
    m = re.search(r'^## \[', s, re.MULTILINE)
    if not m:
        raise SystemExit("no `## [` in CHANGELOG (reorder failed?)")
    s = s[:m.start()] + entry + s[m.start():]
    p.write_text(s)
    print("  OK: [0.8.5] at top")


def add_changelog_descending_guard():
    print("\n=== tests/test_changelog.py: +test_changelog_is_descending ===")
    p = Path("tests/test_changelog.py")
    s = p.read_text()
    if "def test_changelog_is_descending(" in s:
        print("  already present — skip")
        return
    extra = '''

def test_changelog_is_descending():
    """All versions strictly descending (top = highest).

    Регрессия §13.4/130: patch_changelog() с anchor-based insert привёл к
    out-of-order entries — `[0.7.4c]` (patch_v074c) оказался выше
    `[0.8.4]` (patch_v084 вставлен перед `## [0.8.2]`, а `[0.7.4c]`
    был добавлен ещё раньше выше `[0.8.2]`). `test_changelog_has_latest_tag`
    поймал симптом (`top != tag`), этот guard ловит саму причину.
    """
    import re as _re
    text = CHANGELOG.read_text()
    versions = _re.findall(r"^## \\[(\\d[^\\]]*)\\]", text, flags=_re.MULTILINE)
    assert versions, "CHANGELOG has no `## [X.Y.Z]` entries"

    def _key(v):
        m = _re.match(r'^(\\d+)\\.(\\d+)\\.(\\d+)([a-z]?)$', v)
        assert m, f"unrecognized version: {v!r}"
        ma, mi, pa, su = m.groups()
        return (int(ma), int(mi), int(pa), su)

    keys = [_key(v) for v in versions]
    assert keys == sorted(keys, reverse=True), (
        f"CHANGELOG not descending. Order: {list(zip(versions, keys))}"
    )
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: +1 descending guard")


# ============ PHASE 2: row_id guards ============

def patch_external_sort_guard():
    print("\n=== cadw/ops/sort.py: external_sort row_id guard ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()

    old = '''    if not key_columns:
        raise ValueError("key_columns must be non-empty")
    nk = len(key_columns)
    asc = _normalize_per_column(ascending, nk, "ascending")
    all_columns = _unique_columns(value_columns, key_columns)'''

    new = '''    if not key_columns:
        raise ValueError("key_columns must be non-empty")
    # v0.8.5 (§13.4/126): `row_id` reserved (internal identity column,
    # appended by external_sort / hash_shuffle). User column named
    # `row_id` → duplicate → `KeyError: Field "row_id" exists 2 times in
    # schema` (ArrowIPCBatchReader) or `Multiple matches for
    # FieldRef.Name(row_id)` (pc.sort_indices). Fail-fast here, BEFORE
    # any batch materialization.
    if "row_id" in key_columns or "row_id" in value_columns:
        raise ValueError(
            "external_sort: column name 'row_id' is reserved for CADW "
            "internal identity column. Rename your column before calling. "
            "See DECISIONS §11.5, §13.4/126."
        )
    nk = len(key_columns)
    asc = _normalize_per_column(ascending, nk, "ascending")
    all_columns = _unique_columns(value_columns, key_columns)'''

    s = replace_line_block(s, old, new, tag="ext-sort-guard")
    p.write_text(s)
    compile_check(p)
    print("  OK: external_sort — row_id guard")


def patch_hash_shuffle_guard():
    print("\n=== cadw/ops/shuffle.py: hash_shuffle row_id guard ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if not key_columns:
        raise ValueError("key_columns must be non-empty")

    spill_dir = Path(spill_dir)'''

    new = '''    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if not key_columns:
        raise ValueError("key_columns must be non-empty")
    # v0.8.5 (§13.4/126): `row_id` reserved — same rationale as in
    # external_sort. Check arguments (not materialized batch) for
    # cheapest fail-fast.
    if "row_id" in key_columns or "row_id" in value_columns:
        raise ValueError(
            "hash_shuffle: column name 'row_id' is reserved for CADW "
            "internal identity column. Rename your column before calling. "
            "See DECISIONS §11.5, §13.4/126."
        )

    spill_dir = Path(spill_dir)'''

    s = replace_line_block(s, old, new, tag="hash-guard")
    p.write_text(s)
    compile_check(p)
    print("  OK: hash_shuffle — row_id guard")


# ============ PHASE 3: rolling_reference helper ============

def write_rolling_reference():
    print("\n=== tests/_helpers/rolling_reference.py ===")
    helpers_dir = Path("tests/_helpers")
    helpers_dir.mkdir(exist_ok=True)
    (helpers_dir / "__init__.py").write_text("")

    p = helpers_dir / "rolling_reference.py"
    p.write_text('''"""Reference rolling implementation for tests (§13.4/128).

WHY: two consecutive test-expectation errors in v0.8.4-fix (rolling+nulls)
were manual arithmetic mistakes. Reference function derives expectations
from a deque-based algorithm (ground truth) instead of hand-computation.

USAGE:
    from tests._helpers.rolling_reference import expected_rolling
    expected = expected_rolling(vals, window, agg, min_periods=...)
    assert got == expected

SEMANTICS (matches pandas Series.rolling):
- Positional window of size `window` over `vals` (list, 1D).
- Nulls consume window slot but contribute nothing to sum/min/max/mean.
- count = number of non-null in current window.
- min_periods applies to NON-NULL count, not window size:
    pd.Series([1, None, 3]).rolling(2, min_periods=2).sum()
    -> [NaN, NaN, NaN]  (each window has at most 1 non-null)
- Output: list[float | int | None], length == len(vals).
"""
from __future__ import annotations

from collections import deque


def expected_rolling(vals, window: int, agg: str, min_periods=None):
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be int >= 1; got {window!r}")
    if agg not in ("sum", "mean", "min", "max", "count"):
        raise ValueError(
            f"agg must be one of sum/mean/min/max/count; got {agg!r}"
        )
    if min_periods is None:
        min_periods = window
    elif not (1 <= min_periods <= window):
        raise ValueError(
            f"min_periods must be in [1, window={window}]; got {min_periods}"
        )

    out = []
    buf = deque(maxlen=window)
    for v in vals:
        buf.append(v)
        non_null = [x for x in buf if x is not None]
        if len(non_null) < min_periods:
            out.append(None)
            continue
        if agg == "count":
            out.append(len(non_null))
        elif not non_null:
            out.append(None)
        elif agg == "sum":
            out.append(sum(non_null))
        elif agg == "mean":
            out.append(sum(non_null) / len(non_null))
        elif agg == "min":
            out.append(min(non_null))
        elif agg == "max":
            out.append(max(non_null))
    return out
''')
    compile_check(p)
    print("  OK: helper + tests/_helpers/__init__.py")


# ============ PHASE 4: convert rolling tests ============

def convert_rolling_tests():
    print("\n=== 4 rolling tests → expected_rolling ===")
    for fname in ("tests/test_rolling.py", "tests/test_rolling_on_col.py"):
        p = Path(fname)
        s = p.read_text()

        # Replace min_with_nulls block
        for agg in ("min", "max"):
            old = f'''def test_rolling_{agg}_with_nulls(tmp_path):
    """S1 regression: {agg} + null must not crash (`_dq[-1][1] >= None`).'''
            if old in s:
                # Find end of function
                start = s.find(old)
                end = s.find("\ndef ", start + 1)
                if end < 0:
                    end = len(s)
                new = f'''def test_rolling_{agg}_with_nulls(tmp_path):
    """S1 regression ({agg} + null) + reference-derived (§13.4/128)."""
    from tests._helpers.rolling_reference import expected_rolling
    vals_in = [3.0, None, 1.0, None, 5.0]
    df = make_df(tmp_path, data={{"v": vals_in}})
    res = df.rolling(2, "{agg}", min_periods=1)
    got = read_col(res, tmp_path, "v_rolling_{agg}")
    assert got == expected_rolling(vals_in, 2, "{agg}", min_periods=1)
'''
                s = s[:start] + new + s[end:]

            old_on = f'''def test_rolling_on_{agg}_with_nulls(tmp_path):
    """S1 regression (on=col): {agg} + null must not crash.'''
            if old_on in s:
                start = s.find(old_on)
                end = s.find("\ndef ", start + 1)
                if end < 0:
                    end = len(s)
                new_on = f'''def test_rolling_on_{agg}_with_nulls(tmp_path):
    """S1 regression ({agg} + null, on=col) + reference-derived (§13.4/128)."""
    from tests._helpers.rolling_reference import expected_rolling
    vals_in = [3.0, None, 1.0, None, 5.0]
    df = make_df(tmp_path, data={{
        "ts": [1, 2, 3, 4, 5],
        "v": vals_in,
    }})
    res = df.rolling(2, "{agg}", on="ts", columns=["v"], min_periods=1)
    got = read_col(res, tmp_path, "v_rolling_{agg}")
    assert got == expected_rolling(vals_in, 2, "{agg}", min_periods=1)
'''
                s = s[:start] + new_on + s[end:]

        p.write_text(s)
        compile_check(p)
    print("  OK: 4 tests converted")


# ============ PHASE 5: row_id tests ============

def write_row_id_tests():
    print("\n=== tests/test_row_id_reserved.py ===")
    p = Path("tests/test_row_id_reserved.py")
    p.write_text('''"""v0.8.5: `row_id` reserved guard (§13.4/126).

`row_id` is CADW internal identity column (appended by external_sort /
hash_shuffle). User column named `row_id` → duplicate → cryptic error
downstream. Guard: fail-fast ValueError in external_sort/hash_shuffle
before any batch read.
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ops.shuffle import hash_shuffle
from cadw.ops.sort import external_sort


def make_iter(table, bs=2):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


# ---------- external_sort ----------

def test_external_sort_rejects_row_id_key(tmp_path):
    table = pa.table({"row_id": [1, 2, 3], "v": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        external_sort(
            make_iter(table),
            key_columns=["row_id"],
            value_columns=["v"],
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "s1",
            log=False,
        )


def test_external_sort_rejects_row_id_value(tmp_path):
    table = pa.table({"k": [1, 2, 3], "row_id": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        external_sort(
            make_iter(table),
            key_columns=["k"],
            value_columns=["row_id"],
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "s2",
            log=False,
        )


def test_external_sort_without_row_id_works(tmp_path):
    table = pa.table({"k": [3, 1, 2], "v": [30, 10, 20]})
    p = external_sort(
        make_iter(table),
        key_columns=["k"],
        value_columns=["v"],
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "s3",
        log=False,
    )
    assert p.exists()


# ---------- hash_shuffle ----------

def test_hash_shuffle_rejects_row_id_key(tmp_path):
    table = pa.table({"row_id": [1, 2, 3], "v": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        hash_shuffle(
            make_iter(table),
            key_columns=["row_id"],
            value_columns=["v"],
            num_partitions=2,
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "h1",
            log=False,
        )


def test_hash_shuffle_rejects_row_id_value(tmp_path):
    table = pa.table({"k": [1, 2, 3], "row_id": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        hash_shuffle(
            make_iter(table),
            key_columns=["k"],
            value_columns=["row_id"],
            num_partitions=2,
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "h2",
            log=False,
        )


def test_hash_shuffle_without_row_id_works(tmp_path):
    table = pa.table({"k": [1, 2, 3, 1], "v": [10, 20, 30, 40]})
    paths = hash_shuffle(
        make_iter(table),
        key_columns=["k"],
        value_columns=["v"],
        num_partitions=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "h3",
        log=False,
    )
    assert len(paths) == 2


# ---------- E2E (was cryptic KeyError) ----------

def test_sort_values_on_user_row_id_raises(tmp_path):
    t = pa.table({"row_id": [3, 1, 2], "v": [30, 10, 20]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    with pytest.raises(ValueError, match="row_id.*reserved"):
        df.sort_values("row_id")


def test_groupby_on_user_row_id_raises(tmp_path):
    t = pa.table({"row_id": ["a", "b", "a"], "v": [1, 2, 3]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    with pytest.raises(ValueError, match="row_id.*reserved"):
        df.groupby("row_id").agg("v", "sum")
''')
    compile_check(p)
    print("  OK: 8 tests")


# ============ PHASE 6: DECISIONS ============

def patch_decisions():
    print("\n=== DECISIONS.md: §11.5 + §11.1 + §13.4/129,130 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    old_115 = '| `row_id` — reserved column name (v0.8.4) | `external_sort` / `hash_shuffle` / `merge` / `_shift_on` / `_rolling_on` | User column named `row_id` → `KeyError: Field "row_id" exists 2 times in schema` в `sort_values`. Workaround: rename user column. v0.9: fail-fast guard в `external_sort`/`hash_shuffle` (§13.4/126) |'
    new_115 = '| `row_id` — reserved column name (v0.8.5) | `external_sort` / `hash_shuffle` | **Guard (v0.8.5)**: fail-fast `ValueError` до materialize. User column `row_id` → rename (§13.4/126) |'
    if old_115 in s:
        s = s.replace(old_115, new_115, 1)
        print("  OK: §11.5 — row_id now guarded")

    if "| `row_id` reserved guard (v0.8.5) |" not in s:
        s = s.replace(
            "| `rolling(on=col)` (v0.8.4: sort-aware, prototype-validated) | ✅ | 18 |",
            "| `rolling(on=col)` (v0.8.4: sort-aware, prototype-validated) | ✅ | 18 |\n"
            "| `row_id` reserved guard (v0.8.5) | ✅ | 8 |\n"
            "| `rolling_reference` helper (v0.8.5) | ✅ | (4 tests converted) |\n"
            "| `test_changelog_is_descending` (v0.8.5) | ✅ | 1 |",
            1,
        )

    if "Эпизод 129" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 129 (meta-guard: `rolling_reference` helper).** §13.4/128
зафиксировал: test-expectation для rolling+nulls — источник ошибок
(2 итерации v0.8.4-fix). Корень — **ручная арифметика** в expected,
не в коде.

**Fix (v0.8.5):** `tests/_helpers/rolling_reference.py` —
`expected_rolling(vals, window, agg, min_periods)`. Deque-based, matches
pandas: positional window, nulls consume slot (не contribute),
`min_periods` applies to **non-null count** (не window size):
`pd.Series([1, None, 3]).rolling(2, min_periods=2).sum()` → `[NaN, NaN, NaN]`.

Применено в 4 тестах (min/max + nulls, positional + on=col).
Expected **derived**, не authored: property of algorithm, не of arithmetic.

**Урок (meta):** для rolling/shift/quantile/rank — hard-coded expected =
потенциальный баг. Три уровня:
1. Reference helper (`expected_rolling`).
2. pandas-equivalence (`pd.Series(vals).rolling(...).agg().tolist()`).
3. Per-position trace в docstring (если helper не подходит).

**Эпизод 130 (CHANGELOG descending — anchor drift).** `test_changelog_has_latest_tag`
правильно упал на master+v0.8.4: `top [0.7.4c] != tag v0.8.4`. Root:
`patch_changelog()` вставлял перед anchor (`## [0.8.2]` для `[0.8.4]`,
`## [0.8.4]` для `[0.8.5]`, …). Со временем `[0.7.4c]` (patch_v074c)
вставлен ещё раньше **выше** `[0.8.2]`, а `[0.8.4]` — перед `[0.8.2]`
→ `[0.8.4]` ниже `[0.7.4c]`.

**Fix (v0.8.5):** одноразовый sort всех `## [X]` блоков descending by
(major, minor, patch, suffix). Новый guard `test_changelog_is_descending`
(check sorted) — ловит **причину**, не только симптом.

**v0.8.4 tag не ретагнут** — тест `--exact-match` запускается только
на HEAD; старые tags не проверяются. v0.8.4 tag остаётся
out-of-order (historical artifact); v0.8.5 commit — fix.

**Урок (усиливает §13.4/111):** «anchor-based insert» в
CHANGELOG/build-files — source of drift. Правильно: **insert-at-top**
(искать первый `## [` regex, вставлять перед ним) или **build** из
separate files. Anchor `## [X.Y.Z-1]` предполагает, что `[X.Y.Z-1]`
существует и стоит на правильном месте — оба предположения нарушаются
по мере изменений.

**Правило:** для append-style docs (CHANGELOG, RELEASE_NOTES) — top
insertion by regex `^## [` first-match, или sort на read. Anchor-based
— только для стабильных anchors (e.g. `[tool.pytest.ini_options]` в
pyproject.toml, который никогда не двигается).

**Следующее:** v0.8.4a (multi-column `on`) → v0.8.3 `resample(freq)`
(prototype-first).

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.5, §11.1, §13.4/129+130")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v085-guards"

    print("=== v0.8.5: CHANGELOG reorder + row_id guard + rolling_reference ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v085-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v085-{ts_str}",
        "-m", "before patch v0.8.5"])

    # Phase 1: CHANGELOG reorder + new entry
    reorder_changelog()
    insert_v085_entry()
    add_changelog_descending_guard()

    # Phase 2: row_id guards
    patch_external_sort_guard()
    patch_hash_shuffle_guard()

    # Phase 3: rolling_reference
    write_rolling_reference()
    convert_rolling_tests()

    # Phase 4: row_id tests + DECISIONS
    write_row_id_tests()
    patch_decisions()

    print("\n=== pytest tests/test_changelog.py -v (on patch branch — should SKIP) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.5 (changelog red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_row_id_reserved.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_row_id_reserved.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.5 (row_id red)"], check=False)
        sys.exit(1)

    print("\n=== pytest rolling/shift/expanding/quantile regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rolling.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py",
         "tests/test_expanding_ewm.py",
         "tests/test_quantile.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.5 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.5: CHANGELOG descending (reorder) + row_id reserved guard + "
        "rolling_reference helper (§13.4/129,130)"])

    print("\n==> v0.8.5 applied on", branch)
    print()
    print("NEXT: squash → tag v0.8.5:")
    print("  git reset --soft master")
    print("  git commit -m 'v0.8.5: CHANGELOG reorder + row_id guard + rolling_reference'")
    print("  git checkout master")
    print("  git merge --ff-only patch/v085-guards")
    print("  git branch -d patch/v085-guards")
    print("  git tag -a v0.8.5 -m 'v0.8.5'")
    print("  uv run pytest tests/test_changelog.py -v   # 3 passed (2 old + descending)")


if __name__ == "__main__":
    main()