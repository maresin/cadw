#!/usr/bin/env python3
"""v0.9.1f-fix3: pivot pre-check formula (dense memory, not long-format).

BUG (§13.4/159): pre-check estimated `k_pairs_est × row_size × 2`,
but pandas pivot allocates DENSE `K_index × K_columns × value_size`.
For 10K unique × 20 cols, 200 MB budget:
  Current: 10000 × 100 × 108 × 2 = 216 MB > 60 MB → false fail
  Correct: 10000 × 20 × 8 × 2 = 3.2 MB << 100 MB → pass

Also: custom `0.3 × avail` inconsistent with `MemoryGuard.state_limit_bytes`
(0.5). And no pre-check for `grouped_table.to_pandas()` (3× overhead).

FIX:
  1. Pre-check: `K_index_est × K_columns_est × value_size × 2` (dense
     upper bound; value_size from values column type, not row).
  2. Use `self._memory_guard.state_limit_bytes` (0.5).
  3. Add `check_state(grouped_table.nbytes * 3)` before `to_pandas()`.
  4. Test budget 200 MB now works for 10K rows.

GUARDS: pivot tests + guards + fast + slow.

Run: uv run python scripts/patch_v091f_fix3.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def fix_pivot_precheck():
    print("\n=== cadw/dataframe.py::pivot_table — dense pre-check + to_pandas check ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        # §13.4/157: explicit pa.types.is_* — `hasattr(t, "bit_width")`
        # returns True for ALL types, but accessing `.bit_width` raises
        # ValueError on string/binary/list/struct. Use fixed sizes.
        def _safe_type_bytes(t) -> int:
            if (pa.types.is_integer(t) or pa.types.is_floating(t)
                    or pa.types.is_timestamp(t) or pa.types.is_date(t)
                    or pa.types.is_time(t)):
                return max(1, t.bit_width // 8)
            if pa.types.is_boolean(t):
                return 1
            if pa.types.is_string(t) or pa.types.is_binary(t):
                return 50  # conservative default
            return 8  # fallback for list/struct/dict

        row_size = sum(
            _safe_type_bytes(self._dataset.schema.field(c).type)
            for c in index_cols + [columns, values]
        )
        if row_size <= 0:
            row_size = 16
        est_bytes = k_pairs_est * row_size * 2
        state_limit = int(self._memory_guard.available_bytes() * 0.3)
        if est_bytes > state_limit:
            raise MemoryLimitExceeded(
                f"pivot_table: K_pairs ~{k_pairs_est} × row_size "
                f"{row_size} B × 2 = {est_bytes} B > 0.3 × avail "
                f"{state_limit} B. K_index ~{k_index_est}, K_columns "
                f"~{k_cols_est}. Reduce cardinality or use groupby() "
                f"(long format). See §11.5."
            )'''

    new = '''        # §13.4/157: explicit pa.types.is_* — `hasattr(t, "bit_width")`
        # returns True for ALL types, but accessing `.bit_width` raises
        # ValueError on string/binary/list/struct.
        # §13.4/159: pivot allocates DENSE `K_index × K_columns ×
        # value_size`, not long `K_pairs × row_size`. Use value_size.
        def _safe_type_bytes(t) -> int:
            if (pa.types.is_integer(t) or pa.types.is_floating(t)
                    or pa.types.is_timestamp(t) or pa.types.is_date(t)
                    or pa.types.is_time(t)):
                return max(1, t.bit_width // 8)
            if pa.types.is_boolean(t):
                return 1
            if pa.types.is_string(t) or pa.types.is_binary(t):
                return 50
            return 8

        value_size = _safe_type_bytes(
            self._dataset.schema.field(values).type
        )
        # Dense pivot upper bound: K_index × K_columns × value_size × 2
        # (pandas float64 for all aggs except count→int64; both ≤ 8).
        est_bytes = k_index_est * k_cols_est * value_size * 2
        state_limit = self._memory_guard.state_limit_bytes
        if est_bytes > state_limit:
            raise MemoryLimitExceeded(
                f"pivot_table: K_index ~{k_index_est} × K_columns "
                f"~{k_cols_est} × value_size {value_size} B × 2 "
                f"(pandas) = {est_bytes} B > state_limit "
                f"{state_limit} B. Reduce cardinality or use groupby() "
                f"(long format). See §11.5."
            )'''

    s = replace_line_block(s, old, new, tag="pivot-precheck-dense")
    p.write_text(s)
    compile_check(p)
    print("  OK: dense pre-check (K_index × K_columns × value_size)")


def add_to_pandas_precheck():
    print("\n=== cadw/dataframe.py::pivot_table — pre-to_pandas check ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        # v0.9.1f: explicit check_state before to_pandas (AST guard).
        # grouped_table is K_pairs rows, small (< est_bytes by
        # construction — groupby enforces same budget internally).
        self._memory_guard.check_state(
            grouped_table.nbytes, name="pivot_table.grouped",
        )
        value_col = f"{values}_{aggfunc}"

        # Reshape via pandas (small table)
        grouped_pd = grouped_table.to_pandas()'''

    new = '''        # v0.9.1f: explicit check_state before to_pandas (AST guard).
        # grouped_table is K_pairs rows — bounded by groupby internally.
        self._memory_guard.check_state(
            grouped_table.nbytes, name="pivot_table.grouped",
        )
        # §13.4/159: pre-check pandas overhead (× 3 for to_pandas).
        # pandas to_pandas allocates ~3× Arrow (object/StringDtype).
        self._memory_guard.check_state(
            grouped_table.nbytes * 3,
            name="pivot_table.to_pandas",
        )
        value_col = f"{values}_{aggfunc}"

        # Reshape via pandas (small table)
        grouped_pd = grouped_table.to_pandas()'''

    s = replace_line_block(s, old, new, tag="pre-to-pandas-check")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def append_decisions_159():
    print("\n=== DECISIONS.md: §13.4/159 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 159" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/159 (pivot pre-check = dense memory, not long).** "
        s += "K_index × K_columns × value_size, not K_pairs × row_size.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 159 (pivot pre-check = dense memory, not long-format).**
v0.9.1f-fix2: E2E success test 10K rows × 20 cols в 200 MB budget
падал с `MemoryLimitExceeded: K_pairs ~1000000 × row_size 108 × 2 =
216 MB > 0.3 × avail 60 MB`.

**Ошибка формулы.** `pivot_table` использует `groupby([*index,
columns])` (long format) + `pandas.pivot` (dense). Pre-check считал
long-format memory (`k_pairs × row_size`) вместо **dense pivot**
(`K_index × K_columns × value_size`).

- Long format: `K_pairs × 108 B` (row + col + value).
- Dense pivot: `K_index × K_columns × 8 B` (только value, все cells
  включая null).

Для 10K unique × 20 cols:
- Long: `200000 × 108 × 2 = 43 MB` (с точным K_columns=20).
- Extrapolated: `1M × 108 × 2 = 216 MB` (K_columns_est=100, 5× над).
- Dense: `10000 × 100 × 8 × 2 = 16 MB` (extrapolated) / 3.2 MB (точный).

**Dense — правильная метрика для `pandas.pivot`.** Long-format
groupby **уже** streamed на диск (groupby_aggregate → pa.Table, materialized
но bounded). Проблема — `pivot_table` materializes dense.

**Плюс:** custom `0.3 × avail` был inconsistent с
`MemoryGuard.state_limit_bytes` (0.5). Fix: use
`state_limit_bytes` — единая политика.

**Плюс:** отсутствовал pre-check для `grouped_table.to_pandas()`
(× 3 pandas overhead). Fix: `check_state(grouped_table.nbytes * 3)`
before to_pandas.

**Урок (§13.4/159):** для ops с **промежуточной materialization**
(long → dense) — pre-check должен оценивать **максимальный** из
промежуточных буферов, не первый. Pivot: long (bounded by groupby)
→ pandas × 3 → dense. Pre-check = max(long, pandas×3, dense) × safety.

Формула «K_pairs × row_size» была буквально **не той метрикой** — K_pairs
describe long row count, row_size — long row width; их произведение =
long table nbytes, но pivot dense ≠ long nbytes.

**Правило:** для каждого intermediate buffer — своя формула. Не
«универсальный `n_rows × row_width`». Pivot dense:
`K_index × K_columns × value_width`.

**Backlog (§11.6):** audit pre-checks в ops с multi-step materialization:
- `pivot_table` — fixed.
- `apply(fn)` — group materialization (per-group bounded). OK.
- `merge` — build + output. `JoinMemoryModel` покрывает.
- `dedup` — 1.5× table. OK.
- `groupby_aggregate` — output = K rows. `GroupByMemoryModel` покрывает.

**Связанные:** §13.4/158 (E2E respects inherent complexity), §13.4/157
(hasattr pyarrow types).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/159")


def main():
    print("=== v0.9.1f-fix3 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_pivot_precheck()
    add_to_pandas_precheck()
    append_decisions_159()

    print("\n=== pytest: pivot E2E (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pivot fast (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pivot_table.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT FAST REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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
        "v0.9.1f: pivot_table / melt / crosstab — long↔wide; "
        "§13.4/157-159 (pa.types.is_*; E2E respects complexity; "
        "dense pre-check K_index×K_columns×value_size; state_limit_bytes; "
        "to_pandas pre-check ×3)"])

    print("\n==> v0.9.1f-fix3 applied. v0.9.1 complete.")
    print()
    print("SQUASH + TAG v0.9.1:")
    print("  git reset --soft v0.9.0")
    print("  git commit -m 'v0.9.1: pipeline advanced'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.9.1] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.9.1 -m 'v0.9.1: pipeline advanced complete'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()