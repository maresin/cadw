#!/usr/bin/env python3
"""v0.9.1f-fix1: bit_width on variable-length types + crosstab count values.

BUG 1 (§13.4/157): `pivot_table` row_size estimate used
  `if hasattr(t, "bit_width"): t.bit_width`
`hasattr` returns True for ALL types (attribute exists), but
`.bit_width` raises `ValueError: Non-fixed width type` for string /
binary / list / struct. Need explicit `pa.types.is_*` check.

FIX 1: use `cadw.core.models.type_size` (handles string via sample
default; conservative fallback for others) — no bit_width access.
For simplicity in pivot_table context (no sample for string): default
to 50 B for string, 8 B for fixed-width, 8 B fallback.

BUG 2 (§13.4/157): `crosstab` delegates to
`pivot_table(values=index[0], aggfunc='count')`. But pivot_table
validation rejects `values in index_cols` — even for count (where
specific value doesn't matter). 3 crosstab tests fail.

FIX 2: for `aggfunc='count'` only — allow `values` to be in index_cols
(count of any non-null value; value itself unused).

META (§13.4/157): pyarrow type attribute introspection:
- `hasattr(t, "bit_width")` — MISLEADING (attr exists, accessing raises).
- `pa.types.is_*(t)` — only reliable feature detection.
- Same for `num_fields`, `value_type`, `field`, etc. — many raise
  ValueError on wrong type.

GUARDS: pivot tests + guards + fast + slow.

Run: uv run python scripts/patch_v091f_fix1.py
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


# ============ Fix 1: bit_width safe ============

def fix_bit_width():
    print("\n=== cadw/dataframe.py::pivot_table — safe row_size estimate ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        # Estimated row_size for grouped table (index + columns + agg value)
        row_size = sum(
            self._dataset.schema.field(c).type.bit_width // 8
            for c in index_cols + [columns, values]
            if hasattr(self._dataset.schema.field(c).type, "bit_width")
        )
        if row_size <= 0:
            row_size = 16  # conservative default
        est_bytes = k_pairs_est * row_size * 2'''

    new = '''        # §13.4/157: explicit pa.types.is_* — `hasattr(t, "bit_width")`
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
        est_bytes = k_pairs_est * row_size * 2'''

    s = replace_line_block(s, old, new, tag="safe-bit-width")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: crosstab count allows values in index ============

def fix_crosstab_count():
    print("\n=== cadw/dataframe.py::pivot_table — count allows value=index ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        if values in index_cols or values == columns:
            raise ValueError(
                f"pivot_table: values {values!r} must be distinct from "
                f"index/columns"
            )
        if aggfunc not in ("mean", "sum", "count", "min", "max"):
            raise ValueError(
                f"pivot_table: aggfunc must be one of mean/sum/count/"
                f"min/max; got {aggfunc!r}"
            )'''

    new = '''        if aggfunc not in ("mean", "sum", "count", "min", "max"):
            raise ValueError(
                f"pivot_table: aggfunc must be one of mean/sum/count/"
                f"min/max; got {aggfunc!r}"
            )
        # v0.9.1f-fix1 (§13.4/157): `count` doesn't use value semantics —
        # only non-null counting. Allow values ∈ index_cols for
        # aggfunc='count' (used by crosstab: values=index[0]).
        # Other aggs (mean/sum/min/max) require value ≠ index/columns.
        if aggfunc != "count":
            if values in index_cols or values == columns:
                raise ValueError(
                    f"pivot_table: values {values!r} must be distinct "
                    f"from index/columns (aggfunc={aggfunc!r})"
                )
        if values == columns:
            # values=columns makes no sense for any agg
            raise ValueError(
                f"pivot_table: values {values!r} == columns {columns!r}"
            )'''

    s = replace_line_block(s, old, new, tag="crosstab-count-value")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: §13.4/157 ============

def append_decisions_157():
    print("\n=== DECISIONS.md: §13.4/157 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 157" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/157 (hasattr on pyarrow types misleading; count "
        s += "values=index).** Fix: pa.types.is_*, count allows values "
        s += "in index_cols.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 157 (hasattr on pyarrow types misleading; count
values=index).** v0.9.1f: 9 tests fail — 6 на `bit_width`, 3 на
`crosstab values=index`.

**Bug 1 (bit_width):** `if hasattr(t, "bit_width"): t.bit_width` —
`hasattr` returns True для всех типов (атрибут **есть** в dir()),
но accessing `.bit_width` raises `ValueError: Non-fixed width type`
на string/binary/list/struct. Проверка не отличает «fixed-width» от
«variable-length».

Fix: explicit `pa.types.is_integer/is_floating/is_timestamp/is_date/
is_time/is_boolean` → `.bit_width`; string/binary → 50 B default;
остальное → 8 B fallback.

**Bug 2 (crosstab):** `crosstab` delegates to
`pivot_table(values=index[0], aggfunc='count')`. `pivot_table` validation
отклоняет `values in index_cols` без исключения для `count`. Но для
`count` values **не используется** (только non-null counting).

Fix: allow `values in index_cols` **только для `aggfunc='count'`**.
Для `mean/sum/min/max` — strict check остаётся (values должны быть
distinct, иначе результат — тождественный).

**Урок (§13.4/157, расширение 154-156):** `hasattr(t, "attr")` на
**pyarrow types** — MISLEADING для feature detection. Многие атрибуты
существуют на всех типах, но raise при wrong-type access:
- `bit_width` → raises на variable-length.
- `num_fields` → raises на non-struct.
- `value_type` → raises на non-list.
- `field(i)` → raises на non-struct.

Правило: для pyarrow type introspection — **только** `pa.types.is_*(t)`
или `try: t.attr except ValueError`. `hasattr` — никогда.

**Backlog (§11.6):** audit `hasattr` на pyarrow types в `cadw/`:
- `cadw/dataframe.py` — fixed (pivot_table).
- `cadw/core/models.py::type_size` — uses `pa.types.is_*` — OK.
- `cadw/core/memory_guard.py::estimate_row_size_from_schema` —
  uses `pa.types.is_*` — OK.
- `cadw/ops/resample.py::_TS_UNIT_FACTOR[t.unit]` — unit access only
  after `is_timestamp(t)` — OK.
- `cadw/ml/gbdt.py` — no type introspection in hot path.

**Связанные:** §13.4/154 (def-prefix), /155 (module-level AST),
/156 (blanket-walk).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/157")


def main():
    print("=== v0.9.1f-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_bit_width()
    fix_crosstab_count()
    append_decisions_157()

    print("\n=== pytest: pivot tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pivot_table.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pivot E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
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
        "v0.9.1f: pivot_table / melt / crosstab — long↔wide transforms; "
        "melt streaming (uniform-dtype), pivot_table fail-fast (K_pairs "
        "estimate + check_state before groupby+to_pandas), crosstab alias; "
        "§11.5 sparse limits; §13.4/157 (pa.types.is_* not hasattr for "
        "type introspection; count allows values in index)"])

    print("\n==> v0.9.1f-fix1 applied. v0.9.1 complete.")
    print()
    print("SQUASH + TAG v0.9.1:")
    print("  git reset --soft v0.9.0")
    print("  git commit -m 'v0.9.1: DAG pipeline, save/load, "
          "ColumnTransformer, apply(fn), pivot/melt/crosstab'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.9.1] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.9.1 -m 'v0.9.1: pipeline advanced'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()