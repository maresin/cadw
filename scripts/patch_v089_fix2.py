#!/usr/bin/env python3
"""v0.8.9-fix2: final fix for v0.8.9 safety round.

ROOT CAUSE of the remaining failure (`test_uint64_hash_uses_pandas_path`):
  `_hash_column` uint64 branch used `pc.cast(XOR(arr, 2^63), int64)` —
  `pc.cast(safe=True)` raises ArrowInvalid for values >= INT64_MAX, and
  XOR with the sign bit pushes EVERY uint64 value into [2^63, 2^64) →
  guaranteed raise → silent fallback to `_python_hash_column`. Test spy
  on `_pandas_hash_column` never fires.

FIX (v0.8.9-fix2):
  - `pc.cast(arr, pa.int64(), safe=False)` — direct bitcast (two's-
    complement reinterpretation). No XOR, no raise, nulls propagate.
    Bijection over uint64 ↔ int64 → partition-disjoint invariant preserved.

SIDE FIX (was already broken by v0.8.9 P-SH2 but not yet exercised):
  `tests/test_hash_v081.py::test_uint64_falls_back_to_python` asserts the
  OLD (pre-P-SH2) behaviour "uint64 → Python fallback". After P-SH2,
  uint64 → pandas path. Renamed + assertion updated.

META (§13.4/139 — appended to episode 138):
  Third consecutive round where the fix introduced a fresh bug. Pattern:
  "fix code assumes library semantics instead of verifying via REPL". Same
  class as §13.4/128 (expectations) and §13.4/138 (audit formulas).
  Rule: any fix that adds/changes a library call MUST include a REPL
  snippet in the docstring showing the edge case (safe=True vs safe=False,
  null propagation, boundary values).

Run: uv run python scripts/patch_v089_fix2.py
"""
from __future__ import annotations

import os
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


# ============ Fix 1: uint64 branch in _hash_column ============

def fix_uint64_hash_column():
    print("\n=== cadw/ops/shuffle.py::_hash_column — uint64 bitcast (safe=False) ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''    # v0.8.9 (P-SH2): uint64 → XOR sign bit (bijection) → int64 → pandas hash.
    # v0.8.9-fix: Arrow-native (no numpy warning on nulls — to_numpy() gives
    # float64 NaN for nulls, astype(uint64) triggers invalid-cast warning).
    if pa.types.is_uint64(t):
        import pyarrow.compute as _pc
        try:
            flipped = _pc.bit_wise_xor(
                arr, pa.scalar(1 << 63, type=pa.uint64())
            )
            signed = _pc.cast(flipped, pa.int64())
            h = _pandas_hash_column(signed)
            if h is not None:
                return h
        except Exception:
            pass
        # extreme fallback: Python loop
        return _python_hash_column(arr)'''

    new = '''    # v0.8.9-fix2 (P-SH2): uint64 → int64 via `pc.cast(safe=False)` (bitcast),
    # then pandas C-hash. Prior v0.8.9-fix used XOR-2^63-then-cast, which
    # raised ArrowInvalid: pc.cast(safe=True) rejects values >= 2^63, and
    # XOR with sign bit pushes EVERY uint64 value into [2^63, 2^64) —
    # guaranteed raise → silent fallback to Python (invariant preserved
    # but 19× slower; the failure was hidden from tests by the spy).
    #
    # REPL check (§13.4/139 — fix adding/changing a library call must
    # document its boundary via a runnable snippet):
    #   >>> pc.cast(pa.array([2**63+1], type=pa.uint64()), pa.int64())
    #   ArrowInvalid: Integer value 9223372036854775809 not in range
    #   >>> pc.cast(pa.array([2**63+1], type=pa.uint64()), pa.int64(), safe=False)
    #   <pyarrow.Int64Array>[-9223372036854775807]
    # safe=False is a bitcast (two's-complement reinterpretation) — bijection
    # over uint64 ↔ int64, so same uint64 value → same int64 → same hash
    # (partition-disjoint invariant preserved). Nulls propagate unchanged
    # through pc.cast (no numpy roundtrip, no cast warning).
    if pa.types.is_uint64(t):
        import pyarrow.compute as _pc
        try:
            signed = _pc.cast(arr, pa.int64(), safe=False)
            h = _pandas_hash_column(signed)
            if h is not None:
                return h
        except Exception:
            pass
        # extreme fallback: Python loop
        return _python_hash_column(arr)'''

    s = replace_line_block(s, old, new, tag="uint64-fix2")
    p.write_text(s)
    compile_check(p)
    print("  OK: pc.cast(arr, int64, safe=False) (no XOR)")


# ============ Fix 2: test_hash_v081.py stale test ============

def fix_test_hash_v081():
    print("\n=== tests/test_hash_v081.py::test_uint64_falls_back_to_python "
          "→ test_uint64_uses_pandas_via_bitcast ===")
    p = Path("tests/test_hash_v081.py")
    s = p.read_text()

    old = '''def test_uint64_falls_back_to_python(monkeypatch):
    """uint64 can't fit Int64Dtype → must skip pandas, use Python."""
    from cadw.ops import shuffle as sh
    called_pandas = []
    called_python = []
    orig_pd = sh._pandas_hash_column
    orig_py = sh._python_hash_column

    def spy_pd(arr):
        called_pandas.append(str(arr.type))
        return orig_pd(arr)

    def spy_py(arr):
        called_python.append(str(arr.type))
        return orig_py(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy_pd)
    monkeypatch.setattr(sh, "_python_hash_column", spy_py)
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.uint64())], names=["k"]
    )
    compute_hashes(b, ["k"])
    # pandas called but returned None; Python called as fallback
    assert "uint64" in called_python'''

    new = '''def test_uint64_uses_pandas_via_bitcast(monkeypatch):
    """v0.8.9 (P-SH2): uint64 → int64 (safe=False bitcast) → pandas C-hash.

    Replaces pre-v0.8.9 `test_uint64_falls_back_to_python`, which asserted
    the OLD behaviour (uint64 → Python loop, because Int64Dtype can't hold
    uint64 values ≥ 2^63). After P-SH2, `_hash_column` bitcasts uint64 to
    int64 (bijection, partition-disjoint preserved) and routes through
    the same pandas path as int64 (19× faster than Python).

    v0.8.9-fix2: bitcast done via `pc.cast(arr, int64, safe=False)` — a
    direct two's-complement reinterpretation. Prior variant XOR-2^63-then-
    cast(safe=True) raised ArrowInvalid on every value (XOR pushes all
    into ≥ 2^63) and silently fell back to Python.
    """
    from cadw.ops import shuffle as sh
    called_pandas = []
    called_python = []
    orig_pd = sh._pandas_hash_column
    orig_py = sh._python_hash_column

    def spy_pd(arr):
        called_pandas.append(str(arr.type))
        return orig_pd(arr)

    def spy_py(arr):
        called_python.append(str(arr.type))
        return orig_py(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy_pd)
    monkeypatch.setattr(sh, "_python_hash_column", spy_py)
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.uint64())], names=["k"]
    )
    compute_hashes(b, ["k"])
    # P-SH2: uint64 bitcast to int64 → pandas path (NOT Python fallback).
    assert "int64" in called_pandas, (
        f"uint64 should bitcast to int64 and use pandas hash; "
        f"called_pandas={called_pandas}, called_python={called_python}"
    )
    assert "uint64" not in called_python, (
        f"uint64 should NOT fall back to Python hash; "
        f"called_python={called_python}"
    )'''

    s = replace_line_block(s, old, new, tag="test-hash-v081-uint64")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: DECISIONS §13.4/139 addendum to 138 ============

def append_decisions_139():
    print("\n=== DECISIONS.md: §13.4/139 addendum to episode 138 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "v0.8.9-fix2 addendum" in s:
        print("  already present — skip")
        return

    marker = (
        "**Связанные:** §13.4/25, §13.4/26 (тот же класс «trust в библиотеку»);\n"
        "§13.4/133–135 (тот же класс «hidden by green tests», но для type matrix);\n"
        "§3.10 (check before materialize — расширение на library calls)."
    )
    if marker not in s:
        print("  WARNING: §13.4/138 marker not found — skipping")
        return

    addendum = marker + '''

**v0.8.9-fix2 addendum (§13.4/139).** Три последовательных раунда правок
(v0.8.9 → v0.8.9-fix → v0.8.9-fix2) — каждый содержал свежий баг,
внесённый самим же фиксом:

- **v0.8.9**: AST guard путал `Table.from_batches` (материализует) с
  `Dataset.from_batches` (стримингует); широкий `ListComp` guard ловил
  `[f for f in schema]` (безопасно, K-bounded).
- **v0.8.9-fix**: `pc.cast(XOR(arr, 2^63) → int64)` — `safe=True` raises
  `ArrowInvalid` на всех значениях (XOR с sign bit переносит **всё**
  в `[2^63, 2^64)`); `pa.field(...)` по умолчанию `nullable=True` —
  test expected values захардкожены как non-nullable.
- **v0.8.9-fix2**: `pc.cast(arr, int64, safe=False)` = bitcast
  (two's-complement reinterpretation), bijection, nulls propagate;
  заменил XOR-вариант. Добавил REPL-сниппет в docstring `_hash_column`.

**Класс ошибки — тот же, что §13.4/128** («test expectations authored, not
derived») и §13.4/138 (audit formulas assumed, not measured). Здесь
«**fix semantics assumed, not verified**». Правило: **если fix добавляет
или меняет вызов библиотечной функции — REPL-сниппет в docstring
обязателен**, показывающий edge case: `safe=True` vs `safe=False`,
null propagation, boundary values (≥ `INT64_MAX`, empty, all-null).
Это защищает от ровно того класса, что случился трижды подряд.'''
    s = s.replace(marker, addendum, 1)
    p.write_text(s)
    print("  OK: §13.4/139")


# ============ Fix 4: CHANGELOG — update P-SH2 description ============

def fix_changelog_p_sh2():
    print("\n=== CHANGELOG.md: update P-SH2 description ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    old = '''- **P-SH2 (S2)** — `_hash_column` uint64 → XOR 2^63 → int64 bitcast → pandas
  C-hash (19× faster than Python fallback). Preserves partition-disjoint
  invariant (bijection, same value → same hash).'''

    new = '''- **P-SH2 (S2)** — `_hash_column` uint64 → `pc.cast(arr, int64, safe=False)`
  bitcast → pandas C-hash (19× faster than Python fallback). Bitcast =
  two's-complement reinterpretation, bijection over uint64 ↔ int64 →
  same value → same hash (partition-disjoint preserved). Nulls propagate
  through `pc.cast` unchanged. (v0.8.9-fix2: prior XOR-2^63-then-cast
  variant raised `ArrowInvalid` because values ≥ 2^63 overflow
  `pc.cast(safe=True)` — every uint64 value, since XOR with sign bit
  pushes all into [2^63, 2^64). Fix: skip XOR, cast directly with
  `safe=False`. REPL snippet documented in `_hash_column` docstring.)'''

    if old not in s:
        print("  WARNING: P-SH2 paragraph not found — skipping (CHANGELOG "
              "may have been restructured)")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.8.9-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    fix_uint64_hash_column()
    fix_test_hash_v081()
    append_decisions_139()
    fix_changelog_p_sh2()

    print("\n=== pytest: new tests + test_hash_v081 + uint64 regressions ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_pingis_safety.py",
         "tests/test_hash_v081.py",
         "tests/test_shuffle.py",
         "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** RED ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.9-fix2 (red)"], check=False)
        sys.exit(1)

    print("\n=== regression: join + dedup + groupby ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join.py", "tests/test_join_streaming.py",
         "tests/test_join_oom_guard.py", "tests/test_drop_duplicates.py",
         "tests/test_groupby_v2.py", "tests/test_mustfix_local_groupby.py",
         "tests/test_mustfix_skew.py", "tests/test_mustfix_dict_check.py",
         "-q"],
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
        "v0.8.9-fix2: uint64 bitcast via pc.cast(safe=False); "
        "test_hash_v081::test_uint64_falls_back_to_python updated for "
        "P-SH2 semantics; §13.4/139 addendum (fix semantics = verify via "
        "REPL, same class as §13.4/128)"])

    print("\n==> v0.8.9-fix2 applied. Ready to squash → tag v0.8.9.")


if __name__ == "__main__":
    main()