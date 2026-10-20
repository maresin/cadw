#!/usr/bin/env python3
# v0.9.3d: C2 (S2) StdOp int-column truncation fix.
#
# Bug: StdOp.update used pa.scalar(mean_b, type=arr.type) for subtract.
# For int64 arr: pa.scalar(2.5, type=int64) -> 2 (silent truncation).
# M2 computed from wrong deviations -> wrong std when batch mean
# non-integral. Examples:
#   [1,2,3,4] -> buggy 1.4142 (sqrt(2)), correct 1.2910 (sqrt(5/3))
#   [1,2]     -> buggy 1.0,             correct 0.7071 (sqrt(0.5))
# Float columns unaffected (pa.scalar(float, type=float64) is exact).
#
# Public API: cadw.StdOp, cadw.reduce(..., StdOp(...)).
# groupby().agg("std") uses Arrow group_by().aggregate (native, no
# StdOp) -> unaffected. ML fit_* don't use StdOp.
#
# Semantic change for int columns (correctness fix; prior value was
# mathematically wrong). CHANGELOG flags this. No existing test
# pinned the buggy value (both test_std_invariant and
# test_std_precision_large_offset use float data).
#
# Run: uv run python scripts/patch_v093d_stdop_int_fix.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import py_compile; py_compile.compile('" + str(path) +
         "', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("COMPILE FAIL: " + str(path))
        print(r.stderr[-800:])
        sys.exit(1)
    print("  compile OK: " + str(path))


# ============ Change 1: StdOp.update cast to float64 ============

def fix_stdop_update():
    print("\n=== reduce.py: StdOp.update cast to float64 ===")
    p = Path("cadw/ops/reduce.py")
    s = p.read_text()

    if "arr_f = pc.cast(arr, pa.float64())" in s:
        print("  already fixed - skip")
        return

    old = (
        "    def update(self, state: State, batch: pa.RecordBatch) -> State:\n"
        "        arr = batch.column(self.column)\n"
        "        n_b = len(arr) - arr.null_count\n"
        "        if n_b == 0:\n"
        "            return state\n"
        "        mean_b = pc.mean(arr, skip_nulls=True).as_py()\n"
        "        diff = pc.subtract(arr, pa.scalar(mean_b, type=arr.type))\n"
        "        M2_b = pc.sum(pc.multiply(diff, diff), skip_nulls=True).as_py() or 0.0\n"
    )

    new = (
        "    def update(self, state: State, batch: pa.RecordBatch) -> State:\n"
        "        arr = batch.column(self.column)\n"
        "        n_b = len(arr) - arr.null_count\n"
        "        if n_b == 0:\n"
        "            return state\n"
        "        # v0.9.3d (C2, S2): cast to float64 before subtract.\n"
        "        # Prior code used pa.scalar(mean_b, type=arr.type) -- for\n"
        "        # int64 arr, pa.scalar(2.5, type=int64) silently truncates\n"
        "        # to 2 -> wrong std on non-integral batch mean. E.g.\n"
        "        # [1,2,3,4] -> buggy sqrt(2) ~= 1.414, correct sqrt(5/3)\n"
        "        # ~= 1.291 (ddof=1). Float columns unaffected (float64\n"
        "        # scalar is exact). See 13.4/176.\n"
        "        arr_f = pc.cast(arr, pa.float64())\n"
        "        mean_b = pc.mean(arr_f, skip_nulls=True).as_py()\n"
        "        diff = pc.subtract(\n"
        "            arr_f, pa.scalar(float(mean_b), type=pa.float64())\n"
        "        )\n"
        "        M2_b = pc.sum(pc.multiply(diff, diff), skip_nulls=True).as_py() or 0.0\n"
    )

    if old not in s:
        print("  WARNING: StdOp.update block not found. Context:")
        idx = s.find("class StdOp")
        if idx >= 0:
            print("    " + repr(s[idx:idx+900]))
        raise SystemExit("StdOp.update anchor not found")

    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Change 2: new tests ============

TESTS_LINES = [
    '# v0.9.3d: C2 (S2) StdOp int-column correctness.',
    '# Prior bug: int-column mean truncated -> wrong std when batch',
    '# mean non-integral. e.g. [1,2,3,4] int64: buggy sqrt(2),',
    '# correct sqrt(5/3).',
    '',
    'import numpy as np',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.ops.reduce import reduce, StdOp',
    '',
    '',
    'def make_iter(vals, batch_size, type_=pa.int64()):',
    '    for i in range(0, len(vals), batch_size):',
    '        yield pa.RecordBatch.from_arrays(',
    '            [pa.array(vals[i:i+batch_size], type=type_)],',
    '            names=["x"],',
    '        )',
    '',
    '',
    'def test_std_int_column_correct_known():',
    '    # [1,2,3,4] int64: mean 2.5, M2=5, ddof=1 -> sqrt(5/3).',
    '    # Buggy version: mean truncated to 2, M2=6, sqrt(2).',
    '    data = [1, 2, 3, 4]',
    '    got = reduce(',
    '        make_iter(data, 2), StdOp("x"),',
    '        MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    expected = float(np.std(np.array(data, dtype=np.float64), ddof=1))',
    '    assert got == pytest.approx(expected, rel=1e-12)',
    '    # Sanity: must not equal the buggy truncated value.',
    '    assert not (abs(got - 1.4142135623730951) < 1e-9)',
    '',
    '',
    'def test_std_int_vs_float_same():',
    '    # Same data as int vs float -> identical std.',
    '    int_data = [1, 2, 3, 4]',
    '    float_data = [1.0, 2.0, 3.0, 4.0]',
    '    got_int = reduce(',
    '        make_iter(int_data, 2, type_=pa.int64()),',
    '        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    got_float = reduce(',
    '        make_iter(float_data, 2, type_=pa.float64()),',
    '        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    assert got_int == pytest.approx(got_float, rel=1e-12)',
    '',
    '',
    '@pytest.mark.parametrize("data,batch_size", [',
    '    ([1, 2, 3, 4], 2),',
    '    ([1, 2], 1),',
    '    ([0, 0, 0, 10], 2),',
    '    ([3, 7], 2),',
    '    ([10, 20, 30, 40, 50], 3),',
    '])',
    'def test_std_int_matches_numpy(data, batch_size):',
    '    got = reduce(',
    '        make_iter(data, batch_size), StdOp("x"),',
    '        MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    expected = float(np.std(np.array(data, dtype=np.float64), ddof=1))',
    '    assert got == pytest.approx(expected, rel=1e-12)',
    '',
    '',
    'def test_std_float_unchanged():',
    '    # Regression: float path still works (was always correct).',
    '    data = [1.0, 2.0, 3.0, 4.0]',
    '    got = reduce(',
    '        make_iter(data, 2, type_=pa.float64()),',
    '        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    expected = float(np.std(data, ddof=1))',
    '    assert got == pytest.approx(expected, rel=1e-12)',
    '',
    '',
    'def test_std_int_with_nulls():',
    '    # Int64 with nulls; nulls skipped, correct ddof=1.',
    '    data = [1, None, 3, 4, None, 6]',
    '    got = reduce(',
    '        make_iter(data, 2), StdOp("x"),',
    '        MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    valid = [1.0, 3.0, 4.0, 6.0]',
    '    expected = float(np.std(valid, ddof=1))',
    '    assert got == pytest.approx(expected, rel=1e-12)',
    '',
    '',
    'def test_std_int32_type():',
    '    # int32 column too.',
    '    data = [1, 2, 3, 4]',
    '    got = reduce(',
    '        make_iter(data, 2, type_=pa.int32()),',
    '        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,',
    '    )',
    '    expected = float(np.std(data, ddof=1))',
    '    assert got == pytest.approx(expected, rel=1e-12)',
    '',
    '',
    'def test_std_ast_cast_to_float64():',
    '    # AST: StdOp.update casts arr to float64 before subtract.',
    '    import inspect',
    '    from cadw.ops import reduce as reduce_mod',
    '    src = inspect.getsource(reduce_mod.StdOp.update)',
    '    assert "pc.cast(arr, pa.float64())" in src',
    '    assert "pa.float64()" in src',
    '',
]


def write_tests():
    print("\n=== tests/test_stdop_int_correctness.py ===")
    p = Path("tests/test_stdop_int_correctness.py")
    p.write_text("\n".join(TESTS_LINES) + "\n")
    compile_check(p)
    print("  OK")


# ============ Change 3: CHANGELOG ============

CHANGELOG_LINES = [
    "### Исправлено (v0.9.3d - StdOp int-column truncation, S2 fix)",
    "",
    "**C2 (S2, из v0.9.2 audit):** `StdOp.update` использовал",
    "`pa.scalar(mean_b, type=arr.type)` для subtract. Для int64 arr:",
    "`pa.scalar(2.5, type=int64)` — silent truncation до 2. M2 computed",
    "from wrong deviations -> wrong std когда batch mean non-integral.",
    "",
    "**Примеры:**",
    "",
    "| Input (int64) | Buggy | Correct (numpy ddof=1) |",
    "|---|---|---|",
    "| [1,2,3,4] | 1.4142 | 1.2910 |",
    "| [1,2] | 1.0 | 0.7071 |",
    "| [0,0,0,10] | 5.0332 | 5.0 |",
    "| [1,2,3,4,5] | 1.5811 | 1.5811 (integral mean -> OK) |",
    "",
    "Float columns unaffected (`pa.scalar(float, type=float64)` exact).",
    "`groupby().agg(\"std\")` — Arrow native `group_by().aggregate`, не",
    "задет. ML `fit_*` — не использует `StdOp`.",
    "",
    "**Fix:** cast to float64 before subtract.",
    "",
    "```python",
    "arr_f = pc.cast(arr, pa.float64())",
    "diff = pc.subtract(arr_f, pa.scalar(float(mean_b), type=pa.float64()))",
    "```",
    "",
    "**Semantic change (int columns):** correct value where previous was",
    "mathematically wrong. CHANGELOG flags as breaking for downstream",
    "code that relied on incorrect result. No existing test pinned the",
    "buggy value (both `test_std_invariant` and",
    "`test_std_precision_large_offset` use float data).",
    "",
    "**Правило (13.4/176):** при subtract/add со scalar из mean/median —",
    "**всегда** cast array to float64. Int arithmetic truncates silently.",
    "Same class as 13.4/163 (domain validation), 13.4/175 (memory formula",
    "scope) — hidden mismatch between formula и implementation.",
    "",
    "**Тесты (v0.9.3d):** +8 fast (`test_stdop_int_correctness.py`):",
    "known [1,2,3,4] test (+ sanity not buggy), int==float, 5 param cases",
    "vs numpy, float unchanged, nulls, int32, AST guard.",
    "",
    "- fast: **~1580** (+8). slow: **20** (unchanged).",
    "",
]


def append_changelog():
    print("\n=== CHANGELOG v0.9.3d ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3d - StdOp int-column truncation" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import append_unreleased_entry
    entry = "\n".join(CHANGELOG_LINES)
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ Change 4: DECISIONS 13.4/176 + 11.5 ============

EPISODE_176_LINES = [
    "**Эпизод 176 (int arithmetic silent truncation).** v0.9.3d: C2",
    "из v0.9.2 audit.",
    "",
    "**Bug.** `StdOp.update`:",
    "```python",
    "mean_b = pc.mean(arr, skip_nulls=True).as_py()  # 2.5",
    "diff = pc.subtract(arr, pa.scalar(mean_b, type=arr.type))",
    "```",
    "Для int64 arr: `pa.scalar(2.5, type=int64)` = **2** (silent",
    "truncation, no warning). M2 computed from wrong deviations ->",
    "wrong std. `[1,2,3,4]`: buggy sqrt(2) ~= 1.414, correct",
    "sqrt(5/3) ~= 1.291.",
    "",
    "**Почему не поймали в v0.1.0 (исходный):** `test_reduce.py::",
    "test_std_invariant` использует `make_iter(data, bs)` с default",
    "`type=pa.float64()` — int-путь не тестировался. Float path всегда",
    "правильный (`pa.scalar(2.5, type=float64)` exact).",
    "",
    "**Правило (13.4/176, расширение 163, 175):**",
    "",
    "- **Int arithmetic truncates silently.** `pa.scalar(float, type=int)",
    "  обрезает. `np.array(int) / n` даёт float. `pc.divide(int, int)` —",
    "  float. Но `pa.scalar(x, type=int_type)` — truncate.",
    "- **Casting policy при arithmetic с derived values** (mean, median,",
    "  quantile): всегда cast array to float64 перед scalar op.",
    "  Derived values fractional by nature; using original int type for",
    "  scalar is always wrong.",
    "- **Test matrix для числовых ops:** int64 / float64 / int32",
    "  (at least). Float-only tests hide truncation bugs.",
    "",
    "**Backlog (11.6):**",
    "- Grep `pa.scalar(.*type=arr.type)` в cadw/ — каждое место",
    "  проверить: если scalar — derived value (mean/median/etc), нужен",
    "  explicit float64 cast. `PredicateFilter.mask_for_batch` — separate",
    "  case (C3, user value не derived; correct behavior — validate type",
    "  upfront).",
    "- Audit numeric ops (`SumOp`, `MinMaxOp`, `MeanOp`, `CovarianceOp`,",
    "  `HistogramOp`) — float64 cast присутствует везде кроме `StdOp`",
    "  (fixed здесь).",
    "",
    "**Связанные:** 13.4/163 (domain validation), 175 (memory formula",
    "scope), P-CV1 (small-N hides O(N) bugs) — same class",
    "«implementation mismatch с contract», не поймали small-numeric",
    "test'ом.",
    "",
]


NOTE_11_5_LINES = [
    "**v0.9.3d (StdOp int correctness):**",
    "- `StdOp` теперь корректно обрабатывает int колонки (cast to",
    "  float64 перед subtract). До v0.9.3d возвращал wrong std когда",
    "  batch mean был non-integral. Semantic change — downstream code,",
    "  relied on incorrect result, needs update.",
    "- Float колонки: без изменений (были корректны).",
    "- `groupby().agg(\"std\")`: без изменений (Arrow native group_by,",
    "  не использует `StdOp`).",
    "",
]


def append_episode_13_4_176():
    print("\n=== DECISIONS 13.4/176 + 11.5 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    from cadw.patch_utils import find_section_anchor

    if "Эпизод 176" not in s:
        pos = find_section_anchor(s, "13.5")
        if pos is None:
            raise SystemExit("DECISIONS: 13.5 not found")
        block = "\n".join(EPISODE_176_LINES)
        s = s[:pos] + block + "\n" + s[pos:]
        p.write_text(s)
        print("  added 13.4/176")
    else:
        print("  13.4/176 already present")

    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.3d (StdOp int correctness)" in s:
        print("  11.5 note already present")
        return
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        raise SystemExit("DECISIONS: 11.5 not found")
    note = "\n".join(NOTE_11_5_LINES)
    s = s[:pos] + note + "\n" + s[pos:]
    p.write_text(s)
    print("  added 11.5 note")


# ============ main ============

def main():
    print("=== v0.9.3d: C2 StdOp int truncation ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_stdop_update()
    write_tests()
    append_changelog()
    append_episode_13_4_176()

    print("\n=== pytest: new int-correctness tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_stdop_int_correctness.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: existing reduce tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py",
         "tests/test_reduce_state_bytes.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: gbdt/pca regression (use numeric) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_pca.py",
         "tests/test_scalers.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NUMERIC REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
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
        "v0.9.3d: C2 StdOp int-column truncation S2 fix — cast to "
        "float64 before subtract; int64 [1,2,3,4] buggy 1.414 -> correct "
        "1.291; semantic change flagged; 13.4/176 (int arithmetic "
        "truncation; float-only tests hide this)"])

    print("\n==> v0.9.3d applied.")
    print("NEXT: v0.9.3e (C1 dead code, trivial)")


if __name__ == "__main__":
    main()
