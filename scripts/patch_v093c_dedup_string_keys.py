#!/usr/bin/env python3
# v0.9.3c: P-D2 (S2) dedup string keys high-cardinality.
#
# v0.8.11a bound `1.5 * table.nbytes` extrapolated from int-key
# prototype (0.4x). Bench at K=N strings: 11.26x table.nbytes --
# 7.5x underestimate. For tight budget: partition passes 1.5x check
# but real peak is 11x -> OOM, not MemoryLimitExceeded.
#
# Fix: type-based multiplier.
#   - string subset: 12 * table.nbytes  (11.26x measured + safety)
#   - numeric subset: 2.5 * table.nbytes (1.97x measured + safety)
#
# Over-conservative for K<<N strings (measured ~0x). Documented in
# 11.5; sample-based K_est tightening -> v0.9.4 backlog.
#
# Run: uv run python scripts/patch_v093c_dedup_string_keys.py

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


# ============ Change 1: _local_dedup type-based multiplier ============

def fix_local_dedup_bound():
    print("\n=== dedup.py: _local_dedup type-based multiplier ===")
    p = Path("cadw/ops/dedup.py")
    s = p.read_text()
    if "12 * table.nbytes" in s:
        print("  already fixed - skip")
        return

    # Locate current check_state(1.5 * table.nbytes) block. Use line-based
    # extraction: find line with "1.5 * table.nbytes" and inspect context.
    lines = s.split("\n")
    target_idx = None
    for i, l in enumerate(lines):
        if "1.5 * table.nbytes" in l:
            target_idx = i
            break
    if target_idx is None:
        # Show context around 'dedup.arrow_native'
        print("  '1.5 * table.nbytes' not found. Context:")
        for i, l in enumerate(lines):
            if "dedup.arrow_native" in l or "check_state(" in l:
                print("    line " + str(i + 1) + ": " + repr(l))
        raise SystemExit("1.5 * table.nbytes anchor not found")

    # Identify the full check_state block: from "memory_guard.check_state("
    # line to matching ")". Assume single-line-ish call.
    # Walk backward to find start of statement.
    stmt_start = target_idx
    while stmt_start > 0 and "check_state" not in lines[stmt_start]:
        stmt_start -= 1
    if "check_state" not in lines[stmt_start]:
        raise SystemExit(
            "check_state start not found near '1.5 * table.nbytes' "
            "(line " + str(target_idx + 1) + ")"
        )
    # Find closing paren: track depth
    depth = 0
    stmt_end = stmt_start
    for j in range(stmt_start, len(lines)):
        depth += lines[j].count("(") - lines[j].count(")")
        if depth == 0 and j > stmt_start:
            stmt_end = j
            break
        if depth == 0 and j == stmt_start and ")" in lines[j] and "(" in lines[j]:
            # Single-line complete call
            stmt_end = j
            break

    # Rebuild block.
    # Comment update at stmt_start-1 if applicable, but simplest: replace
    # the whole stmt with multi-line version.
    new_block = [
        "    # v0.9.3c (P-D2, S2): type-based conservative estimate.",
        "    # Bench v0.9.2 audit:",
        "    #   - K=N strings, 400k rows: peak 11.26x table.nbytes",
        "    #     (dictionary_encode copies every string; Arrow StringArray",
        "    #     + int32 indices + BufferBuilder geometric growth).",
        "    #   - K=N int64: peak 1.97x table.nbytes.",
        "    #   - K<<N strings: ~0x (buffers reused).",
        "    # Prior 1.5x bound was extrapolated from int K<<N prototype ->",
        "    # 7.5x underestimate for strings at K=N.",
        "    _has_string_subset = any(",
        "        pa.types.is_string(table.schema.field(c).type)",
        "        or pa.types.is_large_string(table.schema.field(c).type)",
        "        for c in subset_cols",
        "    )",
        "    _dedup_peak_mult = 12 if _has_string_subset else 2.5",
        "    memory_guard.check_state(",
        "        _dedup_peak_mult * table.nbytes,",
        "        name=f\"dedup.arrow_native({partition_path.name})\",",
        "    )",
    ]
    lines[stmt_start:stmt_end + 1] = new_block
    p.write_text("\n".join(lines))
    compile_check(p)
    print("  OK (block lines " + str(stmt_start + 1) + "-" +
          str(stmt_end + 1) + ")")


# ============ Change 2: update AST guard in test_dedup_arrow_native ============

def update_stale_ast_guard():
    print("\n=== test_dedup_arrow_native.py: update AST guard ===")
    p = Path("tests/test_dedup_arrow_native.py")
    s = p.read_text()

    old = (
        '    assert "dedup.arrow_native" in src, (\n'
        '        "v0.8.11a: expect `dedup.arrow_native` guard name"\n'
        '    )\n'
        '    assert "1.5 * table.nbytes" in src, (\n'
        '        "v0.8.11a: expect `1.5 * table.nbytes` Arrow-native bound"\n'
        '    )'
    )

    new = (
        '    assert "dedup.arrow_native" in src, (\n'
        '        "v0.8.11a: expect `dedup.arrow_native` guard name"\n'
        '    )\n'
        '    # v0.9.3c (P-D2): 1.5x -> 12x (string) / 2.5x (numeric)\n'
        '    # based on measured peak (11.26x strings at K=N, 1.97x int).\n'
        '    assert "12 * table.nbytes" in src, (\n'
        '        "v0.9.3c: expect `12 * table.nbytes` for string subset"\n'
        '    )\n'
        '    assert "2.5 * table.nbytes" in src, (\n'
        '        "v0.9.3c: expect `2.5 * table.nbytes` for numeric subset"\n'
        '    )\n'
        '    assert "_dedup_peak_mult" in src, (\n'
        '        "v0.9.3c: expect _dedup_peak_mult variable"\n'
        '    )'
    )

    if new in s and old not in s:
        print("  already updated - skip")
        return
    if old not in s:
        print("  WARNING: AST guard block not found. Context:")
        idx = s.find("1.5 * table.nbytes")
        if idx >= 0:
            print("    " + repr(s[max(0, idx-200):idx+200]))
        raise SystemExit("AST guard anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Change 3: new test file ============

NEW_TESTS_LINES = [
    '# v0.9.3c: P-D2 dedup type-based estimator (S2 fix).',
    '# Prior 1.5x bound was 7.5x underestimate for string keys at K=N.',
    '',
    'from pathlib import Path',
    '',
    'import pytest',
    'import pyarrow as pa',
    'import pyarrow.parquet as pq',
    '',
    'from cadw import CadwDataFrame, MemoryGuard',
    'from cadw.core.memory_guard import MemoryLimitExceeded',
    '',
    '',
    'def make_df(tmp_path, data, chunk=200, name="L", mem_mb=20):',
    '    t = pa.table(data)',
    '    p = Path(tmp_path) / (name + ".parquet")',
    '    pq.write_table(t, p)',
    '    return CadwDataFrame.from_parquet(',
    '        p, Path(tmp_path) / ("ds_" + name), chunk_size=chunk,',
    '        memory_guard=MemoryGuard(max_memory_mb=mem_mb),',
    '        spill_dir=Path(tmp_path) / ("spill_" + name),',
    '    )',
    '',
    '',
    'def test_dedup_string_kN_raises_tight_budget(tmp_path):',
    '    # K=N strings. Partition ~20 KB. 12x = 240 KB peak estimate.',
    '    # Budget 200 KB -> state_limit 100 KB. check_state(12x) raises.',
    '    n = 1000',
    '    df = make_df(tmp_path, {',
    '        "k": ["k%04d" % i for i in range(n)],',
    '        "v": list(range(n)),',
    '    }, chunk=200, mem_mb=20)',
    '    df._memory_guard = MemoryGuard(max_memory_bytes=200_000)',
    '    with pytest.raises(MemoryLimitExceeded, match="arrow_native"):',
    '        df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)',
    '',
    '',
    'def test_dedup_string_kN_ample_budget_passes(tmp_path):',
    '    # Same data, ample budget (10 MB). All rows unique -> all kept.',
    '    n = 1000',
    '    df = make_df(tmp_path, {',
    '        "k": ["k%04d" % i for i in range(n)],',
    '        "v": list(range(n)),',
    '    }, chunk=200, mem_mb=10)',
    '    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)',
    '    out = tmp_path / "o.parquet"',
    '    result.write_parquet(out)',
    '    rows = pq.read_table(out).to_pylist()',
    '    assert len(rows) == n',
    '',
    '',
    'def test_dedup_int_kN_ample_budget_passes(tmp_path):',
    '    # K=N int64. All unique -> all kept.',
    '    n = 1000',
    '    df = make_df(tmp_path, {',
    '        "k": list(range(n)),',
    '        "v": list(range(n)),',
    '    }, chunk=200, mem_mb=10)',
    '    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)',
    '    assert result.count() == n',
    '',
    '',
    'def test_dedup_uses_12x_for_string_ast():',
    '    # AST: _local_dedup contains 12 * table.nbytes for string case.',
    '    import inspect',
    '    from cadw.ops import dedup as dedup_mod',
    '    src = inspect.getsource(dedup_mod._local_dedup)',
    '    assert "12 * table.nbytes" in src',
    '',
    '',
    'def test_dedup_uses_2_5x_for_numeric_ast():',
    '    # AST: _local_dedup contains 2.5 * table.nbytes for numeric case.',
    '    import inspect',
    '    from cadw.ops import dedup as dedup_mod',
    '    src = inspect.getsource(dedup_mod._local_dedup)',
    '    assert "2.5 * table.nbytes" in src',
    '',
    '',
    'def test_dedup_type_check_present():',
    '    # AST: _local_dedup checks pa.types.is_string for subset columns.',
    '    import inspect',
    '    from cadw.ops import dedup as dedup_mod',
    '    src = inspect.getsource(dedup_mod._local_dedup)',
    '    assert "is_string" in src',
    '    assert "subset_cols" in src',
    '',
    '',
    'def test_dedup_multiplier_variable():',
    '    # AST: _dedup_peak_mult variable drives check_state.',
    '    import inspect',
    '    from cadw.ops import dedup as dedup_mod',
    '    src = inspect.getsource(dedup_mod._local_dedup)',
    '    assert "_dedup_peak_mult" in src',
    '',
]


def write_new_tests():
    print("\n=== tests/test_dedup_estimator_v093c.py ===")
    p = Path("tests/test_dedup_estimator_v093c.py")
    p.write_text("\n".join(NEW_TESTS_LINES) + "\n")
    compile_check(p)
    print("  OK")


# ============ Change 4: CHANGELOG ============

CHANGELOG_LINES = [
    "### Исправлено (v0.9.3c - P-D2 dedup string keys, S2 fix)",
    "",
    "**P-D2 (S2, из v0.9.2 audit):** `_local_dedup` guard",
    "`check_state(1.5 * table.nbytes)` extrapolated from int-key",
    "prototype v0.8.11a (K<<N, measured 0.23x). Bench v0.9.2 audit",
    "(bench_dedup_phases.py, 400k rows, K=N strings, table 9.20 MB):",
    "",
    "- dictionary_encode +56.95 MB",
    "- group_by +86.14 MB",
    "- is_in +103.38 MB",
    "- filter +103.58 MB (peak)",
    "",
    "Peak / table = **11.26x** (strings at K=N). Int64 K=N = 1.97x.",
    "Prior 1.5x bound was **7.5x underestimate** for string K=N --",
    "OOM instead of MemoryLimitExceeded.",
    "",
    "**Fix:** type-based multiplier (conservative bench-backed):",
    "",
    "- **String subset:** 12 * table.nbytes (11.26x measured + 7% safety).",
    "- **Numeric subset:** 2.5 * table.nbytes (1.97x measured + 27%).",
    "",
    "`_has_string_subset` checked via `pa.types.is_string` /",
    "`is_large_string` on `subset_cols`. `_dedup_peak_mult` drives",
    "`check_state`.",
    "",
    "**Почему dictionary_encode даёт 11x peak:**",
    "- Копирует **каждое** string value (не только dictionary).",
    "- Arrow StringArray: data buffer + offset buffer (int32).",
    "- BufferBuilder geometric growth (~2x peak of final).",
    "- + int32 indices buffer (4 B / row).",
    "- + hash table for group_by (K entries).",
    "",
    "**Over-conservative:** K<<N strings (measured ~0x) still need 12x.",
    "Sample-based K_est tightening -> v0.9.4 backlog.",
    "",
    "**Тесты (v0.9.3c):** +7 fast (test_dedup_estimator_v093c.py):",
    "string K=N tight-budget raises, string K=N ample passes, int K=N",
    "ample passes, 3 AST guards (12x, 2.5x, type check), multiplier",
    "variable. Existing `test_dedup_removes_stopgap_5x` updated: 1.5x",
    "-> 12x/2.5x.",
    "",
    "- fast: **~1572** (+7). slow: **20** (unchanged).",
    "",
]


def append_changelog():
    print("\n=== CHANGELOG v0.9.3c ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3c - P-D2 dedup string keys" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import append_unreleased_entry
    entry = "\n".join(CHANGELOG_LINES)
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ Change 5: DECISIONS 13.4/175 + 11.5 ============

EPISODE_175_LINES = [
    "**Эпизод 175 (memory formula validity scope).** v0.9.3c: P-D2",
    "из v0.9.2 audit.",
    "",
    "**Bug.** `_local_dedup` guard `check_state(1.5 * table.nbytes)`",
    "был выведен из prototype v0.8.11a: int64 keys, K<<N, measured",
    "peak 0.23x table. Экстраполяция на **string keys, K=N** дала",
    "7.5x underestimate (real 11.26x). Consequence: partition passes",
    "1.5x check, but real peak 11x -> OOM вместо MemoryLimitExceeded.",
    "",
    "**Правило (13.4/175, расширение 128, 158, 173):**",
    "",
    "- **Memory formula scope must be measured, not extrapolated.**",
    "  Prototype on int K<<N не доказывает что formula holds для",
    "  string K=N. Kernel cost depends on: dtype (fixed-width vs",
    "  variable-length), cardinality (K vs N), and internal BufferBuilder",
    "  geometric growth (2x peak).",
    "- **Bench matrix for each guard:**",
    "  - dtype: int / float / string (at least)",
    "  - cardinality: K<<N / K=N/2 / K~N",
    "  - table size: small / medium / large",
    "  Формула должна быть conservative на **worst cell** матрицы,",
    "  не average.",
    "- **`pc.dictionary_encode` is not O(table.nbytes).** For variable-",
    "  length types it copies every value + builds offsets + indices +",
    "  hash table. Peak = table.nbytes + K*(avg_len+8)*2 + ",
    "  4*N (indices) + hash table + BufferBuilder 2x. Simplify: 12x",
    "  table.nbytes for string subset (measured, bench-backed).",
    "",
    "**Backlog (11.6):**",
    "- Audit all `check_state` formulas — grep `* table.nbytes`,",
    "  `* grouped_table.nbytes`, etc. Each should have bench matrix",
    "  source in docstring, not single-cell extrapolation.",
    "- Sample-based K_est tightening for dedup: for K<<N strings,",
    "  could use 2.5x instead of 12x via `count_distinct(sample)`",
    "  extrapolation. v0.9.4.",
    "",
    "**Связанные:** 13.4/128, 158, 173, **P-D1 v0.8.11a** (guarded",
    "int path only; strings edge case deferred to v0.9.2 audit).",
    "",
]


NOTE_11_5_LINES = [
    "**v0.9.3c (dedup string keys):**",
    "- `_local_dedup` guard: type-based multiplier.",
    "  - String subset: `12 * table.nbytes` (11.26x measured at K=N,",
    "    +7% safety).",
    "  - Numeric subset: `2.5 * table.nbytes` (1.97x measured at K=N,",
    "    +27% safety).",
    "- **Over-conservative for K<<N strings** (measured ~0x, checked",
    "  against 12x). Budget may need bump for string dedup even when",
    "  K small. Workaround: `encode_categorical` before dedup, or",
    "  v0.9.4 sample-based K_est estimator.",
    "- `pc.dictionary_encode` on all-unique strings copies every value",
    "  (not just dictionary) + Arrow BufferBuilder geometric growth",
    "  (~2x) + int32 indices + hash table. Peak 11x table, not 0.4x.",
    "",
]


def append_episode_13_4_175():
    print("\n=== DECISIONS 13.4/175 + 11.5 dedup note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    from cadw.patch_utils import find_section_anchor

    if "Эпизод 175" not in s:
        pos = find_section_anchor(s, "13.5")
        if pos is None:
            raise SystemExit("DECISIONS: 13.5 not found")
        block = "\n".join(EPISODE_175_LINES)
        s = s[:pos] + block + "\n" + s[pos:]
        p.write_text(s)
        print("  added 13.4/175")
    else:
        print("  13.4/175 already present")

    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.3c (dedup string keys)" in s:
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
    print("=== v0.9.3c: P-D2 dedup string keys ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_local_dedup_bound()
    update_stale_ast_guard()
    write_new_tests()
    append_changelog()
    append_episode_13_4_175()

    print("\n=== pytest: new estimator tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_dedup_estimator_v093c.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: existing dedup tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dedup_arrow_native.py",
         "tests/test_drop_duplicates.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DEDUP REGRESSION FAILED ***")
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
        "v0.9.3c: P-D2 dedup type-based multiplier S2 fix — string 12x "
        "table.nbytes (11.26x measured at K=N); numeric 2.5x (1.97x); "
        "_has_string_subset + _dedup_peak_mult; AST guard updated; "
        "13.4/175 (memory formula scope must be bench-validated, not "
        "extrapolated from one cell)"])

    print("\n==> v0.9.3c applied.")
    print("NEXT: v0.9.3d (C2 StdOp int truncation S2)")


if __name__ == "__main__":
    main()
