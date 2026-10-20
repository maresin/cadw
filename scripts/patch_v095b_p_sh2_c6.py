#!/usr/bin/env python3
# v0.9.5b: P-SH2 (uint64 silent fallback) + C6 (AST guard extends to
# viz/_aggregate.py + core/memory_guard.py).
#
# P-SH2 (S3, from v0.9.2 audit): shuffle._hash_column uint64 branch
# uses `except Exception: pass` before `_python_hash_column` — silent
# fallback to 30x slower Python hash. Same class as K1.2 (fixed with
# warning). Fix: logger.warning on cast failure + fallback.
#
# C6 (S3): AUDITED_FILES in test_arrow_call_memory_guard.py covers
# ops/*.py + dataframe.py + indexers.py. Missing viz/_aggregate.py
# (uses `sort_by` in line_agg, `df.quantile` in box_stats) and
# core/memory_guard.py (uses `dictionary_encode` in
# estimate_unique_keys). Both guarded internally but not AST-enforced.
# Fix: extend AUDITED_FILES + whitelist precomputed violations.
#
# Run: uv run python scripts/patch_v095b_p_sh2_c6.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


# ============ 1. P-SH2: uint64 warning ============

def fix_uint64_warning():
    print("\n=== shuffle.py: uint64 silent fallback warning ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()
    if "uint64 Python hash fallback" in s:
        print("  already fixed - skip")
        return

    old = (
        "    if pa.types.is_uint64(t):\n"
        "        import pyarrow.compute as _pc\n"
        "        try:\n"
        "            signed = _pc.cast(arr, pa.int64(), safe=False)\n"
        "            h = _pandas_hash_column(signed)\n"
        "            if h is not None:\n"
        "                return h\n"
        "        except Exception:\n"
        "            pass\n"
        "        # extreme fallback: Python loop\n"
        "        return _python_hash_column(arr)\n"
    )
    new = (
        "    if pa.types.is_uint64(t):\n"
        "        import pyarrow.compute as _pc\n"
        "        try:\n"
        "            signed = _pc.cast(arr, pa.int64(), safe=False)\n"
        "            h = _pandas_hash_column(signed)\n"
        "            if h is not None:\n"
        "                return h\n"
        "            # _pandas_hash_column already warned (K1.2)\n"
        "        except Exception as exc:\n"
        "            # v0.9.5b (P-SH2): silence hid 30x perf fallback.\n"
        "            # Same class as K1.2 (K1.2._pandas_hash_column).\n"
        "            logger.warning(\n"
        "                '_hash_column: uint64 bitcast failed (%r); '\n"
        "                'falling back to Python hash loop (30x slower)',\n"
        "                exc,\n"
        "            )\n"
        "        # extreme fallback: Python loop\n"
        "        logger.warning(\n"
        "            '_hash_column: uint64 Python hash fallback '\n"
        "            '(slow path); check pyarrow cast/pandas API'\n"
        "        )\n"
        "        return _python_hash_column(arr)\n"
    )
    if old not in s:
        raise SystemExit("uint64 block anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. C6: extend AUDITED_FILES + whitelist ============

def extend_audited_files():
    print("\n=== test_arrow_call_memory_guard.py: extend AUDITED_FILES ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "viz/_aggregate.py" in s and "core/memory_guard.py" in s:
        print("  already extended - skip")
        return

    old = (
        "AUDITED_FILES = (\n"
        "    list((CADW_ROOT / \"ops\").glob(\"*.py\"))\n"
        "    + [CADW_ROOT / \"dataframe.py\", CADW_ROOT / \"indexers.py\"]\n"
        ")\n"
    )
    new = (
        "AUDITED_FILES = (\n"
        "    list((CADW_ROOT / \"ops\").glob(\"*.py\"))\n"
        "    + [CADW_ROOT / \"dataframe.py\", CADW_ROOT / \"indexers.py\"]\n"
        "    # v0.9.5b (C6): extend to viz/_aggregate + memory_guard.\n"
        "    # Both have internally-guarded high-risk calls (sort_by on\n"
        "    # K rows in line_agg; df.quantile in box_stats; dictionary_\n"
        "    # encode on sample in estimate_unique_keys). Whitelisted\n"
        "    # below with rationale.\n"
        "    + [CADW_ROOT / \"viz\" / \"_aggregate.py\",\n"
        "       CADW_ROOT / \"core\" / \"memory_guard.py\"]\n"
        ")\n"
    )
    if old not in s:
        raise SystemExit("AUDITED_FILES anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def add_whitelist_entries():
    print("\n=== test_arrow_call_memory_guard.py: whitelist C6 violations ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "cadw/viz/_aggregate.py:line_agg" in s:
        print("  already whitelisted - skip")
        return

    # Insert after existing describe entry (K1.1 multi-line format).
    anchor_marker = '"cadw/dataframe.py:CadwDataFrame.describe":'
    idx = s.find(anchor_marker)
    if idx < 0:
        raise SystemExit("describe whitelist anchor not found")
    # Find the value-close (next line ending with comma at 8-space indent).
    lines = s.split("\n")
    line_idx = None
    for i, line in enumerate(lines):
        if anchor_marker in line:
            line_idx = i
            break
    if line_idx is None:
        raise SystemExit("describe key line not found in split")
    # Value line is next (or same line if single-line). We insert after
    # the closing comma line.
    end_idx = line_idx
    if not lines[line_idx].strip().endswith(","):
        # multi-line: find value-close
        for j in range(line_idx + 1, min(line_idx + 5, len(lines))):
            if lines[j].strip().endswith(","):
                end_idx = j
                break
        else:
            raise SystemExit("describe value-close not found")
    # Insert new entries after end_idx
    new_entries = [
        "    # v0.9.5b (C6): viz/_aggregate.py high-risk calls.",
        "    # line_agg sorts K-row aggregate table (bounded by K, not N).",
        "    # box_stats delegates to df.quantile (guarded via external_sort).",
        '    "cadw/viz/_aggregate.py:line_agg":',
        '        "table.sort_by on K rows (bounded by group count)",',
        '    "cadw/viz/_aggregate.py:box_stats":',
        '        "delegates to df.quantile (guarded via external_sort)",',
        "    # v0.9.5b (C6): core/memory_guard.py high-risk calls.",
        "    # estimate_unique_keys: dictionary_encode on sample ≤10k.",
        '    "cadw/core/memory_guard.py:MemoryGuard.estimate_unique_keys":',
        '        "pc.dictionary_encode on ≤10k sample batch (bounded)",',
    ]
    for offset, entry in enumerate(new_entries):
        lines.insert(end_idx + 1 + offset, entry)
    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK")


# ============ 3. Tests ============

def add_uint64_warning_test():
    print("\n=== test_hash_v081.py: uint64 warning ===")
    p = Path("tests/test_hash_v081.py")
    s = p.read_text()
    if "def test_uint64_python_fallback_warns" in s:
        print("  already present - skip")
        return
    L = []
    L.append("")
    L.append("")
    L.append("def test_uint64_python_fallback_warns(monkeypatch, caplog):")
    L.append("    # v0.9.5b (P-SH2): silent -> warning on Python fallback.")
    L.append("    import logging")
    L.append("    from cadw.ops import shuffle as sh")
    L.append("    caplog.set_level(logging.WARNING, logger='cadw.ops.shuffle')")
    L.append("    monkeypatch.setattr(")
    L.append("        sh, '_pandas_hash_column', lambda arr, **kw: None,")
    L.append("    )")
    L.append("    arr = pa.array([1, 2, 3], type=pa.uint64())")
    L.append("    h = sh._hash_column(arr)")
    L.append("    assert h is not None and len(h) == 3")
    L.append("    msgs = [r.message for r in caplog.records]")
    L.append("    assert any('uint64' in str(m) and 'fallback' in str(m)")
    L.append("               for m in msgs), (")
    L.append("        'expected uint64 fallback warning; got ' + str(msgs)")
    L.append("    )")
    L.append("")
    s = s + "\n".join(L)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.5b: P-SH2 + C6 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_uint64_warning()
    extend_audited_files()
    add_whitelist_entries()
    add_uint64_warning_test()

    print("\n=== pytest: AST guard (may reveal unexpected violations) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD REVEALED VIOLATIONS — inspect above ***")
        sys.exit(1)

    print("\n=== pytest: hash regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** HASH REGRESSION FAILED ***")
        sys.exit(1)

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

    commit_with_guards(
        "v0.9.5b: P-SH2 — shuffle._hash_column uint64 silent fallback "
        "-> warning (same class as K1.2); C6 — extend AST guard to "
        "viz/_aggregate.py (line_agg sort_by K rows; box_stats "
        "df.quantile) + core/memory_guard.py (estimate_unique_keys "
        "dictionary_encode on ≤10k sample); whitelisted by qualname",
        self_path="scripts/patch_v095b_p_sh2_c6.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.5b applied.")


if __name__ == "__main__":
    main()
