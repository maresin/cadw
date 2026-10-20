#!/usr/bin/env python3
# v0.9.5b-fix1: robust AUDITED_FILES extension (paren-tracking).
#
# Original patch anchor assumed specific multi-line format; actual
# differs (K1.1 may have restructured). Generic approach:
#   1. Print actual AUDITED_FILES context for debugging.
#   2. Find `AUDITED_FILES = ` line; track paren depth to closing.
#   3. Insert `+ [viz..., memory_guard...]` before close.
#
# State: shuffle.py already has uint64 warning (fix applied). Skip
# that step. Continue whitelist + test + commit.
#
# Run: uv run python scripts/patch_v095b_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def show_audited_context():
    print("\n=== inspect current AUDITED_FILES ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    lines = p.read_text().split("\n")
    for i, line in enumerate(lines):
        if "AUDITED_FILES" in line:
            print(f"  line {i+1}: {line!r}")
            # Print following 8 lines
            for j in range(i + 1, min(i + 10, len(lines))):
                print(f"  line {j+1}: {lines[j]!r}")
                if lines[j].strip() == ")":
                    break
            break


def extend_audited_files():
    print("\n=== extend AUDITED_FILES (paren-tracking) ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    lines = p.read_text().split("\n")
    if any("viz/_aggregate.py" in l or '_aggregate.py' in l for l in lines):
        print("  already extended - skip")
        return

    # Find AUDITED_FILES assignment start
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("AUDITED_FILES ="):
            start_idx = i
            break
    if start_idx is None:
        raise SystemExit("AUDITED_FILES assignment not found")

    # Track paren depth to find close
    depth = 0
    end_idx = None
    for i in range(start_idx, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth == 0 and i > start_idx:
            end_idx = i
            break
    if end_idx is None:
        # Single-line assignment case
        stripped = lines[start_idx].strip()
        if stripped.startswith("AUDITED_FILES = [") or stripped.startswith("AUDITED_FILES = list"):
            end_idx = start_idx
        else:
            raise SystemExit(
                f"AUDITED_FILES close not found (start line {start_idx+1})"
            )

    print(f"  start line {start_idx+1}, close line {end_idx+1}")
    print(f"  close: {lines[end_idx]!r}")

    # Determine: is assignment ends with ')' alone? Or full expression?
    close_stripped = lines[end_idx].strip()
    if close_stripped == ")":
        # Insert `+ [...]` before the close line
        new_lines = [
            '    # v0.9.5b (C6): extend to viz/_aggregate + memory_guard.',
            '    # line_agg uses sort_by on K rows; box_stats delegates to',
            '    # df.quantile (guarded via external_sort); estimate_unique_',
            '    # keys uses dictionary_encode on ≤10k sample. Whitelisted.',
            '    + [CADW_ROOT / "viz" / "_aggregate.py",',
            '       CADW_ROOT / "core" / "memory_guard.py"]',
        ]
        lines[end_idx:end_idx] = new_lines
    else:
        # Single-line: append + [...] to end of line
        # e.g. `AUDITED_FILES = list(...) + [...]` -> append `+ [...]`
        if not close_stripped.endswith(","):
            # tuple/list ends with `)` or `]`; append via +
            lines[end_idx] = close_stripped + (
                ' + [CADW_ROOT / "viz" / "_aggregate.py", '
                'CADW_ROOT / "core" / "memory_guard.py"]'
            )

    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK")


def add_whitelist_entries():
    print("\n=== add C6 whitelist entries ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "cadw/viz/_aggregate.py:line_agg" in s:
        print("  already whitelisted - skip")
        return

    # Find describe anchor
    anchor = '"cadw/dataframe.py:CadwDataFrame.describe":'
    lines = s.split("\n")
    line_idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            line_idx = i
            break
    if line_idx is None:
        raise SystemExit("describe whitelist anchor not found")

    # Find value-close
    end_idx = line_idx
    if not lines[line_idx].strip().endswith(","):
        for j in range(line_idx + 1, min(line_idx + 5, len(lines))):
            if lines[j].strip().endswith(","):
                end_idx = j
                break
        else:
            raise SystemExit("describe value-close not found")
    print(f"  describe key line {line_idx+1}, value-close {end_idx+1}")

    new_entries = [
        "    # v0.9.5b (C6): viz/_aggregate.py high-risk calls.",
        "    # line_agg sorts K-row aggregate table (bounded by K).",
        "    # box_stats delegates to df.quantile (guarded).",
        '    "cadw/viz/_aggregate.py:line_agg":',
        '        "table.sort_by on K rows (bounded by group count)",',
        '    "cadw/viz/_aggregate.py:box_stats":',
        '        "delegates to df.quantile (guarded via external_sort)",',
        "    # v0.9.5b (C6): core/memory_guard.py high-risk calls.",
        '    "cadw/core/memory_guard.py:MemoryGuard.estimate_unique_keys":',
        '        "pc.dictionary_encode on <=10k sample batch (bounded)",',
    ]
    for offset, entry in enumerate(new_entries):
        lines.insert(end_idx + 1 + offset, entry)
    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK")


def add_uint64_warning_test():
    print("\n=== test_hash_v081.py: uint64 warning test ===")
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
    L.append("    msgs = [str(r.message) for r in caplog.records]")
    L.append("    assert any('uint64' in m and 'fallback' in m for m in msgs), (")
    L.append("        'expected uint64 fallback warning; got ' + str(msgs)")
    L.append("    )")
    L.append("")
    s = s + "\n".join(L)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def main():
    print("=== v0.9.5b-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    show_audited_context()
    extend_audited_files()
    add_whitelist_entries()
    add_uint64_warning_test()

    print("\n=== pytest: AST guard ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD STILL RED — inspect violations above ***")
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
        "viz/_aggregate.py + core/memory_guard.py; whitelist line_agg "
        "sort_by, box_stats df.quantile, estimate_unique_keys "
        "dictionary_encode",
        self_path="scripts/patch_v095b_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.5b applied.")


if __name__ == "__main__":
    main()
