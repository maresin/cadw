#!/usr/bin/env python3
# v0.9.4b-2-fix4: repair broken whitelist insert + re-insert correctly.
#
# fix3 assumed single-line whitelist entries. _repr_html_ entry is
# multi-line:
#     "cadw/dataframe.py:CadwDataFrame._repr_html_":
#         "20-row head().to_pandas().to_html(), bounded sample",
# fix3 inserted between key and value -> "key1: key2: value" -> parse error.
#
# fix4:
#   1. Remove the 2 bad-inserted lines (comment + "describe": ... on same line)
#   2. Find value-close line (ends with ",")
#   3. Insert "describe": \n "value", in same multi-line format
#
# Run: uv run python scripts/patch_v094b2_fix4.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def repair_broken_and_insert():
    print("\n=== repair + insert whitelist entry ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    lines = p.read_text().split("\n")

    # ============ Step 1: remove broken lines from fix3 ============
    to_remove = set()
    for i, line in enumerate(lines):
        # Comment line inserted by fix3
        if "v0.9.4b-2: describe() delegates to self.quantile" in line:
            to_remove.add(i)
            continue
        # Bad entry line: '"cadw/dataframe.py:describe": "delegates..."'
        if '"cadw/dataframe.py:describe"' in line:
            stripped = line.strip()
            # fix3 inserted this on ONE line ending with ","
            if stripped.endswith('",') and "delegates to self.quantile" in stripped:
                to_remove.add(i)
                continue
            # Or standalone key line (no value on same line, ends with ":")
            if stripped == '"cadw/dataframe.py:describe":':
                to_remove.add(i)
                continue
            # Orphan value line follows standalone-key; separate check below
        # Orphan value: 8-space indent, "delegates to self.quantile"
        if (line.startswith("        ")
                and "delegates to self.quantile (guarded" in line):
            # Only remove if it's the fix3-inserted one (v0.9.4 comment nearby)
            to_remove.add(i)
            continue

    # But be careful: only remove if the file is actually broken.
    # Detect: try compile.
    import py_compile
    try:
        py_compile.compile(str(p), doraise=True)
        print("  file compiles — no repair needed; will insert below")
        is_broken = False
    except py_compile.PyCompileError:
        print("  file is broken (fix3 partial state) — repairing")
        is_broken = True

    if is_broken and to_remove:
        # Remove lines in reverse to preserve indices
        for i in reversed(sorted(to_remove)):
            print("  removing line " + str(i+1) + ": " + repr(lines[i][:90]))
            del lines[i]
        p.write_text("\n".join(lines))
        assert_compiles(p)
        print("  repaired OK")

    # ============ Step 2: idempotency check ============
    s = p.read_text()
    if '"cadw/dataframe.py:describe"' in s:
        print("  describe already present — skip insert")
        return

    # ============ Step 3: find _repr_html_ key line + value-close ============
    lines = s.split("\n")
    key_idx = None
    for i, line in enumerate(lines):
        if "_repr_html_" in line and line.strip().startswith('"'):
            key_idx = i
            break
    if key_idx is None:
        raise SystemExit("_repr_html_ key not found")

    key_line = lines[key_idx]
    print("  key line " + str(key_idx+1) + ": " + repr(key_line))

    # If key line ends with `:` and no value on same line → multi-line.
    # Advance to value-close (line ending with `,` or `)`).
    key_stripped = key_line.strip()
    is_multiline = key_stripped.endswith(":") or (
        key_stripped.endswith('":')  # e.g. '"key":'
    )

    if is_multiline:
        end_idx = None
        for j in range(key_idx + 1, min(key_idx + 10, len(lines))):
            stripped = lines[j].strip()
            if stripped.endswith(",") or stripped.endswith(")"):
                end_idx = j
                break
        if end_idx is None:
            raise SystemExit(
                "_repr_html_ multi-line entry close not found within 10 lines"
            )
        print("  value-close line " + str(end_idx+1) + ": " + repr(lines[end_idx]))
        # Insert after end_idx:
        #   <comment>
        #   "cadw/dataframe.py:describe":
        #       "delegates to self.quantile (guarded via external_sort)",
        comment = (
            "    # v0.9.4b-2: describe() delegates to self.quantile "
            "(guarded via external_sort)."
        )
        new_key = '    "cadw/dataframe.py:describe":'
        new_val = (
            '        "delegates to self.quantile '
            '(guarded via external_sort)",'
        )
        lines.insert(end_idx + 1, new_val)
        lines.insert(end_idx + 1, new_key)
        lines.insert(end_idx + 1, comment)
    else:
        # Single-line entry: '"key": "value",'
        # Insert after key_idx (single line).
        comment = (
            "    # v0.9.4b-2: describe() delegates to self.quantile "
            "(guarded via external_sort)."
        )
        new_entry = (
            '    "cadw/dataframe.py:describe": '
            '"delegates to self.quantile (guarded via external_sort)",'
        )
        lines.insert(key_idx + 1, new_entry)
        lines.insert(key_idx + 1, comment)
        print("  single-line entry; inserted after line " + str(key_idx+1))

    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK (inserted after _repr_html_)")


def main():
    print("=== v0.9.4b-2-fix4 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    repair_broken_and_insert()

    print("\n=== pytest: AST guard ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: v094b2 + dt/str + patch_utils regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe_v094b2.py",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_patch_utils.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
        "v0.9.4b-2: _GroupBy.transform (agg+merge; order=partition); "
        "describe (count/mean/std/min/q/max); dt.floor/ceil/round(freq) "
        "+ week_of_year; str.strip/lstrip/rstrip/starts_with/ends_with/"
        "replace; PredicateFilter starts_with/ends_with; "
        "_DtAccessor._feature_custom; whitelist describe in AST guard",
    )
    print("\n==> v0.9.4b-2 complete (fix1-fix4).")


if __name__ == "__main__":
    main()
