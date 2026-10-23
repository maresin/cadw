#!/usr/bin/env python3
# v0.9.26m-fix3: post-write lazy injection + proper whitelist rewrite.
# §13.4/303/313/317/318.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_indexing.py"

TARGET_SYMBOLS = [
    "iloc", "loc", "at", "iat",
    "head",
    "set_index", "reset_index", "reindex", "sort_values",
]

WHITELIST_FILES = [
    "cadw/patch_utils.py",
    "scripts/service/diagnose_unguarded_to_pylist.py",
    "scripts/service/diagnose_phase_b_split.py",
    "tests/test_arrow_call_memory_guard.py",
    "tests/test_v0925f_a.py",
    "tests/test_v0925d_a.py",
]

LAZY_TARGET = "CadwDataFrame"
LAZY_IMPORT = "from ._core import CadwDataFrame"


# ============ text-based lazy injection ============

def find_body_start(lines, method_name):
    """Find 0-based index for lazy injection (skip sig + docstring)."""
    def_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (s.startswith(f"def {method_name}(")
                or s.startswith(f"def {method_name} (")):
            indent = len(ln) - len(ln.lstrip())
            if indent == 4:  # method level
                def_idx = i
                break
    if def_idx is None:
        return None
    # Sig-end
    acc = 0
    sig_end = None
    for j in range(def_idx, len(lines)):
        for ch in lines[j]:
            if ch == "(":
                acc += 1
            elif ch == ")":
                acc -= 1
        if acc == 0 and lines[j].rstrip().endswith(":"):
            sig_end = j
            break
    if sig_end is None:
        return None
    # Skip docstring
    insert = sig_end + 1
    if insert < len(lines):
        s = lines[insert].strip()
        if s.startswith('"""') or s.startswith("'''"):
            q = '"""' if s.startswith('"""') else "'''"
            if s.count(q) >= 2:
                insert += 1
            else:
                insert += 1
                while insert < len(lines):
                    if q in lines[insert]:
                        insert += 1
                        break
                    insert += 1
    return insert


def method_uses_cadwdf(method_source_text):
    """AST-based: does method body construct/load CadwDataFrame?"""
    # dedent for ast.parse
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(method_source_text))
    except SyntaxError:
        # fallback to simple text search
        return "CadwDataFrame" in method_source_text
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == "CadwDataFrame":
            return True
    return False


def inject_lazy_in_file():
    """Post-write injection into _indexing.py using text-based scan."""
    print("\n--- post-write lazy injection ---")
    p = Path(NEW_MOD)
    src = p.read_text()

    # Already injected?
    if LAZY_IMPORT in src:
        print("  already has lazy import somewhere — check count")
        # Count occurrences
        n = src.count(LAZY_IMPORT)
        print(f"  occurrences: {n}")

    # Find method boundaries via AST
    tree = ast.parse(src)
    cls_node = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "_IndexingMixin"),
        None,
    )
    if cls_node is None:
        raise SystemExit(f"_IndexingMixin not in {NEW_MOD}")

    lines = src.split("\n")
    to_inject = []  # list of (insert_idx, method_name)

    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Extract method source (with dedent for ast.parse)
        m_start = item.lineno - 1
        if item.decorator_list:
            m_start = min(d.lineno for d in item.decorator_list) - 1
        m_end = max(
            getattr(n, "end_lineno", item.lineno)
            for n in ast.walk(item)
        )
        method_src = "\n".join(lines[m_start:m_end])

        if not method_uses_cadwdf(method_src):
            continue

        # Already has lazy?
        if LAZY_IMPORT in method_src:
            print(f"  {item.name}: already injected")
            continue

        insert_idx = find_body_start(lines, item.name)
        if insert_idx is None:
            print(f"  {item.name}: body-start not found")
            continue
        to_inject.append((insert_idx, item.name))

    if not to_inject:
        print("  no injection needed")
        return 0

    # Apply descending by insert_idx (later positions first)
    to_inject.sort(reverse=True)
    for insert_idx, name in to_inject:
        lines.insert(insert_idx, "        " + LAZY_IMPORT)
        print(f"  {name}: injected at line {insert_idx + 1}")

    new_src = "\n".join(lines)
    ast.parse(new_src, filename=NEW_MOD)
    p.write_text(new_src)
    print(f"  wrote {NEW_MOD} ({len(new_src.splitlines())} lines)")
    return len(to_inject)


# ============ whitelist rewrite (§13.4/318) ============

def rewrite_whitelist():
    """Replace {old_file}:{Class}.{sym} -> {new_file}:{NewClass}.{sym}."""
    print("\n--- whitelist rewrite (§13.4/318) ---")

    old_core = "cadw/dataframe/_core.py"
    new_mod = "cadw/dataframe/_indexing.py"
    old_classes = ["CadwDataFrame"]
    new_class = "_IndexingMixin"

    total = 0
    for fname in WHITELIST_FILES:
        fp = Path(fname)
        if not fp.exists():
            continue
        src = fp.read_text()
        orig = src
        replaced = 0

        for sym in TARGET_SYMBOLS:
            for old_cls in old_classes:
                old_key = f"{old_core}:{old_cls}.{sym}"
                new_key = f"{new_mod}:{new_class}.{sym}"
                count = src.count(old_key)
                if count > 0:
                    src = src.replace(old_key, new_key)
                    replaced += count

        if src != orig:
            fp.write_text(src)
            if fname.endswith(".py"):
                ast.parse(src, filename=fname)
            print(f"  {fname}: {replaced} key(s) rewritten")
            total += replaced
        else:
            print(f"  {fname}: no changes")

    if total == 0:
        print("  WARN: no whitelist keys matched — inspect manually")
    return total


# ============ verify + test + commit ============

def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26l":
        raise SystemExit(f"expected HEAD=v0.9.26l, got {head!r}")
    print("HEAD: " + head)

    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing")
    print(f"  {NEW_MOD}: present")

    core_src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin)" not in core_src:
        raise SystemExit("_core.py: missing _IndexingMixin inheritance")
    print("  _core.py: 4-mixin inheritance OK")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26m-fix3: lazy + whitelist ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    n_lazy = inject_lazy_in_file()
    n_wl = rewrite_whitelist()

    print(f"\n  summary: {n_lazy} lazy injections, "
          f"{n_wl} whitelist key(s) rewritten")

    run_tests()

    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist verify failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26m: extract _indexing (B.1 step 7, large mixin) "
        "+ fix1+fix2+fix3. iloc/loc/at/iat + head/set_index/"
        "reset_index/reindex/sort_values (~596 LOC, 9 symbols) "
        "moved verbatim from _core.py to _indexing.py as "
        "_IndexingMixin. CadwDataFrame(_ExportsMixin, "
        "_IteratorsMixin, _MiscMixin, _IndexingMixin). "
        "fix1-2: sanity via hasattr only (§13.4/313/316/317). "
        "fix3: post-write lazy injection (text-based, "
        "§13.4/303 inspect.getsource indented -> ast.parse fail) "
        "+ whitelist rewrite (§13.4/318 class-qualified keys). "
        "0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26m COMPLETE (fix1+2+3)")


if __name__ == "__main__":
    main()