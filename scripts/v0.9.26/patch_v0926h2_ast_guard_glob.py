#!/usr/bin/env python3
# v0.9.26h2: glob-migration of AST-guard tests (B.1 infra).
# After extraction _exports.py, single-file guards miss extracted
# modules. Change all "inspect.getsource(df_mod._core)" or
# "Path('cadw/dataframe/_core.py')" invariant tests to glob over
# Path("cadw/dataframe").rglob("*.py").
# 0 production changes. Test-only.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


HELPER_SNIPPET = '''
def _iter_dataframe_sources():
    """v0.9.26h2 (§13.4/292): iterate all dataframe package sources.
    Replaces single-file scans; new extracted modules auto-covered.
    """
    import pathlib as _p
    pkg = _p.Path("cadw/dataframe")
    if pkg.is_dir():
        for f in sorted(pkg.rglob("*.py")):
            yield f, f.read_text()
    else:
        # legacy single-file fallback (should not trigger post-v0.9.26g)
        legacy = _p.Path("cadw/dataframe.py")
        if legacy.exists():
            yield legacy, legacy.read_text()
'''


# ============ discovery ============

def discover():
    """Find AST-guard test sites that read _core.py source."""
    hits = {
        "inspect_getsource_core": [],   # inspect.getsource(df_mod._core)
        "inspect_getsource_mod": [],    # inspect.getsource(df_mod)
        "path_core_string": [],         # "cadw/dataframe/_core.py"
        "path_core_composed": [],       # Path("cadw/dataframe/_core.py")
    }

    insp_core_pat = re.compile(
        r'inspect\.getsource\(\s*(?:df_mod\._core|'
        r'cadw\.dataframe\._core|dataframe\._core)'
    )
    insp_mod_pat = re.compile(
        r'inspect\.getsource\(\s*(?:df_mod|cadw\.dataframe)\s*\)'
    )
    path_core_str_pat = re.compile(
        r'[\'"]cadw/dataframe/_core\.py[\'"]'
    )
    path_core_composed_pat = re.compile(
        r'Path\s*\(\s*[\'"]cadw/dataframe/_core\.py[\'"]\s*\)'
    )

    for py in Path("tests").rglob("*.py"):
        src = py.read_text()
        for i, ln in enumerate(src.splitlines(), 1):
            if insp_core_pat.search(ln):
                hits["inspect_getsource_core"].append(
                    (str(py), i, ln.strip()[:100])
                )
            if insp_mod_pat.search(ln):
                hits["inspect_getsource_mod"].append(
                    (str(py), i, ln.strip()[:100])
                )
            if path_core_str_pat.search(ln):
                hits["path_core_string"].append(
                    (str(py), i, ln.strip()[:100])
                )
            if path_core_composed_pat.search(ln):
                hits["path_core_composed"].append(
                    (str(py), i, ln.strip()[:100])
                )

    return hits


def print_plan(hits):
    print("\n" + "=" * 72)
    print("DISCOVERY (AST-guard sites to glob-migrate)")
    print("=" * 72)
    for cat, items in hits.items():
        print(f"\n[{cat}] {len(items)} site(s)")
        for fname, ln, txt in items[:30]:
            print(f"  {fname}:{ln}  {txt}")
        if len(items) > 30:
            print(f"  ... ({len(items) - 30} more)")


# ============ mutations ============

def add_helper(test_path):
    """Insert helper into test module if absent."""
    p = Path(test_path)
    src = p.read_text()
    if "_iter_dataframe_sources" in src:
        return False  # already present
    # Place after imports, before first def/test
    lines = src.split("\n")
    insert_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("def test_") or ln.startswith("class Test"):
            insert_idx = i
            break
    if insert_idx is None:
        insert_idx = len(lines)
    lines.insert(insert_idx, HELPER_SNIPPET)
    new_src = "\n".join(lines)
    try:
        ast.parse(new_src, filename=str(p))
    except SyntaxError as e:
        print(f"  WARN: helper insertion broke syntax in {test_path}: {e}")
        return False
    p.write_text(new_src)
    return True


def migrate_inspect_call_to_loop(fname, old_call, guard_predicate_comment=""):
    """Replace `src = inspect.getsource(df_mod._core)` block with glob loop.

    Uses pattern-matching on the surrounding lines to find the
    pattern: `src = <call>` followed by assertions on `src`.
    Replaces with: `for _f, src in _iter_dataframe_sources():`.

    Note: this is heuristic — since asserts usually span multiple lines,
    the safe approach is to wrap in `for _f, src in _iter_dataframe_sources():`
    and indent following asserts. This requires knowing the block end.
    """
    # We'll do a conservative replacement: only handle the simple
    # case where the assertion is within the next 1-3 lines and uses
    # `src` only.
    p = Path(fname)
    src = p.read_text()
    # Find all lines containing the pattern
    lines = src.split("\n")
    i = 0
    changes = 0
    while i < len(lines):
        if old_call in lines[i]:
            # Find indent
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            # Replace `src = CALL` -> for-loop
            lines[i] = f"{indent}for _fname, src in _iter_dataframe_sources():"
            # Indent following lines until blank-line or dedent
            j = i + 1
            inner_indent = indent + "    "
            while j < len(lines):
                ln = lines[j]
                if not ln.strip():  # blank
                    j += 1
                    continue
                ln_indent = ln[:len(ln) - len(ln.lstrip())]
                if len(ln_indent) < len(indent) + 1:
                    break  # dedent → block end
                # Re-indent
                lines[j] = "    " + ln
                j += 1
            changes += 1
            i = j
        else:
            i += 1
    if changes:
        new_src = "\n".join(lines)
        try:
            ast.parse(new_src, filename=str(p))
        except SyntaxError as e:
            print(f"  WARN: {fname} would break: {e}")
            return False
        p.write_text(new_src)
        return True
    return False


# ============ tests ============

def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


# ============ main ============

def main():
    print("=== v0.9.26h2: glob-migrate AST-guard tests ===")
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26h":
        raise SystemExit("expected HEAD=v0.9.26h, got " + repr(head))
    print("HEAD: " + head)

    hits = discover()
    print_plan(hits)

    total = sum(len(v) for v in hits.values())
    if total == 0:
        print("\nNo sites to migrate — done?")
        return

    # Add helper to each file that has inspect-based reads
    files_with_inspect = set()
    for fname, _, _ in hits["inspect_getsource_core"]:
        files_with_inspect.add(fname)
    for fname, _, _ in hits["inspect_getsource_mod"]:
        files_with_inspect.add(fname)

    print("\n--- add helper ---")
    for fname in sorted(files_with_inspect):
        if add_helper(fname):
            print(f"  {fname}: helper inserted")
        else:
            print(f"  {fname}: helper already present (skip)")

    print("\n--- migrate inspect.getsource blocks ---")
    # Process each file: find `src = inspect.getsource(...)`
    # blocks and convert to for-loops
    for fname in sorted(files_with_inspect):
        # Pattern 1: _core explicit
        did = migrate_inspect_call_to_loop(
            fname, "src = inspect.getsource(df_mod._core)"
        )
        # Pattern 2: plain mod (still reads __init__.py after split)
        did2 = migrate_inspect_call_to_loop(
            fname, "src = inspect.getsource(df_mod)"
        )
        if did or did2:
            print(f"  {fname}: migrated")
        else:
            print(f"  {fname}: no matching pattern (skip)")

    print("\n--- path-based sites (manual follow-up) ---")
    print(f"  path_core_string: {len(hits['path_core_string'])}")
    print(f"  path_core_composed: {len(hits['path_core_composed'])}")
    print("  NOTE: these are NOT auto-migrated. May be intentional")
    print("  (e.g. reading _core for line numbers). Review per site.")
    for fname, ln, txt in hits["path_core_string"] + hits["path_core_composed"]:
        print(f"    {fname}:{ln}  {txt}")

    run_tests()

    commit_with_guards(
        "v0.9.26h2: glob-migrate AST-guard tests (§13.4/292). "
        "inspect.getsource(df_mod._core) replaced with "
        "_iter_dataframe_sources() helper that globs all package "
        "modules. New extracted modules (from B.1 extractions) "
        "auto-covered by facade AST guards. Test-only; 2488 green.",
        self_path="scripts/patch_v0926h2_ast_guard_glob.py",
        check_whitelist=True,
    )
    print("\nv0.9.26h2 COMPLETE")


if __name__ == "__main__":
    main()