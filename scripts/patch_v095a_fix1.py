#!/usr/bin/env python3
# v0.9.5a-fix1: _collect_all_qualnames recursive descent (nested funcs).
#
# Bug: _collect_all_qualnames walked only tree.body + ClassDef.body
# (2 levels: module-level funcs, Class.method). Whitelist uses K1.1
# {path}:{Class.method.nested_func} — same as ast.walk sees (which
# descends into FunctionDef bodies). 5 orphan keys were false
# positives: all Class.method.nested (nested `_iter_zipped` inside
# shift_on/rolling_on/etc; `_iter_encoded` inside encode_categorical).
#
# Fix: recursive descent into FunctionDef/ClassDef bodies, building
# qualname as Class.method.nested (dot-separated levels). Symmetric
# to test's _qualname helper.
#
# Run: uv run python scripts/patch_v095a_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


# ============ Fix _collect_all_qualnames ============

def fix_collect():
    print("\n=== patch_utils.py: _collect_all_qualnames recursive ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def _walk_defs(" in s:
        print("  already recursive - skip")
        return

    old_body = (
        "        for node in tree.body:\n"
        "            if isinstance(node, _ast.FunctionDef):\n"
        "                result.add(rel + ':' + node.name)\n"
        "            elif isinstance(node, _ast.ClassDef):\n"
        "                for item in node.body:\n"
        "                    if isinstance(item, _ast.FunctionDef):\n"
        "                        result.add(rel + ':' + node.name + '.'\n"
        "                                   + item.name)\n"
        "    return result\n"
    )

    new_body = (
        "        def _walk_defs(node, prefix):\n"
        "            # Recursively descend into FunctionDef/ClassDef bodies.\n"
        "            # Qualname = Class.method.nested (dot-separated levels).\n"
        "            # Matches test_arrow_call_memory_guard._qualname\n"
        "            # semantics (v0.9.5a-fix1): ast.walk sees nested\n"
        "            # FunctionDef nodes, so whitelist keys must include them.\n"
        "            body = getattr(node, 'body', None)\n"
        "            if body is None:\n"
        "                return\n"
        "            for child in body:\n"
        "                if isinstance(child, _ast.FunctionDef):\n"
        "                    qn = prefix + child.name\n"
        "                    result.add(rel + ':' + qn)\n"
        "                    _walk_defs(child, qn + '.')\n"
        "                elif isinstance(child, _ast.ClassDef):\n"
        "                    cp = prefix + child.name + '.'\n"
        "                    for item in child.body:\n"
        "                        if isinstance(item, _ast.FunctionDef):\n"
        "                            qn = cp + item.name\n"
        "                            result.add(rel + ':' + qn)\n"
        "                            _walk_defs(item, qn + '.')\n"
        "        _walk_defs(tree, '')\n"
        "    return result\n"
    )

    if old_body not in s:
        raise SystemExit("_collect_all_qualnames body anchor not found")
    s = s.replace(old_body, new_body, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Add nested-orphan test ============

def add_nested_test():
    print("\n=== test_patch_utils.py: nested qualname test ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_collect_all_qualnames_nested" in s:
        print("  already present - skip")
        return
    L = []
    L.append("")
    L.append("")
    L.append("def test_collect_all_qualnames_nested(tmp_path):")
    L.append("    # v0.9.5a-fix1: nested funcs (Class.method.nested)")
    L.append("    # must be included (K1.1 whitelist uses them).")
    L.append("    from cadw.patch_utils import _collect_all_qualnames")
    L.append("    cadw = tmp_path / 'cadw'")
    L.append("    cadw.mkdir()")
    L.append("    src = (")
    L.append("        'class C:\\n'")
    L.append("        '    def method(self):\\n'")
    L.append("        '        def nested():\\n'")
    L.append("        '            pass\\n'")
    L.append("        '        return nested\\n'")
    L.append("    )")
    L.append("    (cadw / 'm.py').write_text(src)")
    L.append("    names = _collect_all_qualnames(cadw)")
    L.append("    assert 'cadw/m.py:C.method' in names")
    L.append("    assert 'cadw/m.py:C.method.nested' in names")
    L.append("")
    s = s + "\n".join(L)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def main():
    print("=== v0.9.5a-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_collect()
    add_nested_test()

    print("\n=== pytest: test_patch_utils.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS STILL RED ***")
        sys.exit(1)

    print("\n=== verify_whitelist: real cadw/ (standalone) ===")
    from cadw.patch_utils import verify_whitelist
    res = verify_whitelist(raise_on_orphan=False)
    if res["orphans"] or res["malformed"]:
        print("\n  STILL has orphans — follow-up fix2 needed:")
        for k, r_ in res["orphans"] + res["malformed"]:
            print("    " + str(k) + " (" + r_ + ")")
        sys.exit(1)

    print("\n=== pytest: AST guard ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD FAILED ***")
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
        "v0.9.5a: verify_whitelist() — AST orphan detector for "
        "WHITELISTED_FUNCS; commit_with_guards(verify_whitelist=True) "
        "hook; _collect_all_qualnames recursive (nested Class.method."
        "nested funcs); 5 false-positive orphans resolved",
        self_path="scripts/patch_v095a_fix1.py",
        verify_whitelist=True,
    )
    print("\n==> v0.9.5a applied (initial + fix1).")


if __name__ == "__main__":
    main()
