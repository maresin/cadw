#!/usr/bin/env python3
# v0.9.5a-fix3: parameter name shadowing in commit_with_guards.
#
# Bug: def commit_with_guards(..., verify_whitelist=False):
#          if verify_whitelist:
#              verify_whitelist()   # <- parameter bool, not function
# TypeError: 'bool' object is not callable.
#
# Fix: rename param `verify_whitelist` -> `check_whitelist`. Module
# function `verify_whitelist` becomes visible inside body. Symmetric
# to existing kwarg naming (`stale_patterns`/`stale_dirs` for
# `grep_and_warn`/`assert_no_stale_behavior` — different names).
#
# Run: uv run python scripts/patch_v095a_fix3.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def fix_param_rename():
    print("\n=== patch_utils.py: rename verify_whitelist -> check_whitelist ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "check_whitelist=False" in s:
        print("  already fixed - skip")
        return

    # 1. Signature
    old_sig = "                      allow=None, verify_whitelist=False):"
    new_sig = "                      allow=None, check_whitelist=False):"
    if old_sig not in s:
        raise SystemExit("signature anchor not found")
    s = s.replace(old_sig, new_sig, 1)

    # 2. Comment update (docstring line about kwargs)
    old_doc = "    # verify_whitelist: AST-check WHITELISTED_FUNCS (v0.9.5a).\n"
    new_doc = (
        "    # check_whitelist: AST-check WHITELISTED_FUNCS (v0.9.5a);\n"
        "    # parameter renamed from `verify_whitelist` (fix3): old name\n"
        "    # shadowed the module-level `verify_whitelist()` function ->\n"
        "    # TypeError: 'bool' object is not callable.\n"
    )
    if old_doc in s:
        s = s.replace(old_doc, new_doc, 1)

    # 3. Body: if verify_whitelist: -> if check_whitelist:
    old_body = "    if verify_whitelist:\n"
    new_body = "    if check_whitelist:\n"
    if old_body not in s:
        raise SystemExit("body 'if verify_whitelist:' anchor not found")
    s = s.replace(old_body, new_body, 1)

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def add_shadowing_guard_test():
    print("\n=== test_patch_utils.py: shadowing guard ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_commit_with_guards_no_param_shadowing" in s:
        print("  already present - skip")
        return
    L = []
    L.append("")
    L.append("")
    L.append("def test_commit_with_guards_no_param_shadowing():")
    L.append("    # v0.9.5a-fix3: param names must not shadow module-level")
    L.append("    # function names (else `if param: param()` -> TypeError).")
    L.append("    import inspect")
    L.append("    import cadw.patch_utils as pu")
    L.append("    sig = inspect.signature(pu.commit_with_guards)")
    L.append("    module_funcs = {")
    L.append("        n for n, obj in vars(pu).items()")
    L.append("        if callable(obj) and not n.startswith('__')")
    L.append("    }")
    L.append("    for param_name in sig.parameters:")
    L.append("        if param_name in ('self', 'cls'):")
    L.append("            continue")
    L.append("        assert param_name not in module_funcs, (")
    L.append("            'commit_with_guards param ' + repr(param_name)")
    L.append("            + ' shadows module-level function with same name'")
    L.append("        )")
    L.append("")
    s = s + "\n".join(L)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def reload_and_verify():
    print("\n=== reload + subprocess verify ===")
    import cadw.patch_utils as pu
    importlib.reload(pu)
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist()"],
        capture_output=True, text=True, timeout=120,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:5]:
            print("    " + line)
    if r.returncode != 0:
        print("*** subprocess verify FAILED ***")
        sys.exit(1)
    print("  OK")


def main():
    print("=== v0.9.5a-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_param_rename()
    add_shadowing_guard_test()
    reload_and_verify()

    print("\n=== pytest: test_patch_utils.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS FAILED ***")
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

    # Import reloaded commit_with_guards with new kwarg.
    from cadw.patch_utils import commit_with_guards as cwg
    cwg(
        "v0.9.5a: verify_whitelist() — AST orphan detector for "
        "WHITELISTED_FUNCS; _collect_all_qualnames recursive "
        "(nested Class.method.nested); commit_with_guards "
        "check_whitelist=True hook (param renamed to avoid shadowing "
        "verify_whitelist function); fix1-fix3 combined",
        self_path="scripts/patch_v095a_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.5a complete (fix1+fix2+fix3).")


if __name__ == "__main__":
    main()
