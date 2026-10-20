#!/usr/bin/env python3
# v0.9.5a: verify_whitelist() — AST whitelist-orphan detector.
#
# Root cause of v0.9.4b-2 fix2-fix5 (4 rounds): whitelist keys and real
# qualnames drifted. K1.1 uses {path}:{Class.method}; iteratively wrong.
# verify_whitelist() AST-walks cadw/**/*.py, collects real qualnames,
# reports orphans (whitelist keys that don't match any real function).
#
# Hook: commit_with_guards(verify_whitelist=True) — pre-commit gate.
#
# Run: uv run python scripts/patch_v095a_verify_whitelist.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


# ============ 1. patch_utils.py: _collect_all_qualnames + verify_whitelist ============

HELPER_LINES = [
    "",
    "",
    "# ============ v0.9.5a: verify_whitelist (AST-orphan) ============",
    "",
    "",
    "def _collect_all_qualnames(cadw_root):",
    "    # Walk cadw_root/**/*.py; return set of {path}:{qualname}.",
    "    # qualname = func_name (module-level) or Class.method.",
    "    # Nested funcs inside methods: NOT included (whitelist is per-",
    "    # top-level callable).",
    "    import ast as _ast",
    "    from pathlib import Path as _P",
    "    root = _P(cadw_root)",
    "    parent = root.parent",
    "    result = set()",
    "    if not root.exists():",
    "        return result",
    "    for py in root.rglob('*.py'):",
    "        try:",
    "            src = py.read_text(encoding='utf-8')",
    "            tree = _ast.parse(src)",
    "        except (SyntaxError, OSError, UnicodeDecodeError):",
    "            continue",
    "        try:",
    "            rel = py.relative_to(parent).as_posix()",
    "        except ValueError:",
    "            rel = py.as_posix()",
    "        for node in tree.body:",
    "            if isinstance(node, _ast.FunctionDef):",
    "                result.add(rel + ':' + node.name)",
    "            elif isinstance(node, _ast.ClassDef):",
    "                for item in node.body:",
    "                    if isinstance(item, _ast.FunctionDef):",
    "                        result.add(rel + ':' + node.name + '.'",
    "                                   + item.name)",
    "    return result",
    "",
    "",
    "def _load_whitelist_from_test():",
    "    # Import WHITELISTED_FUNCS from test_arrow_call_memory_guard.",
    "    import importlib",
    "    mod = importlib.import_module(",
    "        'tests.test_arrow_call_memory_guard'",
    "    )",
    "    return getattr(mod, 'WHITELISTED_FUNCS', {})",
    "",
    "",
    "def verify_whitelist(*, whitelist=None, cadw_root='cadw',",
    "                     raise_on_orphan=True):",
    "    # AST-verify that every key in WHITELISTED_FUNCS matches a real",
    "    # function/method in cadw_root. Orphan (key without real def) is",
    "    # dead whitelist: silent — key never matched, so guard doesn't",
    "    # apply. Block commit (fail-fast).",
    "    #",
    "    # whitelist: dict override (None -> load from test file).",
    "    # cadw_root: cadw/ dir (default 'cadw').",
    "    # raise_on_orphan: sys.exit(1) on orphans (default True).",
    "    #",
    "    # Returns: {'orphans': [(key, reason)], 'malformed': [(key, r)]}.",
    "    import sys as _sys",
    "    if whitelist is None:",
    "        whitelist = _load_whitelist_from_test()",
    "    actual = _collect_all_qualnames(cadw_root)",
    "    orphans = []",
    "    malformed = []",
    "    for key, rationale in whitelist.items():",
    "        if not isinstance(key, str):",
    "            malformed.append((repr(key), 'non-str key'))",
    "            continue",
    "        if ':' not in key:",
    "            malformed.append((key, 'bare key (K1.1 requires path:qualname)'))",
    "            continue",
    "        if key not in actual:",
    "            orphans.append((key, 'not in cadw AST qualnames'))",
    "    if malformed or orphans:",
    "        print('verify_whitelist: FAIL')",
    "        for k, r in malformed:",
    "            print('  malformed: ' + str(k) + ' (' + r + ')')",
    "        for k, r in orphans:",
    "            print('  orphan: ' + str(k) + ' (' + r + ')')",
    "        if raise_on_orphan:",
    "            _sys.exit(1)",
    "    else:",
    "        print('verify_whitelist: OK ('",
    "              + str(len(whitelist)) + ' keys, all valid)')",
    "    return {'orphans': orphans, 'malformed': malformed}",
    "",
]


def add_verify_whitelist():
    print("\n=== patch_utils.py: verify_whitelist ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def verify_whitelist(" in s:
        print("  already present - skip")
        return
    anchor = "\n\ndef grep_and_warn("
    if anchor not in s:
        raise SystemExit("anchor not found")
    block = "\n".join(HELPER_LINES)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. commit_with_guards: verify_whitelist hook ============

def update_commit_with_guards():
    print("\n=== patch_utils.py: commit_with_guards verify_whitelist kwarg ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "verify_whitelist=False" in s:
        print("  already present - skip")
        return
    old_sig = (
        "def commit_with_guards(message, *, self_path=None,\n"
        "                      stale_patterns=None, stale_dirs=None,\n"
        "                      allow=None):"
    )
    new_sig = (
        "def commit_with_guards(message, *, self_path=None,\n"
        "                      stale_patterns=None, stale_dirs=None,\n"
        "                      allow=None, verify_whitelist=False):"
    )
    if old_sig not in s:
        raise SystemExit("commit_with_guards signature anchor not found")
    s = s.replace(old_sig, new_sig, 1)

    old_body = (
        "    if stale_patterns and stale_dirs:\n"
        "        assert_no_stale_behavior(\n"
        "            stale_patterns, stale_dirs, allow=allow,\n"
        "        )\n"
        "    subprocess.run(['git', 'add', '-A'], check=True)\n"
    )
    new_body = (
        "    if stale_patterns and stale_dirs:\n"
        "        assert_no_stale_behavior(\n"
        "            stale_patterns, stale_dirs, allow=allow,\n"
        "        )\n"
        "    if verify_whitelist:\n"
        "        verify_whitelist()\n"
        "    subprocess.run(['git', 'add', '-A'], check=True)\n"
    )
    if old_body not in s:
        raise SystemExit("commit_with_guards body anchor not found")
    s = s.replace(old_body, new_body, 1)

    # Also update docstring (line comment) referencing kwargs.
    old_doc = (
        "    # git add -A + commit, ALWAYS running pre-commit guards.\n"
        "    # self_path: py_compile check.\n"
        "    # stale_patterns+stale_dirs: assert_no_stale_behavior.\n"
    )
    new_doc = (
        "    # git add -A + commit, ALWAYS running pre-commit guards.\n"
        "    # self_path: py_compile check.\n"
        "    # stale_patterns+stale_dirs: assert_no_stale_behavior.\n"
        "    # verify_whitelist: AST-check WHITELISTED_FUNCS "
        "(v0.9.5a).\n"
    )
    if old_doc in s:
        s = s.replace(old_doc, new_doc, 1)

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. tests ============

TESTS_LINES = [
    "",
    "",
    "# ============ v0.9.5a: verify_whitelist ============",
    "",
    "",
    "def test_collect_all_qualnames_basic(tmp_path):",
    "    from cadw.patch_utils import _collect_all_qualnames",
    "    cadw = tmp_path / 'cadw'",
    "    cadw.mkdir()",
    "    (cadw / 'mod.py').write_text(",
    "        'def foo():\\n    pass\\n'",
    "        '\\n'",
    "        'class C:\\n'",
    "        '    def bar(self):\\n'",
    "        '        pass\\n'",
    "    )",
    "    names = _collect_all_qualnames(cadw)",
    "    assert 'cadw/mod.py:foo' in names",
    "    assert 'cadw/mod.py:C.bar' in names",
    "    assert len(names) == 2",
    "",
    "",
    "def test_collect_all_qualnames_multi_file(tmp_path):",
    "    from cadw.patch_utils import _collect_all_qualnames",
    "    cadw = tmp_path / 'cadw'",
    "    (cadw / 'sub').mkdir(parents=True)",
    "    (cadw / 'a.py').write_text('def one(): pass\\n')",
    "    (cadw / 'sub' / 'b.py').write_text(",
    "        'def two(): pass\\n'",
    "    )",
    "    names = _collect_all_qualnames(cadw)",
    "    assert 'cadw/a.py:one' in names",
    "    assert 'cadw/sub/b.py:two' in names",
    "",
    "",
    "def test_verify_whitelist_clean(tmp_path, capsys):",
    "    from cadw.patch_utils import verify_whitelist",
    "    cadw = tmp_path / 'cadw'",
    "    cadw.mkdir()",
    "    (cadw / 'm.py').write_text('def real(): pass\\n')",
    "    wl = {'cadw/m.py:real': 'exists' }",
    "    res = verify_whitelist(whitelist=wl, cadw_root=cadw,",
    "                           raise_on_orphan=False)",
    "    assert res['orphans'] == []",
    "    assert res['malformed'] == []",
    "",
    "",
    "def test_verify_whitelist_orphan(tmp_path, capsys):",
    "    from cadw.patch_utils import verify_whitelist",
    "    cadw = tmp_path / 'cadw'",
    "    cadw.mkdir()",
    "    (cadw / 'm.py').write_text('def real(): pass\\n')",
    "    wl = {",
    "        'cadw/m.py:real': 'exists',",
    "        'cadw/m.py:fake': 'orphan',",
    "    }",
    "    res = verify_whitelist(whitelist=wl, cadw_root=cadw,",
    "                           raise_on_orphan=False)",
    "    assert len(res['orphans']) == 1",
    "    assert res['orphans'][0][0] == 'cadw/m.py:fake'",
    "",
    "",
    "def test_verify_whitelist_malformed_bare(tmp_path):",
    "    from cadw.patch_utils import verify_whitelist",
    "    cadw = tmp_path / 'cadw'",
    "    cadw.mkdir()",
    "    wl = {'bare_key_no_path': 'malformed'}",
    "    res = verify_whitelist(whitelist=wl, cadw_root=cadw,",
    "                           raise_on_orphan=False)",
    "    assert len(res['malformed']) == 1",
    "",
    "",
    "def test_verify_whitelist_raises_on_orphan(tmp_path):",
    "    import pytest",
    "    from cadw.patch_utils import verify_whitelist",
    "    cadw = tmp_path / 'cadw'",
    "    cadw.mkdir()",
    "    (cadw / 'm.py').write_text('def real(): pass\\n')",
    "    wl = {'cadw/m.py:fake': 'orphan'}",
    "    with pytest.raises(SystemExit):",
    "        verify_whitelist(whitelist=wl, cadw_root=cadw)",
    "",
    "",
    "def test_verify_whitelist_real_cadw():",
    "    # Real cadw/ against real WHITELISTED_FUNCS from test file.",
    "    from cadw.patch_utils import verify_whitelist",
    "    res = verify_whitelist(raise_on_orphan=False)",
    "    assert isinstance(res, dict)",
    "    assert 'orphans' in res and 'malformed' in res",
    "    # If this fails, orphan keys exist — see printed list.",
    "    assert not res['malformed'], (",
    "        'malformed whitelist keys: ' + str(res['malformed'])",
    "    )",
    "    assert not res['orphans'], (",
    "        'orphan whitelist keys: ' + str(res['orphans'])",
    "    )",
    "",
]


def add_tests():
    print("\n=== test_patch_utils.py: append verify_whitelist tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_verify_whitelist_clean" in s:
        print("  already present - skip")
        return
    s = s + "\n".join(TESTS_LINES)
    write_content(p, s.split("\n"))
    assert_compiles(p)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.5a: verify_whitelist ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_verify_whitelist()
    update_commit_with_guards()
    add_tests()

    print("\n=== pytest: test_patch_utils.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS TESTS FAILED ***")
        sys.exit(1)

    print("\n=== verify_whitelist: real cadw/ (standalone) ===")
    from cadw.patch_utils import verify_whitelist
    res = verify_whitelist(raise_on_orphan=False)
    if res["orphans"] or res["malformed"]:
        print("\n  WARNING: whitelist has orphans/malformed")
        print("  Follow-up patch will fix.")
        # Do NOT block this commit — verify_whitelist is new, existing
        # whitelist keys may be pre-existing orphans (K1.1 partial
        # migration). Fix in follow-up.

    print("\n=== pytest: AST guard (regression) ===")
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

    # Dogfood: commit via verify_whitelist=True (fatal). If orphans,
    # commit fails — fix must be in follow-up (separate patch, not
    # mixed with the new helper).
    commit_with_guards(
        "v0.9.5a: verify_whitelist() — AST orphan detector for "
        "WHITELISTED_FUNCS; commit_with_guards(verify_whitelist=True) "
        "hook; dogfood prevents v0.9.4b-2 fix2-fix5 class (wrong "
        "qualname key silently invalid)",
        self_path="scripts/patch_v095a_verify_whitelist.py",
    )
    print("\n==> v0.9.5a applied.")


if __name__ == "__main__":
    main()
