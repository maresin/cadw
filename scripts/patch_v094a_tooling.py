#!/usr/bin/env python3
# v0.9.4a: tooling — write_content / assert_compiles / commit_with_guards.
#
# Eliminates 13.4/174 class (escape traps, triple-quote collision,
# stale-behavior forgotten) structurally:
#   - write_content(path, lines): list-of-strings instead of template.
#     Each line isolated -> no escape can span lines.
#   - assert_compiles(path): py_compile gate after every write_content.
#   - commit_with_guards(...): mandatory guards before git commit.
#
# Dogfood: after patch_utils.py is mutated, this patch imports and uses
# the new helpers via commit_with_guards().
#
# Run: uv run python scripts/patch_v094a_tooling.py

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


def _new_helper_block():
    # Lines as list; NO triple-quote in content. Literal \n in code
    # lines: written as "\\n" in Python source here, resolves to "\n"
    # (2 chars) in patch_utils.py which is what we want (it's Python
    # source, "\n" is escape in source). Double-checked below.
    L = []
    L.append("")
    L.append("")
    L.append("# ============ v0.9.4a: file write / compile / commit ============")
    L.append("")
    L.append("")
    L.append("def write_content(path, lines):")
    L.append("    # Write list of single-line strings to path.")
    L.append("    # Replaces triple-quote templates: each line is an isolated")
    L.append("    # string literal, so escape traps (backslash-n, triple-single,")
    L.append("    # triple-double) cannot span a multi-line template.")
    L.append("    # Empty list -> empty file. Non-empty -> joined + one trailing NL.")
    L.append("    import sys")
    L.append("    from pathlib import Path as _P")
    L.append("    _p = _P(path)")
    L.append("    if not lines:")
    L.append("        _p.write_text('')")
    L.append("        return")
    L.append("    for i, ln in enumerate(lines):")
    L.append("        if not isinstance(ln, str):")
    L.append("            print('write_content: line ' + str(i) + ' not str: '")
    L.append("                  + repr(ln))")
    L.append("            sys.exit(1)")
    L.append("    _p.write_text('\\n'.join(lines) + '\\n')")
    L.append("")
    L.append("")
    L.append("def assert_compiles(path):")
    L.append("    # py_compile gate. sys.exit(1) on SyntaxError, missing file.")
    L.append("    # Use after every write_content() to catch bad content.")
    L.append("    import py_compile")
    L.append("    import sys")
    L.append("    from pathlib import Path as _P")
    L.append("    _p = _P(path)")
    L.append("    if not _p.exists():")
    L.append("        print('assert_compiles: missing ' + str(_p))")
    L.append("        sys.exit(1)")
    L.append("    try:")
    L.append("        py_compile.compile(str(_p), doraise=True)")
    L.append("    except py_compile.PyCompileError as e:")
    L.append("        print('COMPILE FAIL: ' + str(_p))")
    L.append("        print(str(e)[:800])")
    L.append("        sys.exit(1)")
    L.append("")
    L.append("")
    L.append("def commit_with_guards(message, *, self_path=None,")
    L.append("                      stale_patterns=None, stale_dirs=None,")
    L.append("                      allow=None):")
    L.append("    # git add -A + commit, ALWAYS running pre-commit guards.")
    L.append("    # self_path: py_compile check.")
    L.append("    # stale_patterns+stale_dirs: assert_no_stale_behavior.")
    L.append("    # Use instead of sh(['git','commit','-m',msg]) in patches.")
    L.append("    import subprocess")
    L.append("    if self_path is not None:")
    L.append("        assert_compiles(self_path)")
    L.append("    if stale_patterns and stale_dirs:")
    L.append("        assert_no_stale_behavior(")
    L.append("            stale_patterns, stale_dirs, allow=allow,")
    L.append("        )")
    L.append("    subprocess.run(['git', 'add', '-A'], check=True)")
    L.append("    subprocess.run(['git', 'commit', '-m', message], check=True)")
    L.append("")
    L.append("")
    return L


def inject_to_patch_utils():
    print("\n=== patch_utils.py: inject helpers ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def write_content(" in s:
        print("  already present - skip")
        return
    anchor = "\n\ndef grep_and_warn("
    if anchor not in s:
        raise SystemExit("anchor 'def grep_and_warn' not found")
    block = "\n".join(_new_helper_block())
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)


def append_tests():
    print("\n=== tests/test_patch_utils.py: append ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_write_content_basic" in s:
        print("  already present - skip")
        return
    L = []
    L.append("")
    L.append("")
    L.append("# ============ v0.9.4a: write_content / assert_compiles ============")
    L.append("")
    L.append("")
    L.append("def test_write_content_basic(tmp_path):")
    L.append("    from cadw.patch_utils import write_content")
    L.append("    p = tmp_path / 'f.py'")
    L.append("    write_content(p, ['line1', 'line2'])")
    L.append("    assert p.read_text() == 'line1\\nline2\\n'")
    L.append("")
    L.append("")
    L.append("def test_write_content_empty(tmp_path):")
    L.append("    from cadw.patch_utils import write_content")
    L.append("    p = tmp_path / 'f.py'")
    L.append("    write_content(p, [])")
    L.append("    assert p.read_text() == ''")
    L.append("")
    L.append("")
    L.append("def test_write_content_with_quotes(tmp_path):")
    L.append("    from cadw.patch_utils import write_content")
    L.append("    p = tmp_path / 'f.py'")
    L.append("    write_content(p, [\"a = 'x'\", 'b = \"y\"'])")
    L.append("    text = p.read_text()")
    L.append("    assert 'a = ' in text and \"'x'\" in text")
    L.append("    assert 'b = ' in text and '\"y\"' in text")
    L.append("")
    L.append("")
    L.append("def test_write_content_with_backslashes(tmp_path):")
    L.append("    from cadw.patch_utils import write_content")
    L.append("    p = tmp_path / 'f.py'")
    L.append("    write_content(p, ['x = \"a\\\\nb\"'])")
    L.append("    text = p.read_text()")
    L.append("    # content must contain literal 2-char backslash-n, not newline")
    L.append("    assert 'a\\\\nb' in text")
    L.append("")
    L.append("")
    L.append("def test_write_content_unicode(tmp_path):")
    L.append("    from cadw.patch_utils import write_content")
    L.append("    p = tmp_path / 'f.py'")
    L.append("    write_content(p, ['# \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442', '# \u2192 \u2190'])")
    L.append("    text = p.read_text(encoding='utf-8')")
    L.append("    assert '\u0440\u0443\u0441\u0441\u043a\u0438\u0439' in text")
    L.append("")
    L.append("")
    L.append("def test_assert_compiles_ok(tmp_path):")
    L.append("    from cadw.patch_utils import assert_compiles, write_content")
    L.append("    p = tmp_path / 'ok.py'")
    L.append("    write_content(p, ['x = 1'])")
    L.append("    assert_compiles(p)")
    L.append("")
    L.append("")
    L.append("def test_assert_compiles_bad(tmp_path):")
    L.append("    from cadw.patch_utils import assert_compiles, write_content")
    L.append("    p = tmp_path / 'bad.py'")
    L.append("    write_content(p, ['def f('])")
    L.append("    import pytest")
    L.append("    with pytest.raises(SystemExit):")
    L.append("        assert_compiles(p)")
    L.append("")
    L.append("")
    L.append("def test_assert_compiles_missing(tmp_path):")
    L.append("    from cadw.patch_utils import assert_compiles")
    L.append("    import pytest")
    L.append("    with pytest.raises(SystemExit):")
    L.append("        assert_compiles(tmp_path / 'nope.py')")
    L.append("")
    p.write_text(s + "\n".join(L))
    # Dogfood: use new helper to validate the result.
    from cadw.patch_utils import assert_compiles
    assert_compiles(p)


def main():
    print("=== v0.9.4a tooling ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    inject_to_patch_utils()
    # Compile-check patch_utils.py immediately (bootstrap: uses
    # py_compile directly, since assert_compiles not imported yet).
    import py_compile
    try:
        py_compile.compile("cadw/patch_utils.py", doraise=True)
        print("  compile OK: cadw/patch_utils.py")
    except py_compile.PyCompileError as e:
        print("COMPILE FAIL: cadw/patch_utils.py")
        print(str(e)[:800])
        sys.exit(1)

    # Dogfood from here on: assert_compiles + commit_with_guards.
    from cadw.patch_utils import assert_compiles, commit_with_guards

    append_tests()

    print("\n=== pytest: test_patch_utils.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS TESTS FAILED ***")
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

    # Dogfood: commit via new wrapper.
    commit_with_guards(
        "v0.9.4a: tooling — patch_utils.write_content / "
        "assert_compiles / commit_with_guards; eliminates 13.4/174 "
        "escape-trap class structurally (no template strings)",
    )
    print("\n==> v0.9.4a applied.")


if __name__ == "__main__":
    main()
