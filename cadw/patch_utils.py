"""Shared helpers for CADW patch scripts (v0.9.1a).

Extracted from v0.9.0a/b/c experience: recent patch scripts duplicated
CHANGELOG/§-anchor logic with different bug-patterns each time
(§13.4/140 [Unreleased]-after-tag; §13.4/144 cross-file rename;
§13.4/146 silent-skip on missing anchor; §13.4/147 partial contract).

This module fixes the pattern once. Patch scripts import from
`cadw.patch_utils` and use these helpers, so future patches inherit
the fixes instead of reimplementing them.

Not on any hot path (imported only from `scripts/patch_*.py`).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ============ CHANGELOG helpers ============

def ensure_unreleased_header(text: str) -> str:
    """Ensure `## [Unreleased]` header exists; prepend if missing.

    Rationale (§13.4/140, /146): after a release tag, CHANGELOG starts
    with `## [X.Y.Z]`. Patch scripts that skip on missing [Unreleased]
    silently lose their entry. This helper creates the header as a
    fallback (prepend before first `## [`), never skips.
    """
    if "## [Unreleased]" in text:
        return text
    m = re.search(r'^## \[', text, re.MULTILINE)
    if m:
        return text[:m.start()] + "## [Unreleased]\n\n" + text[m.start():]
    return "## [Unreleased]\n\n" + text


def append_unreleased_entry(text: str, entry: str) -> str:
    """Append `entry` (must start with `### ...`) under `## [Unreleased]`.

    `entry` should be a section like `### Added ...` — the function
    inserts it right after the `## [Unreleased]` header (before any
    existing sub-sections), maintaining descending order by convention.
    """
    text = ensure_unreleased_header(text)
    header = "## [Unreleased]\n\n"
    # If there's already content under [Unreleased] (starting with `###`),
    # insert new entry BEFORE it (newest first).
    pos = text.find(header)
    if pos < 0:
        raise ValueError("ensure_unreleased_header failed to create header")
    insert_at = pos + len(header)
    return text[:insert_at] + entry.rstrip() + "\n\n" + text[insert_at:]


# ============ DECISIONS.md section helpers ============

_SECTION_PATTERN = r"^#{{1,6}}\s+(?:§)?{section}\b"

def find_section_anchor(text: str, section: str) -> Optional[int]:
    """Return char offset of section header (§{section}) or None.

    Matches `# 11.6`, `## 11.6`, `### 11.6`, `#### 11.6.` etc.,
    including `§` prefix (`### §11.6`). Regex-based: tolerates
    trailing dot, sub-section markers, and heading-level variations.
    """
    pat = re.compile(
        _SECTION_PATTERN.format(section=re.escape(section)),
        re.MULTILINE,
    )
    m = pat.search(text)
    return m.start() if m else None


def insert_before_section(text: str, section: str,
                          insertion: str) -> tuple[str, bool]:
    """Insert `insertion` text before §{section} header.

    Returns (new_text, found). If section not found, returns
    (text, False) — caller decides (warn and append at end, or skip).
    Never silently no-ops — caller sees `found` flag.
    """
    pos = find_section_anchor(text, section)
    if pos is None:
        return text, False
    return text[:pos] + insertion.rstrip() + "\n\n" + text[pos:], True


# ============ Guard test helper ============

def assert_guards_green(
    guard_tests: list[str],
    *,
    timeout: int = 900,
) -> None:
    """Run pytest on guard tests; exit(1) on failure.

    Extraction of the try/except that every patch script did:
        r = subprocess.run(["uv", "run", "pytest", *guard_tests, "-v"])
        if r.returncode != 0: print(...); sys.exit(1)
    Called from patch script after code changes, before `git commit`.
    Enforces §13.4/140 discipline (no WIP commit on red).
    """
    print(f"  $ uv run pytest {' '.join(guard_tests)} -q")
    r = subprocess.run(
        ["uv", "run", "pytest", *guard_tests, "-q"],
        timeout=timeout,
    )
    if r.returncode != 0:
        print(f"\n*** GUARD FAILURE: {guard_tests} ***")
        sys.exit(1)


# ============ Grep-based stale-behavior detection (§13.4/152) ============
#
# Recurring pattern: patch changes/removes/adds behavior; a test in the
# same or other file asserts the OLD behavior. Manual "grep before
# commit" discipline failed 3 times (§13.4/144, /149, /152). Mechanical
# tooling is the only reliable fix.

# ============ v0.9.4a: file write / compile / commit ============


def write_content(path, lines):
    # Write list of single-line strings to path.
    # Replaces triple-quote templates: each line is an isolated
    # string literal, so escape traps (backslash-n, triple-single,
    # triple-double) cannot span a multi-line template.
    # Empty list -> empty file. Non-empty -> joined + one trailing NL.
    import sys
    from pathlib import Path as _P
    _p = _P(path)
    if not lines:
        _p.write_text('')
        return
    for i, ln in enumerate(lines):
        if not isinstance(ln, str):
            print('write_content: line ' + str(i) + ' not str: '
                  + repr(ln))
            sys.exit(1)
    _p.write_text('\n'.join(lines) + '\n')


def assert_compiles(path):
    # py_compile gate. sys.exit(1) on SyntaxError, missing file.
    # Use after every write_content() to catch bad content.
    import py_compile
    import sys
    from pathlib import Path as _P
    _p = _P(path)
    if not _p.exists():
        print('assert_compiles: missing ' + str(_p))
        sys.exit(1)
    try:
        py_compile.compile(str(_p), doraise=True)
    except py_compile.PyCompileError as e:
        print('COMPILE FAIL: ' + str(_p))
        print(str(e)[:800])
        sys.exit(1)


def commit_with_guards(message, *, self_path=None,
                      stale_patterns=None, stale_dirs=None,
                      allow=None, check_whitelist=False):
    # git add -A + commit, ALWAYS running pre-commit guards.
    # self_path: py_compile check.
    # stale_patterns+stale_dirs: assert_no_stale_behavior.
    # check_whitelist: AST-check WHITELISTED_FUNCS (v0.9.5a);
    # parameter renamed from `verify_whitelist` (fix3): old name
    # shadowed the module-level `verify_whitelist()` function ->
    # TypeError: 'bool' object is not callable.
    # Use instead of sh(['git','commit','-m',msg]) in patches.
    import subprocess
    if self_path is not None:
        assert_compiles(self_path)
    if stale_patterns and stale_dirs:
        assert_no_stale_behavior(
            stale_patterns, stale_dirs, allow=allow,
        )
    if check_whitelist:
        verify_whitelist()
    subprocess.run(['git', 'add', '-A'], check=True)
    subprocess.run(['git', 'commit', '-m', message], check=True)



def run_pytest_iterate(paths=None, *, timeout=900):
    # pytest -x --lf -q --no-header on `paths` (or full).
    # Iteration helper: stop at first failure, only last-failed
    # (after first run, .pytest_cache stores fails). Full suite is
    # the commit gate; this is for fast feedback loops.
    import subprocess
    import sys
    cmd = ['uv', 'run', 'pytest', '-x', '--lf', '-q',
           '--no-header', '-m', 'not slow']
    if paths:
        cmd.extend(paths)
    print('  $ ' + ' '.join(cmd))
    r = subprocess.run(cmd, timeout=timeout)
    if r.returncode != 0:
        sys.exit(r.returncode)



# ============ v0.9.5a: verify_whitelist (AST-orphan) ============


def _collect_all_qualnames(cadw_root):
    # Walk cadw_root/**/*.py; return set of {path}:{qualname}.
    # qualname = func_name (module-level) or Class.method.
    # Nested funcs inside methods: NOT included (whitelist is per-
    # top-level callable).
    import ast as _ast
    from pathlib import Path as _P
    root = _P(cadw_root)
    parent = root.parent
    result = set()
    if not root.exists():
        return result
    for py in root.rglob('*.py'):
        try:
            src = py.read_text(encoding='utf-8')
            tree = _ast.parse(src)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        try:
            rel = py.relative_to(parent).as_posix()
        except ValueError:
            rel = py.as_posix()
        def _walk_defs(node, prefix):
            # Recursively descend into FunctionDef/ClassDef bodies.
            # Qualname = Class.method.nested (dot-separated levels).
            # Matches test_arrow_call_memory_guard._qualname
            # semantics (v0.9.5a-fix1): ast.walk sees nested
            # FunctionDef nodes, so whitelist keys must include them.
            body = getattr(node, 'body', None)
            if body is None:
                return
            for child in body:
                if isinstance(child, _ast.FunctionDef):
                    qn = prefix + child.name
                    result.add(rel + ':' + qn)
                    _walk_defs(child, qn + '.')
                elif isinstance(child, _ast.ClassDef):
                    cp = prefix + child.name + '.'
                    for item in child.body:
                        if isinstance(item, _ast.FunctionDef):
                            qn = cp + item.name
                            result.add(rel + ':' + qn)
                            _walk_defs(item, qn + '.')
        _walk_defs(tree, '')
    return result


def _load_whitelist_from_test():
    # Import WHITELISTED_FUNCS from test_arrow_call_memory_guard.
    import importlib
    mod = importlib.import_module(
        'tests.test_arrow_call_memory_guard'
    )
    return getattr(mod, 'WHITELISTED_FUNCS', {})


def verify_whitelist(*, whitelist=None, cadw_root='cadw',
                     raise_on_orphan=True):
    # AST-verify that every key in WHITELISTED_FUNCS matches a real
    # function/method in cadw_root. Orphan (key without real def) is
    # dead whitelist: silent — key never matched, so guard doesn't
    # apply. Block commit (fail-fast).
    #
    # whitelist: dict override (None -> load from test file).
    # cadw_root: cadw/ dir (default 'cadw').
    # raise_on_orphan: sys.exit(1) on orphans (default True).
    #
    # Returns: {'orphans': [(key, reason)], 'malformed': [(key, r)]}.
    import sys as _sys
    if whitelist is None:
        whitelist = _load_whitelist_from_test()
    actual = _collect_all_qualnames(cadw_root)
    orphans = []
    malformed = []
    for key, rationale in whitelist.items():
        if not isinstance(key, str):
            malformed.append((repr(key), 'non-str key'))
            continue
        if ':' not in key:
            malformed.append((key, 'bare key (K1.1 requires path:qualname)'))
            continue
        if key not in actual:
            orphans.append((key, 'not in cadw AST qualnames'))
    if malformed or orphans:
        print('verify_whitelist: FAIL')
        for k, r in malformed:
            print('  malformed: ' + str(k) + ' (' + r + ')')
        for k, r in orphans:
            print('  orphan: ' + str(k) + ' (' + r + ')')
        if raise_on_orphan:
            _sys.exit(1)
    else:
        print('verify_whitelist: OK ('
              + str(len(whitelist)) + ' keys, all valid)')
    return {'orphans': orphans, 'malformed': malformed}


def grep_and_warn(
    patterns: list[str],
    dirs: list[str],
    *,
    label: str = "stale-behavior",
) -> list[str]:
    """Grep `dirs` for `patterns`; print warnings on hits. Returns hits.

    Non-fatal. Use in patch scripts BEFORE `git add` to surface stale
    assertions in tests that a behavior change may have broken.
    `patterns` are literal strings (grep -F), not regex — safe for
    strings with special chars.
    """
    hits: list[str] = []
    for pat in patterns:
        for d in dirs:
            if not Path(d).exists():
                continue
            r = subprocess.run(
                ["grep", "-rFn", pat, d],
                capture_output=True, text=True,
            )
            for line in r.stdout.strip().splitlines():
                hits.append(line)
    if hits:
        print(f"\n  [{label}] {len(hits)} reference(s) found:")
        for line in hits[:20]:
            print(f"    {line}")
    return hits


# ============ v0.9.6d: AST + subprocess + scoped-idempotency ============
#
# Six consecutive patch-script meta-bugs (§13.4/178, /179, /182,
# /183, /184, /185) — these helpers make the patterns structurally
# impossible:
#   - substring `assert 'X' not in src` catches comments
#     → assert_no_call_in_source() / assert_call_order() [AST]
#   - `python scripts/foo.py` puts scripts/ on sys.path[0]
#     → run_smoke_inline() [uv run python -c]
#   - whole-file idempotency guard matches tokens from other patches
#     → is_patched(fn, *markers) [inspect.getsource(fn)]


def _collect_call_linenos(tree):
    """Return {attr_or_name: [lineno, ...]} for all Call nodes.

    Handles both `obj.attr(...)` (ast.Attribute, use .attr) and
    `bare(...)` (ast.Name, use .id). Docstrings/comments are
    invisible to ast.walk over Call nodes (§13.4/161, /183, /185).
    """
    import ast as _ast
    out: dict = {}
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        f = node.func
        if isinstance(f, _ast.Name):
            out.setdefault(f.id, []).append(node.lineno)
        elif isinstance(f, _ast.Attribute):
            out.setdefault(f.attr, []).append(node.lineno)
    return out


def assert_no_call_in_source(src, *names, fn_name="<unknown>"):
    """AST: raise AssertionError if any Call node matches `name`.

    `name` may be a bare function call (`foo()`) or an attribute
    call (`obj.foo()`). Comments and docstrings are invisible
    (§13.4/161, /183) — only ast.Call nodes are checked. Replaces
    `assert 'foo' not in src`, which catches `# No foo() here`
    and `Use CadwDataFrame.foo(...)` docstrings.
    """
    import ast as _ast
    import textwrap as _tw
    try:
        tree = _ast.parse(_tw.dedent(src))
    except SyntaxError as e:
        raise AssertionError(
            f"assert_no_call_in_source({fn_name}): parse error {e}"
        ) from None
    calls = _collect_call_linenos(tree)
    violations = {n: calls[n] for n in names if n in calls}
    if violations:
        detail = "; ".join(
            f"{n!r} at line(s) {ls}" for n, ls in violations.items()
        )
        raise AssertionError(
            f"assert_no_call_in_source({fn_name}): forbidden Call "
            f"node(s) found: {detail}"
        )


def assert_call_order(src, first_name, second_name, *, fn_name="<unknown>"):
    """AST: assert min(lineno of first Call) < min(lineno of second).

    Raises AssertionError if either Call is absent, or if first does
    not precede second. Docstrings mentioning `first_name`/`second_name`
    are invisible (§13.4/161, /185). Use instead of
    `src.index('first') < src.index('second')`.
    """
    import ast as _ast
    import textwrap as _tw
    try:
        tree = _ast.parse(_tw.dedent(src))
    except SyntaxError as e:
        raise AssertionError(
            f"assert_call_order({fn_name}): parse error {e}"
        ) from None
    calls = _collect_call_linenos(tree)
    if first_name not in calls:
        raise AssertionError(
            f"assert_call_order({fn_name}): first Call "
            f"{first_name!r} not found"
        )
    if second_name not in calls:
        raise AssertionError(
            f"assert_call_order({fn_name}): second Call "
            f"{second_name!r} not found"
        )
    first_min = min(calls[first_name])
    second_min = min(calls[second_name])
    if first_min >= second_min:
        raise AssertionError(
            f"assert_call_order({fn_name}): {first_name!r} "
            f"(first line {first_min}) must precede {second_name!r} "
            f"(first line {second_min})"
        )


def is_patched(fn, *markers):
    """Scoped idempotency guard: fn source contains ALL markers.

    Use instead of `if marker in whole_file: skip` (§13.4/182).
    Whole-file substring matches tokens from OTHER patches in the
    same file — e.g. v0.9.6b2's `if 'v0.9.6b2' in s and 'CADW-native'
    in s` matched because add_len() wrote 'v0.9.6b2' to __len__
    docstring and _ColumnProxy.mean had 'CADW-native' from v0.9.6b1.

    Returns True iff `inspect.getsource(fn)` contains every marker.
    """
    import inspect as _inspect
    src = _inspect.getsource(fn)
    return all(m in src for m in markers)


def run_smoke_inline(script, *, timeout=60):
    """Run `uv run python -c <script>` (cwd on sys.path[0]).

    Use instead of writing a temp file under scripts/ and running
    `python scripts/foo.py`: that puts `scripts/` (not cwd) on
    sys.path[0] → `import cadw` fails with ModuleNotFoundError
    (§13.4/184). Stdout from the script is echoed (2-space indent);
    on failure prints last 1200 chars of stderr, then sys.exit(1).
    """
    import subprocess as _sp
    import sys as _sys
    r = _sp.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines():
            print("  " + line)
    if r.returncode != 0:
        print("  STDERR: " + r.stderr.strip()[-1200:])
        _sys.exit(1)


def assert_no_stale_behavior(
    patterns: list[str],
    dirs: list[str],
    *,
    label: str = "stale-behavior",
    allow: list[str] | None = None,
) -> None:
    """Like grep_and_warn, but exits(1) if hits (excluding `allow` files).

    Use in patch scripts whose fix changes/removes/adds behavior:
    before `git add`, call with `patterns = old behavior strings` and
    `dirs = ["tests/"]`. Any hit = a test asserting old behavior; fix
    the test or the allow-list BEFORE commit.

    `allow` — list of file path substrings to exclude (e.g., ["test_new"]
    if a test file explicitly tests the removed behavior — rare).
    """
    hits = grep_and_warn(patterns, dirs, label=label)
    if allow:
        hits = [h for h in hits if not any(a in h for a in allow)]
    if hits:
        print(f"\n*** {label}: {len(hits)} stale reference(s) remain ***")
        print("Fix tests or update allow-list before commit.")
        sys.exit(1)
