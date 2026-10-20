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
                      allow=None):
    # git add -A + commit, ALWAYS running pre-commit guards.
    # self_path: py_compile check.
    # stale_patterns+stale_dirs: assert_no_stale_behavior.
    # Use instead of sh(['git','commit','-m',msg]) in patches.
    import subprocess
    if self_path is not None:
        assert_compiles(self_path)
    if stale_patterns and stale_dirs:
        assert_no_stale_behavior(
            stale_patterns, stale_dirs, allow=allow,
        )
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
