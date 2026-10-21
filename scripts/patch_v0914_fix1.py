#!/usr/bin/env python3
# v0.9.14-fix1 — apply_or_fail validate= kwarg (TOML/JSON/none).
# v0.9.14 COMPLETE.
#
# v0.9.14 died at U3 (pyproject.toml version 0.1.0→0.9.14):
#   apply_or_fail: compile(new_s, str(p), 'exec')
#   → SyntaxError: cannot assign to expression here (line 6)
#
# Root: apply_or_fail (v0.9.10-fix3 §13.4/211) compile-before-write
# uses Python `compile()`. pyproject.toml is TOML (`name = "..."`),
# not Python. Non-Python files (`.toml`, `.json`, `.yaml`, `.md`,
# `.txt`) cannot be Python-compiled.
#
# §13.4/221 (new): compile-before-write assumes Python target. For
# non-Python formats, validation must be format-aware. Default:
# `.py`/`.pyi` → python compile; other extensions → `validate="none"`
# (no compile-check, still write-once atomic via temp path? no —
# current impl inlines write_text; format-aware error reporting
# deferred to scaffold v4). Explicit `validate="toml"|"json"|"none"`
# opts in.
#
# State on disk (from v0.9.14 partial):
#   F12-6a bool→float widen  — applied (log OK)
#   F12-6b clear RuntimeError — applied
#   F12-7 §11.5 empty-input note — applied
#   T2.2 receiver_type kwarg — applied
#   Tests (F12-6 x3, T2.2 x2) — applied
#   U3 pyproject version — FAILED (compile error)
#   commit — never reached
#
# fix1:
#   (a) apply_or_fail gets `validate` kwarg:
#       - None (default): infer from path extension
#         (.py/.pyi → "python", else "none")
#       - "python": compile(new_s, path, "exec") — existing
#       - "toml": tomllib.loads(new_s) — stdlib 3.11+
#       - "json": json.loads(new_s)
#       - "none": skip validation
#   (b) _collect_call_linenos unchanged; only apply_or_fail changes
#   (c) Tests: toml valid, toml invalid raises SystemExit
#   (d) Apply U3 with validate="toml"
#   (e) verify + dogfood + pytest + commit
#
# Run: uv run python scripts/patch_v0914_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) apply_or_fail validate= kwarg ============

def fix_apply_or_fail_validate():
    print("\n=== (a) apply_or_fail: validate= kwarg ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def apply_or_fail(path, old, new, *, label, context=10):\n"
            "    \"\"\"Replace old->new in path; return True if applied, False if already.\n"
            "\n"
            "    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.\n"
            "    Prints `context` lines around partial-match zone on miss (§13.4/160).\n"
            "    Idempotent: returns False if `new` already present (no write).\n"
            "\n"
            "    v0.9.12 (F12-3, §13.4/211): compile-before-write. `new_s`\n"
            "    is compile()'d in memory BEFORE p.write_text(). On\n"
            "    SyntaxError — SystemExit, file untouched. Without this,\n"
            "    bad anchor (e.g. 4-space substring of 8-space line) can\n"
            "    corrupt the file, and only then would assert_compiles\n"
            "    fail — but the file is already bad. Never reorder:\n"
            "    `compile()` MUST precede `write_text()`.\n"
            "    \"\"\"\n"
            "    from pathlib import Path as _P\n"
            "    p = _P(path)\n"
            "    s = p.read_text()\n"
            "    if new in s:\n"
            "        return False\n"
            "    if old not in s:\n"
            "        print(f\"  {label}: ANCHOR MISS in {path}\")\n"
            "        first_line = old.split(chr(10))[0].strip()\n"
            "        lines = s.split(chr(10))\n"
            "        for i, line in enumerate(lines):\n"
            "            if first_line and first_line in line:\n"
            "                start = max(0, i - 1)\n"
            "                for j in range(start, min(len(lines), i + context)):\n"
            "                    print(f\"    L{j + 1:4d}: {lines[j]}\")\n"
            "                break\n"
            "        raise SystemExit(f\"{label}: anchor missing in {path}\")\n"
            "    new_s = s.replace(old, new, 1)\n"
            "    try:\n"
            "        compile(new_s, str(p), 'exec')\n"
            "    except SyntaxError as e:\n"
            "        print(f\"  {label}: COMPILE FAIL for {path}: {e}\")\n"
            "        raise SystemExit(1) from e\n"
            "    # v0.9.12 (F12-2): newline=\"\" — Python text-mode 'w'\n"
            "    # converts \\n -> \\r\\n on Windows. Force LF for\n"
            "    # cross-platform deterministic file content.\n"
            "    p.write_text(new_s, newline=\"\")\n"
            "    return True\n"
        ),
        (
            "def apply_or_fail(path, old, new, *, label, context=10,\n"
            "                  validate=None):\n"
            "    \"\"\"Replace old->new in path; return True if applied, False if already.\n"
            "\n"
            "    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.\n"
            "    Prints `context` lines around partial-match zone on miss (§13.4/160).\n"
            "    Idempotent: returns False if `new` already present (no write).\n"
            "\n"
            "    v0.9.12 (F12-3, §13.4/211): compile-before-write for\n"
            "    Python files. `new_s` is compile()'d in memory BEFORE\n"
            "    p.write_text(). On SyntaxError — SystemExit, file untouched.\n"
            "    Never reorder: `compile()` MUST precede `write_text()`.\n"
            "\n"
            "    v0.9.14-fix1 (T2.9, §13.4/221): `validate` kwarg.\n"
            "    - None (default): infer from path extension.\n"
            "      `.py`/`.pyi` → \"python\", else → \"none\".\n"
            "    - \"python\": compile(new_s, path, 'exec').\n"
            "    - \"toml\": tomllib.loads(new_s) — stdlib 3.11+.\n"
            "    - \"json\": json.loads(new_s).\n"
            "    - \"none\": skip validation (non-code files: .md, .txt,\n"
            "      .yaml without parser).\n"
            "    Prior (v0.9.10-fix3-v0.9.14): always Python compile →\n"
            "    pyproject.toml (TOML) failed with cryptic\n"
            "    SyntaxError: cannot assign to expression. Fixed.\n"
            "    \"\"\"\n"
            "    from pathlib import Path as _P\n"
            "    p = _P(path)\n"
            "    s = p.read_text()\n"
            "    if new in s:\n"
            "        return False\n"
            "    if old not in s:\n"
            "        print(f\"  {label}: ANCHOR MISS in {path}\")\n"
            "        first_line = old.split(chr(10))[0].strip()\n"
            "        lines = s.split(chr(10))\n"
            "        for i, line in enumerate(lines):\n"
            "            if first_line and first_line in line:\n"
            "                start = max(0, i - 1)\n"
            "                for j in range(start, min(len(lines), i + context)):\n"
            "                    print(f\"    L{j + 1:4d}: {lines[j]}\")\n"
            "                break\n"
            "        raise SystemExit(f\"{label}: anchor missing in {path}\")\n"
            "    new_s = s.replace(old, new, 1)\n"
            "\n"
            "    # Format-aware validation (v0.9.14-fix1, §13.4/221)\n"
            "    if validate is None:\n"
            "        ext = p.suffix.lower()\n"
            "        if ext in (\".py\", \".pyi\"):\n"
            "            validate = \"python\"\n"
            "        else:\n"
            "            validate = \"none\"\n"
            "    if validate == \"python\":\n"
            "        try:\n"
            "            compile(new_s, str(p), 'exec')\n"
            "        except SyntaxError as e:\n"
            "            print(f\"  {label}: COMPILE FAIL for {path}: {e}\")\n"
            "            raise SystemExit(1) from e\n"
            "    elif validate == \"toml\":\n"
            "        try:\n"
            "            import tomllib as _tl\n"
            "            _tl.loads(new_s)\n"
            "        except Exception as e:\n"
            "            print(f\"  {label}: TOML FAIL for {path}: {e}\")\n"
            "            raise SystemExit(1) from e\n"
            "    elif validate == \"json\":\n"
            "        try:\n"
            "            import json as _json\n"
            "            _json.loads(new_s)\n"
            "        except Exception as e:\n"
            "            print(f\"  {label}: JSON FAIL for {path}: {e}\")\n"
            "            raise SystemExit(1) from e\n"
            "    elif validate == \"none\":\n"
            "        pass\n"
            "    else:\n"
            "        raise SystemExit(\n"
            "            f\"{label}: validate={validate!r} not in \"\n"
            "            f\"{{python, toml, json, none}}\"\n"
            "        )\n"
            "\n"
            "    # v0.9.12 (F12-2): newline=\"\" — Python text-mode 'w'\n"
            "    # converts \\n -> \\r\\n on Windows. Force LF for\n"
            "    # cross-platform deterministic file content.\n"
            "    p.write_text(new_s, newline=\"\")\n"
            "    return True\n"
        ),
        label="apply_or_fail validate",
    )
    print("  OK")


# ============ (c) tests for validate kwarg ============

T29_TESTS = [
    "",
    "",
    "# ============ v0.9.14-fix1 (T2.9, §13.4/221): validate= kwarg ============",
    "",
    "",
    "def test_apply_or_fail_toml_valid(tmp_path):",
    "    \"\"\"T2.9: validate='toml' — TOML file applied with toml validation.\"\"\"",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.toml'",
    "    p.write_text('[project]\\nname = \"x\"\\nversion = \"0.1.0\"\\n')",
    "    applied = apply_or_fail(",
    "        str(p), 'version = \"0.1.0\"', 'version = \"0.2.0\"',",
    "        label='t', validate='toml',",
    "    )",
    "    assert applied is True",
    "    assert 'version = \"0.2.0\"' in p.read_text()",
    "",
    "",
    "def test_apply_or_fail_toml_invalid(tmp_path):",
    "    \"\"\"T2.9: validate='toml' — invalid TOML → SystemExit, no write.\"\"\"",
    "    import pytest",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.toml'",
    "    original = 'name = \"x\"\\n'",
    "    p.write_text(original)",
    "    with pytest.raises(SystemExit):",
    "        apply_or_fail(",
    "            str(p), 'name = \"x\"', 'broken = = =',",
    "            label='t', validate='toml',",
    "        )",
    "    assert p.read_text() == original",
    "",
    "",
    "def test_apply_or_fail_infers_none_for_toml(tmp_path):",
    "    \"\"\"Default (validate=None): .toml → 'none' (no Python compile).\"\"\"",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.toml'",
    "    p.write_text('name = \"x\"\\n')",
    "    applied = apply_or_fail(",
    "        str(p), 'name = \"x\"', 'name = \"y\"', label='t',",
    "    )",
    "    assert applied is True",
    "    assert 'name = \"y\"' in p.read_text()",
    "",
    "",
    "def test_apply_or_fail_python_default_unchanged(tmp_path):",
    "    \"\"\"Default (validate=None): .py → 'python' compile still enforced.\"\"\"",
    "    import pytest",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 1\\n')",
    "    with pytest.raises(SystemExit):",
    "        apply_or_fail(",
    "            str(p), 'x = 1', 'def broken(', label='t',",
    "        )",
    "    assert p.read_text() == 'x = 1\\n'",
]


def add_t29_tests():
    print("\n=== (c) T2.9 tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_apply_or_fail_toml_valid" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(T29_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ (d) Apply U3 with validate="toml" ============

def u3_pyproject():
    print("\n=== (d) U3: pyproject version 0.1.0 → 0.9.14 (validate=toml) ===")
    applied = pu.apply_or_fail(
        "pyproject.toml",
        'version = "0.1.0"',
        'version = "0.9.14"',
        label="U3",
        validate="toml",
    )
    if applied:
        print("  applied")
    else:
        print("  already applied")
    # Sanity: file valid TOML + has new version
    import tomllib
    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert data["project"]["version"] == "0.9.14", (
        f"version is {data['project']['version']!r}, expected '0.9.14'"
    )
    print("  verified: pyproject.toml valid TOML, version = 0.9.14")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.14 + fix1 + regression ===")
    pu.verify_anchors([
        # fix1 (T2.9)
        ("cadw/patch_utils.py", "validate=None"),
        ("cadw/patch_utils.py", "v0.9.14-fix1 (T2.9, §13.4/221)"),
        ("cadw/patch_utils.py", 'elif validate == "toml":'),
        ("tests/test_patch_utils.py", "test_apply_or_fail_toml_valid"),
        ("tests/test_patch_utils.py", "test_apply_or_fail_toml_invalid"),
        # U3
        ("pyproject.toml", 'version = "0.9.14"'),
        # v0.9.14 (applied before fix1)
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/dataframe.py", "§11.5, v0.9.14 F12-7"),
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("tests/test_apply.py", "test_apply_mixed_bool_float_widens"),
        # v0.9.13/12/11/10/9 regression
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.14-fix1 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import tomllib
        from pathlib import Path as _P

        # U3: pyproject version correct + valid TOML
        pj = _P("pyproject.toml").read_text()
        assert 'version = "0.9.14"' in pj
        data = tomllib.loads(pj)
        assert data["project"]["version"] == "0.9.14"

        # T2.9: apply_or_fail accepts validate kwarg
        from cadw.patch_utils import apply_or_fail
        sig = inspect.signature(apply_or_fail)
        assert "validate" in sig.parameters
        assert sig.parameters["validate"].default is None

        # AST: toml branch present, json branch present, python default
        src_aof = _tw.dedent(inspect.getsource(apply_or_fail))
        tree = _ast.parse(src_aof)
        # check for tomllib import
        tomllib_imports = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Import)
            and any(a.name == "tomllib" for a in n.names)
        ]
        assert tomllib_imports, "T2.9: tomllib import missing"
        # check for `validate == "toml"` comparison
        toml_cmp = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Compare)
        ]
        assert toml_cmp, "T2.9: no comparison in apply_or_fail"
        # Sanity: string literal 'toml' present
        assert '"toml"' in src_aof or "'toml'" in src_aof

        # F12-6a: bool widen in _maybe_upgrade_schema
        from cadw.dataframe import CadwDataFrame
        src_pga = _tw.dedent(inspect.getsource(CadwDataFrame._per_group_apply))
        assert "v0.9.14 (F12-6a)" in src_pga
        assert "is_boolean(field.type)" in src_pga
        assert "v0.9.14 (F12-6b)" in src_pga
        assert "inconsistent types across groups" in src_pga
        assert "§11.5, v0.9.14 F12-7" in src_pga

        # T2.2: receiver_type filter
        src_per_row = "def f(x, i):\n    return x[i].as_py()\n"
        try:
            from cadw.patch_utils import assert_no_call_in_source
            assert_no_call_in_source(
                src_per_row, "as_py", receiver_type=_ast.Subscript,
            )
            raise AssertionError("T2.2: per-row as_py not flagged")
        except AssertionError as e:
            if "T2.2: per-row" in str(e):
                raise

        print("  U3 pyproject.toml version 0.9.14 (valid TOML) OK")
        print("  T2.9 validate= kwarg (toml branch present) OK")
        print("  F12-6a/6b/7 unchanged OK")
        print("  T2.2 receiver_type filter unchanged OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.14-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_apply_or_fail_validate()
    add_t29_tests()
    importlib.reload(pu)
    u3_pyproject()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.14-fix1: complete v0.9.14 (apply_or_fail validate= "
        "kwarg + U3 pyproject version). v0.9.14 died at U3 "
        "(pyproject.toml version 0.1.0→0.9.14): apply_or_fail "
        "compile-before-write (v0.9.10-fix3 §13.4/211) uses Python "
        "`compile()`, pyproject.toml is TOML → "
        "SyntaxError: cannot assign to expression. §13.4/221 (new): "
        "compile-before-write assumes Python target; non-Python "
        "formats need format-aware validation. Fix: apply_or_fail "
        "gets `validate=None|'python'|'toml'|'json'|'none'` kwarg. "
        "None (default) infers from extension (.py/.pyi → python, "
        "else → none). Explicit toml/json validate via "
        "tomllib.loads/json.loads (SystemExit on failure, file "
        "untouched). Prior always Python compile — pyproject.toml "
        "and any .md/.json/.yaml/.txt edit would fail. State on "
        "disk from v0.9.14 partial: F12-6a (bool→float64 widen, "
        "parallel to C1 int→float), F12-6b (clear RuntimeError on "
        "non-widenable mismatch, was cryptic ArrowInvalid), F12-7 "
        "(§11.5 empty-input mode note), T2.2 (assert_no_call_in_"
        "source receiver_type kwarg for precise AST checks, F12-5 "
        "dogfood needed this). All applied before U3 (log OK×4). "
        "fix1: (a) apply_or_fail validate= kwarg (tomllib/json/"
        "python/none); (b) 4 new tests (toml valid, toml invalid "
        "no-write, infer none for .toml, .py still compiles); "
        "(c) U3 applied with validate='toml' + sanity tomllib.parse; "
        "(d) verify + dogfood + pytest + commit. v0.9.14 COMPLETE: "
        "15 patches this session (v0.9.8a–v0.9.14-fix1). Deferred: "
        "A1 (dataframe.py split 5404 lines), T2.5/T2.6/T2.7 "
        "(scaffold v4 tooling), N6 (Dataset.iter_batches guard, "
        "user responsibility), R3 (planner spill stub, v0.2 "
        "backlog). Next: Pingis formal (fresh snapshot + bench "
        "TASK2/TASK3-style) OR A1 split.",
        self_path="scripts/patch_v0914_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.14-fix1 applied. v0.9.14 COMPLETE.")


if __name__ == "__main__":
    main()
