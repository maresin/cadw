#!/usr/bin/env python3
# v0.9.6e1 — pandas: required -> optional (+dev-only for tests).
#
# Milestone: user of CADW does not need pandas to use the framework.
# `import cadw` does not pull pandas (verified in dogfood).
#
# Changes:
#   1. pyproject.toml:
#      - dependencies: remove "pandas>=3.0.2"
#      - [project.optional-dependencies].pandas = ["pandas>=3.0"]
#        (for to_batches(format="pandas"))
#      - [dependency-groups].dev: add "pandas>=3.0"
#        (test reference helpers use pandas; e.g.
#        _helpers/per_group_reference.py, test_apply.py equivalence)
#   2. uv lock — regenerate lock (rollback pyproject+lock on failure).
#      Do NOT run `uv sync` — current .venv keeps pandas (dev-group),
#      tests stay green. Fresh user `pip install cadw-framework`
#      does not get pandas.
#   3. cadw/dataframe.py::_convert_batch:
#      - format="pandas": catch ImportError -> friendly message
#        (hint: install cadw-framework[pandas]). Arrow's to_pandas()
#        raises ModuleNotFoundError without pandas.
#   4. tests/test_no_pandas_in_ops.py:
#      - extend CADW_OPS root scope to whole cadw/
#      - add test_no_pandas_import_anywhere_in_cadw
#      - allow-list: cadw/viz/seaborn_adapter.py (seaborn API requires
#        pandas.DataFrame; adapter is intentional bridge)
#   5. tests/test_no_pandas_in_ops.py:
#      - add test_pandas_optional_in_pyproject (parse pyproject, assert
#        pandas in [project.optional-dependencies].pandas and
#        [dependency-groups].dev, NOT in [project.dependencies])
#
# Dogfood:
#   - `import cadw; assert 'pandas' not in sys.modules` (subprocess,
#     fresh interpreter — §13.4/184 run_smoke_inline pattern).
#   - Existing `pandas` still importable from test env (dev-group).
#
# Run: uv run python scripts/patch_v096e1_pandas_optional.py

from __future__ import annotations
import os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    assert_compiles, commit_with_guards, assert_no_call_in_source,
)


# ============ 1. Backup pyproject.toml + uv.lock ============

def backup():
    print("\n=== backup pyproject.toml + uv.lock ===")
    d = Path(".pingis/pre-v096e1")
    d.mkdir(parents=True, exist_ok=True)
    for f in ("pyproject.toml", "uv.lock"):
        src = Path(f)
        if not src.exists():
            print(f"  {f}: missing — skip")
            continue
        dst = d / f
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  {f} -> {dst}")
        else:
            print(f"  {f}: backup already exists")


# ============ 2. Edit pyproject.toml ============

def edit_pyproject():
    print("\n=== pyproject.toml: pandas -> optional + dev ===")
    p = Path("pyproject.toml")
    s = p.read_text()

    if "[project.optional-dependencies]" in s and 'pandas = ["pandas' in s:
        print("  already patched — skip")
        return

    # 2a. Remove pandas from dependencies.
    old_deps = '''dependencies = [
    "numpy>=2.4.4",
    "pandas>=3.0.2",
    "psutil>=7.2.2",
    "pyarrow>=26.0.0",
    "xxhash>=3.4.1",
]
'''
    new_deps = '''dependencies = [
    "numpy>=2.4.4",
    "psutil>=7.2.2",
    "pyarrow>=26.0.0",
    "xxhash>=3.4.1",
]

[project.optional-dependencies]
# v0.9.6e1: pandas is opt-in for `to_batches(format="pandas")`.
# Core (`import cadw`, ops, ML, pivot) is pandas-free.
pandas = ["pandas>=3.0"]
'''
    if old_deps not in s:
        raise SystemExit("dependencies anchor not found (pyproject)")
    s = s.replace(old_deps, new_deps, 1)

    # 2b. Add pandas to [dependency-groups].dev.
    old_dev = '''dev = [
    "matplotlib>=3.8",
    "plotly>=5.18",
    "seaborn>=0.13",
    "pytest>=9.0.3",
    "pylint>=4.0.5",
    "pyright>=1.1.409",
    "ruff>=0.15.12",
    "pandas-stubs>=3.0.0.260204",
]
'''
    new_dev = '''dev = [
    "matplotlib>=3.8",
    "pandas>=3.0",          # v0.9.6e1: test reference helpers use pandas
    "plotly>=5.18",
    "seaborn>=0.13",
    "pytest>=9.0.3",
    "pylint>=4.0.5",
    "pyright>=1.1.409",
    "ruff>=0.15.12",
    "pandas-stubs>=3.0.0.260204",
]
'''
    if old_dev not in s:
        raise SystemExit("[dependency-groups].dev anchor not found")
    s = s.replace(old_dev, new_dev, 1)

    p.write_text(s)
    print("  pyproject.toml updated")


# ============ 3. uv lock (rollback on fail) ============

def regen_lock():
    print("\n=== uv lock (regenerate) ===")
    r = subprocess.run(
        ["uv", "lock"], capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print("  STDERR: " + r.stderr.strip()[-1200:])
        print("  ROLLBACK pyproject.toml + uv.lock from backup")
        d = Path(".pingis/pre-v096e1")
        for f in ("pyproject.toml", "uv.lock"):
            bak = d / f
            if bak.exists():
                shutil.copy2(bak, Path(f))
                print(f"  restored {f}")
        raise SystemExit("uv lock failed")
    # Show git diff summary of uv.lock (line-count only, keep output short).
    print("  OK")
    r = subprocess.run(
        ["git", "diff", "--stat", "uv.lock"],
        capture_output=True, text=True, timeout=30,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:3]:
            print("  " + line)


# ============ 4. Friendly ImportError for format="pandas" ============

def update_convert_batch():
    print("\n=== dataframe.py::_convert_batch: friendly pandas error ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "to_batches(format='pandas') requires pandas" in s:
        print("  already patched — skip")
        return

    old = '''def _convert_batch(batch: pa.RecordBatch, format: str):
    """Конвертация одного батча. Векторизованно, без .as_py()."""
    if format == "arrow":
        return batch
    if format == "pandas":
        return batch.to_pandas()
'''
    new = '''def _convert_batch(batch: pa.RecordBatch, format: str):
    """Конвертация одного батча. Векторизованно, без .as_py()."""
    if format == "arrow":
        return batch
    if format == "pandas":
        # v0.9.6e1: pandas is optional. Arrow's to_pandas() raises
        # ModuleNotFoundError without pandas; convert to a clear
        # install-hint error.
        try:
            return batch.to_pandas()
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "to_batches(format='pandas') requires pandas. "
                "Install with: uv add 'cadw-framework[pandas]' "
                "(or: pip install pandas). For pandas-free usage, "
                "use format='arrow' | 'numpy' | 'torch'."
            ) from e
'''
    if old not in s:
        raise SystemExit("_convert_batch anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 5. Extend test_no_pandas_in_ops.py ============

NEW_TEST_LINES = [
    "",
    "",
    "# ============ v0.9.6e1: whole-cadw pandas guard ============",
    "",
    "",
    "_ALLOWED_PANDAS_FILES = {",
    '    # seaborn API requires pandas.DataFrame; adapter is the',
    '    # intentional bridge (Arrow CADW -> seaborn). Not execution',
    '    # path — only invoked when user explicitly calls',
    '    # df.plot.seaborn.* or cadw.viz.seaborn_adapter.*',
    '    "cadw/viz/seaborn_adapter.py":',
    '        "seaborn API requires pandas.DataFrame; adapter bridge",',
    "}",
    "",
    "",
    "def test_no_pandas_import_anywhere_in_cadw():",
    '    """No `import pandas` in any cadw/**/*.py, except explicit',
    '    allow-list (viz/seaborn_adapter.py)."""',
    "    cadw_root = CADW_OPS.parent",
    "    violations: list[str] = []",
    "    for py in sorted(cadw_root.rglob('*.py')):",
    "        rel = py.relative_to(cadw_root.parent).as_posix()",
    "        if rel in _ALLOWED_PANDAS_FILES:",
    "            continue",
    "        try:",
    "            tree = ast.parse(py.read_text())",
    "        except SyntaxError:",
    "            continue",
    "        if _uses_pandas_import(tree):",
    "            violations.append(rel)",
    "",
    "    assert not violations, (",
    '        "Unexpected `import pandas` in cadw/: "',
    '        + str(violations)',
        '        + ". v0.9.6e1: pandas is optional; use format=arrow/numpy/torch. "',
    '        + "Allow-list: " + str(sorted(_ALLOWED_PANDAS_FILES))',
    "    )",
    "",
    "",
    "def test_pandas_optional_in_pyproject():",
    '    """pyproject: pandas is in optional-dependencies.pandas +',
    '    dev group, NOT in project.dependencies."""',
    "    import tomllib",
    "    from pathlib import Path as _P",
    "    pyproject_path = _P(__file__).parent.parent / 'pyproject.toml'",
    "    data = tomllib.loads(pyproject_path.read_text())",
    "",
    "    core_deps = data['project'].get('dependencies', [])",
    "    for d in core_deps:",
    "        assert not d.startswith('pandas'), (",
    "            f'pandas must not be in [project.dependencies]; got {d}'",
    "        )",
    "",
    "    optional = data['project'].get('optional-dependencies', {})",
    "    assert 'pandas' in optional, (",
    "        'pandas missing from [project.optional-dependencies]'",
    "    )",
    "    assert any('pandas' in spec for spec in optional['pandas'])",
    "",
    "    dev_deps = data.get('dependency-groups', {}).get('dev', [])",
    "    assert any(d.startswith('pandas') for d in dev_deps), (",
    "        'pandas must be in [dependency-groups].dev for test helpers'",
    "    )",
]


def extend_test_no_pandas():
    print("\n=== test_no_pandas_in_ops.py: whole-cadw guard + pyproject ===")
    p = Path("tests/test_no_pandas_in_ops.py")
    s = p.read_text()
    if "def test_no_pandas_import_anywhere_in_cadw" in s:
        print("  already extended — skip")
        return
    s = s + "\n".join(NEW_TEST_LINES)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 6. Dogfood ============

def dogfood():
    print("\n=== dogfood: `import cadw` without pandas in sys.modules ===")
    # Fresh subprocess: pandas must NOT be loaded when only importing cadw.
    # (pandas is installed in .venv via dev-group, so we test *import graph*,
    # not absence of module.)
    script = (
        "import sys\n"
        "import cadw\n"
        "assert 'pandas' not in sys.modules, (\n"
        "    'pandas loaded by `import cadw` — should be lazy/optional'\n"
        ")\n"
        "print('import cadw: no pandas in sys.modules')\n"
    )
    r = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print("  STDOUT: " + r.stdout.strip()[-400:])
        print("  STDERR: " + r.stderr.strip()[-1200:])
        raise SystemExit("dogfood: pandas leaks into import cadw")
    print("  " + r.stdout.strip())

    # Belt-and-suspenders: no to_pandas / pivot Calls at top-level of
    # the four core modules (dataframe, groupby, dedup, pivot).
    print("\n=== dogfood: core modules are pandas-free (AST) ===")
    import inspect
    from cadw.dataframe import CadwDataFrame
    from cadw.ops import groupby as gb, dedup as dd, pivot as pv
    assert_no_call_in_source(
        inspect.getsource(CadwDataFrame.pivot_table),
        "to_pandas", fn_name="pivot_table",
    )
    assert_no_call_in_source(
        inspect.getsource(gb._estimate_k_sample),
        "to_pandas", fn_name="groupby._estimate_k_sample",
    )
    assert_no_call_in_source(
        inspect.getsource(dd._estimate_k_sample),
        "to_pandas", fn_name="dedup._estimate_k_sample",
    )
    assert_no_call_in_source(
        inspect.getsource(pv.pivot_to_disk),
        "to_pandas", "from_pandas", fn_name="pivot_to_disk",
    )
    print("  core ops: no to_pandas/from_pandas Call nodes")


# ============ main ============

def main():
    print("=== v0.9.6e1: pandas optional (+dev-only) ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    backup()
    edit_pyproject()
    regen_lock()
    update_convert_batch()
    extend_test_no_pandas()
    dogfood()

    print("\n=== pytest: no-pandas + pyproject guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** GUARDS FAILED ***")
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
        "v0.9.6e1: pandas -> optional (+dev-only). Milestone: "
        "`import cadw` does not pull pandas; core (ops, ML, pivot) "
        "is pandas-free. pyproject: dependencies -pandas; "
        "[project.optional-dependencies].pandas = ['pandas>=3.0'] "
        "(for to_batches(format='pandas')); "
        "[dependency-groups].dev += pandas (test reference helpers). "
        "uv lock regenerated (backup .pingis/pre-v096e1/; rollback "
        "on fail). _convert_batch(format='pandas'): catch "
        "ImportError -> friendly install-hint message. New AST guard "
        "test_no_pandas_import_anywhere_in_cadw (allow-list: "
        "viz/seaborn_adapter.py — seaborn API requires pandas.DataFrame, "
        "intentional bridge). New test_pandas_optional_in_pyproject "
        "(tomllib parse). Dogfood: fresh subprocess `import cadw` "
        "asserts 'pandas' not in sys.modules. v0.9.6 milestone.",
        self_path="scripts/patch_v096e1_pandas_optional.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6e1 applied. Milestone v0.9.6 reached.")


if __name__ == "__main__":
    main()