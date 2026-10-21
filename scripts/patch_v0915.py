#!/usr/bin/env python3
# v0.9.15 — CHANGELOG v0.9.14 section + __version__ + T2.6/T2.7.
#
# Scope:
#   (1) CHANGELOG.md: `## [0.9.14] — 2026-10-21` consolidated
#       (covers v0.9.7–v0.9.14 since v0.9.6 was last entry).
#   (2) cadw/__init__.py: `__version__ = "0.9.14"`.
#   (3) patch_utils.py: `dedent_method_source(fn)` helper
#       (§13.4/215/216 — existing parse_method_source does half).
#   (4) test_patch_utils.py: T2.6 tests (2).
#   (5) test_patch_utils.py: T2.7 meta-guard — flag inline
#       `_tw.dedent(_inspect.getsource(...))` + `_ast.parse(...)` in
#       cadw/ or scripts/ that bypasses the helper (§13.4/216).
#
# Deferred to v0.9.16:
#   - DECISIONS.md §13.4/211–223 append (big block).
#   - A1: dataframe.py split (5404 lines → 7 modules, ~4h).
#   - Release tag v0.9.14 (after CHANGELOG + __version__).
#
# Run: uv run python scripts/patch_v0915.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (1) CHANGELOG.md: 0.9.14 consolidated section ============

CHANGELOG_SECTION_LINES = [
    "## [0.9.14] — 2026-10-21",
    "",
    "### v0.9.7–v0.9.14: audit round 2 (16 patches, 3 silent data-loss bugs)",
    "",
    "**Milestone**: closes all 21 original findings from",
    "`_FOR_PINGIS/cadw_audit_report.md` (R1–R6, B1–B4, N1–N7)",
    "+ 13 findings discovered during implementation. Three",
    "silent data-loss bugs found and fixed — one 3 months in",
    "prod, undetected by tests:",
    "",
    "- **C1**: `_per_group_apply` mixed int/float groups silently",
    "  truncated fractional floats (`from_pylist(schema=int64)`).",
    "- **§13.4/218**: null-key rows dropped in `groupby` skew path",
    "  (`filter(null)` treats null as false — data loss in both",
    "  heavy and light paths). v0.3.4 → v0.9.12-fix2, ~3 months.",
    "- **§13.4/223**: `_infer_arrow_type_from_value` checked",
    "  `isinstance(v, int)` before `bool` (Python bool is int",
    "  subclass) → `True` inferred as int64 → `ArrowTypeError`.",
    "  F12-6a bool-widen was dead code.",
    "",
    "Plus 13 new §13.4 meta-lessons (§13.4/211–223) on patch-",
    "script safety (compile-before-write, partial-state abort),",
    "AST-check precision (negative→AST, receiver-type filter),",
    "and falsy-default traps (dedent-before-parse, fill_null).",
    "",
    "#### Added",
    "",
    "- **R1**: `MemoryGuard.check_partition_writers(P, bytes_per_writer)`",
    "  — real per-writer footprint (audit bench: 1.2–4.4 KB/writer).",
    "  Prior `check_partition_buffers` was vacuous by construction",
    "  (`P × (limit // P) ≤ limit`).",
    "- **R2**: `MemoryGuard.reestimate_from_batch(batch)` +",
    "  `get_batch_rows(..., sample_batch=)` — schema underestimate",
    "  fixed (10× for string columns: 500 B/row est vs 2040 actual).",
    "- **B1**: `propagate_chars` for `hash_shuffle`/`dedup`/`rank`/",
    "  `quantile`/`resample` — audit `plan_pipeline([filter,",
    "  hash_shuffle, groupby])` crash closed.",
    "- **R4**: `TopKOp` (`topk_to_disk`) streaming — `nlargest`/",
    "  `nsmallest` **~3946×** faster (audit bench N=2M, k=10).",
    "- **R5**: `DescribeOp` (`describe_columns`) fused — `describe`",
    "  **~109×** faster (audit bench N=2M, 2 cols).",
    "- **R6**: `pivot_to_disk` streaming Arrow-native — replaces",
    "  `to_pandas().pivot()` (26.8 MB → ~10 MB for 200k/300×300;",
    "  100k×20 passes where dense failed).",
    "- **F1**: `patch_utils.run_verify_in_subprocess(fn)` — fresh",
    "  subprocess dogfood; 8× §13.4/178 stale-module-cache fixed.",
    "- **T2.1**: `patch_utils.apply_or_fail` + `verify_anchors` —",
    "  fail-fast with context, compile-before-write (§13.4/211).",
    "- **T2.2**: `assert_no_call_in_source(receiver_type=)` — precise",
    "  AST check (§13.4/217): `X[i].as_py()` per-row flagged,",
    "  `fn(...).as_py()` aggregate allowed.",
    "- **T2.6**: `patch_utils.dedent_method_source(fn)` — dedent +",
    "  parse helper (§13.4/215/216).",
    "- **T2.9**: `apply_or_fail(validate=)` — format-aware validation",
    "  (`toml`/`json`/`python`/`none`); extension-inferred default",
    "  (§13.4/221).",
    "- **C1**: `_maybe_upgrade_schema` — int→float64 widen in",
    "  `_per_group_apply` mixed groups.",
    "- **F12-6a/b**: bool→float64 widen + clear `RuntimeError` on",
    "  non-widenable mismatch (was cryptic `ArrowInvalid`).",
    "- **F12-6c**: `_infer_arrow_type_from_value` bool-before-int",
    "  reorder (§13.4/223).",
    "- **F12-5**: `_stream_values_at_positions` Arrow-native —",
    "  quantile ~100–400× faster for high quantiles.",
    "- **F12-4**: `_detect_heavy_keys` Arrow-native — groupby skew",
    "  detection ~10× faster, single C++ `group_by` pass.",
    "- **F12-1**: `topk_to_disk` incremental `acc_bytes` —",
    "  O(P²) → O(P).",
    "",
    "#### Fixed",
    "",
    "- **C1: silent int truncation** in `_per_group_apply` when first",
    "  group returns int and later returns fractional float —",
    "  previously truncated (silent data loss) or `ArrowInvalid`.",
    "- **§13.4/218: silent null-key data loss** in `groupby` skew",
    "  path: `_compute_heavy_mask` produced null mask on null-key",
    "  rows (`pc.equal(null, non_null)` = null via Kleene);",
    "  `filter(null)` drops → rows lost from both heavy and light.",
    "  Fix: `pc.fill_null(mask, pa.scalar(False))`.",
    "- **§13.4/223: bool→int silent misinference** in",
    "  `_infer_arrow_type_from_value` (order: `isinstance(v, int)`",
    "  before `bool`). `True` inferred as int64; `from_pylist(",
    "  schema=int64)` → `ArrowTypeError`. F12-6a bool-widen was",
    "  dead code. Fix: bool before int.",
    "- **PF4**: `_maybe_upgrade_for_row` per-row (v0.9.10) → per-",
    "  flush `_maybe_upgrade_schema` (v0.9.11) — for 10M groups ×",
    "  K=10, ~100M list allocs + ~1B attr checks eliminated.",
    "- **N1–N5**: `ColumnTransformer.to_dict` (`__serialization_name__`),",
    "  `dt.floor/ceil/round` (`TypeError` in init for date32/64),",
    "  groupby combiner `check_state`, `_merge_fanout` duplicate",
    "  column (`RuntimeError`), `PCA` docstring, `_GroupBy.apply`",
    "  docstring.",
    "- **B2–B4**: `pca.py` F821 (`TYPE_CHECKING`), `pipeline.py`",
    "  `-> Path` module-level import, `PCA` fitted-model docstring.",
    "- **U1, U2, U3**: `test_dt_ceil_date64` test data (pa.date32",
    "  → pa.date64 via from_table), `_concat_batches` per-column",
    "  `pa.concat_arrays` (1× alloc vs 3×), `pyproject.toml`",
    "  version 0.1.0 → 0.9.14.",
    "- **F12-8**: `PredicateFilter.mask_for_batch` null semantics",
    "  documented (Kleene-drop numeric vs fill_null string —",
    "  intentional asymmetry).",
    "- **§13.4/220**: shadowed module import (`from cadw.ops import",
    "  quantile` gets function, not module) — tests and dogfoods",
    "  use direct submodule import; meta-guard added.",
    "",
    "#### Changed",
    "",
    "- `pyproject.toml`: version 0.1.0 → 0.9.14 (was stale since",
    "  initial scaffold, drifted 3+ months from CHANGELOG).",
    "- `cadw.__version__` = `\"0.9.14\"` (was undefined; `pip show`",
    "  and `import cadw` now agree).",
    "",
    "#### Removed",
    "",
    "- (none)",
    "",
    "#### Deferred to v0.9.16",
    "",
    "- **A1**: `dataframe.py` split (5404 lines → 7 modules) —",
    "  architectural, ~4h, risky. Do after v0.9.14 tag.",
    "- **§11.5 backlog**: N6 (`Dataset.iter_batches` guard — user",
    "  responsibility), R3 (planner spill stub, v0.2).",
    "- `DECISIONS.md §13.4/211–223` — new meta-lessons doc block.",
    "",
    "",
]


def changelog_v0914():
    print("\n=== (1) CHANGELOG.md: 0.9.14 consolidated section ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    # Idempotency: already present?
    if "## [0.9.14] — 2026-10-21" in s:
        print("  already present — skip")
        return
    anchor = "## [Unreleased]\n\n## [0.9.6] — 2026-10-21"
    if anchor not in s:
        raise SystemExit("CHANGELOG anchor '## [Unreleased]\\n\\n## [0.9.6]' not found")
    section = "\n".join(CHANGELOG_SECTION_LINES)
    new_block = f"## [Unreleased]\n\n{section}## [0.9.6] — 2026-10-21"
    s = s.replace(anchor, new_block, 1)
    p.write_text(s, newline="")
    # Sanity: both headers present in order
    assert "## [Unreleased]" in s
    assert "## [0.9.14] — 2026-10-21" in s
    assert "## [0.9.6] — 2026-10-21" in s
    assert s.index("## [Unreleased]") < s.index("## [0.9.14]")
    assert s.index("## [0.9.14]") < s.index("## [0.9.6]")
    print("  OK (0.9.14 section inserted between Unreleased and 0.9.6)")


# ============ (2) cadw/__init__.py: __version__ ============

def version_init():
    print("\n=== (2) cadw/__init__.py: __version__ = 0.9.14 ===")
    pu.apply_or_fail(
        "cadw/__init__.py",
        (
            '"""CADW: out-of-core dataframe. См. DECISIONS.md."""\n'
            "from .core.memory_guard import MemoryGuard, MemoryLimitExceeded\n"
        ),
        (
            '"""CADW: out-of-core dataframe. См. DECISIONS.md."""\n'
            '__version__ = "0.9.14"\n'
            "\n"
            "from .core.memory_guard import MemoryGuard, MemoryLimitExceeded\n"
        ),
        label="__version__",
    )
    print("  OK")


# ============ (3) T2.6: dedent_method_source helper ============

def t2_6_helper():
    print("\n=== (3) T2.6: dedent_method_source helper ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def parse_method_source(fn):\n"
            "    \"\"\"Return ast.Module for inspect.getsource(fn) (§13.4/191).\n"
            "\n"
            "    inspect.getsource(Class.method) includes the method's\n"
            "    class-level indent (4 spaces before `def`); ast.parse requires\n"
            "    zero-indent top-level. Wraps textwrap.dedent.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import inspect as _inspect\n"
            "    import textwrap as _tw\n"
            "    return _ast.parse(_tw.dedent(_inspect.getsource(fn)))\n"
        ),
        (
            "def parse_method_source(fn):\n"
            "    \"\"\"Return ast.Module for inspect.getsource(fn) (§13.4/191).\n"
            "\n"
            "    inspect.getsource(Class.method) includes the method's\n"
            "    class-level indent (4 spaces before `def`); ast.parse requires\n"
            "    zero-indent top-level. Wraps textwrap.dedent.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import inspect as _inspect\n"
            "    import textwrap as _tw\n"
            "    return _ast.parse(_tw.dedent(_inspect.getsource(fn)))\n"
            "\n"
            "\n"
            "def dedent_method_source(fn):\n"
            "    \"\"\"Return (dedented_source, ast.Module) for fn (§13.4/215/216).\n"
            "\n"
            "    inspect.getsource(fn) on a class method / nested def returns\n"
            "    source WITH class-level indent. Both `ast.parse(src)` and\n"
            "    substring checks (e.g. `'foo' in src`) need the dedented\n"
            "    form. §13.4/215 (dedent before parse), §13.4/216 (existing\n"
            "    helper exists, use it) — v0.9.11-fix1 dogfood reinvented\n"
            "    this inline and got it wrong (ast.parse on indented src →\n"
            "    IndentationError). Use this helper from now on.\n"
            "\n"
            "    Returns (dedented_src, parsed_tree). Either may be\n"
            "    discarded if caller needs only one.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import inspect as _inspect\n"
            "    import textwrap as _tw\n"
            "    src = _tw.dedent(_inspect.getsource(fn))\n"
            "    tree = _ast.parse(src)\n"
            "    return src, tree\n"
        ),
        label="T2.6 helper",
    )
    print("  OK")


# ============ (4) T2.6 tests + (5) T2.7 meta-guard ============

NEW_TESTS = [
    "",
    "",
    "# ============ v0.9.15 (T2.6/T2.7): dedent_method_source + meta-guard ============",
    "",
    "",
    "def test_dedent_method_source_basic():",
    "    \"\"\"T2.6: dedent_method_source returns (str, ast.Module).\"\"\"",
    "    import ast as _ast",
    "    from cadw.patch_utils import dedent_method_source",
    "    from cadw.dataframe import CadwDataFrame",
    "    src, tree = dedent_method_source(CadwDataFrame.head)",
    "    assert isinstance(src, str)",
    "    assert isinstance(tree, _ast.Module)",
    "    # Dedented: no leading whitespace before `def`",
    "    first_line = src.split(chr(10))[0]",
    "    assert first_line.startswith('def '), (",
    "        f'expected `def ...` first line, got {first_line!r}'",
    "    )",
    "",
    "",
    "def test_dedent_method_source_ast_parses():",
    "    \"\"\"T2.6: returned tree is real Module (not exception).\"\"\"",
    "    import ast as _ast",
    "    from cadw.patch_utils import dedent_method_source",
    "    from cadw.dataframe import CadwDataFrame",
    "    src, tree = dedent_method_source(CadwDataFrame._per_group_apply)",
    "    fdefs = [n for n in _ast.walk(tree)",
    "             if isinstance(n, _ast.FunctionDef)]",
    "    assert any(f.name == '_per_group_apply' for f in fdefs)",
    "",
    "",
    "def test_no_inline_dedent_getsource_outside_helper():",
    "    \"\"\"T2.7 (§13.4/216): dedent+getsource on class methods must go",
    "    through dedent_method_source helper. Catches future",
    "    'reinvented inline' bugs (v0.9.11-fix1 got it wrong).",
    "",
    "    Skips patch_utils.py itself (helper) and any file with",
    "    docstring-only references. Uses AST — no substring false",
    "    positives (§13.4/213).",
    "    \"\"\"",
    "    import ast as _ast",
    "    from pathlib import Path as _P",
    "    roots = [_P('cadw'), _P('scripts')]",
    "    violations: list[str] = []",
    "    for root in roots:",
    "        if not root.exists():",
    "            continue",
    "        for py in root.rglob('*.py'):",
    "            if py.name == 'patch_utils.py':",
    "                continue  # helper module",
    "            try:",
    "                src = py.read_text()",
    "                tree = _ast.parse(src)",
    "            except (SyntaxError, OSError, UnicodeDecodeError):",
    "                continue",
    "            # Find `_tw.dedent(_inspect.getsource(...))` calls",
    "            for n in _ast.walk(tree):",
    "                if not isinstance(n, _ast.Call):",
    "                    continue",
    "                f = n.func",
    "                if not (isinstance(f, _ast.Attribute)",
    "                        and f.attr == 'dedent'):",
    "                    continue",
    "                if not n.args:",
    "                    continue",
    "                inner = n.args[0]",
    "                if not (isinstance(inner, _ast.Call)",
    "                        and isinstance(inner.func, _ast.Attribute)",
    "                        and inner.func.attr == 'getsource'):",
    "                    continue",
    "                # Exception: in patch_utils itself already filtered",
    "                violations.append(f'{py}:{n.lineno}')",
    "    assert not violations, (",
    "        '§13.4/216: inline dedent(getsource(...)) outside '",
    "        'patch_utils.dedent_method_source:'",
    "        + chr(10) + chr(10).join(violations)",
    "    )",
]


def add_tests():
    print("\n=== (4)/(5) T2.6 + T2.7 tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_dedent_method_source_basic" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.15 + regression ===")
    pu.verify_anchors([
        # v0.9.15
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("CHANGELOG.md", "### v0.9.7–v0.9.14: audit round 2"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("cadw/patch_utils.py", "def dedent_method_source("),
        ("cadw/patch_utils.py", "v0.9.15"),
        ("tests/test_patch_utils.py", "test_dedent_method_source_basic"),
        ("tests/test_patch_utils.py", "test_no_inline_dedent_getsource_outside_helper"),
        # v0.9.14-fix3 regression
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("tests/test_apply.py", "test_apply_all_bool_stays_bool"),
        ("tests/test_apply.py", "test_infer_arrow_type_from_value_unit"),
        # v0.9.14 (F12-6a/b, F12-7, T2.2, T2.9, U3)
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/dataframe.py", "§11.5, v0.9.14 F12-7"),
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("cadw/patch_utils.py", "v0.9.14-fix1 (T2.9, §13.4/221)"),
        ("pyproject.toml", 'version = "0.9.14"'),
        # v0.9.13/12/11/10/9 regression
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.15 via subprocess ===")

    def _verify_all():
        import ast as _ast
        from cadw.patch_utils import dedent_method_source
        from cadw.dataframe import CadwDataFrame

        # v0.9.15: __version__
        import cadw
        assert cadw.__version__ == "0.9.14", (
            f"cadw.__version__ = {cadw.__version__!r}, expected '0.9.14'"
        )

        # CHANGELOG section
        from pathlib import Path as _P
        cl = _P("CHANGELOG.md").read_text()
        assert "## [0.9.14] — 2026-10-21" in cl
        assert "### v0.9.7–v0.9.14: audit round 2" in cl
        assert cl.index("## [Unreleased]") < cl.index("## [0.9.14]")
        assert cl.index("## [0.9.14]") < cl.index("## [0.9.6]")

        # T2.6: helper works
        src, tree = dedent_method_source(CadwDataFrame.head)
        assert isinstance(src, str) and isinstance(tree, _ast.Module)
        assert src.split("\n")[0].startswith("def ")
        fdefs = [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.FunctionDef) and n.name == "head"]
        assert fdefs, "head not found in parsed tree"

        # T2.7 meta-guard: run over cadw/ + scripts/
        violations = []
        for root in (_P("cadw"), _P("scripts")):
            if not root.exists():
                continue
            for py in root.rglob("*.py"):
                if py.name == "patch_utils.py":
                    continue
                try:
                    src_str = py.read_text()
                    t = _ast.parse(src_str)
                except (SyntaxError, OSError, UnicodeDecodeError):
                    continue
                for n in _ast.walk(t):
                    if not isinstance(n, _ast.Call):
                        continue
                    f = n.func
                    if not (isinstance(f, _ast.Attribute) and f.attr == "dedent"):
                        continue
                    if not n.args:
                        continue
                    inner = n.args[0]
                    if (isinstance(inner, _ast.Call)
                            and isinstance(inner.func, _ast.Attribute)
                            and inner.func.attr == "getsource"):
                        violations.append(f"{py}:{n.lineno}")
        assert not violations, (
            f"§13.4/216: inline dedent(getsource(...)) at: {violations}"
        )

        print("  __version__ = '0.9.14' OK")
        print("  CHANGELOG: 0.9.14 section between Unreleased and 0.9.6 OK")
        print("  T2.6 dedent_method_source (str, tree) OK")
        print("  T2.7 meta-guard: no inline dedent(getsource) violations OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-q"],
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
    print("=== v0.9.15 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    changelog_v0914()
    version_init()
    t2_6_helper()
    add_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.15: CHANGELOG v0.9.14 section + __version__ + "
        "T2.6/T2.7. (1) CHANGELOG.md: consolidated `## [0.9.14] — "
        "2026-10-21` section covering v0.9.7–v0.9.14 (16 patches) — "
        "v0.9.7–v0.9.14 were never documented; last CHANGELOG entry "
        "was v0.9.6. Milestone: closes all 21 original audit "
        "findings + 13 more discovered, 3 silent data-loss bugs "
        "(C1 int truncation, §13.4/218 null-key drop 3 months in "
        "prod, §13.4/223 bool→int misinference), 13 new §13.4 "
        "meta-lessons. (2) cadw/__init__.py: `__version__ = \"0.9.14\"` "
        "— was undefined; `pip show` gave 0.9.14 but `cadw.__version__` "
        "→ AttributeError. (3) T2.6: `dedent_method_source(fn)` "
        "helper — returns (dedented_src, ast.Module) for class "
        "methods / nested defs (§13.4/215/216). inspect.getsource "
        "on class method returns indented source; both ast.parse "
        "and substring checks need dedented form. v0.9.11-fix1 "
        "dogfood reinvented this inline and got it wrong. (4) T2.7: "
        "meta-guard test_no_inline_dedent_getsource_outside_helper "
        "— AST walk over cadw/ and scripts/; fails if "
        "`_tw.dedent(_inspect.getsource(...))` appears outside "
        "patch_utils.dedent_method_source (§13.4/216: helper "
        "exists, use it). Deferred: DECISIONS.md §13.4/211–223 "
        "append, A1 dataframe.py split (5404 lines), release tag "
        "v0.9.14 (after v0.9.15). Fast ~1869, slow 22.",
        self_path="scripts/patch_v0915.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.15 applied.")


if __name__ == "__main__":
    main()
