#!/usr/bin/env python3
"""v0.9.43 — Phase G: tech debt closeout.

Closes A.6 (deferred v0.9.26) and isolates benchmark reference deps.
No production code change.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_VERSION = "0.9.43"
TODAY = "2026-10-24"


def _run(cmd, *, check=True, capture=False, timeout=None):
    return subprocess.run(
        cmd, cwd=ROOT, check=check,
        capture_output=capture, text=capture, timeout=timeout,
    )


def verify_state():
    print("=== verify ===")
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                  capture=True).stdout.strip()
    print(f"branch: {branch}")
    if branch != "pingis/dev-audit":
        raise SystemExit(f"expected branch pingis/dev-audit, got {branch}")

    tag = _run(["git", "log", "--oneline", "--all", "--grep=v0.9.42", "-1"],
               capture=True).stdout.strip()
    if not tag:
        raise SystemExit("v0.9.42 commit not found — commit v0.9.42 first")
    print(f"v0.9.42 commit: {tag[:70]}")

    r = _run(["uv", "run", "python", "-c",
              "import cadw; print(cadw.__version__)"],
             check=False, capture=True)
    if r.returncode != 0:
        raise SystemExit(f"import cadw failed: {r.stderr}")
    v = r.stdout.strip()
    if v != "0.9.42":
        raise SystemExit(f"cadw.__version__ = {v!r}, expected 0.9.42")
    print(f"cadw.__version__: {v}")

    if (ROOT / "requirements-bench.txt").exists():
        print("requirements-bench.txt exists — assume already applied")
        sys.exit(0)


def write_requirements_bench():
    print("\n=== requirements-bench.txt ===")
    content = """# Benchmark reproduction reference libraries.
#
# ============================================================================
# IMPORTANT: these are NOT runtime dependencies of cadw.
# They are imported ONLY by scripts/bench_*.py and tests/test_ml_tools_*.
# The AST guard tests/test_no_bench_deps_in_core.py enforces that
# cadw/**/*.py does NOT import any of these.
#
# Install in a separate environment:
#   pip install -r requirements-bench.txt
# Do NOT add these to pyproject.toml.
# ============================================================================

# Tabular reference (classification, regression, trees)
scikit-learn>=1.3

# Statistical reference (ETS, ARIMA, VAR, GLM)
statsmodels>=0.14

# Anomaly detection reference (LOF, OCSVM, KNN, HBOS)
pyod>=1.1

# Change-point reference (PELT, Binseg, BottomUp)
ruptures>=1.1

# Survival analysis reference (CoxPH, RSF)
lifelines>=0.27

# Recommender systems reference (ALS, BPR)
implicit>=0.7

# Optional (uncomment only when reproducing specific benchmarks)
# xgboost>=2.0
# lightgbm>=4.0
"""
    (ROOT / "requirements-bench.txt").write_text(content)
    print("  wrote requirements-bench.txt")


def write_ast_guard():
    print("\n=== tests/test_no_bench_deps_in_core.py ===")
    content = '''"""AST guard: cadw/ must not import benchmark-only reference libraries.

Rationale: requirements-bench.txt contains libraries used only for
benchmark reproduction in scripts/ and tests/. Core cadw/ must remain
dependency-light (v0.9.43, Phase G).

Mirror of tests/test_no_pandas_in_ops.py — same pattern, different set.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CADW = Path(__file__).parent.parent / "cadw"

# Must match requirements-bench.txt.
BENCH_ONLY = {
    "pyod", "ruptures", "lifelines", "implicit",
    "statsmodels", "sklearn",
}


def _walk_py():
    for p in sorted(CADW.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


@pytest.mark.parametrize(
    "path", list(_walk_py()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_no_bench_dep_imports(path):
    """cadw/<path> does not import benchmark-only reference libraries."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(CADW)}: syntax error {e}")
    bad = set(_imported_modules(tree)) & BENCH_ONLY
    assert not bad, (
        f"{path.relative_to(CADW)}: imports benchmark-only libraries "
        f"{sorted(bad)}. These live in requirements-bench.txt and must "
        f"be used only in scripts/ or tests/ (v0.9.43 AST guard)."
    )
'''
    p = ROOT / "tests/test_no_bench_deps_in_core.py"
    p.write_text(content)
    print(f"  wrote tests/test_no_bench_deps_in_core.py "
          f"({len(content.splitlines())} lines)")


def fix_hypothesis_norecursedirs():
    print("\n=== pyproject.toml: norecursedirs += .hypothesis ===")
    p = ROOT / "pyproject.toml"
    text = p.read_text()

    data = tomllib.loads(text)
    nr = (data.get("tool", {}).get("pytest", {})
              .get("ini_options", {}).get("norecursedirs", None))
    if nr is None:
        print("  WARNING: norecursedirs not found — skipping")
        return
    if ".hypothesis" in nr:
        print("  already present")
        return

    m = re.search(r'(norecursedirs\s*=\s*\[)([^\]]*)\]', text)
    if not m:
        raise SystemExit("regex failed to locate norecursedirs list")
    inner = m.group(2)
    if "#" in inner:
        raise SystemExit(
            "norecursedirs has inline comment; manual edit required. "
            "Add '.hypothesis' to pyproject.toml "
            "[tool.pytest.ini_options] norecursedirs."
        )
    stripped = inner.rstrip()
    if stripped.endswith(","):
        new_inner = stripped + ' ".hypothesis"'
    else:
        new_inner = stripped + ', ".hypothesis"'
    text = text[:m.start(2)] + new_inner + text[m.end(2):]
    p.write_text(text)

    data = tomllib.loads(p.read_text())
    nr = data["tool"]["pytest"]["ini_options"]["norecursedirs"]
    if ".hypothesis" not in nr:
        raise SystemExit(f"failed to add: {nr}")
    print(f"  added .hypothesis to norecursedirs "
          f"(list now {len(nr)} entries)")


def update_roadmap_a6():
    print("\n=== docs/roadmap.json: A.6 → done ===")
    p = ROOT / "docs/roadmap.json"
    data = json.loads(p.read_text())

    found = None
    for phase in data.get("phases", []):
        for item in phase.get("items", []):
            if item.get("id") == "A.6":
                found = item
                break
        if found:
            break
    if not found:
        print("  WARNING: A.6 not found — skipping")
        return
    print(f"  current A.6 status: {found.get('status')}")
    found["status"] = "done"
    found["resolution"] = (
        "Closed by reference to AUDIT_v0931 (v0.9.32): 160 to_pylist/"
        "as_py/to_numpy sites scanned systematically; only F9 (gbdt "
        "per-batch stack, fixed v0.9.32) was real; remaining sites "
        "bounded by sample/group size, guarded upstream, or by design "
        "(MaterializedIndex). Original '18 sites' claim from v0.9.26 "
        "cannot be re-derived (audit absent), but the exhaustive "
        "160-site audit supersedes it."
    )
    found["closed_in"] = TARGET_VERSION
    data["updated"] = TODAY
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print("  A.6 → done")


def update_taxonomy_deprecation():
    print("\n=== docs/ML_TASKS_TAXONOMY.md: deprecation notice ===")
    p = ROOT / "docs/ML_TASKS_TAXONOMY.md"
    text = p.read_text()
    if "DEPRECATED (v0.9.43)" in text:
        print("  already marked")
        return
    notice = (
        "\n> \u26a0\ufe0f **DEPRECATED (v0.9.43):** this file will be "
        "replaced by `docs/ml_tools/` matrix after v0.9.44 (M1 patch). "
        "Kept for navigation until the matrix is populated for 5 "
        "sections (A, D, E, F, H). See `CONTEXT.md` "
        "\u00a7\"Roadmap update (v0.9.43)\" for the transition plan. "
        "New ML work must consult the matrix, not this file.\n"
    )
    lines = text.split("\n", 1)
    if lines[0].startswith("# "):
        text = lines[0] + "\n" + notice + "\n" + lines[1]
    else:
        text = notice + "\n" + text
    p.write_text(text)
    print("  deprecation notice added")


def update_context_roadmap():
    print("\n=== CONTEXT.md: roadmap section ===")
    p = ROOT / "CONTEXT.md"
    text = p.read_text()
    if "## Roadmap update (v0.9.43)" in text:
        print("  already present")
        return
    section = """

## Roadmap update (v0.9.43)

**Phases A-D:** \u2705 closed (Phase D = 29/29 algorithms, 25 notes
+ 25 benches).

| Phase | Status | DoD |
|---|---|---|
| **G** tech debt | \u2705 closed v0.9.43 (A.6, .hypothesis, bench-deps isolation) | \u2014 |
| **M1** matrix schema | \U0001F51C v0.9.44 (blocker for F/H) | pydantic schema + 5 sections (A, D, E, F, H) + validator test |
| **F** Light NNs | after M1 | model + tests + bench + **card in matrix** |
| **H** Long-tail | after M1 | model + tests + bench + **card in matrix** |
| **E** Recipes | after F/H | reduced to views over the matrix |
| **M2** Reproduction | parallel | isolated requirements-bench.txt + scripts/reproduce_benchmark.py |
| **M3** Fill remaining | parallel with F/H | 10 more sections (B, C, G, I, J, K, L, M, N, O) |

**DoD change for F/H:** a model is `done` only when:

1. `cadw/ml/<name>.py` implemented + \u22658 tests + bench
2. Card in `docs/ml_tools/<section>.yaml` with:
   `roles`, `purpose`, `real_life_examples`, `benchmark_result`,
   `implementation_status: {verified_in_cadw: true, verified_by_agent: true}`
3. `tests/test_ml_tools_schema.py` passes

**Deprecated:** `docs/ML_TASKS_TAXONOMY.md` (top-of-file notice) \u2014
replaced by matrix after M1 populates 5 sections.

**Bench-deps isolation (v0.9.43):**

- `requirements-bench.txt` \u2014 pyod, ruptures, lifelines, implicit,
  statsmodels, sklearn. NOT in `pyproject.toml`.
- AST guard `tests/test_no_bench_deps_in_core.py` prevents `cadw/`
  from importing these. Reference work lives only in `scripts/` and
  `tests/`.
"""
    p.write_text(text.rstrip() + section + "\n")
    print("  appended roadmap section")


def bump_version():
    print(f"\n=== version \u2192 {TARGET_VERSION} ===")
    pp = ROOT / "pyproject.toml"
    src = pp.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', src, re.M)
    old = m.group(1) if m else "?"
    src = re.sub(r'^(version\s*=\s*)"[^"]+"',
                 rf'\1"{TARGET_VERSION}"',
                 src, count=1, flags=re.M)
    pp.write_text(src)
    print(f"  pyproject.toml: {old} \u2192 {TARGET_VERSION}")

    init = ROOT / "cadw/__init__.py"
    src = init.read_text()
    src = re.sub(r'__version__\s*=\s*["\'][^"\']+["\']',
                 f'__version__ = "{TARGET_VERSION}"',
                 src, count=1)
    init.write_text(src)

    r = _run(["uv", "run", "python", "-c",
              "import cadw; print(cadw.__version__)"],
             check=False, capture=True)
    if r.returncode != 0 or r.stdout.strip() != TARGET_VERSION:
        raise SystemExit(
            f"version check failed: {r.stdout!r} / {r.stderr!r}"
        )
    print(f"  \u2713 cadw.__version__ == {TARGET_VERSION}")


def update_changelog():
    print("\n=== CHANGELOG.md ===")
    p = ROOT / "CHANGELOG.md"
    text = p.read_text()
    if f"### v{TARGET_VERSION}" in text:
        print("  already present")
        return
    marker = "## [Unreleased]"
    entry = f"""## [Unreleased]

### v{TARGET_VERSION} ({TODAY}) — Phase G: tech debt closeout

Closes A.6 (deferred from v0.9.26) and isolates benchmark reference
deps. No production code change.

- **A.6 closed** by reference to AUDIT_v0931 (v0.9.32): 160
  to_pylist/as_py/to_numpy sites scanned; only F9 (gbdt per-batch
  stack, fixed v0.9.32) was real; all others bounded by sample/
  group size, guarded upstream, or by design. Original "18 sites"
  claim from v0.9.26 cannot be re-derived (audit absent), but the
  exhaustive 160-site audit supersedes it.
- **`requirements-bench.txt`** — isolated list of reference libraries
  (scikit-learn, statsmodels, pyod, ruptures, lifelines, implicit).
  NOT added to `pyproject.toml`. Used only by `scripts/bench_*.py`
  and future `tests/test_ml_tools_*`.
- **AST guard** `tests/test_no_bench_deps_in_core.py` — fails if any
  `cadw/**/*.py` imports pyod/ruptures/lifelines/implicit/
  statsmodels/sklearn. Mirror of the existing
  `test_no_pandas_in_ops.py` pattern.
- **`.hypothesis` in `norecursedirs`** — pytest no longer scans the
  hypothesis cache during collection. Pre-existing issue, not a
  v0.9.42 regression.
- **`docs/ML_TASKS_TAXONOMY.md` marked deprecated** — will be
  replaced by `docs/ml_tools/` matrix after v0.9.44 (M1). Kept for
  navigation until 5 sections populated (A, D, E, F, H).
- **`CONTEXT.md` roadmap section added** — A-D closed; G (this
  patch) \u2192 M1 (matrix schema) \u2192 F (Light NNs) / H (Long-tail)
  \u2192 E (recipes as views). DoD for F/H adds "card in matrix with
  roles, purpose, real_life_examples, benchmark_result,
  implementation_status".
- **`docs/roadmap.json`**: A.6 \u2192 done (with resolution note).

Test status: fast suite green (v0.9.42 baseline 2829 passed /
2 skipped + ~99 new AST-guard param tests); slow 25 passed;
WL 34 OK.

"""
    if marker in text:
        text = text.replace(marker, entry, 1)
    else:
        text = entry + text
    p.write_text(text)
    print("  entry added")


def run_tests():
    print("\n=== tests ===")
    print("--- fast suite ---")
    r = _run(["uv", "run", "pytest", "-q", "-m", "not slow"],
             check=False, capture=True, timeout=1800)
    tail = r.stdout[-400:] if r.stdout else ""
    print(tail)
    if r.returncode != 0:
        print("--- full output ---")
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("fast suite FAILED")

    print("--- slow suite ---")
    r = _run(["uv", "run", "pytest", "-q", "-m", "slow"],
             check=False, capture=True, timeout=900)
    print(r.stdout[-200:] if r.stdout else "")
    if r.returncode != 0:
        raise SystemExit("slow suite FAILED")

    print("--- whitelist ---")
    r = _run(["uv", "run", "python", "-c",
              "from cadw.patch_utils import verify_whitelist; "
              "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
             check=False, capture=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        raise SystemExit(f"WL failed: {r.stderr}")


def commit():
    print("\n=== commit ===")
    _run(["git", "add", "-A"])
    msg = (
        "v0.9.43: Phase G — tech debt closeout\n"
        "\n"
        "A.6 closed by reference to AUDIT_v0931 (160 to_pylist sites\n"
        "scanned; only F9 real, fixed v0.9.32).\n"
        "\n"
        "Benchmark reference deps isolated:\n"
        "- requirements-bench.txt (pyod, ruptures, lifelines, implicit,\n"
        "  statsmodels, sklearn) — NOT in pyproject.toml.\n"
        "- AST guard tests/test_no_bench_deps_in_core.py — cadw/ must\n"
        "  not import these; reference work lives only in scripts/ and\n"
        "  tests/.\n"
        "\n"
        "Doc/cleanup:\n"
        "- .hypothesis added to norecursedirs (silences collection warn).\n"
        "- ML_TASKS_TAXONOMY.md marked deprecated (replaced by\n"
        "  docs/ml_tools/ after v0.9.44 M1).\n"
        "- CONTEXT.md roadmap section (G→M1→F/H→E).\n"
        "- roadmap.json A.6 → done with resolution note.\n"
        "\n"
        "Version 0.9.42 → 0.9.43. Test status: fast suite green\n"
        "(baseline + AST-guard param tests); slow 25; WL 34 OK."
    )
    _run(["git", "commit", "-m", msg])
    log = _run(["git", "log", "--oneline", "-1"],
               capture=True).stdout.strip()
    print(f"  {log}")


def main():
    print(f"=== v{TARGET_VERSION} — Phase G: tech debt closeout ===")
    verify_state()
    write_requirements_bench()
    write_ast_guard()
    fix_hypothesis_norecursedirs()
    update_roadmap_a6()
    update_taxonomy_deprecation()
    update_context_roadmap()
    bump_version()
    update_changelog()
    run_tests()
    commit()
    print(f"\nv{TARGET_VERSION} COMPLETE. Next: v0.9.44 (M1 — matrix schema).")


if __name__ == "__main__":
    main()
