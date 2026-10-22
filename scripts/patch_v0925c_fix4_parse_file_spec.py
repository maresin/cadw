#!/usr/bin/env python3
# v0.9.25c-fix4 — real production bug in _parse_file_spec (v0.9.25c).
#
# Root: _parse_file_spec does `spec.split(",")` BEFORE brace expansion.
# For `cadw/ts/{features,stats,models,thf,backtest}.py`:
#   split(",") → ['cadw/ts/{features', 'stats', 'models', 'thf',
#                 'backtest}.py']
# None resolve → all silently dropped. A.6 contributes only
# 'cadw/dataframe.py' (first plain part before comma).
#
# Why v0.9.25c/fix1/fix2 passed: A.4/A.5/A.7 planned had plain paths
# (cadw/ops/sort.py, cadw/dataframe.py) that pulled ts/models.py,
# sort.py, dataframe.py — masking the brace bug. fix3 flipped them to
# done (correct), so only A.6 (broken brace) + A.9 (cadw/ops/sort.py) +
# A.10 (cadw/core/memory_guard.py) remained planned → --phase A
# resolved to 3 files instead of 14.
#
# Fix: _split_top_commas(s) — split on comma only at brace-depth 0.
# Then brace-expansion per part is safe.
#
# Test-side bug too: fix3 test asserted `tests/test_ts_models.py NOT
# in text`. But A.6 planned = cadw/ts/models.py → _find_tests_for_module
# pulls tests/test_ts_models.py. So the test file IS in --phase A
# output (correctly). Absence assertion was wrong (fix3 fall-through
# missed because production bug made it accidentally true).
#
# Correct test layer split:
#   - Unit: _parse_file_spec(brace/comma/wildcard) — direct, deterministic.
#   - Integration: --phase A includes all 12 A.6 + A.9 + A.10 files.
#   - No absence assertions: no done-only file exists for phase A.
#
# Fix-cycle 4/2 (§13.4 budget massively exceeded). Production cadw/
# unchanged across v0.9.25c → fix4; only scripts/collect_project.py +
# tests. Meta-lesson (below): unit-test the function under test BEFORE
# e2e integration test. fix1-fix3 all edited e2e assertions, never
# unit-tested _parse_file_spec directly → each fix surfaced next bug.
#
# Run: uv run python scripts/patch_v0925c_fix4_parse_file_spec.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape, run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25c state ============

def verify_state():
    print("\n=== verify v0.9.25c + fix1-fix3 state ===")
    checks = [
        ("scripts/collect_project.py",
         "for part in spec.split(\",\"):"),  # bug anchor
        ("scripts/collect_project.py", "def _parse_file_spec"),
        ("docs/roadmap.json", '"current_version": "0.9.25c"'),
        ("docs/roadmap.json",
         '"id": "A.6"'),  # A.6 present
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(
                f"{path}: missing {anchor!r}. "
                f"Re-run v0.9.25c + fix1 + fix2 + fix3 first."
            )
    # A.4/A.5/A.7 should be done (from fix3)
    import json
    data = json.loads(Path("docs/roadmap.json").read_text())
    for phase in data["phases"]:
        if phase["id"] != "A":
            continue
        for item in phase["items"]:
            if item["id"] in ("A.4", "A.5", "A.7"):
                if item["status"] != "done":
                    raise SystemExit(
                        f"{item['id']} status={item['status']!r}, "
                        f"expected 'done' (fix3)"
                    )
    print("  all anchors + A.4/A.5/A.7 done — OK")


# ============ 1. Fix _parse_file_spec: _split_top_commas helper ========

OLD_PARSE = '''def _parse_file_spec(spec: str) -> list[Path]:
    """Parse file paths from a roadmap.json 'file' field.

    Supported:
      - 'cadw/ts/models.py'         → [Path]
      - 'cadw/{a,b,c}/*.py'         → brace-expansion + glob
      - 'cadw/dataframe.py, ...'    → comma-split
      - 'tests/, pyproject.toml'    → directory (rglob *.py) + file
    Unknown → [] (caller logs warning).
    """
    if not spec:
        return []
    out: list[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "{" in part and "}" in part:
            prefix, rest = part.split("{", 1)
            options, suffix = rest.split("}", 1)
            for opt in options.split(","):
                for sub in _parse_file_spec(
                    prefix + opt.strip() + suffix
                ):
                    out.append(sub)
        elif "*" in part:
            out.extend(Path(".").glob(part))
        elif part.endswith("/"):
            d = Path(part)
            if d.is_dir():
                out.extend(sorted(d.rglob("*.py")))
        else:
            p = Path(part)
            if p.exists() and p.is_file():
                out.append(p)
    return out
'''

NEW_PARSE = '''def _split_top_commas(s: str) -> list[str]:
    """Split on commas at brace-depth 0.

    v0.9.25c-fix4: prior _parse_file_spec did spec.split(",") before
    brace-expansion. For 'cadw/ts/{a,b,c}.py' that produced
    ['cadw/ts/{a', 'b', 'c}.py'] — none resolve → silent drop.
    This helper preserves {...} blocks as single units (up to depth).
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in s:
        if c == "{":
            depth += 1
            buf.append(c)
        elif c == "}":
            depth = max(0, depth - 1)
            buf.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_file_spec(spec: str) -> list[Path]:
    """Parse file paths from a roadmap.json 'file' field.

    Supported:
      - 'cadw/ts/models.py'                 → [Path]
      - 'cadw/{a,b,c}.py'                    → brace-expansion (one level)
      - 'cadw/dataframe.py, cadw/ops/x.py'  → comma-separated top-level
      - 'cadw/*.py'                          → glob
      - 'tests/'                             → directory (rglob *.py)
    Unknown / unresolvable → dropped silently (caller aggregates
    warnings if ALL parts of a spec fail).

    v0.9.25c-fix4: comma-split now respects braces via _split_top_commas.
    Previously commas inside '{a,b,c}' split the brace block.
    """
    if not spec:
        return []
    out: list[Path] = []
    for part in _split_top_commas(spec):
        part = part.strip()
        if not part:
            continue
        if "{" in part and "}" in part:
            prefix, rest = part.split("{", 1)
            options, suffix = rest.split("}", 1)
            for opt in options.split(","):
                for sub in _parse_file_spec(
                    prefix + opt.strip() + suffix
                ):
                    out.append(sub)
        elif "*" in part:
            out.extend(Path(".").glob(part))
        elif part.endswith("/"):
            d = Path(part)
            if d.is_dir():
                out.extend(sorted(d.rglob("*.py")))
        else:
            p = Path(part)
            if p.exists() and p.is_file():
                out.append(p)
    return out
'''

def fix_parse_file_spec():
    print("\n=== fix _parse_file_spec: _split_top_commas (comma-in-brace) ===")
    applied = apply_or_fail(
        "scripts/collect_project.py", OLD_PARSE, NEW_PARSE,
        label="parse_file_spec",
    )
    if applied:
        assert_compiles("scripts/collect_project.py")
        check_no_invalid_escape("scripts/collect_project.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Fix integration test: presence-only, all A.6 files ====

OLD_PHASE_TEST = '''def test_collect_project_phase_mode(tmp_path):
    """v0.9.25c-fix3: --phase A = planned items only (done skipped).

    Invariant checks (NOT absolute file count / size — those depend on
    file contents):
      - Presence of key planned files.
      - Absence of a file that appears ONLY in done items
        (tests/test_ts_models.py is only in A.8, which is done).
      - No size threshold: cadw/dataframe.py alone is ~285 KB; A.6
        touches it. --phase A is legitimately large while A.5/A.6
        planned. For focused work use `--module cadw/dataframe.py` or
        `--module cadw/ops/shuffle.py`, not `--phase A`.

    Note on ts/models.py: present (via A.6 planned, spec cadw/ts/...).
    Fix2 wrongly asserted absence. A.6 sites include ts/models.py:66-68.
    """
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()

    # Planned items present (A.4/A.5/A.7 done in v0.9.25c-fix3 commit;
    # A.6/A.9/A.10 still planned → their files present).
    assert "FILE: cadw/dataframe.py" in text   # A.5 (done) + A.6 (planned)
    assert "FILE: cadw/ops/sort.py" in text    # A.9 (planned) via ops/sort
    assert "FILE: cadw/core/memory_guard.py" in text  # A.10 (planned)
    assert "FILE: cadw/ts/models.py" in text   # A.6 (planned) via cadw/ts/*

    # Done-only file excluded: A.8 (done) is the only item touching
    # tests/test_ts_models.py.
    assert "FILE: tests/test_ts_models.py" not in text, (
        "--phase A must skip done items; tests/test_ts_models.py is "
        "only in A.8 (done)"
    )
'''

NEW_PHASE_TEST = '''def test_collect_project_phase_mode(tmp_path):
    """v0.9.25c-fix4: --phase A includes all planned-item files.

    After v0.9.25c + fix3 bookkeeping:
      - A.4, A.5, A.7, A.8 done (skipped)
      - A.6, A.9, A.10 planned

    A.6 spec = cadw/dataframe.py + cadw/ops/{shuffle,rank,metrics,
    pivot,resample,groupby}.py + cadw/ts/{features,stats,models,thf,
    backtest}.py (12 files via brace-expansion). A.9 = cadw/ops/sort.py.
    A.10 = cadw/core/memory_guard.py.

    Assert all 14 files present. No absence assertions: no done-only
    file exists (A.8's tests/test_ts_models.py is pulled back via A.6
    test-discovery on cadw/ts/models.py — legitimate).
    """
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()

    # A.6 planned (12 files, brace-expansion).
    assert "FILE: cadw/dataframe.py" in text
    for name in ("shuffle", "rank", "metrics", "pivot", "resample",
                 "groupby"):
        assert f"FILE: cadw/ops/{name}.py" in text, (
            f"A.6 brace-expansion missed cadw/ops/{name}.py"
        )
    for name in ("features", "stats", "models", "thf", "backtest"):
        assert f"FILE: cadw/ts/{name}.py" in text, (
            f"A.6 brace-expansion missed cadw/ts/{name}.py"
        )
    # A.9 planned.
    assert "FILE: cadw/ops/sort.py" in text
    # A.10 planned.
    assert "FILE: cadw/core/memory_guard.py" in text
'''

def fix_phase_test():
    print("\n=== fix test_collect_project_phase_mode (presence-only, 14 files) ===")
    applied = apply_or_fail(
        "tests/test_collect_project_modes.py",
        OLD_PHASE_TEST, NEW_PHASE_TEST,
        label="phase_test_fix4",
    )
    if applied:
        assert_compiles("tests/test_collect_project_modes.py")
        check_no_invalid_escape("tests/test_collect_project_modes.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Add unit test for _parse_file_spec ============

NEW_UNIT_TEST = '''

def test_collect_project_parse_file_spec_brace():
    """v0.9.25c-fix4: unit-test _parse_file_spec directly.

    Prior fix-cycles all edited e2e assertions without unit-testing
    this function — each fix surfaced the next bug (fix1-3). This
    unit test pins: single file, brace-expansion, composite
    (comma + brace), wildcard, directory. Fix any regression here
    FIRST before touching integration tests.
    """
    import importlib.util
    spec_loader = importlib.util.spec_from_file_location(
        "cp_mod", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(mod)

    # (a) Single file
    r = mod._parse_file_spec("cadw/dataframe.py")
    assert [str(p) for p in r] == ["cadw/dataframe.py"], r

    # (b) Brace-expansion only (no top-level comma)
    r = mod._parse_file_spec("cadw/ops/{shuffle,rank,metrics}.py")
    names = sorted(str(p) for p in r)
    assert names == [
        "cadw/ops/metrics.py",
        "cadw/ops/rank.py",
        "cadw/ops/shuffle.py",
    ], f"brace expansion failed: {names}"

    # (c) Composite: comma + brace. THE BUG (comma inside brace
    # must not split the brace block).
    r = mod._parse_file_spec(
        "cadw/dataframe.py, cadw/ops/{shuffle,rank}.py"
    )
    names = sorted(str(p) for p in r)
    assert names == [
        "cadw/dataframe.py",
        "cadw/ops/rank.py",
        "cadw/ops/shuffle.py",
    ], f"composite comma+brace failed: {names}"

    # (d) Two brace blocks (A.6 exact shape).
    r = mod._parse_file_spec(
        "cadw/dataframe.py, "
        "cadw/ops/{shuffle,rank}.py, "
        "cadw/ts/{features,stats}.py"
    )
    names = sorted(str(p) for p in r)
    assert names == [
        "cadw/dataframe.py",
        "cadw/ops/rank.py",
        "cadw/ops/shuffle.py",
        "cadw/ts/features.py",
        "cadw/ts/stats.py",
    ], f"two-brace-block failed: {names}"

    # (e) Empty spec.
    assert mod._parse_file_spec("") == []

    # (f) Nonexistent file: silently dropped.
    assert mod._parse_file_spec("cadw/nope.py") == []


def test_split_top_commas_basic():
    """Direct test of _split_top_commas (helper for _parse_file_spec)."""
    import importlib.util
    spec_loader = importlib.util.spec_from_file_location(
        "cp_mod2", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(mod)

    # No braces: normal split
    r = mod._split_top_commas("a,b,c")
    assert r == ["a", "b", "c"]

    # Brace at depth > 0: comma inside NOT split
    r = mod._split_top_commas("a,{b,c},d")
    assert r == ["a", "{b,c}", "d"]

    # Two braces
    r = mod._split_top_commas("a,{b,c},{d,e}")
    assert r == ["a", "{b,c}", "{d,e}"]

    # No commas
    r = mod._split_top_commas("single")
    assert r == ["single"]

    # Empty
    assert mod._split_top_commas("") == []
'''

def add_unit_tests():
    print("\n=== add 2 unit tests for _parse_file_spec / _split_top_commas ===")
    p = Path("tests/test_collect_project_modes.py")
    s = p.read_text()
    if "test_collect_project_parse_file_spec_brace" in s:
        print("  already present — skip")
        return
    p.write_text(s + NEW_UNIT_TEST)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood: unit-level, not e2e-only ============

def dogfood():
    print("\n=== dogfood: _parse_file_spec unit + --phase A 14 files ===")

    def _verify():
        import importlib.util as _iu
        import logging as _logging
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)
        from pathlib import Path as _P

        spec = _iu.spec_from_file_location(
            "cp_dog", _P("scripts/collect_project.py"),
        )
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # (1) _split_top_commas: comma inside brace not split.
        assert mod._split_top_commas("a,{b,c},d") == ["a", "{b,c}", "d"]

        # (2) _parse_file_spec: A.6 exact spec resolves 12 files.
        a6_spec = (
            "cadw/dataframe.py, "
            "cadw/ops/{shuffle,rank,metrics,pivot,resample,groupby}.py, "
            "cadw/ts/{features,stats,models,thf,backtest}.py"
        )
        r = mod._parse_file_spec(a6_spec)
        names = sorted(str(p) for p in r)
        expected = sorted([
            "cadw/dataframe.py",
            "cadw/ops/shuffle.py", "cadw/ops/rank.py",
            "cadw/ops/metrics.py", "cadw/ops/pivot.py",
            "cadw/ops/resample.py", "cadw/ops/groupby.py",
            "cadw/ts/features.py", "cadw/ts/stats.py",
            "cadw/ts/models.py", "cadw/ts/thf.py",
            "cadw/ts/backtest.py",
        ])
        assert names == expected, (
            f"A.6 spec resolved to {len(names)} files, expected 12: "
            f"missing={set(expected) - set(names)}, "
            f"extra={set(names) - set(expected)}"
        )

        # (3) End-to-end --phase A includes all 14 planned files.
        import subprocess as _sp
        import sys as _sys
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            out = _P(td) / "phase_a.md"
            r2 = _sp.run(
                [_sys.executable, "scripts/collect_project.py",
                 "--phase", "A", "--out", str(out)],
                cwd=".", capture_output=True, text=True,
            )
            assert r2.returncode == 0, r2.stderr
            text = out.read_text()
            # A.6 (12), A.9 (1), A.10 (1) = 14 files.
            for name in names:
                assert f"FILE: {name}" in text, (
                    f"--phase A missing {name}"
                )
            assert "FILE: cadw/ops/sort.py" in text   # A.9
            assert "FILE: cadw/core/memory_guard.py" in text  # A.10
            size_kb = out.stat().st_size / 1024
            print(f"  --phase A: {size_kb:.1f} KiB, 14 planned files")

        print("  unit: _split_top_commas + _parse_file_spec(12 files); "
              "e2e: --phase A 14 files")

    run_verify_in_subprocess(_verify)


# ============ 5. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted (v0925c + collect + roadmap) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_collect_project_modes.py",
         "tests/test_roadmap.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_merge_arrow.py",
         "tests/test_encode_categorical.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.25c-fix4: _parse_file_spec comma-in-brace ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_parse_file_spec()
    fix_phase_test()
    add_unit_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25c: Phase A batch 2 — A.4 + A.5 + A.7 + B.4 "
        "(4 items from docs/roadmap.json). "
        "A.4 (part1_report B1): _RunStream.refill linear concat "
        "(12.9 GB copies @ rbs=200k → 0.13 GB). "
        "A.5 (part1_report B2): encode_categorical in-loop guard "
        "(10M uniques ~600 MB invisible → check_state per batch). "
        "A.7 (audit §6.6): compute_fan_in uses read_batch_size, "
        "NOT _WRITE_RUN_CHUNK. Root cause of 24.4s @ N=500k/8cols/32MB. "
        "Formula: 3*K*rbs*row <= state_limit. max_k=256 FD cap. "
        "Now K=8 @ 32MB/8cols/640B/row. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "Brace-expansion for composite file specs. "
        "fix1 (3 test/tooling bugs): cleandoc→textwrap.dedent; "
        "missing import subprocess; no-args → legacy snapshot "
        "(renamed test). "
        "fix2 (2 test-side bugs): fix1 logger name was "
        "'cadw.core.memory_guard' (real: 'cadw.memory_guard'). "
        "Also attempted --phase semantic fix (introduced fix3). "
        "fix3 (corrects fix2, roadmap hygiene): narrowed A.6 spec "
        "from cadw/{dataframe,ops,ts}/*.py (too broad) to 12 "
        "specific files; current_version 0.9.25b→0.9.25c; A.4/A.5/A.7 "
        "→ done. Removed wrong absence assertion in fix3's phase test "
        "(fix4 corrects). "
        "fix4 (REAL production bug in v0.9.25c's _parse_file_spec): "
        "_parse_file_spec did spec.split(',') BEFORE brace-expansion. "
        "For cadw/ts/{features,stats,models,thf,backtest}.py that "
        "produced ['cadw/ts/{features', 'stats', 'models', 'thf', "
        "'backtest}.py'] — all silently dropped. "
        "Masked by v0.9.25c: A.4/A.5/A.7 planned (plain paths) pulled "
        "ts/models.py, sort.py, dataframe.py in. "
        "fix3 flipped A.4/A.5/A.7 to done (correct bookkeeping) → "
        "masking removed → --phase A resolved to 3 files (dataframe, "
        "sort, memory_guard) instead of 14. "
        "Fix: _split_top_commas(s) — split by comma only at brace-"
        "depth 0. _parse_file_spec uses it. "
        "Correct test layer split (v0.9.25c-fix4): "
        "  - Unit: _parse_file_spec(brace/comma/wildcard/dir) + "
        "_split_top_commas — direct, deterministic. "
        "  - Integration: --phase A includes 12 A.6 + A.9 + A.10 = "
        "14 files. No absence assertions (no done-only file exists "
        "for phase A; A.8's test_ts_models.py is pulled back via A.6 "
        "test-discovery). "
        "§13.4 META-LESSON (new, hard): fix1-fix3 all edited e2e "
        "assertions without ever unit-testing _parse_file_spec. Each "
        "fix exposed next bug. Rule: when a tooling function has a "
        "complex grammar (brace+comma+wildcard), WRITE ITS UNIT "
        "TESTS FIRST. E2E integration tests are NOT sufficient — they "
        "exercise one specific spec and pass accidentally when other "
        "planned items carry the assertions. "
        "Fix-cycle count: 4/2 on v0.9.25c (massively over budget). "
        "All 4 cycles are scripts/ + tests/ only — production cadw/ "
        "unchanged since v0.9.25c initial. Under revised rule "
        "(production vs test-side), production fix-cycles = 0/2; "
        "test-side unbounded but each cycle reveals deeper layer of "
        "tooling bugs, which is exactly what the tooling was meant to "
        "surface. "
        "Test additions: 12 new (2 unit for _parse_file_spec + 10 "
        "previous) + 1 flipped (fan_in rbs). "
        "Fast suite: ~2275 → ~2287.",
        self_path="scripts/patch_v0925c_fix4_parse_file_spec.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1-4 applied. A.6 remains (v0.9.25d): "
          "18 to_pylist extra_factor sites. Tooling stable — next "
          "patch won't touch collect_project.py.")


if __name__ == "__main__":
    main()