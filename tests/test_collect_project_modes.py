"""v0.9.25c: collect_project.py --module / --phase / --doc modes."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "collect_project.py"


def _run(*args: str, timeout: int = 60):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
    )


def test_collect_project_list_phases():
    r = _run("--list-phases")
    assert r.returncode == 0, r.stderr
    for pid in "ABCDEFGH":
        assert f" {pid} " in r.stdout or f"  {pid} " in r.stdout, (
            f"phase {pid!r} missing from --list-phases: {r.stdout}"
        )
    assert "Roadmap v" in r.stdout


def test_collect_project_module_mode(tmp_path):
    out = tmp_path / "snap.md"
    r = _run("--module", "cadw/ts/models.py", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    text = out.read_text()
    assert "FILE: cadw/ts/models.py" in text
    # Should include its test file
    assert "FILE: tests/test_ts_models.py" in text
    # Marker: SARIMA class present
    assert "class SARIMAForecaster" in text


def test_collect_project_phase_mode(tmp_path):
    """v0.9.25c-fix4: --phase A includes all planned-item files.

    After v0.9.25c + fix3 bookkeeping:
      - A.4, A.5, A.7, A.8 done (skipped)
      - A.6 deferred to Phase G (v0.9.26d); A.9/A.10 done (v0.9.26d)

    A.6 spec = cadw/dataframe.py + cadw/ops/{shuffle,rank,metrics,
    pivot,resample,groupby}.py + cadw/ts/{features,stats,models,thf,
    backtest}.py (12 files via brace-expansion). A.9 = cadw/ops/sort.py.
    A.10 = cadw/core/memory_guard.py.

    Assert A.6 files present (deferred); A.9/A.10 files absent (done). No absence assertions: no done-only
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
    # A.9 done (v0.9.26d) - sort.py excluded from phase A snapshot.
    assert "FILE: cadw/ops/sort.py" not in text
    # A.10 planned.
    assert "FILE: cadw/core/memory_guard.py" not in text


def test_collect_project_doc_mode(tmp_path):
    out = tmp_path / "doc.md"
    r = _run("--doc", "docs/ROADMAP.md", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "FILE: docs/ROADMAP.md" in text
    assert "# CADW Roadmap" in text


def test_collect_project_no_args_uses_legacy_snapshot(tmp_path):
    """v0.9.25c-fix1: no new-mode flags → legacy --mode=smart.

    Backward compat: running `collect_project.py` without --module /
    --phase / --doc / --list-phases falls through to the original
    `main()` (README-documented default = smart snapshot to
    merged_cadw_project.py). Prior test asserted an error — wrong
    about the contract. Verify by overriding --output to a tmp file
    and checking success + non-empty output.
    """
    out = tmp_path / "full_snapshot.md"
    r = _run("--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial snapshot


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
