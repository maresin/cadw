"""Rendered docs (MATRIX.md, docs/tasks/*.md) must match YAML sources.

Regenerates into memory and compares byte-for-byte with disk. If it
fails, run `python scripts/render_matrix.py` and commit the diff.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ML_TOOLS = ROOT / "docs/ml_tools"
TASKS = ROOT / "docs/tasks"

_spec = importlib.util.spec_from_file_location(
    "render_matrix", ROOT / "scripts/render_matrix.py"
)
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)  # type: ignore


def test_matrix_md_in_sync():
    if not (ML_TOOLS / "MATRIX.md").exists():
        pytest.skip("MATRIX.md not generated; run scripts/render_matrix.py")
    sections = render._load_all_sections()
    expected = render.render_matrix_md(sections)
    actual = (ML_TOOLS / "MATRIX.md").read_text()
    assert actual == expected, (
        "MATRIX.md out of sync with YAML. Run: python scripts/render_matrix.py"
    )


@pytest.mark.parametrize("section_yaml", sorted(ML_TOOLS.glob("[A-Z]_*.yaml")),
                         ids=lambda p: p.stem)
def test_subtype_md_in_sync(section_yaml):
    if not TASKS.exists():
        pytest.skip("docs/tasks/ not generated; run scripts/render_matrix.py")
    sec = render.load_section(section_yaml)
    for st in sec.subtypes:
        fname = f"{sec.section}{st.id.replace('.', '_')}.md"
        path = TASKS / fname
        assert path.exists(), f"missing {path.relative_to(ROOT)}"
        expected = render.render_subtype_md(sec, st)
        actual = path.read_text()
        assert actual == expected, (
            f"{path.relative_to(ROOT)} out of sync with YAML. "
            f"Run: python scripts/render_matrix.py"
        )


# ============================================================
# v0.9.48.8: per-subtype render coverage (fine-grained stale-file detection).
# ============================================================


def _all_subtypes():
    out = []
    for section_yaml in sorted(ML_TOOLS.glob("[A-Z]_*.yaml")):
        sec = render.load_section(section_yaml)
        for st in sec.subtypes:
            out.append((section_yaml, sec, st))
    return out


@pytest.mark.parametrize(
    "section_yaml,sec,subtype",
    _all_subtypes(),
    ids=lambda x: (x.id if hasattr(x, 'id') else (x.stem if hasattr(x, 'stem') else str(x))),
)
def test_subtype_md_exists_nonempty_and_in_sync(section_yaml, sec, subtype):
    """Each docs/tasks/<Section><subtype>.md file must (a) exist, (b) be
    non-empty, (c) start with the subtype header, (d) be byte-identical to
    render_subtype_md(). Catches a stale or truncated task file for one
    subtype without failing the whole section."""
    if not TASKS.exists():
        pytest.skip("docs/tasks/ not generated; run scripts/render_matrix.py")
    fname = f"{sec.section}{subtype.id.replace('.', '_')}.md"
    path = TASKS / fname
    assert path.exists(), f"missing {path.relative_to(ROOT)}"
    actual = path.read_text()
    assert actual, f"{path.relative_to(ROOT)} is empty"
    expected_header = f"# {subtype.id} — {subtype.name}"
    first_line = actual.splitlines()[0] if actual.splitlines() else ""
    assert first_line == expected_header, (
        f"{path.relative_to(ROOT)}: header mismatch; "
        f"got {first_line!r}, expected {expected_header!r}"
    )
    expected = render.render_subtype_md(sec, subtype)
    assert actual == expected, (
        f"{path.relative_to(ROOT)} out of sync with YAML. "
        f"Run: python scripts/render_matrix.py"
    )
