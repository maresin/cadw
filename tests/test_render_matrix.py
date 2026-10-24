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
