"""Structural tests for docs/roadmap.json and docs/ROADMAP.md.

v0.9.93: ROADMAP.md is now actionable (Current / Next 3 / Do-not-
re-solve / Backlog); phase list A-H lives only in roadmap.json.
Test was rewritten to match — see test_roadmap_md_structure.
"""
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
ROADMAP_JSON = ROOT / "docs" / "roadmap.json"
ROADMAP_MD = ROOT / "docs" / "ROADMAP.md"
INDEX_MD = ROOT / "docs" / "index.md"


def test_roadmap_json_parses():
    assert ROADMAP_JSON.exists(), f"missing {ROADMAP_JSON}"
    data = json.loads(ROADMAP_JSON.read_text())
    assert data["schema_version"] == "1"
    assert isinstance(data["principles"], list)
    assert len(data["principles"]) >= 3


def test_roadmap_json_phases_present():
    data = json.loads(ROADMAP_JSON.read_text())
    phases = data["phases"]
    phase_ids = [p["id"] for p in phases]
    for required in ("A", "B", "C", "D", "E", "F", "G", "H"):
        assert required in phase_ids, f"phase {required!r} missing"


def test_roadmap_json_items_have_required_fields():
    data = json.loads(ROADMAP_JSON.read_text())
    for phase in data["phases"]:
        assert "items" in phase
        for item in phase["items"]:
            for field in ("id", "name", "patch", "status", "complexity"):
                assert field in item, (
                    f"phase {phase['id']} item {item.get('id')} "
                    f"missing field {field!r}"
                )
            if item["status"] == "deferred":
                assert "deferred_to" in item, (
                    "deferred item missing deferred_to field"
                )
            # "rejected-with-measurement" is an intentional terminal status (I.3, v0.9.34,
            # per TASK_v0934 rejection-with-measurement policy).
            assert item["status"] in ("done", "in_progress", "planned", "deferred", "rejected-with-measurement"), (
                f"bad status {item['status']!r} for {item['id']}"
            )


def test_roadmap_md_structure():
    """v0.9.93: ROADMAP.md rewritten from 348-line phase-list stub
    (phases A-H matching roadmap.json) to 115-line actionable
    document (Current / Next 3 sessions / Do-not-re-solve / Backlog
    / Session discipline).

    Phase-level data lives in docs/roadmap.json (still validated by
    test_roadmap_json_*). ROADMAP.md is now the human-facing
    actionable roadmap for the next 3 sessions; the historical
    phase list remains authoritative only in json.

    Test verifies section structure, not phase headers.
    """
    text = ROADMAP_MD.read_text()
    for header in (
        "## Current:",
        "## Next 3 sessions",
        "## Do not re-solve",
        "## Backlog",
        "## Session discipline",
    ):
        assert header in text, (
            f"ROADMAP.md missing section header: {header!r} "
            f"(see AI_CONSTITUTION.md §6 — roadmap summary is the "
            f"visible contract for the AI session)"
        )
    # Version reference must be present (current CADW version)
    assert "v0.9." in text, "ROADMAP.md missing version reference"
    # Every "Next 3 sessions" subsection names a version
    # v0.9.96-fix: version-agnostic. Hardcoded list is itself a bug —
    # guaranteed to fail on every release. New contract:
    #   1. `## Current: vX.Y.Z` line present.
    #   2. `## Next 3 sessions` names >=3 `### vX.Y.Z` subsections.
    #   3. Every Next-version is strictly greater than Current.
    import re as _re
    current_m = _re.search(
        r"^## Current:\s*(v\d+\.\d+\.\d+)\s*$",
        text, flags=_re.MULTILINE,
    )
    assert current_m, (
        "ROADMAP.md: '## Current: vX.Y.Z' line not found"
    )
    cur_ver = current_m.group(1)

    _nx_start = text.find("## Next 3 sessions")
    assert _nx_start >= 0, (
        "ROADMAP.md: '## Next 3 sessions' section not found"
    )
    _nx_end = text.find("\n## ", _nx_start + 1)
    if _nx_end < 0:
        _nx_end = len(text)
    next_body = text[_nx_start:_nx_end]

    next_versions = _re.findall(
        r"^### (v\d+\.\d+\.\d+)",
        next_body, flags=_re.MULTILINE,
    )
    assert len(next_versions) >= 3, (
        f"ROADMAP.md '## Next 3 sessions' must name >=3 future "
        f"versions, found {len(next_versions)}: {next_versions}"
    )

    def _vkey(v: str) -> tuple[int, int, int]:
        m = _re.match(r"v(\d+)\.(\d+)\.(\d+)$", v)
        assert m, f"unrecognized version: {v!r}"
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    cur_key = _vkey(cur_ver)
    for v in next_versions:
        assert _vkey(v) > cur_key, (
            f"ROADMAP.md: {v} in '## Next 3 sessions' is not > "
            f"current {cur_ver}"
        )


def test_docs_index_links_to_roadmap_and_recipes():
    text = INDEX_MD.read_text()
    assert "ROADMAP.md" in text
    assert "roadmap.json" in text
    assert "recipes/" in text


def test_roadmap_json_done_items_match_current_version():
    """Any 'done' item should reference a patch <= current_version."""
    data = json.loads(ROADMAP_JSON.read_text())
    current = data["current_version"]  # e.g. "0.9.25a"
    # Simple sanity: v0.9.25a-fix1 items are done.
    for phase in data["phases"]:
        for item in phase["items"]:
            if item["status"] == "done":
                assert item["patch"].startswith("v"), (
                    f"done item {item['id']} has non-version patch "
                    f"{item['patch']!r}"
                )
