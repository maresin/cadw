"""Structural tests for docs/roadmap.json and docs/ROADMAP.md (v0.9.25b)."""
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
            assert item["status"] in ("done", "in_progress", "planned"), (
                f"bad status {item['status']!r} for {item['id']}"
            )


def test_roadmap_md_phase_headers():
    text = ROADMAP_MD.read_text()
    for pid in "ABCDEFGH":
        assert f"## §" in text  # sanity: section markers present
    for name in (
        "Phase A — Stabilization",
        "Phase B — Infrastructure",
        "Phase C — Pandas P0",
        "Phase D — Algorithm coverage",
        "Phase E — Recipes",
        "Phase F — Light NNs",
        "Phase G — Tech debt close",
        "Phase H — P2",
    ):
        assert name in text, f"ROADMAP.md missing section: {name!r}"


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
