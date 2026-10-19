"""test_changelog.py — guard: CHANGELOG reflects latest tag.

Регрессия v0.8.1-fix3-fix4: CHANGELOG был сломан с v0.1.5. Все
patch_changelog() были silent no-op'ами (цепочка оборвалась). Тег v0.8.1
был бы создан без записи в CHANGELOG. Guard: при наличии тегов —
top entry CHANGELOG должен совпадать с latest tag.
"""
import re
import subprocess
from pathlib import Path

import pytest


CHANGELOG = Path(__file__).parent.parent / "CHANGELOG.md"


def test_changelog_exists():
    assert CHANGELOG.exists(), f"{CHANGELOG} missing"


def test_changelog_has_latest_tag():
    """Top CHANGELOG entry == tag on HEAD (only checked when HEAD is tagged).

    Why --exact-match: patch-branch workflow has HEAD ahead of latest
    release tag; CHANGELOG legitimately has entries for versions that
    are not yet tagged. Skip is correct — this test fires on release
    commits, not during development.

    Why --match "v[0-9]*": `backup-pre-*` and other non-semver tags
    would otherwise be picked up by `git describe`. Only `vX.Y.Z`.
    """
    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match",
         "--match", "v[0-9]*"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("HEAD is not on a semver tag (in development)")

    tag = r.stdout.strip()          # e.g. "v0.8.1"
    version = tag.lstrip("v")       # "0.8.1"

    text = CHANGELOG.read_text()
    versions = re.findall(r"^## \[(\d[^\]]*)\]", text, flags=re.MULTILINE)
    assert versions, "CHANGELOG has no `## [X.Y.Z]` entries"

    top = versions[0]
    assert top == version, (
        f"CHANGELOG top entry [{top}] != tag [{tag}]. "
        f"Add CHANGELOG entry for {tag} before tagging. "
        f"See §13.4/111."
    )
