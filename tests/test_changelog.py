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
    """Top CHANGELOG entry == tag on HEAD (only checked on `master`).

    Why branch check (v0.7.4c-fix): after `git checkout -b patch/...`
    from a tagged commit (e.g., right after squash+tag v0.8.2), HEAD
    still carries the old tag. `git describe --exact-match` then
    succeeds, but CHANGELOG legitimately has an in-progress entry
    (`[0.7.4c]`) that's not yet tagged. Test would fail.

    Guard fires only in release context: **current branch == master**
    AND HEAD is on a semver tag. On patch branches — skip.

    Why --match "v[0-9]*": `backup-pre-*` and other non-semver tags
    would otherwise be picked up by `git describe`. Only `vX.Y.Z`.
    """
    # v0.7.4c-fix: only on `master`
    branch_r = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True,
    )
    current_branch = branch_r.stdout.strip()
    if current_branch != "master":
        pytest.skip(
            f"not on master (current branch: {current_branch or 'detached'}); "
            f"CHANGELOG guard fires only on master (release context)"
        )

    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match",
         "--match", "v[0-9]*"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("master HEAD is not on a semver tag (post-merge, pre-tag)")

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

def test_changelog_is_descending():
    """All versions strictly descending (top = highest).

    Регрессия §13.4/130: patch_changelog() с anchor-based insert привёл к
    out-of-order entries — `[0.7.4c]` (patch_v074c) оказался выше
    `[0.8.4]` (patch_v084 вставлен перед `## [0.8.2]`, а `[0.7.4c]`
    был добавлен ещё раньше выше `[0.8.2]`). `test_changelog_has_latest_tag`
    поймал симптом (`top != tag`), этот guard ловит саму причину.
    """
    import re as _re
    text = CHANGELOG.read_text()
    versions = _re.findall(r"^## \[(\d[^\]]*)\]", text, flags=_re.MULTILINE)
    assert versions, "CHANGELOG has no `## [X.Y.Z]` entries"

    def _key(v):
        m = _re.match(r'^(\d+)\.(\d+)\.(\d+)([a-z]?)$', v)
        assert m, f"unrecognized version: {v!r}"
        ma, mi, pa, su = m.groups()
        return (int(ma), int(mi), int(pa), su)

    keys = [_key(v) for v in versions]
    assert keys == sorted(keys, reverse=True), (
        f"CHANGELOG not descending. Order: {list(zip(versions, keys))}"
    )

