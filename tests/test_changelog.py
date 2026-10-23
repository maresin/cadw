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




def _read_all_decisions():
    """v0.9.28-fix3: DECISIONS.md split into docs/decisions/*.md.
    Concatenate all topic files for tests asserting specific sections.
    §13.4/349, §13.4/350 (local import — _P is per-test local)."""
    from pathlib import Path as _P
    _docs = _P(__file__).resolve().parent.parent / "docs" / "decisions"
    return "\n".join(f.read_text() for f in sorted(_docs.glob("*.md")))

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



# ============ v0.9.20-fix1 (§13.4/233): DECISIONS order ============


def test_decisions_226_232_order():
    """v0.9.20-fix1: episodes 226-232 ordered after 225, before §13.5.

    §13.4/233: append with fallback anchor must pick LATEST candidate
    explicitly. v0.9.20 used anchor = "**Gap 190-210" (present from
    v0.9.17) with fallback to §13.5 — since gap-note existed, 226-232
    inserted BEFORE it → order became 189...226-232...gap...211-225.
    Test pins correct order: 225 < 226 < 232 < §13.5.
    """
    from pathlib import Path as _P
    path = _P(__file__).resolve().parent.parent / "DECISIONS.md"
    text = path.read_text()
    if "**Эпизод 226 " not in text:
        pytest.skip("226 block not yet present (v0.9.20 not applied)")
    idx_225 = text.index("**Эпизод 225 ")
    idx_226 = text.index("**Эпизод 226 ")
    idx_232 = text.index("**Эпизод 232 ")
    idx_135 = text.index("### 13.5. Открытые вопросы")
    assert idx_225 < idx_226, (
        f"Эпизод 226 ({idx_226}) appears before 225 ({idx_225}) — "
        f"append inserted in wrong position"
    )
    assert idx_226 < idx_232, (
        f"Эпизод 232 ({idx_232}) before 226 ({idx_226})"
    )
    assert idx_232 < idx_135, (
        f"Эпизод 232 ({idx_232}) after §13.5 ({idx_135})"
    )


# ============ v0.9.21: DECISIONS 233-240 order ============


def test_decisions_233_240_order():
    """v0.9.21: episodes 233-240 after 232, before §13.5."""
    from pathlib import Path as _P
    text = _read_all_decisions()
    if "**Эпизод 233 " not in text:
        pytest.skip("233 block not yet present")
    i_232 = text.index("**Эпизод 232 ")
    i_233 = text.index("**Эпизод 233 ")
    i_240 = text.index("**Эпизод 240 ")
    i_135 = text.index("### 13.5. Открытые вопросы")
    assert i_232 < i_233, (
        f"233 ({i_233}) before 232 ({i_232})"
    )
    assert i_233 < i_240, f"240 ({i_240}) before 233 ({i_233})"
    assert i_240 < i_135, f"240 ({i_240}) after §13.5 ({i_135})"


# ============ v0.9.22: DECISIONS sections + order ============


def test_decisions_v0922_sections():
    """v0.9.22: §9.2.20-22 present + §13.4/241-245 present + ordered."""
    from pathlib import Path as _P
    text = _read_all_decisions()
    # 9.2.20-22 (before §9.3)
    for h in ("### 9.2.20. GBDT top-level peak",
              "### 9.2.21. Pipeline.plan()",
              "### 9.2.22. KFold(materialize=True)"):
        assert h in text, f"missing {h}"
    i_920 = text.index("### 9.2.20. GBDT")
    i_922 = text.index("### 9.2.22. KFold")
    i_93 = text.index("### 9.3. Pipeline planner")
    assert i_920 < i_922 < i_93
    # 13.4/241-245 (after 240, before §13.5)
    for n in range(241, 246):
        assert f"**Эпизод {n} " in text, f"missing episode {n}"
    i_240 = text.index("**Эпизод 240 ")
    i_241 = text.index("**Эпизод 241 ")
    i_245 = text.index("**Эпизод 245 ")
    i_135 = text.index("### 13.5. Открытые вопросы")
    assert i_240 < i_241 < i_245 < i_135
