"""Guards against version drift between pyproject, cadw.__version__, CHANGELOG.

Added v0.9.43 after two consecutive version-bump misses (v0.9.42-fix1
bumped the CHANGELOG but not pyproject; the v0.9.43 pre-check caught
it before tests). Also guards against a third miss.
"""
import tomllib
from pathlib import Path

import cadw

ROOT = Path(__file__).parent.parent


def test_pyproject_version_matches_cadw():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pp_v = data["project"]["version"]
    assert pp_v == cadw.__version__, (
        f"pyproject version={pp_v!r}, cadw.__version__={cadw.__version__!r}"
    )


def test_changelog_has_top_entry_for_current_version():
    text = (ROOT / "CHANGELOG.md").read_text()
    assert f"### v{cadw.__version__}" in text, (
        f"CHANGELOG has no '### v{cadw.__version__}' section. "
        f"Add it before tagging."
    )

def test_version_is_semver():
    """Version must be plain MAJOR.MINOR.PATCH (no suffix).

    v0.9.100. Cadw uses tag suffixes (e.g. v0.9.93-logistic) for
    sub-releases, but __version__ / pyproject stay on plain semver
    so pip compares correctly.
    """
    import re
    assert re.fullmatch(r"\d+\.\d+\.\d+", cadw.__version__), (
        f"cadw.__version__={cadw.__version__!r} is not plain "
        f"MAJOR.MINOR.PATCH (no suffix allowed)"
    )


def test_version_matches_git_tag_on_master():
    """On master: git tag at HEAD must be v<version> (skips otherwise).

    v0.9.100. Before this test, test_pyproject_version_matches_cadw
    only checked pyproject==cadw.__version__; a missing git tag
    (tagging v0.9.89 but leaving __version__=0.9.88) was silent.
    Release-context-only to avoid false positives on patch branches
    where HEAD is legitimately not on a tag.
    """
    import subprocess
    import pytest
    branch_r = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True,
    )
    if branch_r.stdout.strip() != "master":
        pytest.skip(f"not on master (branch: "
                    f"{branch_r.stdout.strip() or 'detached'})")
    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match",
         "--match", "v[0-9]*"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("HEAD not on a semver tag (post-merge, pre-tag)")
    tag = r.stdout.strip()          # e.g. "v0.9.100"
    tag_version = tag.lstrip("v")
    assert tag_version == cadw.__version__, (
        f"git tag {tag!r} != cadw.__version__ {cadw.__version__!r}; "
        f"CHANGELOG, pyproject, and tag must move together"
    )

