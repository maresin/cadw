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
