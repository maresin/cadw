"""v0.9.34 (B6): README version drift.

Bug: README hardcodes stale versions (v0.8.8, v0.9.24, ~1096 tests).
Fix: replace hardcoded numbers with pointer to cadw.__version__.
"""
from pathlib import Path


README = Path("README.md")


def test_readme_mentions_version_source():
    txt = README.read_text(encoding="utf-8")
    assert "cadw.__version__" in txt or "__version__" in txt


def test_readme_no_hardcoded_old_version():
    txt = README.read_text(encoding="utf-8")
    assert "v0.8.8" not in txt
    assert "v0.9.24" not in txt
