"""v0.9.1a: patch_utils + Protocol split tests."""
import pytest


# ============ patch_utils: ensure_unreleased_header ============

def test_ensure_unreleased_adds_missing():
    from cadw.patch_utils import ensure_unreleased_header
    text = "## [0.9.0] — 2026-10-20\n\nold"
    result = ensure_unreleased_header(text)
    assert "## [Unreleased]" in result
    assert result.startswith("## [Unreleased]")


def test_ensure_unreleased_keeps_existing():
    from cadw.patch_utils import ensure_unreleased_header
    text = "## [Unreleased]\n\n### Added\n\n- x"
    assert ensure_unreleased_header(text) == text


def test_ensure_unreleased_empty_file():
    from cadw.patch_utils import ensure_unreleased_header
    assert ensure_unreleased_header("").startswith("## [Unreleased]")


# ============ patch_utils: append_unreleased_entry ============

def test_append_unreleased_entry():
    from cadw.patch_utils import append_unreleased_entry
    text = "## [Unreleased]\n\n## [0.9.0] — 2026-10-20"
    result = append_unreleased_entry(text, "### Added\n\n- new")
    assert result.index("### Added") < result.index("## [0.9.0]")
    assert result.index("## [Unreleased]") < result.index("### Added")


def test_append_unreleased_entry_without_header():
    """Missing [Unreleased] → helper creates it (no silent skip)."""
    from cadw.patch_utils import append_unreleased_entry
    text = "## [0.9.0] — 2026-10-20\n\nold"
    result = append_unreleased_entry(text, "### Added\n\n- new")
    assert result.startswith("## [Unreleased]")
    assert "### Added" in result


# ============ patch_utils: find_section_anchor ============

def test_find_section_anchor_hash3():
    from cadw.patch_utils import find_section_anchor
    text = "intro\n\n### 11.6. Технический долг\n\nbody"
    pos = find_section_anchor(text, "11.6")
    assert pos is not None
    assert text[pos:pos+8] == "### 11.6"


def test_find_section_anchor_with_dot():
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("## 11.6. Tech debt", "11.6") is not None


def test_find_section_anchor_section_sign():
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("### §11.6. Tech debt", "11.6") is not None


def test_find_section_anchor_not_found():
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("no section here", "11.6") is None


def test_find_section_anchor_does_not_match_prefix():
    """`11.5` must not match `11.50` or `11.5x`."""
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("### 11.50 something", "11.5") is None


# ============ patch_utils: insert_before_section ============

def test_insert_before_section_found():
    from cadw.patch_utils import insert_before_section
    text = "### 11.6. Tech debt\n\nbody"
    new, found = insert_before_section(text, "11.6", "**Insertion**")
    assert found is True
    assert new.index("**Insertion**") < new.index("### 11.6")
    assert new.endswith("body")


def test_insert_before_section_not_found():
    from cadw.patch_utils import insert_before_section
    text = "no section"
    new, found = insert_before_section(text, "11.6", "**Insertion**")
    assert found is False
    assert new == text


# ============ Protocol split (§13.4/147) ============

def test_predictor_does_not_require_transform():
    """Predictor Protocol must not have transform method (§13.4/147)."""
    from cadw.pipeline import Predictor
    required = [m for m in dir(Predictor) if not m.startswith("_")]
    assert "predict" in required
    assert "fit" in required
    assert "transform" not in required, (
        "Predictor must not declare transform (§13.4/147: strict split)"
    )


def test_transformer_has_transform_but_not_predict():
    """Transformer Protocol must not require predict."""
    from cadw.pipeline import Transformer
    required = [m for m in dir(Transformer) if not m.startswith("_")]
    assert "transform" in required
    assert "fit" in required
    assert "predict" not in required


def test_gbdt_step_satisfies_predictor_not_transformer():
    """GBDTStep: fit + predict, no transform → Predictor only."""
    from cadw.pipeline import GBDTStep, Predictor, Transformer
    step = GBDTStep(["x"], "y", n_estimators=5)
    assert isinstance(step, Predictor)
    assert not isinstance(step, Transformer)


def test_cast_step_satisfies_transformer_not_predictor():
    """CastStep: fit + transform, no predict → Transformer only."""
    from cadw.pipeline import CastStep, Predictor, Transformer
    step = CastStep({"v": "float64"})
    assert isinstance(step, Transformer)
    assert not isinstance(step, Predictor)


# ============ grep tooling (§13.4/152) ============

def test_grep_and_warn_finds_hits(tmp_path, capsys):
    from cadw.patch_utils import grep_and_warn
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("old_behavior_xyz = 1\n")
    (d / "test_b.py").write_text("other = 2\n")
    hits = grep_and_warn(["old_behavior_xyz"], [str(d)])
    assert len(hits) == 1
    assert "old_behavior_xyz" in hits[0]


def test_grep_and_warn_empty_when_no_hits(tmp_path):
    from cadw.patch_utils import grep_and_warn
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("new_behavior = 1\n")
    hits = grep_and_warn(["old_behavior_xyz"], [str(d)])
    assert hits == []


def test_grep_and_warn_missing_dir_skipped(tmp_path):
    from cadw.patch_utils import grep_and_warn
    hits = grep_and_warn(["x"], [str(tmp_path / "nope")])
    assert hits == []


def test_assert_no_stale_behavior_raises_on_hit(tmp_path):
    from cadw.patch_utils import assert_no_stale_behavior
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("old_behavior_xyz\n")
    with pytest.raises(SystemExit):
        assert_no_stale_behavior(["old_behavior_xyz"], [str(d)])


def test_assert_no_stale_behavior_allow_list(tmp_path):
    from cadw.patch_utils import assert_no_stale_behavior
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("old_behavior_xyz\n")
    # allow-list should exclude → no raise
    assert_no_stale_behavior(
        ["old_behavior_xyz"], [str(d)], allow=["test_a.py"],
    )


def test_assert_no_stale_behavior_passes_when_clean(tmp_path):
    from cadw.patch_utils import assert_no_stale_behavior
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("new_behavior = 1\n")
    assert_no_stale_behavior(["old_behavior_xyz"], [str(d)])  # no raise


# ============ v0.9.4a: write_content / assert_compiles ============


def test_write_content_basic(tmp_path):
    from cadw.patch_utils import write_content
    p = tmp_path / 'f.py'
    write_content(p, ['line1', 'line2'])
    assert p.read_text() == 'line1\nline2\n'


def test_write_content_empty(tmp_path):
    from cadw.patch_utils import write_content
    p = tmp_path / 'f.py'
    write_content(p, [])
    assert p.read_text() == ''


def test_write_content_with_quotes(tmp_path):
    from cadw.patch_utils import write_content
    p = tmp_path / 'f.py'
    write_content(p, ["a = 'x'", 'b = "y"'])
    text = p.read_text()
    assert 'a = ' in text and "'x'" in text
    assert 'b = ' in text and '"y"' in text


def test_write_content_with_backslashes(tmp_path):
    from cadw.patch_utils import write_content
    p = tmp_path / 'f.py'
    write_content(p, ['x = "a\\nb"'])
    text = p.read_text()
    # content must contain literal 2-char backslash-n, not newline
    assert 'a\\nb' in text


def test_write_content_unicode(tmp_path):
    from cadw.patch_utils import write_content
    p = tmp_path / 'f.py'
    write_content(p, ['# русский текст', '# → ←'])
    text = p.read_text(encoding='utf-8')
    assert 'русский' in text


def test_assert_compiles_ok(tmp_path):
    from cadw.patch_utils import assert_compiles, write_content
    p = tmp_path / 'ok.py'
    write_content(p, ['x = 1'])
    assert_compiles(p)


def test_assert_compiles_bad(tmp_path):
    from cadw.patch_utils import assert_compiles, write_content
    p = tmp_path / 'bad.py'
    write_content(p, ['def f('])
    import pytest
    with pytest.raises(SystemExit):
        assert_compiles(p)


def test_assert_compiles_missing(tmp_path):
    from cadw.patch_utils import assert_compiles
    import pytest
    with pytest.raises(SystemExit):
        assert_compiles(tmp_path / 'nope.py')
