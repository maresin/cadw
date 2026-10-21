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


def test_run_pytest_iterate_callable():
    # Only shape-check: callable and doesn't raise on import.
    from cadw.patch_utils import run_pytest_iterate
    assert callable(run_pytest_iterate)
    import inspect
    sig = inspect.signature(run_pytest_iterate)
    assert 'paths' in sig.parameters


# ============ v0.9.5a: verify_whitelist ============


def test_collect_all_qualnames_basic(tmp_path):
    from cadw.patch_utils import _collect_all_qualnames
    cadw = tmp_path / 'cadw'
    cadw.mkdir()
    (cadw / 'mod.py').write_text(
        'def foo():\n    pass\n'
        '\n'
        'class C:\n'
        '    def bar(self):\n'
        '        pass\n'
    )
    names = _collect_all_qualnames(cadw)
    assert 'cadw/mod.py:foo' in names
    assert 'cadw/mod.py:C.bar' in names
    assert len(names) == 2


def test_collect_all_qualnames_multi_file(tmp_path):
    from cadw.patch_utils import _collect_all_qualnames
    cadw = tmp_path / 'cadw'
    (cadw / 'sub').mkdir(parents=True)
    (cadw / 'a.py').write_text('def one(): pass\n')
    (cadw / 'sub' / 'b.py').write_text(
        'def two(): pass\n'
    )
    names = _collect_all_qualnames(cadw)
    assert 'cadw/a.py:one' in names
    assert 'cadw/sub/b.py:two' in names


def test_verify_whitelist_clean(tmp_path, capsys):
    from cadw.patch_utils import verify_whitelist
    cadw = tmp_path / 'cadw'
    cadw.mkdir()
    (cadw / 'm.py').write_text('def real(): pass\n')
    wl = {'cadw/m.py:real': 'exists' }
    res = verify_whitelist(whitelist=wl, cadw_root=cadw,
                           raise_on_orphan=False)
    assert res['orphans'] == []
    assert res['malformed'] == []


def test_verify_whitelist_orphan(tmp_path, capsys):
    from cadw.patch_utils import verify_whitelist
    cadw = tmp_path / 'cadw'
    cadw.mkdir()
    (cadw / 'm.py').write_text('def real(): pass\n')
    wl = {
        'cadw/m.py:real': 'exists',
        'cadw/m.py:fake': 'orphan',
    }
    res = verify_whitelist(whitelist=wl, cadw_root=cadw,
                           raise_on_orphan=False)
    assert len(res['orphans']) == 1
    assert res['orphans'][0][0] == 'cadw/m.py:fake'


def test_verify_whitelist_malformed_bare(tmp_path):
    from cadw.patch_utils import verify_whitelist
    cadw = tmp_path / 'cadw'
    cadw.mkdir()
    wl = {'bare_key_no_path': 'malformed'}
    res = verify_whitelist(whitelist=wl, cadw_root=cadw,
                           raise_on_orphan=False)
    assert len(res['malformed']) == 1


def test_verify_whitelist_raises_on_orphan(tmp_path):
    import pytest
    from cadw.patch_utils import verify_whitelist
    cadw = tmp_path / 'cadw'
    cadw.mkdir()
    (cadw / 'm.py').write_text('def real(): pass\n')
    wl = {'cadw/m.py:fake': 'orphan'}
    with pytest.raises(SystemExit):
        verify_whitelist(whitelist=wl, cadw_root=cadw)


def test_verify_whitelist_real_cadw():
    # Real cadw/ against real WHITELISTED_FUNCS from test file.
    from cadw.patch_utils import verify_whitelist
    res = verify_whitelist(raise_on_orphan=False)
    assert isinstance(res, dict)
    assert 'orphans' in res and 'malformed' in res
    # If this fails, orphan keys exist — see printed list.
    assert not res['malformed'], (
        'malformed whitelist keys: ' + str(res['malformed'])
    )
    assert not res['orphans'], (
        'orphan whitelist keys: ' + str(res['orphans'])
    )



def test_collect_all_qualnames_nested(tmp_path):
    # v0.9.5a-fix1: nested funcs (Class.method.nested)
    # must be included (K1.1 whitelist uses them).
    from cadw.patch_utils import _collect_all_qualnames
    cadw = tmp_path / 'cadw'
    cadw.mkdir()
    src = (
        'class C:\n'
        '    def method(self):\n'
        '        def nested():\n'
        '            pass\n'
        '        return nested\n'
    )
    (cadw / 'm.py').write_text(src)
    names = _collect_all_qualnames(cadw)
    assert 'cadw/m.py:C.method' in names
    assert 'cadw/m.py:C.method.nested' in names


def test_commit_with_guards_no_param_shadowing():
    # v0.9.5a-fix3: param names must not shadow module-level
    # function names (else `if param: param()` -> TypeError).
    import inspect
    import cadw.patch_utils as pu
    sig = inspect.signature(pu.commit_with_guards)
    module_funcs = {
        n for n, obj in vars(pu).items()
        if callable(obj) and not n.startswith('__')
    }
    for param_name in sig.parameters:
        if param_name in ('self', 'cls'):
            continue
        assert param_name not in module_funcs, (
            'commit_with_guards param ' + repr(param_name)
            + ' shadows module-level function with same name'
        )


# ============ v0.9.6d: AST + subprocess + scoped helpers ============


def test_assert_no_call_in_source_clean():
    from cadw.patch_utils import assert_no_call_in_source
    src = 'def f():\n    return 1\n'
    assert_no_call_in_source(src, 'to_pandas')  # no raise


def test_assert_no_call_in_source_finds_bare():
    from cadw.patch_utils import assert_no_call_in_source
    src = 'def f():\n    to_pandas()\n'
    with pytest.raises(AssertionError, match="'to_pandas'"):
        assert_no_call_in_source(src, 'to_pandas')


def test_assert_no_call_in_source_finds_attr():
    from cadw.patch_utils import assert_no_call_in_source
    src = 'def f(t):\n    t.to_pandas()\n'
    with pytest.raises(AssertionError, match="'to_pandas'"):
        assert_no_call_in_source(src, 'to_pandas')


def test_assert_no_call_in_source_ignores_comment():
    # §13.4/183: substring would catch `# No to_pandas()`. AST does not.
    from cadw.patch_utils import assert_no_call_in_source
    src = 'def f():\n    # No to_pandas() here\n    return 1\n'
    assert_no_call_in_source(src, 'to_pandas')  # no raise


def test_assert_no_call_in_source_ignores_docstring():
    from cadw.patch_utils import assert_no_call_in_source
    src = (
        'def f():\n'
        '    """Use CadwDataFrame.from_table(pa_table)."""\n'
        '    return 1\n'
    )
    assert_no_call_in_source(src, 'from_table')  # no raise


def test_assert_call_order_ok():
    from cadw.patch_utils import assert_call_order
    src = (
        'def f(t):\n'
        '    t.check_state(1)\n'
        '    CadwDataFrame.from_table(t)\n'
    )
    assert_call_order(src, 'check_state', 'from_table')


def test_assert_call_order_wrong():
    from cadw.patch_utils import assert_call_order
    src = (
        'def f(t):\n'
        '    CadwDataFrame.from_table(t)\n'
        '    t.check_state(1)\n'
    )
    with pytest.raises(AssertionError, match='must precede'):
        assert_call_order(src, 'check_state', 'from_table')


def test_assert_call_order_missing_first():
    from cadw.patch_utils import assert_call_order
    src = 'def f():\n    pass\n'
    with pytest.raises(AssertionError, match='not found'):
        assert_call_order(src, 'check_state', 'from_table')


def test_assert_call_order_ignores_docstring():
    # §13.4/185: docstring mention of `from_table` must not affect ordering.
    from cadw.patch_utils import assert_call_order
    src = (
        'def f(t):\n'
        '    """Use CadwDataFrame.from_table(...)."""\n'
        '    t.check_state(1)\n'
        '    CadwDataFrame.from_table(t)\n'
    )
    assert_call_order(src, 'check_state', 'from_table')


def test_is_patched_scoped():
    # §13.4/182: scoped idempotency uses inspect.getsource(fn).
    from cadw.patch_utils import is_patched
    def fn():
        return 'marker_a'
    assert is_patched(fn, 'marker_a')
    assert not is_patched(fn, 'marker_a', 'marker_b')
    assert not is_patched(fn, 'nonexistent')


def test_run_smoke_inline_ok(capsys):
    from cadw.patch_utils import run_smoke_inline
    run_smoke_inline("print('hello-smoke')")
    captured = capsys.readouterr()
    assert 'hello-smoke' in captured.out


def test_run_smoke_inline_fail():
    from cadw.patch_utils import run_smoke_inline
    with pytest.raises(SystemExit):
        run_smoke_inline("raise ValueError('boom')")


def test_run_smoke_inline_has_cadw_on_path():
    # §13.4/184: cwd on sys.path. `import cadw` works.
    from cadw.patch_utils import run_smoke_inline
    run_smoke_inline("import cadw; print('cadw-import-ok')")
