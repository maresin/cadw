"""v0.9.1a: patch_utils + Protocol split tests."""
import ast
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


# ============ v0.9.6e2: write_content_checked ============


def test_write_content_checked_ok(tmp_path):
    from cadw.patch_utils import write_content_checked
    p = tmp_path / 'f.py'
    write_content_checked(p, ['x = 1'])
    assert p.read_text() == 'x = 1\n'


def test_write_content_checked_bad_raises(tmp_path):
    from cadw.patch_utils import write_content_checked
    p = tmp_path / 'f.py'
    with pytest.raises(SystemExit):
        write_content_checked(p, ['def f('])


# ============ v0.9.6e2: parse_method_source ============


def test_parse_method_source_indented():
    from cadw.patch_utils import parse_method_source
    from cadw.dataframe import CadwDataFrame
    tree = parse_method_source(CadwDataFrame.pivot_table)
    assert any(isinstance(n, ast.FunctionDef) for n in tree.body)


def test_parse_method_source_nested_function():
    from cadw.patch_utils import parse_method_source
    from cadw.dataframe import CadwDataFrame
    tree = parse_method_source(CadwDataFrame._per_group_apply)
    assert tree is not None


# ============ v0.9.6e2: safe_skip_or_write ============


def test_safe_skip_or_write_new(tmp_path):
    from cadw.patch_utils import safe_skip_or_write
    p = tmp_path / 'f.py'
    written = safe_skip_or_write(p, ['x = 1'])
    assert written is False
    assert p.read_text() == 'x = 1\n'


def test_safe_skip_or_write_valid_skips(tmp_path):
    from cadw.patch_utils import safe_skip_or_write
    p = tmp_path / 'f.py'
    p.write_text('x = 1\n')
    written = safe_skip_or_write(p, ['x = 2'],
                                 required_symbols=['x'])
    assert written is True
    assert 'x = 1' in p.read_text()


def test_safe_skip_or_write_broken_rewrites(tmp_path):
    from cadw.patch_utils import safe_skip_or_write
    p = tmp_path / 'f.py'
    p.write_text('def f(\n')
    written = safe_skip_or_write(p, ['x = 1'])
    assert written is False
    assert p.read_text() == 'x = 1\n'


def test_safe_skip_or_write_missing_symbol_rewrites(tmp_path):
    from cadw.patch_utils import safe_skip_or_write
    p = tmp_path / 'f.py'
    p.write_text('y = 1\n')
    written = safe_skip_or_write(p, ['x = 1'],
                                 required_symbols=['x'])
    assert written is False
    assert p.read_text() == 'x = 1\n'


# ============ v0.9.6e2: assert_test_imports_defined ============


def test_assert_test_imports_defined_ok():
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'def test_a():\n'
        '    import os\n'
        '    assert os.path.exists("/") or True\n'
    )
    assert_test_imports_defined(src, fn_name='test_a')


def test_assert_test_imports_defined_catches_missing():
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'def test_a():\n'
        '    missing_helper(1, 2)\n'
    )
    with pytest.raises(AssertionError, match='missing_helper'):
        assert_test_imports_defined(src, fn_name='test_a')


def test_assert_test_imports_defined_module_import():
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'import pytest\n'
        'def test_a():\n'
        '    pytest.raises(ValueError)\n'
    )
    assert_test_imports_defined(src, fn_name='test_a')


def test_assert_test_imports_defined_builtin_ok():
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'def test_a():\n'
        '    assert len([1, 2]) == 2\n'
    )
    assert_test_imports_defined(src, fn_name='test_a')


def test_assert_test_imports_defined_assignment_ok():
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'def test_a():\n'
        '    x = 5\n'
        '    assert x == 5\n'
    )
    assert_test_imports_defined(src, fn_name='test_a')


def test_assert_test_imports_defined_builtin_exceptions_ok():
    # §13.4/195: TypeError/ValueError/etc are builtins; must not
    # be flagged as undefined.
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'def test_a():\n'
        '    raise TypeError("x")\n'
        'def test_b():\n'
        '    try:\n'
        '        int("no")\n'
        '    except ValueError:\n'
        '        pass\n'
    )
    assert_test_imports_defined(src, fn_name='builtin_exceptions')


def test_assert_test_imports_defined_lambda_param_ok():
    # §13.4/195: lambda parameter binds within lambda body; must not
    # be flagged as undefined.
    from cadw.patch_utils import assert_test_imports_defined
    src = (
        'def test_a():\n'
        '    rows = [{"g": "x"}]\n'
        '    out = sorted(rows, key=lambda r: r["g"])\n'
        '    assert len(out) == 1\n'
    )
    assert_test_imports_defined(src, fn_name='lambda_param')

# ============ v0.9.6e2-fix3: meta-guard — all tests bound ============


def test_all_tests_have_bindings():
    """Meta-guard (§13.4/192/196): every tests/test_*.py has all
    bare-Call targets bound (import / local / module / builtin).

    Runs assert_test_imports_defined over every test file. Catches
    future regressions of the class that produced v0.9.6e2-fix3
    (ast missing in test_parse_method_source_indented — helper's
    own bug class, in the helper's own test file).
    """
    from cadw.patch_utils import assert_test_imports_defined
    from pathlib import Path as _P
    root = _P(__file__).parent
    errors: list[str] = []
    for py in sorted(root.glob('test_*.py')):
        try:
            assert_test_imports_defined(
                py.read_text(), fn_name=py.name,
            )
        except AssertionError as e:
            errors.append(str(e))
    assert not errors, (
        'bare-Call target(s) not bound in test file(s):\n'
        + '\n'.join(errors)
    )

# ============ v0.9.9 (F1): run_verify_in_subprocess ============


def test_run_verify_in_subprocess_ok():
    from cadw.patch_utils import run_verify_in_subprocess
    def dogfood_ok():
        import pyarrow  # noqa
        print('inner ok')
    run_verify_in_subprocess(dogfood_ok)


def test_run_verify_in_subprocess_raises_on_failure():
    from cadw.patch_utils import run_verify_in_subprocess
    def dogfood_fails():
        raise ValueError('intentional')
    with pytest.raises(SystemExit):
        run_verify_in_subprocess(dogfood_fails)


def test_run_verify_in_subprocess_has_cadw_fresh():
    from cadw.patch_utils import run_verify_in_subprocess
    def dogfood_import_cadw():
        from cadw.core.memory_guard import MemoryGuard
        g = MemoryGuard(max_memory_mb=10)
        assert g.budget_bytes == 10 * 1024 * 1024
    run_verify_in_subprocess(dogfood_import_cadw)

# ============ v0.9.10 (T2.1): apply_or_fail / verify_anchors ============


def test_apply_or_fail_basic(tmp_path):
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    p.write_text('x = 1\n')
    applied = apply_or_fail(str(p), 'x = 1', 'x = 2', label='t')
    assert applied is True
    assert p.read_text() == 'x = 2\n'


def test_apply_or_fail_idempotent(tmp_path):
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    p.write_text('x = 2\n')
    applied = apply_or_fail(str(p), 'x = 1', 'x = 2', label='t')
    assert applied is False
    assert p.read_text() == 'x = 2\n'


def test_apply_or_fail_raises_on_miss(tmp_path):
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    p.write_text('y = 1\n')
    with pytest.raises(SystemExit):
        apply_or_fail(str(p), 'x = 1', 'x = 2', label='t')


def test_apply_or_fail_prints_context_on_miss(tmp_path, capsys):
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    p.write_text('unique_marker_line = 1\nother = 2\n')
    with pytest.raises(SystemExit):
        apply_or_fail(
            str(p), 'unique_marker_line = 1\nwrong_next',
            'never', label='t',
        )
    captured = capsys.readouterr()
    assert 'unique_marker_line' in captured.out


def test_verify_anchors_ok(tmp_path):
    from cadw.patch_utils import verify_anchors
    p = tmp_path / 'f.py'
    p.write_text('x = 1\n')
    misses = verify_anchors([(str(p), 'x = 1')])
    assert misses == []


def test_verify_anchors_miss_raises(tmp_path):
    import pytest
    from cadw.patch_utils import verify_anchors
    p = tmp_path / 'f.py'
    p.write_text('x = 1\n')
    with pytest.raises(SystemExit):
        verify_anchors([(str(p), 'y = 2')])


def test_verify_anchors_miss_no_fail_fast(tmp_path):
    from cadw.patch_utils import verify_anchors
    p = tmp_path / 'f.py'
    p.write_text('x = 1\n')
    misses = verify_anchors([(str(p), 'y = 2')], fail_fast=False)
    assert len(misses) == 1
    assert misses[0][0] == str(p)

def test_apply_or_fail_no_write_on_compile_fail(tmp_path):
    """v0.9.10-fix3/fix5 (§13.4/211/214): no write on compile fail.

    Prior (fix3): apply_or_fail wrote file, then assert_compiles →
    corrupt file on bad anchor. Fix (fix3): compile in memory, only
    write on success.

    Test data (fix5): `old` and `new` MUST NOT be substrings of each
    other (§13.4/214). Prior test used old='    x = 1\n',
    new='x = 1\n' — new is substring of old (inside original
    'def f():\n    x = 1\n'). apply_or_fail's idempotency check
    `if new in s` returned False (assumed already applied) → no
    SystemExit. Now: old='x = 1', new='def broken(' — no substring
    relation; replacement is syntactically broken → compile fail →
    SystemExit before write.
    """
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    original = 'x = 1\n'
    p.write_text(original)
    with pytest.raises(SystemExit):
        apply_or_fail(str(p), 'x = 1', 'def broken(', label='t')
    assert p.read_text() == original

# ============ v0.9.14-fix1 (T2.9, §13.4/221): validate= kwarg ============


def test_apply_or_fail_toml_valid(tmp_path):
    """T2.9: validate='toml' — TOML file applied with toml validation."""
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.toml'
    p.write_text('[project]\nname = "x"\nversion = "0.1.0"\n')
    applied = apply_or_fail(
        str(p), 'version = "0.1.0"', 'version = "0.2.0"',
        label='t', validate='toml',
    )
    assert applied is True
    assert 'version = "0.2.0"' in p.read_text()


def test_apply_or_fail_toml_invalid(tmp_path):
    """T2.9: validate='toml' — invalid TOML → SystemExit, no write."""
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.toml'
    original = 'name = "x"\n'
    p.write_text(original)
    with pytest.raises(SystemExit):
        apply_or_fail(
            str(p), 'name = "x"', 'broken = = =',
            label='t', validate='toml',
        )
    assert p.read_text() == original


def test_apply_or_fail_infers_none_for_toml(tmp_path):
    """Default (validate=None): .toml → 'none' (no Python compile)."""
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.toml'
    p.write_text('name = "x"\n')
    applied = apply_or_fail(
        str(p), 'name = "x"', 'name = "y"', label='t',
    )
    assert applied is True
    assert 'name = "y"' in p.read_text()


def test_apply_or_fail_python_default_unchanged(tmp_path):
    """Default (validate=None): .py → 'python' compile still enforced."""
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    p.write_text('x = 1\n')
    with pytest.raises(SystemExit):
        apply_or_fail(
            str(p), 'x = 1', 'def broken(', label='t',
        )
    assert p.read_text() == 'x = 1\n'

# ============ v0.9.14 (T2.2): assert_no_call_in_source receiver_type ============


def test_assert_no_call_in_source_receiver_subscript():
    """T2.2: ast.Subscript receiver filter catches X[i].as_py() only."""
    import ast
    from cadw.patch_utils import assert_no_call_in_source
    # Per-row: X[i].as_py() — Subscript receiver → should match
    src_per_row = 'def f(x, i):\n    return x[i].as_py()\n'
    with pytest.raises(AssertionError, match='as_py'):
        assert_no_call_in_source(
            src_per_row, 'as_py', receiver_type=ast.Subscript,
        )
    # Aggregate: pc.sum(x).as_py() — Call receiver → should NOT match
    src_aggregate = 'def f(x):\n    return pc.sum(x).as_py()\n'
    assert_no_call_in_source(  # no raise
        src_aggregate, 'as_py', receiver_type=ast.Subscript,
    )


def test_assert_no_call_in_source_receiver_none_unchanged():
    """Backward compat: receiver_type=None matches any receiver."""
    from cadw.patch_utils import assert_no_call_in_source
    src = 'def f(x):\n    return pc.sum(x).as_py()\n'
    with pytest.raises(AssertionError, match='as_py'):
        assert_no_call_in_source(src, 'as_py')

# ============ v0.9.15 (T2.6/T2.7): dedent_method_source + meta-guard ============


def test_dedent_method_source_basic():
    """T2.6: dedent_method_source returns (str, ast.Module)."""
    import ast as _ast
    from cadw.patch_utils import dedent_method_source
    from cadw.dataframe import CadwDataFrame
    src, tree = dedent_method_source(CadwDataFrame.head)
    assert isinstance(src, str)
    assert isinstance(tree, _ast.Module)
    # Dedented: no leading whitespace before `def`
    first_line = src.split(chr(10))[0]
    assert first_line.startswith('def '), (
        f'expected `def ...` first line, got {first_line!r}'
    )


def test_dedent_method_source_ast_parses():
    """T2.6: returned tree is real Module (not exception)."""
    import ast as _ast
    from cadw.patch_utils import dedent_method_source
    from cadw.dataframe import CadwDataFrame
    src, tree = dedent_method_source(CadwDataFrame._per_group_apply)
    fdefs = [n for n in _ast.walk(tree)
             if isinstance(n, _ast.FunctionDef)]
    assert any(f.name == '_per_group_apply' for f in fdefs)


def test_no_inline_dedent_getsource_outside_helper():
    """T2.7 (§13.4/216): dedent+getsource on class methods must go
    through dedent_method_source helper. Catches future
    'reinvented inline' bugs (v0.9.11-fix1 got it wrong).

    Skips patch_utils.py itself (helper) and any file with
    docstring-only references. Uses AST — no substring false
    positives (§13.4/213).
    """
    import ast as _ast
    from pathlib import Path as _P
    # v0.9.15-fix2 (§13.4/225): cadw/ only. Scripts/
    # skipped — historical patch-scripts (v0.9.6-v0.9.14)
    # predate dedent_method_source helper; they inline
    # _tw.dedent(_inspect.getsource(...)) legitimately
    # (ran once, ephemeral). Future patch-scripts use
    # helper by convention (docstring + code review).
    roots = [_P('cadw')]
    violations: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for py in root.rglob('*.py'):
            if py.name == 'patch_utils.py':
                continue  # helper module
            # v0.9.15-fix2 (§13.4/225): SyntaxWarning —
            # invalid escape sequences in scanned files.
            import warnings as _warn
            try:
                src = py.read_text()
                with _warn.catch_warnings():
                    _warn.simplefilter('ignore', SyntaxWarning)
                    tree = _ast.parse(src)
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            # Find `_tw.dedent(_inspect.getsource(...))` calls
            for n in _ast.walk(tree):
                if not isinstance(n, _ast.Call):
                    continue
                f = n.func
                if not (isinstance(f, _ast.Attribute)
                        and f.attr == 'dedent'):
                    continue
                if not n.args:
                    continue
                inner = n.args[0]
                if not (isinstance(inner, _ast.Call)
                        and isinstance(inner.func, _ast.Attribute)
                        and inner.func.attr == 'getsource'):
                    continue
                # Exception: in patch_utils itself already filtered
                violations.append(f'{py}:{n.lineno}')
    assert not violations, (
        '§13.4/216: inline dedent(getsource(...)) outside '
        'patch_utils.dedent_method_source:'
        + chr(10) + chr(10).join(violations)
    )