"""v0.8.9: AST guard — high-risk Arrow/pandas calls must be memory-guarded.

Principle §3.11: any call that allocates more than O(batch) (Table.join,
Table.group_by, to_pandas, pc.quantile, ...) requires a preceding
`memory_guard.check_state(...)` / `check_batch(...)` / `stat().st_size`
check. Enforced at AST level.

IMPORTANT: `Table.from_batches` (pyarrow) materializes a full table;
`Dataset.from_batches` (cadw.dataset) is our *streaming* API — it does
not materialize. Only the former is guarded.

Helper functions with verified bounded allocation are whitelisted with
an explicit rationale (same discipline as §13.4/138 — trust = explicit
bound, not "Arrow is optimized").

v0.9.2 K1.1: whitelist is keyed by `{relative_path}:{qualname}` instead
of bare function name. The old bare-name keys ("fit", "transform",
"predict") whitelisted EVERY method of that name in `cadw/` — e.g.
`RobustScaler.fit` (calls `df.quantile`, Sort-based) was silently
exempt. Qualnames make each exemption explicit and reviewable.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


CADW_ROOT = Path(__file__).resolve().parent.parent / "cadw"

# High-risk method attribute names. Instance methods (Table.join,
# Table.group_by) and module-level pc.* kernels both arrive as
# Attribute calls, so method-name matching covers both.
HIGH_RISK_ATTRS = {
    "join",          # pa.Table.join
    "group_by",      # pa.Table.group_by
    "sort_by",       # pa.Table.sort_by
    "from_batches",  # pa.Table.from_batches (pyarrow) — NOT Dataset
    "to_pandas",     # Table/DataFrame.to_pandas
    "from_pandas",   # pa.Table.from_pandas
    "sort_indices",  # pc.sort_indices
    "quantile",      # pc.quantile
    "rank",          # pc.rank
    "value_counts",  # pc.value_counts
    "tdigest",       # pc.tdigest
    "index_in",      # pc.index_in
    "is_in",         # pc.is_in
    "dictionary_encode",  # pc.dictionary_encode
}

# Whitelisted call sites with verified bound.
# Key: "{path relative to repo root}:{qualname}" — e.g.
# "cadw/ops/join.py:hash_join" or "cadw/scalers.py:RobustScaler.fit".
# Qualname is Class.method / Class.method.nested_func; module-level
# helpers have no dots. Never a bare method name (K1.1, §13.4/170).
# Value: rationale string (shown in violation message).
WHITELISTED_FUNCS = {
    # --- ops/: helpers with their own guard or bounded by design ---
    "cadw/ops/dedup.py:_estimate_k_sample":
        "first batch only; to_pandas on ≤sample rows",
    "cadw/ops/groupby.py:_estimate_k_sample":
        "first batch only; to_pandas on ≤sample rows",
    "cadw/ops/shuffle.py:_pandas_hash_column":
        "per-batch, caller has check_batch(extra_factor=3.0)",
    "cadw/ops/shuffle.py:_python_hash_column":
        "per-batch Python loop, caller check_batch'd",
    "cadw/ops/shuffle.py:_hash_column":
        "per-batch dispatch to one of the above",
    "cadw/ops/groupby.py:_merge_partials":
        "small group_by after full partition check",
    "cadw/ops/groupby.py:_accumulate_heavy":
        "heavy keys ≤ 20 rows (bounded by sample)",
    "cadw/ops/groupby.py:_shuffle_write":
        "per-batch sort_indices (batch ≤ check_batch'd)",
    "cadw/ops/join.py:_exact_actual_matches_arrow":
        "value_counts/join on K-bounded key sets (§9.2.13)",
    "cadw/ops/join.py:_estimate_multi_key_matches":
        "same — K-bounded key cardinality (§9.2.13)",
    "cadw/ops/join.py:_left_preserve_with_nulls":
        "append nulls, no extra allocation",
    "cadw/ops/join.py:hash_join":
        "backward-compat wrapper, caller check_batch'd",
    "cadw/ops/sort.py:_write_run":
        "buf ≤ 0.2×avail; sort_indices per batch",
    # --- dataframe.py: bounded accumulators / per-batch conversions ---
    "cadw/dataframe.py:_concat_batches":
        "bounded accum list (one group/window), not N",
    "cadw/dataframe.py:_convert_batch":
        "per-batch, caller check_batch'd",
    "cadw/dataframe.py:CadwDataFrame.head":
        "≤ n rows accumulated before Table.from_batches",
    "cadw/dataframe.py:CadwDataFrame.encode_categorical":
        "per-batch value_counts; K uniques, not N",
    "cadw/dataframe.py:CadwDataFrame.encode_categorical._iter_encoded":
        "per-batch index_in against K-value dictionary",
    "cadw/dataframe.py:CadwDataFrame._shift_on._iter_zipped":
        "bounded accum list; guard in _shift_on",
    "cadw/dataframe.py:CadwDataFrame._rolling_on._iter_zipped":
        "bounded accum list; guard in _rolling_on",
    "cadw/dataframe.py:CadwDataFrame._per_group_shift._iter_zipped":
        "bounded accum list; guard in _per_group_shift",
    "cadw/dataframe.py:CadwDataFrame._per_group_rolling._iter_zipped":
        "bounded accum list; guard in _per_group_rolling",
    "cadw/dataframe.py:CadwDataFrame._repr_html_":
        "20-row head().to_pandas().to_html(), bounded sample",
    # --- indexers.py ---
    "cadw/indexers.py:_ILocIndexer._empty_view":
        "empty schema table, zero rows",
    # --- scalers.py (now audited; K1.1 surfaced this) ---
    "cadw/scalers.py:RobustScaler.fit":
        "per-column df.quantile — quantile op self-guards (§9.2.N)",
}

GUARD_ATTRS = {
    "check_state",
    "check_batch",
    "check_partition_buffers",
    "stat",
}

AUDITED_FILES = (
    list((CADW_ROOT / "ops").glob("*.py"))
    + list((CADW_ROOT / "ml").glob("*.py"))
    + [
        CADW_ROOT / "dataframe.py",
        CADW_ROOT / "indexers.py",
        CADW_ROOT / "scalers.py",
        CADW_ROOT / "pipeline.py",
        CADW_ROOT / "cv.py",
    ]
)


def _is_table_from_batches(call: ast.Call) -> bool:
    """`pa.Table.from_batches(...)` or `Table.from_batches(...)`.
    NOT `Dataset.from_batches(...)` (streaming, safe)."""
    f = call.func
    if not isinstance(f, ast.Attribute) or f.attr != "from_batches":
        return False
    base = f.value
    if isinstance(base, ast.Name):
        return base.id == "Table"
    if isinstance(base, ast.Attribute):
        # pa.Table.from_batches → base = Attribute(Name('pa'), 'Table')
        return base.attr == "Table"
    return False


def _is_string_method(call: ast.Call) -> bool:
    """`", ".join(...)` — receiver is a str literal; not Arrow."""
    base = call.func.value  # type: ignore[attr-defined]
    return isinstance(base, ast.Constant) and isinstance(base.value, str)


def _get_high_risk_calls(func: ast.AST):
    """Yield (attr, lineno) for high-risk calls in `func`'s own body.

    Nested function/class definitions are NOT descended into — their
    calls are attributed to their own qualname (K1.1).
    """
    stack: list[ast.AST] = [func]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in HIGH_RISK_ATTRS
                and not _is_string_method(child)
            ):
                attr = child.func.attr
                skip = False
                if attr == "from_batches":
                    if not _is_table_from_batches(child):
                        skip = True
                    elif child.args and isinstance(child.args[0], ast.GeneratorExp):
                        # Generator form is safe (zero-copy) — pyarrow 26.
                        skip = True
                if not skip:
                    yield (attr, child.lineno)
            stack.append(child)


def _iter_functions(tree: ast.AST):
    """Yield (qualname, func_node, enclosing_chain) for every def."""

    def visit(node, qual, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from visit(child, qual + [child.name], chain)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(qual + [child.name])
                yield qualname, child, tuple(chain)
                yield from visit(child, qual + [child.name], chain + [child])
            else:
                yield from visit(child, qual, chain)

    yield from visit(tree, [], [])


def _has_guard_before(chain: Iterable[ast.AST], call_line: int) -> bool:
    """True if any guard call appears earlier in scope.

    `chain` is (func, *enclosing) — a guard in the enclosing method
    (e.g. check_state in `_shift_on` before the nested generator
    `_iter_zipped`) counts: the guard runs at setup, the generator is
    only consumed through the streaming Dataset that setup built.
    """
    for fn in chain:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in GUARD_ATTRS
                and node.lineno < call_line
            ):
                return True
    return False


def _violations_for_source(src: str, rel: str) -> list[str]:
    """Violations for one source string; `rel` is repo-relative path."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    violations = []
    for qualname, func, chain in _iter_functions(tree):
        key = f"{rel}:{qualname}"
        if key in WHITELISTED_FUNCS:
            continue
        for attr, lineno in _get_high_risk_calls(func):
            if not _has_guard_before((func, *chain), lineno):
                violations.append(
                    f"{rel}:{lineno}: `{attr}()` in `{qualname}` without "
                    f"preceding check_state/check_batch/stat"
                )
    return violations


def _collect_violations(files: Iterable[Path] | None = None) -> list[str]:
    violations = []
    for py in (files if files is not None else AUDITED_FILES):
        py = Path(py)
        if not py.exists():
            continue
        rel = py.relative_to(CADW_ROOT.parent).as_posix()
        violations.extend(_violations_for_source(py.read_text(), rel))
    return violations


def test_no_unguarded_high_risk_arrow_calls():
    """Every high-risk Arrow/pandas call has a memory guard in scope.

    Guard = `memory_guard.check_state(...)` / `check_batch(...)` /
    `check_partition_buffers(...)` / `path.stat()` earlier in the same
    function (or its enclosing method), OR the call site is
    whitelisted by `{path}:{qualname}` with a rationale (§3.11).
    """
    violations = _collect_violations()
    assert not violations, (
        "Unguarded high-risk Arrow/pandas calls (§3.11, §13.4/138, "
        "§13.4/170):" + "\n".join(f"  {v}" for v in violations)
    )


def test_list_comp_from_batches_flagged():
    """Sanity: list-comp Table.from_batches is flagged; gen form is not."""
    bad_src = """
def f(batches, schema):
    t = pa.Table.from_batches([b for b in batches], schema=schema)
    return t
"""
    fn = ast.parse(bad_src).body[0]
    assert list(_get_high_risk_calls(fn)), (
        "list-comp Table.from_batches should be flagged"
    )

    good_src = """
def f(batches, schema):
    t = pa.Table.from_batches((b for b in batches), schema=schema)
    return t
"""
    fn = ast.parse(good_src).body[0]
    assert not list(_get_high_risk_calls(fn)), (
        "generator Table.from_batches should NOT be flagged"
    )


def test_dataset_from_batches_not_flagged():
    """`Dataset.from_batches` is our streaming API — NOT high-risk."""
    src = """
def f(schema, gen, dir):
    return Dataset.from_batches(schema, gen, dir)
"""
    fn = ast.parse(src).body[0]
    assert not list(_get_high_risk_calls(fn)), (
        "Dataset.from_batches must not be flagged; it streams, does not "
        "materialize (unlike pa.Table.from_batches)"
    )


def test_string_join_not_flagged():
    """`", ".join(...)` on a str literal is not Table.join."""
    src = """
def f(names):
    return ", ".join(names)
"""
    fn = ast.parse(src).body[0]
    assert not list(_get_high_risk_calls(fn))


def test_whitelisted_funcs_documented():
    """Every whitelisted call site has a non-empty rationale."""
    for key, reason in WHITELISTED_FUNCS.items():
        assert isinstance(key, str) and key
        assert isinstance(reason, str) and reason, (
            f"Whitelisted {key!r} must have a rationale string"
        )


def test_whitelist_keys_are_qualified():
    """K1.1: keys must be `path:qualname`, never a bare method name."""
    for key in WHITELISTED_FUNCS:
        path, sep, qualname = key.partition(":")
        assert sep and path and qualname, (
            f"Whitelist key {key!r} must be 'path:qualname'"
        )
        assert qualname not in {"fit", "transform", "predict"}, (
            f"Bare method name {qualname!r} must not be whitelisted "
            f"(would exempt every class — §13.4/170)"
        )


def test_class_method_named_fit_is_flagged():
    """A method `fit` calling pc.quantile is flagged unless its own
    qualname is whitelisted (this is exactly RobustScaler.fit)."""
    src = """
class A:
    def fit(self, df):
        return df.quantile("x", [0.25, 0.5])
"""
    assert _violations_for_source(src, "cadw/fake.py"), (
        "class method `fit` with high-risk call must be flagged"
    )
    assert "cadw/fake.py:A.fit" not in WHITELISTED_FUNCS


def test_qualname_recorded_for_nested():
    """Nested functions get the dotted qualname Class.method.inner."""
    src = """
class C:
    def outer(self):
        def inner(x):
            return x.to_pandas()
        return inner
"""
    names = [q for q, _f, _c in _iter_functions(ast.parse(src))]
    assert "C.outer" in names
    assert "C.outer.inner" in names


def test_robust_scaler_fit_is_detected_when_unwhitelisted(monkeypatch):
    """Prove the whitelist is doing real work for RobustScaler.fit:
    removing its entry makes the scan flag `df.quantile` in that method.
    (Guards against a regression where the method is silently exempted
    by a bare-name whitelist again — §13.4/170.)"""
    key = "cadw/scalers.py:RobustScaler.fit"
    assert key in WHITELISTED_FUNCS, key
    monkeypatch.delitem(WHITELISTED_FUNCS, key)
    scalers = CADW_ROOT / "scalers.py"
    violations = _violations_for_source(
        scalers.read_text(), "cadw/scalers.py",
    )
    hits = [v for v in violations if "RobustScaler.fit" in v]
    assert hits, (
        "RobustScaler.fit must be flagged when its qualname is not "
        "whitelisted (it calls df.quantile)"
    )
    assert any("quantile" in v for v in hits), hits
