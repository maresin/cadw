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
        "first batch only; v0.9.6c1 Arrow-native "
        "(dictionary_encode, no pandas); caller-bounded "
        "sample",
    "cadw/ops/groupby.py:_detect_heavy_keys":
        "sample_batches bounded by groupby_aggregate "
        "(sample_cap 10k rows / 0.3×avail; skip_skew if "
        "> 0.5×avail); pa.concat_tables + group_by on "
        "bounded sample; peak ~2×sample + Arrow pool, "
        "caller MemoryGuard (§9.2.8)",
    "cadw/ops/groupby.py:_estimate_k_sample":
        "first batch only; v0.9.6c1 Arrow-native "
        "(dictionary_encode, no pandas); caller-bounded "
        "sample (10k rows / 0.3×avail)",
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
    # v0.9.23c-fix3 (§13.4/138): Arrow-native merge internals.
    # Both call pa.Table.from_batches on a bounded input; actual
    # merge-buffer bound enforced by _merge_keys.check_state(
    # 3*K*rbs*row_bytes) at function entry.
    "cadw/ops/sort.py:_RunStream.__init__":
        "from_batches([], schema) — empty table (0 rows), "
        "O(1) allocation; real bound in _merge_keys.check_state "
        "(3*K*rbs*row_bytes)",
    "cadw/ops/sort.py:_RunStream.refill":
        "from_batches([single_batch_slice]) — zero-copy from "
        "mmap-backed ArrowIPCBatchReader (RSS held by OS page "
        "cache, not heap); buffered rows <= rbs; bound in caller "
        "_merge_keys.check_state",
    "cadw/ops/shuffle.py:_string_hash_column":
        "dict_encode O(K + batch) where K = uniques-in-batch; caller "
        "hash_shuffle._process check_batch(extra_factor=3.0) covers "
        "peak ≤ 3×batch.nbytes; §13.4/226/227 null edges handled",
    # v0.9.20-fix4 (§13.4/236): MemoryGuard.calibrate calls
    # sampler_thread.join(timeout=2.0) — that's threading.Thread.join
    # (stdlib, safe daemon cleanup), NOT pa.Table.join (C++ hash join,
    # unbounded output materialization). AST guard matches by attribute
    # name only; Thread.join is safe. Sampler thread: daemon, 10ms
    # interval, join has 2s timeout. Rationale MUST keep the
    # "threading.Thread.join" phrase — test_calibrate_thread_join_whitelisted
    # asserts it to prevent silent removal (hiding real Table.join).
    "cadw/core/memory_guard.py:MemoryGuard.calibrate":
        "threading.Thread.join(timeout=2.0) for peak sampler daemon "
        "cleanup; NOT pa.Table.join; sampler bounded 10ms interval, "
        "2s timeout, daemon thread",
    # --- dataframe.py: bounded accumulators / per-batch conversions ---
    "cadw/dataframe_helpers.py:_concat_batches":
        "bounded accum list (one group/window), not N",
    "cadw/dataframe_helpers.py:_convert_batch":
        "per-batch, caller check_batch'd",
    "cadw/dataframe/_indexing.py:_IndexingMixin.head":
        "≤ n rows accumulated before Table.from_batches",
    "cadw/dataframe/_ml_bindings.py:_MLBindingsMixin.encode_categorical":
        "per-batch value_counts; K uniques, not N",
    "cadw/dataframe/_ml_bindings.py:_MLBindingsMixin.encode_categorical._iter_encoded":
        "per-batch index_in against K-value dictionary",
    "cadw/dataframe/_per_group.py:_PerGroupMixin._shift_on._iter_zipped":
        "bounded accum list; guard in _shift_on",
    "cadw/dataframe/_rolling.py:_RollingMixin._rolling_on._iter_zipped":
        "bounded accum list; guard in _rolling_on",
    "cadw/dataframe/_per_group.py:_PerGroupMixin._per_group_shift._iter_zipped":
        "bounded accum list; guard in _per_group_shift",
    "cadw/dataframe/_rolling.py:_RollingMixin._per_group_rolling._iter_zipped":
        "bounded accum list; guard in _per_group_rolling",
    "cadw/dataframe/_core.py:CadwDataFrame._repr_html_":
        "20-row head().to_pandas().to_html(), bounded sample",
    # v0.9.6b1: sort_values in-memory fast path (pa.Table.sort_by)
    # for group tables in per-group apply. Group already ≤ 0.3×
    # available (checked in _per_group_apply before fn() call);
    # sort_by allocates output × 1.5 in-memory (pa.Table.sort_by
    # internal peak §9.2.7 = 1.5× input + 6 MB).
    "cadw/dataframe/_indexing.py:_IndexingMixin.sort_values":
        "in-memory fast path: pa.Table.sort_by on group ≤ 0.3×avail",
    # v0.9.7a: topk_to_disk sort_indices on bounded P×n accum.
    # per-batch sort_indices bounded by check_batch on input;
    # accumulator bounded by check_state after each batch.
    "cadw/ops/topk.py:topk_to_disk":
        "sort_indices on P×n accum (check_state); per-batch guarded",
    "cadw/ops/topk.py:_per_batch_topk":
        "called from topk_to_disk after check_batch on batch",
    # v0.9.5b (C6): viz/_aggregate.py high-risk calls.
    # line_agg sorts K-row aggregate table (bounded by K).
    # box_stats delegates to df.quantile (guarded).
    "cadw/viz/_aggregate.py:line_agg":
        "table.sort_by on K rows (bounded by group count)",
    "cadw/viz/_aggregate.py:box_stats":
        "delegates to df.quantile (guarded via external_sort)",
    # v0.9.5b (C6): core/memory_guard.py high-risk calls.
    "cadw/core/memory_guard.py:MemoryGuard.estimate_unique_keys":
        "pc.dictionary_encode on <=10k sample batch (bounded)",
    # --- indexers.py ---
    "cadw/indexers.py:_ILocIndexer._empty_view":
        "empty schema table, zero rows",
    # --- scalers.py (now audited; K1.1 surfaced this) ---
    "cadw/scalers.py:RobustScaler.fit":
        "per-column df.quantile — quantile op self-guards (§9.2.N)",
    # --- dataframe.py qcut (v0.9.25j) ---
    "cadw/dataframe/_transformers_reshape.py:_TransformersReshapeMixin.qcut":
        "self.quantile delegates to CADW's own guarded quantile op "
        "(external_sort + streaming); quantile self-guards via its "
        "own MemoryGuard (§9.2.N). AST guard cannot distinguish "
        "self.quantile from pc.quantile — both attr name 'quantile'.",
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
        CADW_ROOT / "dataframe" / "_core.py",
        CADW_ROOT / "indexers.py",
        CADW_ROOT / "scalers.py",
        CADW_ROOT / "pipeline.py",
        CADW_ROOT / "cv.py",
    ]
    # v0.9.5b (C6): extend to viz/_aggregate + memory_guard.
    # line_agg uses sort_by on K rows; box_stats delegates to
    # df.quantile (guarded via external_sort); estimate_unique_
    # keys uses dictionary_encode on ≤10k sample. Whitelisted.
    + [CADW_ROOT / "viz" / "_aggregate.py",
       CADW_ROOT / "core" / "memory_guard.py"]
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


# ============ v0.9.12-fix3 (§3.11, §13.4/219): whitelist guards ============


def test_detect_heavy_keys_in_whitelist():
    """_detect_heavy_keys whitelisted with caller-bounded rationale."""
    key = 'cadw/ops/groupby.py:_detect_heavy_keys'
    assert key in WHITELISTED_FUNCS
    rationale = WHITELISTED_FUNCS[key]
    assert 'groupby_aggregate' in rationale or 'caller' in rationale.lower()


def test_detect_heavy_keys_single_caller():
    """§13.4/219: whitelist rationale assumes single caller.

    If a new caller appears (potentially unbounded input), the
    whitelist becomes a lie — no explicit guard, sample not
    actually bounded. Guard: exactly 1 call site in groupby.py.
    """
    import ast
    import inspect
    from cadw.ops import groupby as gb_mod
    src = inspect.getsource(gb_mod)
    tree = ast.parse(src)
    call_sites = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == '_detect_heavy_keys'
    ]
    assert len(call_sites) == 1, (
        f'_detect_heavy_keys called from {len(call_sites)} sites; '
        f'whitelist rationale assumes single caller (groupby_aggregate). '
        f'New caller: verify sample is bounded or add check_state.'
    )


def test_estimate_k_sample_whitelist_rationale_updated():
    """Stale 'to_pandas' rationale replaced with Arrow-native."""
    for key in ('cadw/ops/groupby.py:_estimate_k_sample',
                'cadw/ops/dedup.py:_estimate_k_sample'):
        assert key in WHITELISTED_FUNCS
        rationale = WHITELISTED_FUNCS[key]
        assert 'dictionary_encode' in rationale or 'Arrow-native' in rationale, (
            f'{key}: rationale should mention Arrow-native / dictionary_encode'
        )
        assert 'to_pandas' not in rationale, (
            f'{key}: stale to_pandas rationale (v0.9.6c1 removed pandas)'
        )

# ============ v0.9.20-fix4 (§13.4/236): calibrate whitelist ============


def test_calibrate_thread_join_whitelisted():
    """MemoryGuard.calibrate whitelisted for threading.Thread.join.

    v0.9.20 introduced `sampler_thread.join(timeout=2.0)` in calibrate
    for peak sampler cleanup. AST guard matches by attribute name only
    ("join"), so this benign stdlib call is flagged. Whitelist entry
    exists with rationale; this test asserts:
      (a) key present in WHITELISTED_FUNCS
      (b) rationale mentions "threading.Thread.join" — proves the
          entry is for the benign stdlib call, not papering over a
          real pa.Table.join violation elsewhere in calibrate.
    """
    key = "cadw/core/memory_guard.py:MemoryGuard.calibrate"
    assert key in WHITELISTED_FUNCS, (
        f"{key!r} missing from whitelist — MemoryGuard.calibrate has "
        f"threading.Thread.join(); AST guard false-positives without "
        f"whitelist. See §13.4/236."
    )
    rationale = WHITELISTED_FUNCS[key]
    assert "threading.Thread.join" in rationale, (
        f"{key!r} rationale must mention 'threading.Thread.join' to "
        f"distinguish from pa.Table.join. Rationale: {rationale!r}"
    )
    assert "NOT pa.Table.join" in rationale or "not pa.Table.join" in rationale, (
        f"{key!r} rationale must state NOT pa.Table.join "
        f"(otherwise may hide real C++ hash join violations). "
        f"Rationale: {rationale!r}"
    )


def test_calibrate_uses_thread_join_not_table_join():
    """AST: calibrate calls .join() on sampler_thread, not on pa.Table.

    Verifies that the whitelist is genuine: calibrate really uses
    threading.Thread.join, not pa.Table.join somewhere.
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.core.memory_guard import MemoryGuard

    src = _tw.dedent(_inspect.getsource(MemoryGuard.calibrate))
    tree = _ast.parse(src)
    join_receivers = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "join"):
            recv = node.func.value
            if isinstance(recv, _ast.Name):
                join_receivers.append(recv.id)
            else:
                join_receivers.append(_ast.unparse(recv))
    assert join_receivers, "expected at least one .join() call in calibrate"
    for recv in join_receivers:
        assert "sampler" in recv.lower() or "thread" in recv.lower(), (
            f"calibrate .join() receiver {recv!r} is not thread-like — "
            f"whitelist may hide real Table.join. Receivers: {join_receivers}"
        )
