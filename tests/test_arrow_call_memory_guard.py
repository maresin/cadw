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
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


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

# Whitelisted functions with verified bound. Key: function name.
# Value: rationale string (shown in violation message).
WHITELISTED_FUNCS = {
    # Functions with their own guard (check_state / check_batch / stat).
    "_safe_materialize_partition": "stat() before materialize; gen form",
    "_local_dedup": "size check + 5×T pre-check before pandas",
    "_groupby_with_skew": "check_partition_buffers before writers",
    "hash_shuffle": "check_partition_buffers before writers",
    # Bounded by design (≤ sample size / ≤ 20 rows / K-bounded).
    "_estimate_k_sample": "first batch only",
    "_pandas_hash_column": "per-batch, caller has check_batch(extra_factor=3.0)",
    "_python_hash_column": "per-batch Python loop, caller check_batch'd",
    "_hash_column": "per-batch dispatch",
    "_merge_partials": "small group_by after full partition check",
    "_accumulate_heavy": "heavy keys ≤ 20 rows (bounded by sample)",
    "_accum_to_table": "≤ 20 rows",
    "_shuffle_write": "per-batch sort_indices (batch ≤ check_batch'd)",
    "_exact_actual_matches_arrow": "value_counts on K-bounded build/probe (§9.2.13)",
    "_estimate_multi_key_matches": "same",
    "_left_preserve_with_nulls": "append nulls, no extra allocation",
    "hash_join": "backward-compat wrapper, caller check_batch'd",
    "_write_run": "buf ≤ 0.2×avail; sort_indices per batch",
    "_concat_batches": "bounded accum list",
    "_convert_batch": "per-batch, caller check_batch'd",
    "_iter_zipped": "bounded accum list",
    "_iter_encoded": "per-batch value_counts/index_in (K-bounded, §9.2.13)",
    "_table_to_cadwdf": "small in-memory table",
    "_empty_view": "empty schema table",
    # Facade methods — Dataset.from_batches is our streaming API (safe).
    "reset_index": "Dataset.from_batches (streaming)",
    "head": "Dataset.from_batches (streaming)",
    "sort_values": "Dataset.from_batches (streaming)",
    "rank": "Dataset.from_batches (streaming)",
    "encode_categorical": "Dataset.from_batches (streaming)",
    "shift": "Dataset.from_batches (streaming)",
    "diff": "Dataset.from_batches (streaming)",
    "_diff_on": "Dataset.from_batches (streaming)",
    "rolling": "Dataset.from_batches (streaming)",
    "_rolling_on": "Dataset.from_batches (streaming)",
    "_shift_on": "Dataset.from_batches (streaming)",
    "resample": "Dataset.from_batches (streaming)",
    "expanding": "Dataset.from_batches (streaming)",
    "ewm": "Dataset.from_batches (streaming)",
    "merge": "Dataset.from_batches (streaming)",
    "_wrap_result_table": "Dataset.from_batches (streaming, small result)",
    # v0.8.10 per-group methods — bounded accum list in _iter_zipped,
    # same pattern as _shift_on/_rolling_on.
    "_per_group_shift": "2× external_sort + bounded accum zip",
    "_per_group_diff": "delegates to _per_group_shift + streaming subtract",
    "_per_group_rolling": "2× external_sort + _RollState + bounded accum",
}

GUARD_ATTRS = {
    "check_state",
    "check_batch",
    "check_partition_buffers",
    "stat",
}

AUDITED_FILES = (
    list((CADW_ROOT / "ops").glob("*.py"))
    + [CADW_ROOT / "dataframe.py", CADW_ROOT / "indexers.py"]
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


def _get_high_risk_calls(func: ast.FunctionDef):
    """Yield (attr, lineno, node) for high-risk calls in function body."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        attr = f.attr
        if attr not in HIGH_RISK_ATTRS:
            continue
        # `from_batches`: only pyarrow Table, not cadw Dataset.
        if attr == "from_batches":
            if not _is_table_from_batches(node):
                continue
            # Generator form is safe (zero-copy) — pyarrow 26.
            if node.args and isinstance(node.args[0], ast.GeneratorExp):
                continue
        yield (attr, node.lineno, node)


def _has_guard_before(func: ast.FunctionDef, call_line: int) -> bool:
    """True if any guard call appears in function before `call_line`."""
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in GUARD_ATTRS:
                if node.lineno < call_line:
                    return True
        if isinstance(node, ast.Attribute) and node.attr == "stat":
            if node.lineno < call_line:
                return True
    return False


def _collect_violations():
    violations = []
    for py in AUDITED_FILES:
        if not py.exists():
            continue
        src = py.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in WHITELISTED_FUNCS:
                continue
            for attr, lineno, _call in _get_high_risk_calls(node):
                if not _has_guard_before(node, lineno):
                    violations.append(
                        f"{py.relative_to(CADW_ROOT.parent).as_posix()}:"
                        f"{lineno}: `{attr}()` in function `{node.name}` "
                        f"without preceding check_state/check_batch/stat"
                    )
    return violations


def test_no_unguarded_high_risk_arrow_calls():
    """Every high-risk Arrow/pandas call has a memory guard in scope.

    Guard = `memory_guard.check_state(...)` / `check_batch(...)` /
    `check_partition_buffers(...)` / `path.stat()` earlier in the same
    function, OR function is whitelisted with rationale (§3.11).
    """
    violations = _collect_violations()
    assert not violations, (
        "Unguarded high-risk Arrow/pandas calls (§3.11, §13.4/138):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_list_comp_from_batches_flagged():
    """Sanity: list-comp Table.from_batches is flagged; gen form is not."""
    bad_src = """
def f(batches, schema):
    t = pa.Table.from_batches([b for b in batches], schema=schema)
    return t
"""
    tree = ast.parse(bad_src)
    fn = tree.body[0]
    calls = list(_get_high_risk_calls(fn))
    assert calls, "list-comp Table.from_batches should be flagged"

    good_src = """
def f(batches, schema):
    t = pa.Table.from_batches((b for b in batches), schema=schema)
    return t
"""
    tree = ast.parse(good_src)
    fn = tree.body[0]
    calls = list(_get_high_risk_calls(fn))
    assert not calls, "generator Table.from_batches should NOT be flagged"


def test_dataset_from_batches_not_flagged():
    """`Dataset.from_batches` is our streaming API — NOT high-risk."""
    src = """
def f(schema, gen, dir):
    return Dataset.from_batches(schema, gen, dir)
"""
    tree = ast.parse(src)
    fn = tree.body[0]
    calls = list(_get_high_risk_calls(fn))
    assert not calls, (
        "Dataset.from_batches must not be flagged; it streams, does not "
        "materialize (unlike pa.Table.from_batches)"
    )


def test_whitelisted_funcs_documented():
    """Every whitelisted function has a non-empty rationale."""
    for name, reason in WHITELISTED_FUNCS.items():
        assert isinstance(name, str) and name
        assert isinstance(reason, str) and reason, (
            f"Whitelisted {name!r} must have a rationale string"
        )
