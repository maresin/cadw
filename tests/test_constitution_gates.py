"""G5, G6 — constitution gates (AI_CONSTITUTION.md §5).

G5: unbounded hyperparameter allocation without SpillDir.
G6: batch-loop append + np.concatenate/np.vstack (O(n) return).

Runs on cadw/ml/*.py and cadw/ts/*.py. Existing G1-G4 in
test_primitive_reuse / test_guard_before_alloc / test_no_dense_windows.

See AI_CONSTITUTION.md §5 AP1, AP2 for correct patterns.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# v0.9.93-prep-fix: whitelist populated; gates run in fast suite.
# G5/G6 are now pure regression guards: new violations fail merge,
# existing backlog items are whitelisted with rationale + roadmap
# reference. See AI_CONSTITUTION.md §5 AP1/AP2.

CADW = Path(__file__).parent.parent / "cadw"
ROOT = Path(__file__).parent.parent

# G5: identifiers used as O(hyperparameter) shape dims (not n).
# G5: identifiers that MUST be treated as data-dependent (unbounded)
# shape dims. Narrowed from the original over-broad set.
#
# REMOVED (false positives — model hyperparameters, bounded by
# design, not data cardinality):
#   k, K, n_clusters, n_topics, n_trees, n_estimators
# These are O(model) state (k·d, n_trees·psi, ...) and do NOT grow
# with data size. Adding them to G5 would fail every clustering /
# forest model on a legitimate np.zeros((k, d)).
#
# KEPT (genuinely data-dependent, can blow up at scale):
#   vocab, vocab_size, V    — NLP corpora can have 10^5..10^6 terms
#   n_unique                — full-column cardinality
#   max_run                 — BOCPD run-length history (unbounded)
#   n_kernels               — ROCKET/minirocket (user-driven, can be
#                             large but typically ≤ 10k; kept for safety)
#   n_classes               — extreme classification (10^6 classes);
#                             normal cases bounded but non-OOC
UNBOUNDED_HYPERPARAMS = {
    "vocab", "vocab_size", "V", "n_unique", "max_run", "n_kernels",
    "n_classes",
}

ALLOC_ATTRS = {"zeros", "empty", "full", "ones"}

# G6: helper calls that materialise a list into O(n) output.
CONCAT_FUNCS = {"concatenate", "vstack", "hstack", "stack", "array"}

# Iterators that signal per-batch iteration.
BATCH_ITER_ATTRS = {"_iter_filtered_batches", "_iter_arrow_batches",
                    "_iter_batches", "batches", "iter_batches"}

# Whitelist for pre-existing acceptable uses (populate as refactors land).
G5_WHITELIST = {
    # (file, function_name): reason
    ("cadw/ml/gbdt.py", "_grad_hess_multiclass"):
        "P1a refactor (plan_v0.9.87): n_classes shape state is "
        "documented bounded_state=false / refuse_at_guard. "
        "Extreme classification (K > 100) is a documented limitation; "
        "normal use (K <= 10) fits in state_limit.",

    ("cadw/ml/lda.py", "_fit_once"):
        "v0.9.95 roadmap (K-spill): V (vocab) can exceed state_limit "
        "for corpora with 10^5+ unique terms. HashShuffle-based vocab "
        "accumulation planned; beta matrix spill pending.",
}

G6_WHITELIST = {
    # (file, function_name, reason)
}


def _walk_ml_ts():
    for sub in ("ml", "ts"):
        d = CADW / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


def _iter_functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _is_spill_alloc(call, hyperparams: set[str]) -> list[str]:
    """Return list of hyperparam names used as first-shape arg in np.zeros/empty/full/ones."""
    if not isinstance(call, ast.Call):
        return []
    if not isinstance(call.func, ast.Attribute):
        return []
    if call.func.attr not in ALLOC_ATTRS:
        return []
    recv = call.func.value
    if not (isinstance(recv, ast.Name) and recv.id == "np"):
        return []
    if not call.args:
        return []
    shape = call.args[0]
    hits: list[str] = []
    if isinstance(shape, (ast.Tuple, ast.List)):
        for el in shape.elts:
            if isinstance(el, ast.Name) and el.id in hyperparams:
                hits.append(el.id)
    elif isinstance(shape, ast.Name) and shape.id in hyperparams:
        hits.append(shape.id)
    return hits


def _is_inside_spilldir(fn: ast.FunctionDef) -> bool:
    """True if function contains `with SpillDir(...)` or `SpillBuffer(` /
    `SpillArray(` call (allocation is bounded by spill lifecycle)."""
    for n in ast.walk(fn):
        if isinstance(n, ast.With):
            for item in n.items:
                ctx = item.context_expr
                if (isinstance(ctx, ast.Call)
                        and isinstance(ctx.func, ast.Name)
                        and ctx.func.id in ("SpillDir", "SpillBuffer",
                                            "SpillArray")):
                    return True
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("SpillBuffer", "SpillArray")):
            return True
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "check_state"):
            return True
    return False


def _has_batch_iter(fn: ast.FunctionDef) -> bool:
    """True if function contains a for-loop over *_iter*batches() or .batches()."""
    for n in ast.walk(fn):
        if not isinstance(n, ast.For):
            continue
        it = n.iter
        if isinstance(it, ast.Call):
            f = it.func
            if isinstance(f, ast.Attribute) and f.attr in BATCH_ITER_ATTRS:
                return True
            if (isinstance(f, ast.Name)
                    and f.id in ("iter", "iter_batches")):
                # fallback: iter(df._iter_filtered_batches()) - rare
                pass
    return False


def _returns_concat(fn: ast.FunctionDef) -> list[int]:
    """Line numbers of `return np.concatenate(...)` /
    `return np.vstack(...)` / etc. within function."""
    lines: list[int] = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Return):
            continue
        val = n.value
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and val.func.attr in CONCAT_FUNCS
                and isinstance(val.func.value, ast.Name)
                and val.func.value.id == "np"):
            lines.append(n.lineno)
    return lines


def _appends_in_loop(fn: ast.FunctionDef) -> bool:
    """True if function has `for ...: X.append(...)` on a local list."""
    for n in ast.walk(fn):
        if not isinstance(n, ast.For):
            continue
        for inner in ast.walk(n):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr in ("append", "extend")):
                return True
    return False


def _has_output_param(fn: ast.FunctionDef) -> bool:
    """True if function accepts an `output` parameter.

    Functions with `output="memory"|"auto"|"disk"` handle both
    branches explicitly: memory branch returns np.concatenate(parts),
    spill branch returns SpillArray/SpillBuffer.finalize(). The
    concatenate is the user-requested memory path — not a silent
    O(n) return. G6 should not fire here.
    """
    all_args = list(fn.args.args) + list(fn.args.kwonlyargs)
    return any(a.arg == "output" for a in all_args)


def _has_spill_call(fn: ast.FunctionDef) -> bool:
    """True if function constructs SpillArray / SpillBuffer / SpillDir.

    Spill branch exists → np.concatenate is the memory branch only.
    G6 should not fire.
    """
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in (
                    "SpillArray", "SpillBuffer", "SpillDir"):
                return True
            if isinstance(n.func, ast.Attribute) and n.func.attr in (
                    "SpillArray", "SpillBuffer", "SpillDir"):
                return True
    return False


@pytest.mark.parametrize(
    "path", list(_walk_ml_ts()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_g5_unbounded_hyperparam_alloc(path):
    """G5: np.zeros((hyperparam, ...)) not in SpillDir context.

    See AI_CONSTITUTION.md §5 AP1. Use SpillDir + np.memmap for
    hyperparam-sized state.
    """
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: syntax error {e}")

    rel = str(path.relative_to(ROOT))
    violations = []
    for fn in _iter_functions(tree):
        key = (rel, fn.name)
        if key in G5_WHITELIST:
            continue
        for call in ast.walk(fn):
            hits = _is_spill_alloc(call, UNBOUNDED_HYPERPARAMS)
            if hits and not _is_inside_spilldir(fn):
                violations.append((fn.name, call.lineno, hits))

    assert not violations, (
        f"{rel}: np.zeros/empty/full/ones with hyperparameter shape "
        f"not wrapped in SpillDir/SpillBuffer/check_state. "
        f"Violations: {violations[:5]}. "
        f"See AI_CONSTITUTION.md §5 AP1 (unbounded hyperparam). "
        f"Fix: SpillDir + np.memmap, or SpillBuffer/HashShuffle. "
        f"If back-compat path is intentional, add to G5_WHITELIST.")


@pytest.mark.parametrize(
    "path", list(_walk_ml_ts()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_g6_batch_loop_concat(path):
    """G6: batch-loop append + np.concatenate(parts) = O(n) return.

    See AI_CONSTITUTION.md §5 AP2. Use SpillArray.finalize().
    """
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: syntax error {e}")

    rel = str(path.relative_to(ROOT))
    violations = []
    for fn in _iter_functions(tree):
        key = (rel, fn.name)
        if key in G6_WHITELIST:
            continue
        concat_lines = _returns_concat(fn)
        if not concat_lines:
            continue
        if not _has_batch_iter(fn):
            continue
        if not _appends_in_loop(fn):
            continue
        # Skip if function explicitly handles the memory/spill split:
        # - has `output` parameter (memory branch is intentional,
        #   user-requested; spill branch uses SpillArray)
        # - contains SpillArray/SpillBuffer/SpillDir call (spill
        #   branch exists somewhere in the function)
        if _has_output_param(fn) or _has_spill_call(fn):
            continue
        violations.append((fn.name, concat_lines[:3]))

    assert not violations, (
        f"{rel}: batch-loop + append + np.concatenate/vstack return — "
        f"O(n) RAM. Violations: {violations[:5]}. "
        f"See AI_CONSTITUTION.md §5 AP2 (SpillArray pattern). "
        f"Fix: SpillArray(guard=mg, width=K, ...).write_array(...) "
        f"+ finalize(). If back-compat path is intentional, add to "
        f"G6_WHITELIST.")
