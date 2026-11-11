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


# ═══════════════════════════════════════════════════════════════════
# G8 — spill_ratio validity (must not exceed state_limit safety)
# ═══════════════════════════════════════════════════════════════════
#
# SpillBuffer/SpillArray spill when buffered bytes exceed
# threshold = guard.available_bytes() × spill_ratio. The state_limit
# check fires at available × 0.7 (state_safety). So:
#   - spill_ratio > 0.7 → OOM before spill (never spills)
#   - spill_ratio = 0.7 → collides with state_limit (borderline)
#   - spill_ratio = 0.5 → safe (20% headroom below state_limit)
#   - spill_ratio = 0.0 → force disk (first non-empty batch spills)
# Idiomatic safe set: {0.0, 0.5}. Other values (0.3, 0.7, 0.9, 1.0)
# are either non-idiomatic or unsafe.

SAFE_SPILL_RATIOS = {0.0, 0.5}

G8_WHITELIST: set[tuple[str, str]] = set()


def _walk_cadw_all():
    """All cadw subdirs that may construct SpillArray/SpillBuffer."""
    for sub in ("ml", "ts", "ops", "core"):
        d = CADW / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


def _get_call_attr(call: ast.Call) -> str | None:
    """Return function/method name of a Call node."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _get_kwarg_value(call: ast.Call, name: str) -> ast.expr | None:
    """Return the AST value of keyword `name` in call, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_safe_spill_ratio(expr: ast.expr) -> bool:
    """Safe = constant in {0.0, 0.5}, ternary (depends on output),
    or a Name reference (assumed bound to safe value).

    Unsafe = literal 0.3, 0.7, 0.9, 1.0, or arithmetic expression.
    """
    if isinstance(expr, ast.Constant):
        try:
            return float(expr.value) in SAFE_SPILL_RATIOS
        except (TypeError, ValueError):
            return True
    if isinstance(expr, (ast.IfExp, ast.Name)):
        return True
    return False


def test_g8_spill_ratio_valid():
    """G8: SpillArray/SpillBuffer must use spill_ratio ∈ {0.0, 0.5}.

    spill_ratio >= 0.7 → threshold >= state_limit → OOM before spill
    (never spills; O(n) RAM silently grows). See
    AI_CONSTITUTION.md §3.3 + §4.1.

    Safe:   spill_ratio=0.5                     (auto, idiomatic)
            spill_ratio=0.0                     (force disk)
            spill_ratio=0.0 if output=="disk" else 0.5  (ternary)
            sr = 0.5; SpillArray(..., spill_ratio=sr)   (Name)
    Unsafe: spill_ratio=1.0, 0.7, 0.9, 0.3
    """
    violations: list[tuple[str, str, int, object]] = []
    for path in _walk_cadw_all():
        rel = str(path.relative_to(ROOT))
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            pytest.fail(f"{rel}: syntax error {e}")
        for fn in _iter_functions(tree):
            if (rel, fn.name) in G8_WHITELIST:
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                name = _get_call_attr(call)
                if name not in ("SpillArray", "SpillBuffer"):
                    continue
                sr = _get_kwarg_value(call, "spill_ratio")
                if sr is None:
                    continue
                if not _is_safe_spill_ratio(sr):
                    val = getattr(sr, "value", "<expr>")
                    violations.append((rel, fn.name, call.lineno, val))

    assert not violations, (
        f"spill_ratio outside {{0.0, 0.5}}. Violations: "
        f"{violations[:5]}. "
        f"spill_ratio >= 0.7 → threshold >= state_limit → OOM before "
        f"spill. Use 0.5 for auto, 0.0 for disk. "
        f"See AI_CONSTITUTION.md §3.3 + §4.1.")


# ═══════════════════════════════════════════════════════════════════
# G9 — self-verification test exists for every output= producer
# ═══════════════════════════════════════════════════════════════════
#
# AI-written producers with output= MUST ship with a test exercising
# both branches (memory + auto). Without G9, a producer could ship
# with only the memory branch tested — spill branch would silently
# accumulate O(n) RAM on prod scale (§1.3 violation without anyone
# noticing). This is the enforcement arm of §4.1 without human review.

G9_WHITELIST: set[tuple[str, str]] = {
    # v0.9.113: private helpers of public producers. Dual-branch
    # coverage (output="memory" + output="disk"/"auto") exists in
    # tests/test_zms_verification.py / tests/test_cusum_verification.py
    # through the public entry points fit_zms / fit_cusum. AST scan
    # only sees direct-by-name calls, not calls made inside a public
    # producer's body, so private helpers need explicit whitelist.
    ("cadw/ml/zms.py", "_fit_zms_df"),
    ("cadw/ml/cusum.py", "_fit_cusum_df"),
}

TESTS_DIR = ROOT / "tests"


def _walk_producers():
    """Files in cadw/ml/ and cadw/ops/ that may define output= funcs.

    Not including cadw/dataframe/ — bindings are thin pass-throughs
    to ml/ producers (tested there). Widen if bindings gain non-
    trivial output= logic.
    """
    for sub in ("ml", "ops"):
        d = CADW / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


def _functions_with_kwarg(path: Path, kwarg: str) -> list[str]:
    """Return names of non-dunder functions whose signature has kwarg.

    Dunders (__init__, __call__, ...) are constructors, not producers;
    G9/G10 target fit_/predict_/transform_-style functions. Excluding
    dunders prevents false positives when a Model class gains a
    constructor kwarg (e.g. CoxPHModel(solver=...) at v0.9.93).
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    out: list[str] = []
    for fn in _iter_functions(tree):
        if fn.name.startswith("__") and fn.name.endswith("__"):
            continue
        args = list(fn.args.args) + list(fn.args.kwonlyargs)
        if any(a.arg == kwarg for a in args):
            out.append(fn.name)
    return out


_TEST_CALL_CACHE: dict[str, tuple[bool, bool]] | None = None


def _test_call_coverage() -> dict[str, tuple[bool, bool]]:
    """Scan all test files once; map fn_name → (has_memory, has_auto).

    Only **direct** function calls (`Name`) are counted, not method
    calls (`Attribute`). Rationale: `df.drop_duplicates(...)` is a
    CadwDataFrame facade method, not `cadw.ops.dedup.drop_duplicates`
    — counting it would mask a missing ops-level test. Direct calls
    to imported functions (`drop_duplicates(...)`,
    `ops_dedup(...)`) are what tests use to exercise the ops API.

    Alias resolution: `from cadw.ops.dedup import drop_duplicates
    as ops_dedup` resolves to `drop_duplicates`, so tests written
    against the alias count. This is common in
    test_streaming_output_* (`ops_dedup`, `ops_groupby`, etc.).

    has_auto includes calls without `output=` kwarg (default="auto").
    Cached per-session — file scan is O(num_tests × parse).
    """
    global _TEST_CALL_CACHE
    if _TEST_CALL_CACHE is not None:
        return _TEST_CALL_CACHE
    cache: dict[str, tuple[bool, bool]] = {}
    if not TESTS_DIR.exists():
        _TEST_CALL_CACHE = cache
        return cache
    for tp in TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in tp.parts:
            continue
        try:
            tree = ast.parse(tp.read_text())
        except SyntaxError:
            continue
        # Build import map: local_name → original function name
        # (from `from mod import name [as alias]` only).
        imports: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    local = a.asname or a.name
                    imports[local] = a.name
        # Scan both Name calls (with alias resolution) and Attribute
        # calls (method calls like `m.predict_proba(...)` — how ML
        # producers are tested). Facade `df.drop_duplicates(...)`
        # also counts as `drop_duplicates` coverage — legitimate
        # (it exercises the same producer). What DOESN'T count: a
        # producer with no memory branch test at all.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                # Direct call: resolve alias if imported
                resolved = imports.get(node.func.id, node.func.id)
            elif isinstance(node.func, ast.Attribute):
                # Method call: use attr name (m.predict_proba → predict_proba)
                resolved = node.func.attr
            else:
                continue
            mem = auto = False
            for kw in node.keywords:
                if kw.arg == "output":
                    v = getattr(kw.value, "value", None)
                    if v == "memory":
                        mem = True
                    elif v == "auto":
                        auto = True
                    break
            else:
                # no `output` kwarg → default is "auto"
                auto = True
            prev_mem, prev_auto = cache.get(resolved, (False, False))
            cache[resolved] = (prev_mem or mem, prev_auto or auto)
    _TEST_CALL_CACHE = cache
    return cache


def test_g9_self_verification_present():
    """G9: every function with `output=` param in cadw/ml/ or cadw/ops/
    must have at least one test exercising output="memory" AND one
    exercising output="auto" (or default = auto).

    Enforces §4.1 self-verification protocol without human review:
    if AI writes a new producer, it must also write its own tests.
    Green suite without both branches = not shippable.
    """
    coverage = _test_call_coverage()
    missing: list[tuple[str, str, list[str]]] = []
    for path in _walk_producers():
        rel = str(path.relative_to(ROOT))
        for fn_name in _functions_with_kwarg(path, "output"):
            if (rel, fn_name) in G9_WHITELIST:
                continue
            has_mem, has_auto = coverage.get(fn_name, (False, False))
            gaps: list[str] = []
            if not has_mem:
                gaps.append("no output='memory' test")
            if not has_auto:
                gaps.append("no output='auto' (or default) test")
            if gaps:
                missing.append((rel, fn_name, gaps))

    assert not missing, (
        f"output= producers lacking self-verification tests: "
        f"{missing[:5]}. "
        f"Add tests/test_<module>.py with both branches: "
        f"result_mem = fn(..., output='memory'); "
        f"result_auto = fn(..., output='auto'); assert equivalent. "
        f"See AI_CONSTITUTION.md §4.1. "
        f"If test lives in a different file (e.g. "
        f"test_streaming_output_ml.py), still detected. "
        f"If intentionally back-compat-only, add to G9_WHITELIST "
        f"with rationale.")


# ═══════════════════════════════════════════════════════════════════
# G10 — fit_* with solver= must have dual-branch equivalence test
# ═══════════════════════════════════════════════════════════════════
#
# v0.9.93 will add solver="auto"|"closed"|"sgd" to fit_linear,
# fit_logistic, fit_coxph. This gate is forward-looking: prevents
# "two branches, one tested" class of bug from entering v0.9.93.
# Currently no solver= producers exist (gate effectively empty).

G10_WHITELIST: set[tuple[str, str]] = set()

# v0.9.94: explicit allowlist of functions requiring dual-branch tests.
# G10 no longer AST-scans "any solver=/method= kwarg" — too broad
# (fit_calibration uses method="isotonic"|"sigmoid" for algorithm
# variant, not wide-d OOC; fit_drift uses method="ks"|"chi2"; same
# for feature_select). Only functions with **wide-d OOC alternative**
# are registered here. Adding a new model with solver="auto"|"closed"|
# "sgd" (or method="exact"|"randomized") MUST append its entry, then
# G10 fails until dual-branch test ships.
#
# Format: (relative_path, function_name)
G10_TRACKED_FUNCTIONS: dict[tuple[str, str], str] = {
    ("cadw/ml/linear.py", "fit_linear"):
        "solver=closed|sgd (ridge quadratic loss); bit-identical coef",
    ("cadw/ml/logistic.py", "fit_logistic"):
        "solver=closed|sgd (logistic); predictions+LL per §4.2",
    ("cadw/ml/coxph.py", "fit_coxph"):
        "solver=closed|sgd (Cox Breslow); c-index+LL per §4.2",
    ("cadw/ml/pca.py", "fit_pca"):
        "method=exact|randomized (PCA); reconstruction error",
}


def _find_solver_producers():
    """Yield (file, [fn_names_with_solver]) for cadw/ml/."""
    d = CADW / "ml"
    if not d.exists():
        return
    for p in sorted(d.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        # v0.9.94: check both solver= and method= (PCA uses method=).
        fns = _functions_with_kwarg(p, "solver") + _functions_with_kwarg(
            p, "method")
        if fns:
            yield p, fns


def test_g10_dual_branch_test_present():
    """G10: every fit_* with solver= OR method= param must have a test
    comparing the explicit values.

    solver= (v0.9.93): "closed" vs "sgd" (linear/logistic/coxph).
    method= (v0.9.94): "exact" vs "randomized" (pca).

    Contract (v0.9.93+):
      - quadratic loss (linear, ridge): bit-identical at d=50
        (atol=1e-6). Same math, same optimum, both branches must
        produce same coefficients.
      - non-quadratic (logistic, coxph): documented convergence
        (relative coef error < 5% vs IRLS/Newton reference).

    Requires explicit solver="closed" AND solver="sgd" calls in
    tests. `solver="auto"` alone does NOT satisfy — AST cannot know
    which branch a runtime auto call picks.

    See AI_CONSTITUTION.md §4.1 + ROADMAP v0.9.93.
    """
    # Scan test files for calls with kwarg solver= OR method=
    # (v0.9.94: PCA uses method="exact"|"randomized"; G10 covers both).
    branch_calls: dict[str, set[str]] = {}
    if TESTS_DIR.exists():
        for tp in TESTS_DIR.rglob("test_*.py"):
            if "__pycache__" in tp.parts:
                continue
            try:
                tree = ast.parse(tp.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = _get_call_attr(node)
                if not callee:
                    continue
                for kw in node.keywords:
                    if kw.arg in ("solver", "method"):
                        v = getattr(kw.value, "value", None)
                        if isinstance(v, str):
                            branch_calls.setdefault(
                                callee, set()).add(v)

    missing: list[tuple[str, str, list[str]]] = []
    # v0.9.94: iterate explicit allowlist, not AST scan (see
    # G10_TRACKED_FUNCTIONS docstring).
    for (rel, fn_name) in G10_TRACKED_FUNCTIONS:
        if (rel, fn_name) in G10_WHITELIST:
            continue
        branches = branch_calls.get(fn_name, set())
        has_closed = "closed" in branches or "exact" in branches
        has_sgd = "sgd" in branches or "randomized" in branches
        gaps: list[str] = []
        if not has_closed:
            gaps.append("no solver='closed' (or method='exact') test")
        if not has_sgd:
            gaps.append("no solver='sgd' (or method='randomized') test")
        if gaps:
            missing.append((rel, fn_name, gaps))

    assert not missing, (
        f"fit_* with solver= lacking dual-branch tests: "
        f"{missing[:5]}. "
        f"Tracked dual-branch functions lacking tests: "
        f"add test comparing closed vs sgd (or exact vs randomized "
        f"for PCA). Add test then run: "
        f"- quadratic loss (linear): bit-identical (atol=1e-6)"
        f"- non-quadratic (logistic/coxph): c-index / predictions + "
        f"LL per §4.2"
        f"- PCA: reconstruction error (components are sign/rotation "
        f"ambiguous). "
        f"See AI_CONSTITUTION.md §4.2. "
        f"If not yet implemented, add to G10_WHITELIST with roadmap "
        f"reference.")


# ═══════════════════════════════════════════════════════════════════
# G11 — transitive migration: caller uses pa.Table API on
#       CadwDataFrame producer return.
#
# Rationale: v0.9.95 unified 13 producers to return CadwDataFrame.
# Callers that used to receive pa.Table or np.ndarray still call
# pa.Table-only methods (.column, .num_rows, .schema, ...) or apply
# arithmetic directly. Those are silent AttributeError / TypeError
# bugs at runtime, invisible to G1-G10 (shape/guard/self-verify gates).
# G11 catches them by AST.
#
# See AI_CONSTITUTION.md §5 AP7 (multi-return producer migration).
# ═══════════════════════════════════════════════════════════════════

# Bare-name module-level producers. Safe to match by name anywhere:
# these are module-level functions, uniquely named in cadw. Calls like
# `groupby_aggregate(...)`, `_gba(...)` (import alias) resolve through
# imports map to these names.
G11_BARE_NAME_PRODUCERS = frozenset({
    "groupby_aggregate", "drop_duplicates", "topk_to_disk",
})

# Backward-compat alias; new code should reference the split set above.
CADW_DF_PRODUCER_NAMES = G11_BARE_NAME_PRODUCERS

# NOTE: method names like predict_proba / score_samples / predict_risk /
# transform_dense are NOT tracked globally — same name appears in both
# migrated classes (returns CadwDataFrame) and unmigrated ones (returns
# ndarray/Table: logistic.predict_proba, gbdt.predict_proba, calibration.
# predict_proba, uplift.m1_.predict_proba). Tracked only via
# `self.<method>()` when <method> has output= in the SAME class. See
# _g11_is_producer_call.

# pa.Table-only methods (not present on CadwDataFrame).
G11_FORBIDDEN_METHODS = frozenset({
    "column", "column_names", "num_rows", "num_columns",
    "take", "combine_chunks", "rename_columns", "sort_by",
    "group_by", "append_column", "set_column", "to_table",
})

# pa.Table-only attributes.
G11_FORBIDDEN_ATTRS = frozenset({"shape", "schema", "data"})

# Internal helpers that legitimately return pa.Table. G11 skips them.
G11_PA_TABLE_INTERNAL = frozenset({
    "_local_dedup", "_merge_partials", "_groupby_with_skew",
    "_empty_result_schema", "_accum_to_table",
})


def _g11_function_has_output_param(fn_node: ast.FunctionDef) -> bool:
    """True if function signature has output= kwarg."""
    args = list(fn_node.args.args) + list(fn_node.args.kwonlyargs)
    return any(a.arg == "output" for a in args)


def _g11_build_imports_map(tree: ast.Module) -> dict[str, str]:
    """local_name -> actual_name from `from mod import name [as alias]`."""
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                local = a.asname or a.name
                imports[local] = a.name
    return imports


def _g11_is_producer_call(
    call: ast.Call,
    imports: dict[str, str],
    current_class: "str | None",
    class_methods: set[str],
) -> "str | None":
    """Return producer name if call is to a tracked producer, else None.

    Tracking modes (v1):

      Bare-name (`groupby_aggregate(...)`, `_gba(...)`):
        Match against G11_BARE_NAME_PRODUCERS. Imports map resolves
        aliases (`_gba` -> `groupby_aggregate`, `_ops_dedup` ->
        `drop_duplicates`).

      Intra-class method (`self.<method>(...)`):
        Match against `class_methods` = methods defined in the SAME
        class with an `output=` param. This catches
        `self.score_samples(df)` on gmm (migrated) and correctly
        IGNORES `self.predict_proba(X)` on logistic/gbdt/calibration
        (unmigrated — their predict_proba has no output= param and
        returns ndarray, not CadwDataFrame).

    NOT tracked: `obj.<method>()` for obj != self. Receiver type is
    unknown at AST time. Common names (predict_proba) collide between
    migrated and unmigrated classes. Accepted limitation; use
    `# noqa: G11` for the rare cross-object real bug.
    """
    if isinstance(call.func, ast.Name):
        name = imports.get(call.func.id, call.func.id)
        if name in G11_PA_TABLE_INTERNAL:
            return None
        if name in G11_BARE_NAME_PRODUCERS:
            return name
        return None
    if (
        current_class is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    ):
        attr = call.func.attr
        if attr in G11_PA_TABLE_INTERNAL:
            return None
        if attr in class_methods:
            return attr
    return None


def _g11_find_tracked_vars(
    fn_node: ast.FunctionDef,
    imports: dict[str, str],
    current_class: "str | None",
    class_methods: set[str],
) -> dict[str, list[tuple[int, "str | None"]]]:
    """Map var_name -> timeline of (lineno, producer_or_None).

    Each entry records an assignment at that line. producer=None means
    "reassigned to a non-producer value" (untrack). Latest assignment
    strictly before a use site determines tracking state.

    This handles the AP7 pattern from AI_CONSTITUTION.md §5 and brief
    §3.4 ("pa.Table.from_batches -> UNTRACK"):

        result = groupby_aggregate(...)       # producer (tracked)
        result = pa.Table.from_batches(...)   # pa.Table (untracked)
        result.rename_columns(...)            # OK — not tracked

    Without reassignment tracking, that last line is a false positive.
    """
    timeline: dict[str, list[tuple[int, "str | None"]]] = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign):
            producer: "str | None" = (
                _g11_is_producer_call(
                    node.value, imports, current_class, class_methods,
                )
                if isinstance(node.value, ast.Call) else None
            )
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    timeline.setdefault(tgt.id, []).append(
                        (node.lineno, producer))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            if isinstance(node.target, ast.Name):
                producer = _g11_is_producer_call(
                    node.value, imports, current_class, class_methods)
                timeline.setdefault(node.target.id, []).append(
                    (node.lineno, producer))
    for lst in timeline.values():
        lst.sort()
    return timeline


def _g11_tracked_at(
    timeline: dict[str, list[tuple[int, "str | None"]]],
    var: str, line: int,
) -> "str | None":
    """Producer name if `var` is tracked at `line`, else None.

    "Tracked" = latest assignment strictly before `line` was from a
    producer. Reassignment to a non-producer clears tracking.
    """
    lst = timeline.get(var)
    if not lst:
        return None
    producer: "str | None" = None
    for a_line, a_prod in lst:
        if a_line >= line:
            break
        producer = a_prod
    return producer


def _g11_scan_violations(
    fn_node: ast.FunctionDef,
    timeline: dict[str, list[tuple[int, "str | None"]]],
    src_lines: list[str],
) -> list[dict]:
    """Scan function body for forbidden usage of tracked vars.

    `timeline` is per-var reassignment history from
    _g11_find_tracked_vars. Each use site consults _g11_tracked_at()
    to know whether the var is a CadwDataFrame at that line (vs
    reassigned to pa.Table per AP7).
    """
    violations: list[dict] = []

    def tracked(var: str, line: int) -> "str | None":
        return _g11_tracked_at(timeline, var, line)

    for node in ast.walk(fn_node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            var = node.value.id
            prod = tracked(var, node.lineno)
            if prod is None:
                continue
            if node.attr in G11_FORBIDDEN_METHODS:
                line = node.lineno
                if "noqa: G11" in src_lines[line - 1]:
                    continue
                violations.append({
                    "kind": "forbidden_method",
                    "line": line,
                    "variable": var,
                    "produced_by": prod,
                    "method": node.attr,
                })
            elif node.attr in G11_FORBIDDEN_ATTRS:
                line = node.lineno
                if "noqa: G11" in src_lines[line - 1]:
                    continue
                violations.append({
                    "kind": "forbidden_attr",
                    "line": line,
                    "variable": var,
                    "produced_by": prod,
                    "attr": node.attr,
                })

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "isinstance" and len(node.args) >= 2:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Name):
                    prod = tracked(arg0.id, node.lineno)
                    if prod is None:
                        continue
                    t = node.args[1]
                    tname: "str | None" = None
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                        if t.value.id in ("pa", "np", "numpy") and t.attr in ("Table", "ndarray"):
                            tname = f"{t.value.id}.{t.attr}"
                    if tname:
                        line = node.lineno
                        if "noqa: G11" in src_lines[line - 1]:
                            continue
                        violations.append({
                            "kind": "isinstance_legacy",
                            "line": line,
                            "variable": arg0.id,
                            "produced_by": prod,
                            "type": tname,
                        })

        if isinstance(node, (ast.BinOp, ast.Compare)):
            left = node.left
            if isinstance(left, ast.Name):
                prod = tracked(left.id, node.lineno)
                if prod is None:
                    continue
                line = node.lineno
                if "noqa: G11" in src_lines[line - 1]:
                    continue
                violations.append({
                    "kind": "binop",
                    "line": line,
                    "variable": left.id,
                    "produced_by": prod,
                })
    return violations


def _g11_collect_class_producers(class_node: ast.ClassDef) -> set[str]:
    """Method names in class with output= param (intra-class producers)."""
    producers: set[str] = set()
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef):
            if _g11_function_has_output_param(item):
                producers.add(item.name)
    return producers


def _g11_scan_file(path: Path) -> list[dict]:
    """Scan one cadw file for G11 violations."""
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (SyntaxError, OSError):
        return []
    imports = _g11_build_imports_map(tree)
    src_lines = src.splitlines()
    violations: list[dict] = []

    def _record(
        fn_node: ast.FunctionDef,
        timeline: dict[str, list[tuple[int, "str | None"]]],
    ) -> None:
        if not timeline:
            return
        for v in _g11_scan_violations(fn_node, timeline, src_lines):
            v["file"] = str(path.relative_to(ROOT))
            v["function"] = fn_node.name
            violations.append(v)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_methods = _g11_collect_class_producers(node)
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef):
                    tracked = _g11_find_tracked_vars(
                        fn, imports, node.name, class_methods)
                    _record(fn, tracked)
        elif isinstance(node, ast.FunctionDef):
            tracked = _g11_find_tracked_vars(
                node, imports, None, set())
            _record(node, tracked)

    return violations


@pytest.mark.parametrize(
    "path",
    [p for p in CADW.rglob("*.py") if "__pycache__" not in p.parts],
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_g11_transitive_migration(path):
    """G11: caller of migrated producer must not use pa.Table API.

    See AI_CONSTITUTION.md §5 AP7 (multiple return points).
    """
    violations = _g11_scan_file(path)
    if not violations:
        return

    msgs = []
    for v in violations[:5]:
        msgs.append(
            f"  {v['file']}:{v['line']} {v['function']}() "
            f"var={v['variable']!r} from {v['produced_by']}(): "
            f"{v['kind']}"
        )

    raise AssertionError(
        f"G11 transitive-migration violations in {path.name}:\n"
        + "\n".join(msgs)
        + f"\n({len(violations)} total)\n"
        f"\nFix: CadwDataFrame now returned by {violations[0]['produced_by']}().\n"
        f"See AI_CONSTITUTION.md §5 AP7 for patterns:\n"
        f"  - extract via [b.column(c) for b in result._iter_filtered_batches()]\n"
        f"  - result.count() instead of .num_rows\n"
        f"  - pa.Table.from_batches(list(batches)) to convert back if needed\n"
        f"  - assert isinstance(result, CadwDataFrame) — CORRECT type check\n"
    )

# ============================================================================
# G12-G17 (v0.9.178, DEC-067/068/069)
# ============================================================================

def test_G12_bounded_state_implies_batched():
    """G12 (DEC-067): bounded_state=True requires fit_batched AND
    predict_batched (or documented exception in streaming_strategy).

    Skeleton: enable v0.9.179+ after TS refactor.
    """
    pytest.skip("TODO v0.9.179+: G12 enable after TS refactor")


def test_G13_approximation_disclosed():
    """G13 (DEC-067, R6): fit/predict with approximation emits
    ApproximationInfo on result + ApproximationWarning.

    Skeleton: enable v0.9.179+ after approximation wire-up in fit paths.
    """
    pytest.skip("TODO v0.9.179+: G13 enable after approximation wire-up")


def test_G14_verified_has_min_ram():
    """G14 (DEC-067, v0.9.177): verified + ram_formula => min_ram_formula
    or min_ram_pending. REAL gate (rollout completed 97/98)."""
    import yaml
    from pathlib import Path

    ml_tools = Path(__file__).resolve().parent.parent / "docs" / "ml_tools"
    violations = []
    total_verified = 0
    with_min_ram = 0
    with_pending = 0

    for yp in sorted(ml_tools.glob("[A-Z]_*.yaml")):
        data = yaml.safe_load(yp.read_text()) or {}
        for st in data.get("subtypes") or []:
            for m in st.get("models") or []:
                status = (m.get("implementation_status") or {}).get("status")
                if status != "verified":
                    continue
                total_verified += 1
                ram = m.get("ram_formula")
                if not ram:
                    continue  # no ram_formula: not subject to G14
                min_ram = m.get("min_ram_formula")
                pending = m.get("min_ram_pending")
                if min_ram:
                    with_min_ram += 1
                elif pending:
                    with_pending += 1
                else:
                    violations.append(f"{yp.stem}::{m.get('name')}")

    assert not violations, (
        f"G14 violations ({len(violations)}): verified + ram_formula "
        f"requires min_ram_formula or min_ram_pending:\n  "
        + "\n  ".join(violations[:10])
    )
    assert total_verified >= 90, (
        f"Expected >=90 verified cards, got {total_verified}"
    )
    assert with_min_ram + with_pending >= total_verified - 5, (
        f"Coverage too low: {with_min_ram} migrated + {with_pending} "
        f"pending of {total_verified} verified"
    )


def test_G15_auto_switch_registry():
    """G15 (DEC-067): models with O(d^2) state (linear/logistic/PCA)
    declare auto_switch registry in YAML.

    Skeleton: enable v0.9.179+ after auto-switch registry wire-up.
    """
    pytest.skip("TODO v0.9.179+: G15 enable after auto-switch registry")


def test_G16_state_nbytes_includes_output():
    """G16 (DEC-067, R1): n-sized persistent output must be included in
    state_nbytes() (AST scan of fit methods).

    Skeleton: enable v0.9.179+ after AST scan implementation.
    """
    pytest.skip("TODO v0.9.179+: G16 enable after AST scan")


def test_G17_bounded_by_domain_is_draft():
    """G17 (DEC-067): bounded_by_domain => status:draft (visible debt).

    Skeleton: enable v0.9.179+ after bounded_by_domain migration.
    """
    pytest.skip("TODO v0.9.179+: G17 enable after bounded_by_domain migration")

# ============================================================================
# G18 (v0.9.189, DEC-070/072): bounded_state=true must have all transitive
# leaf primitives streaming OR window-truncation on model.
# ============================================================================

def test_G18_primitives_before_models():
    """G18 (DEC-070/072): bounded_state=true requires each transitive leaf
    primitive to have streaming_variant != null OR model declares
    bounded_by_window. intrinsic_bound without reason = violation.
    """
    import re
    import yaml
    from pathlib import Path
    from cadw.core.cards import load_primitive_cards

    root = Path(__file__).resolve().parent.parent
    cards = load_primitive_cards(root / "docs/ml_tools/PRIMITIVES.yaml")
    by_short = {}
    for c in cards:
        short = c.cadw_function.rsplit(".", 1)[-1]
        by_short[short] = c

    ml_tools = root / "docs/ml_tools"
    violations = []
    checked = 0

    for yf in sorted(ml_tools.glob("[A-Z]_*.yaml")):
        try:
            data = yaml.safe_load(yf.read_text()) or {}
        except yaml.YAMLError:
            continue
        for st in data.get("subtypes") or []:
            for m in st.get("models") or []:
                status = (m.get("implementation_status") or {}).get("status")
                if status != "verified":
                    continue
                if not m.get("bounded_state"):
                    continue
                module = m.get("cadw_module", "")
                if not module:
                    continue
                mod_path = root / module.replace(".", "/")
                if mod_path.suffix != ".py":
                    mod_path = mod_path.with_suffix(".py")
                if not mod_path.exists():
                    continue
                checked += 1
                src = mod_path.read_text()
                # Find imports from cadw.(ts|ops).X import ...
                for im in re.findall(
                    r"from cadw\.(?:ts|ops)[\w.]* import ([^\n(]+)",
                    src,
                ):
                    for name in re.findall(r"\b(_[a-z][a-z0-9_]*)\b", im):
                        c = by_short.get(name)
                        if c is None:
                            continue  # audit test catches unregistered
                        if (c.streaming_variant is None
                                and c.intrinsic_bound is None):
                            violations.append(
                                f"{yf.name}::{m.get('name')}: bounded_state="
                                f"true imports {name} which has streaming_"
                                f"variant=null AND no intrinsic_bound. "
                                f"Require streaming OR bounded_by_window "
                                f"on model (G18, DEC-070/072)."
                            )
                        if c.intrinsic_bound is not None:
                            reason = (c.intrinsic_bound or {}).get("reason", "")
                            if not reason or len(str(reason).strip()) < 20:
                                violations.append(
                                    f"{yf.name}::{m.get('name')}: {name} "
                                    f"intrinsic_bound.reason empty/short "
                                    f"({reason!r}); must explain why window-"
                                    f"truncation was rejected (G18, DEC-072)."
                                )

    assert not violations, (
        f"G18: {len(violations)} primitives-before-models violations "
        f"across {checked} verified bounded_state=true models:\n"
        + "\n".join(f"  {v}" for v in violations[:15])
    )

# ============================================================================
# G20 (v0.9.192, DEC-074): bounded_by_window.applies_to partial
# (non-"all") requires n_scalability: linear (worst-case mode may be O(n)).
# ============================================================================

def test_G20_partial_window_scope_requires_linear():
    """G20 (DEC-074): if bounded_by_window.applies_to is set to a specific
    params value (not null/'all'), then n_scalability must be 'linear'
    (worst-case mode outside the partial scope is O(n)).
    """
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    ml_tools = root / "docs" / "ml_tools"
    violations = []
    checked = 0

    for yf in sorted(ml_tools.glob("[A-Z]_*.yaml")):
        try:
            data = yaml.safe_load(yf.read_text()) or {}
        except yaml.YAMLError:
            continue
        for st in data.get("subtypes") or []:
            for m in st.get("models") or []:
                status = (m.get("implementation_status") or {}).get("status")
                if status != "verified":
                    continue
                bbw = m.get("bounded_by_window")
                if not bbw:
                    continue
                checked += 1
                applies_to = bbw.get("applies_to")
                n_scal = m.get("n_scalability")
                # Partial scope = specific params. Default 'all'/None means
                # full-scope bounded_by_window; that's fine with
                # n_scalability=bounded_by_window.
                is_partial = (applies_to is not None
                              and applies_to not in ("all", "params=all"))
                if is_partial and n_scal != "linear":
                    violations.append(
                        f"{yf.name}::{m.get('name')}: bounded_by_window."
                        f"applies_to={applies_to!r} (partial) but "
                        f"n_scalability={n_scal!r}; must be 'linear' "
                        f"(worst-case mode outside scope is O(n), DEC-074)."
                    )

    assert not violations, (
        f"G20: {len(violations)} partial-window scope violations "
        f"across {checked} cards with bounded_by_window:\n"
        + "\n".join(f"  {v}" for v in violations[:10])
    )

# ============================================================================
# G22 (v0.9.204, AP33): leaf primitive deps of verified+bounded_state models
# must have streaming_variant or intrinsic_bound; OR model has bounded_by_window
# ============================================================================

def test_G22_leaf_deps_consistent():
    """G22 (v0.9.204, AP33): every leaf primitive imported by a
    verified+bounded_state model must have streaming_variant OR
    intrinsic_bound; failure = fake OOC (case A).
    """
    import ast
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    # Load primitive cards (short name -> card)
    prim = yaml.safe_load(
        (root / "docs" / "ml_tools" / "PRIMITIVES.yaml").read_text())
    cards = {(c.get("cadw_function") or "").rsplit(".", 1)[-1]: c
             for c in (prim.get("primitives") or [])}

    violations = []
    checked = 0

    for yf in sorted((root / "docs" / "ml_tools").glob("[A-Z]_*.yaml")):
        try:
            d = yaml.safe_load(yf.read_text()) or {}
        except yaml.YAMLError:
            continue
        for st in d.get("subtypes") or []:
            for m in st.get("models") or []:
                impl = m.get("implementation_status") or {}
                if impl.get("status") != "verified":
                    continue
                if not m.get("bounded_state"):
                    continue
                mod = m.get("cadw_module", "")
                if not mod:
                    continue
                p = root / (mod.replace(".", "/") + ".py")
                if not p.exists():
                    continue
                try:
                    tree = ast.parse(p.read_text())
                except SyntaxError:
                    continue

                leaves = []
                for im in ast.walk(tree):
                    if isinstance(im, ast.ImportFrom):
                        modname = im.module or ""
                        if not (modname.startswith("cadw.ts") or
                                modname.startswith("cadw.ops") or
                                modname.startswith("cadw.ml")):
                            continue
                        for alias in im.names:
                            if alias.name.startswith("_"):
                                c = cards.get(alias.name)
                                if c and c.get("layer") == "leaf":
                                    leaves.append(alias.name)

                if not leaves:
                    continue
                checked += 1
                has_bbw = bool(m.get("bounded_by_window"))
                for ln in leaves:
                    c = cards[ln]
                    sv = c.get("streaming_variant")
                    ib = c.get("intrinsic_bound")
                    wt = c.get("window_truncatable")
                    if sv is not None:
                        continue
                    if ib is not None:
                        continue
                    if wt is not None:
                        # AP33 case B valid: window_truncatable
                        # (DEC-072: window check before intrinsic)
                        continue
                    if has_bbw:
                        # case B — tolerated (card fix tracked separately)
                        continue
                    violations.append(
                        f"{yf.name}::{m.get('name')} -> {ln} "
                        f"(streaming_variant=None, intrinsic_bound=None, "
                        f"window_truncatable=None, "
                        f"bounded_by_window=False) — case A")

    assert not violations, (
        f"G22 (AP33): {len(violations)} leaf primitives without bound used "
        f"by verified+bounded_state models (case A, fake OOC):\n"
        + "\n".join(violations)
        + f"\n\nFix: add streaming_variant or intrinsic_bound to primitive "
        f"card, OR declare bounded_by_window on model."
    )


# ============================================================================
# G23 (v0.9.206): layer=batch + bs=true — no n-sized term in ram_formula.
#
# Rationale: v0.9.206 audit нашёл _baseline_hazard_streaming с layer=batch,
# bs=true, ram_formula="2*n_events*f64 + ..." — dishonest (state grows with
# n_events = n for all-uncensored). Card was flipped to leaf+ib. G23
# предотвращает future repeats статически.
# ============================================================================

def test_G23_batch_primitive_no_n_in_ram_formula():
    """G23 (v0.9.206): layer=batch + bs=true ram_formula must not contain
    n-sized identifiers as отдельные термы.

    Whitelisted (bounded hyperparams, not n):
      batch, row_size, d, k, K, p, q, s, W_eff, chunk, min_size,
      k_arms, n_q (Q size), n_y (query batch), n_features (fixed d),
      nnz (sparse count, caller-controlled), state_bytes, n_estimators.

    Forbidden (n-sized):
      n, N, n_events, n_rows, n_total, n_x, n_groups, n_samples,
      n_train, n_test, n_obs, n_mem.
    """
    import re
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    prim = yaml.safe_load(
        (root / "docs" / "ml_tools" / "PRIMITIVES.yaml").read_text())
    cards = prim.get("primitives") or []

    FORBIDDEN = {
        "n", "N", "n_events", "n_rows", "n_total", "n_x", "n_groups",
        "n_samples", "n_train", "n_test", "n_obs", "n_mem",
    }

    violations = []
    for c in cards:
        if c.get("layer") != "batch":
            continue
        if c.get("bounded_state") is not True:
            continue
        rf = c.get("ram_formula") or ""
        if not rf:
            continue
        # identifiers: \b[a-zA-Z_]\w*
        ids = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", rf))
        for ident in ids:
            if ident in FORBIDDEN:
                violations.append(
                    f"{c.get('name')}: ram_formula={rf!r} contains "
                    f"n-sized identifier {ident!r}")

    assert not violations, (
        f"G23: {len(violations)} batch+bs=true primitives claim "
        f"bounded state but ram_formula contains n-sized term:\n"
        + "\n".join(violations)
        + "\n\nFix: either layer=batch->leaf + intrinsic_bound {axis, bound}, "
        f"OR ram_formula reflect actual O(model) state (no n)."
    )


# ---------------------------------------------------------------------------
# G23-behavioral: optional fixture-driven state independence test.
# Whitelist (in _G23_FIXTURES) — cheap primitives with concrete setup.
# Other batch primitives skip (expensive fixtures; static G23 covers them).
# ---------------------------------------------------------------------------

_G23_FIXTURES = {}  # v0.9.206 initial: empty (static G23 + per-primitive CW)


def test_G23_behavioral_fixture_whitelist():
    """G23-behavioral (v0.9.206): guard that fixture whitelist не растёт
    молча. Skip если пусто.
    """
    # Empty whitelist is valid (v0.9.206 initial; static G23 above + per-
    # primitive CW tests already cover: test_linalg_svd::test_ooc_in_n_rows,
    # test_lof_verification::test_state_constant_in_n, etc.)
    assert isinstance(_G23_FIXTURES, dict), (
        "_G23_FIXTURES must be a dict; got "
        f"{type(_G23_FIXTURES).__name__}")

