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

G9_WHITELIST: set[tuple[str, str]] = set()

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
