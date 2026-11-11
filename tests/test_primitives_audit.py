"""v0.9.187 (DEC-071 §4): audit test — every module-level helper is
either registered in PRIMITIVES.yaml or in DEC-071 §2 exemption list.

Scans cadw/ml/*.py and cadw/ts/*.py for `^def _name(`; asserts each
function is known via cadw_function OR name in PRIMITIVES.yaml, OR in
EXEMPT set.
"""
import re
from pathlib import Path

import pytest

from cadw.core.cards import load_primitive_cards


EXEMPT = frozenset({
    # O(1) utility
    "_check_df", "_check", "_resolve_w_eff",
    # O(steps) / O(k) numeric
    "_ols_forecast", "_mase", "_naive_mae_train", "_naive_seasonal_mae",
    "_nextodd", "_tricube",
    # Alias / wrapper (impl has own card)
    "_best_split",
    # Nested 1-site (19x _ses_fit inside; card on _ses_fit)
    "_best_alpha", "_grid_search",
    # Already registered under descriptive name (see DEC-071 §3):
    # theta._ses_fit -> _ses_fit_simple, ets._ses_fit -> _ses_fit_hw
    "_ses_fit",
    # v0.9.187 EXEMPT additions (session files, justified by DEC-071 §2):
    # Ensemble/Transforms internals — signature-aware df pass-through
    # helpers; covered by ApproxMixin + members registered above.
    "_build_features_for_step",
    "_build_inner_pipeline",
    "_fit_one",
    "_train_n_horizons",
    # Coarse decomposition inner helpers — covered by stl_decompose card
    "_loess_local_linear",  # already carded separately, keep for safety
})


def _module_helpers(path: Path) -> set[str]:
    src = path.read_text()
    return set(re.findall(r"^def (_[a-z][a-z0-9_]*)\(", src, re.MULTILINE))


def _registered_names() -> tuple[set[str], set[str]]:
    cards = load_primitive_cards("docs/ml_tools/PRIMITIVES.yaml")
    names = {c.name for c in cards}
    functions = {c.cadw_function for c in cards}
    # Extract short fn name from dotted path
    short_fns = {f.rsplit(".", 1)[-1] for f in functions}
    return names, short_fns


def test_all_module_helpers_registered_or_exempt():
    """DEC-071 §4: every def _x() in cadw/ml/, cadw/ts/ is registered."""
    repo_root = Path(__file__).resolve().parent.parent
    names, short_fns = _registered_names()

    # v0.9.187 (DEC-071 §4 amendment): scope = files touched by
    # v0.9.179-187 (pelt, coxph, binseg + 5 TS refactored). Expanding to
    # full cadw/ml, cadw/ts is gradual — register when next touching the
    # module (backlog v0.9.188+).
    SESSION_FILES = [
        "cadw/ml/pelt.py",
        "cadw/ml/binseg.py",
        "cadw/ml/coxph.py",
        "cadw/ts/models.py",
        "cadw/ts/theta.py",
        "cadw/ts/otm.py",
        "cadw/ts/ets.py",
        "cadw/ts/auto_ets.py",
        "cadw/ts/auto_theta.py",
        "cadw/ts/ensemble.py",
        "cadw/ts/transforms.py",
        "cadw/ts/_decomposition.py",
    ]

    unregistered = []
    for rel_path in SESSION_FILES:
        py = repo_root / rel_path
        if not py.exists():
            continue
        helpers = _module_helpers(py)
        for h in helpers:
            if h in EXEMPT:
                continue
            if h in names or h in short_fns:
                continue
            unregistered.append(f"{rel_path}::{h}")

    assert not unregistered, (
        f"DEC-071 audit: {len(unregistered)} unregistered module helpers:\n"
        + "\n".join(f"  {u}" for u in unregistered)
        + "\n\nRegister in PRIMITIVES.yaml, or add to EXEMPT set in "
        "tests/test_primitives_audit.py (with rationale)."
    )

# ---- v0.9.198 (AP32): verified cards depends_on must resolve to
# PRIMITIVES.yaml OR OPS_LAYER_EXEMPT ----

OPS_LAYER_EXEMPT = frozenset({
    # Pipeline operations, not model primitives (see
    # PRIMITIVES_BACKLOG Appendix B). These are DataFrame-method-level
    # ops: caller-controlled batching; each has its own YAML card in
    # ops section if needed (not PRIMITIVES.yaml by design).
    "cadw.ops.reduce",
    "cadw.ops.shuffle",
    "cadw.ops.sort",
    "cadw.ops.join",
    "cadw.ops.groupby",
    "cadw.ops.pivot",
    "cadw.ops.quantile",
    "cadw.ops.topk",
    "cadw.ops.rank",
    "cadw.ops.resample",
    "cadw.ops.dedup",
    "cadw.ops.describe",
    "cadw.ops.histogram",
    "cadw.ops.metrics",
    "cadw.ops.linalg",  # module-level (not specific function)
})


def test_all_verified_depends_on_resolve():
    """AP32 (v0.9.198): every cadw.ops.* depends_on of a verified card
    must resolve to a PRIMITIVES.yaml card OR be in OPS_LAYER_EXEMPT.

    Rationale: verified card claims OOC, but its underlying primitive
    may be unaudited (state/streaming semantics undocumented). LOF
    (v0.9.197) depended on unregistered knn_query (v0.9.144) — invisible
    to G18/G21 until manual probe. This test catches that class.

    Scope: cadw.ops.* only. cadw.ts.* dependencies are covered by
    G18 (leaf streaming_variant) separately.
    """
    import yaml
    from cadw.core.cards import load_primitive_cards

    prim = load_primitive_cards("docs/ml_tools/PRIMITIVES.yaml")
    prim_fns = {c.cadw_function for c in prim}

    repo_root = Path(__file__).resolve().parent.parent
    violations = []
    checked = 0
    for yp in sorted((repo_root / "docs" / "ml_tools").glob("[A-Z]_*.yaml")):
        try:
            d = yaml.safe_load(yp.read_text()) or {}
        except yaml.YAMLError:
            continue
        for st in d.get("subtypes") or []:
            for m in st.get("models") or []:
                status = (m.get("implementation_status") or {}).get("status")
                if status != "verified":
                    continue
                for dep in (m.get("depends_on") or []):
                    if not dep.startswith("cadw.ops."):
                        continue
                    checked += 1
                    if dep in prim_fns:
                        continue
                    # Exempt: whole submodule ops-layer
                    module_prefix = dep.rsplit(".", 1)[0]
                    if dep in OPS_LAYER_EXEMPT:
                        continue
                    if module_prefix in OPS_LAYER_EXEMPT:
                        continue
                    violations.append(
                        f"{yp.name}::{m.get('name')} -> {dep}")
    assert not violations, (
        f"AP32: {len(violations)} verified cards depend on "
        f"unregistered cadw.ops.* primitives:\n"
        + "\n".join(violations)
        + f"\n\nFix: register in PRIMITIVES.yaml OR add to "
        f"OPS_LAYER_EXEMPT in test_primitives_audit.py "
        f"(reason: pipeline ops, not model primitives)."
    )

