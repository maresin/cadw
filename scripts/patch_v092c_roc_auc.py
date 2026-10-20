#!/usr/bin/env python3
"""v0.9.2c: roc_auc (binary + macro) + multiclass log_loss.

SCOPE:
  - cadw/ops/metrics.py:
    * MetricsOp: multiclass log_loss (y_proba_cols: list[str], K>1)
    * module functions _compute_roc_auc_binary / _compute_roc_auc_macro
    * AUC_METRICS = frozenset({"roc_auc"})
  - cadw/dataframe.py::metrics: dispatch (roc_auc separate from ReduceOp)
  - cadw/__init__.py: AUC_METRICS export
  - tests/_helpers/metrics_reference.py: expected_roc_auc (per-threshold)
  - tests/test_metrics_reference.py: helper self-consistency
  - tests/test_metrics.py: +roc_auc +multiclass log_loss (~20 fast)
  - tests/test_metrics_e2e.py: +1 slow (100K roc_auc + multiclass)

DESIGN (locked-in):
  - roc_auc is NOT a pure Reduce — Sort-based, two-pass:
      Pass 1: count n_pos, n_neg (per batch)
      external_sort by y_proba desc
      Pass 2: stream sorted, per-threshold grouping, trapezoid
    Peak RAM = batch + external_sort disk. Independent of N.
  - Multiclass log_loss: y_proba_cols list, y_true in classes. O(1) state.
  - y_proba API: str → binary, list[str] → multiclass (both metrics).
  - roc_auc binary: y_true ∈ {0,1}, pos_label=1 default.
  - roc_auc macro: y_proba list K + classes explicit, ovr K binary passes.
  - Conflict roc_auc + regression on same y_true → ValueError.
  - Fail-fast not needed for roc_auc: Sort already handles N via disk.

PROTOTYPE (§13.4/123, /166): scripts/proto_roc_auc.py PASS 8/8.
  Validated: per-threshold grouping (ties), NaN on all-pos/all-neg.

GUARDS: metrics + reduce + full suite + guards.

Run: uv run python scripts/patch_v092c_roc_auc.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. tests/_helpers/metrics_reference.py ============

REFERENCE = '''"""Reference metrics for tests (per-threshold ROC-AUC, log-loss).

Validated by prototype scripts/proto_roc_auc.py (PASS 8/8). Manual
arithmetic in test bodies prohibited (§13.4/166); use these helpers
instead.
"""
from __future__ import annotations

import numpy as np


def expected_roc_auc(y_true, y_score) -> float:
    """Per-threshold grouping ROC-AUC. Matches sklearn.roc_auc_score.

    All tied scores → one threshold. NaN if n_pos==0 or n_neg==0.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    area = 0.0
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        tp += int((y[i:j] == 1).sum())
        fp += int((y[i:j] == 0).sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr
        i = j
    return float(area)


def expected_log_loss_binary(y_true, y_proba, eps: float = 1e-15) -> float:
    """Binary cross-entropy, matches sklearn.log_loss (skipna=True)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_proba, dtype=np.float64)
    mask = ~(np.isnan(y) | np.isnan(p))
    y = y[mask]
    p = np.clip(p[mask], eps, 1.0 - eps)
    if len(y) == 0:
        return float("nan")
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def expected_log_loss_multiclass(
    y_true_idx, y_proba_matrix, eps: float = 1e-15,
) -> float:
    """Multiclass cross-entropy.

    y_true_idx: (N,) int array in [0, K-1].
    y_proba_matrix: (N, K) float array (rows sum to 1).
    NaN if empty after null filter.
    """
    y = np.asarray(y_true_idx, dtype=np.int64)
    P = np.asarray(y_proba_matrix, dtype=np.float64)
    mask = ~(np.isnan(y)) & ~(np.isnan(P).any(axis=1))
    y = y[mask]
    P = P[mask]
    if len(y) == 0:
        return float("nan")
    P = np.clip(P, eps, 1.0 - eps)
    return float(-np.log(P[np.arange(len(y)), y]).mean())
'''


def write_reference():
    print("\n=== tests/_helpers/metrics_reference.py ===")
    p = Path("tests/_helpers/metrics_reference.py")
    p.write_text(REFERENCE)
    compile_check(p)
    print("  OK")


# ============ 2. MetricsOp: multiclass log_loss + AUC dispatch ============

def extend_metrics_op():
    print("\n=== cadw/ops/metrics.py: multiclass log_loss + _compute_roc_auc_* ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    # ---- Update constants ----
    old_consts = '''REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1"})
SUPPORTED_METRICS = REGRESSION_METRICS | PROBA_METRICS | CLASSIFICATION_METRICS
VALID_AVERAGES = frozenset({"binary", "macro", "micro", "weighted"})'''

    new_consts = '''REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1"})
AUC_METRICS = frozenset({"roc_auc"})
SUPPORTED_METRICS = (
    REGRESSION_METRICS | PROBA_METRICS
    | CLASSIFICATION_METRICS | AUC_METRICS
)
VALID_AVERAGES = frozenset({"binary", "macro", "micro", "weighted"})

# module-level sentinel (§13.4/148 lesson: never `obj is object()`)
_ROC_AUC_UNSET = object()'''

    s = replace_line_block(s, old_consts, new_consts, tag="metrics-consts")

    # ---- __init__: y_proba_cols + multiclass detection ----
    old_init_head = '''    def __init__(
        self,
        y_true_col: str,
        *,
        y_pred_col: Optional[str] = None,
        y_proba_col: Optional[str] = None,
        metrics: Sequence[str] | str,
        classes: Optional[Sequence] = None,
        average: Optional[str] = None,
        pos_label=None,
    ) -> None:'''

    new_init_head = '''    def __init__(
        self,
        y_true_col: str,
        *,
        y_pred_col: Optional[str] = None,
        y_proba_col: Optional[str] = None,       # binary backward compat
        y_proba_cols: Optional[Sequence[str]] = None,  # multiclass/binary
        metrics: Sequence[str] | str,
        classes: Optional[Sequence] = None,
        average: Optional[str] = None,
        pos_label=None,
    ) -> None:'''

    s = replace_line_block(s, old_init_head, new_init_head, tag="metrics-init-head")

    # After self.y_proba_col assignment, unify into list
    old_proba_assign = '''        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col

        ms = set(self.metrics)'''

    new_proba_assign = '''        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col  # legacy single; also unified below

        # v0.9.2c: unified y_proba_cols list for binary + multiclass.
        if y_proba_cols is not None:
            self.y_proba_cols = list(y_proba_cols)
        elif y_proba_col is not None:
            self.y_proba_cols = [y_proba_col]
        else:
            self.y_proba_cols = None
        self._needs_multiclass_proba = (
            self.y_proba_cols is not None
            and len(self.y_proba_cols) > 1
        )

        ms = set(self.metrics)'''

    s = replace_line_block(s, old_proba_assign, new_proba_assign,
                            tag="metrics-proba-unify")

    # Reject roc_auc directly (must go through metrics() dispatch)
    old_unknown_check = '''        unknown = [m for m in self.metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"MetricsOp: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )'''

    new_unknown_check = '''        unknown = [m for m in self.metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"MetricsOp: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )
        # v0.9.2c: roc_auc is multipass (Sort-based); handled by
        # CadwDataFrame.metrics() facade, not MetricsOp.
        if any(m in AUC_METRICS for m in self.metrics):
            raise ValueError(
                "MetricsOp: 'roc_auc' is computed by CadwDataFrame.metrics "
                "via external_sort (multipass). Do not pass roc_auc "
                "directly to MetricsOp."
            )'''

    s = replace_line_block(s, old_unknown_check, new_unknown_check,
                            tag="metrics-reject-auc-direct")

    # ---- update(): multiclass log_loss branch ----
    old_ll_block = '''        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self._needs_proba:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_col))'''

    new_ll_block = '''        # Log loss (multiclass v0.9.2c; else binary)
        if self._needs_proba and self._needs_multiclass_proba:
            # Multiclass: y_true encodes class index (via classes);
            # sum -log(p[y_idx]).
            yt_list = batch.column(self.y_true_col).to_pylist()
            proba_lists = [batch.column(c).to_pylist()
                           for c in self.y_proba_cols]
            missing_yt: list = []
            eps = 1e-15
            for i, v in enumerate(yt_list):
                if v is None:
                    continue
                idx = self._cls_to_idx.get(v, -1)
                if idx < 0:
                    if len(missing_yt) < 5:
                        missing_yt.append(v)
                    continue
                p = proba_lists[idx][i]
                if p is None:
                    continue
                p_c = min(max(float(p), eps), 1.0 - eps)
                state["sum_log_loss"] += -math.log(p_c)
                state["n_proba"] += 1
            if missing_yt:
                raise ValueError(
                    f"MetricsOp: log_loss (multiclass) y_true has "
                    f"labels not in classes {self._classes}: "
                    f"{missing_yt} (first 5)."
                )
        # Log loss (binary; y must be {0,1}, p in [0,1])
        elif self._needs_proba:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_cols[0]))'''

    s = replace_line_block(s, old_ll_block, new_ll_block,
                            tag="metrics-multiclass-log-loss")

    # Add `import math` at top if not present
    if "\nimport math\n" not in s:
        old_imports = "from __future__ import annotations\n\nfrom typing import Optional, Sequence"
        new_imports = "from __future__ import annotations\n\nimport math\nfrom typing import Optional, Sequence"
        if old_imports in s:
            s = s.replace(old_imports, new_imports, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK: multiclass log_loss + AUC_METRICS")


# ============ 3. _compute_roc_auc_* module functions ============

ROC_AUC_FUNCS = '''

# ============ ROC-AUC (v0.9.2c) ============

def _compute_roc_auc_binary(
    stream_factory,
    *,
    y_true_col: str,
    y_proba_col: str,
    pos_label,
    spill_dir,
    memory_guard,
    log: bool = False,
) -> float:
    """ROC-AUC (binary ovr) via external_sort + streaming trapezoid.

    Validated by scripts/proto_roc_auc.py PASS 8/8. All tied probas →
    single threshold point (matches sklearn roc_curve).

    Peak RAM: batch + external_sort disk. Independent of N.

    Args:
        stream_factory: callable() → fresh iterable of RecordBatch
            (called twice: pass 1 counts, pass 2 trapezoid).
        y_true_col: binary label column (any two distinct values).
        y_proba_col: probability column.
        pos_label: value in y_true that denotes positive class.
        spill_dir: temp dir for external_sort.
        memory_guard: MemoryGuard for check_batch.
        log: verbose logging.

    Returns:
        float AUC in [0, 1], or NaN if n_pos==0 or n_neg==0.
    """
    import pyarrow as pa
    import pyarrow.compute as _pc
    from pathlib import Path
    from cadw.ops.sort import external_sort
    from cadw.io.batch_reader import ArrowIPCBatchReader

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: count n_pos / n_neg (both y and proba non-null)
    total_valid = 0
    n_pos = 0
    for batch in stream_factory():
        memory_guard.check_batch(batch, name="roc_auc.count")
        y = batch.column(y_true_col)
        p = batch.column(y_proba_col)
        both_valid = _pc.and_(_pc.is_valid(y), _pc.is_valid(p))
        y_f = y.filter(both_valid)
        total_valid += len(y_f)
        if len(y_f) > 0:
            eq = _pc.equal(y_f, pa.scalar(pos_label, type=y.type))
            n_pos += int(_pc.sum(_pc.cast(eq, pa.int64())).as_py() or 0)
    n_neg = total_valid - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort by proba desc (values: y_true)
    sorted_path = external_sort(
        stream_factory(),
        key_columns=[y_proba_col],
        value_columns=[y_true_col],
        memory_guard=memory_guard,
        spill_dir=spill_dir / "sort_by_proba",
        ascending=False,
        log=False,
    )

    # Pass 2: stream sorted, per-threshold grouping, trapezoid
    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    area = 0.0
    cur_score = _ROC_AUC_UNSET
    cur_tp = 0
    cur_fp = 0

    try:
        reader = ArrowIPCBatchReader(
            str(sorted_path), [y_proba_col, y_true_col]
        )
        for b in reader:
            memory_guard.check_batch(b, name="roc_auc.stream")
            probas = b.column(y_proba_col).to_pylist()
            ys = b.column(y_true_col).to_pylist()
            for s, y in zip(probas, ys):
                if s is None or y is None:
                    continue
                if cur_score is _ROC_AUC_UNSET or s != cur_score:
                    if cur_score is not _ROC_AUC_UNSET:
                        tp += cur_tp
                        fp += cur_fp
                        tpr = tp / n_pos
                        fpr = fp / n_neg
                        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
                        prev_fpr = fpr
                        prev_tpr = tpr
                    cur_score = s
                    cur_tp = 0
                    cur_fp = 0
                if y == pos_label:
                    cur_tp += 1
                else:
                    cur_fp += 1
        # Flush last group
        if cur_score is not _ROC_AUC_UNSET:
            tp += cur_tp
            fp += cur_fp
            tpr = tp / n_pos
            fpr = fp / n_neg
            area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
    finally:
        try:
            Path(sorted_path).unlink()
        except OSError:
            pass

    return float(area)


def _compute_roc_auc_macro(
    stream_factory,
    *,
    y_true_col: str,
    y_proba_cols: Sequence[str],
    classes: Sequence,
    spill_dir,
    memory_guard,
    log: bool = False,
) -> float:
    """Macro-average ROC-AUC (one-vs-rest, K binary passes).

    Peak RAM per pass: batch + Sort disk. Independent of N.
    NaN if no class produces valid AUC (all-pos or all-neg).
    """
    import math as _math
    from pathlib import Path

    spill_dir = Path(spill_dir)
    aucs: list = []
    for k, cls in enumerate(classes):
        auc = _compute_roc_auc_binary(
            stream_factory,
            y_true_col=y_true_col,
            y_proba_col=y_proba_cols[k],
            pos_label=cls,
            spill_dir=spill_dir / f"auc_k{k}",
            memory_guard=memory_guard,
            log=log,
        )
        aucs.append(auc)
    valid = [a for a in aucs if not _math.isnan(a)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))
'''


def add_roc_auc_funcs():
    print("\n=== cadw/ops/metrics.py: _compute_roc_auc_* ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()
    if "_compute_roc_auc_binary" in s:
        print("  already present — skip")
        return
    if not s.endswith("\n"):
        s += "\n"
    s += ROC_AUC_FUNCS
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 4. dataframe.py::metrics — dispatch ============

def extend_metrics_facade():
    print("\n=== cadw/dataframe.py::metrics — roc_auc + multiclass dispatch ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "y_proba_cols=y_proba_cols" in s:
        print("  already present — skip")
        return

    # Update signature (y_proba: str | Sequence[str])
    old_sig = '''        y_proba: "str | None" = None,
        classes: "Sequence | None" = None,
        average: "str | None" = None,
        pos_label=None,
    ) -> dict:'''

    new_sig = '''        y_proba: "str | Sequence[str] | None" = None,
        classes: "Sequence | None" = None,
        average: "str | None" = None,
        pos_label=None,
    ) -> dict:'''

    s = replace_line_block(s, old_sig, new_sig, tag="metrics-facade-signature")

    # Update body: dispatch roc_auc + multiclass handling
    old_body = '''        # §13.4/164: fail-fast for classification K too large — before
        # constructing MetricsOp (clear message, not "state limit").
        ms = set(metrics)
        if ms & CLASSIFICATION_METRICS:
            K = 2 if classes is None else len(classes)
            est = 64 + 24 * K
            state_limit = self._memory_guard.state_limit_bytes
            if est > state_limit:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"metrics: classification K={K} → state "
                    f"{est} B > state_limit {state_limit} B. "
                    f"Reduce class count or increase budget. §13.4/164."
                )

        op = MetricsOp(
            y_true, y_pred_col=y_pred, y_proba_col=y_proba,
            metrics=metrics, classes=classes,
            average=average, pos_label=pos_label,
        )
        return reduce(
            self._iter_filtered_batches(), op,
            self._memory_guard, log=False,
        )'''

    new_body = '''        # §13.4/164: fail-fast for classification K too large — before
        # constructing MetricsOp (clear message, not "state limit").
        ms = set(metrics)
        if (ms & CLASSIFICATION_METRICS) or (ms & AUC_METRICS):
            K = 2 if classes is None else len(classes)
            est = 64 + 24 * K
            state_limit = self._memory_guard.state_limit_bytes
            if est > state_limit:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"metrics: K={K} → state {est} B > state_limit "
                    f"{state_limit} B. Reduce class count or increase "
                    f"budget. §13.4/164."
                )

        # v0.9.2c: y_proba normalization (str → [str], list → list).
        if y_proba is None:
            y_proba_cols = None
        elif isinstance(y_proba, str):
            y_proba_cols = [y_proba]
        else:
            y_proba_cols = list(y_proba)
            if not y_proba_cols:
                raise ValueError(
                    "metrics: y_proba list is empty"
                )

        # Column existence for all y_proba cols
        if y_proba_cols is not None:
            for pc_ in y_proba_cols:
                if pc_ not in self._columns:
                    raise KeyError(
                        f"metrics: y_proba column {pc_!r} not in "
                        f"{self._columns}"
                    )

        is_multiclass_proba = (
            y_proba_cols is not None and len(y_proba_cols) > 1
        )
        if is_multiclass_proba:
            if classes is None:
                raise ValueError(
                    f"metrics: multiclass y_proba (len="
                    f"{len(y_proba_cols)}) requires `classes`"
                )
            if len(y_proba_cols) != len(classes):
                raise ValueError(
                    f"metrics: len(y_proba)={len(y_proba_cols)} != "
                    f"len(classes)={len(classes)}"
                )

        # roc_auc: separate handling (multipass Sort-based)
        has_roc_auc = "roc_auc" in metrics
        other_metrics = [m for m in metrics if m != "roc_auc"]

        # Conflict: roc_auc + regression on same y_true (binary labels vs
        # continuous target — mutually exclusive semantics).
        if has_roc_auc and (set(other_metrics) & REGRESSION_METRICS):
            raise ValueError(
                f"metrics: roc_auc requires binary labels y_true, "
                f"regression ({sorted(set(other_metrics) & REGRESSION_METRICS)}) "
                f"requires continuous — cannot mix. Use separate calls."
            )
        if has_roc_auc and y_proba_cols is None:
            raise ValueError("metrics: roc_auc requires y_proba")

        result: dict = {}

        if other_metrics:
            op = MetricsOp(
                y_true,
                y_pred_col=y_pred,
                y_proba_cols=y_proba_cols,
                metrics=other_metrics,
                classes=classes,
                average=average,
                pos_label=pos_label,
            )
            result.update(reduce(
                self._iter_filtered_batches(), op,
                self._memory_guard, log=False,
            ))

        if has_roc_auc:
            from cadw.ops.metrics import (
                _compute_roc_auc_binary, _compute_roc_auc_macro,
            )
            if is_multiclass_proba:
                result["roc_auc"] = _compute_roc_auc_macro(
                    stream_factory=lambda: self._iter_filtered_batches(),
                    y_true_col=y_true,
                    y_proba_cols=y_proba_cols,
                    classes=list(classes),
                    spill_dir=self._spill_dir / "metrics_roc_auc",
                    memory_guard=self._memory_guard,
                )
            else:
                pos_roc = pos_label if pos_label is not None else 1
                result["roc_auc"] = _compute_roc_auc_binary(
                    stream_factory=lambda: self._iter_filtered_batches(),
                    y_true_col=y_true,
                    y_proba_col=y_proba_cols[0],
                    pos_label=pos_roc,
                    spill_dir=self._spill_dir / "metrics_roc_auc",
                    memory_guard=self._memory_guard,
                )

        return result'''

    s = replace_line_block(s, old_body, new_body, tag="metrics-facade-dispatch")

    # Update imports in facade
    old_imp = '''        from cadw.ops.metrics import (
            CLASSIFICATION_METRICS, REGRESSION_METRICS,
            SUPPORTED_METRICS, MetricsOp,
        )'''
    new_imp = '''        from cadw.ops.metrics import (
            AUC_METRICS, CLASSIFICATION_METRICS, REGRESSION_METRICS,
            SUPPORTED_METRICS, MetricsOp,
        )'''
    if old_imp in s:
        s = s.replace(old_imp, new_imp, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 5. __init__.py — export AUC_METRICS ============

def update_init():
    print("\n=== cadw/__init__.py: export AUC_METRICS ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "AUC_METRICS" in s:
        print("  already present — skip")
        return
    old = "    REGRESSION_METRICS, PROBA_METRICS, CLASSIFICATION_METRICS,\n)\n"
    new = "    REGRESSION_METRICS, PROBA_METRICS, CLASSIFICATION_METRICS,\n    AUC_METRICS,\n)\n"
    if old not in s:
        raise SystemExit("metrics import anchor not found")
    s = s.replace(old, new, 1)
    if '"AUC_METRICS"' not in s:
        old_all = '    "CLASSIFICATION_METRICS",\n'
        if old_all in s:
            s = s.replace(old_all, old_all + '    "AUC_METRICS",\n', 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 6. Tests ============

REFERENCE_TESTS = '''"""tests/metrics_reference.py: helper self-consistency."""
import math
import numpy as np
import pytest

from tests._helpers.metrics_reference import (
    expected_log_loss_binary,
    expected_log_loss_multiclass,
    expected_roc_auc,
)


def test_roc_auc_perfect():
    assert expected_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_roc_auc_inverted():
    assert expected_roc_auc([0, 1, 0, 1], [0.5, 0.4, 0.6, 0.3]) == pytest.approx(0.0)


def test_roc_auc_ties():
    """All same score → 0.5."""
    assert expected_roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_partial_ties():
    """y=[0,1,0,1], s=[0.5,0.5,0.7,0.7] → 0.5 (ties grouped)."""
    assert expected_roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.7, 0.7]) == pytest.approx(0.5)


def test_roc_auc_all_positive_nan():
    assert math.isnan(expected_roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_roc_auc_all_negative_nan():
    assert math.isnan(expected_roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_roc_auc_sklearn_equivalence_if_installed():
    pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = 200
        y = rng.integers(0, 2, size=n)
        s = rng.random(size=n)
        assert expected_roc_auc(y, s) == pytest.approx(
            roc_auc_score(y, s), rel=1e-9,
        )


def test_log_loss_binary_matches_sklearn_if_installed():
    pytest.importorskip("sklearn")
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100).astype(float)
    p = rng.random(size=100)
    assert expected_log_loss_binary(y, p) == pytest.approx(
        log_loss(y, p), rel=1e-9,
    )


def test_log_loss_multiclass_known():
    """3 samples, 3 classes: -mean(log(p[i, y_i])) = -mean(log([0.9,0.7,0.6]))."""
    import numpy as np
    y = np.array([0, 1, 2])
    P = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
    ])
    expected = float(-np.log(np.array([0.9, 0.7, 0.6])).mean())
    assert expected_log_loss_multiclass(y, P) == pytest.approx(expected)


def test_log_loss_multiclass_matches_sklearn_if_installed():
    pytest.importorskip("sklearn")
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(1)
    n, K = 200, 3
    y = rng.integers(0, K, size=n)
    P = rng.random(size=(n, K))
    P = P / P.sum(axis=1, keepdims=True)
    assert expected_log_loss_multiclass(y, P) == pytest.approx(
        log_loss(y, P, labels=list(range(K))), rel=1e-9,
    )
'''


METRICS_TESTS_APPEND = '''

# ============ v0.9.2c: roc_auc + multiclass log_loss ============

from tests._helpers.metrics_reference import (
    expected_roc_auc, expected_log_loss_multiclass,
    expected_log_loss_binary,
)


# --- multiclass log_loss ---

def test_metrics_multiclass_log_loss(tmp_path):
    """3 classes, perfect proba → log_loss ~ 0."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2, 0, 1, 2],
        "p0": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "p1": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "p2": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert res["log_loss"] == pytest.approx(0.0, abs=1e-10)


def test_metrics_multiclass_log_loss_known(tmp_path):
    """y=[0,1,2], p*=[0.9,0.7,0.6] for true class → -mean(log)."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2],
        "p0": [0.9, 0.1, 0.2],
        "p1": [0.05, 0.7, 0.2],
        "p2": [0.05, 0.2, 0.6],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    ref_y = [0, 1, 2]
    ref_P = [[0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]]
    expected = expected_log_loss_multiclass(ref_y, ref_P)
    assert res["log_loss"] == pytest.approx(expected, rel=1e-10)


def test_metrics_multiclass_log_loss_requires_classes(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1], "p0": [0.5, 0.5], "p1": [0.5, 0.5],
    })
    with pytest.raises(ValueError, match="requires .classes."):
        df.metrics(
            y_true="y", y_proba=["p0", "p1"], metrics=["log_loss"],
        )


def test_metrics_multiclass_log_loss_classes_mismatch(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1], "p0": [0.5, 0.5], "p1": [0.5, 0.5],
    })
    with pytest.raises(ValueError, match="len\\(y_proba\\)"):
        df.metrics(
            y_true="y", y_proba=["p0", "p1"], metrics=["log_loss"],
            classes=[0, 1, 2],
        )


# --- roc_auc binary ---

def test_metrics_roc_auc_binary_perfect(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1], "p": [0.1, 0.2, 0.8, 0.9],
    })
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert res["roc_auc"] == pytest.approx(1.0)


def test_metrics_roc_auc_binary_known(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1], "p": [0.1, 0.4, 0.35, 0.8],
    })
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert res["roc_auc"] == pytest.approx(0.75)


def test_metrics_roc_auc_binary_ties(tmp_path):
    """All same proba → 0.5."""
    df = make_df(tmp_path, {
        "y": [0, 1, 0, 1], "p": [0.5, 0.5, 0.5, 0.5],
    })
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert res["roc_auc"] == pytest.approx(0.5)


def test_metrics_roc_auc_binary_all_positive_nan(tmp_path):
    df = make_df(tmp_path, {"y": [1, 1, 1], "p": [0.1, 0.5, 0.9]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert math.isnan(res["roc_auc"])


def test_metrics_roc_auc_binary_all_negative_nan(tmp_path):
    df = make_df(tmp_path, {"y": [0, 0, 0], "p": [0.1, 0.5, 0.9]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert math.isnan(res["roc_auc"])


def test_metrics_roc_auc_binary_nulls_skipped(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, None, 1, 1], "p": [0.1, 0.9, None, 0.8],
    })
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    # valid: (0,0.1), (1,0.8) → perfect → 1.0
    assert res["roc_auc"] == pytest.approx(1.0)


def test_metrics_roc_auc_binary_filtered(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1, 0, 1],
        "p": [0.1, 0.2, 0.8, 0.9, 0.5, 0.7],
        "f": [1, 1, 1, 1, 0, 0],
    })
    filtered = df[df["f"] == 1]
    res = filtered.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    # Only first 4 rows → perfect → 1.0
    assert res["roc_auc"] == pytest.approx(1.0)


def test_metrics_roc_auc_matches_reference(tmp_path):
    rng = np.random.default_rng(7)
    n = 200
    y = rng.integers(0, 2, size=n).tolist()
    p = rng.random(size=n).tolist()
    df = make_df(tmp_path, {"y": y, "p": p}, chunk=20)
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    ref = expected_roc_auc(y, p)
    assert res["roc_auc"] == pytest.approx(ref, rel=1e-9)


# --- roc_auc multiclass (macro) ---

def test_metrics_roc_auc_multiclass_macro(tmp_path):
    """3-class, perfect separation via ovr → macro=1.0."""
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1, 2, 2],
        "p0": [0.9, 0.8, 0.05, 0.1, 0.05, 0.1],
        "p1": [0.05, 0.1, 0.9, 0.8, 0.1, 0.05],
        "p2": [0.05, 0.1, 0.05, 0.1, 0.9, 0.85],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["roc_auc"], classes=[0, 1, 2],
    )
    assert res["roc_auc"] == pytest.approx(1.0)


def test_metrics_roc_auc_multiclass_requires_classes(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1], "p0": [0.5, 0.5], "p1": [0.5, 0.5],
    })
    with pytest.raises(ValueError, match="requires .classes."):
        df.metrics(
            y_true="y", y_proba=["p0", "p1"], metrics=["roc_auc"],
        )


# --- mixed ---

def test_metrics_roc_auc_with_accuracy(tmp_path):
    """roc_auc + accuracy, both binary — different columns."""
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1], "yp": [0, 0, 1, 0], "p": [0.1, 0.2, 0.8, 0.4],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["roc_auc", "accuracy"], classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["roc_auc"] == pytest.approx(0.875)


def test_metrics_roc_auc_with_regression_conflict(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0.1, 0.9], "p": [0.2, 0.8]})
    with pytest.raises(ValueError, match="cannot mix"):
        df.metrics(
            y_true="y", y_pred="yp", y_proba="p",
            metrics=["roc_auc", "r2"],
        )


def test_metrics_roc_auc_without_y_proba_raises(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0.1, 0.9]})
    with pytest.raises(ValueError, match="requires y_proba"):
        df.metrics(y_true="y", y_pred="yp", metrics=["roc_auc"])


# --- direct MetricsOp rejects roc_auc ---

def test_metrics_op_rejects_roc_auc_directly():
    from cadw.ops.metrics import MetricsOp
    with pytest.raises(ValueError, match="roc_auc.*separately"):
        MetricsOp(
            "y", y_proba_col="p", metrics=["roc_auc"],
        )
'''


E2E_APPEND = '''

@pytest.mark.slow
def test_metrics_100k_roc_auc_and_multiclass(tmp_path):
    """100K rows: roc_auc + multiclass log_loss, memory independent of N."""
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    # well-separated probas via one-hot + noise
    P = np.full((n, 3), 0.05)
    P[np.arange(n), y] = 0.9
    P += rng.normal(scale=0.02, size=(n, 3))
    P = np.clip(P, 1e-6, None)
    P = P / P.sum(axis=1, keepdims=True)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "y": y.tolist(),
            "p0": P[:, 0].tolist(),
            "p1": P[:, 1].tolist(),
            "p2": P[:, 2].tolist(),
        }),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss", "roc_auc"], classes=[0, 1, 2],
    )

    # Reference: multiclass log_loss (manual)
    eps = 1e-15
    P_c = np.clip(P, eps, 1.0 - eps)
    ref_ll = float(-np.log(P_c[np.arange(n), y]).mean())
    assert res["log_loss"] == pytest.approx(ref_ll, rel=1e-9)

    # Reference: macro roc_auc (3 ovr passes)
    from tests._helpers.metrics_reference import expected_roc_auc
    aucs = []
    for cls in (0, 1, 2):
        y_bin = (y == cls).astype(np.int64)
        aucs.append(expected_roc_auc(y_bin, P[:, cls]))
    ref_auc = float(np.mean(aucs))
    assert res["roc_auc"] == pytest.approx(ref_auc, rel=1e-6)
'''


def write_tests():
    print("\n=== tests/test_metrics_reference.py ===")
    p = Path("tests/test_metrics_reference.py")
    p.write_text(REFERENCE_TESTS)
    compile_check(p)
    print("  OK")

    print("\n=== tests/test_metrics.py: append v092c tests ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()
    if "test_metrics_multiclass_log_loss" in s:
        print("  already present — skip")
    else:
        if not s.endswith("\n"):
            s += "\n"
        s += METRICS_TESTS_APPEND
        p.write_text(s)
        compile_check(p)
        print("  OK")

    print("\n=== tests/test_metrics_e2e.py: append v092c slow test ===")
    p = Path("tests/test_metrics_e2e.py")
    s = p.read_text()
    if "test_metrics_100k_roc_auc_and_multiclass" in s:
        print("  already present — skip")
    else:
        if not s.endswith("\n"):
            s += "\n"
        s += E2E_APPEND
        p.write_text(s)
        compile_check(p)
        print("  OK")


# ============ 7. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.2c via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2c — roc_auc" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.2c — roc_auc + multiclass log_loss)

- **`roc_auc`** в `CadwDataFrame.metrics()`:
  - **binary** (`y_proba` scalar str): `y_true ∈ {pos_label, other}`.
    `pos_label` default `1`.
  - **macro** (`y_proba` list of K cols, `classes` explicit K>2):
    one-vs-rest, macro average.
  - Algorithm: **external_sort** by proba desc + **per-threshold
    grouping** + streaming trapezoid. Peak RAM = batch + Sort disk
    (independent of N). Validated by prototype 8/8.
  - NaN if `n_pos == 0` or `n_neg == 0` (honest).
- **Multiclass `log_loss`**: `y_proba` list of K cols, `classes` explicit.
  O(1) state (sum_log_loss, n_proba). One pass.
- **`AUC_METRICS`** = `{"roc_auc"}` (exported).
- **`tests/_helpers/metrics_reference.py`**: `expected_roc_auc`,
  `expected_log_loss_binary`, `expected_log_loss_multiclass` —
  reference implementations (validated by prototype + sklearn
  equivalence if installed).

### Design decisions (locked-in, v0.9.2c)

- **roc_auc — не Reduce**. Sort-based (multipass): pass 1 count,
  external_sort, pass 2 trapezoid. `MetricsOp` rejects `roc_auc`
  directly; `CadwDataFrame.metrics()` dispatches it separately.
- **Per-threshold grouping** (ties): all tied scores = one ROC point
  (matches sklearn.roc_curve). Prototype caught this (§13.4/166).
- **y_proba**: str → binary, list[str] → multiclass (both log_loss
  and roc_auc).
- **Conflict**: `roc_auc` (binary labels) + regression (`r2` etc.,
  continuous) on same `y_true` → ValueError. Use separate calls.

### Ограничения (v0.9.2c, §11.5)

- **roc_auc binary requires `y_true ∈ {0,1}`** (or `pos_label` +
  one other value; values not in `{pos_label, …}` treated as
  negative — but validated only for column presence, not value domain).
  String labels → use multiclass form (`y_proba` list + `classes`).
- **roc_auc macro** — `average='macro'` only (unweighted); weighted
  ovr — backlog.
- **`roc_curve`** (TPR/FPR arrays) — не возвращается; если запросят —
  отдельный метод.
- **Sort temp** — `spill_dir/metrics_roc_auc/sort_by_proba/*` (auto
  cleanup after streaming).
- **N × row_size** does not bind peak: Sort handles arbitrary N via
  disk. Fail-fast on `K × 24 + 64 > state_limit` (classification state).

### Тесты (v0.9.2c)

- +10 fast (`test_metrics_reference.py`): helper self-consistency,
  sklearn-equivalence (importorskip), known cases.
- +17 fast (`test_metrics.py`): multiclass log_loss (perfect/known/
  validation), roc_auc binary (perfect/known/ties/all-pos-NaN/
  all-neg-NaN/nulls/filtered/reference), roc_auc macro, mixed
  (accuracy + roc_auc, regression conflict, requires y_proba),
  MetricsOp rejects roc_auc.
- +1 slow (`test_metrics_e2e.py`): 100K rows, roc_auc + multiclass
  log_loss, memory 30 MB budget.

- fast: **~1468** (+27). slow: **18** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: roc_auc note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.2c (roc_auc" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**v0.9.2c (roc_auc):** binary+macro via external_sort + "
        s += "per-threshold grouping. Multiclass log_loss (y_proba list)."
        s += "\\n"
        p.write_text(s)
        return

    note = '''**v0.9.2c (roc_auc + multiclass log_loss):**
- `roc_auc` — multipass (Sort-based, not Reduce). Binary (y_proba scalar)
  or macro (y_proba list K, classes K>2).
- Per-threshold grouping (ties) — matches sklearn.
- Peak RAM = batch + Sort disk. Independent of N.
- NaN if all-pos or all-neg.
- Multiclass log_loss — y_proba list of K, y_true encoded via classes.
- Conflict: roc_auc + regression on same y_true → ValueError.
- **`roc_curve`** arrays (TPR/FPR) — не возвращаются; separate if asked.
- `roc_auc` macro — unweighted (`average='macro'`); weighted — backlog.
- **Binary y_true validation**: y_proba scalar + pos_label → any y_true
  values treated as positive (pos_label) / negative (rest). String
  binary → use multiclass form.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ 8. §13.4/166 ============

def append_decisions_166():
    print("\n=== DECISIONS.md: §13.4/166 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 166" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/166 (np.trapz removed; tie-handling in roc_auc; "
        s += "prototype also test-authoring).**\\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 166 (numpy 2.4 trapz, roc_auc ties, prototype as test-
authoring layer).** v0.9.2c prototype: 2 fix iterations before PASS.

**Layer 1 (external API contract):** `np.trapz` → `np.trapezoid`
(numpy 2.0 deprecation, removed in 2.4). pyproject pins
`numpy>=2.4.4`. Prototype on old API → AttributeError.

**Layer 2 (algorithm — ties):** `roc_auc` requires **per-threshold
grouping**, not per-row. All tied scores → one ROC point (sklearn
behavior). My per-row iteration gave wrong AUC for tied scores:
- Case 4 (all same score 0.5): 0.25 vs correct 0.5
- Case 7 (partial ties 0.5,0.5,0.7,0.7): 0.25 vs correct 0.5

**Layer 3 (test-authoring in prototype):** expected values in prototype
were manual arithmetic. Case 3 expected 5/6=0.833, correct 3/4=0.75
(different sklearn example). Case 2 expected 0.75 (label "random"),
correct 0.0 (fully inverted). **4th time in session** (§13.4/128, /163,
/164, /166).

**Fix (v0.9.2c):**
- `np.trapz` → `np.trapezoid`.
- **Per-threshold grouping** in streaming: `cur_score` sentinel,
  accumulate `cur_tp/cur_fp` per group, flush on score change.
- `tests/_helpers/metrics_reference.py::expected_roc_auc` — reference
  implementation (validated by prototype PASS + sklearn equivalence).
  Tests import it; no manual arithmetic in test bodies.

**Урок (§13.4/166, расширение 128, 129, 163, 164, 165):**
- **Prototype is also code, also requires API contract validation**
  (`np.trapz` in pyproject numpy>=2.4 → would fail on CI). Backlog:
  extend `test_api_availability.py` to numpy (`np.trapezoid`,
  `np.divide(where=)`, `np.int64`).
- **Prototype with expected values = test-authoring.** 4th time manual
  arithmetic failed. Rule: prototype expected — **only** from reference
  library (sklearn) OR **no expected** (A==B consistency only).
  Expected-tests — separate, with `tests/_helpers/metrics_reference.py`.
- **Sort-based algorithms need tie-heavy cases in prototype**
  (all-same-value, partial ties). Random continuous proba misses ties
  because ties are rare — but 100K E2E with quantized proba might
  have them. Prototype case 4 (all 0.5) + case 7 (ties 0.5/0.5/0.7/0.7)
  caught the bug in 5 minutes; without prototype — v092c would ship
  with tie-bug (missed on random tests, caught only in specific
  production data).

**Backlog (§11.6):**
- Extend `test_api_availability.py`: `np.trapezoid` (numpy 2.4+),
  `np.divide(where=)` (numpy 1.13+), `np.int64`, `np.float64`,
  `np.asarray`, `np.clip`, `np.isnan`, `np.log`, `np.exp` — что
  используем в metrics/pca/linear/gbdt.
- Prototype template: `expected = reference_fn(...)` (sklearn/pandas)
  or `expected = None` (consistency only). Manual arithmetic — fail.

**Связанные:** §13.4/128, /129, /163, /164, /165.

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/166")


def main():
    print("=== v0.9.2c: roc_auc + multiclass log_loss ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    write_reference()
    extend_metrics_op()
    add_roc_auc_funcs()
    extend_metrics_facade()
    update_init()
    write_tests()
    append_changelog()
    update_decisions()
    append_decisions_166()

    print("\n=== pytest: metrics_reference ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics_reference.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REFERENCE TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: metrics fast (all) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: metrics E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_metrics_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: reduce + sort regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "tests/test_sort.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE/SORT REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.2c: roc_auc (binary + macro, sort-based per-threshold) + "
        "multiclass log_loss; MetricsOp rejects roc_auc directly "
        "(dispatched by facade); tests/_helpers/metrics_reference.py; "
        "§13.4/166 (numpy np.trapezoid; tie-handling; prototype expected "
        "from reference)"])

    print("\n==> v0.9.2c applied.")
    print("NEXT: v0.9.2d (per-group metrics) or v0.9.2e (CV).")


if __name__ == "__main__":
    main()