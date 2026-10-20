#!/usr/bin/env python3
"""v0.9.2b: classification metrics — accuracy/precision/recall/f1.

SCOPE:
  - ReduceOp.batch_memory_factor() — protocol extension (default 1.0)
  - reduce() passes extra_factor to check_batch
  - MetricsOp: TP/FP/FN per-class state, average modes
  - CadwDataFrame.metrics(): classification dispatch, classes/average
  - SUPPORTED_METRICS update

DESIGN (locked-in):
  - State = tp[K] + fp[K] + fn[K] int64 = 24×K + 56 B. K=1000: 24 KB
    (vs confusion K²×8 = 8 MB). K² not needed for scalar metrics.
  - average: binary (K=2, pos_label) / macro / micro / weighted.
    Auto: binary if K==2 else macro.
  - Mixed regression (r2/mse/rmse/mae) + classification (accuracy/...)
    on same y_pred → ValueError (continuous vs labels).
  - log_loss + classification OK (different columns: y_proba vs y_pred).
  - Classification path uses .to_pylist() per batch (7× overhead).
    reduce() calls check_batch(extra_factor=op.batch_memory_factor())
    → MetricsOp returns 8.0 when classification requested. Honest.
  - Fail-fast: K × 24 + 64 > state_limit → MemoryLimitExceeded in
    estimate_state_nbytes before init.
  - Numeric or string labels supported (dict lookup, nulls skipped).

TESTS (~20 fast + 1 slow):
  - accuracy (binary perfect/mixed)
  - precision/recall/f1 binary (pos_label)
  - multiclass macro/micro/weighted
  - zero-division (no positive predictions → 0.0, sklearn zero_division=0)
  - nulls skipped, empty, filtered
  - unknown label → ValueError
  - mixed regression+classification → ValueError
  - average=binary on K>2 → ValueError
  - K too large → MemoryLimitExceeded
  - string labels, sklearn equivalence
  - 100K rows 3-class (slow), manual numpy reference

GUARDS: new tests + existing metrics + full suite + guards.

Run: uv run python scripts/patch_v092b_classification_metrics.py
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


# ============ 1. ReduceOp.batch_memory_factor + reduce() ============

def extend_reduce():
    print("\n=== cadw/ops/reduce.py: batch_memory_factor + reduce() ===")
    p = Path("cadw/ops/reduce.py")
    s = p.read_text()

    if "batch_memory_factor" in s:
        print("  already present — skip")
        return

    # Add method to ReduceOp (after estimate_state_nbytes)
    old_op = '''    def estimate_state_nbytes(self) -> "int | None":
        """Upper bound on state size BEFORE init() (v0.5.1, P8 fix).

        Called by `reduce()` for `MemoryGuard.check_state` *before*
        `op.init()`, so the state must be estimable without allocation.

        - Built-in ops override with known sizes (no allocation, O(1)).
        - User ops should override — otherwise `MemoryGuard.
          estimate_state_size` falls back to 1 MiB hardcode (P8 in
          _FOR_PINGIS/REPORT.md, DECISIONS §9.2.4).
        - Returning None means "use MemoryGuard fallback".

        Contract: `estimate_state_nbytes() >= state_nbytes(init())`.
        """
        return None'''

    new_op = '''    def estimate_state_nbytes(self) -> "int | None":
        """Upper bound on state size BEFORE init() (v0.5.1, P8 fix).

        Called by `reduce()` for `MemoryGuard.check_state` *before*
        `op.init()`, so the state must be estimable without allocation.

        - Built-in ops override with known sizes (no allocation, O(1)).
        - User ops should override — otherwise `MemoryGuard.
          estimate_state_size` falls back to 1 MiB hardcode (P8 in
          _FOR_PINGIS/REPORT.md, DECISIONS §9.2.4).
        - Returning None means "use MemoryGuard fallback".

        Contract: `estimate_state_nbytes() >= state_nbytes(init())`.
        """
        return None

    def batch_memory_factor(self) -> float:
        """Extra factor for `check_batch` (v0.9.2b).

        Default 1.0 — batch.nbytes covers peak per-batch RAM.

        Override in ops that temporarily exceed `batch.nbytes` inside
        `update` (e.g., `.to_pylist()` overhead — 7× for Python objects).
        `reduce()` passes this to `check_batch(extra_factor=...)`.
        """
        return 1.0'''

    s = replace_line_block(s, old_op, new_op, tag="reduce-op-protocol")

    # Update reduce() to pass extra_factor
    old_reduce = '''    state = op.init()
    batch_id = 0
    for batch in batch_iter:
        memory_guard.check_batch(batch, name=f"{type(op).__name__}.batch")
        state = op.update(state, batch)'''

    new_reduce = '''    state = op.init()
    batch_id = 0
    extra_factor = op.batch_memory_factor()
    for batch in batch_iter:
        memory_guard.check_batch(
            batch,
            name=f"{type(op).__name__}.batch",
            extra_factor=extra_factor,
        )
        state = op.update(state, batch)'''

    s = replace_line_block(s, old_reduce, new_reduce, tag="reduce-extra-factor")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 2. Extend MetricsOp ============

def extend_metrics_op():
    print("\n=== cadw/ops/metrics.py: classification metrics ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    if "CLASSIFICATION_METRICS" in s:
        print("  already present — skip")
        return

    # Update header constants
    old_consts = '''REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
SUPPORTED_METRICS = REGRESSION_METRICS | PROBA_METRICS'''

    new_consts = '''REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1"})
SUPPORTED_METRICS = REGRESSION_METRICS | PROBA_METRICS | CLASSIFICATION_METRICS
VALID_AVERAGES = frozenset({"binary", "macro", "micro", "weighted"})'''

    s = replace_line_block(s, old_consts, new_consts, tag="metrics-consts")

    # Update __init__ signature and validation
    old_init = '''    def __init__(
        self,
        y_true_col: str,
        *,
        y_pred_col: Optional[str] = None,
        y_proba_col: Optional[str] = None,
        metrics: Sequence[str] | str,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = list(metrics)
        unknown = [m for m in self.metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"MetricsOp: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col

        needs_pred = bool(set(self.metrics) & REGRESSION_METRICS)
        needs_proba = bool(set(self.metrics) & PROBA_METRICS)
        if needs_pred and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for "
                f"{sorted(set(self.metrics) & REGRESSION_METRICS)}"
            )
        if needs_proba and y_proba_col is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(set(self.metrics) & PROBA_METRICS)}"
            )'''

    new_init = '''    def __init__(
        self,
        y_true_col: str,
        *,
        y_pred_col: Optional[str] = None,
        y_proba_col: Optional[str] = None,
        metrics: Sequence[str] | str,
        classes: Optional[Sequence] = None,
        average: Optional[str] = None,
        pos_label: int = 1,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = list(metrics)
        unknown = [m for m in self.metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"MetricsOp: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col

        ms = set(self.metrics)
        self._needs_regression = bool(ms & REGRESSION_METRICS)
        self._needs_proba = bool(ms & PROBA_METRICS)
        self._needs_classification = bool(ms & CLASSIFICATION_METRICS)

        # §13.4/164: regression (continuous y_pred) and classification
        # (labels y_pred) can't share the same column.
        if self._needs_regression and self._needs_classification:
            raise ValueError(
                f"MetricsOp: regression ({sorted(ms & REGRESSION_METRICS)}) "
                f"and classification ({sorted(ms & CLASSIFICATION_METRICS)}) "
                f"cannot share y_pred column — one expects continuous "
                f"values, the other labels. Pass distinct calls."
            )

        if self._needs_regression and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for "
                f"{sorted(ms & REGRESSION_METRICS)}"
            )
        if self._needs_proba and y_proba_col is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(ms & PROBA_METRICS)}"
            )
        if self._needs_classification and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for classification "
                f"{sorted(ms & CLASSIFICATION_METRICS)}"
            )

        # Classification setup
        self._classes: list = []
        self._cls_to_idx: dict = {}
        self._k: int = 0
        self._average_method: str = ""
        self._pos_label = pos_label

        if self._needs_classification:
            if classes is None:
                # K=2 default
                self._classes = [0, 1]
            else:
                self._classes = list(classes)
            self._k = len(self._classes)
            if self._k < 2:
                raise ValueError(
                    f"MetricsOp: classification requires at least 2 "
                    f"classes; got {self._k}"
                )
            # Duplicate check
            if len(set(map(repr, self._classes))) != self._k:
                raise ValueError(
                    f"MetricsOp: classes must be unique; got "
                    f"{self._classes}"
                )
            self._cls_to_idx = {c: i for i, c in enumerate(self._classes)}

            # Average method: explicit or auto
            if average is None:
                self._average_method = (
                    "binary" if self._k == 2 else "macro"
                )
            elif average not in VALID_AVERAGES:
                raise ValueError(
                    f"MetricsOp: average must be one of "
                    f"{sorted(VALID_AVERAGES)}; got {average!r}"
                )
            else:
                self._average_method = average
            if self._average_method == "binary" and self._k != 2:
                raise ValueError(
                    f"MetricsOp: average='binary' requires K=2; "
                    f"got K={self._k}. Use 'macro'/'micro'/'weighted'."
                )
            if self._average_method == "binary":
                if pos_label not in self._cls_to_idx:
                    raise ValueError(
                        f"MetricsOp: pos_label {pos_label!r} not in "
                        f"classes {self._classes}"
                    )'''

    s = replace_line_block(s, old_init, new_init, tag="metrics-init")

    # Update estimate_state_nbytes
    old_est = '''    def estimate_state_nbytes(self) -> int:
        # 7 counters (n, sum_y, sum_yy, sum_res_sq, sum_abs_res, sum_ll,
        # n_proba) × 8 B = 56, rounded to 64.
        return 64'''

    new_est = '''    def estimate_state_nbytes(self) -> int:
        # Base: 7 counters × 8 B = 56, rounded to 64.
        base = 64
        if self._needs_classification:
            # tp[K] + fp[K] + fn[K] int64 = 24 × K bytes
            return base + 24 * self._k
        return base

    def batch_memory_factor(self) -> float:
        """§13.4/164: classification uses .to_pylist() per batch
        (~7× Python-object overhead). reduce() checks batch with this
        factor — honest peak RAM, not hidden."""
        return 8.0 if self._needs_classification else 1.0'''

    s = replace_line_block(s, old_est, new_est, tag="metrics-est")

    # Update init() to add classification state
    old_state_init = '''    def init(self):
        return {
            "n": 0,
            "sum_y": 0.0,
            "sum_yy": 0.0,
            "sum_residual_sq": 0.0,
            "sum_abs_residual": 0.0,
            "sum_log_loss": 0.0,
            "n_proba": 0,
        }'''

    new_state_init = '''    def init(self):
        state = {
            "n": 0,
            "sum_y": 0.0,
            "sum_yy": 0.0,
            "sum_residual_sq": 0.0,
            "sum_abs_residual": 0.0,
            "sum_log_loss": 0.0,
            "n_proba": 0,
        }
        if self._needs_classification:
            state["tp"] = np.zeros(self._k, dtype=np.int64)
            state["fp"] = np.zeros(self._k, dtype=np.int64)
            state["fn"] = np.zeros(self._k, dtype=np.int64)
        return state'''

    s = replace_line_block(s, old_state_init, new_state_init, tag="metrics-state-init")

    # Update update() to add classification branch
    old_update_tail = '''                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        return state'''

    new_update_tail = '''                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        # Classification (v0.9.2b): TP/FP/FN per class
        if self._needs_classification:
            yt_list = batch.column(self.y_true_col).to_pylist()
            yp_list = batch.column(self.y_pred_col).to_pylist()
            missing_yt: list = []
            missing_yp: list = []
            yt_idx = np.full(len(yt_list), -1, dtype=np.int64)
            yp_idx = np.full(len(yp_list), -1, dtype=np.int64)
            for i, v in enumerate(yt_list):
                if v is None:
                    continue
                idx = self._cls_to_idx.get(v, -1)
                if idx < 0:
                    if len(missing_yt) < 5:
                        missing_yt.append(v)
                else:
                    yt_idx[i] = idx
            for i, v in enumerate(yp_list):
                if v is None:
                    continue
                idx = self._cls_to_idx.get(v, -1)
                if idx < 0:
                    if len(missing_yp) < 5:
                        missing_yp.append(v)
                else:
                    yp_idx[i] = idx
            if missing_yt:
                raise ValueError(
                    f"MetricsOp: y_true has labels not in classes "
                    f"{self._classes}: {missing_yt} (first 5)."
                )
            if missing_yp:
                raise ValueError(
                    f"MetricsOp: y_pred has labels not in classes "
                    f"{self._classes}: {missing_yp} (first 5)."
                )
            valid = (yt_idx >= 0) & (yp_idx >= 0)
            yt = yt_idx[valid]
            yp = yp_idx[valid]
            same = yt == yp
            diff = ~same
            np.add.at(state["tp"], yt[same], 1)
            np.add.at(state["fp"], yp[diff], 1)
            np.add.at(state["fn"], yt[diff], 1)

        return state'''

    s = replace_line_block(s, old_update_tail, new_update_tail, tag="metrics-update-classification")

    # Update state_nbytes
    old_state_nbytes = '''    def state_nbytes(self, state):
        return 8 * 7  # 7 counters × 8 B'''

    new_state_nbytes = '''    def state_nbytes(self, state):
        base = 8 * 7  # 7 counters × 8 B
        if self._needs_classification:
            base += state["tp"].nbytes + state["fp"].nbytes + state["fn"].nbytes
        return base'''

    s = replace_line_block(s, old_state_nbytes, new_state_nbytes, tag="metrics-state-nbytes")

    # Update finalize() to compute classification metrics
    old_finalize_tail = '''        if "log_loss" in self.metrics:
            results["log_loss"] = (
                state["sum_log_loss"] / state["n_proba"]
                if state["n_proba"] > 0 else nan
            )

        return results'''

    new_finalize_tail = '''        if "log_loss" in self.metrics:
            results["log_loss"] = (
                state["sum_log_loss"] / state["n_proba"]
                if state["n_proba"] > 0 else nan
            )

        if self._needs_classification:
            results.update(self._finalize_classification(state))

        return results

    def _finalize_classification(self, state) -> dict:
        tp = state["tp"].astype(np.float64)
        fp = state["fp"].astype(np.float64)
        fn = state["fn"].astype(np.float64)

        # Per-class precision / recall / f1 (zero_division=0, sklearn)
        pp = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        pr = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = pp + pr
        pf1 = np.where(denom > 0, 2.0 * pp * pr / denom, 0.0)

        total = float(tp.sum() + fp.sum())  # = n_total
        accuracy = float(tp.sum()) / total if total > 0 else float("nan")

        results: dict = {}
        for metric in self.metrics:
            if metric == "accuracy":
                results["accuracy"] = accuracy
            elif metric == "precision":
                results["precision"] = self._average_value(
                    pp, tp, fn, accuracy,
                )
            elif metric == "recall":
                results["recall"] = self._average_value(
                    pr, tp, fn, accuracy,
                )
            elif metric == "f1":
                results["f1"] = self._average_value(
                    pf1, tp, fn, accuracy,
                )
        return results

    def _average_value(self, per_class, tp, fn, accuracy):
        avg = self._average_method
        if avg == "micro":
            return accuracy  # single-label multiclass
        if avg == "macro":
            return float(per_class.mean())
        if avg == "weighted":
            support = tp + fn  # true count per class
            total_support = float(support.sum())
            if total_support <= 0:
                return float("nan")
            return float((per_class * support).sum() / total_support)
        if avg == "binary":
            pi = self._cls_to_idx[self._pos_label]
            return float(per_class[pi])
        raise RuntimeError(
            f"_average_value: unknown average {avg!r} (internal bug)"
        )'''

    s = replace_line_block(s, old_finalize_tail, new_finalize_tail, tag="metrics-finalize-classification")

    p.write_text(s)
    compile_check(p)
    print("  OK: TP/FP/FN state + average modes")


# ============ 3. metrics() dispatch in dataframe.py ============

def extend_metrics_facade():
    print("\n=== cadw/dataframe.py::metrics — classification dispatch ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "classes: \"Sequence | None\"" in s:
        print("  already present — skip")
        return

    # Extend signature + dispatch
    old_sig = '''    def metrics(
        self,
        *,
        y_true: str,
        metrics: "Sequence[str] | str",
        y_pred: "str | None" = None,
        y_proba: "str | None" = None,
    ) -> dict:
        """Regression + log_loss metrics in one pass (Reduce-based).

        Metrics:
          - r2, mse, rmse, mae — require `y_pred` (numeric).
          - log_loss — requires `y_proba` (binary, values in [0,1]).
          - Combinations of regression + log_loss allowed in one call if
            both y_pred and y_proba provided.

        Args:
            y_true: column with true values.
            metrics: metric name(s) — scalar or list.
            y_pred: predicted values (float). Required for
                r2/mse/rmse/mae.
            y_proba: predicted probabilities (float in [0,1]).
                Required for log_loss.

        Returns:
            dict `{metric_name: value}`. Values are NaN for empty input
            (after filter).

        Nulls in y_true / y_pred / y_proba → row skipped (skipna=True).

        State: O(1) (7 counters). Independent of N.
        """
        from cadw.ops.metrics import MetricsOp, SUPPORTED_METRICS
        from cadw.ops.reduce import reduce

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = list(metrics)
        if not metrics:
            raise ValueError("metrics: metrics list is empty")

        # §13.4/162: no generic "at least one required" check —
        # `MetricsOp.__init__` raises per-metric specific error
        # ("y_pred_col required for ['r2']" / "y_proba_col required for
        # ['log_loss']"). Better UX: specific > generic.

        for c, label in (
            (y_true, "y_true"), (y_pred, "y_pred"), (y_proba, "y_proba")
        ):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )

        op = MetricsOp(
            y_true, y_pred_col=y_pred, y_proba_col=y_proba,
            metrics=metrics,
        )
        return reduce(
            self._iter_filtered_batches(), op,
            self._memory_guard, log=False,
        )'''

    new_sig = '''    def metrics(
        self,
        *,
        y_true: str,
        metrics: "Sequence[str] | str",
        y_pred: "str | None" = None,
        y_proba: "str | None" = None,
        classes: "Sequence | None" = None,
        average: "str | None" = None,
        pos_label: int = 1,
    ) -> dict:
        """Regression + log_loss + classification metrics in one pass.

        Metrics:
          - r2, mse, rmse, mae — require `y_pred` (numeric, continuous).
          - log_loss — requires `y_proba` (binary, values in [0,1]).
          - accuracy, precision, recall, f1 — require `y_pred` (labels,
            numeric or string).

        Args:
            y_true: column with true values.
            metrics: metric name(s) — scalar or list.
            y_pred: predicted values (float or labels — depends on
                metric group).
            y_proba: predicted probabilities (float in [0,1], binary).
            classes: explicit class list for classification. K=2 default
                when classes=None.
            average: 'binary' | 'macro' | 'micro' | 'weighted' for
                precision/recall/f1. Auto: 'binary' (K=2) / 'macro' (K>2).
            pos_label: positive class for average='binary'.

        Returns:
            dict `{metric_name: value}`. NaN for empty input.

        Restrictions:
          - Regression (r2/mse/rmse/mae) and classification
            (accuracy/...) cannot share y_pred in one call.
          - Classification K×24+64 > state_limit → MemoryLimitExceeded.

        State: O(1) + O(K) (classification). Independent of N.
        """
        from cadw.ops.metrics import (
            CLASSIFICATION_METRICS, REGRESSION_METRICS,
            SUPPORTED_METRICS, MetricsOp,
        )
        from cadw.ops.reduce import reduce

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = list(metrics)
        if not metrics:
            raise ValueError("metrics: metrics list is empty")
        unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"metrics: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )

        for c, label in (
            (y_true, "y_true"), (y_pred, "y_pred"), (y_proba, "y_proba")
        ):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )

        # §13.4/164: fail-fast for classification K too large — before
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

    s = replace_line_block(s, old_sig, new_sig, tag="metrics-classification-dispatch")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 4. __init__.py export update ============

def update_init():
    print("\n=== cadw/__init__.py: export CLASSIFICATION_METRICS ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()

    old = "from .ops.metrics import MetricsOp, SUPPORTED_METRICS\n"
    if old not in s:
        raise SystemExit("metrics import anchor not found")

    new = "from .ops.metrics import (\n    MetricsOp, SUPPORTED_METRICS,\n    REGRESSION_METRICS, PROBA_METRICS, CLASSIFICATION_METRICS,\n)\n"
    s = s.replace(old, new, 1)

    # __all__
    old_all = '    "MetricsOp", "SUPPORTED_METRICS",\n'
    if old_all in s:
        new_all = (
            '    "MetricsOp", "SUPPORTED_METRICS",\n'
            '    "REGRESSION_METRICS", "PROBA_METRICS",\n'
            '    "CLASSIFICATION_METRICS",\n'
        )
        s = s.replace(old_all, new_all, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 5. Tests ============

TESTS_APPEND = '''

# ============ v0.9.2b: classification metrics ============

def test_metrics_accuracy_binary_perfect(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1, 0, 1], "yp": [0, 1, 0, 1]})
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(1.0)


def test_metrics_accuracy_binary_mixed(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1, 0, 1], "yp": [0, 1, 1, 1]})
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1],
    )
    # 3 correct out of 4
    assert res["accuracy"] == pytest.approx(0.75)


def test_metrics_accuracy_auto_binary_default(tmp_path):
    """classes=None → K=2 default."""
    df = make_df(tmp_path, {"y": [0, 1, 0, 1], "yp": [0, 1, 1, 0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["accuracy"])
    assert res["accuracy"] == pytest.approx(0.5)


def test_metrics_precision_recall_f1_binary(tmp_path):
    """y=[1,1,1,0,0], yp=[1,0,1,1,0] → TP=2, FP=1, FN=1.
    precision=2/3, recall=2/3, f1=2/3."""
    df = make_df(tmp_path, {
        "y": [1, 1, 1, 0, 0],
        "yp": [1, 0, 1, 1, 0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision", "recall", "f1"],
        classes=[0, 1],
        average="binary", pos_label=1,
    )
    assert res["precision"] == pytest.approx(2.0 / 3.0)
    assert res["recall"] == pytest.approx(2.0 / 3.0)
    assert res["f1"] == pytest.approx(2.0 / 3.0)


def test_metrics_precision_binary_pos_label_0(tmp_path):
    """pos_label=0: precision for class 0."""
    df = make_df(tmp_path, {
        "y": [1, 1, 1, 0, 0],
        "yp": [1, 0, 1, 1, 0],
    })
    # class 0: TP=1 (y=0,yp=0), FP=0, FN=1
    # precision(class0) = 1/(1+0) = 1.0
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision"], classes=[0, 1],
        average="binary", pos_label=0,
    )
    assert res["precision"] == pytest.approx(1.0)


def test_metrics_accuracy_multiclass(tmp_path):
    """3 classes, y=yp → accuracy 1.0; classes explicit."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2, 0, 1, 2],
        "yp": [0, 1, 2, 0, 1, 2],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy", "precision", "recall", "f1"],
        classes=[0, 1, 2],
    )
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)


def test_metrics_multiclass_macro_micro(tmp_path):
    """3-class macro vs micro differ on class imbalance."""
    df = make_df(tmp_path, {
        "y": [0, 0, 0, 0, 1, 2],
        "yp": [0, 0, 1, 2, 1, 2],
    })
    macro = df.metrics(
        y_true="y", y_pred="yp", metrics=["precision"],
        classes=[0, 1, 2], average="macro",
    )
    micro = df.metrics(
        y_true="y", y_pred="yp", metrics=["precision"],
        classes=[0, 1, 2], average="micro",
    )
    acc = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1, 2],
    )
    # Per-class precision: c0=2/2=1.0, c1=1/1=1.0, c2=0/1=0.0
    # macro = (1+1+0)/3 = 2/3; micro = accuracy = 4/6 = 2/3
    assert micro["precision"] == pytest.approx(acc["accuracy"])
    assert macro["precision"] == pytest.approx(2.0 / 3.0)


def test_metrics_multiclass_weighted(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 0, 0, 0, 1, 2],
        "yp": [0, 0, 1, 2, 1, 2],
    })
    # Support: c0=4, c1=1, c2=1. precision: 1.0, 1.0, 0.0
    # weighted = (4·1 + 1·1 + 1·0) / 6 = 5/6
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["precision"],
        classes=[0, 1, 2], average="weighted",
    )
    assert res["precision"] == pytest.approx(5.0 / 6.0)


def test_metrics_classification_zero_division(tmp_path):
    """No positive predictions → precision=0.0 (sklearn zero_division=0)."""
    df = make_df(tmp_path, {
        "y": [1, 1, 0],
        "yp": [0, 0, 0],  # never predicts 1
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision"], classes=[0, 1],
        average="binary", pos_label=1,
    )
    assert res["precision"] == 0.0


def test_metrics_classification_nulls_skipped(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, None, 1, 0],
        "yp": [0, 1, None, 0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1],
    )
    # valid rows: y=0 yp=0 (correct), y=0 yp=0 (correct) → accuracy=1.0
    assert res["accuracy"] == pytest.approx(1.0)


def test_metrics_classification_empty(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0, 1]})
    empty = df[df["y"] > 100]
    res = empty.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"], classes=[0, 1],
    )
    assert math.isnan(res["accuracy"])


def test_metrics_classification_filtered(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 0, 1, 0],
        "yp": [0, 1, 1, 1, 1],  # errors on rows 2, 4
    })
    filtered = df[df["y"] == 0]  # only rows 0, 2, 4
    res = filtered.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"], classes=[0, 1],
    )
    # correct: row 0 (0,0); wrong: rows 2,4 (0,1); acc = 1/3
    assert res["accuracy"] == pytest.approx(1.0 / 3.0)


def test_metrics_classification_unknown_label_raises(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 2],  # 2 not in classes
        "yp": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="y_true has labels not in"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["accuracy"],
            classes=[0, 1],
        )


def test_metrics_classification_unknown_pred_label_raises(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 0],
        "yp": [0, 5, 0],
    })
    with pytest.raises(ValueError, match="y_pred has labels not in"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["accuracy"],
            classes=[0, 1],
        )


def test_metrics_classification_and_regression_conflict(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0, 1]})
    with pytest.raises(ValueError, match="cannot share y_pred"):
        df.metrics(
            y_true="y", y_pred="yp",
            metrics=["r2", "accuracy"], classes=[0, 1],
        )


def test_metrics_average_binary_on_multiclass_raises(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 2], "yp": [0, 1, 2],
    })
    with pytest.raises(ValueError, match="average='binary' requires K=2"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["precision"],
            classes=[0, 1, 2], average="binary",
        )


def test_metrics_invalid_average_raises(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0, 1]})
    with pytest.raises(ValueError, match="average must be one of"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["precision"],
            classes=[0, 1], average="bogus",
        )


def test_metrics_classification_k_too_large_raises(tmp_path):
    """K=10000 → state 240 KB > state_limit (2000 B budget)."""
    df = make_df(tmp_path, {"y": [0], "yp": [0]})
    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded, match="state"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["accuracy"],
            classes=list(range(10000)),
        )


def test_metrics_classification_string_labels(tmp_path):
    df = make_df(tmp_path, {
        "y": ["cat", "dog", "cat", "dog"],
        "yp": ["cat", "dog", "dog", "dog"],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy"], classes=["cat", "dog"],
    )
    assert res["accuracy"] == pytest.approx(0.75)


def test_metrics_classification_combined_with_log_loss(tmp_path):
    """classification + log_loss — different columns, OK."""
    df = make_df(tmp_path, {
        "y": [0, 1, 0, 1],
        "yp": [0, 1, 1, 0],
        "p": [0.1, 0.9, 0.6, 0.4],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["accuracy", "log_loss"], classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(0.5)
    assert res["log_loss"] > 0


def test_metrics_classification_pandas_equivalence(tmp_path):
    """Compare to manual numpy reference for binary precision/recall/f1."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, size=n).astype(np.int64)
    yp = (y + rng.integers(0, 2, size=n)) % 2  # noisy
    df = make_df(tmp_path, {
        "y": y.tolist(), "yp": yp.tolist(),
    }, chunk=20)
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy", "precision", "recall", "f1"],
        classes=[0, 1], average="binary", pos_label=1,
    )
    tp = int(((y == 1) & (yp == 1)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    total = n
    ref_acc = (tp + (total - tp - fp - fn)) / total
    ref_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ref_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ref_f1 = (2 * ref_prec * ref_rec / (ref_prec + ref_rec)
              if (ref_prec + ref_rec) > 0 else 0.0)
    assert res["accuracy"] == pytest.approx(ref_acc, rel=1e-9)
    assert res["precision"] == pytest.approx(ref_prec, rel=1e-9)
    assert res["recall"] == pytest.approx(ref_rec, rel=1e-9)
    assert res["f1"] == pytest.approx(ref_f1, rel=1e-9)
'''


E2E_TESTS_APPEND = '''

@pytest.mark.slow
def test_metrics_100k_classification_multiclass(tmp_path):
    """100K rows, 3-class, all metrics verified vs manual numpy."""
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    yp = y.copy()
    flip_mask = rng.random(n) < 0.2  # 20% errors
    yp[flip_mask] = rng.integers(0, 3, size=int(flip_mask.sum()))

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy", "precision", "recall", "f1"],
        classes=[0, 1, 2], average="macro",
    )

    # Manual reference
    K = 3
    tp = np.zeros(K, dtype=np.int64)
    fp = np.zeros(K, dtype=np.int64)
    fn = np.zeros(K, dtype=np.int64)
    for k in range(K):
        tp[k] = int(((y == k) & (yp == k)).sum())
        fp[k] = int(((y != k) & (yp == k)).sum())
        fn[k] = int(((y == k) & (yp != k)).sum())
    pp = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    pr = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    denom = pp + pr
    pf1 = np.where(denom > 0, 2 * pp * pr / denom, 0.0)
    acc = float(tp.sum()) / float(tp.sum() + fp.sum())
    assert res["accuracy"] == pytest.approx(acc, rel=1e-9)
    assert res["precision"] == pytest.approx(float(pp.mean()), rel=1e-9)
    assert res["recall"] == pytest.approx(float(pr.mean()), rel=1e-9)
    assert res["f1"] == pytest.approx(float(pf1.mean()), rel=1e-9)
'''


def write_tests():
    print("\n=== tests/test_metrics.py: append classification tests ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()
    if "test_metrics_accuracy_binary_perfect" in s:
        print("  already present — skip")
        return
    if not s.endswith("\n"):
        s += "\n"
    s += TESTS_APPEND
    p.write_text(s)
    compile_check(p)
    print("  OK")

    print("\n=== tests/test_metrics_e2e.py: append classification E2E ===")
    p = Path("tests/test_metrics_e2e.py")
    s = p.read_text()
    if "test_metrics_100k_classification_multiclass" in s:
        print("  already present — skip")
        return
    if not s.endswith("\n"):
        s += "\n"
    s += E2E_TESTS_APPEND
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 6. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.2b via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2b — classification metrics" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.2b — classification metrics)

- **`accuracy`, `precision`, `recall`, `f1`** в `CadwDataFrame.metrics()`:
  - binary (K=2, `pos_label` = 1 по умолчанию) и multiclass (K≥3).
  - **`average`** ∈ `'binary'|'macro'|'micro'|'weighted'` (auto: binary
    if K=2, macro if K>2).
  - `classes` — explicit class list (K=2 default when None).
  - Numeric или string labels — dict lookup, nulls skipped.
- **`MetricsOp`** state: `tp[K] + fp[K] + fn[K]` int64 = `24×K` bytes
  (vs confusion matrix `K²×8 = 8 MB` для K=1000 — 320× дешевле для
  scalar metrics).
- **`ReduceOp.batch_memory_factor()`** — protocol extension.
  Classification path uses `.to_pylist()` per batch (7× Python-object
  overhead). `reduce()` passes to `check_batch(extra_factor=...)` —
  honest peak RAM, not hidden.
- **`SUPPORTED_METRICS`** = regression + proba + classification
  (exported from `cadw`).

### Design decisions (locked-in, v0.9.2b)

- **TP/FP/FN state** вместо confusion matrix. Accuracy/precision/
  recall/f1 — все выводятся из TP/FP/FN per class. K² state не нужен
  (confusion matrix добавим отдельным методом, если запросят).
- **Average auto-switch**: binary (K=2) / macro (K>2). Matches sklearn
  semantics; user может переопределить.
- **zero_division=0** (sklearn default): precision/recall/f1 при
  пустом знаменателе → 0.0, не NaN.
- **Mixed regression + classification** на одном `y_pred` → ValueError
  (continuous vs labels).
- **`log_loss` + classification** — OK (разные колонки: y_proba vs
  y_pred).
- **Fail-fast**: `K × 24 + 64 > state_limit` → `MemoryLimitExceeded`
  в `metrics()` до `MetricsOp` construction.

### Ограничения (v0.9.2b, §11.5)

- **`roc_auc`** — v0.9.2c (Sort-based, требует полного proba массива).
- **Multiclass log_loss** (proba matrix) — v0.9.2c.
- **Confusion matrix** itself — не возвращается; если запросят —
  отдельный метод (не `metrics()` scalar API).
- **Class mismatch** y_true/y_pred not in `classes` → ValueError
  (fail-fast, первые 5 invalid labels).
- **Empty input** → accuracy=NaN (как regression).
- **`average='binary'`** требует K=2, `pos_label` в `classes`.

### Тесты (v0.9.2b)

- +20 fast (`test_metrics.py`): accuracy binary perfect/mixed/auto,
  precision/recall/f1 binary pos_label, multiclass macro/micro/weighted,
  zero-division, nulls, empty, filtered, unknown label, mixed
  regression conflict, average=binary on K>2, invalid average, K too
  large, string labels, log_loss + classification, pandas equivalence.
- +1 slow (`test_metrics_e2e.py`): 100K rows, 3-class, manual numpy ref.

- fast: **~1441** (+20). slow: **17** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: classification note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.2b (classification metrics)" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**v0.9.2b (classification metrics):** accuracy/precision/"
        s += "recall/f1 binary+multiclass. State tp/fp/fn per class. "
        s += "roc_auc — v0.9.2c.\n"
        p.write_text(s)
        return

    note = '''**v0.9.2b (classification metrics):** `accuracy`, `precision`,
`recall`, `f1` (binary + multiclass).
- State: `tp[K] + fp[K] + fn[K]` int64 = `24×K` bytes.
- `average` ∈ `binary|macro|micro|weighted`; auto binary (K=2) / macro
  (K>2).
- `classes` explicit for K>2; K=2 default when None.
- zero_division=0 (sklearn default).
- Mixed regression + classification on same `y_pred` → ValueError.
- Classification path uses `.to_pylist()` per batch; `reduce()` passes
  `check_batch(extra_factor=8.0)` (honest peak).
- Fail-fast: `K × 24 + 64 > state_limit` → `MemoryLimitExceeded`.
- **`roc_auc`** — v0.9.2c (sort-based).
- **Multiclass log_loss** (proba matrix) — v0.9.2c.
- **Confusion matrix** — separate method if requested, not scalar.
- `average='binary'` requires K=2 and `pos_label` in `classes`.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ 7. §13.4/164 (if bugs encountered) ============
# Not added preemptively — add after test run if needed.


# ============ main ============

def main():
    print("=== v0.9.2b: classification metrics ===")

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

    extend_reduce()
    extend_metrics_op()
    extend_metrics_facade()
    update_init()
    write_tests()
    append_changelog()
    update_decisions()

    print("\n=== pytest: metrics fast (including classification) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
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

    print("\n=== pytest: reduce regression (protocol change) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
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
        "v0.9.2b: classification metrics — accuracy/precision/recall/f1 "
        "(binary + multiclass, average modes); MetricsOp tp/fp/fn state "
        "(24×K vs K² confusion); ReduceOp.batch_memory_factor protocol "
        "extension; §11.5 classification constraints"])

    print("\n==> v0.9.2b applied.")
    print("NEXT: v0.9.2c (roc_auc + multiclass log_loss).")


if __name__ == "__main__":
    main()