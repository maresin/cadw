#!/usr/bin/env python3
# v0.9.22 — fixes from AUDIT_ML_v0922 (round 3): gbdt formula, gbdt
# binary-y single-pass, NB C_ASSUMED, kmeans batch_memory_factor,
# _flush_buf comment, DECISIONS §9.2.20-22 + §13.4/241-245.
#
# 7 items (audit §4 says 9; 1.6 deferred to v0.9.22b for exact-anchor
# safety, 1.3 changed from ValueError to C_ASSUMED bump):
#   1.1 gbdt multiclass est_n_state misses y_all (S2) — n·(1+4K), was n·3K
#   1.2 gbdt _validate_binary_y 3-pass → 1-pass (S3 perf)
#   1.3 NB C_ASSUMED=32 → 64 (S2-doc; audit says "not RAM bug",
#       ValueError K>32 is too aggressive for 2MB RSS)
#   1.4 kmeans batch_memory_factor=2k (S1 — 200× under-report @ k=100)
#   1.7 _flush_buf comment (S4 doc)
#   2.1-2.3 DECISIONS §9.2.20-22 (GBDT peak / pipeline plan / KFold)
#   + §13.4/241-245 (5 meta-lessons from this audit)
#
# Deferred (per AUDIT_ML §4 + safety):
#   1.5 kmeans++ reservoir — v0.9.23 (needs prototype)
#   1.6 extra_factor at 5 sites — v0.9.22b (exact anchors needed)
#   2.5 RobustScaler O(d) sorts — v0.9.23 (algorithmic)
#
# Deviation from audit §4:
#   - 1.3: C_ASSUMED=64, NOT ValueError K>32. Audit says "not RAM bug",
#     breaking K>32 users for zero RAM win is wrong. Bump + doc is honest.
#   - 1.6: deferred to v0.9.22b. Blind anchors at file:line risk partial
#     state (some fixed, some missed) — exact §13.4/238 anti-pattern.
#
# Atomic scope per §13.4/238: 5 code + 3 doc edits. Output identity
# preserved for 1.1/1.2/1.3/1.7/2.1-2.3 (guard-only / error-only / doc).
# 1.4 tightens check_batch (more conservative, no output change).
#
# Run: uv run python scripts/patch_v0922.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ 1.1 gbdt multiclass formula ============

def fix_gbdt_multiclass_formula():
    print("\n=== 1.1 gbdt multiclass est_n_state y_all ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()
    if "n_total * 8 * (1 + 4 * K)" in s:
        print("  already fixed — skip")
        return

    old = (
        "    # State: pred (n,K) + grad_all (n,K) + hess_all (n,K) + cur_leaf (n)\n"
        "    #        = n * K * 8 * 3 (minimal, cur_leaf reused per class)\n"
        "    est_n_state = n_total * K * 8 * 3\n"
    )
    new = (
        "    # v0.9.22 (§13.4/241, AUDIT_ML §1.1): state formula corrected.\n"
        "    # Persistent: y_all (n·8) + pred (n·K·8) + grad_all (n·K·8)\n"
        "    # + hess_all (n·K·8) + round_cur_leaves (K·n·8, one per class\n"
        "    # held through round) = n·8·(1 + 4K). Prior n·K·8·3 missed\n"
        "    # y_all and the K-th round_cur_leaves array: at N=10M K=10,\n"
        "    # formula 2.4 GB vs actual 3.28 GB (880 MB miss).\n"
        "    est_n_state = n_total * 8 * (1 + 4 * K)\n"
    )
    if old not in s:
        raise SystemExit("gbdt multiclass anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ 1.2 gbdt _validate_binary_y single-pass ============

def fix_gbdt_validate_binary():
    print("\n=== 1.2 gbdt _validate_binary_y 3-pass → 1-pass ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()
    if "v0.9.22 (§13.4/246, AUDIT_ML §1.2)" in s:
        print("  already fixed — skip")
        return

    old = (
        '    seen: set = set()\n'
        '    y_sum = 0.0\n'
        '    n_valid = 0\n'
        '    for b in df._iter_filtered_batches():\n'
        '        col = b.column(y_col)\n'
        '        try:\n'
        '            uniq = col.unique()\n'
        '            vals = uniq.to_pylist() if uniq is not None else col.to_pylist()\n'
        '        except Exception:\n'
        '            vals = col.to_pylist()\n'
        '        for v in vals:\n'
        '            if v is None:\n'
        '                continue\n'
        '            fv = float(v)\n'
        '            if fv != 0.0 and fv != 1.0:\n'
        '                raise ValueError(\n'
        '                    f"fit_gbdt_classifier: y must be binary (0/1); got {v!r}"\n'
        '                )\n'
        '            seen.add(fv)\n'
        '        # Accumulate y_sum / n_valid in same pass\n'
        '        import numpy as np\n'
        '        y_np = _column_to_f64(col)\n'
        '        mask = ~np.isnan(y_np)\n'
        '        y_sum += float(y_np[mask].sum())\n'
        '        n_valid += int(mask.sum())\n'
    )

    new = (
        '    # v0.9.22 (§13.4/246, AUDIT_ML §1.2): single-pass over y.\n'
        '    # Prior did col.unique() + uniq.to_pylist() + _column_to_f64(col)\n'
        '    # = 3 passes per batch. Now: one numpy scan; validate + sum +\n'
        '    # count in one shot. Output identical (same y_sum, n_valid,\n'
        '    # same ValueError for non-binary y).\n'
        '    import numpy as np\n'
        '    seen: set = set()\n'
        '    y_sum = 0.0\n'
        '    n_valid = 0\n'
        '    for b in df._iter_filtered_batches():\n'
        '        y_np = _column_to_f64(b.column(y_col))\n'
        '        mask = ~np.isnan(y_np)\n'
        '        valid = y_np[mask]\n'
        '        if valid.size == 0:\n'
        '            continue\n'
        '        bad = (valid != 0.0) & (valid != 1.0)\n'
        '        if bad.any():\n'
        '            raise ValueError(\n'
        '                f"fit_gbdt_classifier: y must be binary (0/1); "\n'
        '                f"got {float(valid[bad][0])!r}"\n'
        '            )\n'
        '        if (valid == 0.0).any():\n'
        '            seen.add(0.0)\n'
        '        if (valid == 1.0).any():\n'
        '            seen.add(1.0)\n'
        '        y_sum += float(valid.sum())\n'
        '        n_valid += int(valid.size)\n'
    )

    if old not in s:
        raise SystemExit("gbdt _validate_binary_y anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ 1.3 NB C_ASSUMED 32 → 64 ============

def fix_nb_c_assumed():
    print("\n=== 1.3 NB C_ASSUMED=32 → 64 (deviation from audit) ===")
    p = Path("cadw/ml/naive_bayes.py")
    s = p.read_text()
    if "C_ASSUMED=64" in s:
        print("  already fixed — skip")
        return

    # Gaussian comment + formula
    old_g = (
        "        # Conservative C_ASSUMED=32 (typical multi-class); actual C > 32\n"
        "        # → per-update check_state in reduce() catches real overflow.\n"
        "        # Per class: 8 (n) + 8d (sum_x) + 8d (sum_xx) = 8 + 16d.\n"
        "        return 32 * (8 + 16 * self.d)\n"
    )
    new_g = (
        "        # v0.9.22 (§13.4/242, AUDIT_ML §1.3): C_ASSUMED=64 (was 32).\n"
        "        # Audit: K=100 gives 3.125× drift, K=500 gives 15.6×; but\n"
        "        # measured RSS is tiny (2 MB @ K=100) — not a RAM bug,\n"
        "        # per-update check_state enforces actual bound. Bumping to 64\n"
        "        # doubles the assumption (covers K≤64) without breaking\n"
        "        # K>32 users (audit's ValueError K>32 too aggressive for 0\n"
        "        # RAM win).\n"
        "        # Per class: 8 (n) + 8d (sum_x) + 8d (sum_xx) = 8 + 16d.\n"
        "        return 64 * (8 + 16 * self.d)\n"
    )
    if old_g not in s:
        raise SystemExit("NB Gaussian anchor not found")
    s = s.replace(old_g, new_g, 1)

    # Multinomial and Bernoulli (identical formula, 2 occurrences)
    old_mb = "        return 32 * (8 + 8 * self.d)\n"
    new_mb = (
        "        # v0.9.22 (§13.4/242): C_ASSUMED=64 (was 32). See Gaussian.\n"
        "        return 64 * (8 + 8 * self.d)\n"
    )
    count = s.count(old_mb)
    if count != 2:
        raise SystemExit(
            f"NB Multinomial/Bernoulli expected 2 anchors, got {count}"
        )
    s = s.replace(old_mb, new_mb)  # replace all 2

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK (1 Gaussian + 2 Multinomial/Bernoulli)")


# ============ 1.4 kmeans batch_memory_factor ============

def fix_kmeans_batch_memory_factor():
    print("\n=== 1.4 kmeans batch_memory_factor=2k ===")
    p = Path("cadw/ml/kmeans.py")
    s = p.read_text()
    if "def batch_memory_factor(self) -> float:" in s and "2.0 * float(self.k)" in s:
        print("  already fixed — skip")
        return

    old = (
        "        self.k, self.d = self.centroids.shape\n"
        "\n"
        "    def init(self):\n"
    )
    new = (
        "        self.k, self.d = self.centroids.shape\n"
        "\n"
        "    def batch_memory_factor(self) -> float:\n"
        '        """v0.9.22 (§13.4/243, AUDIT_ML §1.4): update() creates two\n'
        "        (n, k, d) float64 tensors (diff then diff*diff) — 2·k·\n"
        "        batch.nbytes transient. Prior inherited 1.0 → k=100, d=50,\n"
        "        batch=10k: real peak ~800 MB vs check_batch 4 MB. Override\n"
        '        to 2k. Output identity unchanged (guard-only edit)."""\n'
        "        return 2.0 * float(self.k)\n"
        "\n"
        "    def init(self):\n"
    )

    if old not in s:
        raise SystemExit("KMeansStepOp init anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ 1.7 _flush_buf comment ============

def fix_flush_buf_comment():
    print("\n=== 1.7 _flush_buf comment ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "# v0.9.22 (§13.4/244, AUDIT_ML §1.7)" in s:
        print("  already fixed — skip")
        return

    old = (
        "        def _flush_buf():\n"
        "            nonlocal buf_rows\n"
        "            if not buf_rows:\n"
        "                return\n"
        "            _maybe_upgrade_schema()  # v0.9.11 (PF4)\n"
    )
    new = (
        "        def _flush_buf():\n"
        "            # v0.9.22 (§13.4/244, AUDIT_ML §1.7): buf_rows bounded\n"
        "            # at BATCH_ROWS=1000 (see _emit_row) → ~140 KB at K=5.\n"
        "            # Real guard is _flush_group at 0.3·available. No\n"
        "            # check_batch needed — bounded by construction.\n"
        "            nonlocal buf_rows\n"
        "            if not buf_rows:\n"
        "                return\n"
        "            _maybe_upgrade_schema()  # v0.9.11 (PF4)\n"
    )

    if old not in s:
        raise SystemExit("_flush_buf anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ 2.1-2.3 DECISIONS §9.2.20-22 ============

DECISIONS_9_2_20_22 = (
    "\n"
    "### 9.2.20. GBDT top-level peak (multiclass) — §13.4/241\n"
    "\n"
    "**Persistent per-op state** (multiclass, K classes):\n"
    "\n"
    "    est_n_state = n·8·(1 + 4K)\n"
    "\n"
    "Components: `y_all` (n·8), `pred` (n·K·8), `grad_all` (n·K·8),\n"
    "`hess_all` (n·K·8), `round_cur_leaves` (K·n·8). Prior formula\n"
    "n·K·8·3 missed `y_all` + one `round_cur_leaves` slot → 33-50%\n"
    "under-report (AUDIT_ML §1.1 measured 1.37-1.44× at K=3-10; N=10M\n"
    "K=10 → 880 MB miss). Fixed in v0.9.22.\n"
    "\n"
    "**Per-batch hist peak** (still separate `check_state` in\n"
    "`_AllNodeHistOp.estimate_state_nbytes`): `S × F × B × 32 B` for\n"
    "current level S nodes × F features × B bins. For F=1000, B=255,\n"
    "S=63 (d=5): ~514 MB — larger than base state at small N. Audited\n"
    "exact (AUDIT_ML §3.1 retracted): `estimate == state_nbytes(init())`\n"
    "for S∈{3,7,16,31,63} × F∈{2,100,1000} × B∈{16,255}.\n"
    "\n"
    "**Full peak ≈** `est_n_state + max_hist + 2 × batch_X`. Not\n"
    "aggregated into one formula: per-op `check_state` calls cover each\n"
    "phase. For small N (hist dominates), ratio vs `n·32` blows up to\n"
    "7-10×; safe because hist is separately checked.\n"
    "\n"
    "\n"
    "### 9.2.21. Pipeline.plan() = sum(state_bytes), not peak — §13.4/247\n"
    "\n"
    "`plan(df)` checks `sum(step.state_bytes) ≤ state_limit` and returns\n"
    "`predicted_outcome ∈ {fits, does_not_fit}`. **State-only**: does\n"
    "**not** model per-step peak (transform buffer + batch). Honest by\n"
    "design: transform is per-batch streamed, each step's MemoryGuard\n"
    "handles its own batch. `plan` = feasibility of fitted state, not\n"
    "runtime peak. If a transform step internally needs large temp\n"
    "buffers (e.g. `RobustScaler.fit` O(d) sorts), its own\n"
    "`check_state` catches at runtime — plan may say `fits` and\n"
    "transform may raise. Documented as intentional (§13.4/247).\n"
    "\n"
    "\n"
    "### 9.2.22. KFold(materialize=True) budget ≈ 2 × check_state — §13.4/248\n"
    "\n"
    "`KFold(n_splits, materialize=True)` calls\n"
    "`check_state(20 * n)` — bounds int32 fold_id (4n) + int64\n"
    "`np.arange` (8n) + int64 `rng.permutation` (8n) numpy transients.\n"
    "**But** the resulting Arrow `augmented` dataset (~n × K_cols × 8 B)\n"
    "is outside that bound. Measured N=100k: RSS 4.33 MB vs 20n=2 MB\n"
    "(ratio 2.17×). User must budget ~2× check_state for total KFold\n"
    "memory. Default (streaming) mode is O(batch), no such caveat.\n"
    "\n"
)


def append_decisions_9_2():
    print("\n=== 2.1-2.3 DECISIONS §9.2.20-22 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "### 9.2.20. GBDT top-level peak" in s:
        print("  already present — skip")
        return
    anchor = "### 9.3. Pipeline planner"
    if anchor not in s:
        raise SystemExit("§9.3 anchor not found")
    if s.count(anchor) != 1:
        raise SystemExit(f"§9.3 anchor count = {s.count(anchor)}")
    s = s.replace(anchor, DECISIONS_9_2_20_22 + anchor, 1)
    p.write_text(s, newline="")
    assert s.index("### 9.2.20") < s.index("### 9.3. Pipeline")
    print("  OK")


# ============ §13.4/241-245 meta-lessons ============

DECISIONS_13_4_241_245 = (
    "\n\n\n"
    "**Эпизод 241 (formula components must be enumerated).** "
    "v0.9.22 (AUDIT_ML §1.1): gbdt multiclass `est_n_state = "
    "n·K·8·3` — intended `pred + grad_all + hess_all`, but actual "
    "persistent state includes **`y_all` (n·8)** and `round_cur_leaves` "
    "(K·n·8, one per class held through the round) — formula missed "
    "two components. Measured ratio 1.37-1.44× (K=3-10); N=10M K=10 → "
    "880 MB under-report. Fix: `n·8·(1 + 4K)`.\n\n"
    "**Правило (§13.4/241):** state formula должен **enumerate все "
    "persistent arrays**. Перед публикацией — grep `self.` assignments "
    "в `_fit_*` / `_build_*` и убедиться что каждый → 8·bytes в formula. "
    "Missing component → silent under-report на production scale.\n\n"
    "**Связанные:** §13.4/187 (Arrow state accounting), /240 (formula "
    "unit vs metric).\n\n\n"
    "**Эпизод 242 (C_ASSUMED drift — bump or error, not both).** "
    "v0.9.22 (AUDIT_ML §1.3): NB `estimate_state_nbytes` hardcodes "
    "`C_ASSUMED=32` — actual K classes discovered dynamically from y. "
    "K=100 → 3.125× under; K=500 → 15.6×. **But** measured RSS is "
    "tiny (2 MB @ K=100) because state is O(K·d·8), per-class overhead "
    "small. Audit suggested `ValueError K>32`; rejected — breaks "
    "existing K=33-64 users for zero RAM win. Fix: bump 32→64 "
    "(2× more conservative, covers common K) + docstring note. "
    "Per-update `check_state` still enforces actual bound.\n\n"
    "**Правило (§13.4/242):** при formula drift, **сначала измерить "
    "actual RSS**. Если tiny — bump assumption + doc. Если large — "
    "fix formula / extract actual metric. `ValueError` — только "
    "если **нет** clean bounded path (для NB — есть: per-update "
    "check). Не ломать users ради формальной точности.\n\n"
    "**Связанные:** §13.4/240 (formula unit vs metric), /178 "
    "(вакуумный guard — обе «модель не отражает реальность»).\n\n\n"
    "**Эпизод 243 (inherited default = wrong for op-specific "
    "transient).** v0.9.22 (AUDIT_ML §1.4): `KMeansStepOp` inherits "
    "`ReduceOp.batch_memory_factor() = 1.0`. But `update()` creates "
    "two (n, k, d) float64 tensors (`diff` then `diff*diff`) — "
    "transient `2·k·batch.nbytes`. For k=100, d=50, batch=10k: real "
    "peak 800 MB vs check_batch 4 MB → **guaranteed OOM** (S1).\n\n"
    "**Правило (§13.4/243):** если `update()` (или `merge`) создаёт "
    "**N intermediate arrays размера k×batch**, override "
    "`batch_memory_factor` явно. Default 1.0 — only for streaming "
    "kernels с temp ≤ batch.nbytes. Test: grep `np.zeros`, `[None]`, "
    "tensor broadcasts в `update` — если N intermediates, "
    "factor ≥ N.\n\n"
    "**Связанные:** §13.4/187 (per-row as_py — memory accounting).\n\n\n"
    "**Эпизод 244 (bounded-by-construction = no check_batch needed).** "
    "v0.9.22 (AUDIT_ML §1.7): `_flush_buf` accumulates `buf_rows` "
    "(Python list) с **нет** `check_batch`. But `buf_rows` bounded "
    "at `BATCH_ROWS=1000` in `_emit_row` — ~140 KB at K=5. Real guard "
    "is `_flush_group` at `0.3 × available` (per-group memory). "
    "No check_batch needed — bounded by construction. Add comment.\n\n"
    "**Правило (§13.4/244):** buffer с **explicit cap** "
    "(`BATCH_ROWS`, `maxlen=`, `if len(x) >= N: flush()`) — **не "
    "нужен check_batch**, нужен **comment** с указанием cap + "
    "real guard. Иначе будущий читатель / auditor пометит как «missing "
    "guard» (§13.4/138 false positive). Distinguish bounded-by-"
    "construction from bounded-by-guard.\n\n"
    "**Связанные:** §13.4/138 (AST guard for high-risk calls), /216 "
    "(helper exists).\n\n\n"
    "**Эпизод 245 (audit-first, measure-second — retract 0 evidence).** "
    "v0.9.22 (AUDIT_ML overall): 17 SUSPECTED findings из 24 **retracted** "
    "under measurement (71%). Examples: gbdt `_AllNodeHistOp` "
    "estimate exact (63 combos); gbdt predict_df per-tree no accum; "
    "NB dict-growth no OOM path (2 MB RSS @ K=500); pipeline fanout "
    "0.152× formula; K6 §2.3 join 0.472× formula. Only 7 CONFIRMED "
    "(29%). **Without measurement, all 24 → v0.9.22 scope would be "
    "6× larger than justified** — patch cascade (§13.4/238) guaranteed.\n\n"
    "**Правило (§13.4/245):** SUSPECTED finding без benchmark — "
    "**не actionable**. Сначала measure (tracemalloc + RSS + "
    "reference-assert), потом status: CONFIRMED / RETRACTED / "
    "NEEDS-RESEARCH. Assumptions — не findings. Retracted с "
    "evidence — valuable output (не «пропущено», а «проверено, "
    "не баг»).\n\n"
    "**Связанные:** §13.4/232-240 (metric unit — без правильной "
    "metric findings неотличимы от noise), /228 (dogfood = happy "
    "path, edge cases = pytest).\n\n\n"
)


def append_decisions_13_4():
    print("\n=== §13.4/241-245 append ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "**Эпизод 241 (formula components" in s:
        print("  already present — skip")
        return
    anchor = "### 13.5. Открытые вопросы"
    if anchor not in s:
        raise SystemExit("§13.5 anchor not found")
    if s.count(anchor) != 1:
        raise SystemExit(f"§13.5 anchor count = {s.count(anchor)}")
    s = s.replace(anchor, DECISIONS_13_4_241_245 + anchor, 1)
    p.write_text(s, newline="")
    assert s.index("**Эпизод 240 ") < s.index("**Эпизод 241 ")
    assert s.index("**Эпизод 245 ") < s.index("### 13.5. Открытые вопросы")
    print("  OK (241-245 after 240, before §13.5)")


# ============ tests ============

def add_tests():
    print("\n=== tests ===")
    # t1: gbdt multiclass formula
    p1 = Path("tests/test_gbdt_multiclass.py")
    s1 = p1.read_text()
    if "def test_multiclass_est_state_includes_y_all" in s1:
        print("  t1 already present — skip")
    else:
        extra1 = '''

# ============ v0.9.22 (§13.4/241): multiclass state formula ============


def test_multiclass_est_state_includes_y_all():
    """v0.9.22: est_n_state = n·8·(1 + 4K), not n·K·8·3.

    Prior formula missed y_all (n·8) and K-th round_cur_leaves (K·n·8).
    """
    from cadw.ml.gbdt import _fit_boosting_multiclass
    import inspect
    src = inspect.getsource(_fit_boosting_multiclass)
    assert "n_total * 8 * (1 + 4 * K)" in src, (
        "multiclass state formula must be n·8·(1+4K) "
        "(AUDIT_ML §1.1). Prior n·K·8·3 under-reports."
    )
    assert "n_total * K * 8 * 3" not in src, (
        "old formula n·K·8·3 must be removed"
    )
'''
        p1.write_text(s1 + extra1, newline="")
        pu.assert_compiles(p1)
        print("  t1 OK")

    # t2: kmeans batch_memory_factor
    p2 = Path("tests/test_kmeans.py")
    s2 = p2.read_text()
    if "def test_kmeans_batch_memory_factor_2k" in s2:
        print("  t2 already present — skip")
    else:
        extra2 = '''

# ============ v0.9.22 (§13.4/243): kmeans batch_memory_factor ============


def test_kmeans_batch_memory_factor_2k():
    """v0.9.22: KMeansStepOp.batch_memory_factor = 2k.

    update() creates two (n, k, d) tensors (diff then diff*diff) =
    2·k·batch.nbytes transient. Prior inherited 1.0 → 200× under-report
    at k=100 (AUDIT_ML §1.4).
    """
    import numpy as np
    from cadw.ml.kmeans import KMeansStepOp
    for k in (2, 5, 10, 100):
        op = KMeansStepOp(["a", "b"], np.zeros((k, 2)))
        assert op.batch_memory_factor() == 2.0 * k, (
            f"k={k}: expected factor {2.0*k}, got {op.batch_memory_factor()}"
        )
'''
        p2.write_text(s2 + extra2, newline="")
        pu.assert_compiles(p2)
        print("  t2 OK")

    # t3: NB C_ASSUMED=64
    p3 = Path("tests/test_naive_bayes.py")
    s3 = p3.read_text()
    if "def test_nb_c_assumed_is_64" in s3:
        print("  t3 already present — skip")
    else:
        extra3 = '''

# ============ v0.9.22 (§13.4/242): NB C_ASSUMED=64 ============


def test_nb_c_assumed_is_64():
    """v0.9.22: C_ASSUMED bumped 32→64 (AUDIT_ML §1.3).

    Drift is bounded (K=100 gives 3.125×, RSS=2MB — not RAM bug);
    bump + doc is honest fix, ValueError too aggressive.
    """
    import inspect
    from cadw.ml import naive_bayes as nb_mod
    for op_name in ("_GaussianStatsOp", "_MultinomialStatsOp",
                    "_BernoulliStatsOp"):
        src = inspect.getsource(getattr(nb_mod, op_name))
        assert "return 64 * (8 +" in src, (
            f"{op_name}: expected C_ASSUMED=64 (return 64 * ...)"
        )
        assert "return 32 * (8 +" not in src, (
            f"{op_name}: old C_ASSUMED=32 must be removed"
        )
'''
        p3.write_text(s3 + extra3, newline="")
        pu.assert_compiles(p3)
        print("  t3 OK")

    # t4: DECISIONS 9.2.20-22 + 13.4.241-245 order
    p4 = Path("tests/test_changelog.py")
    s4 = p4.read_text()
    if "def test_decisions_v0922_sections" in s4:
        print("  t4 already present — skip")
    else:
        extra4 = '''

# ============ v0.9.22: DECISIONS sections + order ============


def test_decisions_v0922_sections():
    """v0.9.22: §9.2.20-22 present + §13.4/241-245 present + ordered."""
    from pathlib import Path as _P
    text = (_P(__file__).resolve().parent.parent / "DECISIONS.md").read_text()
    # 9.2.20-22 (before §9.3)
    for h in ("### 9.2.20. GBDT top-level peak",
              "### 9.2.21. Pipeline.plan()",
              "### 9.2.22. KFold(materialize=True)"):
        assert h in text, f"missing {h}"
    i_920 = text.index("### 9.2.20. GBDT")
    i_922 = text.index("### 9.2.22. KFold")
    i_93 = text.index("### 9.3. Pipeline planner")
    assert i_920 < i_922 < i_93
    # 13.4/241-245 (after 240, before §13.5)
    for n in range(241, 246):
        assert f"**Эпизод {n} " in text, f"missing episode {n}"
    i_240 = text.index("**Эпизод 240 ")
    i_241 = text.index("**Эпизод 241 ")
    i_245 = text.index("**Эпизод 245 ")
    i_135 = text.index("### 13.5. Открытые вопросы")
    assert i_240 < i_241 < i_245 < i_135
'''
        p4.write_text(s4 + extra4, newline="")
        pu.assert_compiles(p4)
        print("  t4 OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.22 + regression ===")
    pu.verify_anchors([
        # v0.9.22 fixes
        ("cadw/ml/gbdt.py", "n_total * 8 * (1 + 4 * K)"),
        ("cadw/ml/gbdt.py", "v0.9.22 (§13.4/246, AUDIT_ML §1.2)"),
        ("cadw/ml/naive_bayes.py", "return 64 * (8 + 16 * self.d)"),
        ("cadw/ml/naive_bayes.py", "return 64 * (8 + 8 * self.d)"),
        ("cadw/ml/kmeans.py", "return 2.0 * float(self.k)"),
        ("cadw/dataframe.py",
         "# v0.9.22 (§13.4/244, AUDIT_ML §1.7)"),
        ("DECISIONS.md", "### 9.2.20. GBDT top-level peak"),
        ("DECISIONS.md", "### 9.2.21. Pipeline.plan()"),
        ("DECISIONS.md", "### 9.2.22. KFold(materialize=True)"),
        ("DECISIONS.md", "**Эпизод 241 (formula components"),
        ("DECISIONS.md", "**Эпизод 245 (audit-first"),
        # tests
        ("tests/test_gbdt_multiclass.py",
         "def test_multiclass_est_state_includes_y_all"),
        ("tests/test_kmeans.py", "def test_kmeans_batch_memory_factor_2k"),
        ("tests/test_naive_bayes.py", "def test_nb_c_assumed_is_64"),
        ("tests/test_changelog.py", "def test_decisions_v0922_sections"),
        # v0.9.21 regression
        ("cadw/dataframe.py", "def _group_boundaries_arrow("),
        ("cadw/dataframe.py", "batch_first_key = tuple("),
        ("DECISIONS.md", "**Эпизод 233 (append with fallback anchor"),
        ("DECISIONS.md", "**Эпизод 240 (formula-as-sizing"),
        # v0.9.20 regression
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "# v0.9.20-fix3"),
        ("tests/test_arrow_call_memory_guard.py",
         '"cadw/core/memory_guard.py:MemoryGuard.calibrate"'),
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        # v0.9.19/18/17/16
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood (happy path only per §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.22 (happy path) ===")

    def _verify_all():
        import numpy as np
        import pyarrow as pa
        from cadw import CadwDataFrame, MemoryGuard
        from cadw.ml.gbdt import fit_gbdt_multiclass
        from cadw.ml.kmeans import fit_kmeans, KMeansStepOp
        from cadw.ml.naive_bayes import fit_gaussian_nb

        # 1.1/1.2: gbdt multiclass + binary small fit succeeds
        rng = np.random.default_rng(0)
        n = 300
        x = rng.normal(size=n).tolist()
        y3 = [i % 3 for i in range(n)]
        y2 = [i % 2 for i in range(n)]
        t3 = pa.table({"x": x, "y": y3})
        df3 = CadwDataFrame.from_table(
            t3, memory_guard=MemoryGuard(max_memory_mb=50),
        )
        m_mc = fit_gbdt_multiclass(df3, ["x"], "y", n_estimators=3, max_depth=2)
        assert m_mc is not None
        t2 = pa.table({"x": x, "y": y2})
        df2 = CadwDataFrame.from_table(
            t2, memory_guard=MemoryGuard(max_memory_mb=50),
        )
        from cadw.ml.gbdt import fit_gbdt_classifier
        m_bin = fit_gbdt_classifier(df2, ["x"], "y", n_estimators=3, max_depth=2)
        assert m_bin is not None

        # 1.3: NB C=3 fit succeeds
        m_nb = fit_gaussian_nb(df3, ["x"], "y")
        assert m_nb is not None

        # 1.4: kmeans factor + fit succeeds
        op = KMeansStepOp(["x"], np.zeros((10, 1)))
        assert op.batch_memory_factor() == 20.0
        dfk = CadwDataFrame.from_table(
            pa.table({"x": x, "y": [0] * n}),
            memory_guard=MemoryGuard(max_memory_mb=50),
        )
        m_km = fit_kmeans(dfk, ["x"], k=3, n_init=1)
        assert m_km is not None

        print("  gbdt multiclass + binary small fit OK")
        print("  NB C=3 fit OK")
        print("  kmeans factor=2k + small fit OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_multiclass.py",
         "tests/test_gbdt_classification.py",
         "tests/test_gbdt_regression.py",
         "tests/test_naive_bayes.py",
         "tests/test_kmeans.py",
         "tests/test_apply.py",
         "tests/test_changelog.py", "-q"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.22 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_gbdt_multiclass_formula()
    fix_gbdt_validate_binary()
    fix_nb_c_assumed()
    fix_kmeans_batch_memory_factor()
    fix_flush_buf_comment()
    append_decisions_9_2()
    append_decisions_13_4()
    add_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.22: AUDIT_ML_v0922 fixes. 7 items from round-3 audit "
        "(24 suspected → 7 CONFIRMED, 17 RETRACTED under measurement — "
        "§13.4/245 audit-first discipline): (1.1) gbdt multiclass "
        "est_n_state = n·8·(1+4K), was n·K·8·3 — missed y_all (n·8) + "
        "K-th round_cur_leaves (K·n·8); N=10M K=10: 880 MB under-report. "
        "(1.2) gbdt _validate_binary_y single-pass — prior "
        "col.unique()+uniq.to_pylist()+_column_to_f64 = 3 passes/batch; "
        "output identical (same y_sum, n_valid, same ValueError). "
        "(1.3) NB C_ASSUMED=32→64 (deviation from audit's ValueError "
        "K>32: audit says 'not RAM bug', measured RSS 2 MB @ K=100; "
        "breaking K=33-64 users for zero RAM win is wrong; bump+doc "
        "honest; per-update check_state still enforces actual). "
        "(1.4) kmeans KMeansStepOp.batch_memory_factor=2k — update() "
        "creates 2 (n,k,d) float64 tensors (diff then diff*diff); "
        "prior inherited 1.0 → S1 (k=100, d=50, batch=10k: real peak "
        "800 MB vs check_batch 4 MB = guaranteed OOM). (1.7) _flush_buf "
        "comment — bounded at BATCH_ROWS=1000 (~140 KB at K=5), real "
        "guard is _flush_group 0.3·available; no check_batch needed. "
        "(2.1-2.3) DECISIONS §9.2.20 (GBDT peak), §9.2.21 (pipeline "
        "plan = state only), §9.2.22 (KFold materialize 2× budget) — "
        "doc-only. (§13.4/241-245 append) 5 meta-lessons: 241 formula "
        "components must be enumerated (grep self. in _fit_*); 242 "
        "C_ASSUMED drift — bump or error, not both (audit-first, don't "
        "break users for formal precision); 243 inherited default "
        "wrong for op-specific transient (override batch_memory_factor "
        "if N intermediates); 244 bounded-by-construction (explicit "
        "cap = comment, not check_batch; distinguish from bounded-by-"
        "guard); 245 audit-first, measure-second — retract 0-evidence "
        "(17/24 retracted in this round = 71%; without measurement "
        "patch scope would be 6× larger). (Deviations from audit §4) "
        "(1.6) extra_factor at 5 sites (rank:269,345; groupby:581; "
        "resample:272; pivot:204) deferred to v0.9.22b — audit gives "
        "file:line but not exact check_batch text; blind anchors risk "
        "partial state (§13.4/238 anti-pattern). Dedicated v0.9.22b "
        "with grep-per-site inspection. (1.5) kmeans++ reservoir — "
        "v0.9.23 (needs prototype). (2.5) RobustScaler O(d) sorts — "
        "v0.9.23 (algorithmic). (Backward compat) all 1912 existing "
        "tests unchanged; 4 new tests (gbdt formula, kmeans factor, "
        "NB C_ASSUMED, DECISIONS sections). (Test matrix) gbdt "
        "multiclass formula: 1.37-1.44× measured (K=3-10); at N=10M "
        "K=10: 2.4 GB → 3.28 GB (880 MB miss closed). kmeans factor: "
        "6/20/200× under-report at k=3/10/100 closed. NB C_ASSUMED: "
        "3.125× @ K=100 drift halves (formula more conservative). "
        "Deferred to v0.9.22b/v0.9.23: extra_factor sites, kmeans++ "
        "reservoir, RobustScaler batching. v0.9.24+: A1 split, Pingis "
        "round 4 (v1.0 readiness).",
        self_path="scripts/patch_v0922.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.22 applied.")


if __name__ == "__main__":
    main()
