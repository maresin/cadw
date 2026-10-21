#!/usr/bin/env python3
# v0.9.17 — §13.4/178–189 append + seaborn filter + category-memory tests.
#
# Scope:
#   (1) DECISIONS.md: append §13.4/178–189 (12 distinct meta-lessons
#       from v0.9.7–v0.9.13, not covered by 211–225). Insert BEFORE
#       "**Эпизод 211" (v0.9.16 append). Note in §13.4 tail: 190–210
#       reserved (fewer distinct lessons than initially projected;
#       not fabricated).
#   (2) tests/conftest.py: filter specific MatplotlibDeprecationWarning
#       from seaborn/categorical.py:700 (third-party transition;
#       upgrade tracking → §11.6).
#   (3) tests/test_category_beyond_memory.py: verify CADW's core
#       out-of-core guarantee when K × per_key_bytes > available_bytes.
#       Sizes: K=100k (fast), K=1M (slow). Budgets: 10 MB, 1 MB.
#       Assert: succeed with K rows OR clean MemoryLimitExceeded (never
#       OOM). Includes §13.4/218 regression (null keys in skew path).
#
# Deferred to v0.9.18:
#   - A1: dataframe.py split (5404 lines → 7 modules).
#   - Pingis formal round 3 (fresh bench + snapshot).
#
# Run: uv run python scripts/patch_v0917.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (1) DECISIONS.md §13.4/178–189 append ============

DECISIONS_178_189 = [
    "",
    "**Эпизод 178 (guard, вакуумный по построению).** v0.9.8a "
    "(R1): `check_partition_buffers(P, P*buf)` с "
    "`buf = buffer_limit // P` — произведение "
    "`P × (limit//P) ≤ limit` всегда. Реальная стоимость "
    "ArrowIPCBatchWriter (1.2–4.4 KB/писатель; audit bench_r1) "
    "не участвует в проверке. P=5000 → 5.8 MB RSS; "
    "экстраполяция P=1e6 → 1.2 GB без срабатывания guard.",
    "",
    "**Правило (§13.4/178):** guard должен измерять **реальную "
    "стоимость** операции, а не величину, которая по построению "
    "≤ limit. `check_partition_writers(P, bytes_per_writer)` — "
    "правильная форма (явный per-writer cost, не P×size-of-one-buf).",
    "",
    "**Связанные:** §13.4/185 (per-row vs per-flush allocation — "
    "same class: модели памяти не отражают реальную стоимость).",
    "",
    "",
    "**Эпизод 179 (schema estimate хуже batch-based).** v0.9.8b "
    "(R2): `estimate_row_size_from_schema(default_string_bytes=50)` "
    "для 10 cols × 200 B/value даёт 500 B/row против 2040 B/row "
    "реально (10.1× низко). `get_batch_rows(ef=2) = 3670` → "
    "батч 7.1 MB при бюджете 5 MB. `check_batch` сработает, но "
    "**после** аллокации — sizing неверный.",
    "",
    "**Правило (§13.4/179):** для variable-length типов "
    "(string/binary) estimate с **первого batch** "
    "(`batch.nbytes / num_rows`), не из schema defaults. "
    "Schema-based оценка — только для fixed-width. "
    "`reestimate_from_batch` + `get_batch_rows(sample_batch=)`.",
    "",
    "**Связанные:** §13.4/175 (memory formula validity scope — "
    "formula, выведенная на одном типе, не переносится на другой).",
    "",
    "",
    "**Эпизод 180 (propagator/model asymmetry).** v0.9.8c (B1): "
    "`HashShuffleMemoryModel` зарегистрирован (`plan('hash_shuffle')` "
    "работает), но `propagate_chars(op.name='hash_shuffle')` → "
    "`ValueError: unsupported op` — hard crash. Не "
    "`PipelineMemoryError`.",
    "",
    "**Правило (§13.4/180):** memory model + output-chars "
    "propagator — **отдельные** контракты. Наличие `get_model(op)` "
    "не гарантирует `propagate_chars(op)`. Test для каждой новой "
    "op: `plan_pipeline([filter, op, sort])`.",
    "",
    "**Связанные:** §13.4/172 (registry growth — closed lists "
    "stale-bait; parallel: propagator list stale).",
    "",
    "",
    "**Эпизод 181 (full sort для top-n).** v0.9.7a (R4): "
    "`nlargest(n)` вызывал `external_sort` по всему N → "
    "`head(n)`. O(N log N) + O(N) дискового I/O при k ≪ N. "
    "Streaming per-batch `pc.select_k_unstable` + merge: "
    "**3946×** быстрее / **2.8×** меньше RSS (audit bench N=2M, "
    "k=10).",
    "",
    "**Правило (§13.4/181):** если целевой output — O(k) строк, "
    "полная сортировка не нужна. Partial selection (heap / "
    "`select_k`) + k-way merge. Спросить «какой k относительно N?» "
    "перед выбором алгоритма.",
    "",
    "**Связанные:** §13.4/182 (multi-pass replaceable by single-pass).",
    "",
    "",
    "**Эпизод 182 (multi-pass описание).** v0.9.7b (R5): "
    "`describe()` = 5 Reduce-проходов (count/mean/std/min/max) "
    "+ 1 external_sort per column. Fused single-pass (Welford + "
    "min/max + reservoir для quantiles): **109×** быстрее / "
    "**12.1×** меньше RSS (audit bench N=2M, 2 cols).",
    "",
    "**Правило (§13.4/182):** summary stats на одном столбце — "
    "fused в **один** `ReduceOp`. Multi-pass на полный dataset "
    "оправдан только когда stats действительно независимы "
    "(например, hyperloglog requires distinct state).",
    "",
    "**Связанные:** §13.4/181.",
    "",
    "",
    "**Эпизод 183 (pandas pivot dense matrix).** v0.9.6c2 (R6): "
    "`to_pandas().pivot()` materializes dense matrix: 200k rows, "
    "300×300 cardinality → 26.8 MB RSS vs 720 KB useful (37× "
    "overhead). 100k×20 cols passes where dense failed. Streaming "
    "`pivot_to_disk` (HashShuffle на (index, cols) + Sort + "
    "per-key buffer flush): ~10 MB.",
    "",
    "**Правило (§13.4/183):** pandas pivot — dense reshape; "
    "для K_index × K_columns ≫ buffer streaming reshape "
    "необходим. Peak = O(K_columns × buffer) + O(sort), не "
    "O(K_index × K_columns).",
    "",
    "**Связанные:** §9.2.18 (to_pandas cost model), §13.4/170 "
    "(whitelist by qualname — pivot был за whitelist).",
    "",
    "",
    "**Эпизод 184 (from_pylist int truncation).** v0.9.10 (C1): "
    "`_per_group_apply` first group returns int → "
    "`_infer_arrow_type_from_value(int)` = int64; later float "
    "`2.5` → `pa.Table.from_pylist(rows, schema=int64)` "
    "**silent truncates** до 2 (или ArrowInvalid в редких "
    "случаях). Fix: widen int→float64 при fractional float.",
    "",
    "**Правило (§13.4/184):** при mixed-type numeric groups — "
    "**widen** через один level (int → float64, float64 → "
    "строка не нужна — что не влезает, raise). Very large int "
    "(> 2^53) теряют precision на widen → §11.5 limitation.",
    "",
    "**Связанные:** §13.4/223 (bool→int misinference — same "
    "«schema ≠ actual type» class, other direction), §13.4/218.",
    "",
    "",
    "**Эпизод 185 (per-row hot-path allocation).** v0.9.11 "
    "(PF4): `_maybe_upgrade_for_row` вызывался на **каждую** "
    "строку `_emit_row`, аллоцировал `upgraded: list = []` "
    "(K элементов). Для 10M groups × K=10 → ~100M list allocs + "
    "~1B attr checks. Deferred to flush boundary "
    "(`_maybe_upgrade_schema` раз в BATCH_ROWS=1000): ~1000× "
    "fewer allocs.",
    "",
    "**Правило (§13.4/185):** hot-path per-row allocation с "
    "fixed-size per-call → defer to flush/finalize boundary. "
    "Bounded-по-batch state дешевле bounded-по-row.",
    "",
    "**Связанные:** §13.4/178, /186.",
    "",
    "",
    "**Эпизод 186 (recompute в цикле → incremental).** "
    "v0.9.12 (F12-1): `topk_to_disk` recompute "
    "`sum(tb.nbytes for tb in per_batch)` **на каждой** итерации "
    "→ O(P²). P=10000 → 100M итераций, ~1.6 GB attr accesses "
    "за один вызов. Incremental `acc_bytes += top.nbytes`.",
    "",
    "**Правило (§13.4/186):** если значение в цикле — "
    "**monoid** (sum/count/min/max), поддерживай инкрементально. "
    "Recompute = O(N×K), incremental = O(N).",
    "",
    "**Связанные:** §13.4/185 (per-row fix аналогичен: "
    "«не повторяй вычисление в hot path»).",
    "",
    "",
    "**Эпизод 187 (per-row as_py где есть Arrow primitive).** "
    "v0.9.12 (F12-4): `_detect_heavy_keys` делал "
    "`for i in range(b.num_rows): b.column(c)[i].as_py()` — "
    "10k Python iterations per groupby call (~10-50 ms). "
    "Arrow-native `pa.concat_tables + group_by().aggregate("
    "[(k, 'count')]) + filter`: single C++ pass, ~10× faster.",
    "",
    "**Правило (§13.4/187):** per-row Python loop где Arrow "
    "имеет aggregate kernel — заменяй. Exception: bounded "
    "case (K ≪ N, ~100 keys) — приемлемо, документируй.",
    "",
    "**Связанные:** §13.4/188 (same class, quantile path), "
    "/217 (precise AST check — detect exactly X[i].as_py()).",
    "",
    "",
    "**Эпизод 188 (per-value loop где есть Arrow take).** "
    "v0.9.13 (F12-5): `_stream_values_at_positions` итерировал "
    "**каждое** значение sorted file до `max_target` — O(N) "
    "Python iterations (~500 ms–2 s для N=10M). `pc.take("
    "non_null_arr, local_targets)` per batch: O(K) per batch, "
    "не O(N). ~100–400× для high quantiles.",
    "",
    "**Правило (§13.4/188):** если нужны значения по индексам "
    "`[i1, i2, ...]` (K targets), а не все N — Arrow `take`, "
    "не итерация. Спросить «сколько target'ов относительно N?» "
    "перед выбором цикла.",
    "",
    "**Связанные:** §13.4/187, /217.",
    "",
    "",
    "**Эпизод 189 (asymmetric null semantics — documented "
    "contract).** v0.9.13 (F12-8): `PredicateFilter.mask_for_batch` "
    "даёт numeric `_OPS` (gt/ge/lt/le/eq/ne) — Kleene "
    "null-propagation (`pc.greater(null, 5)` = null); "
    "`b.filter(null)` drops → matches pandas boolean indexing + "
    "SQL WHERE. String `_OPS_NO_SCALAR` (contains/starts_with/"
    "ends_with) — `fill_null(False)` → matches pandas "
    "`.str.contains(na=False)` (CADW default). **Asymmetry by "
    "design.** §13.4/218 (fill_null) применяется только к "
    "fill-default cases, не к Kleene-drop.",
    "",
    "**Правило (§13.4/189):** asymmetry в null semantics — "
    "**документируй как intentional contract**, не «унифицируй». "
    "Обе семантики согласованы с ecosystem (pandas/SQL), и "
    "unification сломает user expectation.",
    "",
    "**Связанные:** §13.4/218 (fill_null — где mandatory), "
    "/223 (bool→int — semantic class != this).",
    "",
    "",
    "**Gap 190–210 (reserved).** Initially projected 178–210 "
    "for v0.9.7–v0.9.13 lessons; после consolidation audit — "
    "**12 distinct meta-lessons** (178–189), не 33. Numbers "
    "190–210 зарезервированы, не fabricated. Fix-chain meta-"
    "lessons v0.9.10–v0.9.15 — 211–225. Итог: 178–189 "
    "(round-1 audit + Arrow-native migrations) + 211–225 "
    "(patch-script safety + AST precision).",
    "",
    "",
]


def append_178_189():
    print("\n=== (1) DECISIONS.md: §13.4/178–189 append ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "**Эпизод 178 (guard, вакуумный по построению)" in s:
        print("  already present — skip")
        return
    anchor = "**Эпизод 211 (compile-before-write"
    if anchor not in s:
        raise SystemExit("DECISIONS.md anchor '**Эпизод 211' not found")
    block = "\n".join(DECISIONS_178_189) + "\n"
    new_block = block + anchor
    s = s.replace(anchor, new_block, 1)
    p.write_text(s, newline="")
    # Sanity: order 177 < 178 < 189 < 190 gap-note < 211 < 225
    assert "**Эпизод 178 (guard, вакуумный" in s
    assert "**Эпизод 189 (asymmetric null semantics" in s
    assert "Gap 190–210 (reserved)" in s
    assert s.index("Эпизод 178") < s.index("Эпизод 189")
    assert s.index("Эпизод 189") < s.index("Gap 190–210")
    assert s.index("Gap 190–210") < s.index("Эпизод 211")
    print("  OK (178–189 + gap note inserted before 211)")


# ============ (2) seaborn warning filter ============

def filter_seaborn_warning():
    print("\n=== (2) tests/conftest.py: filter seaborn MatplotlibDeprecationWarning ===")
    p = Path("tests/conftest.py")
    s = p.read_text()
    if "vert: bool was deprecated" in s:
        print("  already filtered — skip")
        return
    # Append filter at module level (executes at import)
    extra = (
        "\n\n"
        "# v0.9.17 (§13.4/225 + third-party tracking): filter specific\n"
        "# MatplotlibDeprecationWarning from seaborn/categorical.py:700\n"
        "# (seaborn < 0.14 passes vert=True to ax.bxp; matplotlib 3.11\n"
        "# deprecates vert in favor of orientation). Third-party\n"
        "# transition — CADW code correct, upstream fix pending.\n"
        "# Tracking: §11.6 upgrade to seaborn>=0.14 backlog.\n"
        "import warnings as _warnings\n"
        "_warnings.filterwarnings(\n"
        "    'ignore',\n"
        "    message='vert: bool was deprecated in Matplotlib 3.11',\n"
        "    category=DeprecationWarning,\n"
        ")\n"
    )
    p.write_text(s + extra)
    pu.assert_compiles(p)
    print("  OK")


# ============ (3) category-beyond-memory tests ============

CATEGORY_TESTS_LINES = [
    '"""v0.9.17: verify out-of-core guarantee when K × per_key > available.',
    '',
    'These tests exercise the core CADW guarantee on the regime that',
    'justifies the whole framework: **unique group keys do NOT fit in',
    'memory**. `K × per_key_bytes` exceeds `available_bytes`; HashShuffle',
    'partitions by key hash; LocalGroupBy processes each partition;',
    'result is concat (partitions are key-disjoint).',
    '',
    'per_key_bytes ≈ 80 B (40 B key string + tuple/dict state overhead).',
    'K=100k needs ~8 MB hash table; K=1M needs ~80 MB.',
    '',
    'Contract:',
    '  - If P × (K/P) × per_key fits state_limit → succeed with K rows.',
    '  - If check_partition_writers fires → clean MemoryLimitExceeded,',
    '    never OOM.',
    '  - Null keys preserved (§13.4/218 regression).',
    '"""',
    'from __future__ import annotations',
    '',
    'import pyarrow as pa',
    'import pyarrow.parquet as pq',
    'import pytest',
    '',
    'from cadw import CadwDataFrame, MemoryGuard',
    'from cadw.core.memory_guard import MemoryLimitExceeded',
    'from cadw.ops.groupby import groupby_aggregate',
    '',
    '',
    'def _unique_key_table(n: int, *, extra_null: int = 0) -> pa.Table:',
    '    """All-unique string keys; optional extra null-key rows."""',
    '    keys = [f"k{i:07d}" for i in range(n)]',
    '    vals = [1] * n',
    '    if extra_null:',
    '        keys.extend([None] * extra_null)',
    '        vals.extend([1] * extra_null)',
    '    return pa.table({"k": keys, "v": vals})',
    '',
    '',
    'def _iter_batches(table: pa.Table, batch_size: int = 1000):',
    '    for i in range(0, table.num_rows, batch_size):',
    '        yield table.slice(i, batch_size).to_batches()[0]',
    '',
    '',
    '# ============ fast: K=100k ============',
    '',
    '',
    'def test_groupby_100k_unique_10mb(tmp_path):',
    '    """100k unique keys, 10 MB budget. K×per_key ≈ 8 MB →',
    '    P=64 → ~125 KB/partition fits state_limit (~5 MB).',
    '    Result: exactly 100k rows, each v_sum=1."""',
    '    n = 100_000',
    '    table = _unique_key_table(n)',
    '    guard = MemoryGuard(max_memory_mb=10)',
    '    result = groupby_aggregate(',
    '        _iter_batches(table), ["k"], ["v"], ["sum"],',
    '        guard, tmp_path, num_partitions=None, log=False,',
    '    )',
    '    assert result.num_rows == n',
    '    # Spot-check 100 rows',
    '    for r in result.to_pylist()[:100]:',
    '        assert r["v_sum"] == 1',
    '',
    '',
    'def test_groupby_100k_unique_1mb_borderline(tmp_path):',
    '    """100k unique keys, 1 MB budget. K×per_key ≈ 8 MB ≫ 1 MB,',
    '    but P=64 (each 125 KB fits) and writer check ~131 KB ≤ 314 KB',
    '    → succeed. Borderline case validating auto-P + writer check."""',
    '    n = 100_000',
    '    table = _unique_key_table(n)',
    '    guard = MemoryGuard(max_memory_mb=1)',
    '    try:',
    '        result = groupby_aggregate(',
    '            _iter_batches(table), ["k"], ["v"], ["sum"],',
    '            guard, tmp_path, num_partitions=None, log=False,',
    '        )',
    '        assert result.num_rows == n',
    '    except MemoryLimitExceeded:',
    '        pass  # clean error also acceptable; assert not OOM',
    '',
    '',
    'def test_unique_100k_unique_10mb(tmp_path):',
    '    """df.unique("k") on 100k unique keys, 10 MB budget."""',
    '    n = 100_000',
    '    table = _unique_key_table(n)',
    '    p = tmp_path / "src.parquet"',
    '    pq.write_table(table, p)',
    '    df = CadwDataFrame.from_parquet(',
    '        p, tmp_path / "ds", chunk_size=5000,',
    '        memory_guard=MemoryGuard(max_memory_mb=10),',
    '        spill_dir=tmp_path / "spill",',
    '    )',
    '    result = df.unique("k")',
    '    assert result.count() == n',
    '',
    '',
    'def test_dedup_100k_unique_10mb(tmp_path):',
    '    """drop_duplicates(subset=["k"]) on 100k unique keys, 10 MB."""',
    '    n = 100_000',
    '    table = _unique_key_table(n)',
    '    p = tmp_path / "src.parquet"',
    '    pq.write_table(table, p)',
    '    df = CadwDataFrame.from_parquet(',
    '        p, tmp_path / "ds", chunk_size=5000,',
    '        memory_guard=MemoryGuard(max_memory_mb=10),',
    '        spill_dir=tmp_path / "spill",',
    '    )',
    '    result = df.drop_duplicates(',
    '        subset=["k"], keep="first", num_partitions=64,',
    '    )',
    '    assert result.count() == n',
    '',
    '',
    '# ============ §13.4/218 regression: null keys in K≫P regime ============',
    '',
    '',
    'def test_groupby_50k_unique_plus_50k_null_10mb(tmp_path):',
    '    """50k unique keys + 50k null-key rows, 10 MB budget.',
    '    §13.4/218 regression: nulls must survive HashShuffle +',
    '    LocalGroupBy (fill_null(False) in _compute_heavy_mask).',
    '    If nulls were dropped (old bug), result would have only',
    '    50k rows + sum(v_count) = 50k instead of 100k."""',
    '    table = _unique_key_table(50_000, extra_null=50_000)',
    '    guard = MemoryGuard(max_memory_mb=10)',
    '    result = groupby_aggregate(',
    '        _iter_batches(table), ["k"], ["v"], ["count"],',
    '        guard, tmp_path, num_partitions=32, log=False,',
    '    )',
    '    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}',
    '    assert rows.get(None) == 50_000, (',
    '        f"null-key rows dropped — got {rows.get(None)} null count, "',
    '        f"expected 50_000 (regression of §13.4/218 fix)"',
    '    )',
    '    assert sum(rows.values()) == 100_000',
    '',
    '',
    '# ============ slow: K=1M ============',
    '',
    '',
    '@pytest.mark.slow',
    'def test_groupby_1m_unique_10mb(tmp_path):',
    '    """1M unique keys, 10 MB budget. K×per_key ≈ 80 MB ≫ 10 MB.',
    '    P=64 partitions × ~15.6k keys each ≈ 1.25 MB/partition',
    '    (fits state_limit 5 MB) + writer check ~131 KB (fits).',
    '    Full out-of-core path: HashShuffle + N × LocalGroupBy + Concat."""',
    '    n = 1_000_000',
    '    table = _unique_key_table(n)',
    '    guard = MemoryGuard(max_memory_mb=10)',
    '    result = groupby_aggregate(',
    '        _iter_batches(table, 5000), ["k"], ["v"], ["sum"],',
    '        guard, tmp_path, num_partitions=None, log=False,',
    '    )',
    '    assert result.num_rows == n',
    '    for r in result.to_pylist()[:50]:',
    '        assert r["v_sum"] == 1',
    '',
    '',
    '@pytest.mark.slow',
    'def test_groupby_1m_unique_1mb_clean_raise(tmp_path):',
    '    """1M unique keys, 1 MB budget. P auto → ~400 → 512;',
    '    check_partition_writers(512, 2048) = ~1 MB > 0.3 MB limit',
    '    → clean MemoryLimitExceeded. NOT OOM (that would be silent',
    '    contract violation)."""',
    '    n = 1_000_000',
    '    table = _unique_key_table(n)',
    '    guard = MemoryGuard(max_memory_mb=1)',
    '    with pytest.raises(MemoryLimitExceeded):',
    '        groupby_aggregate(',
    '            _iter_batches(table, 5000), ["k"], ["v"], ["sum"],',
    '            guard, tmp_path, num_partitions=None, log=False,',
    '        )',
]


def add_category_tests():
    print("\n=== (3) tests/test_category_beyond_memory.py ===")
    p = Path("tests/test_category_beyond_memory.py")
    if p.exists():
        print("  already present — skip")
        return
    pu.write_content_checked(p, CATEGORY_TESTS_LINES)
    print("  written + compiled")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.17 + regression ===")
    pu.verify_anchors([
        # v0.9.17 (1)
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный по построению)"),
        ("DECISIONS.md", "**Эпизод 179 (schema estimate хуже batch-based)"),
        ("DECISIONS.md", "**Эпизод 180 (propagator/model asymmetry)"),
        ("DECISIONS.md", "**Эпизод 181 (full sort для top-n)"),
        ("DECISIONS.md", "**Эпизод 182 (multi-pass описание)"),
        ("DECISIONS.md", "**Эпизод 183 (pandas pivot dense matrix)"),
        ("DECISIONS.md", "**Эпизод 184 (from_pylist int truncation)"),
        ("DECISIONS.md", "**Эпизод 185 (per-row hot-path allocation)"),
        ("DECISIONS.md", "**Эпизод 186 (recompute в цикле"),
        ("DECISIONS.md", "**Эпизод 187 (per-row as_py где есть Arrow"),
        ("DECISIONS.md", "**Эпизод 188 (per-value loop где есть Arrow take)"),
        ("DECISIONS.md", "**Эпизод 189 (asymmetric null semantics"),
        ("DECISIONS.md", "Gap 190–210 (reserved)"),
        # v0.9.17 (2)
        ("tests/conftest.py", "vert: bool was deprecated in Matplotlib 3.11"),
        ("tests/conftest.py", "v0.9.17 (§13.4/225"),
        # v0.9.17 (3)
        ("tests/test_category_beyond_memory.py", "test_groupby_100k_unique_10mb"),
        ("tests/test_category_beyond_memory.py", "test_groupby_1m_unique_10mb"),
        ("tests/test_category_beyond_memory.py",
         "test_groupby_1m_unique_1mb_clean_raise"),
        ("tests/test_category_beyond_memory.py",
         "test_groupby_50k_unique_plus_50k_null_10mb"),
        # v0.9.16 regression (DECISIONS 211-225)
        ("DECISIONS.md", "**Эпизод 211 (compile-before-write"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        # v0.9.15/14 regression
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("pyproject.toml", 'version = "0.9.14"'),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.17 via subprocess ===")

    def _verify_all():
        from pathlib import Path as _P

        d = _P("DECISIONS.md").read_text()
        for n in range(178, 190):
            assert f"**Эпизод {n} " in d, f"DECISIONS.md missing 178–189 episode {n}"
        assert "Gap 190–210 (reserved)" in d
        assert d.index("Эпизод 177") < d.index("Эпизод 178") \
            if "Эпизод 177" in d else True
        assert d.index("Эпизод 189") < d.index("Gap 190–210")
        assert d.index("Gap 190–210") < d.index("Эпизод 211")
        assert d.index("Эпизод 211") < d.index("Эпизод 225")

        cc = _P("tests/conftest.py").read_text()
        assert "vert: bool was deprecated in Matplotlib 3.11" in cc
        assert "warnings.filterwarnings" in cc

        tm = _P("tests/test_category_beyond_memory.py").read_text()
        for name in (
            "test_groupby_100k_unique_10mb",
            "test_groupby_100k_unique_1mb_borderline",
            "test_unique_100k_unique_10mb",
            "test_dedup_100k_unique_10mb",
            "test_groupby_50k_unique_plus_50k_null_10mb",
            "test_groupby_1m_unique_10mb",
            "test_groupby_1m_unique_1mb_clean_raise",
        ):
            assert f"def {name}(" in tm, f"missing {name}"

        # Smoke: fast-path 100k actually runs end-to-end (not just
        # import-level). Budget 10 MB, expect success.
        import pyarrow as pa
        from cadw import MemoryGuard
        from cadw.ops.groupby import groupby_aggregate
        import tempfile
        from pathlib import Path as _PP
        with tempfile.TemporaryDirectory() as td:
            td = _PP(td)
            n = 100_000
            keys = [f"k{i:07d}" for i in range(n)]
            t = pa.table({"k": keys, "v": [1] * n})
            def _it():
                for i in range(0, n, 1000):
                    yield t.slice(i, 1000).to_batches()[0]
            guard = MemoryGuard(max_memory_mb=10)
            result = groupby_aggregate(
                _it(), ["k"], ["v"], ["sum"], guard, td,
                num_partitions=None, log=False,
            )
            assert result.num_rows == n, (
                f"smoke groupby(100k unique): got {result.num_rows} rows"
            )

        print("  DECISIONS 178–189 + gap note intact OK")
        print("  conftest seaborn filter OK")
        print("  test_category_beyond_memory: 7 tests present OK")
        print("  smoke: groupby(100k unique, 10 MB) succeeded OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target (fast) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_category_beyond_memory.py", "-q",
         "-m", "not slow"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE (fast subset) ***")
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
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.17 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    append_178_189()
    filter_seaborn_warning()
    add_category_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.17: §13.4/178–189 append + seaborn warning filter + "
        "category-beyond-memory tests. (1) DECISIONS.md: 12 distinct "
        "meta-lessons from v0.9.7-v0.9.13 — 178 (vacuous guard: "
        "check_partition_buffers P*(limit//P) ≤ limit by construction, "
        "real writer 1.2-4.4 KB unaccounted), 179 (schema estimate "
        "10× low for strings; reestimate from batch), 180 "
        "(propagator/model asymmetry: model registered, "
        "propagate_chars missing → hard crash), 181 (full external_sort "
        "for top-n → streaming top-k, 3946×), 182 (5-pass describe → "
        "fused single-pass, 109×), 183 (pandas pivot dense 37× "
        "overhead → streaming reshape), 184 (from_pylist int64 silent "
        "truncation mixed int/float groups → widen), 185 (per-row "
        "hot-path alloc → defer to flush, ~1000× fewer), 186 "
        "(recompute sum per iter O(P²) → incremental O(P)), 187 "
        "(per-row as_py loop → Arrow group_by, ~10×), 188 "
        "(per-value loop for quantile → pc.take, ~100-400×), 189 "
        "(asymmetric null semantics: numeric Kleene-drop vs string "
        "fill_null, documented intentional contract). Gap note "
        "190–210: reserved, fewer distinct lessons than projected, "
        "not fabricated. Insert before v0.9.16's 211–225. (2) "
        "tests/conftest.py: filter specific "
        "MatplotlibDeprecationWarning 'vert: bool was deprecated' from "
        "seaborn/categorical.py:700 (third-party transition, upgrade "
        "to seaborn>=0.14 deferred §11.6). (3) "
        "tests/test_category_beyond_memory.py: 7 tests verifying "
        "out-of-core guarantee when K × per_key_bytes > "
        "available_bytes. Fast: 100k unique keys 10 MB / 1 MB "
        "(borderline ok); unique(100k); drop_duplicates(100k); "
        "§13.4/218 null-key regression (50k unique + 50k null, "
        "assert None count = 50k). Slow: 1M unique keys 10 MB "
        "(succeeds, P=64 ~15.6k keys/partition fits); 1M unique 1 MB "
        "(clean MemoryLimitExceeded via check_partition_writers, "
        "not OOM).",
        self_path="scripts/patch_v0917.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.17 applied.")


if __name__ == "__main__":
    main()
