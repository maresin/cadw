#!/usr/bin/env bash
# patch_02b2-fix: 3 бага в 02b2 (A, B, C) + скрытый D (tee duplicate).
set -euo pipefail

BR="patch/02b2-skew-isolation"
CUR="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CUR" != "$BR" ]]; then
    echo "ERROR: expected branch ${BR}, got ${CUR}"; exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-02b2-fix-${TS}"
mkdir -p ".backups/${BK}"
git diff > ".backups/${BK}/wip.diff"
echo "WIP diff saved: .backups/${BK}/wip.diff"

# ---- Fix A + D: sample без tee (без дубликатов), всегда строится ----
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()

old_block = '''    if num_partitions is None:
        # Sample первых ~10k строк, оценить K, выбрать P.
        # Бюджет на партицию — половина available: вторая половина нужна
        # на батчи чтения и состояние агрегации.
        sample_cap = 10_000
        sample_iter, rest_iter = itertools.tee(batch_iter, 2)
        sample_batches: list[pa.RecordBatch] = []
        n_sample = 0
        for b in sample_iter:
            sample_batches.append(b)
            n_sample += b.num_rows
            if n_sample >= sample_cap:
                break
        sample_bytes = sum(b.nbytes for b in sample_batches)
        sample_limit = int(memory_guard.available_bytes() * 0.5)
        if sample_bytes > sample_limit:
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B). "
                f"Укажите num_partitions явно, чтобы избежать sample."
            )

        batch_iter = itertools.chain(sample_batches, rest_iter)

        k_est = _estimate_k_sample(sample_batches, by_columns)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "groupby_aggregate: auto num_partitions=%d (k_est=%d, sample=%d rows)",
                num_partitions, k_est, n_sample,
            )
'''

new_block = '''    # Sample строится ВСЕГДА: он нужен и для auto-P, и для skew detection.
    # Важно: break-and-chain, не tee. С tee chain(sample, rest) дублирует
    # sample (count/sum удваиваются, mean маскирует).
    sample_cap = 10_000
    batch_iter = iter(batch_iter)
    sample_batches: list[pa.RecordBatch] = []
    n_sample = 0
    for b in batch_iter:
        sample_batches.append(b)
        n_sample += b.num_rows
        if n_sample >= sample_cap:
            break
    sample_bytes = sum(b.nbytes for b in sample_batches)
    sample_limit = int(memory_guard.available_bytes() * 0.5)
    if sample_bytes > sample_limit:
        raise MemoryLimitExceeded(
            f"groupby_aggregate: sample ({sample_bytes} B) > "
            f"available*0.5 ({sample_limit} B). "
            f"Укажите num_partitions явно, чтобы избежать sample."
        )
    batch_iter = itertools.chain(sample_batches, batch_iter)

    if num_partitions is None:
        k_est = _estimate_k_sample(sample_batches, by_columns)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "groupby_aggregate: auto num_partitions=%d (k_est=%d, sample=%d rows)",
                num_partitions, k_est, n_sample,
            )
'''

if old_block not in s:
    raise SystemExit("Fix A/D: sample block not found (было ли применено 02b2?)")
s = s.replace(old_block, new_block, 1)
open(p, "w").write(s)
print("Fix A/D: sample always built, no duplication")
PY

# ---- Fix B: _accumulate_heavy on Table ----
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()
old = '''    grouped = batch.select(list(by_columns) + list(agg_columns)).group_by(
        list(by_columns)
    ).aggregate(specs)'''
new = '''    # .group_by есть у pa.Table, не у RecordBatch.
    table = pa.Table.from_batches([batch.select(list(by_columns) + list(agg_columns))])
    grouped = table.group_by(list(by_columns)).aggregate(specs)'''
if old not in s:
    raise SystemExit("Fix B: pattern not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("Fix B: RecordBatch -> Table in _accumulate_heavy")
PY

# ---- Fix C: test_groupby_90_10_skew ----
python3 - <<'PY'
p = "tests/test_mustfix_skew.py"
s = open(p).read()
old = '''    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["sum", "count"],
        guard, tmp_path, num_partitions=4, log=False,
    )'''
new = '''    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val", "val"], ["sum", "count"],
        guard, tmp_path, num_partitions=4, log=False,
    )'''
if old not in s:
    raise SystemExit("Fix C: pattern not found")
s = s.replace(old, new, 1)

# ---- Add 2 tests: no-duplication invariant ----
extra_tests = '''

# ---------- no-duplication (Fix D) ----------

def test_no_sample_duplication_heavy(tmp_path):
    """Все ключи heavy: count не удваивается (sample без tee)."""
    table = pa.table({"cat": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val"], ["count"],
        guard, tmp_path, num_partitions=None, log=False,
    )
    rows = {r["cat"]: r["val_count"] for r in result.to_pylist()}
    assert rows == {"A": 2, "B": 2}, f"sample duplicated: {rows}"


def test_no_sample_duplication_light(tmp_path):
    """Все ключи light: count=1, не 2."""
    n = 200
    table = pa.table({"cat": [f"k{i}" for i in range(n)], "val": [1] * n})
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        make_iter(table, 20), ["cat"], ["val"], ["count"],
        guard, tmp_path, num_partitions=None, log=False,
    )
    assert result.num_rows == n
    for r in result.to_pylist():
        assert r["val_count"] == 1, f"key {r['cat']} count={r['val_count']}"
'''
if "test_no_sample_duplication_heavy" not in s:
    s += extra_tests
open(p, "w").write(s)
print("Fix C + 2 duplication tests")
PY

# ---- Run tests ----
set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests still failing (rc=${RC}). Not committing."
    exit 1
fi

git add -A
git commit -m "02b2-fix: sample без tee (no dup), RecordBatch→Table в _accumulate_heavy, test agg_columns, +2 dup-tests"

echo "==> sha256"
sha256sum cadw/ops/groupby.py tests/test_mustfix_skew.py
echo
echo "==> 02b2-fix applied on ${BR}"
echo "    WIP diff: .backups/${BK}/wip.diff"
echo "    Revert:   git reset --hard 80851dd"