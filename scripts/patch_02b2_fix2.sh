#!/usr/bin/env bash
# patch_02b2-fix2: 2 бага (E, F) из fix1.
set -euo pipefail

BR="patch/02b2-skew-isolation"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-02b2-fix2-${TS}"
git diff > ".backups/backup-pre-02b2-fix2-${TS}/wip.diff"

# ---- Fix E: _accumulate_heavy — дедупликация select-колонок ----
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()
old = '''    # .group_by есть у pa.Table, не у RecordBatch.
    table = pa.Table.from_batches([batch.select(list(by_columns) + list(agg_columns))])
    grouped = table.group_by(list(by_columns)).aggregate(specs)'''
new = '''    # .group_by есть у pa.Table, не у RecordBatch.
    # Дедупликация: agg_columns может содержать повторы (одна колонка,
    # несколько агрегаций), но select с дубликатом даёт ArrowInvalid.
    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))
    table = pa.Table.from_batches([batch.select(select_cols)])
    grouped = table.group_by(list(by_columns)).aggregate(specs)'''
if old not in s:
    raise SystemExit("Fix E: _accumulate_heavy body not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("Fix E: _accumulate_heavy dedup select")
PY

# ---- Fix F: sample > limit при явном num_partitions — skip skew, не падать ----
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()
old = '''    sample_bytes = sum(b.nbytes for b in sample_batches)
    sample_limit = int(memory_guard.available_bytes() * 0.5)
    if sample_bytes > sample_limit:
        raise MemoryLimitExceeded(
            f"groupby_aggregate: sample ({sample_bytes} B) > "
            f"available*0.5 ({sample_limit} B). "
            f"Укажите num_partitions явно, чтобы избежать sample."
        )
    batch_iter = itertools.chain(sample_batches, batch_iter)'''
new = '''    sample_bytes = sum(b.nbytes for b in sample_batches)
    sample_limit = int(memory_guard.available_bytes() * 0.5)
    if sample_bytes > sample_limit:
        if num_partitions is None:
            # P вывести не из чего, а sample не влезает — отказ.
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B) и num_partitions не задан. "
                f"Передайте num_partitions явно."
            )
        # Пользователь задал P явно: sample нужен только для skew detection.
        # Если не влезает — пропускаем skew, полагаемся на P и рекурсию (02b1).
        if log:
            logger.warning(
                "groupby_aggregate: sample (%d B) > available*0.5 (%d B); "
                "skew detection skipped (num_partitions=%d задан явно)",
                sample_bytes, sample_limit, num_partitions,
            )
        sample_batches = []
    batch_iter = itertools.chain(sample_batches, batch_iter)'''
if old not in s:
    raise SystemExit("Fix F: sample limit block not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("Fix F: sample>limit при явном P — skip skew")
PY

set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests still failing (rc=${RC}). Not committing."
    exit 1
fi

git add -A
git commit -m "02b2-fix2: _accumulate_heavy dedup (agg_columns с повторами), sample>limit при явном P — skip skew"

echo "==> sha256"
sha256sum cadw/ops/groupby.py
echo "==> 02b2-fix2 applied on ${BR}"