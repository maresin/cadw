#!/usr/bin/env bash
# patch_02b2-fix3: баги G (mean+явный sum) и H (sample потерян при skip skew).
set -euo pipefail

BR="patch/02b2-skew-isolation"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-02b2-fix3-${TS}"
git diff > ".backups/backup-pre-02b2-fix3-${TS}/wip.diff"

# ---- Fix G: _accumulate_heavy — внутренние ключи для mean ----
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()

old = '''def _accumulate_heavy(accum: dict, batch: pa.RecordBatch, by_columns, agg_columns, agg_funcs) -> None:
    specs = []
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            specs.extend([(col, "sum"), (col, "count")])
        else:
            specs.append((col, f))
    # .group_by есть у pa.Table, не у RecordBatch.
    # Дедупликация: agg_columns может содержать повторы (одна колонка,
    # несколько агрегаций), но select с дубликатом даёт ArrowInvalid.
    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))
    table = pa.Table.from_batches([batch.select(select_cols)])
    grouped = table.group_by(list(by_columns)).aggregate(specs)
    for row in grouped.to_pylist():
        key = tuple(row[c] for c in by_columns)
        if key not in accum:
            accum[key] = {}
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                sk, ck = f"{col}_sum", f"{col}_count"
                accum[key][sk] = accum[key].get(sk, 0.0) + row[sk]
                accum[key][ck] = accum[key].get(ck, 0) + row[ck]
            else:
                out = f"{col}_{f}"
                new = row[out]
                old = accum[key].get(out)
                if old is None:
                    accum[key][out] = new
                elif f in ("sum", "count"):
                    accum[key][out] = old + new
                elif f == "min":
                    accum[key][out] = min(old, new)
                elif f == "max":
                    accum[key][out] = max(old, new)'''

new = '''def _accumulate_heavy(accum: dict, batch: pa.RecordBatch, by_columns, agg_columns, agg_funcs) -> None:
    # Base-specs без дублей: (col, base_func).
    # mean -> (sum, count); остальное — как есть.
    needed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            for bf in ("sum", "count"):
                if (col, bf) not in seen:
                    seen.add((col, bf))
                    needed.append((col, bf))
        else:
            if (col, f) not in seen:
                seen.add((col, f))
                needed.append((col, f))

    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))
    table = pa.Table.from_batches([batch.select(select_cols)])
    grouped = table.group_by(list(by_columns)).aggregate(needed)

    for row in grouped.to_pylist():
        key = tuple(row[c] for c in by_columns)
        if key not in accum:
            accum[key] = {}
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                # Внутренние ключи — не пересекаются с явным sum/count.
                sk, ck = f"__mean_sum_{col}", f"__mean_count_{col}"
                accum[key][sk] = accum[key].get(sk, 0.0) + row[f"{col}_sum"]
                accum[key][ck] = accum[key].get(ck, 0) + row[f"{col}_count"]
            else:
                out = f"{col}_{f}"
                new = row[out]
                old = accum[key].get(out)
                if old is None:
                    accum[key][out] = new
                elif f in ("sum", "count"):
                    accum[key][out] = old + new
                elif f == "min":
                    accum[key][out] = min(old, new)
                elif f == "max":
                    accum[key][out] = max(old, new)'''

if old not in s:
    raise SystemExit("Fix G: _accumulate_heavy body not found")
s = s.replace(old, new, 1)

# _accum_to_table: mean из внутренних ключей
old_fin = '''def _accum_to_table(accum: dict, by_columns, agg_columns, agg_funcs) -> pa.Table:
    rows = []
    for key, d in accum.items():
        row = dict(zip(by_columns, key))
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                s, c = d[f"{col}_sum"], d[f"{col}_count"]
                row[f"{col}_mean"] = s / c if c > 0 else float("nan")
            else:
                row[f"{col}_{f}"] = d[f"{col}_{f}"]
        rows.append(row)'''
new_fin = '''def _accum_to_table(accum: dict, by_columns, agg_columns, agg_funcs) -> pa.Table:
    rows = []
    for key, d in accum.items():
        row = dict(zip(by_columns, key))
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                s = d.get(f"__mean_sum_{col}", 0.0)
                c = d.get(f"__mean_count_{col}", 0)
                row[f"{col}_mean"] = s / c if c > 0 else float("nan")
            else:
                row[f"{col}_{f}"] = d[f"{col}_{f}"]
        rows.append(row)'''
if old_fin not in s:
    raise SystemExit("Fix G: _accum_to_table body not found")
s = s.replace(old_fin, new_fin, 1)

open(p, "w").write(s)
print("Fix G: mean — внутренние ключи __mean_sum/__mean_count")
PY

# ---- Fix H: sample сохранён при skip skew ----
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()

old = '''    sample_bytes = sum(b.nbytes for b in sample_batches)
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

new = '''    sample_bytes = sum(b.nbytes for b in sample_batches)
    sample_limit = int(memory_guard.available_bytes() * 0.5)
    skip_skew = False
    if sample_bytes > sample_limit:
        if num_partitions is None:
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B) и num_partitions не задан. "
                f"Передайте num_partitions явно."
            )
        # P задан явно: sample не влезает, skew detection пропускаем,
        # но прочитанное не выбрасываем (batch_iter уже исчерпан sample-циклом).
        skip_skew = True
        if log:
            logger.warning(
                "groupby_aggregate: sample (%d B) > available*0.5 (%d B); "
                "skew detection skipped (num_partitions=%d задан явно)",
                sample_bytes, sample_limit, num_partitions,
            )
    batch_iter = itertools.chain(sample_batches, batch_iter)'''

if old not in s:
    raise SystemExit("Fix H: sample block not found")
s = s.replace(old, new, 1)

old_detect = '''    # --- Skew detection & isolation (§11.4 п.3) ---
    heavy_keys = _detect_heavy_keys(sample_batches, by_columns)'''
new_detect = '''    # --- Skew detection & isolation (§11.4 п.3) ---
    heavy_keys = set() if skip_skew else _detect_heavy_keys(sample_batches, by_columns)'''

if old_detect not in s:
    raise SystemExit("Fix H: detect call not found")
s = s.replace(old_detect, new_detect, 1)

open(p, "w").write(s)
print("Fix H: skip_skew сохраняет sample_batches в chain")
PY

set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests still failing (rc=${RC})"
    exit 1
fi

git add -A
git commit -m "02b2-fix3: mean+явный sum (внутренние ключи), skip_skew не теряет sample"

echo "==> sha256"
sha256sum cadw/ops/groupby.py
echo "==> 02b2-fix3 applied on ${BR}"