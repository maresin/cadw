#!/usr/bin/env bash
# patch_02b2: must-fix §11.4 п.3 — skew-детектор + изоляция тяжёлых ключей.
set -euo pipefail

# ---------- inline require_clean_tree ----------
require_clean_tree() {
  local bad_tracked bad_untracked
  bad_tracked="$(git status --porcelain --untracked-files=no)"
  if [[ -n "$bad_tracked" ]]; then
    echo "ERROR: tracked files have uncommitted changes:" >&2
    echo "$bad_tracked" >&2
    return 1
  fi
  bad_untracked=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      scripts/*|.backups/*|__pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*) : ;;
      *) bad_untracked+="  $f"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  if [[ -n "$bad_untracked" ]]; then
    echo "ERROR: untracked files outside scripts/, .backups/, caches:" >&2
    echo "$bad_untracked" >&2
    return 1
  fi
  return 0
}
require_clean_tree || exit 1

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-02b2-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_02b2"

BR="patch/02b2-skew-isolation"
git rev-parse --verify "${BR}" >/dev/null 2>&1 && { echo "branch exists"; exit 1; }
git checkout -b "${BR}"

# ---------- 1. groupby.py ----------
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()

# 1a. imports
if "import numpy as np" not in s:
    s = s.replace("import itertools\n", "import itertools\n\nimport numpy as np\n", 1)
if "from cadw.io.batch_writer import ArrowIPCBatchWriter" not in s:
    s = s.replace(
        "from cadw.io.batch_reader import ArrowIPCBatchReader",
        "from cadw.io.batch_reader import ArrowIPCBatchReader\nfrom cadw.io.batch_writer import ArrowIPCBatchWriter",
        1,
    )
if "_boundaries" not in s.split("def groupby_aggregate")[0]:
    s = s.replace(
        "from cadw.ops.shuffle import hash_shuffle, estimate_num_partitions",
        "from cadw.ops.shuffle import hash_shuffle, estimate_num_partitions, compute_hashes, _mod_p, _boundaries",
        1,
    )

# 1b. helpers
helpers = '''
_HEAVY_RATIO = 0.05


def _detect_heavy_keys(sample_batches, by_columns, threshold_ratio: float = _HEAVY_RATIO) -> set:
    """Тяжёлые ключи по sample: доля в sample > threshold_ratio."""
    if not sample_batches:
        return set()
    from collections import Counter
    counter: Counter = Counter()
    for b in sample_batches:
        for i in range(b.num_rows):
            key = tuple(b.column(c)[i].as_py() for c in by_columns)
            counter[key] += 1
    total = sum(counter.values())
    if total == 0:
        return set()
    return {k for k, cnt in counter.items() if cnt / total > threshold_ratio}


def _compute_heavy_mask(batch: pa.RecordBatch, by_columns, heavy_keys: set) -> pa.Array:
    n = batch.num_rows
    if not heavy_keys or n == 0:
        return pa.array([False] * n, type=pa.bool_())
    mask = pa.array([False] * n, type=pa.bool_())
    for key in heavy_keys:
        k_mask = pa.array([True] * n, type=pa.bool_())
        for col, v in zip(by_columns, key):
            k_mask = pc.and_(k_mask, pc.equal(batch.column(col), pa.scalar(v)))
        mask = pc.or_(mask, k_mask)
    return mask


def _accumulate_heavy(accum: dict, batch: pa.RecordBatch, by_columns, agg_columns, agg_funcs) -> None:
    specs = []
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            specs.extend([(col, "sum"), (col, "count")])
        else:
            specs.append((col, f))
    grouped = batch.select(list(by_columns) + list(agg_columns)).group_by(
        list(by_columns)
    ).aggregate(specs)
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
                    accum[key][out] = max(old, new)


def _accum_to_table(accum: dict, by_columns, agg_columns, agg_funcs) -> pa.Table:
    rows = []
    for key, d in accum.items():
        row = dict(zip(by_columns, key))
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                s, c = d[f"{col}_sum"], d[f"{col}_count"]
                row[f"{col}_mean"] = s / c if c > 0 else float("nan")
            else:
                row[f"{col}_{f}"] = d[f"{col}_{f}"]
        rows.append(row)
    if not rows:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    return pa.Table.from_pylist(
        rows, schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
    )


def _shuffle_write(writers, batch: pa.RecordBatch, key_columns, value_columns,
                   num_partitions: int, hash_seed: int, row_id_offset: int) -> None:
    hashes = compute_hashes(batch, key_columns, hash_seed)
    parts = _mod_p(hashes, num_partitions)
    order = pc.sort_indices(parts)
    sorted_batch = batch.select(list(key_columns) + list(value_columns)).take(order)
    sorted_parts = parts.take(order)
    parts_np = sorted_parts.to_numpy(zero_copy_only=False)
    order_np = order.to_numpy(zero_copy_only=False)
    if parts_np.size == 0:
        return
    bounds = _boundaries(parts_np)
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        p = int(parts_np[start])
        piece = sorted_batch.slice(start, end - start)
        row_ids = pa.array(
            (order_np[start:end].astype(np.int64) + row_id_offset).astype(np.uint64),
            type=pa.uint64(),
        )
        writers[p].write_batch(piece.append_column("row_id", row_ids))


def _groupby_with_skew(batch_iter, by_columns, agg_columns, agg_funcs, heavy_keys,
                       num_partitions, memory_guard, spill_dir, log) -> pa.Table:
    all_columns: list[str] = []
    seen: set[str] = set()
    for c in list(by_columns) + list(agg_columns):
        if c not in seen:
            seen.add(c)
            all_columns.append(c)

    it = iter(batch_iter)
    try:
        first_batch = next(it)
    except StopIteration:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    memory_guard.check_batch(first_batch, name="groupby_with_skew.first", extra_factor=3.0)

    fields = [first_batch.schema.field(c) for c in all_columns]
    fields.append(pa.field("row_id", pa.uint64()))
    schema = pa.schema(fields)
    shuf_dir = spill_dir / "shuffle"
    shuf_dir.mkdir(parents=True, exist_ok=True)
    paths = [shuf_dir / f"partition_{p:05d}.arrow" for p in range(num_partitions)]
    writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]

    accum: dict = {}
    row_id_offset = 0
    try:
        for batch in itertools.chain([first_batch], it):
            memory_guard.check_batch(
                batch, name="groupby_with_skew.batch", extra_factor=3.0
            )
            heavy_mask = _compute_heavy_mask(batch, by_columns, heavy_keys)
            light_mask = pc.invert(heavy_mask)
            heavy_batch = batch.filter(heavy_mask)
            light_batch = batch.filter(light_mask)
            if heavy_batch.num_rows > 0:
                _accumulate_heavy(accum, heavy_batch, by_columns, agg_columns, agg_funcs)
            if light_batch.num_rows > 0:
                _shuffle_write(
                    writers, light_batch, by_columns, agg_columns,
                    num_partitions, hash_seed=0, row_id_offset=row_id_offset,
                )
            row_id_offset += batch.num_rows
    finally:
        for w in writers:
            w.close()

    light_results: list[pa.Table] = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        t = _local_groupby(
            p, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard, spill_dir=spill_dir,
        )
        if t.num_rows > 0:
            light_results.append(t)
        try:
            p.unlink()
        except OSError:
            pass

    heavy_result = _accum_to_table(accum, by_columns, agg_columns, agg_funcs)
    all_results = light_results + ([heavy_result] if heavy_result.num_rows > 0 else [])
    if not all_results:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    if log:
        logger.info(
            "groupby_with_skew: heavy=%d, light_partitions=%d",
            len(accum), len(light_results),
        )
    return pa.concat_tables(all_results)


'''
if "_detect_heavy_keys" not in s:
    s = s.replace("def groupby_aggregate(", helpers + "def groupby_aggregate(", 1)

# 1c. ветвление после auto-P
old_call = '''    partitions = hash_shuffle(
        batch_iter,
        key_columns=by_columns,
        value_columns=value_columns,
        num_partitions=num_partitions,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "shuffle",
        hash_seed=hash_seed,
        log=log,
    )'''
new_call = '''    # --- Skew detection & isolation (§11.4 п.3) ---
    heavy_keys = _detect_heavy_keys(sample_batches, by_columns)
    if heavy_keys:
        if log:
            logger.info(
                "groupby_aggregate: skew detected, %d heavy key(s) -> streaming",
                len(heavy_keys),
            )
        return _groupby_with_skew(
            batch_iter, by_columns, agg_columns, agg_funcs, heavy_keys,
            num_partitions, memory_guard, spill_dir, log,
        )

    partitions = hash_shuffle(
        batch_iter,
        key_columns=by_columns,
        value_columns=value_columns,
        num_partitions=num_partitions,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "shuffle",
        hash_seed=hash_seed,
        log=log,
    )'''
if old_call not in s:
    raise SystemExit("hash_shuffle call in groupby_aggregate not found")
s = s.replace(old_call, new_call, 1)

open(p, "w").write(s)
print("patched groupby.py (skew isolation)")
PY

# ---------- 2. tests ----------
cat > tests/test_mustfix_skew.py <<'EOF'
"""Тесты must-fix §11.4 п.3 (патч 02b2): skew isolation."""
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import groupby_aggregate, _detect_heavy_keys


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def sample_batches_from(table: pa.Table, n: int = 10_000) -> list:
    out = []
    for i in range(0, min(table.num_rows, n), 1000):
        out.append(table.slice(i, min(1000, table.num_rows - i)).to_batches()[0])
    return out


def test_detect_heavy_none_uniform():
    n = 1000
    table = pa.table({"cat": [f"g{i:04d}" for i in range(n)], "val": list(range(n))})
    sample = sample_batches_from(table)
    heavy = _detect_heavy_keys(sample, ["cat"])
    assert heavy == set(), f"uniform data should have no heavy keys; got {heavy}"


def test_detect_heavy_90_10():
    n = 1000
    table = pa.table({"cat": ["A"] * 900 + ["B"] * 100, "val": list(range(n))})
    sample = sample_batches_from(table)
    heavy = _detect_heavy_keys(sample, ["cat"])
    assert ("A",) in heavy
    assert ("B",) in heavy


def test_detect_heavy_one_outlier():
    n = 1000
    keys = [f"g{i:03d}" for i in range(100)] + ["BIG"] * 900
    table = pa.table({"cat": keys, "val": list(range(n))})
    sample = sample_batches_from(table)
    heavy = _detect_heavy_keys(sample, ["cat"])
    assert ("BIG",) in heavy


def test_groupby_90_10_skew(tmp_path):
    n = 1000
    table = pa.table({
        "cat": ["A"] * 900 + ["B"] * 100,
        "val": [1.0] * 900 + [2.0] * 100,
    })
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["sum", "count"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    assert rows[0]["cat"] == "A" and rows[0]["val_sum"] == 900.0 and rows[0]["val_count"] == 900
    assert rows[1]["cat"] == "B" and rows[1]["val_sum"] == 200.0 and rows[1]["val_count"] == 100


def test_groupby_one_hot_skew(tmp_path):
    n = 1000
    table = pa.table({"cat": ["HOT"] * n, "val": list(range(n))})
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    assert result.num_rows == 1
    assert result.to_pylist()[0]["val_mean"] == pytest.approx(499.5)


def test_groupby_mixed_skew_and_light(tmp_path):
    n_light = 1000
    n_heavy = 4000
    keys = [f"L{i:03d}" for i in range(200) for _ in range(5)]
    keys += ["HEAVY"] * n_heavy
    vals = [1.0] * (n_light + n_heavy)
    table = pa.table({"cat": keys, "val": vals})
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        make_iter(table, 200), ["cat"], ["val"], ["sum"],
        guard, tmp_path, num_partitions=8, log=False,
    )
    rows = {r["cat"]: r["val_sum"] for r in result.to_pylist()}
    assert len(rows) == 201
    assert rows["HEAVY"] == float(n_heavy)
    assert rows["L000"] == 5.0


def test_groupby_no_heavy_uses_shuffle(monkeypatch, tmp_path):
    called = []
    orig = gb_mod._groupby_with_skew
    def spy(*a, **kw):
        called.append(True)
        return orig(*a, **kw)
    monkeypatch.setattr(gb_mod, "_groupby_with_skew", spy)
    n = 200
    table = pa.table({"cat": [f"g{i:03d}" for i in range(n)], "val": list(range(n))})
    groupby_aggregate(
        make_iter(table, 50), ["cat"], ["val"], ["count"],
        MemoryGuard(max_memory_mb=10), tmp_path, num_partitions=4, log=False,
    )
    assert not called


def test_groupby_heavy_mean_correct(tmp_path):
    n = 1000
    table = pa.table({"cat": ["A"] * 950 + ["B"] * 50, "val": [10.0] * 950 + [20.0] * 50})
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    assert rows[0]["val_mean"] == 10.0
    assert rows[1]["val_mean"] == 20.0


def test_groupby_multiple_heavy_keys(tmp_path):
    keys = ["A"] * 600 + ["B"] * 300 + [f"L{i:02d}" for i in range(50) for _ in range(2)]
    table = pa.table({"cat": keys, "val": list(range(len(keys)))})
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["count"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = {r["cat"]: r["val_count"] for r in result.to_pylist()}
    assert rows["A"] == 600
    assert rows["B"] == 300
    for i in range(50):
        assert rows[f"L{i:02d}"] == 2
EOF

# ---------- 3. DECISIONS §11.4 status ----------
python3 - <<'PY'
p = "DECISIONS.md"
s = open(p).read()
marker = "**Обновление (патч 02b1):**"
if marker not in s:
    raise SystemExit("marker 02b1 not found")
addition = (
    "**Обновление (патч 02b2):** пункт **3** (skew) — **закрыт**: "
    "`_detect_heavy_keys` по sample (>5% sample → тяжёлый), `_groupby_with_skew` "
    "single-pass: тяжёлые ключи агрегируются потоково (`_accumulate_heavy`, "
    "память O(heavy_keys × aggs)), лёгкие — `hash_shuffle + _local_groupby`, "
    "конкатенация. Тест `test_groupby_90_10_skew` и `test_groupby_one_hot_skew` "
    "проходят без OOM при бюджете 200 KB. **§11.4 закрыт полностью** "
    "(пункты 1, 2, 3, 4, 5).\n\n"
)
if "патч 02b2" not in s:
    s = s.replace(marker, addition + marker, 1)
open(p, "w").write(s)
print("patched DECISIONS §11.4 — all must-fix closed")
PY

# ---------- 4. Sync + tests ----------
uv lock
uv sync
uv run pytest -q

# ---------- 5. Commit ----------
git add -A
git commit -m "02b2: must-fix §11.4 п.3 — skew detection + isolation (streaming for heavy keys, shuffle for light)"

echo "==> sha256"
sha256sum cadw/ops/groupby.py tests/test_mustfix_skew.py
echo
echo "==> Patch 02b2 applied on ${BR}"
echo "    Base:   ${CUR}"
echo "    Backup: ${BK}  (.backups/${BK}/repo.tgz)"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"