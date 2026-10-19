#!/usr/bin/env bash
# patch_11b1_inner_join: inner_join (HashShuffle обеих сторон + pd.merge
# per partition) + CadwDataFrame.merge(how='inner').
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11b1-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11b1"

BR="patch/11b1-inner-join"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/join.py ----------
cat > cadw/ops/join.py <<'PY'
"""
Inner join (v0.2.1).

Grace hash join: HashShuffle обеих сторон с одним seed → партиции с
одинаковыми ключами → для каждой пары (left_p, right_p):
  - build = меньшая партиция (по file size), materialize как pd.DataFrame;
    check_state(build.nbytes);
  - probe = другая, батчами; check_batch(probe_batch);
  - проверка max_output_rows * probe_rows до pd.merge (защита от cartesian);
  - null-ключи отбрасываются (SQL-семантика; pandas.merge по умолчанию
    матчит NaN==NaN, что не то);
  - pd.merge(how="inner", suffixes=...).

Ограничения v0.2.1:
- how='inner' только.
- on — общие колонки в обеих сторонах (по имени).
- left_on/right_on, on=index — v0.2.2.
- row_id не сохраняется (join создаёт новую идентичность).
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.shuffle import hash_shuffle


def _drop_null_keys(df: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """SQL-семантика: строки с NULL в любом on-ключе не участвуют."""
    if len(df) == 0:
        return df
    mask = np.ones(len(df), dtype=bool)
    for c in on:
        mask &= df[c].notna().to_numpy()
    return df.loc[mask]


def _empty_result_schema(
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
) -> pa.Schema:
    on_set = set(on)
    left_rest = [f for f in left_schema if f.name not in on_set]
    right_rest = [f for f in right_schema if f.name not in on_set]
    overlap = {f.name for f in left_rest} & {f.name for f in right_rest}
    fields: list[pa.Field] = []
    for c in on:
        fields.append(left_schema.field(c))
    for f in left_rest:
        name = f.name + suffixes[0] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type))
    for f in right_rest:
        name = f.name + suffixes[1] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type))
    return pa.schema(fields)


def _materialize_partition(path: Path, memory_guard: MemoryGuard) -> pa.Table:
    batches = []
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        batches.append(b.select(keep))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(table.nbytes, name=f"join_build({path.name})")
    return table


def _iter_probe_batches(path: Path) -> Iterable[pa.RecordBatch]:
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        yield b.select(keep)


def inner_join(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash inner join. Возвращает pa.Table."""
    if not on:
        raise ValueError("inner_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"inner_join: on column {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"inner_join: on column {c!r} not in right schema")

    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    # Peek: пустой вход -> пустой результат
    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    try:
        first_right = next(right_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    left_batches = chain([first_left], left_it)
    right_batches = chain([first_right], right_it)

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions, memory_guard,
        spill_dir / "join_left", hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions, memory_guard,
        spill_dir / "join_right", hash_seed=hash_seed, log=False,
    )

    if max_output_rows is None:
        # Консервативная оценка: 0.3 * available / 256 B на строку
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )

    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    all_batches: list[pa.RecordBatch] = []

    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None or rp is None:
            continue

        # build = меньшая партиция по file size (proxy для памяти)
        if lp.stat().st_size <= rp.stat().st_size:
            build_path, probe_path = lp, rp
        else:
            build_path, probe_path = rp, lp

        build_table = _materialize_partition(build_path, memory_guard)
        build_df = _drop_null_keys(build_table.to_pandas(), list(on))
        if len(build_df) == 0:
            continue

        for probe_batch in _iter_probe_batches(probe_path):
            memory_guard.check_batch(
                probe_batch, name=f"join_probe({probe_path.name})"
            )
            probe_df = _drop_null_keys(probe_batch.to_pandas(), list(on))
            if len(probe_df) == 0:
                continue
            # Верхняя оценка output до materialize
            if len(build_df) * len(probe_df) > max_output_rows:
                raise MemoryLimitExceeded(
                    f"inner_join: estimated output rows "
                    f"{len(build_df)} * {len(probe_df)} = "
                    f"{len(build_df) * len(probe_df)} > max_output_rows="
                    f"{max_output_rows}. Likely skew (DECISIONS §11.4)."
                )
            merged = pd.merge(
                build_df, probe_df, on=list(on), how="inner", suffixes=suffixes
            )
            if len(merged) == 0:
                continue
            out_table = pa.Table.from_pandas(merged, preserve_index=False)
            # Привести порядок колонок к empty_schema
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            memory_guard.check_state(
                out_table.nbytes, name=f"join_output(p={p})"
            )
            for b in out_table.to_batches():
                all_batches.append(b)

    if not all_batches:
        return pa.Table.from_batches([], schema=empty_schema)
    return pa.Table.from_batches(all_batches)
PY

# ---------- cadw/ops/__init__.py: экспорт ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/__init__.py")
s = p.read_text()
if "from .join import inner_join" not in s:
    s = s.replace(
        "from .sort import external_sort, read_sorted, compute_fan_in",
        "from .sort import external_sort, read_sorted, compute_fan_in\n"
        "from .join import inner_join",
        1,
    )
    s = s.replace(
        '"external_sort", "read_sorted", "compute_fan_in",',
        '"external_sort", "read_sorted", "compute_fan_in", "inner_join",',
        1,
    )
    p.write_text(s)
    print("cadw/ops/__init__.py: inner_join exported")
else:
    print("cadw/ops/__init__.py: inner_join already exported")
PY

# ---------- dataframe.py: merge ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# import
if "from cadw.ops.join import inner_join" not in s:
    s = s.replace(
        "from cadw.ops.groupby import groupby_aggregate",
        "from cadw.ops.groupby import groupby_aggregate\n"
        "from cadw.ops.join import inner_join",
        1,
    )
    print("dataframe.py: import inner_join")

# Вставить merge() перед "# ---------------- write ----------------"
anchor = "    # ---------------- write ----------------"
if anchor not in s:
    raise SystemExit("write anchor not found")
if "    def merge(" not in s:
    method = '''    # ---------------- merge ----------------

    def merge(
        self,
        right: "CadwDataFrame",
        on: str | Sequence[str],
        how: str = "inner",
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
        num_partitions: int = 8,
    ) -> "CadwDataFrame":
        """Inner join (v0.2.1). how ∈ {'inner'} только.

        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle (обе стороны).
        """
        if how != "inner":
            raise NotImplementedError(
                f"merge: how={how!r}; only 'inner' supported in v0.2.1. "
                f"'left'/'right'/'outer' — v0.2.2."
            )
        if not isinstance(right, CadwDataFrame):
            raise TypeError("merge: right must be CadwDataFrame")
        on_list = [on] if isinstance(on, str) else list(on)
        for c in on_list:
            if c not in self.columns:
                raise KeyError(f"merge: on column {c!r} not in self")
            if c not in right.columns:
                raise KeyError(f"merge: on column {c!r} not in right")

        left_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self.columns]
        )
        right_schema = pa.schema(
            [right._dataset.schema.field(c) for c in right.columns]
        )

        result_table = inner_join(
            self._iter_filtered_batches(),
            right._iter_filtered_batches(),
            left_schema,
            right_schema,
            on=on_list,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "join",
            suffixes=suffixes,
            num_partitions=num_partitions,
        )
        return self._wrap_result_table(result_table)

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    print("dataframe.py: merge() added")
else:
    print("dataframe.py: merge() already exists")
PY

# ---------- tests/test_join.py ----------
cat > tests/test_join.py <<'EOF'
"""Тесты inner_join / CadwDataFrame.merge (11-B-1)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _sorted_by(table, key):
    return sorted(table.to_pylist(), key=lambda r: r[key])


def test_merge_basic(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["x", "y", "z", "w"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k")
    head = result.head(20)
    assert head.num_rows == 2
    rows = _sorted_by(head, "k")
    assert rows == [
        {"k": 1, "a": "x", "b": "p"},
        {"k": 3, "a": "z", "b": "q"},
    ]


def test_merge_multi_key(tmp_path):
    left = make_df(tmp_path, "L", {
        "k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30],
    })
    right = make_df(tmp_path, "R", {
        "k1": [1, 2], "k2": ["a", "a"], "w": [100, 200],
    })
    result = left.merge(right, on=["k1", "k2"])
    head = result.head(20)
    rows = sorted(head.to_pylist(), key=lambda r: r["v"])
    assert rows == [
        {"k1": 1, "k2": "a", "v": 10, "w": 100},
        {"k1": 2, "k2": "a", "v": 30, "w": 200},
    ]


def test_merge_suffixes(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2], "v": [10, 20]})
    right = make_df(tmp_path, "R", {"k": [1, 2], "v": [100, 200]})
    result = left.merge(right, on="k", suffixes=("_l", "_r"))
    head = result.head(10)
    names = set(head.schema.names)
    assert "v_l" in names and "v_r" in names
    rows = _sorted_by(head, "k")
    assert rows[0]["v_l"] == 10 and rows[0]["v_r"] == 100


def test_merge_empty_right(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k")
    assert result.shape[0] == 0


def test_merge_null_not_matched(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, None, 2], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [None, 2], "b": [100, 200]})
    result = left.merge(right, on="k")
    head = result.head(10)
    assert head.num_rows == 1
    rows = head.to_pylist()
    assert rows[0]["k"] == 2
    assert rows[0]["a"] == 30
    assert rows[0]["b"] == 200


def test_merge_cartesian_on_duplicates(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 1, 2], "a": ["x", "y", "z"]})
    right = make_df(tmp_path, "R", {"k": [1, 1], "b": ["p", "q"]})
    result = left.merge(right, on="k")
    # k=1: 2 left × 2 right = 4; k=2: 0
    assert result.shape[0] == 4


def test_merge_after_filter(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 2, 3], "b": [100, 200, 300]})
    filtered = left[left["a"] > 15]  # k=2, 3
    result = filtered.merge(right, on="k")
    assert result.shape[0] == 2


def test_merge_missing_on_raises(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"other": [1], "b": [100]})
    with pytest.raises(KeyError, match="not in right"):
        left.merge(right, on="k")


def test_merge_unsupported_how(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
    with pytest.raises(NotImplementedError, match="only 'inner'"):
        left.merge(right, on="k", how="left")


def test_merge_requires_cadwdataframe(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    with pytest.raises(TypeError, match="CadwDataFrame"):
        left.merge("not a df", on="k")
EOF

# ---------- DECISIONS ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `inner_join` (v0.2.1)" not in s:
    s = s.replace(
        "| dict check в `_local_groupby` (11-A-4a, must-fix §11.4) | ✅ | 5 |",
        "| dict check в `_local_groupby` (11-A-4a, must-fix §11.4) | ✅ | 5 |\n"
        "| `inner_join` (v0.2.1: HashShuffle обеих сторон + pd.merge) | ✅ | 9 |",
        1,
    )

# §12.6 v0.2 progress
if "§12.6. v0.2 (следующее)" in s:
    s = s.replace(
        "§12.6. v0.2 (следующее)",
        "§12.6. v0.2 (в работе: inner_join ✅)",
        1,
    )

# §13.4 эпизод 36
marker_135 = "### 13.5. Открытые вопросы"
episode_36 = """**Эпизод 36 (первый join, SQL null semantics).** При первом
`inner_join` через `pd.merge` обнаружилось: **pandas.merge по умолчанию
матчит NaN==NaN**, что отличается от SQL (NULL не совпадает). Для
CADW выбрана SQL-семантика: `_drop_null_keys` фильтрует строки с NULL в
on-колонках до `pd.merge`. Без фильтра `null` в key дал бы cartesian
NaN×NaN — скрытый OOM. **Урок:** `pd.merge` — не «SQL join»; на key
колонках NULL — отдельная семантика, её нужно реализовывать явно.
Также: `inner_join` проверяет `len(build) * len(probe) > max_output_rows`
**до** `pd.merge` — защита от cartesian-skew, который pyarrow/pandas
молча материализуют. max_output_rows = 0.3 * available / 256 B
(консервативно); в v0.2.2 — из `JoinMemoryModel.solve`.

"""
if "Эпизод 36" not in s:
    s = s.replace(marker_135, episode_36 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §12.6, §13.4/36")
PY

# ---------- CHANGELOG ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.2.0] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(right, on, how='inner', suffixes=('_x','_y'),
  num_partitions=8)`: Grace hash inner join.
- `cadw.ops.join.inner_join`: HashShuffle обеих сторон (один seed) →
  партиции с одинаковыми ключами → для каждой пары: build = меньшая
  партиция (`check_state`), probe батчами (`check_batch`),
  `pd.merge(how='inner', suffixes=...)`.
- SQL-семантика NULL: `_drop_null_keys` до `pd.merge` (pandas.merge
  по умолчанию матчит NaN==NaN — не то).
- Защита от cartesian-skew: `len(build) * len(probe) > max_output_rows`
  → `MemoryLimitExceeded` **до** `pd.merge`.

### Ограничения v0.2.1
- `how='inner'` только; `left`/`right`/`outer` — v0.2.2.
- `on` по имени в обеих сторонах; `left_on`/`right_on`, `on=index` —
  v0.2.2.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2.

"""
if "## [0.2.0]" not in s:
    if "## [0.1.7]" in s:
        s = s.replace("## [0.1.7]", entry + "## [0.1.7]", 1)
    else:
        s = s.replace("## [0.1.0]", entry + "## [0.1.0]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.0")
PY

# ---------- run ----------
echo "=== pytest tests/test_join.py ==="
set +e
uv run pytest tests/test_join.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "test_join.py: rc=$RC"
  git add -A
  git commit -m "WIP: 11-B-1 (test red; fix follows)" || true
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-1: inner_join (Grace hash: HashShuffle обеих сторон + pd.merge per partition) + CadwDataFrame.merge(how='inner') + 10 tests"

echo "==> 11-B-1 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"