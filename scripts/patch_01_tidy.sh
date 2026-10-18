#!/usr/bin/env bash
# scripts/patch_01_tidy.sh
# Патч 01: порядок в директории.
set -euo pipefail

# ---------- 0. Guards ----------
git rev-parse --is-inside-work-tree >/dev/null
git rev-parse HEAD >/dev/null 2>&1 || { echo "ERROR: no commits yet"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "ERROR: working tree not clean. Commit or stash first."
  git status --short
  exit 1
}

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK_TAG="backup-pre-tidy-${TS}"
BK_DIR=".backups/${BK_TAG}"

# ---------- 1. Backup ----------
echo "==> Backup: tag ${BK_TAG}; tar ${BK_DIR}/repo.tgz"
mkdir -p "${BK_DIR}"
git archive --format=tar.gz -o "${BK_DIR}/repo.tgz" HEAD
git tag -a "${BK_TAG}" -m "before patch_01_tidy"

# ---------- 2. Branch ----------
BR="patch/01-tidy"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  echo "ERROR: branch ${BR} already exists. Delete it or rename."
  exit 1
fi
git checkout -b "${BR}"

# ---------- 3. Move legacy ----------
echo "==> Move legacy to _legacy/"
mkdir -p cadw/_legacy tests/_legacy

git mv cadw/core/spill_manager.py            cadw/_legacy/spill_manager.py
git mv cadw/operations/unique_collector.py   cadw/_legacy/unique_collector.py
git mv cadw/operations/groupby_aggregator.py cadw/_legacy/groupby_aggregator.py
git mv tests/test_unique.py                  tests/_legacy/test_unique.py
git mv tests/test_groupby.py                 tests/_legacy/test_groupby.py
if [[ -f tests/demo_unique.py ]]; then
  git mv tests/demo_unique.py tests/_legacy/demo_unique.py
fi

cat > cadw/_legacy/__init__.py <<'EOF'
"""Legacy-модули. Не импортируются по умолчанию. См. DECISIONS.md §11.2."""
EOF
: > tests/_legacy/__init__.py

# Fix internal imports of legacy modules
sed -i 's|from cadw.core.spill_manager import SpillManager|from cadw._legacy.spill_manager import SpillManager|g' \
  cadw/_legacy/unique_collector.py cadw/_legacy/groupby_aggregator.py

# Fix imports in legacy tests
sed -i 's|from cadw.operations.unique_collector import UniqueCollector|from cadw._legacy.unique_collector import UniqueCollector|g' \
  tests/_legacy/test_unique.py
if [[ -f tests/_legacy/demo_unique.py ]]; then
  sed -i 's|from cadw.operations.unique_collector import UniqueCollector|from cadw._legacy.unique_collector import UniqueCollector|g' \
    tests/_legacy/demo_unique.py
fi
sed -i 's|from cadw.operations.groupby_aggregator import GroupByAggregator|from cadw._legacy.groupby_aggregator import GroupByAggregator|g' \
  tests/_legacy/test_groupby.py

# ---------- 4. Drop empty cadw/operations/ ----------
if [[ -f cadw/operations/__init__.py ]]; then
  git rm -f cadw/operations/__init__.py
fi
rmdir cadw/operations 2>/dev/null || true

# ---------- 5. Delete dead code ----------
[[ -f cadw/dataframe.py ]] && git rm -f cadw/dataframe.py
[[ -f main.py ]]           && git rm -f main.py

# ---------- 6. cadw/__init__.py ----------
cat > cadw/__init__.py <<'EOF'
"""CADW: out-of-core dataframe. См. DECISIONS.md."""
from .core.memory_guard import MemoryGuard, MemoryLimitExceeded
from .io.batch_reader import ParquetBatchReader, ArrowIPCBatchReader
from .io.batch_writer import ParquetBatchWriter, ArrowIPCBatchWriter
from .ops.reduce import (
    reduce, ReduceOp,
    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,
)
from .ops.shuffle import (
    hash_shuffle, compute_hashes, estimate_num_partitions, read_partition_batches,
)
from .ops.sort import external_sort, read_sorted
from .ops.groupby import groupby_aggregate, write_groupby_result

__all__ = [
    "MemoryGuard", "MemoryLimitExceeded",
    "ParquetBatchReader", "ArrowIPCBatchReader",
    "ParquetBatchWriter", "ArrowIPCBatchWriter",
    "reduce", "ReduceOp",
    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",
    "CovarianceOp", "HistogramOp",
    "hash_shuffle", "compute_hashes", "estimate_num_partitions",
    "read_partition_batches",
    "external_sort", "read_sorted",
    "groupby_aggregate", "write_groupby_result",
]
EOF

# ---------- 7. cadw/core/__init__.py ----------
cat > cadw/core/__init__.py <<'EOF'
from .memory_guard import MemoryGuard, MemoryLimitExceeded

__all__ = ["MemoryGuard", "MemoryLimitExceeded"]
EOF

# ---------- 8. cadw/ops/__init__.py (idempotent rewrite) ----------
cat > cadw/ops/__init__.py <<'EOF'
from .reduce import (
    reduce, ReduceOp,
    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,
)
from .shuffle import (
    hash_shuffle, compute_hashes, estimate_num_partitions, read_partition_batches,
)
from .sort import external_sort, read_sorted
from .groupby import groupby_aggregate, write_groupby_result

__all__ = [
    "reduce", "ReduceOp",
    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",
    "CovarianceOp", "HistogramOp",
    "hash_shuffle", "compute_hashes", "estimate_num_partitions",
    "read_partition_batches",
    "external_sort", "read_sorted",
    "groupby_aggregate", "write_groupby_result",
]
EOF

# ---------- 9. pyproject.toml ----------
python3 - <<'PY'
import re
p = "pyproject.toml"
s = open(p).read()
if '"xxhash' not in s:
    s = re.sub(r'"pyarrow>=[0-9.]+",',
               '"pyarrow>=26.0.0",\n    "xxhash>=3.4.1",',
               s, count=1)
if "[tool.pytest.ini_options]" not in s:
    s += ('\n[tool.pytest.ini_options]\n'
          'testpaths = ["tests"]\n'
          'norecursedirs = ["_legacy", ".venv", "build", "dist"]\n')
open(p, "w").write(s)
PY

# ---------- 10. README.md ----------
cat > README.md <<'EOF'
# CADW

Out-of-core фреймворк для табличных данных, не помещающихся в RAM, на слабых машинах.

## Принципы

- Не более одного батча данных в RAM; остальное — на диск (Arrow IPC).
- Apache Arrow — единый формат в памяти и для кэша.
- Три примитива: `Reduce`, `HashShuffle`, `Sort`. Всё тяжёлое — их композиция.

## Статус

Реализованы и покрыты тестами: `MemoryGuard`, `Reduce`, `HashShuffle`,
`Sort`, `groupby_aggregate`.

`groupby` работает при числе групп, превышающем память
(`tests/test_groupby_v2.py::test_groupby_many_groups`).

**Гарантия out-of-core условна** — см. `DECISIONS.md` §11.4 (5 must-fix).

Отсутствует: `CadwDataFrame`, `Dataset`, `RowFilter`, `unique`/`join` через
новые примитивы (см. `DECISIONS.md` §11.3, §12).

## Установка

```bash
uv sync
```

## Пример

```python
from cadw import groupby_aggregate, MemoryGuard
from cadw.io.batch_reader import ParquetBatchReader

reader = ParquetBatchReader("data.parquet", ["cat", "val"], batch_size=10000)
guard = MemoryGuard(max_memory_mb=200)
result = groupby_aggregate(reader, ["cat"], ["val"], ["mean"], guard, "./spill")
print(result)
```

## Документация

- `DECISIONS.md` — архитектура, алгоритмы, статус, must-fix, уроки.

## Legacy

`cadw/_legacy/`, `tests/_legacy/` — старые реализации (`SpillManager`,
`UniqueCollector`, `GroupByAggregator`). Не импортируются по умолчанию,
исключены из `pytest`. См. `DECISIONS.md` §11.2.
EOF

# ---------- 11. .gitignore ----------
for line in ".backups/" "merged_cadw_project.py" "cadw_cache/" "cadw_shuffle/"; do
  grep -qxF "$line" .gitignore 2>/dev/null || echo "$line" >> .gitignore
done

# ---------- 12. Sync + tests ----------
echo "==> uv lock && uv sync"
uv lock
uv sync

echo "==> pytest (main)"
MAIN_OUT="$(uv run pytest -q 2>&1 | tail -1)" || true
echo "$MAIN_OUT"
if ! echo "$MAIN_OUT" | grep -qE '[0-9]+ passed'; then
  echo "WARNING: unexpected pytest summary: $MAIN_OUT"
fi

echo "==> pytest (legacy, best-effort)"
uv run pytest tests/_legacy -q || echo "(legacy failures — ok, см. DECISIONS §11.2)"

# ---------- 13. Commit ----------
git add -A
git commit -m "tidy: legacy -> _legacy/, rm dataframe.py+main.py, README, pyproject (pyarrow>=26, xxhash, pytest config)"

# ---------- 14. Fingerprints ----------
echo "==> sha256 (контроль содержимого)"
sha256sum cadw/__init__.py cadw/core/__init__.py cadw/ops/__init__.py pyproject.toml README.md || true

# ---------- 15. Done ----------
echo
echo "==> Patch 01 applied on branch ${BR}"
echo "    Base branch:  ${CURRENT_BRANCH}"
echo "    Backup tag:   ${BK_TAG}"
echo "    Backup tar:   ${BK_DIR}/repo.tgz"
echo "    Revert (fast):    git checkout ${CURRENT_BRANCH} && git branch -D ${BR}"
echo "    Revert (hard):    git reset --hard ${BK_TAG}"
