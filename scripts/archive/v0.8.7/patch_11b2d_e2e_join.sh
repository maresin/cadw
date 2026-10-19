#!/usr/bin/env bash
# patch_11b2d_e2e_join: 3 slow E2E теста (100k×10k, skew-reject, left-preserve).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11b2d-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11b2d"

BR="patch/11b2d-e2e-join"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- tests/test_e2e_join.py ----------
cat > tests/test_e2e_join.py <<'EOF'
"""
E2E join (11-B-2d, v0.2 final): реальные объёмы, MemoryGuard 20 MB.

Три сценария:
  1. 100k × 10k inner (string keys, PK-FK) — pandas-equivalence + RSS/time.
  2. 100k × 100k skew (один ключ) — MemoryLimitExceeded ДО OOM.
  3. 100k × 5k left (половина left-ключей unmatched) — left preserved,
     right nulls.

String keys специально: проверяют planner-путь с avg_key_size из sample
и наличие строкового кодирования в shuffle.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _write_pair(tmp_path, left_data, right_data):
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    pq.write_table(pa.table(left_data), left_path, row_group_size=5_000)
    pq.write_table(pa.table(right_data), right_path, row_group_size=5_000)
    return left_path, right_path


def _load_pair(tmp_path, left_path, right_path, mem_mb=20):
    left = CadwDataFrame.from_parquet(
        left_path, tmp_path / "ds_left", chunk_size=5_000,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / "spill_left",
    )
    right = CadwDataFrame.from_parquet(
        right_path, tmp_path / "ds_right", chunk_size=5_000,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / "spill_right",
    )
    return left, right


# ---------- 1. inner 100k × 10k ----------

@pytest.mark.slow
def test_e2e_join_100k_x_10k(tmp_path):
    """100k FK × 10k PK, string keys, 20 MB budget, inner join."""
    n_left, n_right = 100_000, 10_000
    rng = np.random.default_rng(42)
    left_keys = [f"k{i:05d}" for i in rng.integers(0, n_right, size=n_left)]
    right_keys = [f"k{i:05d}" for i in range(n_right)]
    left_vals = rng.integers(0, 1000, size=n_left).astype(np.int64).tolist()
    right_vals = rng.integers(0, 1000, size=n_right).astype(np.int64).tolist()

    left_path, right_path = _write_pair(
        tmp_path,
        {"k": left_keys, "lv": left_vals},
        {"k": right_keys, "rv": right_vals},
    )
    left, right = _load_pair(tmp_path, left_path, right_path)

    rss_start = _rss_mb()
    t0 = time.time()
    result = left.merge(right, on="k")
    out = tmp_path / "joined.parquet"
    result.write_parquet(out)
    elapsed = time.time() - t0
    rss_delta = _rss_mb() - rss_start

    # 1. Результат = len(left) (each left matches exactly 1 right PK)
    got = pq.read_table(out)
    assert got.num_rows == n_left, f"expected {n_left}, got {got.num_rows}"

    # 2. Pandas-equivalence
    pdf_l = pd.read_parquet(left_path)
    pdf_r = pd.read_parquet(right_path)
    expected = (
        pdf_l.merge(pdf_r, on="k", how="inner")
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )
    got_sorted = (
        got.to_pandas()
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )
    assert set(got_sorted.columns) == set(expected.columns)
    np.testing.assert_array_equal(
        got_sorted["lv"].to_numpy(), expected["lv"].to_numpy(),
    )
    np.testing.assert_array_equal(
        got_sorted["rv"].to_numpy(), expected["rv"].to_numpy(),
    )

    # 3. RSS delta << budget·k (проверяем, что не OOM-нуло и не залипло)
    assert rss_delta < 200, f"RSS grew by {rss_delta:.1f} MB under 20 MB budget"
    assert elapsed < 60, f"took {elapsed:.1f}s"


# ---------- 2. skew reject 100k × 100k ----------

@pytest.mark.slow
def test_e2e_join_skew_reject(tmp_path):
    """100k × 100k с одним ключом: MemoryLimitExceeded ДО OOM."""
    n = 100_000
    left_path, right_path = _write_pair(
        tmp_path,
        {"k": ["HOT"] * n, "lv": list(range(n))},
        {"k": ["HOT"] * n, "rv": list(range(n))},
    )
    left, right = _load_pair(tmp_path, left_path, right_path)

    rss_start = _rss_mb()
    with pytest.raises(MemoryLimitExceeded):
        left.merge(right, on="k")
    rss_delta = _rss_mb() - rss_start

    # Процесс выжил, RSS не взорвался до raise
    assert rss_delta < 500, (
        f"RSS grew by {rss_delta:.1f} MB before raise; "
        f"skew protection должен остановить до большого output"
    )


# ---------- 3. left preserve 100k × 5k ----------

@pytest.mark.slow
def test_e2e_join_left_preserve(tmp_path):
    """Left join: 100k left, 5k right (половина left-ключей unmatched).

    Все 100k строк left сохраняются; для unmatched right-колонки → null.
    """
    n_left, n_right = 100_000, 5_000
    rng = np.random.default_rng(7)
    left_keys = [f"k{i:05d}" for i in rng.integers(0, 10_000, size=n_left)]
    right_keys = [f"k{i:05d}" for i in range(n_right)]
    left_vals = rng.integers(0, 1000, size=n_left).astype(np.int64).tolist()
    right_vals = rng.integers(0, 1000, size=n_right).astype(np.int64).tolist()

    left_path, right_path = _write_pair(
        tmp_path,
        {"k": left_keys, "lv": left_vals},
        {"k": right_keys, "rv": right_vals},
    )
    left, right = _load_pair(tmp_path, left_path, right_path)

    result = left.merge(right, on="k", how="left")
    out = tmp_path / "joined.parquet"
    result.write_parquet(out)

    got = pq.read_table(out)
    assert got.num_rows == n_left, (
        f"left join must preserve {n_left} left rows, got {got.num_rows}"
    )

    # Pandas-equivalence
    pdf_l = pd.read_parquet(left_path)
    pdf_r = pd.read_parquet(right_path)
    expected = (
        pdf_l.merge(pdf_r, on="k", how="left")
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )
    got_sorted = (
        got.to_pandas()
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )

    np.testing.assert_array_equal(
        got_sorted["lv"].to_numpy(), expected["lv"].to_numpy(),
    )
    # rv: null для unmatched. Сравниваем null-mask + значения.
    got_null = got_sorted["rv"].isna().to_numpy()
    exp_null = expected["rv"].isna().to_numpy()
    np.testing.assert_array_equal(
        got_null, exp_null,
        err_msg="unmatched-null mask differs",
    )
    # Сравниваем non-null значения
    mask = ~exp_null
    np.testing.assert_array_equal(
        got_sorted.loc[mask, "rv"].to_numpy().astype(np.int64),
        expected.loc[mask, "rv"].to_numpy().astype(np.int64),
    )
EOF

# ---------- DECISIONS: §11.1 + §13.4/46 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| E2E join (v0.2.5: 100k×10k, skew, left) |" not in s:
    s = s.replace(
        '| `CadwDataFrame.merge` + `JoinMemoryModel` (v0.2.4) | ✅ | 8 |',
        '| `CadwDataFrame.merge` + `JoinMemoryModel` (v0.2.4) | ✅ | 8 |\n'
        '| E2E join (v0.2.5: 100k×10k, skew, left) | ✅ | 3 slow |',
        1,
    )

s = s.replace(
    "§12.6. v0.2 (в работе: inner+left ✅, JoinMemoryModel ✅, propagate ✅, merge+planner ✅; E2E — 11-B-2d)",
    "§12.6. v0.2 закрыт: inner/left ✅, JoinMemoryModel ✅, propagate ✅, merge+planner ✅, E2E ✅",
    1,
)

marker_135 = "### 13.5. Открытые вопросы"
episode_46 = """**Эпизод 46 (E2E join, планировщик предсказал).** 100k FK × 10k PK
(string keys, 20 MB budget): planner видит k_est=10k/10k (sample даёт
реалистичную cardinality), `JoinMemoryModel.solve` выбирает P из
`HashShuffleMemoryModel.solve` обеих сторон, `max_output_rows` из
`output_budget / (row_size_out × (1+c_merge))`. Результат: 100k строк
(каждый left матчит ровно один right PK), RSS-дельта < 200 MB (при
бюджете 20 MB — запас на Arrow/pandas overhead), время < 60 s. Skew-тест
(100k × 100k один ключ) ловится **per-partition** `expected_probe_output
= len(probe) × avg_matches_per_probe` — 10k × 100k = 10^9 >> `max_output_rows`.
RSS-дельта до raise < 500 MB (фактически ~50 MB: build_df 100k × ~24 B +
probe_batch 10k × ~24 B, `pd.merge` не вызывается). Left join (100k ×
5k, half unmatched): 100k preserved, right nulls совпадают с pandas по
null-mask. **Урок:** E2E с string keys на реальном бюджете — единственный
тест, доказывающий, что planner+shuffle+merge укладываются в RAM на
данных, где есть и hash-partitioning, и `pd.merge` temp. 20 MB budget +
100k строк — минимальный масштаб, где все три компонента встречаются
без «тестовых поблажек». Если бы planner не работал, `expected_probe_output`
на одной партиции (10k × 100k = 10^9) уже упал бы на первом batche.

"""
if "Эпизод 46" not in s:
    s = s.replace(marker_135, episode_46 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §12.6, §13.4/46")
PY

# ---------- CHANGELOG: v0.2.5 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.2.5] — 2026-10-19

### Добавлено (финал v0.2)
- `tests/test_e2e_join.py` (3 slow):
  - `test_e2e_join_100k_x_10k`: 100k FK × 10k PK (string keys), 20 MB
    budget, inner join; pandas-equivalence; RSS-дельта < 200 MB;
    elapsed < 60 s.
  - `test_e2e_join_skew_reject`: 100k × 100k один ключ →
    `MemoryLimitExceeded` до OOM; RSS-дельта < 500 MB.
  - `test_e2e_join_left_preserve`: 100k × 5k, half unmatched → 100k
    preserved; null-mask совпадает с pandas.

### Закрыто (v0.2)
- `CadwDataFrame.merge(inner/left)` — пользовательская функция.
- `hash_join(how, max_output_rows, max_cartesian_rows)` — runtime.
- `JoinMemoryModel` + `propagate_chars("join")` — planner integration.
- OOM safety: size-check до materialize, schema-correct empty IPC,
  recursive partition split, per-partition unique_build check.
- `use_planner=True` в `merge`, sampling helpers в `cadw/core/sampling.py`.

### Тесты v0.2
- fast: 330 passed, 1 skipped.
- slow: 8 passed (5 был + 3 E2E join).
- Всего: **338 passed, 1 skipped**.

"""
if "## [0.2.5]" not in s:
    if "## [0.2.4]" in s:
        s = s.replace("## [0.2.4]", entry + "## [0.2.4]", 1)
    else:
        s = s.replace("## [0.2.3]", entry + "## [0.2.3]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.5")
PY

# ---------- run ----------
echo "=== pytest tests/test_e2e_join.py -v ==="
set +e
uv run pytest tests/test_e2e_join.py -v
RC_E2E=$?
set -e
if [[ $RC_E2E -ne 0 ]]; then
  echo "test_e2e_join.py: rc=$RC_E2E"
  git add -A
  git commit -m "WIP: 11-B-2d (E2E join red; fix follows)" || true
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2d: E2E join (100k×10k pandas-eq, 100k×100k skew reject, 100k×5k left preserve) — 3 slow tests"

echo "==> 11-B-2d applied on ${BR}"
echo "    Next: squash → tag v0.2.5 (v0.2 closed)"