#!/usr/bin/env bash
# patch_v034_combiner_fix:
#   1. _detect_heavy_keys: +_HEAVY_ABSOLUTE_MIN (skew only if count >= min
#      AND ratio > threshold). Fixes combiner path never being used.
#   2. Rewrite combiner tests: n=1000, k=50 (no skew) → combiner active.
#   3. Remove test_combiner_std (std rejected by _AGG_FUNCS before combiner).
#   4. Fix test_combiner_skew: n=1000 (heavy fires).
#   5. Add test_combiner_path_actually_used (monkeypatch spy).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v034-combiner"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v034-fix-${TS}"
git diff > ".backups/backup-pre-v034-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.3.4 combiner (skew detection disables combiner on tiny data; fix follows)" || true
require_clean_tree || exit 1

# ---------- Fix 1: groupby.py — _HEAVY_ABSOLUTE_MIN ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

# Add constant
old_const = "_HEAVY_RATIO = 0.05"
new_const = (
    "_HEAVY_RATIO = 0.05\n"
    "# Skew isolation is worth the overhead only when the key is BOTH\n"
    "# large in absolute terms and dominant in ratio. Without absolute\n"
    "# min, tiny test data (5 rows, 3 keys, 40/40/20%) incorrectly\n"
    "# triggers skew path — combiner never activates. §13.4/52b.\n"
    "_HEAVY_ABSOLUTE_MIN = 100"
)
if old_const not in s:
    raise SystemExit("_HEAVY_RATIO not found")
s = s.replace(old_const, new_const, 1)

# Update _detect_heavy_keys filter
old_filter = "    return {k for k, cnt in counter.items() if cnt / total > threshold_ratio}"
new_filter = (
    "    return {\n"
    "        k for k, cnt in counter.items()\n"
    "        if cnt >= _HEAVY_ABSOLUTE_MIN and cnt / total > threshold_ratio\n"
    "    }"
)
if old_filter not in s:
    raise SystemExit("_detect_heavy_keys filter not found")
s = s.replace(old_filter, new_filter, 1)

p.write_text(s)
print("groupby.py: _HEAVY_ABSOLUTE_MIN=100 added; _detect_heavy_keys requires both")
PY

# ---------- Fix 2-4: rewrite test_combiner.py ----------
cat > tests/test_combiner.py <<'EOF'
"""
Combiner для groupby (v0.3.4): partial aggregation до shuffle.

Тесты используют n=1000, k=50 (20 rows/group = 2%) — БЕЗ skew-trigger,
чтобы combiner путь реально исполнялся (не fallback). Skew-detection
требует count >= _HEAVY_ABSOLUTE_MIN=100 AND ratio > 0.05 (§13.4/52b).
"""
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.groupby import groupby_aggregate


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def _run(tmp_path, table, by, agg_cols, funcs, *, combiner,
         num_partitions=4, mem_mb=20, batch_size=10):
    guard = MemoryGuard(max_memory_mb=mem_mb)
    return groupby_aggregate(
        make_iter(table, batch_size), by, agg_cols, funcs,
        guard, tmp_path, num_partitions=num_partitions,
        combiner=combiner, log=False,
    )


# helper: non-skew data
def _flat(n=1000, k=50):
    """n rows, k uniform groups (n/k rows each, no key is heavy)."""
    return pa.table({
        "k": [f"g{i % k:03d}" for i in range(n)],
        "v": list(range(n)),
    })


# ---------- equivalence: combiner vs no-combiner ----------

def test_combiner_count_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["count"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["count"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_sum_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["sum"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["sum"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_min_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["min"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["min"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_max_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["max"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["max"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_mean_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["mean"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["mean"], combiner=True)
    rows0 = r0.sort_by("k").to_pylist()
    rows1 = r1.sort_by("k").to_pylist()
    # Float comparison for mean.
    assert len(rows0) == len(rows1)
    for a, b in zip(rows0, rows1):
        assert a["k"] == b["k"]
        assert a["v_mean"] == pytest.approx(b["v_mean"], rel=1e-10)


# ---------- mixed aggs ----------

def test_combiner_mixed_aggs(tmp_path):
    table = pa.table({
        "k": [f"g{i % 50:03d}" for i in range(1000)],
        "v": list(range(1000)),
        "w": [i * 1.5 for i in range(1000)],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v", "w"], ["count", "mean"],
              combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v", "w"], ["count", "mean"],
              combiner=True)
    rows0 = r0.sort_by("k").to_pylist()
    rows1 = r1.sort_by("k").to_pylist()
    assert len(rows0) == len(rows1)
    for a, b in zip(rows0, rows1):
        assert a["k"] == b["k"]
        assert a["v_count"] == b["v_count"]
        assert a["w_mean"] == pytest.approx(b["w_mean"], rel=1e-10)


def test_combiner_same_col_sum_and_count(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v", "v"], ["sum", "count"],
              combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v", "v"], ["sum", "count"],
              combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


# ---------- std rejected (not in _AGG_FUNCS) ----------

def test_combiner_std_rejected_by_groupby(tmp_path):
    """std не входит в _AGG_FUNCS — ValueError до combiner-логики."""
    table = _flat(n=10, k=3)
    with pytest.raises(ValueError, match="std"):
        _run(tmp_path, table, ["k"], ["v"], ["std"], combiner=True)


# ---------- skew: warning + fallback ----------

def test_combiner_skew_warns_and_falls_back(tmp_path):
    """90/10 на n=1000 → оба ключа heavy → combiner off, warning."""
    import warnings
    table = pa.table({
        "k": ["A"] * 900 + ["B"] * 100,
        "v": [1.0] * 1000,
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(
            tmp_path, table, ["k"], ["v"], ["count"],
            combiner=True, num_partitions=4,
        )
    assert any("combiner" in str(w.message) and "skew" in str(w.message)
               for w in caught), (
        f"expected combiner-skew warning; got {[str(w.message) for w in caught]}"
    )
    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
    assert rows == {"A": 900, "B": 100}


# ---------- large N, low K: combiner actually active ----------

def test_combiner_path_actually_used(tmp_path, monkeypatch):
    """Spy on _combine_batches — proves combiner path executes (not fallback)."""
    from cadw.ops import groupby as gb_mod
    calls: list[bool] = []
    orig = gb_mod._combine_batches

    def spy(*args, **kwargs):
        calls.append(True)
        return orig(*args, **kwargs)

    monkeypatch.setattr(gb_mod, "_combine_batches", spy)

    table = _flat(n=1000, k=50)
    result = _run(tmp_path, table, ["k"], ["v"], ["count"], combiner=True)
    assert calls, (
        "combiner path NOT used — data still triggers skew detection, "
        "or combiner_active is False. Check _HEAVY_ABSOLUTE_MIN + ratio."
    )
    for r in result.to_pylist():
        assert r["v_count"] == 20  # 1000 / 50


def test_combiner_large_n_low_k(tmp_path):
    """1000 rows, 50 groups → each count = 20. Combiner path is non-skew."""
    table = _flat(n=1000, k=50)
    r_off = _run(tmp_path / "off", table, ["k"], ["v"], ["count"], combiner=False)
    r_on = _run(tmp_path / "on", table, ["k"], ["v"], ["count"], combiner=True)
    assert r_off.sort_by("k").to_pylist() == r_on.sort_by("k").to_pylist()
    for r in r_on.to_pylist():
        assert r["v_count"] == 20


# ---------- multi-key combiner ----------

def test_combiner_multi_by(tmp_path):
    n = 1000
    table = pa.table({
        "k1": [f"a{i % 10}" for i in range(n)],
        "k2": [f"b{(i // 100) % 10}" for i in range(n)],
        "v": list(range(n)),
    })
    r0 = _run(tmp_path / "off", table, ["k1", "k2"], ["v"], ["sum"],
              combiner=False)
    r1 = _run(tmp_path / "on", table, ["k1", "k2"], ["v"], ["sum"],
              combiner=True)
    rows0 = sorted(r0.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
    rows1 = sorted(r1.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
    assert rows0 == rows1


# ---------- empty input ----------

def test_combiner_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, pa.table({"k": [], "v": []}), ["k"], ["v"], ["sum"],
             combiner=True)


# ---------- API: через CadwDataFrame ----------

def test_dataframe_agg_with_combiner(tmp_path):
    """df.groupby('k').agg('v', 'sum', combiner=True) — non-skew data."""
    from cadw import CadwDataFrame, MemoryGuard
    import pyarrow.parquet as pq

    t = pa.table({
        "k": [f"g{i % 50:03d}" for i in range(1000)],
        "v": list(range(1000)),
    })
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.groupby("k").agg("v", "sum", combiner=True)
    out = tmp_path / "out.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 50
    # sum of v per group g{i}: values i, i+50, i+100, ... (20 items)
    # = 20*i + 50*(0+1+...+19) = 20*i + 50*190 = 20*i + 9500
    for r in rows:
        i = int(r["k"][1:])
        assert r["v_sum"] == pytest.approx(20 * i + 9500, rel=1e-10)
EOF

# ---------- DECISIONS.md: episode 52b ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_52b = """**Эпизод 52b (skew detection без абсолютного минимума = combiner
не тестируется).** Первая версия `test_combiner.py` прошла 12/13, но с
9 предупреждениями «combiner отключён в skew-пути». Причина:
`_detect_heavy_keys` флагил как heavy любой ключ с `ratio > 0.05` — без
абсолютного минимума. На данных 5 строк / 3 ключа (40/40/20%) — все
heavy; на 1000 строк / 10 групп (10% каждый) — все heavy. Combiner
**никогда не активировался**, тесты проходили через fallback (тот же
результат → ложные зелёные).

**Fix:** `_HEAVY_ABSOLUTE_MIN = 100`. Ключ считается heavy только если
`count >= 100 AND ratio > 0.05`. Существующие skew-тесты (`test_mustfix_skew
.py`) не сломались: 90/10 на n=1000 (A=900, B=100) — оба >= 100; 100/1000
uniform — каждый 1 < 100. Combiner-тесты переписаны на n=1000, k=50
(20 rows/group = 2% < 5%) — combiner активен.

**Дополнительно:** `test_combiner_std_warns_and_falls_back` был
некорректным — `std` не входит в `_AGG_FUNCS` и отклоняется **до**
combiner-логики. Заменён на `test_combiner_std_rejected_by_groupby`
(ожидает `ValueError`).

**Новый guard:** `test_combiner_path_actually_used` — monkeypatch
`_combine_batches`, проверяет что путь реально исполняется. Это
template для будущих оптимизаций: **«зелёные тесты не значит —
упражняется новая функциональность»**. Если fallback даёт тот же
результат, тест на эквивалентность проходит без вызова нового кода.

**Урок:** любой `if new_feature_enabled` требует теста, который
проверяет, что `new_feature` **вызывается** (spy/monkeypatch). Иначе
все «equivalence»-тесты могут проходить через старую ветку.

"""
if "Эпизод 52b" not in s:
    s = s.replace(marker_135, episode_52b + marker_135, 1)

# Обновить §11.5 с новым знанием про skew min
old_115 = "| `groupby_aggregate` combiner не работает в skew-пути (v0.3.4) |"
if old_115 in s and "`_HEAVY_ABSOLUTE_MIN = 100`" not in s.split(old_115)[0][-200:]:
    s = s.replace(
        old_115,
        "| `_detect_heavy_keys` absolute min = 100 (v0.3.4) | skew detection | Без абсолютного минимума tiny data (5 строк) ложно флагалась как skew. §13.4/52b. |\n"
        + old_115,
        1,
    )

p.write_text(s)
print("DECISIONS: §11.5, §13.4/52b")
PY

# ---------- run ----------
echo "=== pytest tests/test_combiner.py -v ==="
set +e
uv run pytest tests/test_combiner.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "combiner fix: rc=$RC"
  exit 1
fi

echo "=== pytest tests/test_mustfix_skew.py -v (regression check) ==="
set +e
uv run pytest tests/test_mustfix_skew.py -v
RC_SKEW=$?
set -e
if [[ $RC_SKEW -ne 0 ]]; then
  echo "skew tests: rc=$RC_SKEW"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.4-fix: _HEAVY_ABSOLUTE_MIN=100 (combiner не был активен на tiny data); combiner tests rewritten; §13.4/52b"

echo "==> v0.3.4-fix applied"