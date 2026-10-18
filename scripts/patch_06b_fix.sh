#!/usr/bin/env bash
# patch_06b-fix: ds.num_data() (метод), categorical test -> правильная семантика.
set -euo pipefail

BR="patch/06b-lightgbm"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-06b-fix-${TS}"
git diff > ".backups/backup-pre-06b-fix-${TS}/wip.diff"

# ---- Fix A: num_data -> num_data() во всех assert ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_lightgbm.py")
s = p.read_text()
if "ds.num_data()" not in s:
    s = s.replace("ds.num_data ==", "ds.num_data() ==")
    s = s.replace("ds.num_data <=", "ds.num_data() <=")
    p.write_text(s)
    print("fixed: ds.num_data() in assertions")
PY

# ---- Fix B: заменить categorical test на семантически корректный ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_lightgbm.py")
s = p.read_text()

old = '''def test_to_lightgbm_categorical(tmp_path):
    """categorical_cols передаётся в lgb.Dataset(categorical_feature=[...])."""
    rng = np.random.default_rng(0)
    n = 500
    cat = rng.choice(["a", "b", "c"], size=n).astype(object)
    x1 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 > 0).astype(np.int64)
    t = pa.table({"cat": cat, "x1": x1, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    ds = df.to_lightgbm(["x1"], "y", max_rows=None, categorical_cols=["cat"])
    # передача cat как y не работает; но мы проверяем только ошибку
    # (categorical_feature указывает на cat, которого нет в X)
    with pytest.raises(KeyError):
        df.to_lightgbm(["x1"], "y", max_rows=None, categorical_cols=["cat"])
'''

new = '''def test_to_lightgbm_categorical_must_be_in_x(tmp_path):
    """categorical_cols[i] обязан быть в X_cols (LightGBM categorical_feature = позиции в X)."""
    rng = np.random.default_rng(0)
    n = 200
    cat = rng.integers(0, 3, size=n).astype(np.int64)  # int-encoded
    x1 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 > 0).astype(np.int64)
    t = pa.table({"cat": cat, "x1": x1, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )

    # 1. cat не в X_cols -> KeyError
    with pytest.raises(KeyError, match="not in X_cols"):
        df.to_lightgbm(["x1"], "y", max_rows=None, categorical_cols=["cat"])

    # 2. cat в X_cols -> OK, categorical_feature=[индекс]
    ds = df.to_lightgbm(
        ["cat", "x1"], "y", max_rows=None, categorical_cols=["cat"]
    )
    ds.construct()
    assert ds.num_data() == n


def test_to_lightgbm_categorical_string_limitation(tmp_path):
    """Документированное ограничение: сырые строки не конвертируются автоматически.

    LightGBM требует int-encoded категории (или pandas category dtype).
    CADW не кодирует строки — это ответственность пользователя.
    """
    rng = np.random.default_rng(0)
    n = 100
    cat = rng.choice(["a", "b", "c"], size=n).astype(object)
    x1 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 > 0).astype(np.int64)
    t = pa.table({"cat": cat, "x1": x1, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=50,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    # np.column_stack со строками даст object dtype; LightGBM упадёт
    # (или мы упадём раньше на конвертации) — это ожидаемо.
    # Проверяем, что хотя бы один из вариантов даёт явную ошибку.
    with pytest.raises((ValueError, TypeError, KeyError)):
        ds = df.to_lightgbm(
            ["cat", "x1"], "y", max_rows=None, categorical_cols=["cat"]
        )
        ds.construct()
'''

if old not in s:
    raise SystemExit("categorical test: pattern not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("replaced: test_to_lightgbm_categorical -> 2 семантически корректных теста")
PY

# ---- Run ----
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

git add -A
git commit -m "06b-fix: ds.num_data() (метод), categorical test -> правильная семантика (int-encoded + KeyError path)"

echo "==> sha256"
sha256sum tests/test_lightgbm.py
echo "==> 06b-fix applied"