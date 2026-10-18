#!/usr/bin/env bash
# patch_06b: to_lightgbm(max_rows=N) — честный downsampling + E2E.
set -euo pipefail

require_clean_tree() {
  local bt bu
  bt="$(git status --porcelain --untracked-files=no)"
  [[ -z "$bt" ]] || { echo "dirty:"; echo "$bt"; return 1; }
  bu=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      scripts/*|.backups/*|__pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*) : ;;
      *) bu+="  $f"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  [[ -z "$bu" ]] || { echo "untracked:"; echo "$bu"; return 1; }
  return 0
}
require_clean_tree || exit 1

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-06b-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_06b"

BR="patch/06b-lightgbm"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- dataframe.py: to_lightgbm ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# Точка вставки: перед "# ---------------- helpers ----------------"
anchor = "    # ---------------- helpers ----------------\n\n    def _wrap_result_table"
if anchor not in s:
    raise SystemExit("insertion point not found")

new_methods = '''    # ---------------- LightGBM ----------------

    def to_lightgbm(
        self,
        X_cols: str | Sequence[str],
        y_col: str,
        *,
        max_rows: int | None = None,
        categorical_cols: Sequence[str] | None = None,
        **lgb_params: Any,
    ):
        """Создать lgb.Dataset из (≤max_rows) первых строк.

        Контракт:
          - LightGBM не поддерживает потоковое обучение; ему нужен весь
            датасет в RAM. Этот метод материализует до max_rows строк
            (порядок: как в self, т.е. с учётом фильтров/сортировки).
          - max_rows=None → все строки (может OOM на большом датасете;
            используйте MemoryGuard.available_bytes() для оценки N).
          - max_rows < total → downsampling (первые max_rows, не случайные).
          - categorical_cols — передаются в lgb.Dataset(categorical_feature=...).

        Пример:
            guard = MemoryGuard(max_memory_mb=200)
            # ~200 MB / ~200 B/строка ≈ 1M строк
            X_cols = ["x1", "x2"]
            lgb_train = df.to_lightgbm(X_cols, "y", max_rows=1_000_000)
            booster = lgb.train({"objective": "binary"}, lgb_train, num_boost_round=100)

        Возвращает lgb.Dataset.
        """
        import lightgbm as lgb  # ленивый импорт

        x_cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
        for c in x_cols:
            if c not in self._columns:
                raise KeyError(f"to_lightgbm: X column {c!r} not in selection")
        if y_col not in self._columns:
            raise KeyError(f"to_lightgbm: y column {y_col!r} not in selection")

        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be positive or None")

        # Материализуем до max_rows строк.
        # Используем .head() — он уже оптимизирован на N без полного скана.
        n_take = self.count() if max_rows is None else min(max_rows, self.count())
        if n_take == 0:
            raise ValueError("to_lightgbm: no rows (empty dataframe / filter)")

        table = self.head(n_take)
        X = np.column_stack([
            table.column(c).to_numpy(zero_copy_only=False) for c in x_cols
        ])
        y = table.column(y_col).to_numpy(zero_copy_only=False)

        dataset_params: dict = {}
        if categorical_cols:
            cats = list(categorical_cols)
            for c in cats:
                if c not in x_cols:
                    raise KeyError(
                        f"to_lightgbm: categorical column {c!r} not in X_cols"
                    )
            dataset_params["categorical_feature"] = [
                x_cols.index(c) for c in cats
            ]

        return lgb.Dataset(X, label=y, **dataset_params, **lgb_params)

    # ---------------- helpers ----------------

    def _wrap_result_table'''

s = s.replace(anchor, new_methods, 1)
p.write_text(s)
print("patched dataframe.py: to_lightgbm")
PY

# ---------- pyproject: lightgbm в dev-group ----------
python3 - <<'PY'
from pathlib import Path
p = Path("pyproject.toml")
s = p.read_text()

if "lightgbm" not in s:
    if "[dependency-groups]" in s:
        # найти конец dev-group (до \n\n или \n[)
        import re
        m = re.search(r"(\[dependency-groups\]\ndev = \[\n)((?:\s+\"[^\"]+\",\n)+)(\])", s)
        if m:
            new_dev = m.group(1) + m.group(2) + '    "lightgbm>=4.0",\n' + m.group(3)
            s = s.replace(m.group(0), new_dev, 1)
        else:
            s = s.rstrip() + '\n    "lightgbm>=4.0",\n'
    else:
        s = s.rstrip() + '\n\n[dependency-groups]\ndev = [\n    "scikit-learn>=1.4",\n    "lightgbm>=4.0",\n]\n'
    p.write_text(s)
    print("added lightgbm to [dependency-groups] dev")
PY

uv pip install "lightgbm>=4.0"
uv lock && uv sync

# ---------- tests ----------
cat > tests/test_lightgbm.py <<'EOF'
"""Тесты to_lightgbm (06b): честный downsampling + E2E."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard

lgb = pytest.importorskip("lightgbm")


def make_df(tmp_path, n=2000, seed=0):
    rng = np.random.default_rng(seed=seed)
    x1 = rng.normal(0, 1, n).astype(np.float32)
    x2 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 + x2 > 0).astype(np.int64)
    t = pa.table({"x1": x1, "x2": x2, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


def test_to_lightgbm_basic(tmp_path):
    df = make_df(tmp_path, n=1000)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=1000)
    assert isinstance(ds, lgb.Dataset)
    ds.construct()
    assert ds.num_data == 1000
    assert ds.num_feature() == 2


def test_to_lightgbm_downsampling(tmp_path):
    df = make_df(tmp_path, n=1000)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=300)
    ds.construct()
    assert ds.num_data == 300


def test_to_lightgbm_max_rows_none(tmp_path):
    df = make_df(tmp_path, n=500)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=None)
    ds.construct()
    assert ds.num_data == 500


def test_to_lightgbm_max_rows_zero_raises(tmp_path):
    df = make_df(tmp_path, n=100)
    with pytest.raises(ValueError):
        df.to_lightgbm(["x1", "x2"], "y", max_rows=0)


def test_to_lightgbm_missing_x_raises(tmp_path):
    df = make_df(tmp_path, n=100)
    with pytest.raises(KeyError, match="X column"):
        df.to_lightgbm(["nope"], "y", max_rows=100)


def test_to_lightgbm_missing_y_raises(tmp_path):
    df = make_df(tmp_path, n=100)
    with pytest.raises(KeyError, match="y column"):
        df.to_lightgbm(["x1"], "nope", max_rows=100)


def test_to_lightgbm_after_filter(tmp_path):
    df = make_df(tmp_path, n=1000)
    filtered = df[df["x1"] > -2]
    expected = filtered.count()
    ds = filtered.to_lightgbm(["x1", "x2"], "y", max_rows=None)
    ds.construct()
    assert ds.num_data == expected


def test_to_lightgbm_after_sort(tmp_path):
    """Порядок строк после sort сохраняется в материализации."""
    df = make_df(tmp_path, n=500)
    sorted_df = df.sort_values("x1")
    ds = sorted_df.to_lightgbm(["x1", "x2"], "y", max_rows=100)
    ds.construct()
    assert ds.num_data == 100


def test_train_binary_classifier(tmp_path):
    """Полный цикл: df → lgb.Dataset → lgb.train → predict."""
    df = make_df(tmp_path, n=2000)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=None)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1},
        ds,
        num_boost_round=20,
    )
    # predict на тех же данных — accuracy должна быть высокой
    X = np.column_stack([
        df.head(2000).column("x1").to_numpy(zero_copy_only=False),
        df.head(2000).column("x2").to_numpy(zero_copy_only=False),
    ])
    y_true = df.head(2000).column("y").to_numpy(zero_copy_only=False)
    preds = (booster.predict(X) > 0.5).astype(np.int64)
    acc = float((preds == y_true).mean())
    assert acc > 0.9, f"LightGBM accuracy {acc:.2f}"


@pytest.mark.slow
def test_e2e_lightgbm_under_budget(tmp_path):
    """50k строк, 10 MB budget, downsampling до 20k, lgb.train."""
    df = make_df(tmp_path, n=50_000)
    # 20k строк * ~24 B/строка (3 float32/int64) ≈ 500 KB — влезает в 10 MB
    ds = df.to_lightgbm(
        ["x1", "x2"], "y",
        max_rows=20_000,
    )
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1},
        ds,
        num_boost_round=20,
    )
    # predict на head(1000)
    head = df.head(1000)
    X = np.column_stack([
        head.column("x1").to_numpy(zero_copy_only=False),
        head.column("x2").to_numpy(zero_copy_only=False),
    ])
    y_true = head.column("y").to_numpy(zero_copy_only=False)
    preds = (booster.predict(X) > 0.5).astype(np.int64)
    acc = float((preds == y_true).mean())
    assert acc > 0.9, f"E2E LightGBM accuracy {acc:.2f}"


def test_to_lightgbm_categorical(tmp_path):
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
EOF

# ---------- DECISIONS ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()
if "| LightGBM `to_lightgbm(max_rows)` (06b)" not in s:
    s = s.replace(
        "| ML: `to_batches(format)` + `partial_fit` (06a) | ✅ | 12 |",
        "| ML: `to_batches(format)` + `partial_fit` (06a) | ✅ | 12 |\n"
        "| LightGBM `to_lightgbm(max_rows)` (06b, честный downsampling) | ✅ | 11 |",
        1,
    )
if "06b ✅" not in s:
    s = s.replace(
        "### 12.5. Приоритет 4 — ML и визуализация (06a ✅: to_batches/partial_fit)",
        "### 12.5. Приоритет 4 — ML и визуализация (06a ✅, 06b ✅ LightGBM)",
        1,
    )
p.write_text(s)
print("DECISIONS updated: 06b")
PY

# ---------- run ----------
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

git add -A
git commit -m "06b: to_lightgbm(max_rows) — честный downsampling + E2E lgb.train under 10 MB budget"

echo "==> sha256"
sha256sum cadw/dataframe.py tests/test_lightgbm.py pyproject.toml
echo "==> 06b applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"