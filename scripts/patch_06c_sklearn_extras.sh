#!/usr/bin/env bash
# patch_06c: sklearn extras (MiniBatchKMeans, PCA) + README ML-секция.
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
BK="backup-pre-06c-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_06c"

BR="patch/06c-sklearn-extras"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- tests ----------
cat > tests/test_sklearn_extras.py <<'EOF'
"""Тесты 06c: sklearn-пайплайн через partial_fit (без LightGBM)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard

sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def make_df(tmp_path, n=2000, seed=0, n_features=4):
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(0, 1, n).astype(np.float32) for i in range(n_features)}
    t = pa.table(cols)
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


# ---------- MiniBatchKMeans ----------

def test_minibatch_kmeans_partial_fit(tmp_path):
    df = make_df(tmp_path, n=1000, n_features=2)
    model = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)
    df.partial_fit(model, X_cols=["x0", "x1"], y_col=None, batch_size=100)
    # model обучен: центроиды имеют форму (3, 2)
    assert model.cluster_centers_.shape == (3, 2)
    # предсказания работают
    head = df.head(50)
    X = np.column_stack([
        head.column("x0").to_numpy(zero_copy_only=False),
        head.column("x1").to_numpy(zero_copy_only=False),
    ])
    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1, 2})


def test_minibatch_kmeans_after_filter(tmp_path):
    df = make_df(tmp_path, n=500, n_features=2)
    filtered = df[df["x0"] > -2]
    model = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=50)
    filtered.partial_fit(model, X_cols=["x0", "x1"], y_col=None, batch_size=50)
    assert model.cluster_centers_.shape == (2, 2)


# ---------- PCA ----------

def test_pca_partial_fit(tmp_path):
    """PCA требует n_features в первом вызове (API sklearn, не CADW)."""
    df = make_df(tmp_path, n=1000, n_features=4)
    pca = PCA(n_components=2, random_state=0)
    # n_features=4 — количество X-колонок, требуемое sklearn на первом вызове
    df.partial_fit(
        pca, X_cols=["x0", "x1", "x2", "x3"], y_col=None,
        batch_size=100, n_features=4,
    )
    assert pca.components_.shape == (2, 4)
    # transform работает
    head = df.head(10)
    X = np.column_stack([
        head.column(c).to_numpy(zero_copy_only=False)
        for c in ["x0", "x1", "x2", "x3"]
    ])
    Z = pca.transform(X)
    assert Z.shape == (10, 2)


def test_pca_explained_variance(tmp_path):
    """На коррелированных данных PCA даёт высокий explained_variance_ratio[0]."""
    rng = np.random.default_rng(42)
    n = 1000
    x0 = rng.normal(0, 1, n).astype(np.float32)
    x1 = x0 + 0.01 * rng.normal(0, 1, n).astype(np.float32)  # почти линейно
    t = pa.table({"x0": x0, "x1": x1})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=200,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    pca = PCA(n_components=2, random_state=0)
    df.partial_fit(
        pca, X_cols=["x0", "x1"], y_col=None,
        batch_size=200, n_features=2,
    )
    # Первая компонента объясняет почти всё
    assert pca.explained_variance_ratio_[0] > 0.99


# ---------- E2E: sklearn pipeline без LightGBM ----------

@pytest.mark.slow
def test_e2e_sklearn_pipeline_kmeans_pca(tmp_path):
    """50k строк, 10 MB budget: scaler → kmeans, scaler → pca."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed=42)
    n = 50_000
    x0 = rng.normal(0, 1, n).astype(np.float32)
    x1 = rng.normal(0, 1, n).astype(np.float32)
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"x0": x0, "x1": x1}), src,
                   row_group_size=10_000)

    guard = MemoryGuard(max_memory_mb=10)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5_000,
        memory_guard=guard, spill_dir=tmp_path / "spill",
    )

    scaler = StandardScaler()
    df.partial_fit(scaler, X_cols=["x0", "x1"], y_col=None, batch_size=5_000)

    kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=1_000)
    df.partial_fit(kmeans, X_cols=["x0", "x1"], y_col=None, batch_size=5_000)

    pca = PCA(n_components=2, random_state=0)
    df.partial_fit(
        pca, X_cols=["x0", "x1"], y_col=None,
        batch_size=5_000, n_features=2,
    )

    assert scaler.mean_.shape == (2,)
    assert kmeans.cluster_centers_.shape == (5, 2)
    assert pca.components_.shape == (2, 2)
EOF

# ---------- DECISIONS §11.5: PCA ограничение ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker = "| `to_lightgbm` и категориальные признаки."
if marker in s and "`PCA.partial_fit` требует `n_features`" not in s:
    # Найти конец таблицы §11.5 и вставить до следующего раздела
    import re
    m = re.search(r"(### 11\.5\..*?)(?=###|\Z)", s, re.DOTALL)
    if m:
        block = m.group(1)
        new_row = (
            "| `PCA.partial_fit` требует `n_features` в первом вызове | "
            "`CadwDataFrame.partial_fit(pca, ..., n_features=d)` | "
            "передайте `n_features=len(X_cols)` в kwargs (API sklearn) |\n"
        )
        # Вставить после последней строки таблицы
        lines = block.splitlines(keepends=True)
        last_table_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("|") and not line.startswith("|---"):
                last_table_idx = i
        if last_table_idx > 0:
            lines.insert(last_table_idx + 1, new_row)
            s = s.replace(block, "".join(lines), 1)
            p.write_text(s)
            print("DECISIONS §11.5: PCA n_features row added")
PY

# ---------- README: ML-секция ----------
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
s = p.read_text()

ml_section = """
## ML-интеграция

### sklearn: partial_fit (потоковое обучение)

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

guard = MemoryGuard(max_memory_mb=100)
df = CadwDataFrame.from_parquet("data.parquet", "cache/", memory_guard=guard)

scaler = StandardScaler()
df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

model = SGDClassifier()
for _ in range(3):
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5000,
                   classes=[0, 1])
```

Батчи конвертируются в numpy на лету, память ограничена `MemoryGuard`.

### LightGBM: честный downsampling

```python
ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=1_000_000)
booster = lgb.train({"objective": "binary"}, ds, num_boost_round=100)
```

**Ограничение:** LightGBM требует весь датасет в RAM. `to_lightgbm(max_rows=N)`
материализует до N строк; для N больше бюджета — distributed LightGBM
(Spark/Dask) или downsampling. См. `DECISIONS.md` §11.5.

### Экспорт батчей

```python
for batch in df.to_batches(format="numpy", batch_size=10000):
    # batch: dict[str, np.ndarray]
    ...
for batch in df.to_batches(format="torch"):
    # batch: dict[str, torch.Tensor]
    ...
```

### Категориальные признаки (LightGBM)

LightGBM принимает `categorical_feature=[индексы в X]`; X должен быть
числовым. CADW не кодирует строки автоматически. Для строковых категорий
сначала кодируйте:

```python
# TODO: df.encode_categorical(["cat"]) — кандидат в §12.5-ext
df['cat_code'] = ...  # ваше кодирование
ds = df.to_lightgbm(["cat_code", "x1"], "y", categorical_cols=["cat_code"])
```
"""

if "## ML-интеграция" not in s:
    # Вставить перед "## Документация"
    anchor = "## Документация"
    if anchor in s:
        s = s.replace(anchor, ml_section + "\n" + anchor, 1)
        p.write_text(s)
        print("README: ML section added")
    else:
        s = s.rstrip() + "\n\n" + ml_section
        p.write_text(s)
        print("README: ML section appended")
else:
    print("README: ML section already present")
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
git commit -m "06c: sklearn extras (MiniBatchKMeans, PCA via partial_fit) + README ML-секция"

echo "==> sha256"
sha256sum tests/test_sklearn_extras.py README.md DECISIONS.md
echo "==> 06c applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"