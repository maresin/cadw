"""Тесты 06c: sklearn-пайплайн через partial_fit (без LightGBM)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard

sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


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
    """IncrementalPCA.partial_fit: n_features не нужен (выводится из X.shape[1])."""
    df = make_df(tmp_path, n=1000, n_features=4)
    pca = IncrementalPCA(n_components=2)
    # IncrementalPCA выводит n_features из X.shape[1] первого батча
    df.partial_fit(
        pca, X_cols=["x0", "x1", "x2", "x3"], y_col=None,
        batch_size=100,
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
    pca = IncrementalPCA(n_components=2)
    df.partial_fit(
        pca, X_cols=["x0", "x1"], y_col=None,
        batch_size=200,
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

    pca = IncrementalPCA(n_components=2)
    df.partial_fit(
        pca, X_cols=["x0", "x1"], y_col=None,
        batch_size=5_000,
    )

    assert scaler.mean_.shape == (2,)
    assert kmeans.cluster_centers_.shape == (5, 2)
    assert pca.components_.shape == (2, 2)
