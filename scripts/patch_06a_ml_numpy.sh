#!/usr/bin/env bash
# patch_06a: to_batches(format=...) + partial_fit + E2E со sklearn.
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
BK="backup-pre-06a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_06a"

BR="patch/06a-ml-numpy"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- dataframe.py: добавить to_batches(format=...) + partial_fit ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 1. Импорт numpy + лениво torch внутри функций (не в шапке).
if "import numpy as np" not in s:
    s = s.replace(
        "from __future__ import annotations\n\nimport tempfile",
        "from __future__ import annotations\n\nimport tempfile",
        1,
    )
    # Вставить numpy после typing
    s = s.replace(
        "from typing import Iterator, Sequence",
        "from typing import Any, Iterator, Sequence\n\nimport numpy as np",
        1,
    )

# 2. Заменить to_batches на версию с format.
old_tobatches = '''    def to_batches(self, batch_size: int | None = None) -> Iterator[pa.RecordBatch]:
        """Поток батчей с учётом фильтров. batch_size=None — как в Dataset."""
        if not self._row_filters and batch_size is None:
            yield from self._dataset.iter_batches(columns=self._columns)
            return
        if not self._row_filters:
            # re-chunk без фильтров
            yield from self._dataset.iter_batches(
                batch_size=batch_size, columns=self._columns
            )
            return
        # С фильтрами: фильтруем поток, потом re-chunk на выходе
        if batch_size is None:
            yield from self._iter_filtered_batches()
            return
        accum: list[pa.RecordBatch] = []
        accum_n = 0
        for b in self._iter_filtered_batches():
            j = 0
            n = b.num_rows
            while j < n:
                take = min(batch_size - accum_n, n - j)
                accum.append(b.slice(j, take))
                accum_n += take
                j += take
                if accum_n == batch_size:
                    yield pa.Table.from_batches(accum).combine_chunks().to_batches()[0] \\
                        if len(accum) == 1 else _concat_batches(accum)
                    accum = []
                    accum_n = 0
        if accum_n > 0:
            yield _concat_batches(accum)'''

new_tobatches = '''    def to_batches(
        self,
        batch_size: int | None = None,
        format: str = "arrow",
    ):
        """Поток батчей с учётом фильтров.

        format:
          - "arrow": pa.RecordBatch (default, обратная совместимость);
          - "pandas": pd.DataFrame per batch;
          - "numpy": dict[str, np.ndarray] per batch;
          - "torch": dict[str, torch.Tensor] per batch (ленивый импорт torch).

        batch_size=None — как в Dataset (stored boundaries); N — re-chunk.
        """
        if format not in ("arrow", "pandas", "numpy", "torch"):
            raise ValueError(
                f"unsupported format {format!r}; "
                f"expected 'arrow' | 'pandas' | 'numpy' | 'torch'"
            )

        for b in self._iter_arrow_batches(batch_size=batch_size):
            yield _convert_batch(b, format)

    def _iter_arrow_batches(
        self, batch_size: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Внутренний поток RecordBatch с учётом фильтров и re-chunk."""
        if not self._row_filters and batch_size is None:
            yield from self._dataset.iter_batches(columns=self._columns)
            return
        if not self._row_filters:
            yield from self._dataset.iter_batches(
                batch_size=batch_size, columns=self._columns
            )
            return
        if batch_size is None:
            yield from self._iter_filtered_batches()
            return
        accum: list[pa.RecordBatch] = []
        accum_n = 0
        for b in self._iter_filtered_batches():
            j = 0
            n = b.num_rows
            while j < n:
                take = min(batch_size - accum_n, n - j)
                accum.append(b.slice(j, take))
                accum_n += take
                j += take
                if accum_n == batch_size:
                    yield _concat_batches(accum)
                    accum = []
                    accum_n = 0
        if accum_n > 0:
            yield _concat_batches(accum)'''

if old_tobatches not in s:
    raise SystemExit("to_batches: pattern not found (было ли применено 03c?)")
s = s.replace(old_tobatches, new_tobatches, 1)

# 3. Добавить partial_fit после write_parquet.
old_write = '''    # ---------------- helpers ----------------

    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":'''

new_write = '''    # ---------------- partial_fit ----------------

    def partial_fit(
        self,
        model: Any,
        X_cols: str | Sequence[str],
        y_col: str | None = None,
        *,
        batch_size: int | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Потоковое обучение модели с методом .partial_fit.

        - X_cols: колонки признаков (str или list[str]);
        - y_col=None: модель без y (StandardScaler, PCA). partial_fit(X, **kw).
        - y_col=str: partial_fit(X, y, **kw).

        Внутри: per-batch конвертация X (np.column_stack) и y (.to_numpy);
        никаких построчных циклов. Модель обучается батч за батчем.

        Возвращает обученную модель (тот же объект, изменённый in-place).
        """
        x_cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
        for c in x_cols:
            if c not in self._columns:
                raise KeyError(f"partial_fit: X column {c!r} not in selection")
        if y_col is not None and y_col not in self._columns:
            raise KeyError(f"partial_fit: y column {y_col!r} not in selection")

        n_batches = 0
        for b in self._iter_arrow_batches(batch_size=batch_size):
            X = np.column_stack(
                [b.column(c).to_numpy(zero_copy_only=False) for c in x_cols]
            )
            if y_col is not None:
                y = b.column(y_col).to_numpy(zero_copy_only=False)
                model.partial_fit(X, y, **fit_kwargs)
            else:
                model.partial_fit(X, **fit_kwargs)
            n_batches += 1
        if n_batches == 0:
            raise ValueError("partial_fit: no batches produced (empty data?)")
        return model

    # ---------------- helpers ----------------

    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":'''

if old_write not in s:
    raise SystemExit("partial_fit insertion point: pattern not found")
s = s.replace(old_write, new_write, 1)

# 4. Добавить _convert_batch в конец файла (module-level).
conv = '''

def _convert_batch(batch: pa.RecordBatch, format: str):
    """Конвертация одного батча. Векторизованно, без .as_py()."""
    if format == "arrow":
        return batch
    if format == "pandas":
        return batch.to_pandas()
    if format == "numpy":
        return {
            name: batch.column(i).to_numpy(zero_copy_only=False)
            for i, name in enumerate(batch.schema.names)
        }
    if format == "torch":
        import torch  # ленивый импорт
        return {
            name: torch.from_numpy(
                batch.column(i).to_numpy(zero_copy_only=False)
            )
            for i, name in enumerate(batch.schema.names)
        }
    raise ValueError(f"unsupported format {format!r}")
'''
if "_convert_batch(" not in s.split("class _GroupBy")[0]:
    s = s.rstrip() + conv

p.write_text(s)
print("patched dataframe.py: to_batches(format), partial_fit, _convert_batch")
PY

# ---------- tests ----------
cat > tests/test_ml_integration.py <<'EOF'
"""Тесты ML-интеграции (06a): to_batches(format) и partial_fit."""
import inspect
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw.dataframe as df_mod
from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path: Path, n: int = 100) -> CadwDataFrame:
    rng = np.random.default_rng(seed=0)
    x1 = rng.normal(0, 1, n).astype(np.float64)
    x2 = rng.normal(0, 1, n).astype(np.float64)
    y = (x1 + x2 > 0).astype(np.int64)
    t = pa.table({"x1": x1, "x2": x2, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=20,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )


# ---------- format conversion ----------

def test_to_batches_arrow_default(tmp_path):
    df = make_df(tmp_path)
    for b in df.to_batches():
        assert isinstance(b, pa.RecordBatch)


def test_to_batches_format_numpy(tmp_path):
    df = make_df(tmp_path)
    total = 0
    for d in df.to_batches(format="numpy"):
        assert isinstance(d, dict)
        assert set(d.keys()) == {"x1", "x2", "y"}
        for k, v in d.items():
            assert isinstance(v, np.ndarray)
        n = len(d["x1"])
        assert all(len(v) == n for v in d.values())
        total += n
    assert total == 100


def test_to_batches_format_pandas(tmp_path):
    import pandas as pd
    df = make_df(tmp_path)
    total = 0
    for pdf in df.to_batches(format="pandas"):
        assert isinstance(pdf, pd.DataFrame)
        assert list(pdf.columns) == ["x1", "x2", "y"]
        total += len(pdf)
    assert total == 100


def test_to_batches_format_torch_lazy(tmp_path):
    torch = pytest.importorskip("torch")
    df = make_df(tmp_path)
    for d in df.to_batches(format="torch"):
        assert isinstance(d, dict)
        for v in d.values():
            assert isinstance(v, torch.Tensor)


def test_to_batches_unknown_format_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="unsupported format"):
        list(df.to_batches(format="parquet"))


def test_to_batches_format_with_batch_size(tmp_path):
    df = make_df(tmp_path)
    sizes = [len(d["x1"]) for d in df.to_batches(batch_size=30, format="numpy")]
    assert sizes == [30, 30, 30, 10]


def test_to_batches_format_with_filter(tmp_path):
    df = make_df(tmp_path)
    sub = df[df["x1"] > 0]
    total = sum(len(d["x1"]) for d in sub.to_batches(format="numpy"))
    assert total == int((df.to_batches(format="numpy").__iter__().__next__() is not None)) * 0 + sub.count()


# ---------- partial_fit ----------

def test_partial_fit_sgd(tmp_path):
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path, n=200)
    model = SGDClassifier(loss="log_loss", random_state=0)
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50)
    # проверка: модель обучена, делает предсказания
    batches = list(df.to_batches(format="numpy"))
    X = np.vstack([np.column_stack([b["x1"], b["x2"]]) for b in batches])
    preds = model.predict(X)
    # на тренировочных данных accuracy > 0.9 (данные линейно разделимы)
    acc = float((preds == np.concatenate([b["y"] for b in batches])).mean())
    assert acc > 0.9, f"SGD accuracy {acc:.2f}"


def test_partial_fit_standard_scaler(tmp_path):
    from sklearn.preprocessing import StandardScaler
    df = make_df(tmp_path, n=200)
    scaler = StandardScaler()
    df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=50)
    # scaler обучен: mean ~0, scale ~1 (данные из N(0,1))
    assert abs(float(scaler.mean_[0])) < 0.3
    assert abs(float(scaler.scale_[0])) < 1.5
    assert abs(float(scaler.scale_[0])) > 0.5


def test_partial_fit_missing_x_col_raises(tmp_path):
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="X column"):
        df.partial_fit(SGDClassifier(), X_cols=["nope"], y_col="y")


def test_partial_fit_missing_y_col_raises(tmp_path):
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="y column"):
        df.partial_fit(SGDClassifier(), X_cols=["x1"], y_col="nope")


def test_partial_fit_after_filter(tmp_path):
    """partial_fit работает на отфильтрованном df."""
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path, n=200)
    filtered = df[df["x1"] > -2]  # отбрасываем экстремальные
    model = SGDClassifier(loss="log_loss", random_state=0)
    filtered.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50)
    # модель обучена (проверка: coef_ не пустой)
    assert model.coef_.shape == (1, 2)


# ---------- E2E: full sklearn pipeline ----------

@pytest.mark.slow
def test_e2e_sklearn_pipeline(tmp_path):
    """Стандартный sklearn-пайплайн: scaler.partial_fit → sgd.partial_fit → predict."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    # Больше данных — 50k строк, 5 MB бюджет
    rng = np.random.default_rng(seed=42)
    n = 50_000
    x1 = rng.normal(0, 1, n).astype(np.float32)
    x2 = rng.normal(0, 1, n).astype(np.float32)
    y = ((x1 + x2 > 0)).astype(np.int64)
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"x1": x1, "x2": x2, "y": y}), src,
                   row_group_size=10_000)

    guard = MemoryGuard(max_memory_mb=10)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5_000,
        memory_guard=guard, spill_dir=tmp_path / "spill",
    )

    # 1. scaler.partial_fit (без y)
    scaler = StandardScaler()
    df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=5_000)

    # 2. sgd.partial_fit (с y, на 3 эпохах для сходимости)
    model = SGDClassifier(loss="log_loss", random_state=0)
    for _ in range(3):
        df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5_000)

    # 3. predict на 1000 строк — accuracy
    head = df.head(1000)
    X = np.column_stack([
        head.column("x1").to_numpy(zero_copy_only=False),
        head.column("x2").to_numpy(zero_copy_only=False),
    ])
    y_true = head.column("y").to_numpy(zero_copy_only=False)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    acc = float((preds == y_true).mean())
    assert acc > 0.85, f"E2E sklearn pipeline accuracy {acc:.2f}"
EOF

# ---------- DECISIONS ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()
if "| ML: `to_batches(format)` + `partial_fit` (06a)" not in s:
    s = s.replace(
        "| E2E (04, filter→groupby→sort→parquet, 20 MB budget) | ✅ | 2 (slow) |",
        "| E2E (04, filter→groupby→sort→parquet, 20 MB budget) | ✅ | 2 (slow) |\n"
        "| ML: `to_batches(format)` + `partial_fit` (06a) | ✅ | 12 |",
        1,
    )
if "§12.5" in s and "06a ✅" not in s:
    s = s.replace(
        "### 12.5. Приоритет 4 — ML и визуализация",
        "### 12.5. Приоритет 4 — ML и визуализация (06a ✅: to_batches/partial_fit)",
        1,
    )
p.write_text(s)
print("DECISIONS §11.1 / §12.5 updated")
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

# ---------- commit ----------
git add -A
git commit -m "06a: to_batches(format='arrow'|'pandas'|'numpy'|'torch') + partial_fit (sklearn-compatible) + E2E sklearn pipeline"

echo "==> sha256"
sha256sum cadw/dataframe.py tests/test_ml_integration.py
echo "==> 06a applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"