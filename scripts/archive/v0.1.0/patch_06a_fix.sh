#!/usr/bin/env bash
# patch_06a-fix: _convert_batch (unconditional append) + sklearn dep.
set -euo pipefail

BR="patch/06a-ml-numpy"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-06a-fix-${TS}"
git diff > ".backups/backup-pre-06a-fix-${TS}/wip.diff"

# ---- A. Append _convert_batch if no module-level def ----
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

if "\ndef _convert_batch(" in s:
    print("_convert_batch already defined; skip")
else:
    s = s.rstrip() + '''


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
    p.write_text(s)
    print("appended _convert_batch")
PY

# ---- B. sklearn в pyproject + установка ----
python3 - <<'PY'
from pathlib import Path
p = Path("pyproject.toml")
s = p.read_text()

# dev-group: создать, если нет
if "[dependency-groups]" not in s:
    s = s.rstrip() + '\n\n[dependency-groups]\ndev = [\n    "scikit-learn>=1.4",\n]\n'
    print("added [dependency-groups] dev = scikit-learn")
elif "scikit-learn" not in s.split("[dependency-groups]")[1].split("\n[")[0]:
    # вставить в dev-group
    import re
    def add(match):
        body = match.group(1)
        if "scikit-learn" in body:
            return match.group(0)
        return f"[dependency-groups]{body}    \"scikit-learn>=1.4\",\n"
    s2 = re.sub(r"\[dependency-groups\](.*?)(?=\n\[|\Z)", add, s, count=1, flags=re.DOTALL)
    if s2 != s:
        s = s2
        p.write_text(s)
        print("added scikit-learn to [dependency-groups] dev")
    else:
        print("WARN: could not auto-insert scikit-learn; check pyproject manually")
else:
    print("scikit-learn already in [dependency-groups]")

# если зависимость в основных deps (на случай, если dev-group не сработает)
if s != p.read_text():
    p.write_text(s)
PY

# Установить в venv (на случай, если lock не подхватит)
uv pip install "scikit-learn>=1.4"

# Sync lock + env
uv lock && uv sync

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
git commit -m "06a-fix: _convert_batch (unconditional), scikit-learn in dev-group"

echo "==> sha256"
sha256sum cadw/dataframe.py pyproject.toml
echo "==> 06a-fix applied"