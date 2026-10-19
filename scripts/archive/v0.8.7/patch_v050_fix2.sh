#!/usr/bin/env bash
# patch_v050_fix2: test_hash_join_streaming_writes_incrementally — spy
# фильтрует по join_output.arrow (hash_shuffle тоже использует writer).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v050-ram-fixes"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v050-fix2-${TS}"
git diff > ".backups/backup-pre-v050-fix2-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.1-fix2 (streaming write test asserts 100; hash_shuffle also writes)" || true
require_clean_tree || exit 1

python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_streaming.py")
s = p.read_text()

old = '''def test_hash_join_streaming_writes_incrementally(tmp_path, monkeypatch):
    """Each output batch must be written immediately (not accumulated)."""
    write_calls = []
    orig_write = ArrowIPCBatchWriter.write_batch

    def spy(self, batch):
        write_calls.append(batch.num_rows)
        return orig_write(self, batch)

    monkeypatch.setattr(ArrowIPCBatchWriter, "write_batch", spy)

    left = make_df(tmp_path, "L", {"k": list(range(100)), "a": list(range(100))})
    right = make_df(tmp_path, "R", {"k": list(range(100)), "b": list(range(100))})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=4,
    )
    assert write_calls, "hash_join_streaming must write batches to disk"
    assert sum(write_calls) == 100'''

new = '''def test_hash_join_streaming_writes_incrementally(tmp_path, monkeypatch):
    """hash_join_streaming writes output incrementally (not accumulated).

    Note: hash_shuffle also uses ArrowIPCBatchWriter (for partition files).
    Spy filters by target path so we only count writes to join_output.arrow.
    """
    write_calls = []
    orig_init = ArrowIPCBatchWriter.__init__
    orig_write = ArrowIPCBatchWriter.write_batch

    def patched_init(self, path, schema, *args, **kwargs):
        self._test_path = str(path)
        return orig_init(self, path, schema, *args, **kwargs)

    def spy(self, batch):
        # Only count writes to the final join_output.arrow (P2 target).
        if "join_output.arrow" in getattr(self, "_test_path", ""):
            write_calls.append(batch.num_rows)
        return orig_write(self, batch)

    monkeypatch.setattr(ArrowIPCBatchWriter, "__init__", patched_init)
    monkeypatch.setattr(ArrowIPCBatchWriter, "write_batch", spy)

    left = make_df(tmp_path, "L", {"k": list(range(100)), "a": list(range(100))})
    right = make_df(tmp_path, "R", {"k": list(range(100)), "b": list(range(100))})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=4,
    )
    assert write_calls, "hash_join_streaming must write to join_output.arrow"
    assert sum(write_calls) == 100, (
        f"total rows written to join_output.arrow: {sum(write_calls)} (expected 100)"
    )
    assert len(write_calls) > 1, (
        f"must write incrementally (>1 batch); got {len(write_calls)} call(s)"
    )'''

if old not in s:
    raise SystemExit("target test not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("test_join_streaming.py: spy filters by join_output.arrow")
PY

echo "=== pytest: v0.5.1 target tests ==="
set +e
uv run pytest \
  tests/test_join.py \
  tests/test_join_streaming.py \
  tests/test_join_oom_guard.py \
  tests/test_reduce_state_bytes.py \
  tests/test_merge_planner.py \
  -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "target: rc=$RC"
  exit 1
fi

echo "=== pytest: full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest: full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.1-fix2: streaming write test filters by join_output.arrow (hash_shuffle writes too)"

echo "==> v0.5.1-fix2 applied"