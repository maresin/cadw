#!/usr/bin/env bash
# patch_v032_rank_bufferable_fix:
#   1. dataframe.py: rank_dtype учитывает method='average' (float64).
#   2. test_rank.py: убрать pandas 'last' (не существует), заменить на
#      reverse-first identity.
#   3. docstring: 'last' — CADW extension.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v032-rank-bufferable"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v032-fix-${TS}"
git diff > ".backups/backup-pre-v032-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.3.2 (rank bufferable; average dtype + 'last' extension; fix follows)" || true
require_clean_tree || exit 1

# ---------- Fix 1: dataframe.py — rank_dtype with method ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

old = 'rank_dtype = pa.float64() if pct else pa.int64()'
new = 'rank_dtype = pa.float64() if (pct or method == "average") else pa.int64()'
count = s.count(old)
if count == 0:
    raise SystemExit("rank_dtype line not found in dataframe.py")
s = s.replace(old, new)
print(f"dataframe.py: {count}× rank_dtype -> учитывает method=='average'")

# Docstring: пометить 'last' как extension
old_doc = "            - 'last':    обратный min (по позициям в группе)."
new_doc = "            - 'last':    reverse of 'first' (CADW extension, НЕ в pandas)."
if old_doc in s:
    s = s.replace(old_doc, new_doc, 1)
    print("dataframe.py: docstring — 'last' помечен как extension")

p.write_text(s)
print("saved")
PY

# ---------- Fix 2: test_rank.py — manual 'last' identity ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_rank.py")
s = p.read_text()

old = '''def test_rank_last_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="last")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="last").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )'''

new = '''def test_rank_last_is_reverse_first(tmp_path):
    """method='last' — CADW extension (pandas не поддерживает 'last').

    Семантика: reverse of 'first' within each tie group — элемент с
    меньшим row_id получает БОЛЬШИЙ rank. Identity:
        last(i) = group_first + group_last - first(i)
    где group_first/group_last — min/max first-rank в группе.
    """
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="last")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    tmp = pdf.copy()
    tmp["first"] = tmp["k"].rank(method="first").astype("int64")
    tmp["group_first"] = tmp.groupby("k")["first"].transform("min")
    tmp["group_last"] = tmp.groupby("k")["first"].transform("max")
    expected = (tmp["group_first"] + tmp["group_last"] - tmp["first"]).to_numpy()

    np.testing.assert_array_equal(got["k_rank"].to_numpy(), expected)'''

if old not in s:
    raise SystemExit("test_rank_last_pandas_equivalence block not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("test_rank.py: test_rank_last_pandas_equivalence → test_rank_last_is_reverse_first")
PY

# ---------- Fix 3: DECISIONS §11.5 — 'last' extension ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "`rank(method='last')` — CADW extension" not in s:
    anchor = "| `rank` nulls участвуют как значения"
    if anchor in s:
        s = s.replace(
            anchor,
            "| `rank(method='last')` — CADW extension (pandas не имеет 'last') | `rank` (v0.3.2) | Семантика: reverse of 'first' в tie-группе; identity `last(i) = group_first + group_last - first(i)`. Тестируется без pandas-эталона. |\n"
            + anchor,
            1,
        )
        print("DECISIONS §11.5: 'last' extension note")
    else:
        print("WARNING: §11.5 anchor not found; skipping note")
else:
    print("DECISIONS: 'last' extension note already present")

p.write_text(s)
PY

# ---------- run ----------
echo "=== pytest tests/test_rank.py -v ==="
set +e
uv run pytest tests/test_rank.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "rank fix: rc=$RC"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.2-fix: rank average → float64 (dtype); 'last' — CADW extension (reverse-of-first identity); 35/35 rank tests"

echo "==> v0.3.2-fix applied"