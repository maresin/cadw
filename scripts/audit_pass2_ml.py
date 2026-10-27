"""Pass 2 state-class audit for cadw/ml/ (47 models).

Complements phase0c gates (which flag specific violations) by
emitting a full per-function table across all ML models. Feeds
priority ranking for P2b-full / P1a-b / P6.

Limitations (documented, see METHODOLOGY §8):
  - misses np.array([... for _ in range(n)])
  - misses broadcasting (n,) + (n,k)
  - misses allocs inside helpers (each function tracked separately)
  - may false-positive on predict paths where n = batch_size

Usage:
  uv run python scripts/audit_pass2_ml.py --write
"""
import argparse
import ast
from pathlib import Path

ML_DIR = Path("cadw/ml")
OUT_PATH = Path(".pingis/audits/state_class_pass2_ml.md")

N_NAMES = {
    'n', 'N', 'n_total', 'n_w', 'n_windows', 'n_obs', 'n_rows',
    'n_samples', 'n_valid', 'n_train', 'n_test', 'n_patterns',
    'n_queries', 'n_s', 'n_points',
}
ALLOC_CTORS = {'zeros', 'empty', 'full', 'ones'}
TILE_LIKE = {'tile', 'repeat'}
VSTACK_LIKE = {'vstack', 'hstack', 'stack', 'concatenate'}
GUARD_METHODS = {'check_state', 'check_batch',
                 '_probe_memory_limit_mb', 'check_partition_buffers'}


def _is_n_shape_expr(node):
    if isinstance(node, ast.Name):
        return (node.id in N_NAMES
                or node.id.endswith('_total')
                or node.id.endswith('_size'))
    if isinstance(node, ast.Tuple):
        return bool(node.elts) and _is_n_shape_expr(node.elts[0])
    if isinstance(node, ast.BinOp):
        return (_is_n_shape_expr(node.left)
                or _is_n_shape_expr(node.right))
    return False


def _func_name(call):
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _scan_function(fn):
    allocs = []
    reduce_calls = []
    spill_uses = []
    sliding_copy = []
    vstack_like = []
    guards = []

    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        fname = _func_name(node)
        if fname is None:
            continue

        if fname in GUARD_METHODS:
            guards.append(node.lineno)
        if fname == 'reduce':
            reduce_calls.append(node.lineno)
        if fname == 'memmap':
            spill_uses.append(node.lineno)
        if isinstance(node.func, ast.Name) and fname == 'SpillDir':
            spill_uses.append(node.lineno)

        if fname == 'copy' and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            if (isinstance(recv, ast.Call)
                    and isinstance(recv.func, ast.Attribute)
                    and recv.func.attr == 'sliding_window_view'):
                sliding_copy.append(node.lineno)

        if fname in ALLOC_CTORS and node.args:
            if _is_n_shape_expr(node.args[0]):
                try:
                    expr = ast.unparse(node.args[0])
                except Exception:
                    expr = '?'
                allocs.append(
                    (node.lineno, f"np.{fname}({expr}, ...)"))

        if fname in TILE_LIKE and len(node.args) >= 2:
            if _is_n_shape_expr(node.args[1]):
                try:
                    expr = ast.unparse(node.args[1])
                except Exception:
                    expr = '?'
                allocs.append(
                    (node.lineno, f"np.{fname}(..., {expr})"))

        if fname in VSTACK_LIKE and node.args:
            arg0 = node.args[0]
            is_name = isinstance(arg0, ast.Name)
            is_list = isinstance(arg0, ast.List)
            vstack_like.append((node.lineno, fname, is_name, is_list))

    has_nalloc = bool(allocs)
    has_sliding = bool(sliding_copy)
    has_guard = bool(guards)
    has_reduce = bool(reduce_calls)
    has_spill = bool(spill_uses)
    vstack_suspects = [v for v in vstack_like if v[2]]

    # v0.9.87-P5b: memmap/SpillDir is MORE informative than guarded-RAM
    # (function uses disk). Check it first so disk-users are visible.
    if has_spill:
        pattern = "memmap-SpillDir"
    elif has_nalloc:
        pattern = "guarded-RAM" if has_guard else "UNGUARDED-RAM"
    elif vstack_suspects:
        pattern = "list-accumulation"
    elif has_sliding:
        pattern = "windowed-copy"
    elif has_reduce:
        pattern = "streaming-via-reduce"
    else:
        pattern = "no-n-state"

    return {
        'name': fn.name, 'lineno': fn.lineno, 'pattern': pattern,
        'allocs': allocs, 'sliding_copy': sliding_copy,
        'vstack_suspects': vstack_suspects,
        'reduce_calls': reduce_calls, 'spill_uses': spill_uses,
        'guards': guards,
    }


def _scan_file(path):
    try:
        src = path.read_text()
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [{'file': str(path), 'error': str(e)}]
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        end = getattr(node, 'end_lineno', node.lineno)
        if end - node.lineno < 5:
            continue
        info = _scan_function(node)
        info['file'] = str(path)
        out.append(info)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--dirs', nargs='+', default=None,
                    help="Directories to scan (default: cadw/ml). "
                         "E.g. 'cadw/ops cadw/ts'")
    ap.add_argument('--out', default=None,
                    help="Output path (default: state_class_pass2_ml.md)")
    args = ap.parse_args()

    scan_dirs = [Path(d) for d in args.dirs] if args.dirs else [ML_DIR]
    out_path = Path(args.out) if args.out else OUT_PATH

    files = []
    for d in scan_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.py")))
    rows = []
    for f in files:
        if f.name == '__init__.py':
            continue
        rows.extend(_scan_file(f))

    counts = {}
    for r in rows:
        if 'error' in r:
            continue
        counts[r['pattern']] = counts.get(r['pattern'], 0) + 1

    by_file = {}
    for r in rows:
        if 'error' in r:
            continue
        by_file.setdefault(Path(r['file']).name, []).append(r)

    L = []
    L.append("# State-class audit Pass 2 — ML models (2026-10-28)")
    L.append("")
    L.append("Generated by `scripts/audit_pass2_ml.py`. Auto-heuristic.")
    L.append("Manual review of §Priority + predict-path false positives")
    L.append("required (predict-path n = batch_size is common).")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(f"- Files scanned: {len(by_file)}")
    L.append(f"- Functions classified: {sum(counts.values())}")
    L.append("")
    L.append("| pattern | count | meaning |")
    L.append("|---|---|---|")
    interps = {
        "streaming-via-reduce": "reduce() present, no n-alloc — SAFE",
        "no-n-state": "no n-sized alloc — SAFE (streaming or bounded)",
        "guarded-RAM": "n-alloc with check_state — inspected, OK",
        "memmap-SpillDir": "memmap or SpillDir present — SAFE (disk)",
        "list-accumulation": "vstack(Xs) of accumulated list — REVIEW",
        "UNGUARDED-RAM": "n-alloc, no guard — HIGH PRIORITY",
        "windowed-copy": "sliding_window_view.copy() — should be 0 (P4)",
    }
    for p in sorted(counts, key=lambda x: -counts[x]):
        L.append(f"| {p} | {counts[p]} | {interps.get(p, '')} |")
    L.append("")

    unguarded = [r for r in rows if r.get('pattern') == 'UNGUARDED-RAM']
    list_acc = [r for r in rows if r.get('pattern') == 'list-accumulation']
    windowed = [r for r in rows if r.get('pattern') == 'windowed-copy']

    L.append("## Priority (manual review)")
    L.append("")

    if unguarded:
        L.append(f"### UNGUARDED-RAM ({len(unguarded)})")
        L.append("")
        for r in unguarded:
            fn = r['file'].replace('cadw/ml/', '')
            a0 = r['allocs'][0] if r['allocs'] else (0, '?')
            L.append(f"- `{fn}::{r['name']}` (L{r['lineno']}) "
                     f"first_alloc@L{a0[0]}: `{a0[1]}`")
        L.append("")

    if list_acc:
        L.append(f"### list-accumulation ({len(list_acc)})")
        L.append("")
        for r in list_acc:
            fn = r['file'].replace('cadw/ml/', '')
            v = r['vstack_suspects'][0] if r['vstack_suspects'] else (0, '?', False, False)
            L.append(f"- `{fn}::{r['name']}` (L{r['lineno']}) "
                     f"`{v[1]}()`@L{v[0]} on Name arg")
        L.append("")

    if windowed:
        L.append(f"### windowed-copy ({len(windowed)}) — EXPECTED 0 after P4")
        L.append("")
        for r in windowed:
            fn = r['file'].replace('cadw/ml/', '')
            ln = r['sliding_copy'][0] if r['sliding_copy'] else 0
            L.append(f"- `{fn}::{r['name']}` (L{ln})")
        L.append("")

    L.append("## All functions (per-file)")
    L.append("")
    for fname in sorted(by_file):
        L.append(f"### {fname}")
        L.append("")
        L.append("| fn | L | pattern | allocs | guards |")
        L.append("|---|---|---|---|---|")
        for r in by_file[fname]:
            a_str = (', '.join(f"L{a[0]}" for a in r['allocs'])
                     if r['allocs'] else '-')
            g_str = str(len(r['guards'])) if r['guards'] else '-'
            L.append(f"| {r['name']} | {r['lineno']} | {r['pattern']} "
                     f"| {a_str} | {g_str} |")
        L.append("")

    errors = [r for r in rows if 'error' in r]
    if errors:
        L.append("## Errors")
        for r in errors:
            L.append(f"- {r['file']}: {r['error']}")

    text = '\n'.join(L)
    print(text)
    if args.write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + '\n')
        print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
