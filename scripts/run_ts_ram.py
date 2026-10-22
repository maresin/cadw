import json, subprocess, sys

SCEN = [
    ("rolling_stats(n=2000,4g)", "scenario_rolling_stats", 32),
    ("rolling_crosscorr(n=2000,4g)", "scenario_crosscorr", 32),
    ("make_features(n=2000,4g)", "scenario_make_features", 32),
    ("GBDT(n=500)", "scenario_gbdt", 64),
    ("THF_OLS(n=300)", "scenario_thf_ols", 32),
    ("THF_MinT_WLS(n=300)", "scenario_thf_mint_wls", 32),
    ("backtest(n=200)", "scenario_backtest", 32),
    ("BASELINE_noop", "scenario_baseline_noop", 32),
]

CHILD = r"""
import gc, json, sys, time
sys.path.insert(0, '.')
import psutil, tracemalloc
import scripts.verify_ts_ram as m
fn = getattr(m, sys.argv[1])
gc.collect()
proc = psutil.Process(); base = proc.memory_info().rss
tracemalloc.start()
t0 = time.perf_counter()
fn()
dt = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({'rss': (proc.memory_info().rss-base)/1048576, 'peak': peak/1048576, 'sec': dt}))
"""

print('%-30s %10s %10s %8s %6s %s' % ('scenario','rss_delta','tm_peak','budget','sec','status'))
print('-'*85)
for name, fn, budget in SCEN:
    r = subprocess.run([sys.executable, '-c', CHILD, fn], capture_output=True, text=True)
    if r.returncode != 0:
        print(name, 'ERR', (r.stderr or '').strip().splitlines()[-1] if r.stderr else '?')
        continue
    d = json.loads(r.stdout.strip().splitlines()[-1])
    ok = 'OK' if d['peak'] <= budget else 'OVER'
    print('%-30s %10.2f %10.2f %8d %6.2f %s' % (name, d['rss'], d['peak'], budget, d['sec'], ok))
