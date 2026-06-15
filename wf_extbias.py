#!/usr/bin/env python3
"""Follow-up to the workflow: extend the Phase 1 P10-bias grid above 0.8 (it pinned at the grid
ceiling in closedloop_regimes, so Phase 1 was likely understated and Phase 2's marginal edge
overstated). Re-test Phase 2 vs a properly-tuned deterministic Phase 1 at aft_sd=0.3.
Run: uv run --with pulp --with numpy wf_extbias.py"""
import numpy as np
import model as M

cfg = M.default_cfg()
AFT, NDAYS = 0.30, 80
BIASES = [0.0, 0.6, 0.8, 0.9, 1.0]   # 0.0 = naive P50 deterministic
rng = np.random.default_rng(2026)
cols = {f"P1@{b}": [] for b in BIASES}
cols["P2"] = []
for _ in range(NDAYS):
    pv = M.sample_pv(cfg, rng, AFT)
    day, ok = {}, True
    for b in BIASES:
        r = M.run_closedloop(cfg, pv, False, b, AFT)
        if r is None: ok = False; break
        day[f"P1@{b}"] = r["cost"]
    if ok:
        r2 = M.run_closedloop(cfg, pv, True, 0.0, AFT)
        if r2 is None: ok = False
        else: day["P2"] = r2["cost"]
    if ok:
        for k, v in day.items(): cols[k].append(v)

arr = {k: np.array(v) for k, v in cols.items()}
p95 = lambda a: np.percentile(a, 95)

# CONTROL: re-running the same day is bit-identical (CBC deterministic); and P1@0.0 == naive by construction.
pv0 = M.sample_pv(cfg, np.random.default_rng(2026), AFT)
c1 = M.run_closedloop(cfg, pv0, False, 0.0, AFT)["cost"]
c2 = M.run_closedloop(cfg, pv0, False, 0.0, AFT)["cost"]
print(f"CONTROL determinism: repeat run diff = {abs(c1-c2):.8f} (must be 0)\n")

print(f"closed-loop, aft_sd={AFT}, {len(arr['P2'])} days")
print(f"{'controller':<10} {'mean':>8} {'p90':>8} {'p95':>8} {'worst':>8}")
for k in [f"P1@{b}" for b in BIASES] + ["P2"]:
    a = arr[k]
    print(f"{k:<10} {a.mean():>8.4f} {np.percentile(a,90):>8.4f} {p95(a):>8.4f} {a.max():>8.4f}")

p1keys = [f"P1@{b}" for b in BIASES if b > 0]
best_mean = min(p1keys, key=lambda k: arr[k].mean())
best_p95 = min(p1keys, key=lambda k: p95(arr[k]))
pm = lambda base: 100*(arr[base].mean()-arr["P2"].mean())/abs(arr[base].mean())
pt = lambda base: 100*(p95(arr[base])-p95(arr["P2"]))/abs(p95(arr[base]))
print(f"\nPhase1 means by bias: " + ", ".join(f"{b}={arr[f'P1@{b}'].mean():.4f}" for b in BIASES))
print(f"best Phase1 by mean = {best_mean}; by p95 = {best_p95}")
print(f"P2 vs {best_mean} (best mean): mean {pm(best_mean):+.2f}%  p95 {pt(best_mean):+.2f}%  (+ = P2 better)")
print(f"P2 vs {best_p95} (best p95): mean {pm(best_p95):+.2f}%  p95 {pt(best_p95):+.2f}%")
print("\nRead: if Phase1 mean keeps falling to bias=1.0, the optimum is at/above the ceiling and")
print("Phase 2's mean edge vanishes; the question is whether P2 still beats the BEST bias on p95.")
