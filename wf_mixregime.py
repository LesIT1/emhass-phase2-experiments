#!/usr/bin/env python3
"""Decisive Phase 2 test: does the ADAPTIVE stochastic tree beat a single FIXED deterministic
P10-bias across a MIX of day types? bias=1.0 wins when uncertainty is always high, but it
over-hedges confident days; Phase 2's bands widen/narrow with the day's uncertainty. If no single
fixed bias matches Phase 2 across the mix, Phase 2's adaptivity has real value; if one does, the
cheap deterministic plan is enough. Run: uv run --with pulp --with numpy wf_mixregime.py"""
import numpy as np
import model as M

cfg = M.default_cfg()
NDAYS = 100
AFT_CHOICES = [0.0, 0.0, 0.15, 0.30, 0.45]   # mix: many confident days, some very uncertain
BIASES = [0.0, 0.4, 0.6, 0.8, 1.0]
rng = np.random.default_rng(7)
cols = {f"P1@{b}": [] for b in BIASES}
cols["P2"] = []
for _ in range(NDAYS):
    aft = float(rng.choice(AFT_CHOICES))
    pv = M.sample_pv(cfg, rng, aft)
    day, ok = {}, True
    for b in BIASES:
        r = M.run_closedloop(cfg, pv, False, b, aft)   # fixed bias, but forecast bands reflect this day's aft
        if r is None: ok = False; break
        day[f"P1@{b}"] = r["cost"]
    if ok:
        r2 = M.run_closedloop(cfg, pv, True, 0.0, aft)  # Phase 2: bands adapt to this day's uncertainty
        if r2 is None: ok = False
        else: day["P2"] = r2["cost"]
    if ok:
        for k, v in day.items(): cols[k].append(v)

arr = {k: np.array(v) for k, v in cols.items()}
p95 = lambda a: np.percentile(a, 95)
# CONTROL: determinism
pv0 = M.sample_pv(cfg, np.random.default_rng(7), 0.3)
d = abs(M.run_closedloop(cfg, pv0, True, 0, 0.3)["cost"] - M.run_closedloop(cfg, pv0, True, 0, 0.3)["cost"])
print(f"CONTROL determinism: {d:.8f} (must be 0)\n")

print(f"MIXED regimes (aft_sd drawn per day from {sorted(set(AFT_CHOICES))}), {len(arr['P2'])} days")
print(f"{'controller':<10} {'mean':>8} {'p90':>8} {'p95':>8} {'worst':>8}")
for k in [f"P1@{b}" for b in BIASES] + ["P2"]:
    a = arr[k]
    print(f"{k:<10} {a.mean():>8.4f} {np.percentile(a,90):>8.4f} {p95(a):>8.4f} {a.max():>8.4f}")

p1keys = [f"P1@{b}" for b in BIASES if b > 0]
bm = min(p1keys, key=lambda k: arr[k].mean())
bp = min(p1keys, key=lambda k: p95(arr[k]))
pm = lambda base: 100*(arr[base].mean()-arr["P2"].mean())/abs(arr[base].mean())
pt = lambda base: 100*(p95(arr[base])-p95(arr["P2"]))/abs(p95(arr[base]))
print(f"\nbest fixed-bias Phase1 by mean = {bm}; by p95 = {bp}")
print(f"P2 vs {bm} (best mean): mean {pm(bm):+.2f}%  p95 {pt(bm):+.2f}%  (+ = P2 better)")
print(f"P2 vs {bp} (best p95):  mean {pm(bp):+.2f}%  p95 {pt(bp):+.2f}%")
print("\nRead: if the SAME fixed bias is best on both mean and p95 AND ties P2, a fixed conservative")
print("deterministic plan suffices. If P2 beats every fixed bias (esp. on p95) because no single bias")
print("is right for both confident and uncertain days, that is Phase 2's adaptivity earning its keep.")
