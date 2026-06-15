#!/usr/bin/env python3
"""The real competitor to Phase 2: an ADAPTIVE-bias deterministic plan (bias = clamp(k*u), where
u = forecast relative band width), mirroring our Solcast risk-aware bias. If adaptive-bias cheaply
matches the stochastic tree across mixed regimes, Phase 2 is largely redundant; if Phase 2 still
beats it on the tail, the full tree earns its place. Run: uv run --with pulp --with numpy wf_adaptbias.py"""
import numpy as np
import model as M

cfg = M.default_cfg()
NDAYS = 80
AFT_CHOICES = [0.0, 0.0, 0.15, 0.30, 0.45]


def run_adaptive(cfg, pv_true, aft, k, bias_max=1.0):
    hour, base, load, imp, exp = M.profiles(cfg)
    soc, done, cost = cfg["soc0"], 0, 0.0
    for tau in range(cfg["n"]):
        p10, p50, p90 = M.forecast(cfg, hour, base, tau, pv_true, 1.0, aft)
        mask = p50 > 0.05
        u = float(np.mean((p90[mask] - p10[mask]) / p50[mask])) if mask.any() else 0.0
        bias = min(bias_max, k * u)
        center = (1.0 - bias) * p50 + bias * p10
        r = M.plan(cfg, hour, imp, exp, load, tau, soc, cfg["def_hrs"] - done,
                   [("m", center)], [1.0], "shared")
        if not r["ok"]:
            return None
        cost += (imp[tau] * r["gi0"] - exp[tau] * r["ge0"]) * cfg["dt"]
        soc += (r["bc0"] * cfg["eff"] - r["bd0"] / cfg["eff"]) * cfg["dt"] / cfg["cap"]
        done += r["d0"]
    return cost


KS = [0.0, 0.75, 1.5]          # k=0 -> bias 0 -> naive (CONTROL)
FIXED = [0.8, 1.0]
rng = np.random.default_rng(7)
cols = {f"adapt_k{k}": [] for k in KS}
for b in FIXED: cols[f"fix@{b}"] = []
cols["P2"] = []
for _ in range(NDAYS):
    aft = float(rng.choice(AFT_CHOICES))
    pv = M.sample_pv(cfg, rng, aft)
    day, ok = {}, True
    for k in KS:
        c = run_adaptive(cfg, pv, aft, k)
        if c is None: ok = False; break
        day[f"adapt_k{k}"] = c
    if ok:
        for b in FIXED:
            r = M.run_closedloop(cfg, pv, False, b, aft)
            if r is None: ok = False; break
            day[f"fix@{b}"] = r["cost"]
    if ok:
        r2 = M.run_closedloop(cfg, pv, True, 0.0, aft)
        if r2 is None: ok = False
        else: day["P2"] = r2["cost"]
    if ok:
        for kk, vv in day.items(): cols[kk].append(vv)

arr = {k: np.array(v) for k, v in cols.items()}
p95 = lambda a: np.percentile(a, 95)
# CONTROL: adapt_k0 must equal the naive fixed-bias-0 plan day-for-day.
chk = M.run_closedloop(cfg, M.sample_pv(cfg, np.random.default_rng(7), 0.3), False, 0.0, 0.3)["cost"]
chk2 = run_adaptive(cfg, M.sample_pv(cfg, np.random.default_rng(7), 0.3), 0.3, 0.0)
print(f"CONTROL adapt_k0 == naive: diff {abs(chk-chk2):.8f} (must be 0)\n")

print(f"MIXED regimes, {len(arr['P2'])} days")
print(f"{'controller':<11} {'mean':>8} {'p90':>8} {'p95':>8} {'worst':>8}")
order = [f"adapt_k{k}" for k in KS] + [f"fix@{b}" for b in FIXED] + ["P2"]
for k in order:
    a = arr[k]
    print(f"{k:<11} {a.mean():>8.4f} {np.percentile(a,90):>8.4f} {p95(a):>8.4f} {a.max():>8.4f}")

cand = [c for c in order if c != "P2" and c != "adapt_k0.0"]
bm = min(cand, key=lambda c: arr[c].mean())
bp = min(cand, key=lambda c: p95(arr[c]))
pm = lambda base: 100*(arr[base].mean()-arr["P2"].mean())/abs(arr[base].mean())
pt = lambda base: 100*(p95(arr[base])-p95(arr["P2"]))/abs(p95(arr[base]))
print(f"\nbest non-P2 by mean = {bm}; by p95 = {bp}")
print(f"P2 vs {bm}: mean {pm(bm):+.2f}%  p95 {pt(bm):+.2f}%  (+ = P2 better)")
print(f"P2 vs {bp}: mean {pm(bp):+.2f}%  p95 {pt(bp):+.2f}%")
print("\nRead: if an adaptive-bias deterministic plan matches P2 on mean AND p95, the stochastic")
print("tree is redundant and the cheap Solcast-style adaptive bias is the answer. If P2 still leads")
print("the p95, the tree adds tail value the scalar adaptive bias cannot.")
