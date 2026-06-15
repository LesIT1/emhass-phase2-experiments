#!/usr/bin/env python3
"""
Monte Carlo sweep of the Phase 2 two-stage model over many randomized states.

For each random state (day brightness, forecast spread, SoC, battery size, tariff, load,
deferrable size/hours/window) it measures two clean quantities:

  VSS (value of the stochastic solution): EEV - RP_shared.
      EEV       = expected cost when the deferrable SCHEDULE is committed from a deterministic
                  P50 plan, then continuous grid/battery recourse adapts per scenario.
      RP_shared = expected cost when the schedule is chosen accounting for all scenarios
                  (one shared schedule), continuous recourse per scenario.
      VSS >= 0 is how much planning the schedule under uncertainty beats planning on P50 alone.

  BINARY GAP (value of per-scenario binary recourse): RP_shared - RP_free.
      RP_free lets the deferrable schedule differ per scenario (t=0 tied). >= 0 is the extra
      value of paying for 3x the integer variables, the thing lutorm asked about.

Also tracks solve times (incl. LOCKED to recheck the presolve claim) and reports the
distribution of each gap across states, not just the mean.

Run:  uv run --with pulp --with numpy sim.py
Stylised, NOT EMHASS code. Directional evidence to bring to #841.
"""

import time
import numpy as np
import pulp

EFF, SOCMIN, SOCMAX, GMAX = 0.95, 0.1, 1.0, 20.0


def make_state(rng):
    n = 48
    hour = np.arange(n) * (24.0 / n)
    bright = rng.uniform(0.3, 1.1)
    pv = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi)) ** 1.3 * 5.0 * bright
    pv[(hour < 6) | (hour > 18)] = 0.0
    load = np.full(n, rng.uniform(0.4, 1.0))
    load[(hour >= 6) & (hour < 9)] += rng.uniform(0.3, 1.0)
    load[(hour >= 17) & (hour < 22)] += rng.uniform(1.0, 2.2)
    imp = np.full(n, rng.uniform(0.18, 0.28))
    cheap_end = rng.choice([11, 12, 13])
    imp[(hour >= 9) & (hour < cheap_end)] = rng.uniform(0.05, 0.12)
    peak_start = rng.choice([15, 16, 17])
    imp[(hour >= peak_start) & (hour < 21)] = rng.uniform(0.35, 0.65)
    exp = np.full(n, rng.uniform(0.02, 0.10))
    wopen = rng.choice([8, 9, 10])
    wclose = rng.choice([14, 15, 16])
    win = [t for t in range(n) if wopen <= hour[t] < wclose]
    def_steps = int(min(len(win), rng.integers(4, 11)))
    return {
        "n": n, "dt": 24.0 / n, "hour": hour, "pv": pv, "load": load, "imp": imp, "exp": exp,
        "win": win, "def_steps": def_steps, "def_kw": rng.uniform(1.0, 3.0),
        "cap": rng.uniform(2.0, 12.0), "pmax": rng.uniform(3.0, 6.0), "soc0": rng.uniform(0.1, 0.9),
        "lo": rng.uniform(0.30, 0.85), "hi": rng.uniform(1.10, 1.80),    # P10 / P90 PV multipliers
        "w": [0.30, 0.40, 0.30],
    }


def solve(st, scen, weights, binary_mode, fixed_sched=None):
    n, dt, win, K, defkw = st["n"], st["dt"], st["win"], st["def_steps"], st["def_kw"]
    cap, pmax, soc0 = st["cap"], st["pmax"], st["soc0"]
    m = pulp.LpProblem("p", pulp.LpMinimize)
    names = [s[0] for s in scen]
    gi, ge, bc, bd, soc = {}, {}, {}, {}, {}
    for s in names:
        for t in range(n):
            gi[t, s] = pulp.LpVariable(f"gi_{t}_{s}", 0, GMAX)
            ge[t, s] = pulp.LpVariable(f"ge_{t}_{s}", 0, GMAX)
            bc[t, s] = pulp.LpVariable(f"bc_{t}_{s}", 0, pmax)
            bd[t, s] = pulp.LpVariable(f"bd_{t}_{s}", 0, pmax)
            soc[t, s] = pulp.LpVariable(f"soc_{t}_{s}", SOCMIN, SOCMAX)

    y = {}
    if binary_mode == "shared":
        for t in win:
            y[t] = pulp.LpVariable(f"y_{t}", cat="Binary")
        m += pulp.lpSum(y[t] for t in win) == K
        defp = {(t, s): defkw * (y[t] if t in win else 0) for s in names for t in range(n)}
    elif binary_mode == "free":
        for s in names:
            for t in win:
                y[t, s] = pulp.LpVariable(f"y_{t}_{s}", cat="Binary")
            m += pulp.lpSum(y[t, s] for t in win) == K
        if win and win[0] == 0:
            for s in names[1:]:
                m += y[0, s] == y[0, names[0]]
        defp = {(t, s): defkw * (y[t, s] if t in win else 0) for s in names for t in range(n)}
    else:  # fixed schedule
        fs = set(fixed_sched)
        defp = {(t, s): defkw * (1.0 if t in fs else 0.0) for s in names for t in range(n)}

    for (s, mult) in scen:
        pv = st["pv"] * mult
        for t in range(n):
            m += pv[t] + gi[t, s] + bd[t, s] == st["load"][t] + defp[t, s] + bc[t, s] + ge[t, s]
            prev = soc0 if t == 0 else soc[t - 1, s]
            m += soc[t, s] == prev + (bc[t, s] * EFF - bd[t, s] / EFF) * dt / cap

    for s in names[1:]:   # non-anticipativity: shared t=0 continuous controls
        m += gi[0, s] == gi[0, names[0]]
        m += ge[0, s] == ge[0, names[0]]
        m += bc[0, s] == bc[0, names[0]]
        m += bd[0, s] == bd[0, names[0]]

    wmap = dict(zip(names, weights))
    m += pulp.lpSum(
        wmap[s] * pulp.lpSum((st["imp"][t] * gi[t, s] - st["exp"][t] * ge[t, s]) * dt for t in range(n))
        for s in names
    )
    nbin = sum(1 for v in y.values() if isinstance(v, pulp.LpVariable))
    t0 = time.perf_counter()
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    secs = time.perf_counter() - t0
    if binary_mode == "shared":
        sched = [t for t in win if pulp.value(y[t]) and pulp.value(y[t]) > 0.5]
    elif binary_mode == "free":
        s0 = names[0]
        sched = [t for t in win if pulp.value(y[t, s0]) and pulp.value(y[t, s0]) > 0.5]
    else:
        sched = list(fixed_sched)
    ok = pulp.LpStatus[m.status] == "Optimal"
    return {"obj": pulp.value(m.objective) if ok else None, "sched": sched, "secs": secs,
            "nbin": nbin, "ok": ok}


def run_state(st):
    scen3 = [("P10", st["lo"]), ("P50", 1.0), ("P90", st["hi"])]
    w3 = st["w"]
    det = solve(st, [("P50", 1.0)], [1.0], "shared")               # deterministic P50 plan
    rp_shared = solve(st, scen3, w3, "shared")                     # stochastic, one shared schedule
    rp_free = solve(st, scen3, w3, "free")                         # stochastic, per-scenario schedule
    # EEV: commit det's schedule, let continuous recourse adapt per scenario
    eev = solve(st, scen3, w3, "fixed", fixed_sched=det["sched"])
    for r in (det, rp_shared, rp_free, eev):
        if not r["ok"]:
            return None
    vss = eev["obj"] - rp_shared["obj"]
    bgap = rp_shared["obj"] - rp_free["obj"]
    base = abs(rp_shared["obj"]) if abs(rp_shared["obj"]) > 1e-6 else 1e-6
    return {
        "vss": vss, "vss_pct": 100 * vss / base,
        "bgap": bgap, "bgap_pct": 100 * bgap / base,
        "t_shared": rp_shared["secs"], "t_free": rp_free["secs"],
        "nbin_shared": rp_shared["nbin"], "nbin_free": rp_free["nbin"],
        "diverged": det["sched"] != rp_shared["sched"],
        "spread": st["hi"] - st["lo"], "cap": st["cap"],
    }


def pctl(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


if __name__ == "__main__":
    N = 250
    rng = np.random.default_rng(12345)
    rows = []
    t_all = time.perf_counter()
    for i in range(N):
        st = make_state(rng)
        r = run_state(st)
        if r:
            rows.append(r)
    took = time.perf_counter() - t_all

    vss = np.array([r["vss_pct"] for r in rows])
    bg = np.array([r["bgap_pct"] for r in rows])
    ts = np.array([r["t_shared"] for r in rows])
    tf = np.array([r["t_free"] for r in rows])

    print(f"states solved: {len(rows)}/{N}  in {took:.1f}s\n")
    print("VALUE OF STOCHASTIC SCHEDULING (VSS = EEV - RP_shared, % of expected cost)")
    print(f"  mean {vss.mean():+.3f}%  median {np.median(vss):+.3f}%  "
          f"p90 {pctl(vss,90):+.3f}%  p99 {pctl(vss,99):+.3f}%  max {vss.max():+.3f}%")
    print(f"  states with VSS > 1%: {100*np.mean(vss>1):.1f}%   > 0.1%: {100*np.mean(vss>0.1):.1f}%")
    print("\nVALUE OF FREE PER-SCENARIO BINARIES (RP_shared - RP_free, % of expected cost)")
    print(f"  mean {bg.mean():+.4f}%  median {np.median(bg):+.4f}%  "
          f"p90 {pctl(bg,90):+.4f}%  p99 {pctl(bg,99):+.4f}%  max {bg.max():+.4f}%")
    print(f"  states with binary gap > 0.5%: {100*np.mean(bg>0.5):.1f}%   > 0.1%: {100*np.mean(bg>0.1):.1f}%")
    print("\nSOLVE TIME (per solve, seconds)")
    print(f"  shared: mean {ts.mean():.3f} p90 {pctl(ts,90):.3f}    "
          f"free: mean {tf.mean():.3f} p90 {pctl(tf,90):.3f}    "
          f"free/shared mean ratio {tf.mean()/ts.mean():.2f}x")
    print(f"  nbin shared~{rows[0]['nbin_shared']} free~{rows[0]['nbin_free']} (per state varies with window)")
    print(f"  det schedule differed from stochastic-shared in {100*np.mean([r['diverged'] for r in rows]):.0f}% of states")

    spread = np.array([r["spread"] for r in rows])
    cap = np.array([r["cap"] for r in rows])
    sp_med, cap_med = np.median(spread), np.median(cap)
    print("\nWHERE THE STOCHASTIC VALUE LIVES (mean VSS %)")
    print(f"  wide forecast spread (>{sp_med:.2f}): {vss[spread> sp_med].mean():+.3f}%   "
          f"narrow (<=): {vss[spread<=sp_med].mean():+.3f}%")
    print(f"  small battery (<{cap_med:.1f} kWh): {vss[cap< cap_med].mean():+.3f}%   "
          f"large (>=): {vss[cap>=cap_med].mean():+.3f}%")
    hi = (spread > sp_med) & (cap < cap_med)
    print(f"  wide-spread AND small-battery: {vss[hi].mean():+.3f}%  (n={hi.sum()}, "
          f"p90 {pctl(vss[hi],90):+.3f}%)")
