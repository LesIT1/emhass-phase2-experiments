#!/usr/bin/env python3
"""
Shared, verified model library for the EMHASS #841 Phase 2 experiments.

Why this exists: the stochastic two-stage LP is where formulation bugs crept in (1-vs-3-scenario
alternate optima; band-collapse zeroing the Phase 1 blend; pure-CVaR over-hedging). Those were
all caught by CONTROLS. This module centralises the verified LP + controls so experiment drivers
compose on correct code instead of re-deriving it. Run `uv run --with pulp --with numpy model.py`
to execute the self-test (asserts the controls hold).

Core ideas:
  - 3+ PV scenarios, weighted; t=0 controls SHARED across scenarios (non-anticipativity),
    t>=1 is per-scenario recourse.
  - deferrable binary schedule: 'shared' (one schedule), 'free' (per-scenario, t=0 tied),
    'locked' (per-scenario tied equal == presolve test), or 'fixed' (given schedule).
  - optional CVaR objective (Rockafellar-Uryasev): (1-lam)*E[cost] + lam*CVaR_alpha.
  - paths entries may be (name, pv) or (name, pv, load) for per-scenario LOAD uncertainty.

API:
  default_cfg(**overrides) -> cfg dict
  profiles(cfg) -> hour, base_pv, load, imp, exp   (arrays length n)
  forecast(cfg, hour, base, tau, pv_obs, spread_mult=1.0, aft_sd=0.0) -> (p10,p50,p90) for [tau..n)
  plan(cfg, hour, imp, exp, load, tau, soc0, rem_def, paths, weights,
       binary_mode='shared', cvar=None, fixed_sched=None) -> dict(ok,obj,secs,nbin,sched,gi0,ge0,bc0,bd0,d0)
  open_loop(cfg, scenarios, weights, binary_mode='shared', cvar=None, fixed_sched=None) -> plan dict (full day)
  sample_pv(cfg, rng, aft_sd=0.0) -> realized PV array (morning/afternoon brightness + cloud noise)
  run_closedloop(cfg, pv_true, stochastic, bias=0.0, aft_sd=0.0, cvar=None, weights=(.3,.4,.3)) -> dict(cost,done)
"""

import time
import numpy as np
import pulp

Z90 = 1.2815515594


def default_cfg(**over):
    cfg = dict(
        n=24, dt=1.0, cap=8.0, pmax=4.0, eff=0.95, soc0=0.4, socmin=0.1, socmax=1.0,
        def_kw=2.0, def_hrs=4, win=(9, 15), split=13, pv_peak=5.0,
        load_base=0.6, load_morn=0.6, load_eve=1.5,
        t_base=0.2369, cheap=(9, 12, 0.0862), peakt=(15, 21, 0.5384), exp=0.05,
        eps_sd=0.14, theta_prior=(0.85, 0.28),
    )
    cfg.update(over)
    return cfg


def profiles(cfg):
    n = cfg["n"]
    hour = np.arange(n) * (24.0 / n)
    base = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi)) ** 1.3 * cfg["pv_peak"]
    base[(hour < 6) | (hour > 18)] = 0.0
    load = np.full(n, cfg["load_base"])
    load[(hour >= 6) & (hour < 9)] += cfg["load_morn"]
    load[(hour >= 17) & (hour < 22)] += cfg["load_eve"]
    imp = np.full(n, cfg["t_base"])
    c0, c1, cp = cfg["cheap"]; imp[(hour >= c0) & (hour < c1)] = cp
    p0, p1, pp = cfg["peakt"]; imp[(hour >= p0) & (hour < p1)] = pp
    exp = np.full(n, cfg["exp"])
    return hour, base, load, imp, exp


def forecast(cfg, hour, base, tau, pv_obs, spread_mult=1.0, aft_sd=0.0):
    n = cfg["n"]; eps_sd = cfg["eps_sd"]; split = cfg["split"]; mu0, sd0 = cfg["theta_prior"]
    obs = [pv_obs[k] / base[k] for k in range(tau + 1) if base[k] > 1e-6]
    if obs:
        th = float(np.mean(obs)); sd = max(eps_sd / np.sqrt(len(obs)), 0.02)
    else:
        th = mu0; sd = sd0
    p10, p50, p90 = [], [], []
    for t in range(tau, n):
        if t == tau:
            v = pv_obs[tau]; p10.append(v); p50.append(v); p90.append(v); continue
        mu = base[t] * th
        var = sd ** 2 + (th * eps_sd) ** 2
        if hour[t] >= split and hour[tau] < split:
            var += aft_sd ** 2
        dev = Z90 * base[t] * np.sqrt(var) * spread_mult
        p50.append(mu); p10.append(max(0.0, mu - dev)); p90.append(mu + dev)
    return np.array(p10), np.array(p50), np.array(p90)


def plan(cfg, hour, imp, exp, load, tau, soc0, rem_def, paths, weights,
         binary_mode="shared", cvar=None, fixed_sched=None):
    n = cfg["n"]; dt = cfg["dt"]; h = n - tau
    cap, pmax, eff = cfg["cap"], cfg["pmax"], cfg["eff"]
    smin, smax, dkw = cfg["socmin"], cfg["socmax"], cfg["def_kw"]
    wlo, whi = cfg["win"]
    win_l = [i for i in range(h) if wlo <= hour[tau + i] < whi]
    rem = int(max(0, min(rem_def, len(win_l))))
    names = [p[0] for p in paths]
    pvby = {p[0]: p[1] for p in paths}
    loadby = {p[0]: (p[2] if len(p) > 2 else None) for p in paths}

    m = pulp.LpProblem("p", pulp.LpMinimize)
    gi, ge, bc, bd, soc = {}, {}, {}, {}, {}
    for s in names:
        for i in range(h):
            gi[i, s] = pulp.LpVariable(f"gi_{i}_{s}", 0, 50)
            ge[i, s] = pulp.LpVariable(f"ge_{i}_{s}", 0, 50)
            bc[i, s] = pulp.LpVariable(f"bc_{i}_{s}", 0, pmax)
            bd[i, s] = pulp.LpVariable(f"bd_{i}_{s}", 0, pmax)
            soc[i, s] = pulp.LpVariable(f"soc_{i}_{s}", smin, smax)

    y = {}
    if binary_mode == "shared":
        for i in win_l:
            y[i] = pulp.LpVariable(f"y_{i}", cat="Binary")
        if win_l:
            m += pulp.lpSum(y[i] for i in win_l) == rem
        defp = {(i, s): dkw * (y[i] if i in win_l else 0.0) for s in names for i in range(h)}
    elif binary_mode in ("free", "locked"):
        for s in names:
            for i in win_l:
                y[i, s] = pulp.LpVariable(f"y_{i}_{s}", cat="Binary")
            if win_l:
                m += pulp.lpSum(y[i, s] for i in win_l) == rem
        if binary_mode == "free":
            if win_l and win_l[0] == 0:
                for s in names[1:]:
                    m += y[0, s] == y[0, names[0]]
        else:  # locked: tie every scenario's schedule equal (presolve target)
            for i in win_l:
                for s in names[1:]:
                    m += y[i, s] == y[i, names[0]]
        defp = {(i, s): dkw * (y[i, s] if i in win_l else 0.0) for s in names for i in range(h)}
    else:  # fixed
        fs = set(fixed_sched or [])
        defp = {(i, s): dkw * (1.0 if (tau + i) in fs else 0.0) for s in names for i in range(h)}

    for s in names:
        pv = pvby[s]
        for i in range(h):
            t = tau + i
            ld = loadby[s][i] if loadby[s] is not None else load[t]
            m += pv[i] + gi[i, s] + bd[i, s] == ld + defp[i, s] + bc[i, s] + ge[i, s]
            prev = soc0 if i == 0 else soc[i - 1, s]
            m += soc[i, s] == prev + (bc[i, s] * eff - bd[i, s] / eff) * dt / cap
    for s in names[1:]:
        m += gi[0, s] == gi[0, names[0]]
        m += ge[0, s] == ge[0, names[0]]
        m += bc[0, s] == bc[0, names[0]]
        m += bd[0, s] == bd[0, names[0]]

    wmap = dict(zip(names, weights))
    reg = 1e-5 * pulp.lpSum(bc[i, s] + bd[i, s] + gi[i, s] for s in names for i in range(h))
    cost_s = {s: pulp.lpSum((imp[tau + i] * gi[i, s] - exp[tau + i] * ge[i, s]) * dt for i in range(h))
              for s in names}
    ev = pulp.lpSum(wmap[s] * cost_s[s] for s in names)
    if cvar is not None and len(names) > 1:
        alpha, lam = cvar
        eta = pulp.LpVariable("eta")
        u = {s: pulp.LpVariable(f"u_{s}", lowBound=0) for s in names}
        for s in names:
            m += u[s] >= cost_s[s] - eta
        cv = eta + (1.0 / (1.0 - alpha)) * pulp.lpSum(wmap[s] * u[s] for s in names)
        m += (1.0 - lam) * ev + lam * cv + reg
    else:
        m += ev + reg

    t0 = time.perf_counter()
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    secs = time.perf_counter() - t0
    ok = pulp.LpStatus[m.status] == "Optimal"
    s0 = names[0]
    v = lambda x: (pulp.value(x) or 0.0)
    if binary_mode == "shared":
        sched = [tau + i for i in win_l if v(y[i]) > 0.5]
    elif binary_mode in ("free", "locked"):
        sched = [tau + i for i in win_l if v(y[i, s0]) > 0.5]
    else:
        sched = sorted(set(fixed_sched or []))
    nbin = sum(1 for var in y.values() if isinstance(var, pulp.LpVariable))
    return dict(ok=ok, obj=(pulp.value(m.objective) if ok else None), secs=secs, nbin=nbin,
                sched=sched, gi0=v(gi[0, s0]), ge0=v(ge[0, s0]), bc0=v(bc[0, s0]), bd0=v(bd[0, s0]),
                d0=int(tau in sched))


def open_loop(cfg, scenarios, weights, binary_mode="shared", cvar=None, fixed_sched=None):
    hour, base, load, imp, exp = profiles(cfg)
    return plan(cfg, hour, imp, exp, load, 0, cfg["soc0"], cfg["def_hrs"], scenarios, list(weights),
                binary_mode, cvar, fixed_sched)


def sample_pv(cfg, rng, aft_sd=0.0):
    hour, base, _, _, _ = profiles(cfg)
    tm = rng.uniform(0.4, 1.3)
    ta = float(np.clip(tm + rng.normal(0, aft_sd), 0.2, 1.5)) if aft_sd > 0 else tm
    bright = np.where(hour < cfg["split"], tm, ta)
    eps = np.where(base > 1e-6, rng.uniform(0.75, 1.25, cfg["n"]), 1.0)
    return base * bright * eps


def run_closedloop(cfg, pv_true, stochastic, bias=0.0, aft_sd=0.0, cvar=None,
                   weights=(0.3, 0.4, 0.3), spread_mult=1.0):
    # spread_mult scales the bands the controller PLANS with (truth is unchanged): use it to test
    # miscalibration (over/under-dispersed bands) and discretisation rules (e.g. ~P5/P95 ~ 1.283).
    hour, base, load, imp, exp = profiles(cfg)
    soc, done, cost = cfg["soc0"], 0, 0.0
    for tau in range(cfg["n"]):
        p10, p50, p90 = forecast(cfg, hour, base, tau, pv_true, spread_mult, aft_sd)
        half = (p90 - p10) / 2.0 if stochastic else np.zeros_like(p50)
        center = (1.0 - bias) * p50 + bias * p10
        paths = [("lo", np.maximum(0.0, center - half)), ("m", center), ("hi", center + half)]
        r = plan(cfg, hour, imp, exp, load, tau, soc, cfg["def_hrs"] - done, paths, list(weights),
                 "shared", cvar)
        if not r["ok"]:
            return None
        cost += (imp[tau] * r["gi0"] - exp[tau] * r["ge0"]) * cfg["dt"]
        soc += (r["bc0"] * cfg["eff"] - r["bd0"] / cfg["eff"]) * cfg["dt"] / cfg["cap"]
        done += r["d0"]
    return dict(cost=cost, done=done)


def self_test():
    cfg = default_cfg()
    hour, base, load, imp, exp = profiles(cfg)
    pv = base * 0.9
    fails = []

    # Control 1: single scenario -> shared == free == locked obj.
    a = open_loop(cfg, [("m", pv)], [1.0], "shared")
    b = open_loop(cfg, [("m", pv)], [1.0], "free")
    c = open_loop(cfg, [("m", pv)], [1.0], "locked")
    if not (a["ok"] and abs(a["obj"] - b["obj"]) < 1e-6 and abs(a["obj"] - c["obj"]) < 1e-6):
        fails.append(f"C1 single-scenario binary modes differ: {a['obj']},{b['obj']},{c['obj']}")

    # Control 2: 3 IDENTICAL scenarios -> shared == locked obj, and free <= shared (more freedom).
    s3 = [("lo", pv), ("m", pv), ("hi", pv)]
    sh = open_loop(cfg, s3, [0.3, 0.4, 0.3], "shared")
    lk = open_loop(cfg, s3, [0.3, 0.4, 0.3], "locked")
    fr = open_loop(cfg, s3, [0.3, 0.4, 0.3], "free")
    if not (abs(sh["obj"] - lk["obj"]) < 1e-6 and fr["obj"] <= sh["obj"] + 1e-6):
        fails.append(f"C2 identical-scenario: shared {sh['obj']} locked {lk['obj']} free {fr['obj']}")
    if lk["nbin"] != fr["nbin"] or lk["nbin"] <= sh["nbin"]:
        fails.append(f"C2 nbin: shared {sh['nbin']} locked {lk['nbin']} free {fr['nbin']}")

    # Control 3: closed-loop CVaR lam=0 == no CVaR (same realized cost).
    rng = np.random.default_rng(0)
    pvt = sample_pv(cfg, rng, 0.0)
    e0 = run_closedloop(cfg, pvt, True, 0.0, 0.0, cvar=None)
    e1 = run_closedloop(cfg, pvt, True, 0.0, 0.0, cvar=(0.8, 0.0))
    if not (e0 and e1 and abs(e0["cost"] - e1["cost"]) < 1e-6):
        fails.append(f"C3 cvar lam=0 != no-cvar: {e0} {e1}")

    # Control 4: deterministic closed loop completes the deferrable (carryover works).
    d = run_closedloop(cfg, pvt, False, 0.0, 0.0)
    if not (d and d["done"] == cfg["def_hrs"]):
        fails.append(f"C4 deferrable not completed: {d}")

    if fails:
        print("SELF-TEST FAILED:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("SELF-TEST PASSED: all 4 controls hold (single-scenario binary equivalence; "
          "identical-scenario shared==locked & free<=shared & nbin 3x; CVaR lam=0==EV; deferrable carryover).")


if __name__ == "__main__":
    self_test()
