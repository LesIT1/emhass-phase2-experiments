#!/usr/bin/env python3
"""
Closed-loop receding-horizon MC for EMHASS #841 Phase 2.

The honest test: actually run the MPC loop. Each hour the controller observes current state
(SoC, deferrable hours done, today's PV so far), forecasts the rest of the day, plans, commits
ONLY the current step's action, then reality advances and it re-plans. Two controllers face the
SAME realized day:

  DET  : plans on the P50 forecast only (today's behaviour).
  STOCH: plans over P10/P50/P90 scenarios, shared deferrable schedule + continuous recourse,
         t=0 non-anticipativity (the Phase 2 model).

Realized cost is accrued from each committed t=0 action against the KNOWN current PV (standard
receding horizon: current PV observed, only the future is forecast). Forecast uncertainty
shrinks through the day as more daylight is observed (a day-level brightness Theta inferred from
observations, plus irreducible per-step cloud noise). The deferrable is must-run with elapsed
hours carried across steps (the operating-hours carryover that #983 is about).

Reports the distribution of realized DET-minus-STOCH savings across many sampled days.

Run:  uv run --with pulp --with numpy sim_closedloop.py
Stylised, NOT EMHASS code. Directional.
"""

import time
import numpy as np
import pulp

N = 24
HOUR = np.arange(N)
BASE = np.maximum(0.0, np.sin((HOUR - 6.0) / 12.0 * np.pi)) ** 1.3 * 5.0
BASE[(HOUR < 6) | (HOUR > 18)] = 0.0
LOAD = np.full(N, 0.6)
LOAD[(HOUR >= 6) & (HOUR < 9)] += 0.6
LOAD[(HOUR >= 17) & (HOUR < 22)] += 1.5
IMP = np.full(N, 0.2369)
IMP[(HOUR >= 9) & (HOUR < 12)] = 0.0862
IMP[(HOUR >= 15) & (HOUR < 21)] = 0.5384
EXP = np.full(N, 0.05)
WIN = [t for t in range(N) if 9 <= HOUR[t] < 15]     # deferrable window (6 slots)
DEF_KW, DEF_HRS = 2.0, 4                              # must-run 2 kW for 4 h
CAP, PMAX, EFF, SOC0, SOCMIN, SOCMAX = 8.0, 4.0, 0.95, 0.4, 0.1, 1.0
DT = 1.0
Z90 = 1.2815515594
THETA_PRIOR_MU, THETA_PRIOR_SD, EPS_SD = 0.85, 0.28, 0.14
Z_W = [0.30, 0.40, 0.30]
SPLIT = 13          # hour separating "morning" from "afternoon" brightness


def forecast(tau, pv_obs, spread_mult=1.0, aft_sd=0.0):
    """Return P10/P50/P90 PV arrays for the remaining horizon [tau..N-1].
    spread_mult scales the band width: 0 => P10=P50=P90 (STOCH collapses to DET, a control).
    aft_sd adds IRREDUCIBLE afternoon uncertainty when forecasting the afternoon from a morning
    decision (an afternoon cloud front the morning's PV cannot reveal) - it does not shrink with
    morning observations, so re-planning cannot resolve it before the morning commitment bites."""
    obs = [pv_obs[k] / BASE[k] for k in range(tau + 1) if BASE[k] > 1e-6]
    if obs:
        th = float(np.mean(obs)); sd = max(EPS_SD / np.sqrt(len(obs)), 0.02)
    else:
        th = THETA_PRIOR_MU; sd = THETA_PRIOR_SD
    p10, p50, p90 = [], [], []
    for t in range(tau, N):
        if t == tau:                       # current step: PV observed, no uncertainty
            v = pv_obs[tau]; p10.append(v); p50.append(v); p90.append(v); continue
        mu = BASE[t] * th
        var = sd ** 2 + (th * EPS_SD) ** 2
        if HOUR[t] >= SPLIT and HOUR[tau] < SPLIT:    # forecasting afternoon from the morning
            var += aft_sd ** 2
        dev = Z90 * BASE[t] * np.sqrt(var) * spread_mult
        p50.append(mu); p10.append(max(0.0, mu - dev)); p90.append(mu + dev)
    return np.array(p10), np.array(p50), np.array(p90)


def plan_step(tau, soc_cur, rem_def, paths, weights, cvar=None):
    """Solve the remaining-horizon plan; return committed t=0 (gi,ge,bc,bd,d).
    cvar=(alpha, lam): blend objective (1-lam)*E[cost] + lam*CVaR_alpha[cost] (Rockafellar-Uryasev,
    LP-representable). lam=0 == expected cost; lam=1 == pure tail. Only applies with >1 scenario."""
    h = N - tau
    win_l = [i for i in range(h) if (tau + i) in WIN]
    rem = int(max(0, min(rem_def, len(win_l))))
    names = [p[0] for p in paths]
    m = pulp.LpProblem("step", pulp.LpMinimize)
    gi, ge, bc, bd, soc = {}, {}, {}, {}, {}
    for s in names:
        for i in range(h):
            gi[i, s] = pulp.LpVariable(f"gi_{i}_{s}", 0, 50)
            ge[i, s] = pulp.LpVariable(f"ge_{i}_{s}", 0, 50)
            bc[i, s] = pulp.LpVariable(f"bc_{i}_{s}", 0, PMAX)
            bd[i, s] = pulp.LpVariable(f"bd_{i}_{s}", 0, PMAX)
            soc[i, s] = pulp.LpVariable(f"soc_{i}_{s}", SOCMIN, SOCMAX)
    y = {}
    for i in win_l:
        y[i] = pulp.LpVariable(f"y_{i}", cat="Binary")
    if win_l:
        m += pulp.lpSum(y[i] for i in win_l) == rem
    defp = {i: (DEF_KW * y[i] if i in win_l else 0.0) for i in range(h)}

    pv_by = dict(zip(names, [p[1] for p in paths]))
    for s in names:
        pv = pv_by[s]
        for i in range(h):
            t = tau + i
            m += pv[i] + gi[i, s] + bd[i, s] == LOAD[t] + defp[i] + bc[i, s] + ge[i, s]
            prev = soc_cur if i == 0 else soc[i - 1, s]
            m += soc[i, s] == prev + (bc[i, s] * EFF - bd[i, s] / EFF) * DT / CAP
    for s in names[1:]:                    # t=0 non-anticipativity (shared commit)
        m += gi[0, s] == gi[0, names[0]]
        m += ge[0, s] == ge[0, names[0]]
        m += bc[0, s] == bc[0, names[0]]
        m += bd[0, s] == bd[0, names[0]]
    wmap = dict(zip(names, weights))
    # tiny throughput regularizer canonicalizes degenerate optima (same-cost plans that differ
    # only in battery cycling / grid churn) so the committed t=0 action is deterministic.
    reg = 1e-5 * pulp.lpSum(bc[i, s] + bd[i, s] + gi[i, s] for s in names for i in range(h))
    cost_s = {s: pulp.lpSum((IMP[tau + i] * gi[i, s] - EXP[tau + i] * ge[i, s]) * DT for i in range(h))
              for s in names}
    ev = pulp.lpSum(wmap[s] * cost_s[s] for s in names)
    if cvar is not None and len(names) > 1:
        alpha, lam = cvar
        eta = pulp.LpVariable("eta")                       # VaR level (free)
        u = {s: pulp.LpVariable(f"u_{s}", lowBound=0) for s in names}
        for s in names:
            m += u[s] >= cost_s[s] - eta
        cvar_term = eta + (1.0 / (1.0 - alpha)) * pulp.lpSum(wmap[s] * u[s] for s in names)
        m += (1.0 - lam) * ev + lam * cvar_term + reg
    else:
        m += ev + reg
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    s0 = names[0]
    d0 = 1 if (0 in win_l and pulp.value(y[0]) and pulp.value(y[0]) > 0.5) else 0
    return (pulp.value(gi[0, s0]), pulp.value(ge[0, s0]),
            pulp.value(bc[0, s0]), pulp.value(bd[0, s0]), d0,
            pulp.LpStatus[m.status] == "Optimal")


def run_day(pv_true, stochastic, bias=0.0, aft_sd=0.0, cvar=None):
    # Always the SAME 3-scenario structure (removes the 1-vs-3 alternate-optima asymmetry) using
    # the REAL forecast bands. `stochastic` decides whether the 3 points spread or collapse:
    #   stochastic=False, bias=0   -> deterministic P50 plan (naive baseline)
    #   stochastic=False, bias>0   -> Phase 1: deterministic on a P10-biased blend (cheap hedge)
    #   stochastic=True,  bias=0   -> Phase 2: stochastic scenarios around P50
    soc, done, cost = SOC0, 0, 0.0
    for tau in range(N):
        p10, p50, p90 = forecast(tau, pv_true, 1.0, aft_sd)   # real bands always
        half = (p90 - p10) / 2.0 if stochastic else np.zeros_like(p50)
        center = (1.0 - bias) * p50 + bias * p10            # bias shifts the planned PV toward P10
        paths = [("lo", np.maximum(0.0, center - half)), ("m", center), ("hi", center + half)]
        gi0, ge0, bc0, bd0, d0, ok = plan_step(tau, soc, DEF_HRS - done, paths, Z_W, cvar)
        if not ok:
            return None
        cost += (IMP[tau] * gi0 - EXP[tau] * ge0) * DT
        soc += (bc0 * EFF - bd0 / EFF) * DT / CAP
        done += d0
    return cost, done


def compare(a, b):
    """realized saving of b vs a, per day, % of a's cost."""
    sav = np.array([100 * (ai - bi) / (abs(ai) if abs(ai) > 1e-6 else 1e-6) for ai, bi in zip(a, b)])
    return sav


if __name__ == "__main__":
    NDAYS, AFT = 50, 0.30
    LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]      # lam=0 is a CONTROL: must equal P2 expected-cost
    print(f"closed-loop, {NDAYS} days, aft_sd={AFT}. Sweeping CVaR risk weight lambda (alpha=0.8).")
    print("lam=0 == pure expected cost (control); lam=1 == pure worst-case. Tail metrics matter.\n")
    t0 = time.perf_counter()
    rng = np.random.default_rng(2026)
    cols = {"naive": [], "P1@0.8": []}
    for lam in LAMS:
        cols[f"P2 lam={lam}"] = []
    for _ in range(NDAYS):
        tm = rng.uniform(0.4, 1.3)
        ta = float(np.clip(tm + rng.normal(0, AFT), 0.2, 1.5))
        bright = np.where(HOUR < SPLIT, tm, ta)
        eps = np.where(BASE > 1e-6, rng.uniform(0.75, 1.25, N), 1.0)
        pv = BASE * bright * eps
        runs = {"naive": run_day(pv, False, 0.0, AFT), "P1@0.8": run_day(pv, False, 0.8, AFT)}
        for lam in LAMS:
            runs[f"P2 lam={lam}"] = run_day(pv, True, 0.0, AFT, cvar=(0.8, lam) if lam > 0 else None)
        if any(v is None for v in runs.values()):
            continue
        for k, v in runs.items():
            cols[k].append(v[0])
    arr = {k: np.array(v) for k, v in cols.items()}
    print(f"  {'controller':<12} {'mean':>8} {'p90':>8} {'p95':>8} {'worst':>8}")
    for k in ["naive", "P1@0.8"] + [f"P2 lam={lam}" for lam in LAMS]:
        a = arr[k]
        print(f"  {k:<12} {a.mean():>8.4f} {np.percentile(a,90):>8.4f} "
              f"{np.percentile(a,95):>8.4f} {a.max():>8.4f}")
    ev = arr["P2 lam=0.0"]
    print(f"\n  control check: P2 lam=0 vs P2-EV identical? mean diff "
          f"{abs(ev.mean()-arr['P2 lam=0.0'].mean()):.6f} (0 = same path)")
    print("  CVaR vs expected-cost Phase 2 (+ = CVaR better):")
    for lam in LAMS[1:]:
        a = arr[f"P2 lam={lam}"]
        print(f"    lam={lam}: mean {100*(ev.mean()-a.mean())/abs(ev.mean()):+6.2f}%  "
              f"p95 {100*(np.percentile(ev,95)-np.percentile(a,95))/abs(np.percentile(ev,95)):+6.2f}%  "
              f"worst {100*(ev.max()-a.max())/abs(ev.max()):+6.2f}%")
    print(f"\n(took {time.perf_counter()-t0:.0f}s)  Read: a small lambda that cuts p95/worst without")
    print("raising the mean would be the sweet spot; if every lambda hurts, CVaR is not the lever here.")
