#!/usr/bin/env python3
"""
Two-stage stochastic LP/MILP spike for EMHASS #841 Phase 2.

Purpose: get real numbers behind the claims made to lutorm/David before any PR:
  1. Non-anticipativity: t=0 controls shared, t>=1 per-scenario recourse.
  2. Binary handling, three ways:
       SHARED  - one deferrable schedule y[t] for all scenarios (cheap reference).
       LOCKED  - per-scenario y[t,s] but tied equal by constraints (tests whether
                 MILP presolve aggregates them back down to ~SHARED cost).
       FREE    - per-scenario y[t,s], only t=0 tied (true two-stage recourse, the
                 faithful-but-expensive model).
  3. DET - deterministic solve on P50 only (today's behaviour), as the baseline.

Reports per variant: objective (expected cost), wall-clock solve time, nominal
binary count. The headline checks:
  - LOCKED solve time ~ SHARED  => presolve really does collapse the equalities
    (backs the "duplicate + lock is cheap" claim).
  - FREE objective <= SHARED     => value of binary recourse (lutorm's point that
    forcing one schedule across scenarios can be suboptimal).

Run:  uv run --with pulp --with numpy spike.py
Pure illustration, NOT EMHASS code. Profiles are stylised but asymmetric like our tariff.
"""

import time
import numpy as np
import pulp


# ---- stylised day (half-hourly) ----
def make_day(n=48):
    hour = np.arange(n) * (24.0 / n)
    # PV: daylight bell 6..18, ~5 kW peak, peak shifted to ~13:00
    pv = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi)) ** 1.3 * 5.0
    pv[(hour < 6) | (hour > 18)] = 0.0
    # Load: base + morning + evening
    load = np.full(n, 0.8)
    load[(hour >= 6) & (hour < 9)] += 0.8
    load[(hour >= 17) & (hour < 22)] += 1.5
    # Import tariff: cheap 09-12 ONLY, super peak 15-21, mid otherwise.
    # NOTE cheap-import window (09-12) is deliberately OFFSET from the PV peak (12-15)
    # so the best deferrable slot differs between a low-PV and a high-PV day:
    #   low-PV  -> run in 09-12 on cheap grid; high-PV -> run in 12-15 on free surplus PV.
    imp = np.full(n, 0.2369)
    imp[(hour >= 9) & (hour < 12)] = 0.0862
    imp[(hour >= 15) & (hour < 21)] = 0.5384
    exp = np.full(n, 0.03)  # near-zero feed-in -> asymmetric vs import
    return hour, pv, load, imp, exp


SCEN = {"P10": 0.4, "P50": 1.0, "P90": 1.4}     # PV multipliers
WEIGHTS = {"P10": 0.25, "P50": 0.5, "P90": 0.25}

# battery / deferrable params
CAP, PMAX, EFF, SOC0, SOCMIN, SOCMAX = 8.0, 4.0, 0.95, 0.3, 0.1, 1.0
DEF_KW, DEF_STEPS = 3.0, 6           # must-run 3 kW for 6 half-hours (3 h)


def solve(mode, n=48, gmax=10.0, cap=CAP, pmax=PMAX):
    hour, pv0, load, imp, exp = make_day(n)
    dt = 24.0 / n
    scen = ["P50"] if mode == "det" else list(SCEN.keys())
    w = {"P50": 1.0} if mode == "det" else WEIGHTS
    win = [t for t in range(n) if 9 <= hour[t] < 15]   # deferrable allowed in the cheap/solar window

    m = pulp.LpProblem("phase2", pulp.LpMinimize)

    gi, ge, bc, bd, soc = {}, {}, {}, {}, {}
    for s in scen:
        for t in range(n):
            gi[t, s] = pulp.LpVariable(f"gi_{t}_{s}", 0, gmax)
            ge[t, s] = pulp.LpVariable(f"ge_{t}_{s}", 0, gmax)
            bc[t, s] = pulp.LpVariable(f"bc_{t}_{s}", 0, pmax)
            bd[t, s] = pulp.LpVariable(f"bd_{t}_{s}", 0, pmax)
            soc[t, s] = pulp.LpVariable(f"soc_{t}_{s}", SOCMIN, SOCMAX)

    # deferrable binaries
    y = {}
    if mode == "shared":
        for t in range(n):
            y[t] = pulp.LpVariable(f"y_{t}", cat="Binary") if t in win else 0
        m += pulp.lpSum(y[t] for t in win) == DEF_STEPS
        defp = {(t, s): DEF_KW * (y[t] if t in win else 0) for s in scen for t in range(n)}
    else:  # det / locked / free  -> per-scenario binaries
        for s in scen:
            for t in range(n):
                y[t, s] = pulp.LpVariable(f"y_{t}_{s}", cat="Binary") if t in win else 0
        for s in scen:
            m += pulp.lpSum(y[t, s] for t in win) == DEF_STEPS
        defp = {(t, s): DEF_KW * (y[t, s] if t in win else 0) for s in scen for t in range(n)}
        if mode == "locked":      # tie every scenario's schedule equal (presolve target)
            for t in win:
                for s in scen[1:]:
                    m += y[t, s] == y[t, scen[0]]
        elif mode == "free":      # non-anticipativity at t=0 only
            if 0 in win:
                for s in scen[1:]:
                    m += y[0, s] == y[0, scen[0]]

    # balance + soc dynamics per scenario
    for s in scen:
        pv = pv0 * SCEN[s]
        for t in range(n):
            m += pv[t] + gi[t, s] + bd[t, s] == load[t] + defp[t, s] + bc[t, s] + ge[t, s]
            prev = SOC0 if t == 0 else soc[t - 1, s]
            m += soc[t, s] == prev + (bc[t, s] * EFF - bd[t, s] / EFF) * dt / cap

    # non-anticipativity: shared continuous t=0 controls across scenarios
    if mode != "det":
        for s in scen[1:]:
            m += gi[0, s] == gi[0, scen[0]]
            m += ge[0, s] == ge[0, scen[0]]
            m += bc[0, s] == bc[0, scen[0]]
            m += bd[0, s] == bd[0, scen[0]]

    m += pulp.lpSum(
        w[s] * pulp.lpSum((imp[t] * gi[t, s] - exp[t] * ge[t, s]) * dt for t in range(n))
        for s in scen
    )

    nbin = sum(1 for v in y.values() if isinstance(v, pulp.LpVariable))
    t0 = time.perf_counter()
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    dt_solve = time.perf_counter() - t0

    sched = {}
    if mode == "shared":
        sched["all"] = [round(hour[t], 1) for t in win if pulp.value(y[t]) and pulp.value(y[t]) > 0.5]
    else:
        for s in scen:
            sched[s] = [round(hour[t], 1) for t in win if pulp.value(y[t, s]) and pulp.value(y[t, s]) > 0.5]
    return {
        "mode": mode, "status": pulp.LpStatus[m.status],
        "obj": pulp.value(m.objective), "secs": dt_solve, "nbin": nbin,
        "t0_import": pulp.value(gi[0, scen[0]]), "t0_chg": pulp.value(bc[0, scen[0]]),
        "sched": sched,
    }


if __name__ == "__main__":
    print(f"{'variant':<8} {'status':<9} {'exp.cost':>9} {'solve_s':>9} {'nbin':>5}  t0(imp/chg)")
    rows = {}
    for mode in ["det", "shared", "locked", "free"]:
        best = min((solve(mode) for _ in range(3)), key=lambda r: r["secs"])  # min of 3 for timing
        rows[mode] = best
        print(f"{best['mode']:<8} {best['status']:<9} {best['obj']:>9.4f} "
              f"{best['secs']:>9.4f} {best['nbin']:>5}  "
              f"{best['t0_import']:.2f}/{best['t0_chg']:.2f}")

    print("\n--- read-out ---")
    sh, lo, fr = rows["shared"], rows["locked"], rows["free"]
    print(f"presolve check : LOCKED/SHARED solve-time ratio = {lo['secs']/sh['secs']:.2f}x "
          f"(near 1.0 => presolve collapses the equality-locked binaries)")
    print(f"recourse value : SHARED-FREE expected cost = {sh['obj']-fr['obj']:+.4f} "
          f"({100*(sh['obj']-fr['obj'])/abs(sh['obj']):+.2f}% ; >0 => free per-scenario binaries help)")
    print(f"FREE vs SHARED binaries: {fr['nbin']} vs {sh['nbin']} "
          f"({fr['nbin']/max(sh['nbin'],1):.1f}x nominal)")

    # Sensitivity: how the recourse value depends on battery flexibility.
    # A big battery arbitrages energy across time so WHEN the load runs barely matters;
    # a small/constrained battery exposes the per-scenario placement value.
    print("\n--- recourse value vs battery size (shared vs free) ---")
    print(f"{'cap_kWh':>7} {'shared':>9} {'free':>9} {'gap':>9} {'gap%':>7}")
    for cap in [8.0, 4.0, 2.0, 0.5]:
        s_sh = min((solve("shared", cap=cap) for _ in range(2)), key=lambda r: r["secs"])
        s_fr = min((solve("free", cap=cap) for _ in range(2)), key=lambda r: r["secs"])
        gap = s_sh["obj"] - s_fr["obj"]
        print(f"{cap:>7.1f} {s_sh['obj']:>9.4f} {s_fr['obj']:>9.4f} {gap:>+9.4f} "
              f"{100*gap/abs(s_sh['obj']):>+6.2f}%")

    print("\n--- divergence diagnostic (cap=2.0): does FREE actually place differently per scenario? ---")
    diag = solve("free", cap=2.0)
    for s, hrs in diag["sched"].items():
        print(f"  {s}: on-hours {hrs}")
    print(f"  shared schedule: {solve('shared', cap=2.0)['sched']['all']}")
