#!/usr/bin/env python3
"""
CVaR risk frontier experiment for EMHASS #841 Phase 2.

Sweeps cvar alpha in {0.7, 0.8, 0.9} x lam in {0, 0.25, 0.5, 0.75, 1.0}
at aft_sd=0.3 over ~60 simulated days (fixed seed for reproducibility).

CONTROL: lam=0 with any alpha must equal cvar=None (pure EV baseline).
"""

import sys
import json
import time
import numpy as np

# Import verified model library - do NOT re-derive the LP
from model import default_cfg, run_closedloop, sample_pv

SEED = 42
N_DAYS = 60
AFT_SD = 0.3
ALPHAS = [0.7, 0.8, 0.9]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
WEIGHTS = (0.3, 0.4, 0.3)

def run_experiment():
    t_start = time.perf_counter()
    cfg = default_cfg()
    rng = np.random.default_rng(SEED)

    # Pre-generate all true PV traces so every setting sees the same days
    pv_traces = [sample_pv(cfg, rng, AFT_SD) for _ in range(N_DAYS)]

    print(f"Running CVaR frontier: {N_DAYS} days, aft_sd={AFT_SD}")
    print(f"Alphas: {ALPHAS}")
    print(f"Lambdas: {LAMS}")
    print()

    # -----------------------------------------------------------------------
    # CONTROL: Phase 2 baseline (cvar=None) - pure expected-cost stochastic
    # -----------------------------------------------------------------------
    print("=== CONTROL: Phase 2 baseline (cvar=None) ===")
    baseline_costs = []
    for pv_true in pv_traces:
        r = run_closedloop(cfg, pv_true, stochastic=True, bias=0.0,
                           aft_sd=AFT_SD, cvar=None, weights=WEIGHTS)
        if r is None:
            print("ERROR: baseline solve failed", file=sys.stderr)
            sys.exit(1)
        baseline_costs.append(r["cost"])

    baseline_costs = np.array(baseline_costs)
    bl_mean = float(np.mean(baseline_costs))
    bl_p95 = float(np.percentile(baseline_costs, 95))
    bl_worst = float(np.max(baseline_costs))
    print(f"  mean={bl_mean:.4f}  p95={bl_p95:.4f}  worst={bl_worst:.4f}")
    print()

    # -----------------------------------------------------------------------
    # CONTROL CHECK: lam=0 with any alpha must equal cvar=None (same costs)
    # -----------------------------------------------------------------------
    print("=== CONTROL CHECK: lam=0 == cvar=None ===")
    lam0_costs = []
    for pv_true in pv_traces:
        r = run_closedloop(cfg, pv_true, stochastic=True, bias=0.0,
                           aft_sd=AFT_SD, cvar=(ALPHAS[0], 0.0), weights=WEIGHTS)
        if r is None:
            print("ERROR: lam=0 solve failed", file=sys.stderr)
            sys.exit(1)
        lam0_costs.append(r["cost"])

    lam0_costs = np.array(lam0_costs)
    max_diff = float(np.max(np.abs(lam0_costs - baseline_costs)))
    control_passed = max_diff < 1e-5
    print(f"  max |lam=0 - baseline| = {max_diff:.2e}  -> CONTROL {'PASSED' if control_passed else 'FAILED'}")
    if not control_passed:
        print(f"CONTROL FAILED: lam=0 CVaR does not equal cvar=None. max_diff={max_diff:.6e}")
        sys.exit(2)
    print()

    # -----------------------------------------------------------------------
    # MAIN SWEEP
    # -----------------------------------------------------------------------
    print("=== MAIN SWEEP ===")
    print(f"{'alpha':>6} {'lam':>6} {'mean':>8} {'p95':>8} {'worst':>8} {'delta_mean':>11} {'delta_p95':>10}")
    print("-" * 72)

    results = {}
    for alpha in ALPHAS:
        for lam in LAMS:
            cvar_param = (alpha, lam) if lam > 0.0 else None  # lam=0 == no CVaR
            costs = []
            for pv_true in pv_traces:
                r = run_closedloop(cfg, pv_true, stochastic=True, bias=0.0,
                                   aft_sd=AFT_SD, cvar=cvar_param, weights=WEIGHTS)
                if r is None:
                    print(f"ERROR: solve failed alpha={alpha} lam={lam}", file=sys.stderr)
                    sys.exit(1)
                costs.append(r["cost"])

            costs = np.array(costs)
            mean = float(np.mean(costs))
            p95 = float(np.percentile(costs, 95))
            worst = float(np.max(costs))
            delta_mean = mean - bl_mean
            delta_p95 = p95 - bl_p95

            key = f"alpha={alpha}_lam={lam}"
            results[key] = {
                "alpha": alpha,
                "lam": lam,
                "mean": mean,
                "p95": p95,
                "worst": worst,
                "delta_mean": delta_mean,
                "delta_p95": delta_p95,
            }

            print(f"{alpha:>6.1f} {lam:>6.2f} {mean:>8.4f} {p95:>8.4f} {worst:>8.4f}"
                  f" {delta_mean:>+11.4f} {delta_p95:>+10.4f}")

    t_elapsed = time.perf_counter() - t_start
    print()
    print(f"Total elapsed: {t_elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # ANALYSIS: find best risk setting
    # -----------------------------------------------------------------------
    print()
    print("=== ANALYSIS ===")

    # Best = minimises p95 (tail reduction) without increasing mean
    # Filter to settings that don't raise mean by more than a small threshold
    MEAN_TOLERANCE = 0.002  # AUD - trivial cost penalty OK
    candidates = {k: v for k, v in results.items()
                  if v["delta_mean"] <= MEAN_TOLERANCE and v["lam"] > 0.0}

    if candidates:
        best_key = min(candidates, key=lambda k: candidates[k]["delta_p95"])
        best = candidates[best_key]
        print(f"Best risk setting (cuts tail without raising mean by >{MEAN_TOLERANCE}):")
        print(f"  {best_key}: mean_delta={best['delta_mean']:+.4f} p95_delta={best['delta_p95']:+.4f}")
    else:
        best_key = None
        best = None
        print("No setting cuts tail without raising mean (all add mean cost).")

    # Check pure CVaR (lam=1) over-hedging
    pure_cvar_keys = [k for k in results if results[k]["lam"] == 1.0]
    print()
    print("Pure CVaR (lam=1.0) over-hedging check:")
    for k in sorted(pure_cvar_keys):
        v = results[k]
        print(f"  {k}: mean_delta={v['delta_mean']:+.4f} p95_delta={v['delta_p95']:+.4f}")

    # -----------------------------------------------------------------------
    # OUTPUT STRUCTURE
    # -----------------------------------------------------------------------
    output = {
        "experiment": "cvar_frontier",
        "config": {
            "n_days": N_DAYS,
            "aft_sd": AFT_SD,
            "seed": SEED,
            "alphas": ALPHAS,
            "lams": LAMS,
        },
        "control": {
            "name": "lam=0 CVaR == cvar=None (pure EV)",
            "max_diff": max_diff,
            "passed": control_passed,
        },
        "baseline": {
            "cvar": None,
            "mean": bl_mean,
            "p95": bl_p95,
            "worst": bl_worst,
        },
        "results": results,
        "best_risk_setting": best_key,
        "best_risk_detail": best,
        "elapsed_s": t_elapsed,
    }

    # Pretty-print summary table for results field
    rows = []
    rows.append(f"{'alpha':>6} {'lam':>6} {'mean':>8} {'p95':>8} {'worst':>8} {'delta_mean':>11} {'delta_p95':>10}")
    rows.append("-" * 72)
    rows.append(f"{'--':>6} {'none':>6} {bl_mean:>8.4f} {bl_p95:>8.4f} {bl_worst:>8.4f} {'(baseline)':>11} {'':>10}")
    for alpha in ALPHAS:
        for lam in LAMS:
            key = f"alpha={alpha}_lam={lam}"
            v = results[key]
            rows.append(f"{v['alpha']:>6.1f} {v['lam']:>6.2f} {v['mean']:>8.4f} {v['p95']:>8.4f}"
                        f" {v['worst']:>8.4f} {v['delta_mean']:>+11.4f} {v['delta_p95']:>+10.4f}")
    output["results_table"] = "\n".join(rows)

    return output


if __name__ == "__main__":
    result = run_experiment()

    out_path = "workflow-results/cvar_frontier.result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Result written to {out_path}")
    print()
    print("=== RESULTS TABLE ===")
    print(result["results_table"])
    print()
    print(f"CONTROL PASSED: {result['control']['passed']}")
    print(f"Best risk setting: {result['best_risk_setting']}")
    if result["best_risk_detail"]:
        b = result["best_risk_detail"]
        print(f"  mean_delta={b['delta_mean']:+.4f}  p95_delta={b['delta_p95']:+.4f}")
