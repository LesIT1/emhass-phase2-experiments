#!/usr/bin/env python3
"""
Stability experiment: Phase2 vs tuned-Phase1 vs naive over ~300 simulated days.
Reports mean / p90 / p95 with bootstrap 90% CIs, plus a control check.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, ".")
import model as M

SEED = 42
N_DAYS = 300
AFT_SD = 0.3
BOOTSTRAP_REPS = 2000
BOOTSTRAP_CI = 0.90

# Tuned Phase-1: P10-bias blend = 0.20 (from earlier experiments)
PHASE1_BIAS = 0.20

# ---------------------------------------------------------------
# CONTROL: zero-spread stochastic must equal deterministic P50
# This is a formulation-integrity check: if the forecast bands
# collapse to zero, stochastic (all 3 paths identical) must give
# the same cost as the naive (deterministic P50) controller.
# ---------------------------------------------------------------
def run_control():
    cfg = M.default_cfg()
    rng = M.np.random.default_rng(SEED)
    pv_true = M.sample_pv(cfg, rng, aft_sd=0.0)

    det  = M.run_closedloop(cfg, pv_true, stochastic=False, bias=0.0, aft_sd=0.0, spread_mult=1.0)
    zero = M.run_closedloop(cfg, pv_true, stochastic=True,  bias=0.0, aft_sd=0.0, spread_mult=0.0)

    if det is None or zero is None:
        return False, f"solver failed: det={det} zero={zero}"

    tol = 1e-4
    ok = abs(det["cost"] - zero["cost"]) < tol and det["done"] == zero["done"]
    detail = (
        f"zero-spread-stoch cost={zero['cost']:.6f}, done={zero['done']} | "
        f"deterministic cost={det['cost']:.6f}, done={det['done']} | "
        f"diff={abs(det['cost']-zero['cost']):.2e} tol={tol}"
    )
    return ok, detail


# ---------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------
def run_experiment():
    cfg = M.default_cfg()
    rng = np.random.default_rng(SEED)

    costs = {"naive": [], "phase1": [], "phase2": []}
    done_counts = {"naive": [], "phase1": [], "phase2": []}

    print(f"Running {N_DAYS} days  aft_sd={AFT_SD}  phase1_bias={PHASE1_BIAS} ...")
    t0 = time.perf_counter()

    for day in range(N_DAYS):
        pv_true = M.sample_pv(cfg, rng, aft_sd=AFT_SD)

        naive = M.run_closedloop(cfg, pv_true, stochastic=False, bias=0.0,   aft_sd=AFT_SD)
        ph1   = M.run_closedloop(cfg, pv_true, stochastic=False, bias=PHASE1_BIAS, aft_sd=AFT_SD)
        ph2   = M.run_closedloop(cfg, pv_true, stochastic=True,  bias=0.0,   aft_sd=AFT_SD)

        if naive is None or ph1 is None or ph2 is None:
            print(f"  WARNING: solver failure on day {day}, skipping")
            continue

        costs["naive"].append(naive["cost"])
        costs["phase1"].append(ph1["cost"])
        costs["phase2"].append(ph2["cost"])
        done_counts["naive"].append(naive["done"])
        done_counts["phase1"].append(ph1["done"])
        done_counts["phase2"].append(ph2["done"])

        if (day + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  day {day+1:3d}/{N_DAYS}  elapsed={elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    n = len(costs["naive"])
    print(f"Completed {n} days in {elapsed:.1f}s")
    return costs, done_counts, n


def bootstrap_stats(arr, rng, reps=BOOTSTRAP_REPS, ci=BOOTSTRAP_CI):
    """Return (mean, p90, p95) each with (point_est, ci_lo, ci_hi)."""
    arr = np.array(arr)
    n = len(arr)
    boot_mean = np.zeros(reps)
    boot_p90  = np.zeros(reps)
    boot_p95  = np.zeros(reps)
    for r in range(reps):
        s = rng.choice(arr, size=n, replace=True)
        boot_mean[r] = s.mean()
        boot_p90[r]  = np.percentile(s, 90)
        boot_p95[r]  = np.percentile(s, 95)

    alpha = (1 - ci) / 2
    def ci_interval(boot):
        return float(np.percentile(boot, alpha*100)), float(np.percentile(boot, (1-alpha)*100))

    return {
        "mean": {"est": float(arr.mean()),      "ci": ci_interval(boot_mean)},
        "p90":  {"est": float(np.percentile(arr, 90)), "ci": ci_interval(boot_p90)},
        "p95":  {"est": float(np.percentile(arr, 95)), "ci": ci_interval(boot_p95)},
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


def main():
    print("=" * 60)
    print("CONTROL CHECK")
    print("=" * 60)
    ctrl_ok, ctrl_detail = run_control()
    print(f"Control passed: {ctrl_ok}")
    print(f"Detail: {ctrl_detail}")
    if not ctrl_ok:
        print("\nCONTROL FAILED - stopping experiment.")
        result = {
            "control_passed": False,
            "control_detail": ctrl_detail,
            "error": "Control check failed - results would be unreliable"
        }
        with open("workflow-results/stability.result.json", "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    print()
    print("=" * 60)
    print("MAIN EXPERIMENT")
    print("=" * 60)
    costs, done_counts, n_days = run_experiment()

    rng_boot = np.random.default_rng(SEED + 1)
    stats = {}
    for key in ["naive", "phase1", "phase2"]:
        stats[key] = bootstrap_stats(costs[key], rng_boot)

    # Deferrable completion rate
    def_rate = {k: float(np.mean([d == 4 for d in done_counts[k]])) for k in done_counts}

    # Print results table
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Days simulated: {n_days}  aft_sd={AFT_SD}  bootstrap_reps={BOOTSTRAP_REPS}  CI={int(BOOTSTRAP_CI*100)}%")
    print()
    header = f"{'Metric':<22} {'Naive':>14} {'Phase1(b=0.20)':>16} {'Phase2':>14}"
    print(header)
    print("-" * len(header))

    def fmt_ci(s, key):
        lo, hi = s[key]["ci"]
        return f"{s[key]['est']:+.4f} [{lo:+.4f},{hi:+.4f}]"

    for metric in ["mean", "p90", "p95"]:
        n_s = stats["naive"][metric]
        p1_s = stats["phase1"][metric]
        p2_s = stats["phase2"][metric]
        print(f"  {metric.upper():<20} {fmt_ci(stats['naive'],metric):>28} {fmt_ci(stats['phase1'],metric):>28} {fmt_ci(stats['phase2'],metric):>28}")

    print()
    print("Min/Max cost (AUD):")
    for k in ["naive", "phase1", "phase2"]:
        print(f"  {k:<12} min={stats[k]['min']:+.4f}  max={stats[k]['max']:+.4f}")

    print()
    print("Deferrable completion rate (target=1.00):")
    for k in ["naive", "phase1", "phase2"]:
        print(f"  {k:<12} {def_rate[k]:.4f}")

    # Savings vs naive
    print()
    print("Mean savings vs naive (AUD/day):")
    naive_mean = stats["naive"]["mean"]["est"]
    for k in ["phase1", "phase2"]:
        saving = naive_mean - stats[k]["mean"]["est"]
        # CI for saving: bootstrap the difference
        rng2 = np.random.default_rng(SEED + 2)
        arr_n = np.array(costs["naive"])
        arr_k = np.array(costs[k])
        boot_diff = np.zeros(BOOTSTRAP_REPS)
        for r in range(BOOTSTRAP_REPS):
            idx = rng2.integers(0, n_days, size=n_days)
            boot_diff[r] = arr_n[idx].mean() - arr_k[idx].mean()
        alpha = (1 - BOOTSTRAP_CI) / 2
        ci_lo = float(np.percentile(boot_diff, alpha * 100))
        ci_hi = float(np.percentile(boot_diff, (1 - alpha) * 100))
        sig = "SOLID" if ci_lo > 0 else "NOT SIGNIFICANT"
        print(f"  vs {k:<12} {saving:+.4f}  90% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  -> {sig}")

    # Phase2 vs Phase1
    arr_p1 = np.array(costs["phase1"])
    arr_p2 = np.array(costs["phase2"])
    rng3 = np.random.default_rng(SEED + 3)
    boot_p2vp1 = np.zeros(BOOTSTRAP_REPS)
    for r in range(BOOTSTRAP_REPS):
        idx = rng3.integers(0, n_days, size=n_days)
        boot_p2vp1[r] = arr_p1[idx].mean() - arr_p2[idx].mean()
    alpha = (1 - BOOTSTRAP_CI) / 2
    ci_lo_p2vp1 = float(np.percentile(boot_p2vp1, alpha * 100))
    ci_hi_p2vp1 = float(np.percentile(boot_p2vp1, (1 - alpha) * 100))
    mean_p2vp1 = float(arr_p1.mean() - arr_p2.mean())
    sig_p2vp1 = "SOLID" if ci_lo_p2vp1 > 0 else ("NEGATIVE" if ci_hi_p2vp1 < 0 else "NOT SIGNIFICANT")
    print(f"  Phase2 vs Phase1     {mean_p2vp1:+.4f}  90% CI [{ci_lo_p2vp1:+.4f}, {ci_hi_p2vp1:+.4f}]  -> {sig_p2vp1}")

    print()
    print("Statistical notes:")
    print(f"  - worst-day is a single realization (max), not statistically stable")
    print(f"  - CI is bootstrap {int(BOOTSTRAP_CI*100)}% percentile interval over {BOOTSTRAP_REPS} resamples")
    print(f"  - stylized model: absolute AUD values not directly real-world comparable")

    # Collect results into a dict for JSON
    results_table = []
    for k in ["naive", "phase1", "phase2"]:
        for metric in ["mean", "p90", "p95"]:
            results_table.append({
                "controller": k,
                "metric": metric,
                "est": stats[k][metric]["est"],
                "ci_lo": stats[k][metric]["ci"][0],
                "ci_hi": stats[k][metric]["ci"][1],
            })

    result_json = {
        "experiment": "stability",
        "control_passed": True,
        "control_detail": ctrl_detail,
        "n_days": n_days,
        "aft_sd": AFT_SD,
        "phase1_bias": PHASE1_BIAS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_ci": BOOTSTRAP_CI,
        "results_table": results_table,
        "def_rate": def_rate,
        "min_max": {k: {"min": stats[k]["min"], "max": stats[k]["max"]} for k in stats},
        "savings_vs_naive": {
            "phase1": {
                "mean_saving": float(naive_mean - stats["phase1"]["mean"]["est"]),
            },
            "phase2": {
                "mean_saving": float(naive_mean - stats["phase2"]["mean"]["est"]),
            },
        },
        "phase2_vs_phase1": {
            "mean_saving": mean_p2vp1,
            "ci_lo": ci_lo_p2vp1,
            "ci_hi": ci_hi_p2vp1,
            "significant": sig_p2vp1,
        },
    }

    # Build the human-readable results string
    lines = []
    lines.append(f"Days={n_days}  aft_sd={AFT_SD}  bootstrap CI={int(BOOTSTRAP_CI*100)}%  reps={BOOTSTRAP_REPS}")
    lines.append("")
    lines.append(f"{'':22} {'Naive':>28} {'Phase1(b=0.20)':>28} {'Phase2':>28}")
    for metric in ["mean", "p90", "p95"]:
        row = f"  {metric.upper():<20}"
        for k in ["naive", "phase1", "phase2"]:
            s = stats[k][metric]
            lo, hi = s["ci"]
            row += f" {s['est']:+.4f}[{lo:+.4f},{hi:+.4f}]"
        lines.append(row)
    lines.append("")
    lines.append("Deferrable completion rates: " + "  ".join(f"{k}={def_rate[k]:.4f}" for k in def_rate))
    lines.append("")
    naive_m = stats["naive"]["mean"]["est"]
    p1_m = stats["phase1"]["mean"]["est"]
    p2_m = stats["phase2"]["mean"]["est"]
    lines.append(f"Phase1 saves {naive_m - p1_m:+.4f}/day vs naive")
    lines.append(f"Phase2 saves {naive_m - p2_m:+.4f}/day vs naive")
    lines.append(f"Phase2 vs Phase1: {mean_p2vp1:+.4f}/day 90%CI[{ci_lo_p2vp1:+.4f},{ci_hi_p2vp1:+.4f}] -> {sig_p2vp1}")

    results_str = "\n".join(lines)
    result_json["results_str"] = results_str

    print()
    print("=" * 60)
    print("Writing result JSON ...")
    with open("workflow-results/stability.result.json", "w") as f:
        json.dump(result_json, f, indent=2)
    print("Done -> workflow-results/stability.result.json")


if __name__ == "__main__":
    main()
