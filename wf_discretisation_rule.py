#!/usr/bin/env python3
"""
EMHASS #841 Phase 2 – Experiment: discretisation_rule

Compares different 3-point quadrature rules for discretising the PV forecast
distribution in closed-loop operation.

Rules tested:
  A  – Deciles P10/P50/P90, weights [0.3, 0.4, 0.3], spread_mult=1.0
  B  – Pearson-Tukey ~P5/P95,  weights [0.185, 0.63, 0.185], spread_mult=1.283
  C1 – Narrow tails,  spread_mult=0.5, weights [0.3, 0.4, 0.3]
  C2 – Moderate narrow, spread_mult=0.7, weights [0.3, 0.4, 0.3]
  DET – Deterministic baseline (spread_mult=0 -> single P50 scenario, stochastic=False)

CONTROL: stochastic run with spread_mult=0 (bands collapse to zero) must equal
         the deterministic (bias=0) baseline within tolerance, because both
         reduce to a single P50 path with shared t=0 decision.

Uses fixed seeds for reproducibility. ~80 days.
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory so model.py can be found when running from elsewhere
sys.path.insert(0, os.path.dirname(__file__))
import model

SEED = 42
N_DAYS = 80
AFT_SD = 0.3

# --- Rule definitions ---
RULES = [
    # (label, stochastic, spread_mult, weights)
    ("DET",  False, 1.0,   (0.3, 0.4, 0.3)),   # deterministic baseline (bias=0, stochastic=False)
    ("A",    True,  1.0,   (0.3, 0.4, 0.3)),   # deciles P10/P50/P90
    ("B",    True,  1.283, (0.185, 0.63, 0.185)), # Pearson-Tukey ~P5/P95
    ("C1",   True,  0.5,   (0.3, 0.4, 0.3)),   # narrow
    ("C2",   True,  0.7,   (0.3, 0.4, 0.3)),   # moderate narrow
    ("CTRL_0spread", True, 0.0, (0.3, 0.4, 0.3)),  # control: zero spread -> must match DET
]


def run_experiment():
    cfg = model.default_cfg()

    rng = np.random.default_rng(SEED)
    # Pre-sample all true PV paths so every rule sees the same weather
    pv_days = [model.sample_pv(cfg, rng, aft_sd=AFT_SD) for _ in range(N_DAYS)]

    print(f"Discretisation rule comparison: {N_DAYS} days, aft_sd={AFT_SD}, seed={SEED}")
    print("=" * 65)

    results = {}
    t_start = time.perf_counter()

    for label, stochastic, spread_mult, weights in RULES:
        costs = []
        dones = []
        for day_pv in pv_days:
            r = model.run_closedloop(
                cfg, day_pv,
                stochastic=stochastic,
                bias=0.0,
                aft_sd=AFT_SD,
                cvar=None,
                weights=weights,
                spread_mult=spread_mult,
            )
            if r is None:
                print(f"  WARNING: infeasible day skipped for {label}")
                continue
            costs.append(r["cost"])
            dones.append(r["done"])

        costs = np.array(costs)
        mean_cost = float(np.mean(costs))
        p95_cost  = float(np.percentile(costs, 95))
        p5_cost   = float(np.percentile(costs, 5))
        std_cost  = float(np.std(costs))
        mean_done = float(np.mean(dones))

        results[label] = dict(
            mean=mean_cost, p95=p95_cost, p5=p5_cost,
            std=std_cost, mean_done=mean_done, n=len(costs),
            spread_mult=spread_mult, stochastic=stochastic,
        )
        print(f"{label:15s}  mean={mean_cost:+.4f}  p95={p95_cost:+.4f}  "
              f"p5={p5_cost:+.4f}  std={std_cost:.4f}  done={mean_done:.2f}  "
              f"(n={len(costs)}, smult={spread_mult})")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # ----------------------------------------------------------------
    # CONTROL: zero-spread stochastic must equal deterministic baseline
    # ----------------------------------------------------------------
    det_mean  = results["DET"]["mean"]
    ctrl_mean = results["CTRL_0spread"]["mean"]
    tol = 1e-4
    control_passed = abs(det_mean - ctrl_mean) < tol
    control_detail = (
        f"CTRL_0spread mean={ctrl_mean:+.6f}  DET mean={det_mean:+.6f}  "
        f"diff={abs(det_mean - ctrl_mean):.2e}  tol={tol:.0e}  "
        f"PASS={control_passed}"
    )
    print()
    print("CONTROL CHECK:")
    print(f"  {control_detail}")
    if not control_passed:
        print("  *** CONTROL FAILED – formulation bug suspected ***")
        sys.exit(2)
    else:
        print("  Control PASSED.")

    # ----------------------------------------------------------------
    # Summary table (excluding internal control run)
    # ----------------------------------------------------------------
    report_rules = ["DET", "A", "B", "C1", "C2"]
    print()
    print("Summary (lower cost = better):")
    print(f"{'Rule':<8} {'spread_mult':>11} {'mean':>9} {'p95':>9} {'p5':>9} {'std':>8} {'done':>6}")
    print("-" * 65)
    for lbl in report_rules:
        r = results[lbl]
        print(f"{lbl:<8} {r['spread_mult']:>11.3f} {r['mean']:>+9.4f} {r['p95']:>+9.4f} "
              f"{r['p5']:>+9.4f} {r['std']:>8.4f} {r['mean_done']:>6.2f}")

    # Best by mean
    best_mean = min(report_rules, key=lambda k: results[k]["mean"])
    best_p95  = min(report_rules, key=lambda k: results[k]["p95"])
    print(f"\nBest mean cost: {best_mean}  |  Best (lowest) p95: {best_p95}")

    return results, control_passed, control_detail, best_mean, best_p95, elapsed


def build_results_text(results, best_mean, best_p95):
    lines = ["Rule       spread_mult   mean      p95       p5        std     done"]
    for lbl in ["DET", "A", "B", "C1", "C2"]:
        r = results[lbl]
        lines.append(
            f"{lbl:<9}  {r['spread_mult']:>9.3f}  {r['mean']:>+8.4f}  {r['p95']:>+8.4f}  "
            f"{r['p5']:>+8.4f}  {r['std']:>7.4f}  {r['mean_done']:>5.2f}"
        )
    lines.append(f"Best mean: {best_mean}  |  Best p95: {best_p95}")
    return "\n".join(lines)


if __name__ == "__main__":
    results, control_passed, control_detail, best_mean, best_p95, elapsed = run_experiment()

    results_text = build_results_text(results, best_mean, best_p95)

    # Build structured output
    structured = dict(
        experiment="discretisation_rule",
        control_passed=control_passed,
        control_detail=control_detail,
        results=results_text,
        takeaways=[
            f"Best realized mean cost: rule {best_mean} "
            f"(mean={results[best_mean]['mean']:+.4f} AUD/day).",
            f"Best tail (lowest p95): rule {best_p95} "
            f"(p95={results[best_p95]['p95']:+.4f} AUD/day).",
            "Sensitivity across A/B/C1/C2 spread_mult range: "
            f"mean range {max(results[r]['mean'] for r in ['A','B','C1','C2']) - min(results[r]['mean'] for r in ['A','B','C1','C2']):.4f} AUD/day "
            f"({100*(max(results[r]['mean'] for r in ['A','B','C1','C2']) - min(results[r]['mean'] for r in ['A','B','C1','C2'])) / abs(results['DET']['mean']):.1f}% of DET).",
            "Control: zero-spread stochastic matched deterministic baseline -- LP is self-consistent.",
            f"Runtime: {elapsed:.1f}s for {N_DAYS} days x {len(RULES)} rules.",
        ],
        caveats=[
            "Stylised 24-slot model; absolute AUD values are not real tariff costs.",
            f"Only {N_DAYS} days -- p95 estimate has moderate sampling uncertainty.",
            "aft_sd=0.3 represents moderate afternoon uncertainty; results may differ for other regimes.",
            "All rules use the same true PV paths (same seed), so comparisons are paired -- noise cancels.",
            "Pearson-Tukey spread_mult=1.283 approximates P5/P95 under Gaussian; the PV forecast "
            "distribution here is not exactly Gaussian so the rule is approximate.",
        ],
        driver_path="wf_discretisation_rule.py",
        output_path="workflow-results/discretisation_rule.out.txt",
        raw_results={k: v for k, v in results.items()},
    )

    out_path = os.path.join(
        os.path.dirname(__file__),
        "workflow-results", "discretisation_rule.result.json"
    )
    with open(out_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"\nResult written to {out_path}")
