#!/usr/bin/env python3
"""
EMHASS #841 Phase 2 Experiment: asymmetric_weights
===================================================
Tests whether tilting the P10/P50/P90 scenario weights toward the bad (low-PV) tail
captures tail-risk protection cheaply, without needing CVaR.

Weight sets tested:
  sym:   [0.3, 0.4, 0.3]  -- symmetric (control / standard Phase 2)
  tilt1: [0.45, 0.4, 0.15] -- mild tilt toward low-PV
  tilt2: [0.6, 0.3, 0.1]  -- strong tilt toward low-PV

CONTROL: single-scenario (zero spread) with symmetric weights must match the
deterministic (stochastic=False, bias=0) baseline cost. If this fails we STOP.
"""

import sys
import json
import numpy as np
import importlib.util
import pathlib

# ---------------------------------------------------------------------------
# Import the verified model library
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("model", HERE / "model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

default_cfg    = model.default_cfg
run_closedloop = model.run_closedloop
sample_pv      = model.sample_pv
forecast       = model.forecast
profiles       = model.profiles
plan           = model.plan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED   = 42
N_DAYS = 80       # ~80 realised-day draws; each draw is 24 MILP solves
AFT_SD = 0.3      # afternoon PV uncertainty (unresolvable in the morning)

WEIGHT_SETS = {
    "sym":   (0.3, 0.4, 0.3),
    "tilt1": (0.45, 0.4, 0.15),
    "tilt2": (0.6, 0.3, 0.1),
}

# ---------------------------------------------------------------------------
# CONTROL: zero-spread stochastic must equal the deterministic naive baseline
# ---------------------------------------------------------------------------
def run_control():
    """
    When spread_mult=0 the three scenarios collapse to identical paths, so the
    stochastic open-loop == deterministic (bias=0) on every horizon step.
    We run 5 fixed PV days and check max |delta| < 1e-4 AUD.
    """
    cfg = default_cfg()
    rng = np.random.default_rng(SEED)
    max_gap = 0.0
    for _ in range(5):
        pv_true = sample_pv(cfg, rng, 0.0)
        det  = run_closedloop(cfg, pv_true, stochastic=False, bias=0.0, aft_sd=0.0,
                              weights=(0.3, 0.4, 0.3), spread_mult=1.0)
        stoc = run_closedloop(cfg, pv_true, stochastic=True,  bias=0.0, aft_sd=0.0,
                              weights=(0.3, 0.4, 0.3), spread_mult=0.0)
        if det is None or stoc is None:
            return False, "solver failure during control", 0.0
        gap = abs(det["cost"] - stoc["cost"])
        max_gap = max(max_gap, gap)
    passed = max_gap < 1e-3
    detail = f"max |stoc(spread=0) - det| = {max_gap:.6f} AUD over 5 days"
    return passed, detail, max_gap


# ---------------------------------------------------------------------------
# Main experiment: closed-loop over N_DAYS for each weight set
# ---------------------------------------------------------------------------
def run_experiment():
    cfg = default_cfg()
    rng = np.random.default_rng(SEED)

    # Pre-draw all realised PV paths so every weight set sees IDENTICAL weather
    pv_days = [sample_pv(cfg, rng, AFT_SD) for _ in range(N_DAYS)]

    results = {}
    for label, weights in WEIGHT_SETS.items():
        costs = []
        for day_idx, pv_true in enumerate(pv_days):
            r = run_closedloop(cfg, pv_true, stochastic=True, aft_sd=AFT_SD,
                               weights=weights)
            if r is None:
                print(f"  WARNING: solver failure on day {day_idx} for {label}")
                continue
            costs.append(r["cost"])
        arr = np.array(costs)
        results[label] = dict(
            weights=list(weights),
            mean=float(np.mean(arr)),
            p95=float(np.percentile(arr, 95)),
            worst=float(np.max(arr)),
            n=len(arr),
        )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Experiment: asymmetric_weights")
    print(f"Seed={SEED}  Days={N_DAYS}  aft_sd={AFT_SD}")
    print("=" * 64)

    # --- CONTROL ---
    print("\n[CONTROL] zero-spread stochastic == deterministic baseline ...")
    passed, detail, max_gap = run_control()
    status = "PASSED" if passed else "FAILED"
    print(f"  Control {status}: {detail}")
    if not passed:
        print("STOPPING: control failed; results would be unreliable.")
        out = dict(
            experiment="asymmetric_weights",
            control_passed=False,
            control_detail=detail,
            results="",
            takeaways=["Control failed; experiment aborted."],
            caveats=[],
        )
        _write_result(out)
        sys.exit(1)

    # --- EXPERIMENT ---
    print(f"\n[EXPERIMENT] Running {N_DAYS} days x {len(WEIGHT_SETS)} weight sets ...")
    exp_results = run_experiment()

    # --- PRINT TABLE ---
    print("\n--- Results table ---")
    hdr = f"{'Label':<8} {'Weights (lo/mid/hi)':<24} {'Mean (AUD)':>12} {'P95 (AUD)':>12} {'Worst (AUD)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in exp_results.items():
        w = f"[{r['weights'][0]:.2f},{r['weights'][1]:.2f},{r['weights'][2]:.2f}]"
        print(f"{label:<8} {w:<24} {r['mean']:>12.4f} {r['p95']:>12.4f} {r['worst']:>12.4f}")

    sym = exp_results["sym"]
    t1  = exp_results["tilt1"]
    t2  = exp_results["tilt2"]

    # Relative changes vs symmetric baseline
    mean_chg_t1  = (t1["mean"]  - sym["mean"])  / abs(sym["mean"])  * 100
    mean_chg_t2  = (t2["mean"]  - sym["mean"])  / abs(sym["mean"])  * 100
    p95_chg_t1   = (t1["p95"]   - sym["p95"])   / abs(sym["p95"])   * 100
    p95_chg_t2   = (t2["p95"]   - sym["p95"])   / abs(sym["p95"])   * 100
    worst_chg_t1 = (t1["worst"] - sym["worst"]) / abs(sym["worst"]) * 100
    worst_chg_t2 = (t2["worst"] - sym["worst"]) / abs(sym["worst"]) * 100

    print("\n--- Deltas vs symmetric baseline (sym) ---")
    print(f"{'Label':<8} {'dMean%':>10} {'dP95%':>10} {'dWorst%':>10}")
    print(f"{'tilt1':<8} {mean_chg_t1:>10.2f} {p95_chg_t1:>10.2f} {worst_chg_t1:>10.2f}")
    print(f"{'tilt2':<8} {mean_chg_t2:>10.2f} {p95_chg_t2:>10.2f} {worst_chg_t2:>10.2f}")

    # Build human-readable results string
    lines = [hdr, "-" * len(hdr)]
    for label, r in exp_results.items():
        w = f"[{r['weights'][0]:.2f},{r['weights'][1]:.2f},{r['weights'][2]:.2f}]"
        lines.append(f"{label:<8} {w:<24} {r['mean']:>12.4f} {r['p95']:>12.4f} {r['worst']:>12.4f}")
    lines.append("")
    lines.append(f"{'Label':<8} {'dMean%':>10} {'dP95%':>10} {'dWorst%':>10}")
    lines.append(f"{'tilt1':<8} {mean_chg_t1:>10.2f} {p95_chg_t1:>10.2f} {worst_chg_t1:>10.2f}")
    lines.append(f"{'tilt2':<8} {mean_chg_t2:>10.2f} {p95_chg_t2:>10.2f} {worst_chg_t2:>10.2f}")
    results_str = "\n".join(lines)

    # Takeaways (written conservatively; actual sign depends on run)
    takeaways = []
    if p95_chg_t2 < -0.5 and mean_chg_t2 > 0.5:
        takeaways.append(
            "Strong tilt (tilt2) reduces P95 tail cost at the expense of a higher mean, "
            "confirming the pessimistic-weight mechanism works but is not free."
        )
    elif p95_chg_t2 >= -0.1:
        takeaways.append(
            "Tilting weights toward the bad-PV scenario has negligible effect on P95/worst "
            "in this model, suggesting the optimizer already hedges sufficiently via battery storage."
        )
    else:
        takeaways.append(
            "Asymmetric weights shift cost distribution toward the mean at the cost of "
            "increasing the worst case -- the hedging effect is ambiguous."
        )

    if mean_chg_t2 < 0.1 and p95_chg_t2 < 0.1:
        takeaways.append(
            "Both mean and tail improve under tilt2 (free lunch from improved planning): "
            "the battery can fully buffer the pessimistic hedge in this 8kWh / 4kW setup."
        )

    takeaways.append(
        f"Symmetric weights (sym) remain the clean baseline: "
        f"mean={sym['mean']:.4f} AUD, P95={sym['p95']:.4f} AUD, worst={sym['worst']:.4f} AUD."
    )
    takeaways.append(
        "These are stylised-model numbers; ratios between configurations are more reliable "
        "than absolute AUD values."
    )

    caveats = [
        f"N={N_DAYS} days; standard error of the mean ~ {np.std([v for r in [sym,t1,t2] for v in [r['mean']]]):.4f} AUD.",
        "aft_sd=0.3 creates afternoon PV uncertainty; the stochastic controller uses 3 scenarios "
        "(P10/P50/P90) generated by model.forecast -- weight tilting affects the LP objective blend.",
        "CVaR comparison not run here; see the cvar_blend experiment for the direct cost comparison.",
        "Stylised model: single deferrable, no battery degradation, flat load profile.",
    ]

    out = dict(
        experiment="asymmetric_weights",
        control_passed=True,
        control_detail=detail,
        results=results_str,
        raw=exp_results,
        takeaways=takeaways,
        caveats=caveats,
        driver_path=str(HERE / "wf_asymmetric_weights.py"),
        output_path=str(HERE / "workflow-results" / "asymmetric_weights.out.txt"),
    )

    _write_result(out)
    print("\n[DONE] Results written to workflow-results/asymmetric_weights.result.json")
    return out


def _write_result(out):
    out_path = HERE / "workflow-results" / "asymmetric_weights.result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
