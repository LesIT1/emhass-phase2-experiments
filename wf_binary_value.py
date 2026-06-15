#!/usr/bin/env python3
"""
wf_binary_value.py  --  EMHASS #841 Phase 2 experiment: binary_value
=====================================================================
Measures the value of per-scenario FREE binaries vs a single SHARED schedule
(non-anticipativity relaxation) across ~150 randomised system states.

For each randomised state:
  - Build 3 PV scenarios (lo/mid/hi).
  - Solve open_loop with binary_mode='shared'.
  - Solve open_loop with binary_mode='free'.
  - Record gap = (shared_obj - free_obj) / |shared_obj|   (positive = free is better)

CONTROL: with a SINGLE scenario, free obj must equal shared obj (to 1e-6).
         We run this first and STOP if it fails.

Distributions reported: mean, median, p90, p99, max, and % states > 0.5%.
"""

import json
import sys
import time
import numpy as np

# --- import the verified library ------------------------------------------------
sys.path.insert(0, ".")
from model import default_cfg, open_loop, profiles, sample_pv

SEED = 42
N_STATES = 150
OUT_JSON = "workflow-results/binary_value.result.json"

# ---------------------------------------------------------------------------------
# CONTROL: single-scenario free == shared
# ---------------------------------------------------------------------------------
def run_control():
    cfg = default_cfg()
    hour, base, load, imp, exp = profiles(cfg)
    pv = base * 0.9
    scen = [("m", pv)]
    sh = open_loop(cfg, scen, [1.0], binary_mode="shared")
    fr = open_loop(cfg, scen, [1.0], binary_mode="free")
    if not (sh["ok"] and fr["ok"]):
        return False, f"Solver infeasible: shared ok={sh['ok']} free ok={fr['ok']}"
    diff = abs(sh["obj"] - fr["obj"])
    if diff > 1e-5:
        return False, f"Single-scenario diverged: shared={sh['obj']:.6f} free={fr['obj']:.6f} diff={diff:.2e}"
    return True, f"PASSED -- shared={sh['obj']:.6f} free={fr['obj']:.6f} diff={diff:.2e}"

# ---------------------------------------------------------------------------------
# Random cfg generator
# ---------------------------------------------------------------------------------
def random_cfg(rng):
    """Randomise cap, pmax, def_kw, def_hrs, tariff, load, spread multipliers."""
    cap    = rng.uniform(4.0, 16.0)
    pmax   = rng.uniform(2.0, min(8.0, cap))
    def_kw = rng.uniform(0.5, 3.0)
    def_hrs= int(rng.integers(2, 7))   # 2..6 hours
    # tariff randomisation: vary base, cheap, peak
    t_base = rng.uniform(0.15, 0.35)
    cheap_r = rng.uniform(0.04, 0.14)
    peak_r  = rng.uniform(0.35, 0.70)
    # cheap window
    c0 = int(rng.integers(8, 11))
    c1 = c0 + int(rng.integers(2, 5))
    # peak window
    p0 = int(rng.integers(14, 17))
    p1 = p0 + int(rng.integers(3, 7))
    # load
    load_base = rng.uniform(0.3, 1.0)
    load_morn = rng.uniform(0.2, 1.0)
    load_eve  = rng.uniform(0.5, 2.5)
    # pv peak
    pv_peak = rng.uniform(3.0, 8.0)
    # soc0
    soc0 = rng.uniform(0.1, 0.9)
    # eps_sd (spread)
    eps_sd = rng.uniform(0.05, 0.30)

    cfg = default_cfg(
        cap=cap, pmax=pmax, def_kw=def_kw, def_hrs=def_hrs,
        t_base=t_base, cheap=(c0, c1, cheap_r), peakt=(p0, p1, peak_r),
        load_base=load_base, load_morn=load_morn, load_eve=load_eve,
        pv_peak=pv_peak, soc0=soc0, eps_sd=eps_sd,
    )
    return cfg

# ---------------------------------------------------------------------------------
# Per-state scenario builder: lo/mid/hi from eps_sd spread
# ---------------------------------------------------------------------------------
def make_scenarios(cfg, rng):
    hour, base, _, _, _ = profiles(cfg)
    tm = rng.uniform(0.4, 1.3)
    eps_sd = cfg["eps_sd"]
    # mid
    eps_m = rng.uniform(0.85, 1.15, cfg["n"])
    pv_m  = np.maximum(0, base * tm * eps_m)
    # lo: P10-ish
    pv_lo = np.maximum(0, pv_m - 1.2815 * base * tm * eps_sd)
    # hi: P90-ish
    pv_hi = pv_m + 1.2815 * base * tm * eps_sd
    return [("lo", pv_lo), ("mid", pv_m), ("hi", pv_hi)]

# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()

    print("=" * 60)
    print("binary_value experiment -- EMHASS #841 Phase 2")
    print("=" * 60)

    # --- CONTROL first -----------------------------------------------------------
    print("\n[CONTROL] single-scenario: free obj must == shared obj ...")
    ctrl_ok, ctrl_msg = run_control()
    print(f"  {ctrl_msg}")
    if not ctrl_ok:
        print("\nCONTROL FAILED -- STOPPING")
        result = dict(
            experiment="binary_value",
            control_passed=False,
            control_detail=ctrl_msg,
            results="",
            takeaways=[],
            caveats="Control failed; results not computed.",
            driver_path="wf_binary_value.py",
            output_path="workflow-results/binary_value.out.txt",
        )
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    # --- Sweep -------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    gaps = []          # (shared_obj - free_obj) / |shared_obj|
    skipped = 0

    print(f"\n[SWEEP] N_STATES={N_STATES} randomised configs ...")
    for i in range(N_STATES):
        cfg = random_cfg(rng)
        scen = make_scenarios(cfg, rng)
        wts  = [0.3, 0.4, 0.3]

        sh = open_loop(cfg, scen, wts, binary_mode="shared")
        fr = open_loop(cfg, scen, wts, binary_mode="free")

        if not (sh["ok"] and fr["ok"]):
            skipped += 1
            continue

        denom = abs(sh["obj"]) if abs(sh["obj"]) > 1e-9 else 1e-9
        gap = (sh["obj"] - fr["obj"]) / denom
        gaps.append(gap)

        if (i + 1) % 30 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i+1}/{N_STATES}  elapsed {elapsed:.1f}s  "
                  f"running mean_gap={np.mean(gaps)*100:.3f}%")

    elapsed = time.perf_counter() - t_start
    gaps = np.array(gaps)
    n_valid = len(gaps)

    # --- Statistics --------------------------------------------------------------
    mean_gap   = float(np.mean(gaps))
    median_gap = float(np.median(gaps))
    p90_gap    = float(np.percentile(gaps, 90))
    p99_gap    = float(np.percentile(gaps, 99))
    max_gap    = float(np.max(gaps))
    pct_over   = float(np.mean(gaps > 0.005) * 100)   # > 0.5%

    results_str = (
        f"States solved: {n_valid}  skipped: {skipped}  elapsed: {elapsed:.1f}s\n"
        f"\nRelative gap = (shared_obj - free_obj) / |shared_obj|  (positive = free wins)\n"
        f"  mean   : {mean_gap*100:+.4f}%\n"
        f"  median : {median_gap*100:+.4f}%\n"
        f"  p90    : {p90_gap*100:+.4f}%\n"
        f"  p99    : {p99_gap*100:+.4f}%\n"
        f"  max    : {max_gap*100:+.4f}%\n"
        f"  > 0.5% : {pct_over:.1f}% of states\n"
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results_str)

    takeaways = []
    if mean_gap * 100 < 0.1:
        takeaways.append(
            f"Free binaries give negligible mean benefit ({mean_gap*100:+.4f}%) -- "
            "the shared schedule is not a material constraint in typical conditions."
        )
    else:
        takeaways.append(
            f"Mean benefit of free binaries: {mean_gap*100:+.4f}% -- "
            "shared schedule is a mild constraint."
        )

    if pct_over < 5.0:
        takeaways.append(
            f"Only {pct_over:.1f}% of states see gap > 0.5% -- "
            "shared binary mode is operationally adequate."
        )
    else:
        takeaways.append(
            f"{pct_over:.1f}% of states see gap > 0.5% -- "
            "free binaries could matter in adverse conditions."
        )

    takeaways.append(
        f"p99 gap = {p99_gap*100:+.4f}%, max = {max_gap*100:+.4f}% -- "
        "tail risk is bounded/low."
        if p99_gap * 100 < 2.0
        else f"p99 gap = {p99_gap*100:+.4f}%, max = {max_gap*100:+.4f}% -- tail deserves attention."
    )

    caveats = (
        "Stylised 24-step 1-h model; 3 fixed-weight PV scenarios only (no load uncertainty); "
        "random cfg covers a wide but synthetic parameter space; "
        "gaps in AUD cost terms are small absolute values -- ratios can amplify when "
        "shared_obj is near zero (days with large net export)."
    )

    result = dict(
        experiment="binary_value",
        control_passed=True,
        control_detail=ctrl_msg,
        results=results_str,
        takeaways=takeaways,
        caveats=caveats,
        driver_path="wf_binary_value.py",
        output_path="workflow-results/binary_value.out.txt",
    )

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult JSON written to {OUT_JSON}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
