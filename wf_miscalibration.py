#!/usr/bin/env python3
"""
Miscalibration robustness experiment — EMHASS #841 Phase 2.

Tests: closed-loop Phase 2 (stochastic=True) with spread_mult in {0.5, 0.75, 1.0, 1.5, 2.0}.
spread_mult scales the forecast bands the CONTROLLER plans with; the true PV realisation is fixed.

Control: spread_mult=~0 (very narrow bands) should approach the deterministic (stochastic=False)
result because the three paths collapse to near-identical scenarios — the stochastic controller
degenerates to the naive P50 controller. We use spread_mult=0.01 as "near-zero" and assert the
cost difference vs deterministic is < 5% of the deterministic cost.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, ".")
from model import default_cfg, sample_pv, run_closedloop

SEED = 42
N_DAYS = 80
AFT_SD = 0.3
SPREAD_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]
WEIGHTS = (0.3, 0.4, 0.3)

print(f"=== Miscalibration Robustness Experiment ===")
print(f"Days: {N_DAYS}, aft_sd={AFT_SD}, seed={SEED}")
print(f"Spread multipliers: {SPREAD_MULTS}")
print()

cfg = default_cfg()
rng = np.random.default_rng(SEED)

# Pre-sample all PV true realisations so every spread_mult sees the same days.
pv_days = [sample_pv(cfg, rng, AFT_SD) for _ in range(N_DAYS)]

# -----------------------------------------------------------------------
# CONTROL: near-zero spread -> stochastic must converge to deterministic.
# We run both on the same days and assert they are within 5% of each other.
# -----------------------------------------------------------------------
print("=== CONTROL: near-zero spread_mult (0.01) should approach deterministic ===")
t_ctrl = time.perf_counter()

det_costs = []
zero_costs = []
for pv_true in pv_days:
    rd = run_closedloop(cfg, pv_true, stochastic=False, bias=0.0, aft_sd=AFT_SD, weights=WEIGHTS)
    rz = run_closedloop(cfg, pv_true, stochastic=True,  bias=0.0, aft_sd=AFT_SD, weights=WEIGHTS, spread_mult=0.01)
    if rd is None or rz is None:
        print("ERROR: LP infeasible in control run")
        sys.exit(1)
    det_costs.append(rd["cost"])
    zero_costs.append(rz["cost"])

det_mean = float(np.mean(det_costs))
zero_mean = float(np.mean(zero_costs))
rel_diff = abs(zero_mean - det_mean) / (abs(det_mean) + 1e-9)
ctrl_elapsed = time.perf_counter() - t_ctrl

print(f"  Deterministic mean cost:          {det_mean:.4f}")
print(f"  Near-zero spread mean cost:       {zero_mean:.4f}")
print(f"  Relative difference:              {rel_diff*100:.2f}%")
CONTROL_THRESHOLD = 0.05
control_passed = rel_diff < CONTROL_THRESHOLD
print(f"  Control PASSED (< {CONTROL_THRESHOLD*100:.0f}%): {control_passed}")
print(f"  Control elapsed: {ctrl_elapsed:.1f}s")
print()

if not control_passed:
    print("CONTROL FAILED — stopping experiment.")
    sys.exit(1)

# -----------------------------------------------------------------------
# MAIN EXPERIMENT: sweep spread_mult
# -----------------------------------------------------------------------
print("=== MAIN EXPERIMENT: sweep spread_mult ===")
print(f"{'spread_mult':>12}  {'mean_cost':>10}  {'p95_cost':>10}  {'n_failed':>8}  {'elapsed_s':>9}")

results = {}
for sm in SPREAD_MULTS:
    t0 = time.perf_counter()
    costs = []
    n_failed = 0
    for pv_true in pv_days:
        r = run_closedloop(cfg, pv_true, stochastic=True, bias=0.0, aft_sd=AFT_SD,
                           weights=WEIGHTS, spread_mult=sm)
        if r is None:
            n_failed += 1
        else:
            costs.append(r["cost"])
    elapsed = time.perf_counter() - t0
    mean_c = float(np.mean(costs)) if costs else float("nan")
    p95_c  = float(np.percentile(costs, 95)) if costs else float("nan")
    results[sm] = dict(spread_mult=sm, mean=mean_c, p95=p95_c, n_failed=n_failed,
                       n_ok=len(costs), elapsed=elapsed)
    print(f"  {sm:>12.2f}  {mean_c:>10.4f}  {p95_c:>10.4f}  {n_failed:>8d}  {elapsed:>9.1f}s")

print()
print("=== Deterministic baseline (stochastic=False, bias=0) ===")
print(f"  det mean: {det_mean:.4f}  (from control run, same {N_DAYS} days)")

print()
print("=== Relative cost vs nominal (spread_mult=1.0) ===")
nom_mean = results[1.0]["mean"]
nom_p95  = results[1.0]["p95"]
print(f"{'spread_mult':>12}  {'rel_mean%':>10}  {'rel_p95%':>10}")
for sm in SPREAD_MULTS:
    rm = (results[sm]["mean"] - nom_mean) / (abs(nom_mean) + 1e-9) * 100
    rp = (results[sm]["p95"]  - nom_p95)  / (abs(nom_p95)  + 1e-9) * 100
    print(f"  {sm:>12.2f}  {rm:>+10.2f}%  {rp:>+10.2f}%")

print()
print(f"Control passed: {control_passed}")
print("Done.")

# -----------------------------------------------------------------------
# Write structured result for resumability
# -----------------------------------------------------------------------
import os
out_dir = os.path.join(os.path.dirname(__file__), "workflow-results")
os.makedirs(out_dir, exist_ok=True)
result_path = os.path.join(out_dir, "miscalibration.result.json")

output = dict(
    experiment="miscalibration",
    seed=SEED,
    n_days=N_DAYS,
    aft_sd=AFT_SD,
    spread_mults=SPREAD_MULTS,
    weights=list(WEIGHTS),
    control=dict(
        description="near-zero spread_mult=0.01 must be within 5% of deterministic",
        det_mean=det_mean,
        zero_spread_mean=zero_mean,
        rel_diff_pct=float(rel_diff * 100),
        threshold_pct=float(CONTROL_THRESHOLD * 100),
        passed=control_passed,
    ),
    det_baseline_mean=det_mean,
    results={str(sm): results[sm] for sm in SPREAD_MULTS},
)

with open(result_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Result written to: {result_path}")
