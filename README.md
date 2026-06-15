# EMHASS Phase 2 — stochastic-optimisation experiments

Supporting evidence for the discussion on **[EMHASS issue #841](https://github.com/davidusb-geek/emhass/issues/841)** ("Include the effect of forecast uncertainty"). It exists so that claims made in that thread can be **checked and reproduced**, rather than taken on trust.

> [!WARNING]
> **This is a small, stylised toy model — it is NOT EMHASS.** It is a single-household linear program with synthetic PV/load/tariff profiles, built only to compare the *shape* of stochastic vs deterministic planning. **Read the directions and ratios, not the absolute numbers.** The dollar figures are meaningless outside the toy; the value is in *which approach wins, by roughly how much, and why*.

## The question

EMHASS plans a home battery (and deferrable loads) against a PV forecast. "Phase 2" of #841 proposes replacing the single deterministic forecast with a **two-stage stochastic program** over a small set of PV scenarios (e.g. P10/P50/P90), with the t=0 decision shared across scenarios (non-anticipativity) and later steps free to adapt per scenario (recourse).

Is that worth building, versus a much cheaper **deterministic plan that just biases the forecast toward a conservative percentile**? These experiments try to answer that honestly.

## Headline findings

- **vs a naive P50 controller:** the stochastic plan is clearly better — roughly **16–18% lower mean and tail cost**. (Large, statistically solid.)
- **vs a *well-tuned* conservative deterministic plan (bias toward P10):** it is close. It **ties on the mean** and wins a **small, consistent tail edge (~1–3% at p95)**. That edge comes from the **recourse structure** (per-scenario dispatch adapting to which scenario unfolds), which no single-trajectory deterministic plan can replicate — even one with an adaptive bias.
- **The dominant lever is scenario band *calibration*, not the optimisation:** err **wide**, never narrow. Narrowing the bands hurts ~2–3× more than widening helps.
- **Keep it simple:** symmetric scenario weights (heavy-centre Pearson-Tukey weights *worsen* the tail); a **shared** deferrable schedule across scenarios (per-scenario binary freedom buys ~0.03%); **skip CVaR as a default** (pure CVaR over-hedges, and its `alpha` knob is structurally inert with only 3 scenarios).

**Verdict:** Phase 2 is worth building as an **opt-in robust / tail-risk mode**, not as the default planner. A tuned conservative deterministic bias captures most of the everyday value cheaply.

Full write-ups: **[FINDINGS-WORKFLOW.md](FINDINGS-WORKFLOW.md)** (the 9-experiment battery) and **[FINDINGS-ADDENDUM.md](FINDINGS-ADDENDUM.md)** (three follow-ups). [FINDINGS.md](FINDINGS.md) is the earlier exploratory lineage.

## The model (`model.py`)

A two-stage stochastic LP/MILP over PV scenarios: battery (charge/discharge/SoC), grid import/export, and a must-run deferrable; `binary_mode` in `shared` / `free` / `locked` / `fixed`; optional CVaR (Rockafellar–Uryasev) objective; optional per-scenario load. `python model.py` runs a **self-test that asserts four formulation controls** (single-scenario binary equivalence; identical-scenario shared==locked and free<=shared; CVaR λ=0 == expected cost; deferrable carryover). Every experiment driver builds on this verified core rather than re-deriving the LP.

## Experiments

Each `wf_*.py` driver includes a **control** that would catch a formulation bug (e.g. a zero-spread case that must equal the deterministic baseline). Results were independently re-checked (`workflow-results/<key>.verdict.json` are the adversarial verdicts, including corrections to several of our own first-pass claims).

| Driver | Question |
|---|---|
| `wf_presolve_scaling` | Does tying per-scenario binaries cost solve time? (presolve) |
| `wf_binary_value` | How much do free per-scenario deferrable binaries help? |
| `wf_closedloop_regimes` | Phase 2 vs tuned Phase 1 vs naive across regimes (closed loop) |
| `wf_cvar_frontier` | CVaR risk frontier (α × λ) |
| `wf_miscalibration` | Cost of mis-sized scenario bands (too narrow / too wide) |
| `wf_discretisation_rule` | Band width vs weighting scheme (deciles, Pearson-Tukey, blends) |
| `wf_asymmetric_weights` | Tail-tilted weights as a cheap CVaR substitute? |
| `wf_stability` | 300-day bootstrap CIs on the headline result |
| `wf_load_uncertainty` | Does adding load uncertainty widen the edge? |
| `wf_extbias` | Extend the Phase 1 bias grid (follow-up) |
| `wf_mixregime` | Adaptive tree vs a single fixed bias across mixed day-types (follow-up) |
| `wf_adaptbias` | Adaptive-bias deterministic plan vs Phase 2 (follow-up) |

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py            # self-test + all drivers; writes workflow-results/*.out.txt
# or, with uv:
uv run --with pulp --with numpy run_all.py
```

CBC is deterministic with the fixed seeds used here, so a faithful re-run reproduces the committed `workflow-results/` outputs; `git diff` after a run shows any divergence.

## Layout

```
model.py                  verified two-stage stochastic LP + self-test
wf_*.py                   experiment drivers (build on model.py)
run_all.py                reproduce everything
spike.py / sim.py /
  sim_closedloop.py       earlier exploratory scripts (lineage for FINDINGS.md)
FINDINGS-WORKFLOW.md      consolidated findings (9-experiment battery)
FINDINGS-ADDENDUM.md      three follow-up experiments
FINDINGS.md              earlier exploratory findings
workflow-results/         raw outputs (*.out.txt), structured results (*.result.json),
                          and the adversarial verdicts (*.verdict.json)
```

## Honest caveats

- Stylised single-household LP, synthetic AU-style tariffs, one deferrable, hourly steps, 3 fixed scenarios, deterministic prices, additive-Gaussian afternoon-only PV uncertainty (no skew, no real spatio-temporal correlation). **Not EMHASS.**
- Absolute figures are not deployable; `p95`/worst-day are single order statistics with real sampling variance at these day counts.
- Likely **over**-states Phase 2's edge vs a tuned deterministic plan (the bias grid and uncertainty model are idealised/symmetric); could **under**-state the value of real, skewed, correlated load+PV uncertainty.
- A couple of first-pass magnitudes were noise and are flagged as such in the verdicts and findings.

## Credit

The forecast-uncertainty idea on #841 is **@lutorm**'s; the two-phase framing is **@purcell-lab**'s; EMHASS is **@davidusb-geek**'s. This repo is just the supporting simulation work for that discussion. MIT licensed.
