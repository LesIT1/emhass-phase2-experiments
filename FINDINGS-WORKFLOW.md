# EMHASS #841 Phase 2 — Stochastic Planning: Consolidated Findings

> [!IMPORTANT]
> **TL;DR.** Across 9 adversarially-verified experiments, **Phase 2 (multi-scenario stochastic MPC) is real but modest, and its value is overwhelmingly tail insurance, not everyday savings.** Against a *naive* P50 controller it cuts mean cost ~16-17% and the P95 tail ~18% on a stylised single-household model — but against a *bias-tuned* deterministic plan (Phase 1\*) it is essentially a wash on the mean (within noise) and buys only a low-single-digit P95 edge, and *only* when there is genuinely unresolvable afternoon-PV uncertainty plus battery headroom. The single highest-leverage knob is **scenario band calibration** (err *wide*, never narrow). CVaR, per-scenario binary freedom, asymmetric weights, and explicit load uncertainty all proved to be second-order or noise on this model. **Recommendation: ship a small, fixed 3-scenario tree with wide (decile-or-wider) symmetric bands as an opt-in "robust" mode; make scenario count/spread configurable; skip CVaR and per-scenario binaries by default; keep a tuned deterministic P10-bias as the cheap baseline.** All numeric magnitudes are from a toy LP and should be read as *directions and ratios*, not deployable dollar figures.

---

## 1. Headline verdict on Phase 2 — worth building, and as what?

**Build it, but as opt-in tail insurance, not as the default planner.**

- **Versus a naive single-P50 planner**, Phase 2's multi-scenario tree is clearly and significantly better: in the `stability` experiment (300 days, paired bootstrap) it beats naive by **+0.176 AUD/day mean (90% CI [+0.149, +0.205], excludes zero -> SOLID)** and cuts the P95 tail from 4.09 to 3.33 (**~18% lower**, P90 ~17% lower). `load_uncertainty` independently shows ~**16-17% mean** improvement vs naive. This gap is the largest, most robust effect in the whole battery.
- **Versus a *bias-tuned* deterministic plan (Phase 1\*)**, the case collapses to marginal. In `closedloop_regimes` Phase 2 is **+0.09% *worse* on mean cost on average** and only wins on the tail under irresolvable afternoon uncertainty (`aft_sd=0.3`: P90 ~ -0.35% avg, up to -1.8%); at `aft_sd=0` it is slightly worse on both mean and tail. The `stability` experiment confirms full 3-scenario recourse still beats the *scalar* P10-bias heuristic by +0.115/day (CI excludes zero) — so the recourse structure does add value the bias can't capture — **but** that comparison used an *untuned* Phase 1 (b=0.20), whereas `closedloop_regimes` let Phase 1\* tune its bias in-sample. The honest read: **the tree's edge over a *well-tuned* bias is small and tail-only.**
- **The mechanism is hedging, not cleverness.** Phase 2's only robust win is reducing bad-afternoon-PV-draw cost when (a) the uncertainty is genuinely unresolvable at decision time and (b) the battery has headroom to exploit the hedge. Where the battery trivially covers peak load (large-battery, `aft_sd=0`), Phase 2 and the baselines are bit-identical — the stochastic machinery is inert.

**Verdict:** Phase 2 is worth building as a **robust/tail-risk mode** for users with meaningful afternoon PV uncertainty and battery headroom. It is **not** justified as a general-purpose everyday cost-saver — a tuning-free biased deterministic plan reaches mean-cost parity. Frame it to users as "insurance against bad-forecast afternoons," priced in complexity accordingly.

---

## 2. What matters most, ranked (high-leverage levers)

1. **Scenario band calibration (width + direction) — by far the dominant lever.**
   `miscalibration` is decisive: narrowing bands (`spread_mult` < 1) is the costly mistake (mean +3.4%, P95 **+10.5%** vs nominal at 0.5x), while over-dispersing (>1) is cheap and slightly beneficial (mean -1.6 to -1.8%, P95 -2.3 to -6.4%). Mis-sizing to half-width erodes **~38% of Phase 2's edge** (8.1pp -> 5.0pp vs deterministic). The effect is self-limiting: cost saturates by `spread_mult~3` because the low scenario floors at PV=0. **Rule: when unsure, err wide.** `discretisation_rule` corroborates: at fixed weights, *wider* quadrature bands monotonically lower both mean and P95 (A < C2 < C1, all pairwise significant), and the true tail-optimal config was `spread_mult=1.283` with *standard* weights.

2. **Scenario *weighting* scheme — matters, and the obvious choice is wrong.**
   `discretisation_rule` decomposed the confounded "rule B" and found the **heavy-centre Pearson-Tukey weighting (0.185/0.63/0.185) is what produces a bad tail** (P95 3.81), not band width. Standard 0.3/0.4/0.3 weights with a wide band give the best tail (P95 3.135). **Keep symmetric, non-peaked weights.**

3. **Phase 1 bias tuning — the cheap competitor that sets the bar Phase 2 must clear.**
   A tuned P10-bias deterministic plan reaches mean-cost parity with the full tree (`closedloop_regimes`). Critically, **the optimal bias pinned at the top of the {0.4,0.6,0.8} grid (0.8) in every regime** — the true optimum may be higher, which would *strengthen* Phase 1\* and further erode Phase 2's marginal deltas. This lever is under-explored and should be extended upward before trusting Phase 2's relative magnitudes.

4. **CVaR blend (lambda) — minor, and the alpha dimension is structurally vacuous here.**
   `cvar_frontier`: pure CVaR (lambda=1) **over-hedges and is strictly dominated** (raises both mean +4.8% and P95; robust across seeds — avoid). A light blend (lambda~0.25) gives a mild, seed-robust mean/P95 reduction, but this is ordinary closed-loop realised-vs-planned divergence, "harmless-to-slightly-helpful," not a guaranteed free lunch. **The alpha sweep gives ZERO information**: with 3 scenarios and a 0.3-weight worst atom, CVaR_alpha collapses to the worst-scenario cost for every alpha in {0.7,0.8,0.9} *by construction* (verified analytically). Any alpha claim needs >=5 scenarios.

5. **Per-scenario binary freedom (non-anticipativity on deferrables) — negligible.**
   `binary_value`: free per-scenario deferrable binaries beat a single shared schedule by only **+0.0345% mean, with median and P90 both exactly 0%**. The relaxation bound caps how much it could ever help. `presolve_scaling`: a shared/tied schedule costs almost nothing extra in solve time (CBC presolve collapses the redundant binaries; locked/shared ~1.0x). **A shared binary schedule is operationally adequate** and simpler.

6. **Asymmetric (tail-tilted) scenario weights — within noise, not a CVaR substitute.**
   `asymmetric_weights`: tilting toward the low-PV tail leaves **P95 completely unmoved** and the ~0.69% mean "improvement" is ~1/8 of the true day-level SEM — **statistically indistinguishable from zero**. It only flips the t=0 decision on ~10/80 days and *hurts* on 2 of them. Use explicit CVaR for quantile-risk control, not weight bias.

7. **Explicit load uncertainty — smallest robust lever.**
   `load_uncertainty`: adding coupled load uncertainty on top of PV scenarios widens the edge by only **+0.20% of cost on the mean** (cheaper on just 28/60 days, ~coin flip), though the *sign* is robust across 5 seeds. The earlier tail-improvement claim did **not** survive verification (worst-day moves both ways across seeds). Read as "a small, robust reduction in *expected* cost," and note it scales directly with the *assumed* PV-load anti-correlation (k=0.6), which is unfitted.

---

## 3. Concrete recommendations for the EMHASS build

**Defaults (ship these):**
- **3-scenario tree** (low/P50/high PV), **symmetric weights 0.3/0.4/0.3**, **wide bands at decile-or-wider width** (`spread_mult ~ 1.0-1.3`). This captures essentially all the robust value at minimal complexity.
- **Shared (non-anticipative) deferrable binary schedule** across scenarios — the per-scenario freedom buys ~0.03% and adds variables; CBC presolves the tied version cheaply anyway.
- **Keep a tuned deterministic P10-bias plan as the standing baseline** (it reaches mean parity and is far cheaper to compute). Expose Phase 2 as an opt-in "robust mode."

**Make configurable:**
- **Scenario count** (3 default; allow >=5 for users who want meaningful CVaR-alpha behaviour).
- **`spread_mult` / band width** — the dominant lever; document clearly that *narrow is the dangerous direction* and the default should never go below decile width.
- **P10-bias factor** for the deterministic baseline — and **extend its search grid above 0.8** (the optimum pinned at the grid ceiling in every regime).
- **Optional light CVaR blend (lambda~0.25)** for tail-averse users, defaulted **off**.

**Keep simple / skip:**
- **Skip CVaR-alpha as a tuning knob** at the default 3-scenario tree — it is structurally inert. Only surface alpha if scenario count >=5.
- **Skip pure CVaR (lambda=1)** entirely — strictly dominated.
- **Skip asymmetric/tail-tilted weights** — effect is within noise and not a tail-risk tool.
- **Skip explicit per-scenario load-uncertainty modelling by default** — +0.20% mean, and it leans on an unfitted correlation assumption. Could be a future enhancement once real PV-load coupling is measured.
- **Skip per-scenario deferrable binaries** — negligible benefit, more variables.

---

## 4. Trust notes — what's solid, what's stylised, what was distrusted

**Verifier trust:** All 9 experiments were re-run/re-read by a skeptic and returned **trustworthy=true, confidence=high**. No result was distrusted; numbers reproduced bit-for-bit (CBC is deterministic, seed 42). The corrections below are to *interpretation/magnitude*, not to whether the underlying data is sound.

**Statistically SOLID (CI-supported, paired tests):**
- Phase 2 beats *naive* on mean (+0.176/day, CI [+0.149,+0.205]) and P95 (~18%) — `stability`, 300 days, correct paired bootstrap.
- Phase 2 beats *scalar* Phase 1 P10-bias (+0.115/day, CI excludes zero) — `stability`.
- Stochastic beats deterministic P50 regardless of discretisation rule (rule A vs DET -0.154/day, paired t=-4.71); at fixed weights wider bands monotonically lower mean+P95 (all pairwise significant) — `discretisation_rule`.
- Pure CVaR (lambda=1) over-hedges, robust across seeds 42/7/123 — `cvar_frontier`.
- Calibration asymmetry (narrow hurts ~2-3x more than wide helps) robust in direction — `miscalibration`.

**NOT statistically supported (claimed but within noise — do not rely on):**
- "Best rule = B" on mean cost: B-vs-A gap only -0.0186/day, t=-0.75 at n=80 — **noise** (`discretisation_rule`).
- Asymmetric-weight "0.69% free lunch": ~1/8 of the true day-level SEM — **indistinguishable from zero** (`asymmetric_weights`). The driver's own SEM was computed wrong (spread of 3 aggregate means, not a sampling error).
- `load_uncertainty` worst-day tail improvement: **mixed across seeds**, partly seed-specific; only the small mean edge is robust.
- Phase 2-vs-Phase 1\* magnitudes (`closedloop_regimes`): the optimal bias pinned at the grid ceiling (0.8), so Phase 1\* is likely *understated* and Phase 2's marginal `aft_sd=0` deltas could erode or flip. **Extend the bias grid before trusting magnitudes.**

**Structurally inert dimensions (give zero information by construction):**
- CVaR-**alpha** sweep at 3 scenarios — collapses to worst-scenario cost for all alpha (proven analytically). "Pick 0.8" is an unevidenced default here (`cvar_frontier`).

**Interpretation corrections the verifier flagged (data fine, story wrong):**
- `presolve_scaling`: the "free is only t=0 tied" mechanism wording is wrong (window 9-15 excludes t=0, so free binaries are fully untied); conclusion (less presolve leverage) still holds.
- `miscalibration`: prior "0.5x is 8.2% below deterministic" conflated cases — 0.5x is -5.0%, the -8.1% is the 1.0x case; mis-sizing erodes ~38% of the edge (more material than "barely dented").
- `discretisation_rule`: "wider bands caused B's bad tail" is **backwards** — wider bands *improve* the tail; B's heavy-centre *weights* cause it.
- `asymmetric_weights`: tilt1 != tilt2 exactly (differ on 2/80 days); they coincide only after 4-dp rounding.

**Stylisation caveats — where the toy model likely mis-states reality:**
- All experiments use a **stylised single-household 8kWh/4kW (some 12kWh) LP with synthetic AU tariffs**, fixed seed 42, **additive-Gaussian afternoon-only uncertainty (no skew, no spatial/temporal correlation), and no real load uncertainty** in most runs. **This is not EMHASS.**
- **Absolute AUD figures are not deployable** — trust rankings and ratios, not magnitudes. P95 is a single order statistic (n=80 carries ~+/-10% sampling variance on the tail).
- **Likely *over*-states Phase 2's relative value** vs a tuned deterministic plan, because the bias grid was capped at 0.8 (true Phase 1\* probably better) and the uncertainty model is idealised/symmetric (real forecast errors are skewed and correlated, which a 3-scenario symmetric tree captures poorly).
- **Likely *under*-states** the value of real load and PV-load coupling: `load_uncertainty`'s +0.20% scales directly with the assumed k=0.6 anti-correlation; with stronger real coupling (or correlated multi-deferrable loads, which EMHASS has and the toy lacks) the stochastic edge could grow.
- The **saturation behaviour** (cost plateaus past `spread_mult~3` because the low scenario floors at PV=0) is a model artefact of the PV=max(0,.) clip; in a richer model with non-PV uncertainty sources the "err wide" guidance still holds but the saturation point would differ.

---

*Source data: `workflow-results/<key>.{result,verdict}.json` for all 9 keys. Driver/model: `wf_<key>.py` + `model.py` in the parent `stochastic-spike/` dir. All verdicts trustworthy=true, confidence=high.*
