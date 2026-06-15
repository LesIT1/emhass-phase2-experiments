# Phase 2 findings — ADDENDUM (post-workflow follow-ups, 2026-06-16)

Three follow-up experiments run after the 9-experiment workflow, targeting the synthesis's biggest
flagged caveat: the Phase 1 P10-bias grid was capped at 0.8 and pinned at the ceiling, so Phase 1
was likely understated and Phase 2's marginal edge overstated. All on the verified `model.py`; each
included a passing control (determinism diff = 0; adaptive k=0 == naive). Drivers: `wf_extbias.py`,
`wf_mixregime.py`, `wf_adaptbias.py`. Stylised toy LP — read ratios, not magnitudes.

## 1. Extended bias grid (wf_extbias, aft_sd=0.3, 80 days)
Phase 1 mean falls monotonically all the way to **bias=1.0** (plan-for-the-P10-worst-case):
0.6=1.668, 0.8=1.642, 0.9=1.632, **1.0=1.625**. At bias=1.0 the deterministic plan **ties Phase 2**
(P2 mean 1.626 = -0.09%; p95 +0.26%). So at *uniformly high* uncertainty, a maximally-conservative
fixed bias matches the stochastic tree. The workflow's 0.8 cap did overstate Phase 2's edge.

## 2. Mixed regimes (wf_mixregime, aft_sd drawn per day from {0,0.15,0.3,0.45}, 100 days)
With realistically *varying* day uncertainty, **no single fixed bias is best on both axes**: bias 1.0
wins mean (1.778), bias 0.8 wins p95 (3.076). **Phase 2 dominates both** (mean 1.772, p95 2.996):
+0.36% mean / **+3.05% p95** vs best-mean fixed; +0.99% / +2.60% vs best-p95 fixed. A static bias
either over-hedges calm days or under-hedges wild ones; Phase 2's bands adapt to each day.

## 3. Adaptive-bias deterministic vs Phase 2 (wf_adaptbias, mixed regimes, 80 days)
The real competitor: a deterministic plan with bias = clamp(k*u), u = forecast relative band width
(our Solcast risk-aware bias). Result: the adaptive-bias plans **did not beat a well-tuned fixed
bias** here (adapt_k1.5 mean 1.842 vs fix@1.0 1.804) — reducing hedging on lower-uncertainty days
gave up more than it saved under the asymmetric tariff (this model rewards conservatism even on
calmer days). **Phase 2 still beat every deterministic variant**: vs the best deterministic, mean
+0.45-1.1%, **p95 +1.1-1.2%**.

## Refined verdict
- Phase 2's advantage is the **recourse structure** (per-scenario continuous dispatch adapting to
  which scenario unfolds), NOT merely the amount of hedging — a single-trajectory deterministic
  plan cannot replicate it, whether its bias is fixed or adaptive.
- The advantage is **small but consistent**: ~0.5-1.1% mean and ~1.1-3% p95 over the best
  deterministic plan across mixed regimes, concentrated in the tail. This is a modest *upgrade* to
  the workflow's "ties tuned Phase 1" — Phase 2 does have a unique edge, but it is low single digits.
- **Practical call unchanged:** Phase 2 = opt-in robust/tail mode. A well-tuned conservative
  deterministic bias captures most of the mean benefit cheaply; the stochastic tree adds a small
  tail edge for users who want it.
- **Caveats:** stylised toy, ~1% magnitudes, single deferrable, symmetric Gaussian uncertainty; my
  adaptive-bias rule was uncalibrated (a better k-schedule, or bias tied to the actual realised
  spread from the calibration log, might do better). The recourse edge could grow with richer/
  skewed/correlated real uncertainty, or shrink under a better-tuned adaptive bias. NOT EMHASS.

## Suggested next (not yet run)
- Calibrate scenario bands AND the adaptive-bias k from the real Solcast forecast-vs-actual log
  (filling now) and re-run on realistic, possibly-skewed spreads.
- Test >=5 scenarios so the CVaR-alpha dimension is non-vacuous (workflow flagged it inert at 3).
