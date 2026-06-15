# Two-stage stochastic spike - findings (EMHASS #841 Phase 2)

Local de-risking spike, NOT EMHASS code. Stylised 1-day, half-hourly, 1 battery + 1 must-run
deferrable (3 kW x 3 h) in a 09-15 window, 3 PV scenarios (P10/P50/P90 = 0.4/1.0/1.4 x base),
asymmetric tariff (cheap import 09-12, peak 15-21, near-zero feed-in), PuLP + CBC.
Run: `uv run --with pulp --with numpy spike.py`.

Built to put numbers behind the claims made to @lutorm / @davidusb-geek before any PR.

## Finding 1 - presolve collapses equality-locked binaries (the "duplicate + lock is cheap" claim)

LOCKED (per-scenario binaries tied equal by constraints) has **3x the nominal binaries** of
SHARED (36 vs 12) but solves in the **same wall time (ratio ~1.0x)**. CBC presolve aggregates
the equality-locked variables before branching, so lutorm's "duplicate everything then lock the
first stage together" build pattern costs essentially nothing for the locked part. Confirmed.

## Finding 2 - per-scenario binary recourse is real but economically marginal here

FREE (per-scenario binaries, only t=0 tied) genuinely places the load differently per scenario:

```
P10 (low PV) : 09:30-11:30 + 13:00   -> cheap-import window 09-12
P50          : 11:00-13:30
P90 (high PV): 09:00-09:30 + 13:00-14:30 -> soaks the 13-15 PV peak
shared       : 09:30-11:30 + 13:00   (= the P10 placement)
```

So the model IS exploiting the freedom (not a bug). But the expected-cost gain over SHARED is
only **~0.02%**, and it stays ~0 across battery sizes (8 / 4 / 2 / 0.5 kWh):

```
cap_kWh   shared     free      gap     gap%
   8.0    3.1212    3.1206   +0.0006  +0.02%
   4.0    4.9756    4.9750   +0.0006  +0.01%
   2.0    5.9048    5.9047   +0.0001  +0.00%
   0.5    6.6391    6.6386   +0.0006  +0.01%
```

The continuous recourse (grid + battery, always free per scenario) absorbs almost all of the
PV spread, so forcing one shared binary schedule costs next to nothing. A near-zero battery does
not change this.

## What this means for the build

Backs exactly the plan agreed on the thread:
- Build the variables duplicated with t=0 non-anticipativity equalities (presolve makes the
  locked/shared case cheap).
- **Default to shared t>=1 binaries** (continuous recourse free) - captures ~all the value at
  flat integer cost.
- Expose **free per-scenario binaries behind a flag** so the gain can be measured on the real
  EMHASS model + logged data, where it may differ from this toy.

## Finding 3 - Monte Carlo sweep over 250 randomized states (sim.py)

`sim.py` randomizes day brightness, forecast spread (P10/P90 width), SoC, battery size, tariff
shape, load, and deferrable size/hours/window, and per state measures two things out-of-sample:

- **VSS** = EEV - RP_shared: value of choosing the deferrable schedule *stochastically* vs from a
  deterministic P50 plan (continuous grid/battery recourse adapts per scenario in both).
- **Binary gap** = RP_shared - RP_free: extra value of per-scenario binary recourse.

Results (250/250 solved):

```
VSS (value of stochastic scheduling, % of expected cost)
  mean +0.27%  median 0.00%  p90 +0.61%  p99 +3.75%  max +19.3%
  >1% of cost in 7.6% of states ; >0.1% in 18.4%
  (det schedule differed from the stochastic one in 52% of states)

Binary gap (free per-scenario vs shared schedule)
  mean +0.014%  median 0.00%  p90 +0.015%  p99 +0.26%  max +0.94%
  >0.5% in 0.8% of states

Solve time: free ~1.09x shared at this toy size (nbin ~14 shared vs ~42 free)

Where the stochastic value lives (mean VSS):
  wide forecast spread  +0.40%   vs narrow +0.15%
  (battery-size split is weak / tail-driven, do not over-read)
```

Read-out:
- **Phase 2 is a tail-risk tool, not an average-case saver.** On the median day it changes the
  schedule (52% of the time) but barely the cost; the value is in the right tail (p99 ~3.7%,
  max ~19%, >1% on ~8% of days), and it grows with forecast spread. That is exactly the
  "opt-in tool for advanced users" framing - it hedges the bad-forecast days.
- **Free per-scenario binaries are not worth it.** Across 250 randomized states the gap is
  <0.3% at p99 and never above ~1%. Shared schedule + continuous recourse captures essentially
  all the value. So: ship shared as the default; the free-binary flag is genuinely low priority
  (build it if cheap, do not block on it).

## Finding 4 - closed-loop receding-horizon (sim_closedloop.py): Phase 2 vs Phase 1

The honest test: run the actual MPC loop (observe state, forecast the rest of the day, plan,
commit only t=0, advance reality, re-plan), with forecast uncertainty that shrinks as the day
is observed (a day-level brightness inferred from observations + irreducible per-step cloud
noise). Realized cost is accrued from each committed t=0 action against the known current PV.
Three controllers face the SAME sampled days:
  - **naive** : deterministic on P50.
  - **Phase 1**: deterministic on a P10-biased blend `bias*P10 + (1-bias)*P50` (the merged knob).
  - **Phase 2**: stochastic scenarios around P50.

> Two methodology bugs were caught here, both by the spread=0 control (which MUST make the
> stochastic run identical to deterministic): (1) using a 1-scenario det model vs a 3-scenario
> stochastic model let CBC pick different alternate optima -> fixed by making all controllers
> the same 3-scenario structure + a tiny throughput regularizer; (2) collapsing the bands to
> zero also zeroed the P10-vs-P50 gap, so the Phase 1 blend silently equalled naive -> fixed by
> always using real bands and only collapsing the *scenario spread*. Trust the numbers only
> because the control now reads exactly 0.00% / 100% tie.

Results (40 sampled days, asymmetric tariff peak/cheap ~6x, PV brightness ~U[0.4,1.3]):

```
Phase 1 (blend) vs naive P50, by bias:   bias 0.2 +3.0% | 0.4 +5.0% | 0.6 +6.4%  (win ~82%)
Phase 2 (stochastic) vs naive P50:                                    +8.9%  (win 82%)
Phase 2 vs BEST-tuned Phase 1 (bias 0.6):  agg +2.6%  median +0.66%  win 62%  worst -2.1%
```

Read-out (sober, and the important one):
- Hedging the morning cheap-grid commitment against a possibly-cloudy afternoon is worth a lot
  here (~9% vs a naive P50 controller) because the tariff is steeply asymmetric.
- **A well-tuned Phase 1 blend captures most of that** - at bias 0.6 it gets +6.4% of the +8.9%.
  And more bias kept helping (monotonic 0.2->0.6), so a data-tuned blend may close more still.
- **Phase 2's marginal value over a tuned Phase 1 is real but modest**: ~2.6% aggregate, median
  +0.66%, winning only ~62% of days (it loses on the rest, worst -2.1%). It is a tail-hedging
  refinement, not a large win over Phase 1 done well.
- The single highest-leverage lever is therefore **tuning the Phase 1 bias from real
  forecast-vs-actual data** (the calibration log). Phase 2 then adds an opt-in increment for
  users who want the extra tail protection.

## Finding 5 - where Phase 2's value lives, and how to make it bigger

Sweeping `aft_sd` = irreducible afternoon PV uncertainty that the morning's observations cannot
resolve (an afternoon cloud front), so the morning cheap-grid commitment is made under
uncertainty that re-planning cannot remove:

```
aft_sd   P1* vs naive   P2 vs naive   P2 vs P1*(agg)   P2 vs P1*(median)   win   best_bias
 0.00       +6.99%        +8.43%          +1.55%             +0.01%         57%     0.8
 0.15       +9.20%        +9.69%          +0.54%             +0.00%         53%     0.8
 0.30      +10.33%       +12.15%          +2.03%             -0.00%         47%     0.8
 0.45      +14.35%       +18.42%          +4.75%             +0.00%         60%     0.8
```
(P1* = best-tuned Phase 1 blend in that regime; the tuner always wanted the heaviest bias, 0.8.)

Two things, both important:
- **Phase 2's edge over a tuned Phase 1 grows when uncertainty is unresolvable at commit time**
  (+4.75% agg at aft_sd 0.45 vs +1.55% when it all resolves). My first model under-sold Phase 2
  by making the uncertainty too easy to learn away. The realistic case for us - afternoon cloud
  vs a morning grid-charge decision - is exactly the unresolvable kind, so Phase 2 is worth more
  than the early runs implied.
- **But it is tail insurance, not an everyday saver**: in every regime the MEDIAN Phase-2-over-
  Phase-1 gain is ~0 and it wins only ~50-60% of days. The aggregate edge is carried by a
  minority of bad-forecast days. So Phase 2 protects the tail; the typical day it ties a tuned
  blend.

### How to get MORE from Phase 2 (levers, largest first)
1. **Risk-aware objective (CVaR / asymmetric tail weights), not symmetric expected cost.**
   Phase 2's whole point is the tail; optimising the tail (David's CVaR option) instead of the
   mean is the biggest untested lever and should widen the gap. NOT yet tested here.
2. **Calibrate scenarios to the unresolvable component** (afternoon-specific spread, not a day
   scalar) from logged forecast-vs-actual - that is the spread Phase 2 actually monetises.
3. **Add load (and later price) uncertainty + PV/load correlation** - more unresolvable
   dimensions a single biased point cannot represent.
4. **Judge it on tail metrics** (p95 cost, unmet must-run-load avoidance), not mean cost - that
   is the axis it is built for.
The cheap high-ROI baseline win remains: **tune the Phase 1 bias from data** (P1* at bias 0.8
captured most of the value in every regime).

## Finding 6 - CVaR (risk-aware objective): a small knob, and Phase 2's edge IS in the tail

Tested David's CVaR option (Rockafellar-Uryasev, LP form) at aft_sd=0.30, 50 days, sweeping the
risk weight lambda (lam=0 = expected cost [control], lam=1 = pure worst-case), alpha=0.8:

```
controller        mean      p90      p95    worst
naive P50        1.777    3.179    3.698    4.397
P1@0.8 (blend)   1.608    2.931    3.166    3.508
P2 expected      1.581    2.884    3.056    3.272
P2 CVaR .25-.75  1.576    2.837    3.056    3.272
P2 CVaR 1.0      1.693    2.929    3.130    3.300
```
(Control passed: lam=0 == expected-cost Phase 2 exactly, mean diff 0.000000.)

- **A moderate CVaR weight (lam 0.25-0.75) is a small free bonus** over plain expected cost
  (mean +0.3%, p90 a touch better, p95/worst unchanged), with no downside. Worth exposing as a
  knob, but it is not the big lever.
- **Pure CVaR (lam=1) backfires** - over-hedges every step of the receding horizon (plans each
  morning for the cloudy P10), raising mean ~7% and not improving the tail. Myopic worst-case at
  every replan compounds into chronic conservatism. So expose CVaR but default it low / off.
- **The clearest result: Phase 2's advantage over a well-tuned Phase 1 blend is a TAIL effect.**
  vs P1@0.8: mean only ~1.7% better, but **p95 ~3.5% and worst-day ~6.7% better** (3.27 vs 3.51).
  This sharpens the earlier "median ~0" framing: Phase 2 ties the tuned blend on a typical day
  and pays off on the bad-forecast days - genuine tail insurance, now quantified.

Net on "can we get more from Phase 2": yes, a little, via a small CVaR weight; but the real value
is already in the expected-cost stochastic model's tail behaviour. The big levers remain scenario
calibration from data and (cheap) Phase 1 bias tuning. (Worst-day over 50 days is one realization,
so treat p90/p95 as the stabler tail signal.)

## Caveats

Stylised: synthetic profiles, simple battery, one deferrable, hourly, 3 fixed scenarios,
deterministic prices, day-level + per-step PV uncertainty model that is not calibrated to a real
site. The ABSOLUTE percentages are inflated by the extreme tariff asymmetry, wide PV spread, and
a small battery with a no-margin naive baseline; a 32 kWh battery and a gentler spread would
shrink them. Treat the RATIOS (Phase 1 captures most of the hedge; Phase 2 adds a modest, mostly-
tail increment; bias tuning is high-leverage) as the takeaways, not the raw numbers. Real-model +
logged-data benchmarking is still the final word.
