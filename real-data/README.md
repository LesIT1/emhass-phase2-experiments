# Real-data calibration log and ACI backtest

Real forecast-vs-actual calibration data from a single Perth, Australia site
(three-array rooftop PV), logged daily since 2026-06-16, plus an offline
backtest of the adaptive-bias (ACI) recursion discussed in
[emhass#841](https://github.com/davidusb-geek/emhass/issues/841).

## Files

- `calibration_log.csv` - one row per day: `label` (date), `p10`, `p50`, `p90`
  (whole-day Solcast percentile totals in kWh, site correction applied, as
  planned against at ~07:00 local), `actual` (realised whole-day PV in kWh).
- `aci_backtest.py` - replays `bias <- clip(bias + gamma*(shortfall - alpha), 0, 1)`
  over the log, where the planning forecast is the emhass #961 blend
  `bias*P10 + (1-bias)*P50` and shortfall is scored against the forecast
  planned that day (bias updates after the day resolves). Prints the static
  shortfall-rate curve, a parameter grid (alpha x gamma x start), and a
  day-by-day trace. Pure stdlib.

## Run

    python aci_backtest.py --csv calibration_log.csv

## Caveats

Single site, southern-winter conditions, whole-day totals (one update per
day, so gamma values are per-day), n grows daily. The percentiles include a
site correction factor, so the log calibrates the forecast as consumed by the
optimizer, not raw Solcast.
