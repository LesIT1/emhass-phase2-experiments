"""Offline backtest of the ACI bias recursion against the calibration log.

Replays bias <- clip(bias + gamma * (shortfall - alpha), 0, 1) over the logged
daily P10/P50 pairs, where the planning forecast is the EMHASS #961 blend
    forecast = bias * p10 + (1 - bias) * p50
and shortfall = 1 if the realised actual came in below the forecast that was
planned against that day (bias updated AFTER the day resolves, so day t is
planned with the bias produced by day t-1).

Usage: python aci_backtest.py --csv calibration_log.csv
"""

import argparse
import csv


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "label": r["label"],
                    "p10": float(r["p10"]),
                    "p50": float(r["p50"]),
                    "actual": float(r["actual"]),
                }
            )
    return rows


def run_aci(rows, alpha, gamma, bias0):
    bias = bias0
    trace = []
    for r in rows:
        forecast = bias * r["p10"] + (1.0 - bias) * r["p50"]
        shortfall = 1 if r["actual"] < forecast else 0
        trace.append(
            {
                "label": r["label"],
                "bias_used": bias,
                "forecast": forecast,
                "actual": r["actual"],
                "shortfall": shortfall,
            }
        )
        bias = min(1.0, max(0.0, bias + gamma * (shortfall - alpha)))
    return bias, trace


def static_curve(rows, grid):
    out = []
    for b in grid:
        n = sum(1 for r in rows if r["actual"] < b * r["p10"] + (1 - b) * r["p50"])
        out.append((b, n / len(rows)))
    return out


def summarize(trace, final_bias, tail=20):
    n = len(trace)
    rate_all = sum(t["shortfall"] for t in trace) / n
    tail_rows = trace[-tail:]
    rate_tail = sum(t["shortfall"] for t in tail_rows) / len(tail_rows)
    biases = [t["bias_used"] for t in trace]
    mean_tail_bias = sum(t["bias_used"] for t in tail_rows) / len(tail_rows)
    return {
        "final_bias": final_bias,
        "mean_bias_last": mean_tail_bias,
        "rate_all": rate_all,
        "rate_tail": rate_tail,
        "bias_min": min(biases),
        "bias_max": max(biases),
    }


def worst_stretch(rows, k=5):
    """Consecutive-k window where the forecast over-called hardest (most
    negative mean actual - p50). Not necessarily low-yield days in absolute
    terms; it is the worst over-forecast window."""
    best_i, best_v = 0, float("inf")
    for i in range(len(rows) - k + 1):
        v = sum(r["actual"] - r["p50"] for r in rows[i : i + k]) / k
        if v < best_v:
            best_i, best_v = i, v
    return best_i, best_i + k, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tail", type=int, default=20)
    args = ap.parse_args()
    rows = load_rows(args.csv)
    n = len(rows)
    print(f"rows: {n}  ({rows[0]['label']} .. {rows[-1]['label']})")

    print("\nSTATIC SHORTFALL CURVE (fixed bias -> realised shortfall rate)")
    for b, rate in static_curve(rows, [i / 10 for i in range(11)]):
        print(f"  bias={b:.1f}  rate={rate:.3f}")

    i0, i1, v = worst_stretch(rows)
    print(
        f"\nWORST OVER-FORECAST 5-DAY STRETCH: {rows[i0]['label']}..{rows[i1 - 1]['label']} "
        f"(mean actual-p50 = {v:+.2f} kWh)"
    )

    print("\nACI RUNS (once through the log, day-by-day)")
    header = (
        "alpha gamma bias0 | final_bias mean_bias_last{t} rate_all rate_last{t} "
        "bias_range".format(t=args.tail)
    )
    print(header)
    for alpha in (0.10, 0.25):
        for gamma in (0.02, 0.05, 0.10, 0.20):
            for bias0 in (0.0, 0.5):
                final_bias, trace = run_aci(rows, alpha, gamma, bias0)
                s = summarize(trace, final_bias, tail=args.tail)
                print(
                    f"  {alpha:.2f}  {gamma:.2f}  {bias0:.1f} |  "
                    f"{s['final_bias']:.3f}      {s['mean_bias_last']:.3f}      "
                    f"{s['rate_all']:.3f}    {s['rate_tail']:.3f}     "
                    f"[{s['bias_min']:.2f}, {s['bias_max']:.2f}]"
                )

    print("\nDAY-BY-DAY TRACE (alpha=0.10, gamma=0.10, bias0=0.0)")
    _, trace = run_aci(rows, 0.10, 0.10, 0.0)
    for t in trace:
        flag = " SHORTFALL" if t["shortfall"] else ""
        print(
            f"  {t['label']}  bias={t['bias_used']:.3f}  "
            f"fcst={t['forecast']:6.2f}  actual={t['actual']:6.2f}{flag}"
        )


if __name__ == "__main__":
    main()
