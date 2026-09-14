#!/usr/bin/env python3
"""Run the full momentum study on the cached monthly panel and save results."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum import backtest as bt
from momentum import data, metrics, signals as sig, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results")
SECTORS = ["XLE", "XLF", "XLK", "XLP", "XLV", "XLY"]
ASOF = pd.Timestamp("2026-09-14")
COST_BPS = 10.0


def banner(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    # ---------------------------------------------------------------- data
    banner("1. DATA AUDIT")
    raw_panel = data.load_panel(RAW, SECTORS + ["SPY"])
    print(f"raw bars loaded          : {len(raw_panel)}  "
          f"({raw_panel.index[0].date()} -> {raw_panel.index[-1].date()})")

    panel = data.drop_incomplete_last_month(raw_panel, asof=ASOF)
    dropped = len(raw_panel) - len(panel)
    print(f"partial final month      : dropped {dropped} bar(s)  -> last bar "
          f"{panel.index[-1].date()}")

    panel = data.align_panel(panel)
    print(f"common sample            : {len(panel)} months, {panel.shape[1]} symbols")
    audit = data.audit(panel)
    print()
    print(audit.to_string())
    audit.to_csv(os.path.join(OUT, "data_audit.csv"))
    report["data"] = {
        "symbols": list(panel.columns),
        "months": len(panel),
        "start": str(panel.index[0].date()),
        "end": str(panel.index[-1].date()),
        "dropped_partial_bars": int(dropped),
    }

    sectors = panel[SECTORS]
    spy = panel[["SPY"]]

    # ----------------------------------------------------------- strategies
    banner("2. STRATEGY RUNS  (12-1 momentum, monthly rebalance, 10bp one-way)")
    base = bt.BacktestConfig(lookback=12, skip=1, top_q=1/3, bottom_q=1/3, cost_bps=COST_BPS)

    runs = {}
    runs["XS long/short"] = bt.cross_sectional_momentum(sectors, base)
    runs["XS long-only"] = bt.cross_sectional_momentum(
        sectors, bt.BacktestConfig(lookback=12, skip=1, top_q=1/3, long_only=True, cost_bps=COST_BPS)
    )
    runs["XS L/S vol-tgt 10%"] = bt.cross_sectional_momentum(
        sectors, bt.BacktestConfig(lookback=12, skip=1, cost_bps=COST_BPS,
                                   target_vol=0.10, max_leverage=2.0)
    )
    runs["TS absolute (L/S)"] = bt.timeseries_momentum(sectors, base)
    runs["TS absolute (long/flat)"] = bt.timeseries_momentum(
        sectors, bt.BacktestConfig(lookback=12, skip=1, long_only=True, cost_bps=COST_BPS)
    )
    runs["Equal-weight sectors"] = bt.buy_and_hold(sectors, "Equal-weight sectors")
    runs["SPY buy & hold"] = bt.buy_and_hold(spy, "SPY buy & hold")

    # Compare every strategy over the SAME window: the longest lookback
    # determines the first investable month, so benchmarks are truncated to
    # match. Otherwise the benchmark gets a free extra year of 2000-01 data.
    common_start = max(r.returns.index[0] for r in runs.values())
    common_end = min(r.returns.index[-1] for r in runs.values())
    print(f"common evaluation window : {common_start.date()} -> {common_end.date()}")
    series = {k: r.returns.loc[common_start:common_end] for k, r in runs.items()}

    table = metrics.summary_table(series)
    # metrics.summarise reports frequency-neutral column names ("bars",
    # "worst_bar_%") now that the library also runs daily studies. At monthly
    # frequency a bar is a month, so relabel for display only.
    display = table[["bars", "start", "end", "cagr_%", "vol_%", "sharpe",
                     "max_dd_%", "hit_rate_%", "worst_bar_%", "skew"]].round(3)
    display = display.rename(columns={"bars": "months", "worst_bar_%": "worst_month_%"})
    print()
    print(display.to_string())
    table.round(6).to_csv(os.path.join(OUT, "strategy_summary.csv"))
    pd.DataFrame(series).to_csv(os.path.join(OUT, "strategy_returns.csv"))
    report["summary"] = json.loads(table.round(6).to_json(orient="index"))

    print("\nturnover & cost drag (annualised):")
    for name, r in runs.items():
        t = r.turnover.loc[common_start:common_end]
        c = r.costs.loc[common_start:common_end]
        print(f"  {name:<24} turnover {12*t.mean():6.2f}x/yr   "
              f"cost drag {100*12*c.mean():5.2f}%/yr")
        report.setdefault("turnover", {})[name] = {
            "ann_turnover_x": round(float(12 * t.mean()), 4),
            "ann_cost_drag_pct": round(float(100 * 12 * c.mean()), 4),
        }

    # ------------------------------------------------------------ inference
    banner("3. STATISTICAL INFERENCE")
    ls = series["XS long/short"]
    bench = series["SPY buy & hold"]

    nw = stats.newey_west_tstat(ls, overlap=12)
    print(f"XS L/S mean monthly      : {100*nw['mean']:.4f}%")
    print(f"HAC (Newey-West) se      : {100*nw['hac_se']:.4f}%  "
          f"[{nw['lags_used']} lags, floored at overlap-1 = 11]")
    print(f"t-statistic              : {nw['t_stat']:.3f}   (n = {nw['n']})")
    report["hac"] = nw

    boot = stats.moving_block_bootstrap(ls, metrics.sharpe, n_boot=4000, seed=1)
    print(f"\nSharpe (point)           : {boot['point']:.3f}")
    print(f"Moving-block 95% CI      : [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]  "
          f"(block = {boot['block']} months)")
    print(f"P(Sharpe > 0)            : {boot['p_gt_zero']:.3f}")
    report["bootstrap_sharpe"] = boot

    excess = ls - bench
    nw_ex = stats.newey_west_tstat(excess, overlap=12)
    print(f"\nvs SPY: mean excess      : {100*12*nw_ex['mean']:.2f}%/yr, "
          f"t = {nw_ex['t_stat']:.3f}")
    report["hac_excess_vs_spy"] = nw_ex

    rets = sectors.pct_change().iloc[1:]
    mom = sig.compound_momentum(rets, 12, 1)
    reg = stats.predictive_regression(mom, rets, overlap=12)
    print(f"\nPredictive regression R_t+1 = a + b*M_t + e   (pooled, {reg['n_obs']} obs)")
    print(f"  alpha {100*reg['alpha']:.4f}%/mo (t={reg['alpha_t']:.2f})   "
          f"beta {reg['beta']:.5f} (t={reg['beta_t']:.2f})   R2 {reg['r_squared']:.5f}")
    report["predictive_regression"] = reg

    # ---------------------------------------------------------- bucket ladder
    banner("4. FORWARD RETURN BY PRIOR-MOMENTUM BUCKET (terciles; 6 sectors)")
    buckets = bt.decile_portfolios(sectors, lookback=12, skip=1, n_buckets=3)
    print(buckets[["mean_%", "ann_%", "count", "t_stat"]].round(3).to_string())
    buckets.round(6).to_csv(os.path.join(OUT, "momentum_buckets.csv"))
    report["buckets"] = json.loads(buckets.round(6).to_json(orient="index"))
    spread = buckets.loc[3, "mean"] - buckets.loc[1, "mean"]
    print(f"\nwinner - loser spread    : {100*spread:.3f}%/mo "
          f"({100*((1+spread)**12-1):.2f}%/yr)")
    report["bucket_spread_monthly_pct"] = round(100 * float(spread), 4)

    # ------------------------------------------------------------ robustness
    banner("5. PARAMETER GRID  (each cell is a separate test)")
    grid = []
    for lookback in (3, 6, 9, 12, 18, 24):
        for skip in (0, 1):
            for long_only in (False, True):
                cfg = bt.BacktestConfig(lookback=lookback, skip=skip,
                                        long_only=long_only, cost_bps=COST_BPS)
                try:
                    r = bt.cross_sectional_momentum(sectors, cfg).returns
                    r = r.loc[common_start:common_end]
                    nwg = stats.newey_west_tstat(r, overlap=lookback)
                    grid.append({
                        "lookback": lookback, "skip": skip,
                        "book": "long-only" if long_only else "long/short",
                        "cagr_%": round(100 * metrics.cagr(r), 2),
                        "sharpe": round(metrics.sharpe(r), 3),
                        "max_dd_%": round(100 * metrics.max_drawdown(r), 1),
                        "t_stat": round(nwg["t_stat"], 2),
                    })
                except Exception as exc:  # surfaced, never swallowed
                    grid.append({"lookback": lookback, "skip": skip,
                                 "book": "long-only" if long_only else "long/short",
                                 "error": str(exc)})
    grid_df = pd.DataFrame(grid)
    print(grid_df.to_string(index=False))
    grid_df.to_csv(os.path.join(OUT, "parameter_grid.csv"), index=False)

    n_trials = len(grid_df)
    crit = stats.deflated_threshold(n_trials)
    hits = int((grid_df["t_stat"].abs() > 1.96).sum())
    expected_by_chance = 0.05 * n_trials
    print(f"\ntests run                : {n_trials}")
    print(f"cells with |t| > 1.96    : {hits}   (chance alone predicts ~{expected_by_chance:.1f})")
    print(f"Bonferroni |t| threshold : {crit:.2f}   "
          f"cells clearing it: {int((grid_df['t_stat'].abs() > crit).sum())}")
    report["grid"] = {
        "n_trials": n_trials, "raw_hits": hits,
        "expected_by_chance": expected_by_chance,
        "bonferroni_t": round(crit, 3),
        "bonferroni_hits": int((grid_df["t_stat"].abs() > crit).sum()),
    }

    # ------------------------------------------------------------- subperiods
    banner("6. SUB-PERIOD STABILITY")
    splits = {
        "2001-2008 (pre-GFC)": ("2001-01-01", "2008-12-31"),
        "2009-2016 (recovery)": ("2009-01-01", "2016-12-31"),
        "2017-2026 (recent)": ("2017-01-01", "2026-12-31"),
    }
    sub_rows = []
    for label, (lo, hi) in splits.items():
        for name in ["XS long/short", "XS long-only", "TS absolute (long/flat)", "SPY buy & hold"]:
            window = series[name].loc[lo:hi]
            if len(window) < 24:
                continue
            sub_rows.append({
                "period": label, "strategy": name, "months": len(window),
                "cagr_%": round(100 * metrics.cagr(window), 2),
                "sharpe": round(metrics.sharpe(window), 3),
                "max_dd_%": round(100 * metrics.max_drawdown(window), 1),
            })
    sub = pd.DataFrame(sub_rows)
    print(sub.pivot(index="strategy", columns="period", values="sharpe").to_string())
    sub.to_csv(os.path.join(OUT, "subperiods.csv"), index=False)
    report["subperiods"] = sub_rows

    # ------------------------------------------------------------ crash risk
    banner("7. MOMENTUM CRASH EPISODES (the risk the poster flags)")
    dd = metrics.drawdown_series(ls)
    worst = metrics.worst_drawdown_window(ls)
    print(f"deepest drawdown         : {100*worst['depth']:.1f}%  "
          f"peak {worst['peak'].date()} -> trough {worst['trough'].date()} "
          f"({worst['months_to_trough']} months)")
    print(f"recovery                 : "
          f"{worst['recovery'].date() if worst['recovery'] is not None else 'not yet recovered'}")
    worst5 = ls.nsmallest(5)
    print("\nfive worst months for XS long/short:")
    for date, value in worst5.items():
        print(f"  {date.date()}   {100*value:7.2f}%   (SPY {100*bench.loc[date]:7.2f}%)")
    report["worst_drawdown"] = {
        "depth_pct": round(100 * worst["depth"], 2),
        "peak": str(worst["peak"].date()), "trough": str(worst["trough"].date()),
        "recovery": str(worst["recovery"].date()) if worst["recovery"] is not None else None,
        "months_to_trough": worst["months_to_trough"],
    }
    report["worst_months"] = {str(d.date()): round(100 * float(v), 3) for d, v in worst5.items()}

    # ------------------------------------------------- benchmark-relative tests
    banner("8. IS IT MOMENTUM, OR IS IT JUST BEING LONG EQUITIES?")
    # A long-only book of equities has a large positive mean by construction.
    # Testing its mean against ZERO therefore tests the equity risk premium,
    # not the momentum signal. The signal has to be tested against the same
    # assets held passively.
    pairs = [
        ("XS long-only", "Equal-weight sectors"),
        ("XS long/short", "SPY buy & hold"),
        ("TS absolute (long/flat)", "Equal-weight sectors"),
        ("TS absolute (long/flat)", "SPY buy & hold"),
    ]
    rel_rows = []
    for strat, base_name in pairs:
        diff = series[strat] - series[base_name]
        nw_rel = stats.newey_west_tstat(diff, overlap=12)
        row = {
            "strategy": strat, "benchmark": base_name,
            "excess_%/yr": round(100 * 12 * nw_rel["mean"], 2),
            "t_stat": round(nw_rel["t_stat"], 2),
            "sharpe_strat": round(metrics.sharpe(series[strat]), 3),
            "sharpe_bench": round(metrics.sharpe(series[base_name]), 3),
        }
        rel_rows.append(row)
        print(f"  {strat:<24} vs {base_name:<22} "
              f"{row['excess_%/yr']:+7.2f}%/yr   t = {row['t_stat']:+.2f}")
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(os.path.join(OUT, "benchmark_relative.csv"), index=False)
    report["benchmark_relative"] = rel_rows

    # Where does the trend filter actually earn its keep?
    ts = series["TS absolute (long/flat)"]
    ew = series["Equal-weight sectors"]
    worst_market = ew.nsmallest(20).index
    print(f"\n  in the 20 worst months for equal-weight sectors:")
    print(f"    equal-weight sectors  {100*ew.loc[worst_market].mean():+.2f}%/mo avg")
    print(f"    TS momentum long/flat {100*ts.loc[worst_market].mean():+.2f}%/mo avg")
    best_market = ew.nlargest(20).index
    print(f"  in the 20 best months:")
    print(f"    equal-weight sectors  {100*ew.loc[best_market].mean():+.2f}%/mo avg")
    print(f"    TS momentum long/flat {100*ts.loc[best_market].mean():+.2f}%/mo avg")
    report["crash_protection"] = {
        "worst20_bench_pct": round(100 * float(ew.loc[worst_market].mean()), 3),
        "worst20_ts_pct": round(100 * float(ts.loc[worst_market].mean()), 3),
        "best20_bench_pct": round(100 * float(ew.loc[best_market].mean()), 3),
        "best20_ts_pct": round(100 * float(ts.loc[best_market].mean()), 3),
    }

    exposure = runs["TS absolute (long/flat)"].weights.abs().sum(axis=1)
    exposure = exposure.loc[common_start:common_end]
    print(f"\n  TS long/flat average gross exposure: {100*exposure.mean():.1f}% "
          f"(fully invested {100*(exposure > 0.99).mean():.0f}% of months, "
          f"fully flat {100*(exposure < 0.01).mean():.0f}%)")
    report["ts_exposure"] = {
        "avg_gross_pct": round(100 * float(exposure.mean()), 2),
        "pct_months_fully_invested": round(100 * float((exposure > 0.99).mean()), 2),
        "pct_months_flat": round(100 * float((exposure < 0.01).mean()), 2),
    }

    # The decisive control. TS long/flat cuts the worst months roughly in half
    # -- but it cuts the best months by the same proportion. That is what
    # holding less equity does, not what timing does. So compare it against a
    # passive book held at its OWN average exposure: if the timing carries
    # information, it must beat the constant-exposure twin.
    avg_exposure = float(exposure.mean())
    twin = ew * avg_exposure
    twin_nw = stats.newey_west_tstat(ts - twin, overlap=12)
    up_capture = report["crash_protection"]["best20_ts_pct"] / report["crash_protection"]["best20_bench_pct"]
    down_capture = report["crash_protection"]["worst20_ts_pct"] / report["crash_protection"]["worst20_bench_pct"]
    print(f"\n  CONTROL: equal-weight sectors held at a constant {100*avg_exposure:.0f}% exposure")
    print(f"    constant-exposure twin  Sharpe {metrics.sharpe(twin):.3f}, "
          f"CAGR {100*metrics.cagr(twin):.2f}%, maxDD {100*metrics.max_drawdown(twin):.1f}%")
    print(f"    TS momentum long/flat   Sharpe {metrics.sharpe(ts):.3f}, "
          f"CAGR {100*metrics.cagr(ts):.2f}%, maxDD {100*metrics.max_drawdown(ts):.1f}%")
    print(f"    timing excess           {100*12*twin_nw['mean']:+.2f}%/yr, t = {twin_nw['t_stat']:+.2f}")
    print(f"    up-capture {100*up_capture:.1f}%  vs  down-capture {100*down_capture:.1f}%  "
          f"-> asymmetry {100*(down_capture-up_capture):+.1f}pp")
    report["constant_exposure_control"] = {
        "avg_exposure_pct": round(100 * avg_exposure, 2),
        "twin_sharpe": round(metrics.sharpe(twin), 4),
        "ts_sharpe": round(metrics.sharpe(ts), 4),
        "twin_cagr_pct": round(100 * metrics.cagr(twin), 3),
        "ts_cagr_pct": round(100 * metrics.cagr(ts), 3),
        "twin_maxdd_pct": round(100 * metrics.max_drawdown(twin), 2),
        "ts_maxdd_pct": round(100 * metrics.max_drawdown(ts), 2),
        "timing_excess_pct_yr": round(100 * 12 * twin_nw["mean"], 3),
        "timing_t": round(twin_nw["t_stat"], 3),
        "up_capture_pct": round(100 * up_capture, 2),
        "down_capture_pct": round(100 * down_capture, 2),
    }

    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/report.json and 6 CSVs")


if __name__ == "__main__":
    main()
