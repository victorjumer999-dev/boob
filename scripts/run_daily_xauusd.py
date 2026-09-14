#!/usr/bin/env python3
"""Daily time-series momentum on XAUUSD, 5 years.

One instrument, so the cross-sectional strategy from the monthly study is not
merely weak here -- it is undefined. Only absolute (time-series) momentum
applies, and the benchmark that matters is buy & hold.
"""

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
OUT = os.path.join(ROOT, "results_daily")
CSV = os.path.join(ROOT, "data", "raw_daily", "XAUUSD.csv")

BPY = data.BARS_PER_YEAR["daily"]   # 261 weekday bars/yr
COST_BPS = 2.0                      # one-way, spot gold; swept below
LOOKBACK = 252                      # ~12 months of weekday bars
SKIP = 21                           # ~1 month, the 12-1 convention


def banner(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    # ------------------------------------------------------------- data
    banner("1. DATA AUDIT  (XAUUSD spot, daily)")
    frame = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()
    prices = frame[["close"]].rename(columns={"close": "XAUUSD"})

    observed = data.infer_bars_per_year(prices.index)
    print(f"bars                     : {len(prices)}  "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")
    print(f"observed bars/year       : {observed:.2f}   (using {BPY} for annualisation)")
    print(f"weekend bars present     : {int((prices.index.dayofweek >= 5).sum())}")
    if abs(observed - BPY) > 12:
        raise SystemExit(
            f"frequency mismatch: series runs {observed:.0f} bars/yr but the "
            f"annualisation constant is {BPY}. Refusing to report scaled statistics."
        )

    rets = prices.pct_change().iloc[1:]
    print(f"daily sd                 : {float(rets['XAUUSD'].std()):.5f}  "
          f"-> ann vol {100 * metrics.annual_vol(rets['XAUUSD'], BPY):.2f}%")
    print(f"worst / best day         : {100 * float(rets.min().iloc[0]):.2f}% / "
          f"{100 * float(rets.max().iloc[0]):.2f}%")
    print(f"price {float(prices.iloc[0, 0]):.2f} -> {float(prices.iloc[-1, 0]):.2f}  "
          f"({100 * (float(prices.iloc[-1, 0]) / float(prices.iloc[0, 0]) - 1):+.1f}%)")
    report["data"] = {
        "bars": len(prices), "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()), "observed_bars_per_year": round(observed, 2),
        "bars_per_year_used": BPY,
        "ann_vol_pct": round(100 * metrics.annual_vol(rets["XAUUSD"], BPY), 2),
    }

    # ------------------------------------ cross-sectional is undefined here
    banner("2. WHY THERE IS NO CROSS-SECTIONAL RESULT")
    try:
        sig.quantile_weights(sig.compound_momentum(rets, LOOKBACK, SKIP))
        print("  unexpected: ranking succeeded on one asset")
    except Exception as exc:
        print(f"  cross-sectional ranking on 1 asset -> {type(exc).__name__}: {exc}")
        print("  Ranking needs a cross-section. With a single instrument the")
        print("  'long winners / short losers' strategy has no meaning, so every")
        print("  result below is time-series (absolute) momentum.")
    report["cross_sectional"] = "undefined for a single instrument"

    # ------------------------------------------------------------ strategies
    banner(f"3. STRATEGIES  ({LOOKBACK}-{SKIP} daily momentum, {COST_BPS}bp one-way)")
    base = dict(lookback=LOOKBACK, skip=SKIP, cost_bps=COST_BPS, periods_per_year=BPY)

    runs = {}
    runs["TS long/short"] = bt.timeseries_momentum(prices, bt.BacktestConfig(**base))
    runs["TS long/flat"] = bt.timeseries_momentum(
        prices, bt.BacktestConfig(**base, long_only=True))
    runs["TS long/flat, 15% vol"] = bt.timeseries_momentum(
        prices, bt.BacktestConfig(**base, long_only=True, target_vol=0.15,
                                  vol_window=63, max_leverage=2.0))
    runs["Buy & hold"] = bt.buy_and_hold(prices, "Buy & hold")

    # Price vs SMA -- the poster's other momentum formula.
    sma_sig = sig.price_to_sma(prices, lookback=LOOKBACK)
    # price_to_sma is indexed on prices (N bars); returns have N-1. Reindex
    # onto the returns index rather than trusting the shapes to line up.
    sma_dir = sig.timeseries_signal(sma_sig, allow_short=False).reindex(rets.index)
    runs["Price > SMA(252)"] = bt.run_weighted_backtest(
        sma_dir.fillna(0.0), rets, cost_bps=COST_BPS, label="Price > SMA(252)",
        config=bt.BacktestConfig(**base, long_only=True))

    start = max(r.returns.index[0] for r in runs.values())
    end = min(r.returns.index[-1] for r in runs.values())
    series = {k: r.returns.loc[start:end] for k, r in runs.items()}
    print(f"common window            : {start.date()} -> {end.date()}  "
          f"({len(series['Buy & hold'])} bars, {len(series['Buy & hold']) / BPY:.2f} yrs)")

    table = metrics.summary_table(series, periods_per_year=BPY)
    show = table[["bars", "years", "cagr_%", "vol_%", "sharpe", "max_dd_%",
                  "hit_rate_%", "worst_bar_%", "skew"]].round(3)
    print(); print(show.to_string())
    table.round(6).to_csv(os.path.join(OUT, "daily_summary.csv"))
    pd.DataFrame(series).to_csv(os.path.join(OUT, "daily_returns.csv"))
    report["summary"] = json.loads(table.round(6).to_json(orient="index"))

    print("\nturnover & costs:")
    for name, r in runs.items():
        t = r.turnover.loc[start:end]; c = r.costs.loc[start:end]
        print(f"  {name:<24} {BPY * t.mean():6.2f} turns/yr   "
              f"cost drag {100 * BPY * c.mean():5.3f}%/yr")
        report.setdefault("turnover", {})[name] = {
            "ann_turnover": round(float(BPY * t.mean()), 4),
            "ann_cost_drag_pct": round(float(100 * BPY * c.mean()), 4)}

    # ------------------------------------------------------------ inference
    banner("4. INFERENCE  (and why it is weak here)")
    ts = series["TS long/flat"]
    bh = series["Buy & hold"]
    excess = ts - bh

    nw = stats.newey_west_tstat(excess, overlap=LOOKBACK)
    print(f"TS long/flat vs buy & hold: {100 * BPY * nw['mean']:+.2f}%/yr, "
          f"t = {nw['t_stat']:+.3f}  ({nw['lags_used']} HAC lags)")
    n_independent = len(ts) / LOOKBACK
    print(f"\n  A {LOOKBACK}-bar lookback over {len(ts)} bars is only "
          f"{n_independent:.1f} independent momentum episodes.")
    print("  Daily sampling multiplies observations, not information: the HAC")
    print("  bandwidth is floored at 251 lags for exactly that reason. Treat")
    print("  every t-statistic below as indicative, not decisive.")
    report["hac_vs_buyhold"] = nw
    report["independent_episodes"] = round(n_independent, 2)

    boot = stats.moving_block_bootstrap(
        ts, lambda s: metrics.sharpe(s, periods_per_year=BPY),
        n_boot=3000, block=63, seed=1)
    print(f"\nTS long/flat Sharpe      : {boot['point']:.3f}, "
          f"95% CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] (63-day blocks)")
    boot_bh = stats.moving_block_bootstrap(
        bh, lambda s: metrics.sharpe(s, periods_per_year=BPY),
        n_boot=3000, block=63, seed=1)
    print(f"Buy & hold Sharpe        : {boot_bh['point']:.3f}, "
          f"95% CI [{boot_bh['ci_low']:.3f}, {boot_bh['ci_high']:.3f}]")
    report["bootstrap"] = {"ts_long_flat": boot, "buy_hold": boot_bh}

    # ----------------------------------------------------------- parameter grid
    banner("5. LOOKBACK GRID")
    rows = []
    for lb in (21, 42, 63, 126, 189, 252):
        for sk in (0, 21):
            if sk >= lb:
                continue
            for lo in (False, True):
                cfg = bt.BacktestConfig(lookback=lb, skip=sk, long_only=lo,
                                        cost_bps=COST_BPS, periods_per_year=BPY)
                try:
                    r = bt.timeseries_momentum(prices, cfg).returns.loc[start:end]
                    rows.append({
                        "lookback": lb, "skip": sk,
                        "book": "long/flat" if lo else "long/short",
                        "cagr_%": round(100 * metrics.cagr(r, BPY), 2),
                        "sharpe": round(metrics.sharpe(r, periods_per_year=BPY), 3),
                        "max_dd_%": round(100 * metrics.max_drawdown(r), 1),
                        "vs_bh": round(metrics.sharpe(r, periods_per_year=BPY)
                                       - metrics.sharpe(bh, periods_per_year=BPY), 3)})
                except Exception as exc:
                    rows.append({"lookback": lb, "skip": sk,
                                 "book": "long/flat" if lo else "long/short",
                                 "error": str(exc)[:40]})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv(os.path.join(OUT, "daily_grid.csv"), index=False)
    bh_sharpe = metrics.sharpe(bh, periods_per_year=BPY)
    beat = int((grid["sharpe"] > bh_sharpe).sum())
    print(f"\nbuy & hold Sharpe        : {bh_sharpe:.3f}")
    print(f"grid cells beating it    : {beat} of {len(grid)}")
    report["grid"] = {"n_cells": len(grid), "buy_hold_sharpe": round(bh_sharpe, 4),
                      "cells_beating_buy_hold": beat}

    # ------------------------------------------------------------ cost sweep
    banner("6. COST SENSITIVITY  (TS long/flat)")
    cost_rows = []
    for bps in (0.0, 1.0, 2.0, 5.0, 10.0, 25.0):
        r = bt.timeseries_momentum(prices, bt.BacktestConfig(
            lookback=LOOKBACK, skip=SKIP, long_only=True,
            cost_bps=bps, periods_per_year=BPY)).returns.loc[start:end]
        cost_rows.append({"cost_bps": bps,
                          "cagr_%": round(100 * metrics.cagr(r, BPY), 2),
                          "sharpe": round(metrics.sharpe(r, periods_per_year=BPY), 3)})
    costs = pd.DataFrame(cost_rows)
    print(costs.to_string(index=False))
    costs.to_csv(os.path.join(OUT, "daily_costs.csv"), index=False)
    report["cost_sweep"] = cost_rows

    # -------------------------------------------------- exposure control
    banner("7. IS IT TIMING, OR JUST LESS GOLD?")
    exposure = runs["TS long/flat"].weights.abs().sum(axis=1).loc[start:end]
    avg = float(exposure.mean())
    twin = bh * avg
    tw = stats.newey_west_tstat(ts - twin, overlap=LOOKBACK)
    worst = bh.nsmallest(50).index; best = bh.nlargest(50).index
    down = float(ts.loc[worst].mean() / bh.loc[worst].mean())
    up = float(ts.loc[best].mean() / bh.loc[best].mean())
    print(f"average exposure         : {100 * avg:.1f}%  "
          f"(long {100 * (exposure > 0.99).mean():.0f}% of bars, "
          f"flat {100 * (exposure < 0.01).mean():.0f}%)")
    print(f"constant-{100 * avg:.0f}%-exposure twin : Sharpe "
          f"{metrics.sharpe(twin, periods_per_year=BPY):.3f}, "
          f"CAGR {100 * metrics.cagr(twin, BPY):.2f}%, "
          f"maxDD {100 * metrics.max_drawdown(twin):.1f}%")
    print(f"TS long/flat             : Sharpe "
          f"{metrics.sharpe(ts, periods_per_year=BPY):.3f}, "
          f"CAGR {100 * metrics.cagr(ts, BPY):.2f}%, "
          f"maxDD {100 * metrics.max_drawdown(ts):.1f}%")
    print(f"timing excess vs twin    : {100 * BPY * tw['mean']:+.2f}%/yr, t = {tw['t_stat']:+.2f}")
    print(f"up-capture {100 * up:.1f}%  vs  down-capture {100 * down:.1f}%  "
          f"-> asymmetry {100 * (down - up):+.1f}pp")
    report["exposure_control"] = {
        "avg_exposure_pct": round(100 * avg, 2),
        "twin_sharpe": round(metrics.sharpe(twin, periods_per_year=BPY), 4),
        "ts_sharpe": round(metrics.sharpe(ts, periods_per_year=BPY), 4),
        "twin_maxdd_pct": round(100 * metrics.max_drawdown(twin), 2),
        "ts_maxdd_pct": round(100 * metrics.max_drawdown(ts), 2),
        "timing_excess_pct_yr": round(100 * BPY * tw["mean"], 3),
        "timing_t": round(tw["t_stat"], 3),
        "up_capture_pct": round(100 * up, 2), "down_capture_pct": round(100 * down, 2)}

    # --------------------------------------------------------- crash episode
    banner("8. THE JANUARY 2026 MOMENTUM CRASH")
    worst5 = ts.nsmallest(5)
    print("five worst days for TS long/flat:")
    for d, v in worst5.items():
        print(f"  {d.date()} {d.strftime('%a')}  {100 * v:7.2f}%   "
              f"(gold {100 * bh.loc[d]:7.2f}%)")
    dd = metrics.worst_drawdown_window(ts)
    print(f"\ndeepest drawdown         : {100 * dd['depth']:.1f}%  "
          f"{dd['peak'].date()} -> {dd['trough'].date()} ({dd['months_to_trough']} bars)")
    print(f"buy & hold drawdown      : {100 * metrics.max_drawdown(bh):.1f}%")
    report["worst_days"] = {str(d.date()): round(100 * float(v), 3) for d, v in worst5.items()}
    report["drawdown"] = {"ts_depth_pct": round(100 * dd["depth"], 2),
                          "peak": str(dd["peak"].date()), "trough": str(dd["trough"].date()),
                          "bh_depth_pct": round(100 * metrics.max_drawdown(bh), 2)}

    with open(os.path.join(OUT, "report_daily.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/report_daily.json and 4 CSVs")


if __name__ == "__main__":
    main()
