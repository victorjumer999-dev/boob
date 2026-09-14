#!/usr/bin/env python3
"""Cross-sectional momentum on a wide, multi-asset-class universe.

The monthly study ran on 6 sector ETFs + SPY. Seven names is too thin for a
cross-sectional strategy: the top quintile is one or two assets, so the
result is mostly idiosyncratic noise, and every name is an equity, so there
is no diversification to harvest. This widens the universe to 13 instruments
across four asset classes and re-runs the same engine.

IMPORTANT LIMITATION: the commodity series are SPOT prices. A spot commodity
is not investable -- a real position rolls futures and earns (or pays) the
roll yield, which for energy in particular dominates. These results test
whether the momentum SIGNAL ranks assets usefully; they are not an
achievable P&L for the commodity legs.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum import backtest as bt, data, metrics, signals as sig, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_wide")
WIDE = os.path.join(ROOT, "data", "raw_wide")

CLASSES = {
    "equity": ["SPY", "XLE", "XLF", "XLK", "XLP", "XLV", "XLY"],
    "energy": ["WTI", "BRENT", "NATGAS"],
    "metals": ["COPPER", "GOLD", "SILVER"],
}
ORIGINAL = ["SPY", "XLE", "XLF", "XLK", "XLP", "XLV", "XLY"]
COST_BPS, LOOKBACK, SKIP = 10.0, 12, 1


def banner(t): print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {}

    # ------------------------------------------------------------- 1 data
    banner("1. UNIVERSE")
    # Normalise EACH series to a month-end stamp before joining. The feeds
    # disagree: the commodity endpoints stamp month-start (1999-01-01), the
    # ETFs month-end (1999-01-31). Joining first and normalising afterwards
    # creates two rows per month that then collapse into each other.
    cols = {}
    for cls, syms in CLASSES.items():
        for s in syms:
            ser = pd.read_csv(f"{WIDE}/{s}.csv", parse_dates=["date"]).set_index("date")["close"]
            ser.index = ser.index.to_period("M").to_timestamp("M")
            cols[s] = ser[~ser.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols).sort_index()

    of = {s: c for c, ss in CLASSES.items() for s in ss}
    aud = data.audit(panel)
    aud.insert(0, "class", [of[s] for s in aud.index])
    print(aud[["class", "bars", "start", "end", "ann_ret_%", "ann_vol_%"]].to_string())

    # Ragged panel: an asset joins when its history starts. Requiring the
    # common sample would throw away 1999-2011 to wait for the metals feed.
    wide = data.align_panel(panel, min_coverage=8 / len(panel.columns))
    n_names = panel.loc[wide.index].notna().sum(axis=1)
    print(f"\nragged panel   : {wide.index[0].date()} -> {wide.index[-1].date()}  "
          f"({len(wide)} months)")
    print(f"names available: min {n_names.min()}, median {int(n_names.median())}, max {n_names.max()}")
    rep["universe"] = {"n_assets": len(panel.columns), "months": len(wide),
                       "start": str(wide.index[0].date()), "end": str(wide.index[-1].date()),
                       "min_names": int(n_names.min()), "max_names": int(n_names.max())}

    # ------------------------------------------------- 2 diversification
    banner("2. IS THE WIDER UNIVERSE ACTUALLY MORE DIVERSIFIED?")
    rets_all = wide.pct_change().iloc[1:]
    def avg_corr(frame):
        c = frame.corr().to_numpy()
        iu = np.triu_indices_from(c, k=1)
        return float(np.nanmean(c[iu]))
    print(f"average pairwise correlation, original 7 equities : {avg_corr(rets_all[ORIGINAL]):.3f}")
    print(f"average pairwise correlation, wide 13             : {avg_corr(rets_all):.3f}")
    print("\nwithin / across asset classes:")
    for a in CLASSES:
        print(f"  {a:8} internal : {avg_corr(rets_all[CLASSES[a]]):+.3f}")
    for a, b in (("equity", "energy"), ("equity", "metals"), ("energy", "metals")):
        c = rets_all[CLASSES[a]].corrwith(rets_all[CLASSES[b]].mean(axis=1)).mean()
        print(f"  {a:8} vs {b:8}: {c:+.3f}")
    print(f"\n  gold vs silver                : {rets_all['GOLD'].corr(rets_all['SILVER']):+.3f}"
          "   <- near-twins, see report")
    print(f"  WTI vs Brent                  : {rets_all['WTI'].corr(rets_all['BRENT']):+.3f}"
          "   <- near-twins too")
    rep["diversification"] = {
        "avg_corr_original7": round(avg_corr(rets_all[ORIGINAL]), 4),
        "avg_corr_wide13": round(avg_corr(rets_all), 4),
        "gold_silver": round(float(rets_all["GOLD"].corr(rets_all["SILVER"])), 4),
        "wti_brent": round(float(rets_all["WTI"].corr(rets_all["BRENT"])), 4)}

    # ------------------------------------------------------- 3 strategies
    banner(f"3. STRATEGIES  ({LOOKBACK}-{SKIP} momentum, {COST_BPS}bp one-way)")
    cfg = bt.BacktestConfig(lookback=LOOKBACK, skip=SKIP, cost_bps=COST_BPS,
                            top_q=0.2, bottom_q=0.2, min_names=6)
    base = dict(lookback=LOOKBACK, skip=SKIP, cost_bps=COST_BPS,
                top_q=0.2, bottom_q=0.2, min_names=6)
    runs = {}
    runs["XS long/short (13)"] = bt.cross_sectional_momentum(wide, cfg)
    runs["XS long/short (13, RP)"] = bt.cross_sectional_momentum(
        wide, bt.BacktestConfig(**base, risk_parity=True))
    runs["XS long-only (13)"] = bt.cross_sectional_momentum(
        wide, bt.BacktestConfig(**base, long_only=True))
    runs["XS long-only (13, RP)"] = bt.cross_sectional_momentum(
        wide, bt.BacktestConfig(**base, long_only=True, risk_parity=True))
    runs["TS long/flat (13)"] = bt.timeseries_momentum(
        wide, bt.BacktestConfig(lookback=LOOKBACK, skip=SKIP, cost_bps=COST_BPS, long_only=True))
    runs["Equal-weight (13)"] = bt.buy_and_hold(wide, "Equal-weight (13)")
    # like-for-like control: the ORIGINAL 7, same window, same engine
    orig = wide[ORIGINAL].dropna(how="all")
    runs["XS long/short (orig 7)"] = bt.cross_sectional_momentum(
        orig, bt.BacktestConfig(lookback=LOOKBACK, skip=SKIP, cost_bps=COST_BPS,
                                top_q=0.3, bottom_q=0.3, min_names=5))
    runs["XS long/short (orig 7, RP)"] = bt.cross_sectional_momentum(
        orig, bt.BacktestConfig(lookback=LOOKBACK, skip=SKIP, cost_bps=COST_BPS,
                                top_q=0.3, bottom_q=0.3, min_names=5, risk_parity=True))
    runs["SPY buy & hold"] = bt.buy_and_hold(wide[["SPY"]], "SPY buy & hold")

    s = max(r.returns.index[0] for r in runs.values())
    e = min(r.returns.index[-1] for r in runs.values())
    series = {k: r.returns.loc[s:e] for k, r in runs.items()}
    print(f"common window : {s.date()} -> {e.date()}  ({len(series['SPY buy & hold'])} months)")
    tbl = metrics.summary_table(series)
    print()
    print(tbl[["bars", "cagr_%", "vol_%", "sharpe", "max_dd_%", "ruined",
               "hit_rate_%", "skew"]].round(3).to_string())
    tbl.round(6).to_csv(f"{OUT}/wide_summary.csv")
    rets_all.to_csv(f"{OUT}/asset_returns.csv")
    pd.DataFrame(series).to_csv(f"{OUT}/wide_returns.csv")
    rep["summary"] = json.loads(tbl.round(6).to_json(orient="index"))

    print("\nturnover:")
    for k, r in runs.items():
        print(f"  {k:<24} {12 * r.turnover.loc[s:e].mean():5.2f} turns/yr")

    # -------------------------------------------------------- 4 inference
    banner("4. DID WIDENING THE UNIVERSE HELP?")
    w13, o7 = series["XS long/short (13)"], series["XS long/short (orig 7)"]
    w13rp = series["XS long/short (13, RP)"]
    for name, x in (("wide 13", w13), ("wide 13 RP", w13rp), ("original 7", o7)):
        nw = stats.newey_west_tstat(x, overlap=LOOKBACK)
        bo = stats.moving_block_bootstrap(x, metrics.sharpe, n_boot=3000, block=12, seed=1)
        print(f"  {name:<12} mean {100*12*nw['mean']:+6.2f}%/yr  HAC t {nw['t_stat']:+.2f}   "
              f"Sharpe {bo['point']:+.3f}  95% CI [{bo['ci_low']:+.3f}, {bo['ci_high']:+.3f}]")
        rep.setdefault("inference", {})[name] = {"ann_mean_pct": round(100*12*nw["mean"], 3),
                                                 "hac_t": round(nw["t_stat"], 3),
                                                 "sharpe": round(bo["point"], 4),
                                                 "ci": [round(bo["ci_low"], 4), round(bo["ci_high"], 4)]}
    diff = stats.newey_west_tstat(w13rp - o7, overlap=LOOKBACK)
    print(f"\n  difference (13 RP - 7): {100*12*diff['mean']:+.2f}%/yr, HAC t = {diff['t_stat']:+.2f}")
    rep["inference"]["difference"] = {"ann_pct": round(100*12*diff["mean"], 3),
                                      "hac_t": round(diff["t_stat"], 3)}

    pooled = stats.predictive_regression(
        sig.compound_momentum(rets_all, LOOKBACK, SKIP), rets_all, overlap=LOOKBACK)
    print(f"\n  pooled R_t+1 = a + b*M_t : b = {pooled['beta']:+.4f}, HAC t = {pooled['beta_t']:+.2f}, "
          f"R^2 = {pooled['r_squared']:.5f}, n = {pooled['n_obs']}")
    rep["pooled_regression"] = pooled

    # ------------------------------------------------------------ 5 grid
    banner("5. LOOKBACK GRID (wide universe)")
    rows = []
    for lb in (3, 6, 9, 12, 18, 24):
        for sk in (0, 1):
            for lo in (False, True):
                try:
                    r = bt.cross_sectional_momentum(wide, bt.BacktestConfig(
                        lookback=lb, skip=sk, cost_bps=COST_BPS, top_q=0.2, bottom_q=0.2,
                        min_names=6, long_only=lo, risk_parity=True)).returns.loc[s:e]
                    ruined = metrics.is_ruined(r)
                    rows.append({"lookback": lb, "skip": sk,
                                 "book": "long-only" if lo else "long/short",
                                 # Past ruin these are arithmetic, not outcomes.
                                 "cagr_%": "WIPED OUT" if ruined else round(100*metrics.cagr(r), 2),
                                 "sharpe": round(metrics.sharpe(r), 3),
                                 "max_dd_%": "WIPED OUT" if ruined else round(100*metrics.max_drawdown(r), 1)})
                except Exception as exc:
                    rows.append({"lookback": lb, "skip": sk,
                                 "book": "long-only" if lo else "long/short",
                                 "error": str(exc)[:38]})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv(f"{OUT}/wide_grid.csv", index=False)
    crit = stats.deflated_threshold(n_trials=len(grid))
    best = grid.loc[grid["sharpe"].idxmax()]
    print(f"\n  best cell: lookback {int(best['lookback'])}, skip {int(best['skip'])}, "
          f"{best['book']}, Sharpe {best['sharpe']:.3f}")
    print(f"  Bonferroni critical |t| for {len(grid)} trials: {crit:.2f}")
    rep["grid"] = {"n_cells": len(grid), "best_sharpe": float(best["sharpe"]),
                   "bonferroni_t": round(crit, 3)}

    # ------------------------------------------- 6 the one thing that worked
    banner("6. TIME-SERIES MOMENTUM ON THE WIDE UNIVERSE vs SPY")
    ts, spy = series["TS long/flat (13)"], series["SPY buy & hold"]
    print(f"  TS long/flat (13) : CAGR {100*metrics.cagr(ts):5.2f}%  vol {100*metrics.annual_vol(ts):5.2f}%  "
          f"Sharpe {metrics.sharpe(ts):.3f}  maxDD {100*metrics.max_drawdown(ts):6.1f}%")
    print(f"  SPY buy & hold    : CAGR {100*metrics.cagr(spy):5.2f}%  vol {100*metrics.annual_vol(spy):5.2f}%  "
          f"Sharpe {metrics.sharpe(spy):.3f}  maxDD {100*metrics.max_drawdown(spy):6.1f}%")
    # Scale TS up to SPY's volatility so the comparison is like-for-like.
    k = metrics.annual_vol(spy) / metrics.annual_vol(ts)
    tsx = ts * k
    print(f"\n  TS scaled to SPY vol ({k:.2f}x): CAGR {100*metrics.cagr(tsx):5.2f}%  "
          f"Sharpe {metrics.sharpe(tsx):.3f}  maxDD {100*metrics.max_drawdown(tsx):6.1f}%")
    nw = stats.newey_west_tstat(tsx - spy, overlap=LOOKBACK)
    print(f"  excess over SPY at equal vol : {100*12*nw['mean']:+.2f}%/yr, HAC t = {nw['t_stat']:+.2f}")
    bo = stats.moving_block_bootstrap(ts, metrics.sharpe, n_boot=3000, block=12, seed=1)
    bs = stats.moving_block_bootstrap(spy, metrics.sharpe, n_boot=3000, block=12, seed=1)
    print(f"  Sharpe 95% CI  TS {bo['point']:.3f} [{bo['ci_low']:.3f}, {bo['ci_high']:.3f}]   "
          f"SPY {bs['point']:.3f} [{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]")
    rep["ts_vs_spy"] = {"ts_sharpe": round(bo["point"], 4), "spy_sharpe": round(bs["point"], 4),
                        "ts_ci": [round(bo["ci_low"], 4), round(bo["ci_high"], 4)],
                        "spy_ci": [round(bs["ci_low"], 4), round(bs["ci_high"], 4)],
                        "vol_scalar": round(float(k), 4),
                        "excess_at_equal_vol_pct": round(100*12*nw["mean"], 3),
                        "excess_t": round(nw["t_stat"], 3),
                        "ts_maxdd": round(100*metrics.max_drawdown(ts), 2),
                        "spy_maxdd": round(100*metrics.max_drawdown(spy), 2)}

    with open(f"{OUT}/report_wide.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/report_wide.json and 3 CSVs")


if __name__ == "__main__":
    main()
