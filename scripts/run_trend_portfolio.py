#!/usr/bin/env python3
"""One pre-specified test: is there an edge anywhere in this project?

97 configurations have been tested across three studies. Searching further on
the same data manufactures findings. So this runs exactly ONE strategy, whose
every design choice was dictated by what those studies already established:

  * TIME-SERIES, not cross-sectional. Cross-sectional momentum failed on 7
    equities (Sharpe 0.148), on 13 assets across 4 classes (0.041), and its
    pooled predictive coefficient is negative. Time-series was the only
    construction that was ever competitive.
  * BLENDED lookbacks (3/6/12), not the best grid cell. Grid Sharpe ranged
    0.19-1.19 at daily frequency; that spread is noise, and picking its
    maximum is how a backtest lies. Averaging lowers the backtest number and
    raises the chance the live result resembles it.
  * INVERSE-VOL sized. Volatility spans 12% to 71% across this universe;
    equal weights produced a -99.8% drawdown.
  * PORTFOLIO VOL TARGET. The one lever that improved risk-adjusted return in
    every single study.

Then it is judged against a deliberately hostile bar: a passive benchmark, a
split-sample stability check, a permutation null that destroys signal timing
while preserving everything else, and the Bonferroni threshold for all 97
trials (|t| = 3.47).
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum import backtest as bt, data, metrics, signals as sig, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_wide")
WIDE = os.path.join(ROOT, "data", "raw_wide")
SYMS = ["SPY","XLE","XLF","XLK","XLP","XLV","XLY","WTI","BRENT","NATGAS","COPPER","GOLD","SILVER"]
LOOKBACKS = (3, 6, 12)
VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 12, 0.10, 2.0, 10.0
N_TRIALS_SO_FAR = 97


def banner(t): print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def load_panel():
    cols = {}
    for s in SYMS:
        ser = pd.read_csv(f"{WIDE}/{s}.csv", parse_dates=["date"]).set_index("date")["close"]
        ser.index = ser.index.to_period("M").to_timestamp("M")
        cols[s] = ser[~ser.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols).sort_index()
    return data.align_panel(panel, min_coverage=8 / len(SYMS))


def blended_signal(returns, long_only):
    """Average of the sign of 3-, 6- and 12-month trend. No skip: this is
    trend, not the cross-sectional 12-1 convention."""
    parts = [np.sign(sig.compound_momentum(returns, lb, 0)) for lb in LOOKBACKS]
    s = sum(parts) / len(parts)
    return s.clip(lower=0.0) if long_only else s


def build_weights(returns, long_only):
    s = blended_signal(returns, long_only)
    gross = s.abs().sum(axis=1).replace(0.0, np.nan)
    s = s.div(gross, axis=0).fillna(0.0)            # gross 1 before sizing
    return sig.inverse_vol_weights(s, returns, window=VOL_WINDOW)


def run(returns, long_only, cost_bps=COST_BPS, vol_target=True):
    w = build_weights(returns, long_only)
    cfg = bt.BacktestConfig(cost_bps=cost_bps, target_vol=TARGET_VOL if vol_target else None,
                            vol_window=VOL_WINDOW, max_leverage=MAX_LEV)
    unlev = bt.run_weighted_backtest(w, returns, cost_bps=cost_bps, config=cfg)
    if not vol_target:
        return unlev
    lev = sig.volatility_target_scalar(unlev.gross_returns, TARGET_VOL, VOL_WINDOW,
                                       MAX_LEV).reindex(w.index)
    return bt.run_weighted_backtest(w, returns, cost_bps=cost_bps, leverage=lev, config=cfg)


def main():
    rep = {"n_prior_trials": N_TRIALS_SO_FAR}
    panel = load_panel()
    rets = panel.pct_change().iloc[1:]
    banner("1. THE PRE-SPECIFIED STRATEGY")
    print(f"universe   : {len(SYMS)} assets, {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"signal     : mean of sign(3m), sign(6m), sign(12m) trend, no skip")
    print(f"sizing     : inverse trailing volatility ({VOL_WINDOW}m), gross 1")
    print(f"risk       : {100*TARGET_VOL:.0f}% portfolio vol target, max {MAX_LEV}x leverage")
    print(f"costs      : {COST_BPS}bp one-way")
    print(f"prior trials in this project: {N_TRIALS_SO_FAR}  ->  Bonferroni |t| = "
          f"{stats.deflated_threshold(N_TRIALS_SO_FAR):.2f}")

    lf, ls = run(rets, True), run(rets, False)
    ew = bt.buy_and_hold(panel, "Equal-weight (13)")
    spy = bt.buy_and_hold(panel[["SPY"]], "SPY")
    s = max(x.returns.index[0] for x in (lf, ls, ew, spy))
    e = min(x.returns.index[-1] for x in (lf, ls, ew, spy))
    series = {"Trend long/flat": lf.returns.loc[s:e], "Trend long/short": ls.returns.loc[s:e],
              "Equal-weight (13)": ew.returns.loc[s:e], "SPY": spy.returns.loc[s:e]}

    banner("2. RESULT")
    print(f"window: {s.date()} -> {e.date()}  ({len(series['SPY'])} months)")
    tbl = metrics.summary_table(series)
    print()
    print(tbl[["bars","cagr_%","vol_%","sharpe","max_dd_%","hit_rate_%","skew"]].round(3).to_string())
    tbl.round(6).to_csv(f"{OUT}/trend_summary.csv")
    pd.DataFrame(series).to_csv(f"{OUT}/trend_returns.csv")
    rep["summary"] = json.loads(tbl.round(6).to_json(orient="index"))
    turn_lf = float(12 * lf.turnover.loc[s:e].mean())
    print(f"\nturnover: long/flat {turn_lf:.2f}/yr, "
          f"long/short {12*ls.turnover.loc[s:e].mean():.2f}/yr")
    rep["turnover_long_flat"] = round(turn_lf, 3)

    best = series["Trend long/flat"]
    banner("3. IS IT BETTER THAN HOLDING EVERYTHING PASSIVELY?")
    bench = series["Equal-weight (13)"]
    k = metrics.annual_vol(bench) / metrics.annual_vol(best)
    scaled = best * k
    print(f"  trend scaled to benchmark vol ({k:.2f}x): CAGR {100*metrics.cagr(scaled):.2f}%  "
          f"Sharpe {metrics.sharpe(scaled):.3f}  maxDD {100*metrics.max_drawdown(scaled):.1f}%")
    print(f"  equal-weight benchmark               : CAGR {100*metrics.cagr(bench):.2f}%  "
          f"Sharpe {metrics.sharpe(bench):.3f}  maxDD {100*metrics.max_drawdown(bench):.1f}%")
    nw = stats.newey_west_tstat(scaled - bench, overlap=12)
    print(f"  excess at equal vol : {100*12*nw['mean']:+.2f}%/yr, HAC t = {nw['t_stat']:+.2f}  "
          f"(needs |t| > {stats.deflated_threshold(N_TRIALS_SO_FAR):.2f})")
    rep["vs_benchmark"] = {"excess_pct_yr": round(100*12*nw["mean"], 3),
                           "hac_t": round(nw["t_stat"], 3),
                           "scaled_sharpe": round(metrics.sharpe(scaled), 4),
                           "bench_sharpe": round(metrics.sharpe(bench), 4),
                           "scaled_maxdd": round(100*metrics.max_drawdown(scaled), 2),
                           "bench_maxdd": round(100*metrics.max_drawdown(bench), 2)}

    banner("4. IS IT STABLE ACROSS TIME?")
    mid = best.index[len(best) // 2]
    rows = []
    for name, sl in (("first half", best.loc[:mid]), ("second half", best.loc[mid:])):
        b = bench.loc[sl.index]
        rows.append({"period": name, "start": sl.index[0].date(), "end": sl.index[-1].date(),
                     "months": len(sl), "cagr_%": round(100*metrics.cagr(sl), 2),
                     "sharpe": round(metrics.sharpe(sl), 3),
                     "max_dd_%": round(100*metrics.max_drawdown(sl), 1),
                     "bench_sharpe": round(metrics.sharpe(b), 3)})
    half = pd.DataFrame(rows)
    print(half.to_string(index=False))
    rep["stability"] = rows

    banner("5. PERMUTATION NULL  (does the TIMING carry information?)")
    print("  Shuffling the signal rows destroys the alignment between signal and")
    print("  subsequent return while preserving the positions, the sizing and the")
    print("  asset returns exactly. If the real strategy is inside this null")
    print("  distribution, its performance comes from exposure, not from timing.")
    w_real = build_weights(rets, True)
    rng = np.random.default_rng(7)
    null = []
    for _ in range(1000):
        perm = w_real.to_numpy()[rng.permutation(len(w_real))]
        wp = pd.DataFrame(perm, index=w_real.index, columns=w_real.columns)
        try:
            r = bt.run_weighted_backtest(wp, rets, cost_bps=COST_BPS).returns.loc[s:e]
            null.append(metrics.sharpe(r))
        except Exception:
            continue
    null = np.array(null)
    real_unlev = run(rets, True, vol_target=False).returns.loc[s:e]
    real_sharpe = metrics.sharpe(real_unlev)
    pct = float((null < real_sharpe).mean())
    print(f"\n  real (unlevered, same construction) Sharpe : {real_sharpe:.3f}")
    print(f"  permutation null: mean {null.mean():.3f}, sd {null.std():.3f}, "
          f"95th pct {np.percentile(null, 95):.3f}")
    print(f"  the real strategy sits at the {100*pct:.1f}th percentile of the null")
    print(f"  empirical p-value (one-sided) = {1-pct:.4f}")
    rep["permutation"] = {"real_sharpe": round(real_sharpe, 4), "null_mean": round(float(null.mean()), 4),
                          "null_sd": round(float(null.std()), 4),
                          "null_p95": round(float(np.percentile(null, 95)), 4),
                          "percentile": round(100*pct, 2), "p_value": round(1-pct, 4),
                          "n_draws": len(null)}

    banner("6. VERDICT")
    crit = stats.deflated_threshold(N_TRIALS_SO_FAR)
    passes_t = abs(nw["t_stat"]) > crit
    passes_perm = (1 - pct) < 0.05
    stable = min(r["sharpe"] for r in rows) > 0
    dd_better = abs(metrics.max_drawdown(scaled)) < abs(metrics.max_drawdown(bench))
    for label, ok in (("beats passive benchmark at Bonferroni |t|", passes_t),
                      ("beats the permutation null at p < 0.05", passes_perm),
                      ("positive Sharpe in BOTH halves", stable),
                      ("shallower drawdown than the benchmark", dd_better)):
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    rep["verdict"] = {"beats_benchmark_bonferroni": bool(passes_t),
                      "beats_permutation": bool(passes_perm),
                      "stable_both_halves": bool(stable),
                      "shallower_drawdown": bool(dd_better)}
    with open(f"{OUT}/report_trend.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/report_trend.json")


if __name__ == "__main__":
    main()
