#!/usr/bin/env python3
"""Gold and silver: correlation, beta, and whether either leads the other.

The claim under test is that silver is correlated with gold and can
influence how gold moves. Those are two different claims -- one is about
contemporaneous co-movement (easy to confirm, and it matters for portfolio
risk), the other is about prediction (which would be an exploitable
inefficiency, and is the one worth being sceptical about).
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum import backtest as bt, data, metrics, signals as sig, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_daily")
BPY = data.BARS_PER_YEAR["daily"]


def banner(t): print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def load(sym):
    f = os.path.join(ROOT, "data", "raw_daily", f"{sym}.csv")
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["close"].astype(float)


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    # Full history for the statistical tests -- the 5-year file used by the
    # momentum study would throw away two thirds of the sample, and lead-lag
    # is exactly the test that needs the observations.
    au, ag = load("XAUUSD_full"), load("XAGUSD")
    px = pd.DataFrame({"XAU": au, "XAG": ag}).dropna()
    r = px.pct_change().dropna()
    banner("1. SAMPLE")
    print(f"common window : {r.index[0].date()} -> {r.index[-1].date()}  ({len(r)} weekday bars, "
          f"{len(r)/BPY:.1f} yrs)")
    for c in px.columns:
        print(f"  {c}: {px[c].iloc[0]:8.2f} -> {px[c].iloc[-1]:8.2f}   "
              f"ann vol {100*metrics.annual_vol(r[c], BPY):5.2f}%   "
              f"CAGR {100*metrics.cagr(r[c], BPY):5.2f}%")
    rep["sample"] = {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "bars": len(r)}

    # ---------------------------------------------------------------- 2
    banner("2. CORRELATION  (the part of the claim that is true)")
    rho = float(r["XAU"].corr(r["XAG"]))
    print(f"daily return correlation       : {rho:.3f}")
    roll = r["XAU"].rolling(261).corr(r["XAG"]).dropna()
    print(f"1-year rolling correlation     : min {roll.min():.3f}  median {roll.median():.3f}  "
          f"max {roll.max():.3f}")
    print(f"  never falls below            : {roll.min():.3f}  -> the link is persistent, not episodic")
    fit = stats.ols_hac(r["XAG"].to_numpy(), {"beta_to_gold": r["XAU"].to_numpy()})
    print(f"\nsilver = a + b x gold          : b = {fit['beta_to_gold']:.3f} "
          f"(t = {fit['beta_to_gold_t']:.1f}), R^2 = {fit['r_squared']:.3f}")
    print(f"  silver moves {fit['beta_to_gold']:.2f}x gold and is {metrics.annual_vol(r['XAG'],BPY)/metrics.annual_vol(r['XAU'],BPY):.2f}x as volatile")
    rep["correlation"] = {"daily_rho": round(rho, 4), "rolling_min": round(float(roll.min()), 4),
                          "rolling_max": round(float(roll.max()), 4),
                          "beta": round(fit["beta_to_gold"], 4), "r2": round(fit["r_squared"], 4)}

    # ---------------------------------------------------------------- 3
    banner("3. LEAD-LAG  (the part that would be exploitable)")
    print("Does yesterday's move in one predict today's move in the other?\n")
    rows = []
    for src, dst in (("XAG", "XAU"), ("XAU", "XAG")):
        y = r[dst].iloc[1:].to_numpy()
        x = r[src].iloc[:-1].to_numpy()
        f = stats.ols_hac(y, {"lag1": x})
        rows.append({"predictor": f"{src}[t-1]", "target": f"{dst}[t]",
                     "beta": round(f["lag1"], 4), "hac_t": round(f["lag1_t"], 3),
                     "r2": round(f["r_squared"], 5), "n": f["n_obs"]})
        print(f"  {src}[t-1] -> {dst}[t] : beta {f['lag1']:+.4f}  "
              f"HAC t {f['lag1_t']:+.2f}  R^2 {f['r_squared']:.5f}")
    # own-lag control: is any of this just autocorrelation?
    for a in ("XAU", "XAG"):
        f = stats.ols_hac(r[a].iloc[1:].to_numpy(), {"own_lag1": r[a].iloc[:-1].to_numpy()})
        print(f"  {a}[t-1] -> {a}[t] : beta {f['own_lag1']:+.4f}  "
              f"HAC t {f['own_lag1_t']:+.2f}  (own autocorrelation, for reference)")
        rows.append({"predictor": f"{a}[t-1]", "target": f"{a}[t]",
                     "beta": round(f["own_lag1"], 4), "hac_t": round(f["own_lag1_t"], 3),
                     "r2": round(f["r_squared"], 5), "n": f["n_obs"]})
    # joint: does silver add anything beyond gold's own lag?
    j = stats.ols_hac(r["XAU"].iloc[1:].to_numpy(),
                      {"gold_lag1": r["XAU"].iloc[:-1].to_numpy(),
                       "silver_lag1": r["XAG"].iloc[:-1].to_numpy()})
    print(f"\n  joint  XAU[t] ~ XAU[t-1] + XAG[t-1]:")
    print(f"     gold_lag1   {j['gold_lag1']:+.4f}  (t {j['gold_lag1_t']:+.2f})")
    print(f"     silver_lag1 {j['silver_lag1']:+.4f}  (t {j['silver_lag1_t']:+.2f})   R^2 {j['r_squared']:.5f}")
    crit = stats.deflated_threshold(n_trials=6)
    print(f"\n  Bonferroni critical |t| for these 6 tests: {crit:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gold_silver_leadlag.csv"), index=False)
    rep["lead_lag"] = rows
    rep["joint"] = {k: round(v, 5) for k, v in j.items() if isinstance(v, float)}
    rep["bonferroni_t"] = round(crit, 3)

    # ---------------------------------------------------------------- 4
    banner("4. DOES SILVER HELP TIME GOLD?  (the practical version)")
    px5 = px.loc[px.index[-1] - pd.DateOffset(years=5):]
    r5 = px5.pct_change().dropna()
    cfg = dict(lookback=252, skip=21, cost_bps=2.0, periods_per_year=BPY, long_only=True)
    gold_only = bt.timeseries_momentum(px5[["XAU"]], bt.BacktestConfig(**cfg))
    # signal from silver, traded in gold
    mom_ag = sig.compound_momentum(px5[["XAG"]].pct_change().iloc[1:], 252, 21)
    w = sig.timeseries_signal(mom_ag, allow_short=False).rename(columns={"XAG": "XAU"})
    silver_times_gold = bt.run_weighted_backtest(
        w.reindex(r5.index).fillna(0.0), r5[["XAU"]], cost_bps=2.0,
        label="silver signal -> gold", config=bt.BacktestConfig(**cfg))
    bh = bt.buy_and_hold(px5[["XAU"]], "Buy & hold gold")
    s = max(x.returns.index[0] for x in (gold_only, silver_times_gold, bh))
    e = min(x.returns.index[-1] for x in (gold_only, silver_times_gold, bh))
    series = {"Buy & hold gold": bh.returns.loc[s:e],
              "Gold signal -> gold": gold_only.returns.loc[s:e],
              "Silver signal -> gold": silver_times_gold.returns.loc[s:e]}
    tbl = metrics.summary_table(series, periods_per_year=BPY)
    print(tbl[["bars", "cagr_%", "vol_%", "sharpe", "max_dd_%"]].round(3).to_string())
    rep["timing_gold"] = json.loads(tbl.round(6).to_json(orient="index"))

    # ---------------------------------------------------------------- 5
    banner("5. WHAT THE CORRELATION COSTS YOU IN A MOMENTUM BOOK")
    both = r5[["XAU", "XAG"]]
    eq = both.mean(axis=1)
    print(f"correlation over the 5y window : {both['XAU'].corr(both['XAG']):.3f}")
    print(f"equal-weight gold+silver vol   : {100*metrics.annual_vol(eq, BPY):.2f}%")
    print(f"gold alone                     : {100*metrics.annual_vol(both['XAU'], BPY):.2f}%")
    ratio = 100 * metrics.annual_vol(eq, BPY) / (
        0.5 * 100 * metrics.annual_vol(both["XAU"], BPY) + 0.5 * 100 * metrics.annual_vol(both["XAG"], BPY))
    print(f"diversification ratio          : {ratio:.3f}  (1.00 = no benefit at all)")
    print(f"\n  Two names this correlated are close to one position. A cross-sectional")
    print(f"  book that ranks them against each other is choosing between near-twins;")
    print(f"  one that holds both is running a concentrated precious-metals bet.")
    rep["diversification"] = {"rho_5y": round(float(both["XAU"].corr(both["XAG"])), 4),
                              "ratio": round(float(ratio), 4)}

    with open(os.path.join(OUT, "report_gold_silver.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/report_gold_silver.json")


if __name__ == "__main__":
    main()
