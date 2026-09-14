#!/usr/bin/env python3
"""Recreate the reference sheet's three chart panels with the real results."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum import backtest as bt
from momentum import data, metrics, signals as sig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results")
SECTORS = ["XLE", "XLF", "XLK", "XLP", "XLV", "XLY"]

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
BLUE = "#4c9aff"
RED = "#f4685c"
GREY = "#c9d1d9"
GREEN = "#3fb950"


def style(ax, title=None):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=FG, fontsize=11, loc="left", pad=10, weight="bold")


def main():
    panel = data.align_panel(
        data.drop_incomplete_last_month(
            data.load_panel(RAW, SECTORS + ["SPY"]), asof=pd.Timestamp("2026-09-14")
        )
    )
    sectors = panel[SECTORS]
    rets = sectors.pct_change().iloc[1:]

    cfg_long = bt.BacktestConfig(lookback=12, skip=1, top_q=1/3, long_only=True, cost_bps=10.0)
    top = bt.cross_sectional_momentum(sectors, cfg_long)

    # The short leg on its own, as a long book, so it can be plotted as the
    # poster's "Bottom Decile" line rather than inferred from the spread.
    mom = sig.compound_momentum(rets, 12, 1)
    inverted = sig.quantile_weights(-mom, top_q=1/3, long_only=True)
    bottom = bt.run_weighted_backtest(inverted, rets, cost_bps=10.0, label="bottom tercile")
    market = bt.buy_and_hold(panel[["SPY"]])

    start = max(top.returns.index[0], bottom.returns.index[0], market.returns.index[0])
    end = min(top.returns.index[-1], bottom.returns.index[-1], market.returns.index[-1])
    curves = {
        "Momentum (top tercile)": top.returns.loc[start:end],
        "S&P 500 (SPY)": market.returns.loc[start:end],
        "Bottom tercile": bottom.returns.loc[start:end],
    }

    fig = plt.figure(figsize=(13, 11), facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1], hspace=0.42, wspace=0.22)

    # ---- Panel 1: cumulative growth ---------------------------------------
    ax = fig.add_subplot(gs[0, :])
    style(ax, f"Sector momentum vs market  ({start.strftime('%Y')}–{end.strftime('%Y')}, "
              f"monthly, 10bp one-way costs)")
    colours = {"Momentum (top tercile)": BLUE, "S&P 500 (SPY)": GREY, "Bottom tercile": RED}
    for name, series in curves.items():
        equity = (1 + series).cumprod()
        equity = pd.concat([pd.Series([1.0], index=[start - pd.offsets.MonthEnd(1)]), equity])
        ax.plot(equity.index, equity.to_numpy(), label=name, color=colours[name], linewidth=1.6)
        total = 100 * (equity.iloc[-1] - 1)
        ax.annotate(f"{total:+,.0f}%", xy=(equity.index[-1], equity.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=colours[name], fontsize=9, weight="bold", va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Cumulative growth of 1", color=FG, fontsize=9)
    ax.set_xlim(equity.index[0], equity.index[-1] + pd.offsets.MonthEnd(14))
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper left")
    leg.get_frame().set_alpha(0.9)
    # State what the lines actually show. Winners do beat losers -- but both
    # legs are long books riding the equity premium, and the long/short
    # spread that is supposed to isolate the momentum effect earns a Sharpe
    # of 0.15 (t = 0.86) after costs. The gap on this chart is mostly beta.
    ax.text(0.5, 0.075,
            "Winners do beat losers — but the long/short spread that isolates that gap\n"
            "earns Sharpe 0.15 (t = 0.86) after costs, and the top tercile beats SPY by\n"
            "only 1.1%/yr (t = 0.74). Both legs are long books riding the equity premium.",
            transform=ax.transAxes, color=FG, fontsize=8.5, ha="center",
            bbox=dict(facecolor="#1c2128", edgecolor=RED, boxstyle="round,pad=0.5", alpha=0.95))

    # ---- Panel 2: bucket ladder -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style(ax, "Avg next-month return by prior 12-1 momentum")
    buckets = bt.decile_portfolios(sectors, 12, 1, n_buckets=3)
    means = 100 * buckets["mean"]
    errs = 100 * buckets["std"] / np.sqrt(buckets["count"])
    bars = ax.bar(buckets.index, means.to_numpy(), yerr=errs.to_numpy(),
                  color=[RED, GREY, BLUE], capsize=5, ecolor=FG, width=0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1\nlosers", "2", "3\nwinners"], color=FG)
    ax.set_ylabel("Next-month return (%)", color=FG, fontsize=9)
    ax.axhline(0, color=FG, linewidth=0.8)
    for bar, value, err in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + err + 0.05, f"{value:.2f}%",
                ha="center", color=FG, fontsize=9)
    ax.set_ylim(0, max(means) * 1.45)
    ax.text(0.5, 0.9, "ladder is monotone, but the error bars overlap",
            transform=ax.transAxes, color=FG, fontsize=8, ha="center", style="italic")

    # ---- Panel 3: rolling z-score -----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style(ax, "Rolling 12-1 momentum z-score (XLK)")
    z = sig.rolling_zscore(mom["XLK"], window=36).dropna().loc["2015":]
    ax.plot(z.index, z.to_numpy(), color=BLUE, linewidth=1.2)
    ax.axhline(1, color=GREEN, linestyle="--", linewidth=1)
    ax.axhline(-1, color=RED, linestyle="--", linewidth=1)
    ax.axhline(0, color=FG, linewidth=0.6, alpha=0.5)
    ax.fill_between(z.index, 1, z.to_numpy(), where=(z.to_numpy() > 1), color=GREEN, alpha=0.25)
    ax.fill_between(z.index, -1, z.to_numpy(), where=(z.to_numpy() < -1), color=RED, alpha=0.25)
    ax.text(0.99, 0.93, "Long (z > 1)", transform=ax.transAxes, color=GREEN,
            fontsize=8, ha="right")
    ax.text(0.99, 0.06, "Short (z < -1)", transform=ax.transAxes, color=RED,
            fontsize=8, ha="right")
    ax.set_ylabel("Signal (z-score)", color=FG, fontsize=9)
    ax.set_ylim(-3, 3)

    # ---- Panel 4: drawdowns ------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    style(ax, "Drawdown: trend filter vs passive")
    ls_cfg = bt.BacktestConfig(lookback=12, skip=1, cost_bps=10.0)
    ts = bt.timeseries_momentum(
        sectors, bt.BacktestConfig(lookback=12, skip=1, long_only=True, cost_bps=10.0)
    ).returns.loc[start:end]
    ew = bt.buy_and_hold(sectors).returns.loc[start:end]
    for series, colour, name in [(ew, GREY, "Equal-weight sectors"), (ts, BLUE, "Time-series momentum (long/flat)")]:
        dd = 100 * metrics.drawdown_series(series)
        ax.fill_between(dd.index, dd.to_numpy(), 0, color=colour, alpha=0.45)
        ax.plot(dd.index, dd.to_numpy(), color=colour, linewidth=1.0,
                label=f"{name}  (max {dd.min():.0f}%)")
    ax.set_ylabel("Drawdown (%)", color=FG, fontsize=9)
    ax.set_xlim(start, end)
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="lower left")
    leg.get_frame().set_alpha(0.9)

    # ---- Panel 5: parameter grid ------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    style(ax, "Sharpe across the parameter grid")
    grid = pd.read_csv(os.path.join(OUT, "parameter_grid.csv"))
    for book, colour, marker in [("long-only", BLUE, "o"), ("long/short", RED, "s")]:
        sub = grid[grid["book"] == book]
        for skip, ls in [(0, "-"), (1, "--")]:
            cell = sub[sub["skip"] == skip].sort_values("lookback")
            ax.plot(cell["lookback"], cell["sharpe"], ls, marker=marker, color=colour,
                    markersize=4, linewidth=1.2, alpha=1.0 if skip == 1 else 0.55,
                    label=f"{book}, skip={skip}")
    ax.axhline(0, color=FG, linewidth=0.8)
    ax.axhline(0.720, color=GREEN, linestyle=":", linewidth=1.2)
    ax.text(24, 0.745, "SPY = 0.72", color=GREEN, fontsize=8, ha="right")
    ax.set_xlabel("Lookback (months)", color=FG, fontsize=9)
    ax.set_ylabel("Sharpe ratio", color=FG, fontsize=9)
    ax.set_xticks([3, 6, 9, 12, 18, 24])
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7, loc="center right")
    leg.get_frame().set_alpha(0.9)

    fig.suptitle("Momentum Trading — reproduction on real data (6 SPDR sectors + SPY, "
                 f"monthly, {start.year}–{end.year})",
                 color=FG, fontsize=13, weight="bold", x=0.5, y=0.975)
    path = os.path.join(OUT, "momentum_report.png")
    fig.savefig(path, dpi=140, facecolor=BG, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
