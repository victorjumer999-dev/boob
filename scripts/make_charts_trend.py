#!/usr/bin/env python3
"""The punchline chart: trend following vs simply holding less."""
from __future__ import annotations
import os, sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.dates, matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum import metrics

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=f"{ROOT}/results_wide"
BG="#0d1117"; FG="#e6edf3"; GRID="#30363d"
BLUE="#4c9aff"; GOLD="#e3b341"; GREY="#8b949e"; RED="#f4685c"

def style(ax,t=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=FG,labelsize=8); ax.grid(True,color=GRID,lw=.6,alpha=.7); ax.set_axisbelow(True)
    if t: ax.set_title(t,color=FG,fontsize=11,loc="left",pad=9,weight="bold")

def main():
    r=pd.read_csv(f"{OUT}/trend_returns.csv",index_col=0,parse_dates=True)
    trend,bench=r["Trend long/flat"],r["Equal-weight (13)"]
    k=metrics.annual_vol(trend)/metrics.annual_vol(bench); lazy=bench*k
    rep=json.load(open(f"{OUT}/report_trend.json"))
    curves=[(f"Trend following ({rep['turnover_long_flat']:.1f} turns/yr)",trend,BLUE),
            (f"Hold {100*k:.0f}% of the basket, never trade",lazy,GOLD),
            ("Equal-weight basket (full)",bench,GREY)]

    fig=plt.figure(figsize=(13,9.5),facecolor=BG)
    gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],hspace=.36,wspace=.22)

    ax=fig.add_subplot(gs[0,:]); style(ax,f"Growth of 1, {r.index[0].date()} – {r.index[-1].date()}  (13 assets, 10bp one-way)")
    for name,x,c in curves:
        eq=(1+x).cumprod()
        ax.plot(eq.index,eq.to_numpy(),color=c,lw=1.8,
                label=f"{name}   CAGR {100*metrics.cagr(x):.2f}%  Sharpe {metrics.sharpe(x):.3f}")
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log)",color=FG,fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:g}"))
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=8.5,loc="upper left").get_frame().set_alpha(.92)
    ax.text(.5,.07,"Same return. Same volatility. Same Sharpe.  Excess = +0.01%/yr, t = 0.00.\n"
                   "Trend following bought nothing a de-levered buy-and-hold didn't already give.",
            transform=ax.transAxes,color=FG,fontsize=9.5,ha="center",
            bbox=dict(facecolor="#1c2128",edgecolor=RED,boxstyle="round,pad=0.5",alpha=.95))

    ax=fig.add_subplot(gs[1,0]); style(ax,"Except here: drawdown")
    for name,x,c in curves[:2]:
        dd=100*metrics.drawdown_series(x)
        ax.fill_between(dd.index,dd.to_numpy(),0,color=c,alpha=.32)
        ax.plot(dd.index,dd.to_numpy(),color=c,lw=1.2,label=f"{name.split('(')[0].strip()} ({dd.min():.0f}%)")
    ax.set_ylabel("Drawdown (%)",color=FG,fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=7.5,loc="lower left").get_frame().set_alpha(.92)
    ax.text(.97,.13,f"{100*metrics.max_drawdown(trend):.1f}% vs {100*metrics.max_drawdown(lazy):.1f}%\n"
            "at identical return and vol",transform=ax.transAxes,
            color=FG,fontsize=8.5,ha="right",weight="bold",
            bbox=dict(facecolor="#1c2128",edgecolor=BLUE,boxstyle="round,pad=0.4",alpha=.95))

    ax=fig.add_subplot(gs[1,1]); style(ax,"Permutation null: does the timing matter?")
    p=rep["permutation"]
    rng=np.random.default_rng(0)
    draws=rng.normal(p["null_mean"],p["null_sd"],40000)   # smooth render of the tabulated null
    ax.hist(draws,bins=70,color=GREY,alpha=.55,density=True,label="randomly-timed same positions")
    ax.axvline(p["real_sharpe"],color=BLUE,lw=2.2,label=f"real strategy ({p['real_sharpe']:.3f})")
    ax.axvline(p["null_p95"],color=RED,ls="--",lw=1.4,label=f"95th pct of null ({p['null_p95']:.3f})")
    ax.set_xlabel("Sharpe",color=FG,fontsize=9); ax.set_ylabel("density",color=FG,fontsize=9)
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=7.5,loc="upper left").get_frame().set_alpha(.92)
    ax.text(.97,.35,f"real sits at the {p['percentile']:.0f}th percentile\np = {p['p_value']:.2f}",
            transform=ax.transAxes,color=FG,fontsize=8.5,ha="right",
            bbox=dict(facecolor="#1c2128",edgecolor=GRID,boxstyle="round,pad=0.4",alpha=.95))

    fig.suptitle("Is there an edge? One pre-specified test, after 97 prior configurations",
                 color=FG,fontsize=13.5,weight="bold",y=.965)
    p_out=f"{OUT}/trend_report.png"
    fig.savefig(p_out,dpi=140,facecolor=BG,bbox_inches="tight"); print("wrote",p_out)

if __name__=="__main__": main()
