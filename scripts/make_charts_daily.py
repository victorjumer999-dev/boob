#!/usr/bin/env python3
"""Charts for the daily XAUUSD study."""
from __future__ import annotations
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum import backtest as bt, data, metrics, signals as sig

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,"results_daily"); BPY=data.BARS_PER_YEAR["daily"]
BG="#0d1117"; FG="#e6edf3"; GRID="#30363d"
BLUE="#4c9aff"; RED="#f4685c"; GREY="#c9d1d9"; GREEN="#3fb950"; GOLD="#e3b341"

def style(ax,title=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=FG,labelsize=8); ax.grid(True,color=GRID,lw=.6,alpha=.7)
    ax.set_axisbelow(True)
    if title: ax.set_title(title,color=FG,fontsize=11,loc="left",pad=9,weight="bold")

def main():
    prices=pd.read_csv(os.path.join(ROOT,"data","raw_daily","XAUUSD.csv"),
                       parse_dates=["date"]).set_index("date")[["close"]].rename(columns={"close":"XAUUSD"})
    base=dict(lookback=252,skip=21,cost_bps=2.0,periods_per_year=BPY)
    tsf=bt.timeseries_momentum(prices,bt.BacktestConfig(**base,long_only=True))
    vt =bt.timeseries_momentum(prices,bt.BacktestConfig(**base,long_only=True,
                               target_vol=0.15,vol_window=63,max_leverage=2.0))
    bh =bt.buy_and_hold(prices)
    s=max(tsf.returns.index[0],vt.returns.index[0],bh.returns.index[0])
    e=min(tsf.returns.index[-1],vt.returns.index[-1],bh.returns.index[-1])
    curves={"Buy & hold gold":(bh.returns.loc[s:e],GOLD),
            "TS momentum (long/flat)":(tsf.returns.loc[s:e],BLUE),
            "TS + 15% vol target":(vt.returns.loc[s:e],GREEN)}

    fig=plt.figure(figsize=(13,12),facecolor=BG)
    gs=fig.add_gridspec(3,2,height_ratios=[1.25,1,1],hspace=.45,wspace=.22)

    ax=fig.add_subplot(gs[0,:])
    style(ax,f"XAUUSD daily, {s.date()} – {e.date()}  ({len(bh.returns.loc[s:e])} bars = "
             f"{len(bh.returns.loc[s:e])/BPY:.2f} yrs after a 273-bar warm-up; 2bp one-way)")
    # Stagger the end labels: the three curves finish within a few percent of
    # each other, so anchoring each to its own y-value overprints them.
    for off,(name,(r,c)) in zip((14,0,-14),curves.items()):
        eq=(1+r).cumprod(); eq=pd.concat([pd.Series([1.0],index=[s-pd.Timedelta(days=1)]),eq])
        ax.plot(eq.index,eq.to_numpy(),color=c,lw=1.7,label=f"{name}  (Sharpe {metrics.sharpe(r,periods_per_year=BPY):.2f})")
        ax.annotate(f"{100*(eq.iloc[-1]-1):+.0f}%",xy=(eq.index[-1],eq.iloc[-1]),
                    xytext=(10,off),textcoords="offset points",color=c,fontsize=9,weight="bold",va="center")
    ax.set_yscale("log"); ax.set_yticks([1,1.5,2,2.5,3])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Growth of 1",color=FG,fontsize=9)
    ax.set_xlim(s,e+pd.Timedelta(days=95))
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=9,loc="upper left").get_frame().set_alpha(.9)
    ax.text(.5,.06,"Buy & hold wins. The trend filter is long 95% of the time in a 2.4x bull market,\n"
                   "so it can only subtract — and it subtracts 2.2%/yr.",
            transform=ax.transAxes,color=FG,fontsize=9,ha="center",
            bbox=dict(facecolor="#1c2128",edgecolor=RED,boxstyle="round,pad=0.5",alpha=.95))

    # price + on/off state
    ax=fig.add_subplot(gs[1,:]); style(ax,"Gold price and the momentum filter's position (shaded = long)")
    px=prices["XAUUSD"].loc[s:e]
    ax.plot(px.index,px.to_numpy(),color=GOLD,lw=1.3)
    expo=tsf.weights.abs().sum(axis=1).loc[s:e]
    ax.fill_between(px.index,px.min()*.97,px.max()*1.03,where=(expo>.5).to_numpy(),
                    color=BLUE,alpha=.16,step="pre")
    top=px.idxmax()
    ax.annotate(f"top {px.max():,.0f}\n{top.date()}",xy=(top,px.max()),xytext=(-95,-38),
                textcoords="offset points",color=RED,fontsize=8.5,
                arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
    ax.set_ylabel("USD / oz",color=FG,fontsize=9); ax.set_ylim(px.min()*.97,px.max()*1.06)
    ax.text(.5,.06,"The filter was fully long through the January 2026 top and the −13.1% break. "
                   "A 252-day lookback is far too slow to step aside.",
            transform=ax.transAxes,color=FG,fontsize=8.5,ha="center",
            bbox=dict(facecolor="#1c2128",edgecolor=GRID,boxstyle="round,pad=0.4",alpha=.95))

    ax=fig.add_subplot(gs[2,0]); style(ax,"Drawdown")
    for name,(r,c) in curves.items():
        dd=100*metrics.drawdown_series(r)
        ax.fill_between(dd.index,dd.to_numpy(),0,color=c,alpha=.32)
        ax.plot(dd.index,dd.to_numpy(),color=c,lw=1.0,label=f"{name.split('(')[0].strip()} ({dd.min():.0f}%)")
    ax.set_ylabel("Drawdown (%)",color=FG,fontsize=9); ax.set_xlim(s,e)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=7.5,loc="lower left").get_frame().set_alpha(.9)

    ax=fig.add_subplot(gs[2,1]); style(ax,"Sharpe across the lookback grid")
    grid=pd.read_csv(os.path.join(OUT,"daily_grid.csv"))
    bh_s=metrics.sharpe(bh.returns.loc[s:e],periods_per_year=BPY)
    for book,c,m in [("long/flat",BLUE,"o"),("long/short",RED,"s")]:
        for sk,ls in [(0,"-"),(21,"--")]:
            cell=grid[(grid["book"]==book)&(grid["skip"]==sk)].sort_values("lookback")
            if len(cell): ax.plot(cell["lookback"],cell["sharpe"],ls,marker=m,color=c,
                                  ms=4,lw=1.2,alpha=1 if sk==21 else .55,label=f"{book}, skip={sk}")
    ax.axhline(bh_s,color=GOLD,ls=":",lw=1.6)
    ax.text(252,bh_s+.03,f"buy & hold = {bh_s:.2f}",color=GOLD,fontsize=8,ha="right")
    ax.set_xlabel("Lookback (trading days)",color=FG,fontsize=9)
    ax.set_ylabel("Sharpe",color=FG,fontsize=9); ax.set_xticks([21,63,126,189,252])
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=7,loc="lower right").get_frame().set_alpha(.9)

    fig.suptitle("Time-series momentum on XAUUSD — daily bars, 5 years",
                 color=FG,fontsize=13,weight="bold",y=.972)
    p=os.path.join(OUT,"xauusd_daily.png")
    fig.savefig(p,dpi=140,facecolor=BG,bbox_inches="tight"); print("wrote",p)

if __name__=="__main__": main()
