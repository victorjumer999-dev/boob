#!/usr/bin/env python3
"""Charts for the wide-universe study."""
from __future__ import annotations
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.dates, matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum import metrics

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,"results_wide")
BG="#0d1117"; FG="#e6edf3"; GRID="#30363d"
BLUE="#4c9aff"; RED="#f4685c"; GREEN="#3fb950"; GOLD="#e3b341"; PURPLE="#bc8cff"; GREY="#8b949e"

def style(ax,title=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=FG,labelsize=8); ax.grid(True,color=GRID,lw=.6,alpha=.7); ax.set_axisbelow(True)
    if title: ax.set_title(title,color=FG,fontsize=11,loc="left",pad=9,weight="bold")

def main():
    rets=pd.read_csv(f"{OUT}/wide_returns.csv",parse_dates=["date" if "date" in open(f"{OUT}/wide_returns.csv").readline() else "Unnamed: 0"],index_col=0)
    rets.index=pd.to_datetime(rets.index)
    panel=pd.read_csv(f"{OUT}/asset_returns.csv",index_col=0,parse_dates=True)

    fig=plt.figure(figsize=(14,12.5),facecolor=BG)
    gs=fig.add_gridspec(3,2,height_ratios=[1.15,1,1],hspace=.42,wspace=.24)

    # ---- equity curves
    ax=fig.add_subplot(gs[0,:]); style(ax,f"Growth of 1, {rets.index[0].date()} – {rets.index[-1].date()}  (12-1 momentum, 10bp one-way)")
    show=[("SPY buy & hold",GOLD),("Equal-weight (13)",GREY),("TS long/flat (13)",GREEN),
          ("XS long-only (13, RP)",BLUE),("XS long/short (13, RP)",PURPLE),("XS long/short (13)",RED)]
    for name,c in show:
        if name not in rets: continue
        eq=(1+rets[name]).cumprod()
        ax.plot(eq.index,eq.to_numpy(),color=c,lw=1.6,
                label=f"{name}  (Sharpe {metrics.sharpe(rets[name]):+.2f})")
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log)",color=FG,fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:g}"))
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=8.5,loc="upper left",ncol=2).get_frame().set_alpha(.92)
    ax.text(.5,.055,"Widening the universe did not rescue cross-sectional momentum.\n"
                    "The passive equal-weight book still wins.",
            transform=ax.transAxes,color=FG,fontsize=9,ha="center",
            bbox=dict(facecolor="#1c2128",edgecolor=RED,boxstyle="round,pad=0.5",alpha=.95))

    # ---- correlation heatmap
    ax=fig.add_subplot(gs[1,0]); style(ax,"Why breadth should help: pairwise correlation")
    c=panel.corr()
    im=ax.imshow(c.to_numpy(),cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns,rotation=90,fontsize=7)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index,fontsize=7)
    ax.grid(False)
    # Lower triangle only -- the matrix is symmetric and annotating both
    # halves overprints adjacent cells.
    for (i,j),v in np.ndenumerate(c.to_numpy()):
        if i>j and abs(v)>0.75:
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.5,color="white",weight="bold")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.ax.tick_params(colors=FG,labelsize=7)
    cb.outline.set_edgecolor(GRID)

    # ---- asset vols
    ax=fig.add_subplot(gs[1,1]); style(ax,"Why equal weights break: annualised volatility")
    vols=(panel.std()*np.sqrt(12)*100).sort_values()
    colors=[GREEN if v<20 else (GOLD if v<40 else RED) for v in vols]
    ax.barh(range(len(vols)),vols.to_numpy(),color=colors,alpha=.85)
    ax.set_yticks(range(len(vols))); ax.set_yticklabels(vols.index,fontsize=7.5)
    ax.set_xlabel("Annualised volatility (%)",color=FG,fontsize=9)
    for i,v in enumerate(vols): ax.text(v+1,i,f"{v:.0f}%",va="center",color=FG,fontsize=7)
    ax.set_xlim(0,vols.max()*1.18)
    ax.text(.97,.06,"A 5.7x spread. Equal weights make this\na natural-gas bet with equities attached.",
            transform=ax.transAxes,color=FG,fontsize=8,ha="right",
            bbox=dict(facecolor="#1c2128",edgecolor=GRID,boxstyle="round,pad=0.4",alpha=.95))

    # ---- drawdowns
    ax=fig.add_subplot(gs[2,0]); style(ax,"Drawdown")
    for name,c in [("SPY buy & hold",GOLD),("TS long/flat (13)",GREEN),("XS long/short (13, RP)",PURPLE)]:
        dd=100*metrics.drawdown_series(rets[name])
        ax.fill_between(dd.index,dd.to_numpy(),0,color=c,alpha=.30)
        ax.plot(dd.index,dd.to_numpy(),color=c,lw=1.0,label=f"{name} ({dd.min():.0f}%)")
    ax.set_ylabel("Drawdown (%)",color=FG,fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=7.5,loc="lower left").get_frame().set_alpha(.92)

    # ---- grid
    ax=fig.add_subplot(gs[2,1]); style(ax,"Sharpe across the lookback grid")
    grid=pd.read_csv(f"{OUT}/wide_grid.csv")
    for book,c,m in [("long-only",BLUE,"o"),("long/short",RED,"s")]:
        for sk,ls in [(0,"--"),(1,"-")]:
            cell=grid[(grid["book"]==book)&(grid["skip"]==sk)].sort_values("lookback")
            ax.plot(cell["lookback"],cell["sharpe"],ls,marker=m,color=c,ms=4,lw=1.2,
                    alpha=1 if sk==1 else .55,label=f"{book}, skip={sk}")
    spy=metrics.sharpe(rets["SPY buy & hold"])
    ax.axhline(spy,color=GOLD,ls=":",lw=1.6); ax.axhline(0,color=GREY,lw=.8)
    ax.text(24,spy+.02,f"SPY = {spy:.2f}",color=GOLD,fontsize=8,ha="right")
    ax.set_xlabel("Lookback (months)",color=FG,fontsize=9); ax.set_ylabel("Sharpe",color=FG,fontsize=9)
    ax.set_xticks([3,6,9,12,18,24])
    ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=FG,fontsize=7,loc="lower right").get_frame().set_alpha(.92)

    fig.suptitle("Cross-sectional momentum on a wide universe — 13 assets, 4 classes, 2001–2026",
                 color=FG,fontsize=13.5,weight="bold",y=.968)
    p=f"{OUT}/wide_report.png"
    fig.savefig(p,dpi=140,facecolor=BG,bbox_inches="tight"); print("wrote",p)

if __name__=="__main__": main()
